import os
import uuid
import base64
import asyncio
import re
import random
import io
import pdfplumber
import edge_tts
from gtts import gTTS
from fastapi import FastAPI, WebSocket, UploadFile, File, Form, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from dotenv import load_dotenv

# --- NEW SDK IMPORTS ---
from google import genai
from google.genai import types

# --- CONFIGURATION ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("API_KEY") 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for robustness in dev/prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- IN-MEMORY SESSION STORE ---
sessions: Dict[str, "InterviewSession"] = {}

def clean_text_for_tts(text: str) -> str:
    text = re.sub(r'\*', '', text)
    text = re.sub(r'#', '', text)
    return text.strip()

# --- PERSONA CONFIGURATION ---
# Note: gTTS (Backup) does NOT support gender selection (it is always female-sounding).
# We map accents (tld) to try and create some distinction in the backup.
PERSONAS = [
    # --- MALES ---
    {
        "name": "Elon Musk", 
        "style": "Visionary, direct, slightly intense, futuristic, bold", 
        "voice": "en-US-ChristopherNeural", 
        "backup_tld": "us", # US Accent
        "gender": "Male"
    }, 
    {
        "name": "Bill Gates", 
        "style": "Analytical, calm, intellectual, thoughtful, precise", 
        "voice": "en-US-EricNeural", 
        "backup_tld": "ca", # Canadian (Neutral)
        "gender": "Male"
    },
    {
        "name": "Tony Stark", 
        "style": "Witty, fast-paced, confident, slightly arrogant, genius", 
        "voice": "en-GB-RyanNeural", 
        "backup_tld": "co.uk", # British
        "gender": "Male"
    }, 
    {
        "name": "Mark Zuckerberg",
        "style": "Data-driven, strategic, monotone but sharp, focused",
        "voice": "en-US-GuyNeural",
        "backup_tld": "us",
        "gender": "Male"
    },
    # --- FEMALES ---
    {
        "name": "Donna Paulsen", 
        "style": "Confident, sharp, professional, witty, empathetic", 
        "voice": "en-US-AriaNeural",
        "backup_tld": "us",
        "gender": "Female"
    }, 
    {
        "name": "Shuri",
        "style": "Genius-level intellect, youthful, energetic, witty, innovative",
        "voice": "en-NG-EzinneNeural", # Nigerian Accent
        "backup_tld": "co.za", # South African backup
        "gender": "Female"
    },
    {
        "name": "Susan Wojcicki",
        "style": "Experienced, insightful, strategic, nurturing yet firm",
        "voice": "en-US-AnaNeural",
        "backup_tld": "us",
        "gender": "Female"
    }
]

class InterviewSession:
    def __init__(self, resume_text: str, jd_text: str):
        self.resume_text = resume_text
        self.jd_text = jd_text
        self.transcript = []
        self.question_count = 0
        self.max_questions = 15 
        
        # 1. Select Persona randomly
        self.persona = random.choice(PERSONAS)
        self.voice_id = self.persona["voice"]
        self.backup_tld = self.persona["backup_tld"]
        
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

        self.system_instruction = f"""
        You are acting as **{self.persona['name']}**.
        Persona Style: {self.persona['style']}.
        
        CONTEXT:
        RESUME: {self.resume_text[:3000]}
        JOB DESCRIPTION: {self.jd_text[:1500]}
        
        INTERVIEW STRUCTURE:
        0. Introduction: Say "I am {self.persona['name']}...".
        1. Intro Question: Ask the candidate to introduce themself and ask 1 relevant intro/behavioral question.
        2. **DSA & Problem Solving(if applicable)**: Use the JD to set the difficulty. Minimum 2 main dsa/problem-solving question. Ask rigorous questions with one cross-question on it if seems feasible, always rely on last response.
        3. **Resume Deep Dive**: Drill down into skills+projects+other key things.
        4. **Conceptual Questions**: Ask 3 standard medium to medium-hard to hard questions for this role based on internet searches(past experiences).Cross question when needed.
        6. **System Design**: If for this role in real-world in past system design applicable then, ask 1 system design question.
        5. **Conclusion**: End the interview. 
        
        RULES:
        - ONE question at a time.
        - Be concise (spoken style).
        - If user says "TIME_IS_UP_SIGNAL", conclude immediately.
        """

        self.chat = self.client.aio.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                temperature=0.7 
            )
        )

    async def get_next_response(self, user_input: str = None, is_silence_trigger: bool = False, is_time_up: bool = False):
        if is_time_up:
            prompt = "Time is up. Briefly thank the candidate and end the interview."
            response = await self.chat.send_message(prompt)
            return clean_text_for_tts(response.text), True 

        if self.question_count >= self.max_questions:
            return "The interview is now over. I will generate your feedback.", True 

        if is_silence_trigger:
            prompt = f"The candidate is silent. As {self.persona['name']}, politely nudge them."
        else:
            if self.question_count == 0 and not user_input:
                prompt = "Start the interview. Introduce yourself."
            else:
                prompt = f"""
                Candidate Answer: "{user_input}"
                Instructions: Evaluate. Cross-question if needed. Otherwise ask next question. Keep it short.
                """

        response = await self.chat.send_message(prompt)
        ai_text = clean_text_for_tts(response.text)
        
        self.question_count += 1
        
        self.transcript.append(f"User: {user_input if user_input else '[SILENCE]'}")
        self.transcript.append(f"AI ({self.persona['name']}): {ai_text}")
        
        is_finished = "interview is now over" in ai_text.lower() or "time is up" in ai_text.lower()
        
        return ai_text, is_finished

    async def generate_feedback(self):
        # prompt = "Provide detailed feedback: First providing analysis of Strengths, Weaknesses, Improvements, Resume Score (0-100), Final Remarks. Be constructive."
        prompt = """
        Based on the entire transcript, provide detailed feedback as a mentor/guide to the student/interviewee.
        Structure:
        1. **Strengths**
        2. **Weaknesses**
        3. **Improvements**
        4. **Resources**
        5. **RESUME Feedback and MATCHING SCORE with JD and is the resume potential enough to get accepted against ATS filters for this role** (0-100%)
        6. **RESUME IMPROVEMENT SUGGESTIONS**
        7. **FINAL REMARKS**
        """
        response = await self.chat.send_message(prompt)
        return clean_text_for_tts(response.text)

# --- 3-LAYER AUDIO GENERATION ---
async def generate_audio_stream(text: str, voice_id: str, backup_tld: str) -> str:
    """
    1. EdgeTTS (Best, Male/Female) - Timeout 1.5s
    2. gTTS Accented (Reliable, always Female) - Timeout 3s
    3. gTTS US Standard (Ultimate Backup)
    """
    
    # --- LAYER 1: EdgeTTS (High Quality) ---
    try:
        # We set a VERY strict timeout. If it lags, we skip it instantly.
        communicate = edge_tts.Communicate(text, voice_id)
        mp3_data = b""
        
        async def run_edge():
            nonlocal mp3_data
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_data += chunk["data"]
        
        # Strict 1.5s timeout. If Render network is slow, drop it.
        await asyncio.wait_for(run_edge(), timeout=1.5)
        
        if len(mp3_data) > 0:
            return base64.b64encode(mp3_data).decode("utf-8")
            
    except (asyncio.TimeoutError, Exception) as e:
        print(f"⚠️ Layer 1 (EdgeTTS) Failed/Slow: {e}")

    # --- LAYER 2: gTTS (Accented) ---
    try:
        def run_gtts_accent():
            fp = io.BytesIO()
            # gTTS is reliable but synchronous, run in thread
            tts = gTTS(text=text, lang='en', tld=backup_tld)
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read()

        gtts_data = await asyncio.wait_for(
            asyncio.to_thread(run_gtts_accent), 
            timeout=3.0
        )
        return base64.b64encode(gtts_data).decode("utf-8")

    except Exception as e:
        print(f"⚠️ Layer 2 (gTTS Accent) Failed: {e}")

    # --- LAYER 3: gTTS (Standard US - Failsafe) ---
    try:
        def run_gtts_std():
            fp = io.BytesIO()
            tts = gTTS(text=text, lang='en', tld='us')
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read()

        gtts_data = await asyncio.to_thread(run_gtts_std)
        return base64.b64encode(gtts_data).decode("utf-8")
        
    except Exception as e:
        print(f"❌ ALL AUDIO LAYERS FAILED: {e}")
        return "" # Frontend will just show text

# --- API ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "active"}

@app.post("/upload-context")
async def upload_context(
    resume: UploadFile = File(None), 
    resume_text: str = Form(None), 
    jd: str = Form(...)
):
    final_resume_text = ""
    if resume:
        with pdfplumber.open(resume.file) as pdf:
            for page in pdf.pages:
                final_resume_text += page.extract_text() or ""
    elif resume_text:
        final_resume_text = resume_text

    session_id = str(uuid.uuid4())
    sessions[session_id] = InterviewSession(final_resume_text, jd)
    return {"session_id": session_id}

@app.websocket("/ws/interview/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    if session_id not in sessions:
        await websocket.close(code=4004, reason="Session not found")
        return

    session = sessions[session_id]

    try:
        # Initial Question
        ai_text, _ = await session.get_next_response(user_input=None)
        
        # Audio Gen
        audio_b64 = await generate_audio_stream(ai_text, session.voice_id, session.backup_tld)
        
        await websocket.send_json({"type": "audio", "data": audio_b64, "text": ai_text})

        # Loop
        while True:
            data = await websocket.receive_json()
            user_text = data.get("text")
            msg_type = data.get("type") 
            
            is_silence = (msg_type == 'silence_timeout')
            is_time_up = (msg_type == 'time_up')

            ai_text, is_finished = await session.get_next_response(
                user_text, 
                is_silence_trigger=is_silence, 
                is_time_up=is_time_up
            )
            
            audio_b64 = await generate_audio_stream(ai_text, session.voice_id, session.backup_tld)

            if is_finished:
                await websocket.send_json({"type": "audio", "data": audio_b64, "text": ai_text})
                
                feedback_text = await session.generate_feedback()
                await websocket.send_json({
                    "type": "feedback", 
                    "text": feedback_text, 
                    "is_finished": True
                })
                break
            else:
                await websocket.send_json({"type": "audio", "data": audio_b64, "text": ai_text})

    except WebSocketDisconnect:
        if session_id in sessions: del sessions[session_id]
        print(f"Session {session_id} disconnected")