import os
import uuid
import base64
import asyncio
import re
import random
import io # NEW: For in-memory processing
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
    allow_origins=["http://localhost:3000", "https://voca-rouge.vercel.app"], 
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

# --- VOICE & PERSONA CONFIGURATION ---
# We map EdgeTTS voices AND gTTS accents (tld) for backup
PERSONAS = [
    {
        "name": "Elon Musk", 
        "style": "Visionary, direct, slightly stuttery but intense", 
        "voice": "en-US-ChristopherNeural", 
        "backup_tld": "us" # US Accent
    }, 
    {
        "name": "Bill Gates", 
        "style": "Analytical, calm, intellectual", 
        "voice": "en-US-EricNeural", 
        "backup_tld": "us" # US Accent
    },
    {
        "name": "Tony Stark", 
        "style": "Witty, fast-paced, confident, slightly arrogant", 
        "voice": "en-GB-RyanNeural", # UK Voice for Tony fits the wit better
        "backup_tld": "co.uk" # UK Accent backup
    }, 
    {
        "name": "Friday", 
        "style": "Professional, sharp, tactical, calm", 
        "voice": "en-US-AriaNeural", 
        "backup_tld": "com.au" # Australian backup for distinction
    }, 
]

class InterviewSession:
    def __init__(self, resume_text: str, jd_text: str):
        self.resume_text = resume_text
        self.jd_text = jd_text
        self.transcript = []
        self.question_count = 0
        self.max_questions = 15 
        
        # 1. Select Persona & Voice for this session
        self.persona = random.choice(PERSONAS)
        self.voice_id = self.persona["voice"]
        self.backup_tld = self.persona["backup_tld"]
        
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

        self.system_instruction = f"""
        You are acting as **{self.persona['name']}**.
        Persona Style: {self.persona['style']}.
        
        You are conducting a strict but professional Technical Interview.
        
        CONTEXT:
        RESUME: {self.resume_text[:3000]}
        JOB DESCRIPTION (JD): {self.jd_text[:1500]}
        
        INTERVIEW STRUCTURE:
        0. **Introduction**: Introduce yourself as {self.persona['name']}. Do NOT explain why you were chosen. just say "I am...".
        1. **INTRODUCTION QUESTION**: Start with asking intro like one would expect in technical interviews start Just one question, eg introduce yourself and what you do, why /JD ROLE(based on JD)/ and one cross question then move to CORE.
        2. **DSA & Problem Solving(if applicable)**: Use the JD to set the difficulty. Minimum 2 main dsa/problem-solving question. Ask rigorous questions with one cross-question on it if seems feasible, always rely on last response.
        3. **Resume Deep Dive**: Drill down into skills+projects+other key things.
        4. **Conceptual Questions**: Ask standard medium to medium-hard to hard questions for this role based on internet searches(past experiences).
        6. **System Design**: If for this role in real-world in past system design applicable then, ask 1 system design question.
        5. **Conclusion**: End the interview.
        
        CRITICAL INSTRUCTIONS:
        - Ask exactly ONE question at a time.
        - Adopt the speech patterns of {self.persona['name']} slightly, but prioritize clarity.
        - IGNORE phonetic errors (e.g. "casing" -> "caching").
        - If user says "TIME_IS_UP_SIGNAL", immediately conclude.
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
            prompt = "The interview time has expired. Politely tell the candidate the time is up and you will now provide feedback."
            response = await self.chat.send_message(prompt)
            return clean_text_for_tts(response.text), True 

        if self.question_count >= self.max_questions:
            termination_msg = "The interview is now over. Please give me a moment to generate your feedback."
            return termination_msg, True 

        if is_silence_trigger:
            prompt = f"The candidate is silent. As {self.persona['name']}, politely nudge them or ask if they are stuck."
        else:
            if self.question_count == 0 and not user_input:
                prompt = "Start the interview now. Introduce yourself."
            else:
                prompt = f"""
                Candidate Answer: "{user_input}"
                Instructions: Evaluate answer based on technical accuracy. Cross-question if vague. Move to next topic if satisfied. Keep it spoken and concise.
                """

        response = await self.chat.send_message(prompt)
        ai_text = clean_text_for_tts(response.text)
        
        self.question_count += 1
        
        self.transcript.append(f"User: {user_input if user_input else '[SILENCE]'}")
        self.transcript.append(f"AI ({self.persona['name']}): {ai_text}")
        
        is_finished = "interview is now over" in ai_text.lower() or "time is up" in ai_text.lower()
        
        return ai_text, is_finished

    async def generate_feedback(self):
        prompt = """
        Based on the entire transcript, provide detailed feedback as a mentor.
        Structure:
        1. **Strengths**
        2. **Weaknesses**
        3. **Improvements**
        4. **Resources**
        5. **RESUME FEEDBACK & ATS SCORE**: (0-100%)
        6. **IMPROVEMENT SUGGESTIONS**
        7. **FINAL REMARKS**
        """
        response = await self.chat.send_message(prompt)
        return clean_text_for_tts(response.text)

# --- OPTIMIZED AUDIO GENERATION (IN-MEMORY) ---
async def generate_audio_stream(text: str, voice_id: str, backup_tld: str) -> str:
    """
    Generates audio without touching the disk.
    Prioritizes EdgeTTS (Memory Stream). Falls back to gTTS (Memory Stream).
    """
    # 1. Try EdgeTTS (High Quality)
    try:
        communicate = edge_tts.Communicate(text, voice_id)
        mp3_data = b""
        
        # Stream chunks directly into a bytes variable
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_data += chunk["data"]
        
        if len(mp3_data) > 0:
            return base64.b64encode(mp3_data).decode("utf-8")
        else:
            raise Exception("EdgeTTS yielded empty audio")

    # 2. Fallback to gTTS (Reliable + Accented)
    except Exception as e:
        print(f"⚠️ EdgeTTS Failed ({e}). Switching to gTTS Backup ({backup_tld}).")
        try:
            def run_gtts():
                # Write to in-memory bytes buffer, NOT disk
                fp = io.BytesIO()
                tts = gTTS(text=text, lang='en', tld=backup_tld)
                tts.write_to_fp(fp)
                fp.seek(0)
                return fp.read()

            # Run blocking gTTS in a thread
            gtts_data = await asyncio.to_thread(run_gtts)
            return base64.b64encode(gtts_data).decode("utf-8")
            
        except Exception as e2:
            print(f"CRITICAL: Both TTS engines failed. {e2}")
            return "" # Return silence instead of crashing

# --- API ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "active", "service": "AI Interviewer Backend"}

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
        
        # Generate Audio (Optimized)
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
            
            # Generate Audio (Optimized)
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