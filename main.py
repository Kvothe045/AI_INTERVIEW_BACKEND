from multiprocessing import process
import os
import uuid
import base64
import asyncio
import re
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

class InterviewSession:
    def __init__(self, resume_text: str, jd_text: str):
        self.resume_text = resume_text
        self.jd_text = jd_text
        self.transcript = []
        self.question_count = 0
        self.max_questions = 15 
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

        self.system_instruction = f"""
        You are a strict but professional Technical Interviewer named Mr. Elon Musk, Mr. Bill Gates, Mr. Tony Stark or Mr Bruce Wayne, pick one randomly very randomly, based on current date number DONT TELL THE PROCESS OF SELECTION OF PERSONALIT. Just say I am..., and I will be conducting your technical interview for a software engineering role.
        
        CONTEXT:
        RESUME: {self.resume_text[:3000]}
        JOB DESCRIPTION (JD): {self.jd_text[:1500]}
        
        INTERVIEW STRUCTURE:
        1. Introduction
        2. DSA & Problem Solving (JD based)-use JD to set tone and difficulty of interview, search internet to determine what questions have been asked in past interviews for this JD.
        3. Resume Deep Dive
        4. Conceptual Questions
        5. Conclusion
        
        CRITICAL INSTRUCTIONS:
        - Ask exactly ONE question at a time.
        - IGNORE phonetic/transcription errors (e.g. "casing" -> "caching").
        - If the user says "TIME_IS_UP_SIGNAL", immediately say: "Time is up. Let's conclude the interview." and stop asking questions.
        """

        self.chat = self.client.aio.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                temperature=0.6
            )
        )

    async def get_next_response(self, user_input: str = None, is_silence_trigger: bool = False, is_time_up: bool = False):
        # 1. Handle Time Up Force Quit
        if is_time_up:
            prompt = "The interview time has expired. Politely tell the candidate the time is up and you will now provide feedback."
            response = await self.chat.send_message(prompt)
            return clean_text_for_tts(response.text), True # True = Finished

        # 2. Normal Logic
        if self.question_count >= self.max_questions:
            termination_msg = "The interview is now over. Please give me a moment to generate your feedback."
            return termination_msg, True 

        if is_silence_trigger:
            prompt = "The candidate has been silent for 10 seconds. Politely nudge them."
        else:
            if self.question_count == 0 and not user_input:
                prompt = "Start the interview now. Ask the Introduction question."
            else:
                prompt = f"""
                Candidate Answer: "{user_input}"
                Instructions: Evaluate answer, cross-question if needed, otherwise move to next topic. Keep it spoken and concise.
                """

        response = await self.chat.send_message(prompt)
        ai_text = clean_text_for_tts(response.text)
        
        self.question_count += 1
        
        self.transcript.append(f"User: {user_input if user_input else '[SILENCE]'}")
        self.transcript.append(f"AI: {ai_text}")
        
        is_finished = "interview is now over" in ai_text.lower() or "time is up" in ai_text.lower()
        
        return ai_text, is_finished

    async def generate_feedback(self):
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

# --- AUDIO GENERATION ---
async def generate_audio(text: str, filename: str):
    try:
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        await asyncio.wait_for(communicate.save(filename), timeout=5.0)
    except Exception:
        def run_gtts():
            tts = gTTS(text=text, lang='en')
            tts.save(filename)
        await asyncio.to_thread(run_gtts)

# --- API ENDPOINTS ---

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
    audio_file = f"temp_{session_id}.mp3"

    try:
        # Initial Question
        ai_text, _ = await session.get_next_response(user_input=None)
        await generate_audio(ai_text, audio_file)
        with open(audio_file, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8") 
        await websocket.send_json({"type": "audio", "data": audio_data, "text": ai_text})
        if os.path.exists(audio_file): os.remove(audio_file)

        # Loop
        while True:
            data = await websocket.receive_json()
            user_text = data.get("text")
            msg_type = data.get("type") # 'answer', 'silence_timeout', 'time_up'
            
            is_silence = (msg_type == 'silence_timeout')
            is_time_up = (msg_type == 'time_up')

            ai_text, is_finished = await session.get_next_response(
                user_text, 
                is_silence_trigger=is_silence, 
                is_time_up=is_time_up
            )
            
            # Generate response audio
            await generate_audio(ai_text, audio_file)
            with open(audio_file, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode("utf-8")

            if is_finished:
                # Send the "Interview Over" audio first
                await websocket.send_json({"type": "audio", "data": audio_data, "text": ai_text})
                if os.path.exists(audio_file): os.remove(audio_file)
                
                # THEN send feedback
                feedback_text = await session.generate_feedback()
                await websocket.send_json({
                    "type": "feedback", 
                    "text": feedback_text, 
                    "is_finished": True
                })
                break
            else:
                await websocket.send_json({"type": "audio", "data": audio_data, "text": ai_text})
                if os.path.exists(audio_file): os.remove(audio_file)

    except WebSocketDisconnect:
        if session_id in sessions: del sessions[session_id]
        print(f"Session {session_id} disconnected")