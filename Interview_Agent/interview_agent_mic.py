#!/usr/bin/env python3
# ================================================
# Voice Interview Agent (New OpenAI SDK >=1.0)
# Python 3.13 compatible
# ================================================

print("SCRIPT LOADED OK")

import os
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from services import s3_service
import json
import tempfile
from datetime import datetime
from typing import List, Dict

from openai import OpenAI
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()  # NEW SDK client

# ----------------------------------------
# CONFIG
# ----------------------------------------
MODEL = "gpt-4o-mini"
STT_MODEL = "gpt-4o-mini-transcribe"  # New STT model

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Missing OPENAI_API_KEY")
    sys.exit(1)

# ----------------------------------------
# PROMPTS
# ----------------------------------------
interviewer_prompt_system = """
You are a highly skilled technical interviewer.
Ask clear, concise questions.
Ask follow-up questions that dig deeper based on previous answers.
NEVER answer the questions yourself.
Only ask the next best question.
"""

interviewer_prompt_user = """
Role: {job}
Question #{qn}

Previous answers:
{history}

Ask the next best interview question.
"""

# ----------------------------------------
# RECORD AUDIO (Optional)
# ----------------------------------------
def record_audio(seconds=12, samplerate=16000):
    try:
        print(f"Recording for {seconds} seconds. Speak now...")
        audio = sd.rec(int(seconds * samplerate), samplerate=samplerate,
                       channels=1, dtype='int16')
        sd.wait()
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav.write(temp.name, samplerate, audio)
        return temp.name
    except Exception as e:
        print("Audio recording failed, fallback to text input:", e)
        return None

# ----------------------------------------
# SPEECH TO TEXT (NEW API)
# ----------------------------------------
def speech_to_text(filepath):
    if not filepath:
        return input("Type your answer manually: ")
    try:
        with open(filepath, "rb") as f:
            result = client.audio.transcriptions.create(
                model=STT_MODEL,
                file=f
            )
        return result.text.strip()
    except Exception as e:
        print("Transcription failed:", e)
        return input("Type your answer manually: ")

# ----------------------------------------
# ASK QUESTION (NEW API)
# ----------------------------------------
def ask_question(job_title: str, qn: int, history: List[Dict]):
    prev = "\n\n".join([f"Q{i+1}: {h['q']}\nA{i+1}: {h['a']}" 
                        for i, h in enumerate(history)]) or "No previous responses."

    user_prompt = interviewer_prompt_user.format(job=job_title, qn=qn, history=prev)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": interviewer_prompt_system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6
    )
    return response.choices[0].message.content.strip()

# ----------------------------------------
# FINAL EVALUATION (NEW API)
# ----------------------------------------
def evaluate(job_title: str, history: List[Dict]):
    convo = "\n\n".join([f"Q{i+1}: {h['q']}\nA{i+1}: {h['a']}" 
                         for i, h in enumerate(history)])

    prompt = f"""
You are a senior hiring manager.
Evaluate the candidate for the role: {job_title}.

Interview transcript:
{convo}

Give a structured evaluation:
1. PASS/FAIL
2. Score (0–100)
3. Strengths
4. Weaknesses
5. Detailed feedback
6. Final hiring recommendation
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You evaluate candidates with precision and fairness."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4
    )
    return response.choices[0].message.content.strip()

# ----------------------------------------
# MAIN INTERVIEW LOOP
# ----------------------------------------
def run():
    print("\n=== VOICE INTERVIEW AGENT ===")
    job = input("Job title: ").strip() or "Software Engineer"
    try:
        qcount = int(input("Number of questions (default 6): ") or 6)
    except:
        qcount = 6

    history = []

    for qn in range(1, qcount + 1):
        question = ask_question(job, qn, history)
        print(f"\n--- QUESTION {qn} ---")
        print(question)

        input("Press ENTER to start recording...")
        audio_path = record_audio()
        print("Transcribing...")

        answer = speech_to_text(audio_path)
        print(f"Answer: {answer}\n")
        history.append({"q": question, "a": answer})

    print("\nGenerating final evaluation...\n")
    evaluation = evaluate(job, history)

    print("=== FINAL EVALUATION ===\n")
    print(evaluation)

    out = {
        "job_title": job,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "history": history,
        "evaluation": evaluation
    }

    s3_key = f"interviews/interview_{int(time.time())}.json"
    url = s3_service.upload_text(json.dumps(out, indent=2, ensure_ascii=False), s3_key)
    print(f"Saved result to S3: {url}")


if __name__ == "__main__":
    run()
