#!/usr/bin/env python3
# Voice + Video Interview Agent with emotion timeline
# Python 3.13, OpenAI SDK >=1.0
# -----------------------------------------------

import os
import sys
import time
import json
import tempfile
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from services import s3_service
import threading
from datetime import datetime
from typing import List, Dict
from collections import Counter

import cv2
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# ----------------------------------------
# CONFIG
# ----------------------------------------
MODEL = "gpt-4o-mini"
STT_MODEL = "gpt-4o-mini-transcribe"

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Missing OPENAI_API_KEY")
    sys.exit(1)

# ----------------------------------------
# Prompts
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
# OpenCV Emotion Detection
# ----------------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    emotions_detected = []
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        # Improved rule-based detection
        if len(smiles) > 0:
            emotion = "Happy"
        elif len(eyes) >= 2 and np.mean(roi_gray) > 100:
            emotion = "Surprised"
        else:
            emotion = "Neutral"

        emotions_detected.append(emotion)
        # Overlay rectangle + emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    if not emotions_detected:
        emotions_detected.append("Neutral")
    return frame, emotions_detected

# ----------------------------------------
# Video Recording Thread
# ----------------------------------------
def record_video(seconds, emotion_timeline):
    cap = cv2.VideoCapture(0)
    start = time.time()
    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            break
        frame, emotions = detect_emotion(frame)
        # Store dominant emotion per frame
        emotion_timeline.append(emotions[0])
        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

# ----------------------------------------
# Audio Recording Thread
# ----------------------------------------
def record_audio(seconds, audio_path_container):
    samplerate = 16000
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp.name, samplerate, audio)
    audio_path_container.append(temp.name)

def speech_to_text(filepath):
    if not filepath:
        return input("Type your answer manually: ")
    try:
        with open(filepath, "rb") as f:
            result = client.audio.transcriptions.create(model=STT_MODEL, file=f)
        return result.text.strip()
    except Exception as e:
        print("Transcription failed:", e)
        return input("Type your answer manually: ")

# ----------------------------------------
# GPT Interaction
# ----------------------------------------
def ask_question(job_title: str, qn: int, history: List[Dict]):
    prev = "\n\n".join([f"Q{i+1}: {h['q']}\nA{i+1}: {h['a']}" for i, h in enumerate(history)]) or "No previous responses."
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

def evaluate(job_title: str, history: List[Dict], emotion_timeline: List[str]):
    # Convert timeline to counts
    timeline_counts = Counter(emotion_timeline)
    timeline_str = ", ".join([f"{k}:{v}" for k, v in timeline_counts.items()])
    convo = "\n\n".join([f"Q{i+1}: {h['q']}\nA{i+1}: {h['a']}" for i, h in enumerate(history)])
    prompt = f"""
You are a senior hiring manager.
Evaluate the candidate for the role: {job_title}.

Interview transcript:
{convo}

Candidate emotion timeline: {timeline_str}

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
    print("\n=== VOICE + VIDEO INTERVIEW AGENT ===")
    job = input("Job title: ").strip() or "Software Engineer"
    try:
        qcount = int(input("Number of questions (default 6): ") or 6)
    except:
        qcount = 6

    history = []
    overall_emotions = []

    for qn in range(1, qcount + 1):
        question = ask_question(job, qn, history)
        print(f"\n--- QUESTION {qn} ---")
        print(question)
        input("Press ENTER to start recording audio & video...")

        video_emotions = []
        audio_path_container = []

        # Threads
        t_video = threading.Thread(target=record_video, args=(12, video_emotions))
        t_audio = threading.Thread(target=record_audio, args=(12, audio_path_container))
        t_video.start()
        t_audio.start()
        t_video.join()
        t_audio.join()

        audio_path = audio_path_container[0] if audio_path_container else None
        answer = speech_to_text(audio_path)
        print(f"Answer: {answer}\n")

        history.append({"q": question, "a": answer})
        overall_emotions.extend(video_emotions)

    # Generate final evaluation
    print("\nGenerating final evaluation...\n")
    evaluation = evaluate(job, history, overall_emotions)

    print("=== FINAL EVALUATION ===\n")
    print(evaluation)

    out = {
        "job_title": job,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "history": history,
        "emotion_timeline": overall_emotions,
        "evaluation": evaluation
    }

    s3_key = f"interviews/interview_{int(time.time())}.json"
    url = s3_service.upload_text(json.dumps(out, indent=2, ensure_ascii=False), s3_key)
    print(f"Saved result to S3: {url}")


if __name__ == "__main__":
    run()
