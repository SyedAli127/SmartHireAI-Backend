#!/usr/bin/env python3
# ==========================================================
#   Voice Interview Agent (LangChain + LangGraph + LangSmith)
# ==========================================================

import os
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from services import s3_service
import json
from datetime import datetime
from typing import List, Dict

import openai
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from dotenv import load_dotenv

# LangChain + LangGraph + LangSmith
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langsmith import Client
from langgraph.graph import StateGraph, END

load_dotenv()

# ----------------------------------------
# LANGSMITH SETUP
# ----------------------------------------
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

client = Client()

# ----------------------------------------
# CONFIG
# ----------------------------------------
MODEL = "gpt-4o-mini"
STT_MODEL = "gpt-4o-mini-tts"

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("ERROR: Missing OPENAI_API_KEY")
    sys.exit(1)

# LangChain LLM
llm = ChatOpenAI(
    model=MODEL,
    temperature=0.6,
)

# ----------------------------------------
# PROMPTS
# ----------------------------------------
interviewer_prompt = PromptTemplate.from_template("""
You are a highly skilled technical interviewer.
Ask clear, concise questions.
NEVER answer the questions yourself.
Ask the next best question.

Role: {job}
Question #{qn}

Previous answers:
{history}

Ask the next question ONLY.
""")

evaluation_prompt = PromptTemplate.from_template("""
You are a senior hiring manager.
Evaluate the candidate for the role: {job}

Interview transcript:
{transcript}

Provide evaluation:
1. PASS/FAIL
2. Score (0–100)
3. Strengths
4. Weaknesses
5. Detailed feedback
6. Final recommendation
""")

# ----------------------------------------
# RECORD AUDIO (text fallback)
# ----------------------------------------
def record_audio(seconds=10, samplerate=16000):
    answer = input("Type your answer here: ")
    return None, answer

# ----------------------------------------
# SPEECH TO TEXT
# ----------------------------------------
def speech_to_text(filepath):
    with open(filepath, "rb") as f:
        result = openai.Audio.transcriptions.create(
            model=STT_MODEL,
            file=f
        )
    return result.text.strip()

# ==========================================================
#  LANGGRAPH INTERVIEW FLOW
# ==========================================================

# STATE SHAPE
class InterviewState(dict):
    job: str
    qn: int
    total_questions: int
    history: List[Dict]
    last_question: str
    last_answer: str
    evaluation: str


# -------- Node: Ask Question --------
def node_ask_question(state: InterviewState):
    prev = "\n\n".join(
        f"Q{i+1}: {h['q']}\nA{i+1}: {h['a']}" for i, h in enumerate(state["history"])
    ) or "No previous responses."

    prompt = interviewer_prompt.format(
        job=state["job"],
        qn=state["qn"],
        history=prev
    )

    question = llm.invoke(prompt).content.strip()
    print(f"\n--- QUESTION {state['qn']} ---")
    print(question)

    state["last_question"] = question
    return state

# -------- Node: Get Answer --------
def node_get_answer(state: InterviewState):
    input("Press ENTER to answer...")
    path, text = record_audio()

    if path:
        try:
            text = speech_to_text(path)
        except:
            text = input("Transcription failed. Type answer: ")

    print(f"Answer: {text}")

    state["last_answer"] = text
    state["history"].append({"q": state["last_question"], "a": text})
    return state

# -------- Router: continue or evaluate --------
def router_should_continue(state: InterviewState):
    if state["qn"] >= state["total_questions"]:
        return "evaluate"
    return "next"

# -------- Node: Final Evaluation --------
def node_evaluate(state: InterviewState):
    transcript = "\n\n".join(
        f"Q{i+1}: {h['q']}\nA{i+1}: {h['a']}"
        for i, h in enumerate(state["history"])
    )

    evaluation = llm.invoke(
        evaluation_prompt.format(job=state["job"], transcript=transcript)
    ).content.strip()

    print("\n=== FINAL EVALUATION ===\n")
    print(evaluation)

    state["evaluation"] = evaluation
    return state


# ==========================================================
# BUILD LANGGRAPH WORKFLOW
# ==========================================================
workflow = StateGraph(InterviewState)

workflow.add_node("ask", node_ask_question)
workflow.add_node("answer", node_get_answer)
workflow.add_node("evaluate", node_evaluate)

workflow.set_entry_point("ask")

workflow.add_edge("ask", "answer")
workflow.add_conditional_edges(
    "answer",
    router_should_continue,
    {
        "next": "ask",
        "evaluate": "evaluate"
    }
)

workflow.add_edge("evaluate", END)

graph = workflow.compile()


# ==========================================================
# RUN INTERVIEW
# ==========================================================
def run():
    print("\n=== LangChain + LangGraph INTERVIEW AGENT ===")

    job = input("Job Title: ").strip() or "Software Engineer"
    qcount = int(input("Number of questions (default 6): ") or 6)

    initial_state = InterviewState(
        job=job,
        qn=1,
        total_questions=qcount,
        history=[]
    )

    final_state = None

    for state in graph.stream(initial_state):
        # LangGraph streams partial states
        keys = list(state.keys())
        if "qn" in state and state["qn"] <= qcount:
            state["qn"] += 1
        final_state = state

    # Upload output to S3
    s3_key = f"interviews/interview_{int(time.time())}.json"
    url = s3_service.upload_text(json.dumps({
        "job_title": final_state["job"],
        "history": final_state["history"],
        "evaluation": final_state["evaluation"],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }, indent=2), s3_key)
    print(f"Saved result to S3: {url}")

    print(f"Saved result → {fname}")


if __name__ == "__main__":
    run()
