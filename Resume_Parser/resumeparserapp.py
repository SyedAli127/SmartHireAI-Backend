import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
import uuid
from functools import lru_cache
from typing import Any, Dict

from langgraph.graph import StateGraph, START, END
from langsmith import traceable
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from sentence_transformers import CrossEncoder
from Parser.vectorstore import VectorStore
from Parser.resume_parser import parse_resume

import numpy as np
import json
import re

load_dotenv()
RANKER_MODEL_NAME = os.getenv("RANKER_MODEL_NAME", "cross-encoder/stsb-distilroberta-base")

dummy_jd_text = """
We are looking for a Software Engineer with experience in Python, REST APIs, and cloud platforms.
Strong problem-solving skills and ability to work in a team are required.
"""

HYBRID_EVAL_PROMPT = """
You are an expert technical recruiter. You will evaluate a resume against a job description using a strict scoring rubric.

Evaluate ONLY using the text provided. Do NOT hallucinate.

### Job Description:
{jd}

### Resume:
{resume}

### Scoring Rubric (0 to 1 for each):
1. Skill Match (40%)
2. Experience Relevance (30%)
3. Seniority Fit (10%)
4. Tech Stack Similarity (20%)

### Output JSON (ONLY structured JSON):
{{
  "skill_score": float,
  "experience_score": float,
  "seniority_score": float,
  "tech_stack_score": float,
  "explanation": "short reason why candidate is/is not a fit"
}}
"""

class ResumeState(Dict):
    resume_text: str
    embedded: Any
    vectordb: Any
    resume_path: str
    filename: str

@traceable
def parse_node(state: ResumeState):
    resume_path = state["resume_path"]
    parsed = parse_resume(resume_path)
    state["resume_text"] = parsed["cleaned"]
    return state

@traceable
def embed_node(state: ResumeState):
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    text = state["resume_text"]
    vector = embedder.embed_documents([text])[0]
    state["embedded"] = vector
    return state


@lru_cache(maxsize=1)
def get_cross_encoder_model() -> CrossEncoder:
    return CrossEncoder(RANKER_MODEL_NAME)

@traceable
def store_node(state: ResumeState):
    vectordb = state["vectordb"]
    vectordb.add_document(
        text=state["resume_text"],
        metadata={
            "filename": state["filename"],
            "text": state["resume_text"]
        }
    )
    return state

def hybrid_semantic_score(jd_text, resume_text, embedding_model):
    """Combine embeddings similarity + LLM reasoning"""

    legacy_breakdown = compute_legacy_score_breakdown(jd_text, resume_text, embedding_model)
    
    # 1️⃣ Embedding similarity
    jd_emb = embedding_model.embed_query(jd_text)
    resume_emb = embedding_model.embed_query(resume_text)
    emb_similarity = float(np.dot(jd_emb, resume_emb) / (np.linalg.norm(jd_emb) * np.linalg.norm(resume_emb)))

    # 2️⃣ LLM scoring
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = HYBRID_EVAL_PROMPT.format(jd=jd_text, resume=resume_text)
    result = llm.invoke(prompt)
    raw = result.content.strip()

    # Parse JSON safely
    try:
        scores = json.loads(raw)
    except:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise ValueError("LLM output does not contain JSON")
        scores = json.loads(match.group(0))

    # 3️⃣ Weighted hybrid score
    weighted_llm_score = (
        scores["skill_score"] * 0.4 +
        scores["experience_score"] * 0.3 +
        scores["seniority_score"] * 0.1 +
        scores["tech_stack_score"] * 0.2
    )
    hybrid_score = (0.7 * weighted_llm_score) + (0.3 * emb_similarity)

    scores["hybrid_score"] = hybrid_score
    scores["embedding_score"] = emb_similarity
    scores.update(legacy_breakdown)
    scores["legacy_breakdown"] = legacy_breakdown
    scores["structured_scores"] = None
    return scores


def _tokenize(text: str) -> set:
    return set(re.findall(r"[A-Za-z]+", text.lower()))


def _generate_match_summary(cos_sim: float, ce_score: float, overlap_ratio: float, keyword_density: float, len_ratio: float) -> str:
    strengths = []
    weaknesses = []
    if cos_sim > 0.7:
        strengths.append("strong semantic alignment")
    elif cos_sim > 0.5:
        strengths.append("moderate semantic match")
    if overlap_ratio > 0.4:
        strengths.append("high keyword overlap")
    elif overlap_ratio > 0.2:
        strengths.append("decent keyword coverage")
    if keyword_density > 0.3:
        strengths.append("rich relevant content")
    if cos_sim < 0.4:
        weaknesses.append("weak semantic alignment")
    if overlap_ratio < 0.15:
        weaknesses.append("low keyword match")
    if len_ratio < 0.3:
        weaknesses.append("brief resume")
    elif len_ratio > 2.5:
        weaknesses.append("verbose resume")

    if ce_score > 0.7:
        line1 = f"Strong candidate with {', '.join(strengths[:2]) if strengths else 'good overall fit'}."
    elif ce_score > 0.5:
        line1 = f"Moderate match with {', '.join(strengths[:2]) if strengths else 'some relevant experience'}."
    else:
        line1 = f"Weak match due to {', '.join(weaknesses[:2]) if weaknesses else 'limited alignment'}."

    line2 = f"Semantic similarity: {cos_sim:.2f}, Keyword overlap: {overlap_ratio:.1%}, Content density: {keyword_density:.1%}."
    return f"{line1} {line2}"


def compute_legacy_score_breakdown(
    jd_text: str,
    resume_text: str,
    embedding_model: OpenAIEmbeddings,
    faiss_similarity: float = 0.0,
) -> dict:
    """Compute the legacy ranking metrics used by the Streamlit scoring UI."""
    jd_emb = embedding_model.embed_query(jd_text)
    resume_emb = embedding_model.embed_query(resume_text)

    ce_raw = get_cross_encoder_model().predict([(jd_text, resume_text)])[0]
    ce_score = float(1.0 / (1.0 + np.exp(-ce_raw)))
    denom = max(float(np.linalg.norm(jd_emb) * np.linalg.norm(resume_emb)), 1e-8)
    cos_sim = float(np.dot(jd_emb, resume_emb) / denom)

    jd_terms = _tokenize(jd_text)
    resume_terms = _tokenize(resume_text)
    overlap = len(jd_terms & resume_terms)
    overlap_ratio = overlap / max(len(jd_terms), 1)
    kw_density = overlap / max(len(resume_terms), 1)
    len_ratio = min(len(resume_text) / max(len(jd_text), 1), 3.0)

    neural_score = (
        0.60 * ce_score
        + 0.20 * cos_sim
        + 0.15 * overlap_ratio
        + 0.05 * kw_density
    )

    # In API/runtime mode we do not have a persistent FAISS index in hand,
    # so the embedding cosine similarity is the best available proxy.
    faiss_proxy = float(faiss_similarity or cos_sim)

    return {
        "ce_score": round(ce_score, 3),
        "cos_sim": round(cos_sim, 3),
        "overlap_ratio": round(overlap_ratio, 3),
        "kw_density": round(kw_density, 3),
        "len_ratio": round(len_ratio, 2),
        "faiss_similarity": round(faiss_proxy, 3),
        "neural_score": round(neural_score, 3),
        "summary": _generate_match_summary(cos_sim, ce_score, overlap_ratio, kw_density, len_ratio),
        "structured_scores": None,
    }

# -----------------------------
# Build LangGraph
# -----------------------------
workflow = StateGraph(ResumeState)
workflow.add_node("parse", parse_node)
workflow.add_node("embed", embed_node)
workflow.add_node("store", store_node)

workflow.add_edge(START, "parse")
workflow.add_edge("parse", "embed")
workflow.add_edge("embed", "store")
workflow.add_edge("store", END)

graph = workflow.compile()

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.title("SmartHire AI – Hybrid Semantic Resume Matching")
    st.write("Upload resumes and match using AI reasoning + embeddings.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # --- Initialize VectorStore ---
    if "vectordb" not in st.session_state:
        st.session_state.vectordb = VectorStore(
            persist_path="faiss_index.pkl",
            embedding_dim=1536
        )
    vectordb = st.session_state.vectordb

    # --- Resume Upload ---
    resume_file = st.file_uploader("Upload Resume (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
    if resume_file:
        st.success("Resume uploaded!")

    # --- Parse & Store Resume ---
    if resume_file and st.button("Parse & Store Resume"):
        suffix = "." + resume_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(resume_file.read())
            resume_path = tmp.name

        state = graph.invoke({
            "resume_path": resume_path,
            "filename": resume_file.name,
            "vectordb": vectordb
        })

        st.subheader("Extracted Resume Text:")
        st.text_area("Resume Text", state["resume_text"], height=300)

        st.success("Resume stored successfully!")

    # --- Hybrid Matching ---
    st.subheader("Match Resumes Using Hybrid Semantic AI")
    if st.button("Find Top Matches"):
        results = vectordb.search(dummy_jd_text, top_k=10)

        if not results:
            st.info("No resumes stored yet.")
        else:
            st.write("### Ranked Matches (Hybrid AI)")
            for res in results:
                metadata = res["metadata"]
                resume_text = metadata.get("text", "")

                scores = hybrid_semantic_score(dummy_jd_text, resume_text, embeddings)

                st.write(f"**File:** {metadata.get('filename', 'N/A')}")
                st.write(f"**FAISS Similarity Score:** {metadata.get('similarity', 0):.3f}")
                st.write(f"**Hybrid Score:** {scores['hybrid_score']:.3f}")
                st.write(f"**LLM Explanation:** {scores['explanation']}")
                st.markdown("---")

if __name__ == "__main__":
    main()
