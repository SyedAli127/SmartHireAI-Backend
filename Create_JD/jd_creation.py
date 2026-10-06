from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langgraph.graph import START,StateGraph,END
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Optional, List, Dict, Any, Literal,Annotated
import os,sqlite3,uuid,operator
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter 

def build_query(x: Dict[str, Any]) -> str:
    """Construct a compact retrieval query from intake fields."""
    return (
        f"JD policy guidance for '{x.get('title','')}' in '{x.get('department','')}', "
        f"seniority '{x.get('seniority','')}', location '{x.get('location','')}', "
        f"employment '{x.get('employment_type','')}'. "
        f"Focus: responsibilities; must-have vs nice-to-have; skills/tools; "
        f"hiring stages & approvals; compensation/benefits phrasing; legal notes. "
        f"HR notes: {x.get('hr_notes','')}"
    )

outline_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an HR policy outliner. Using ONLY the provided policy passages and intake, "
     "produce a structured Job Description OUTLINE aligned with policy. "
     "If guidance is missing, mark [POLICY-GAP]. Return ONLY the outline text."),
    ("human",
     "Intake\n"
     "- Title: {title}\n"
     "- Department: {department}\n"
     "- Seniority: {seniority}\n"
     "- Location: {location}\n"
     "- Employment Type: {employment_type}\n"
     "- HR Notes: {hr_notes}\n\n"
     "Policy passages\n<<<\n{policy_passages}\n>>>\n\n"
     "Produce a hierarchical OUTLINE with numbered sections:\n"
     "1. Role Summary (1–2 bullets)\n"
     "2. Key Responsibilities (6–10 bullets, outcome-focused, non-overlapping)\n"
     "3. Must-Have Qualifications (policy-aligned, non-redundant)\n"
     "4. Nice-to-Have Qualifications\n"
     "5. Skills & Tools (grouped logically)\n"
     "6. Hiring Process (stages & approvals per policy)\n"
     "7. Compensation & Benefits (policy phrasing or [POLICY-GAP])\n"
     "8. Location & Work Type\n"
     "9. Legal/Compliance Notes (required statements per policy)\n\n"
     "Constraints:\n"
     "- Be specific and concise.\n"
     "- Do not invent content beyond the passages.\n"
     "- Use [POLICY-GAP] where guidance is missing.\n")
])

jd_creation_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior HR writer. Expand the OUTLINE into a complete, professional Job Description. "
     "Follow the OUTLINE strictly. Do not invent benefits or process steps that are not implied. "
     "No fluff; keep it crisp and usable by HR."),
    ("human",
     "OUTLINE\n<<<\n{outline}\n>>>\n\n"
     "Write the full Job Description with these sections (exact order & headings):\n"
     "- Role Summary (short paragraph)\n"
     "- Responsibilities (6–10 bullets, outcome-oriented, non-overlapping)\n"
     "- Must-Have Qualifications\n"
     "- Nice-to-Have Qualifications\n"
     "- Skills & Tools\n"
     "- Hiring Process\n"
     "- Compensation & Benefits (use neutral placeholders if [POLICY-GAP])\n"
     "- Location & Work Type\n"
     "- Legal/Compliance Notes\n\n"
     "Constraints:\n"
     "- Stay faithful to the OUTLINE; keep placeholders where [POLICY-GAP].\n"
     "- Avoid redundancy and vague phrasing.\n"
     "- Return ONLY the JD text, no extra commentary.")
])

judge_scores_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an impartial HR evaluator. Compare three candidate Job Descriptions (JDs). "
             "Give each JD a numeric score (1–10) and a short feedback. Use the format:\n"
             "JD_A: <score> – <feedback>\n"
             "JD_B: <score> – <feedback>\n"
             "JD_C: <score> – <feedback>"),
            ("human",
             "INTAKE\n"
             "- Title: {title}\n- Department: {department}\n- Seniority: {seniority}\n"
             "- Location: {location}\n- Employment Type: {employment_type}\n- HR Notes: {hr_notes}\n\n"
             "OUTLINE\n<<<\n{outline}\n>>>\n\n"
             "CANDIDATES:\n"
             "- JD_A (model_id: {model_id_A})\n<<<\n{JD_A}\n>>>\n\n"
             "- JD_B (model_id: {model_id_B})\n<<<\n{JD_B}\n>>>\n\n"
             "- JD_C (model_id: {model_id_C})\n<<<\n{JD_C}\n>>>\n")
        ])

load_dotenv()
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION = os.getenv("QDRANT_COLLECTION", "test")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
conn=sqlite3.connect(database="HR.db",check_same_thread=False)
checkpointer=SqliteSaver(conn=conn)

emb_llm=OpenAIEmbeddings()

outline_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# JD Candidate Generator 1 → balanced, strong baseline
jd_llm_1 = ChatOpenAI(model="gpt-5-mini", temperature=0.7)
jd_llm_2 = ChatOpenAI(model="gpt-5-nano", temperature=1.0)

# JD Candidate Generator 3 → conservative, concise
jd_llm_3 = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Judge → must be consistent and reproducible
jd_judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

parser=StrOutputParser()

class JDCreationState(TypedDict):
    title: str
    department: Optional[str]
    seniority: Optional[str]
    location: Optional[str]
    employment_type: Optional[str]
    hr_notes: Optional[str]

    outline: str
    candidates: Annotated[List[Dict[str, Any]], operator.add]

    judgements: Annotated[Dict[str, Dict[str, Any]], operator.or_]  # CHANGED
    winner: Dict[str, Any]

    pass_threshold: float
    passes_threshold: bool

    status: Literal["running","needs_review","done","failed"]
    hr_approved: Optional[bool]     # <-- add
    final_jd: Optional[str] 

def get_vectorstore(chunk):
    try:
        client.get_collection(COLLECTION)
        return QdrantVectorStore(
            client=client,
            collection_name=COLLECTION,
            embedding=emb_llm
        )
    except Exception:
        return QdrantVectorStore.from_documents(
            documents=chunk,
            embedding=emb_llm,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION
        )
    
def formatted_docs(retrieved_doc):
    context = "\n\n".join(d.page_content for d in retrieved_doc)
    return context

def outline_creation(state:JDCreationState):
    title=state['title']
    department=state["department"]
    seniority=state['seniority']
    location=state['location']
    employment_type=state['employment_type']
    hr_notes=state['hr_notes']


    loader=PyPDFLoader('Create_JD/HR_Hiring_Policy.pdf')
    doc = list(loader.lazy_load())
    split=RecursiveCharacterTextSplitter(chunk_size=900,chunk_overlap=150)
    chunk=split.split_documents(doc)
    vectorstore=get_vectorstore(chunk)
    retriever=vectorstore.as_retriever(search_kwargs={'k':6},search_type='mmr')
    parallel_inputs = RunnableParallel({
            "policy_passages": RunnableLambda(build_query) | retriever | RunnableLambda(formatted_docs),
            "title": itemgetter("title"),
            "department": itemgetter("department"),
            "seniority": itemgetter("seniority"),
            "location": itemgetter("location"),
            "employment_type": itemgetter("employment_type"),
            "hr_notes": itemgetter("hr_notes"),
        })

    outline_chain = parallel_inputs | outline_prompt | outline_llm | parser

    result=outline_chain.invoke({
            "title": title,
            "department": department,
            "seniority": seniority,
            "location": location,
            "employment_type": employment_type,
            "hr_notes": hr_notes,
        })
    return {
        'outline':result,
        "candidates": [],    # reset to avoid duplicates on revise cycle
        "judgements": {},
            }

def jd_creation_1(state:JDCreationState):
    outline=state['outline']
    
    jd_chain=jd_creation_prompt|jd_llm_1|parser
    jd_text=jd_chain.invoke({'outline':outline})
    # return {"candidates": [{"model_id": "jd_llm_1", "jd_text": jd_text}]}
    return {"candidates": [{"id": "JD_A", "model_id": "jd_llm_1", "jd_text": jd_text}]}


def jd_creation_2(state:JDCreationState):
    outline=state['outline']
    
    jd_chain=jd_creation_prompt|jd_llm_2|parser
    jd_text=jd_chain.invoke({'outline':outline})
    # return {"candidates": [{"model_id": "jd_llm_2", "jd_text": jd_text}]}
    return {"candidates": [{"id": "JD_B", "model_id": "jd_llm_2", "jd_text": jd_text}]}

def jd_creation_3(state:JDCreationState):
    outline=state['outline']
    
    jd_chain=jd_creation_prompt|jd_llm_3|parser
    jd_text=jd_chain.invoke({'outline':outline})
    # return {"candidates": [{"model_id": "jd_llm_3", "jd_text": jd_text}]}
    return {"candidates": [{"id": "JD_C", "model_id": "jd_llm_3", "jd_text": jd_text}]}


def jd_judge(state:JDCreationState):
    candidates = state["candidates"]
    if len(candidates) < 3:
        return {
            "judgements": {},
            "status": "waiting_for_candidates"
        }
    A, B, C = candidates[0], candidates[1], candidates[2]

    outline=state['outline']

    msgs = judge_scores_prompt.format_messages(
        title=state["title"],
        department=state["department"],
        seniority=state["seniority"],
        location=state["location"],
        employment_type=state["employment_type"],
        hr_notes=state["hr_notes"],
        outline=outline,
        model_id_A=A["model_id"], JD_A=A["jd_text"],
        model_id_B=B["model_id"], JD_B=B["jd_text"],
        model_id_C=C["model_id"], JD_C=C["jd_text"],
    )
    resp = jd_judge_llm.invoke(msgs)
    output_text = (resp.content or "").strip()
    
    judgements: Dict[str, Dict[str, Any]] = {}
    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("JD_"):
            continue
        try:
            jd_id, rest = line.split(":", 1)
        # support en dash or hyphen, split once
            if "–" in rest:
                score_str, feedback = rest.split("–", 1)
            elif "-" in rest:
                score_str, feedback = rest.split("-", 1)
            else:
                continue
            judgements[jd_id.strip()] = {
                "score": float(score_str.strip().replace("/10","")),
                "feedback": feedback.strip(),
            }
        except Exception:
            continue
    return {
            "judgements": judgements,   # <-- now a dict
        }

def jd_score(state: JDCreationState):
    judgements = state.get("judgements") or {}
    if not judgements:
        return {}

    threshold = state["pass_threshold"]

    # pick highest scorer
    winner_id, winner_data = max(
        judgements.items(),
        key=lambda item: item[1].get("score", 0)
    )

    winner = {
        "id": winner_id,
        "score": winner_data.get("score", 0.0),
        "feedback": winner_data.get("feedback", ""),
        "jd_text": None,
        "model_id": None,
    }

    # attach JD text + model_id for the winner
    for cand in state.get("candidates", []):
        if cand.get("id") == winner_id:
            winner["jd_text"] = cand.get("jd_text")
            winner["model_id"] = cand.get("model_id")
            break

    # build feedback summary (used if we need to refine)
    outline_feedback = "\n".join(
        f"- {jid}: {data.get('score', 0)}/10 — {data.get('feedback','').strip()}"
        for jid, data in judgements.items()
    ).strip()

    passes_threshold = bool(winner["score"] >= threshold)

    updates = {
        "winner": winner,
        "passes_threshold": passes_threshold,
        "status": "ready_for_hr" if passes_threshold else "running",
    }

    if not passes_threshold:
        updates.update({
            "outline_feedback": outline_feedback,
            "candidates": [],
            "judgements": {},
        })

    return updates

def decision_router(state: JDCreationState) -> str:
    pt = state.get("passes_threshold")
    if pt is False:
        return "refine"     # failed score → regenerate via outline
    if pt is True:
        approved = state.get("hr_approved")
        if approved is True:
            return "approve"
        if approved is False:
            return "revise" # HR rejected → back to outline
        return "await"      # show JD to HR and wait for input
    return "await"
def hr_approve(state:JDCreationState):
    jd=state["winner"]
    if state.get("hr_approved", False):
        return {
                "final_jd": jd["jd_text"],
                "status": "done"
            }
    else:
        return {
                "final_jd": None,
                "status": "needs_review"
        }

def await_hr(state: JDCreationState):

    return {}


graph=StateGraph(JDCreationState)
graph.add_node('outline_creation',outline_creation)
graph.add_node('jd_creation_1',jd_creation_1)
graph.add_node('jd_creation_2',jd_creation_2)
graph.add_node('jd_creation_3',jd_creation_3)
graph.add_node('jd_judge',jd_judge)

##############################################
graph.add_node('jd_score', jd_score)
graph.add_node('hr_approve', hr_approve)
graph.add_node('await_hr', await_hr)
##############################################

graph.add_edge(START,'outline_creation')
graph.add_edge('outline_creation','jd_creation_1')
graph.add_edge('outline_creation','jd_creation_2')
graph.add_edge('outline_creation','jd_creation_3')
graph.add_edge('jd_creation_1','jd_judge')
graph.add_edge('jd_creation_2','jd_judge')
graph.add_edge('jd_creation_3','jd_judge')
graph.add_edge('jd_judge', 'jd_score')

###############################################

graph.add_conditional_edges(
    "jd_score",
    decision_router,
    {
        "refine": "outline_creation",
        "await": "await_hr",
        "approve": "hr_approve",
        "revise": "outline_creation",
    },
)
graph.add_edge('hr_approve', END)

jd_workflow=graph.compile(checkpointer=checkpointer)

def retrieve_threads():
    all_thread=set()

    for checkpoint in checkpointer.list(None):
        all_thread.add(checkpoint.config['configurable']['thread_id'])

    return list(all_thread)

initial_state = {
    "title": "AI Engineer",
    "department": "R&D",
    "seniority": "Mid-level",
    "location": "Karachi",
    "employment_type": "Full-time",
    "hr_notes": "Focus on Python, ML, and teamwork.",
    "pass_threshold": 6.0,
    "status": "running",
    "candidates": [],
    "judgements": {},
    "hr_approved": None,  # auto-approve for now (so flow doesn’t stall)

}
thread_id = str(uuid.uuid4())

result = jd_workflow.invoke(
    initial_state,
    config={"configurable": {"thread_id": thread_id}}
)

if result.get("status") == "ready_for_hr" and not result.get("final_jd"):
    print("\n=== Winner JD (for HR review) ===\n")
    print(result["winner"]["jd_text"])

    # Ask HR
    ans = input("\nAre you satisfied with this JD? (y = approve / n = revise): ").strip().lower()
    approved = ans.startswith("y")

    # Feed the decision back on the SAME thread_id
    result = jd_workflow.invoke(
        {"hr_approved": approved},
        config={"configurable": {"thread_id": thread_id}}
    )

# If approved, show ONLY the final JD text
if result.get("status") == "done" and result.get("final_jd"):
    print("\n=== Final JD ===\n")
    print(result["final_jd"])
    with open("Create_JD/jd_output.txt", "w", encoding="utf-8") as f:
        f.write(str(result["final_jd"]))
else:
    # If HR rejected, the graph has already gone back to outline to regenerate.
    # You can loop here if you want repeated HR review cycles.
    print("\nJD sent back for revision. Regenerating based on outline...\n")