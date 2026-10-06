from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel
import sqlite3, uuid, json, os
from dotenv import load_dotenv

load_dotenv()

# ------------------ Pydantic Schemas ------------------ #

class EssentialsSchema(BaseModel):
    role: str
    location: str
    skills: List[str]
    benefits: List[str]
    impact: str

class PlatformBriefSchema(BaseModel):
    platform: str
    tone: str
    length: str
    cta_style: str
    hashtags_required: bool
    extra_assets: List[str]

class PlatformBriefList(BaseModel):
    briefs: List[PlatformBriefSchema]

class AdVariantSchema(BaseModel):
    platform: str
    variant: Literal["A", "B"]
    headline: str
    text: str
    cta: str
    hashtags: List[str]
    image_prompt: Optional[str] = None
    alt_text: Optional[str] = None
    video_script: Optional[str] = None

class AdVariantList(BaseModel):
    ads: List[AdVariantSchema]

# ------------------ State ------------------ #

class AdCreationState(TypedDict):
    jd: str
    platforms: List[Literal["LinkedIn", "Facebook", "Rozee.pk", "Indeed", "CareersPage"]]
    essentials: Optional[EssentialsSchema]
    platform_briefs: Optional[List[PlatformBriefSchema]]
    ad_variants: Optional[List[AdVariantSchema]]
    checks: Optional[dict]
    ad_pack_id: Optional[str]
    created_at: Optional[str]

# ------------------ LLMs ------------------ #

llm_extract   = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
llm_brief     = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
llm_adgen     = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

llm_extract_structured = llm_extract.with_structured_output(EssentialsSchema)
llm_brief_structured   = llm_brief.with_structured_output(PlatformBriefList)
llm_adgen_structured   = llm_adgen.with_structured_output(AdVariantList)

# ------------------ Prompts ------------------ #

extract_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant that extracts structured essentials from a Job Description (JD). "
     "Respond ONLY with valid JSON."),
    ("human", "JD:\n{jd}\n\nExtract the essentials now.")
])

brief_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a recruitment marketing strategist. "
     "Respond ONLY with a JSON array of platform briefs."),
    ("human",
     "JD essentials:\n{essentials}\n\n"
     "Platforms to target: {platforms}\n\n"
     "Keys: platform, tone, length, cta_style, hashtags_required, extra_assets.\n"
     "Output JSON array only.")
])

adgen_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a recruitment ad generator. "
     "Always create TWO variants:\n"
     "- Variant A: Skills/tech-focused hook\n"
     "- Variant B: Benefits/impact-focused hook\n"
     "Respond ONLY with valid JSON array of ads."),
    ("human",
     "Essentials:\n{essentials}\n\n"
     "Platform Brief:\n{brief}\n\n"
     "JD reference:\n{jd}\n\n"
     "Now write the ads.")
])

# ------------------ Node Functions ------------------ #

def extract_essentials_node(state: AdCreationState):
    msgs = extract_prompt.format_messages(jd=state["jd"])
    result = llm_extract_structured.invoke(msgs)
    return {"essentials": result}

def briefs_node(state: AdCreationState):
    msgs = brief_prompt.format_messages(
        essentials=state["essentials"],
        platforms=", ".join(state["platforms"])
    )
    result = llm_brief_structured.invoke(msgs)
    return {"platform_briefs": result.briefs}

def adgen_node(state: AdCreationState):
    essentials = state["essentials"]
    jd = state["jd"]
    briefs = state["platform_briefs"]
    all_ads: List[AdVariantSchema] = []

    for brief in briefs:
        msgs = adgen_prompt.format_messages(
            essentials=essentials,
            brief=brief,
            jd=jd
        )
        result = llm_adgen_structured.invoke(msgs)
        for ad in result.ads:
            ad.platform = brief.platform  # enforce platform if missing
            all_ads.append(ad)

    return {"ad_variants": all_ads}

def checks_node(state: AdCreationState):
    ads = state.get("ad_variants") or []
    forbidden = ["rockstar", "ninja", "guru"]
    hits = []
    length_ok = True

    for ad in ads:
        txt = (ad.headline or "") + " " + (ad.text or "")
        for bad in forbidden:
            if bad.lower() in txt.lower():
                hits.append({"platform": ad.platform, "variant": ad.variant, "phrase": bad})
        if len(txt.split()) < 30:
            length_ok = False

    return {
        "checks": {
            "brand_ok": len(hits) == 0,
            "dei_ok": len(hits) == 0,
            "length_ok": length_ok,
            "forbidden_phrases_found": hits
        }
    }

def package_node(state: AdCreationState):
    return {
        "ad_pack_id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

# ------------------ Graph ------------------ #

conn = sqlite3.connect("adgen.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(AdCreationState)
graph.add_node("extract_essentials", extract_essentials_node)
graph.add_node("briefs", briefs_node)
graph.add_node("adgen", adgen_node)
graph.add_node("checks", checks_node)
graph.add_node("package", package_node)

graph.add_edge(START, "extract_essentials")
graph.add_edge("extract_essentials", "briefs")
graph.add_edge("briefs", "adgen")
graph.add_edge("adgen", "checks")
graph.add_edge("checks", "package")
graph.add_edge("package", END)

adgen_workflow = graph.compile(checkpointer=checkpointer)

# ------------------ Run ------------------ #

with open("Create_JD/jd_output.txt", "r", encoding="utf-8") as file:
    content = file.read()

init_state = {
    "jd": content,
    "platforms": ["LinkedIn", "Facebook", "Rozee.pk"]
}

thread_id = str(uuid.uuid4())
result = adgen_workflow.invoke(init_state, config={"configurable": {"thread_id": thread_id}})

print("\n=== Final Ad Pack State ===\n")
ads = result.get("ad_variants", [])
for ad in ads:
    print("\n--- Ad Variant ---")
    print(f"Platform: {ad.platform}")
    print(f"Variant: {ad.variant}")
    print(f"Headline: {ad.headline}")
    print(f"Text: {ad.text}")
    print(f"CTA: {ad.cta}")
    print(f"Hashtags: {' '.join(ad.hashtags)}")

# ------------------ Save Ads ------------------ #

output_dir = "Generated_Ads"
os.makedirs(output_dir, exist_ok=True)

for idx, ad in enumerate(ads, start=1):
    platform = ad.platform
    variant = ad.variant
    base_name = f"{platform}_{variant}_{idx}"

    # Save text ad
    if ad.text:
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Platform: {platform}\n")
            f.write(f"Variant: {variant}\n")
            f.write(f"Headline: {ad.headline}\n\n")
            f.write(ad.text + "\n\n")
            f.write(f"CTA: {ad.cta}\n")
            f.write(f"Hashtags: {' '.join(ad.hashtags)}\n")
        print(f"✅ Saved text ad → {txt_path}")

    # Save image prompt if exists
    if ad.image_prompt:
        img_path = os.path.join(output_dir, f"{base_name}_image.txt")
        with open(img_path, "w", encoding="utf-8") as f:
            f.write(ad.image_prompt)
        print(f"🖼️ Saved image prompt → {img_path}")

    # Save video script if exists
    if ad.video_script:
        vid_path = os.path.join(output_dir, f"{base_name}_video.txt")
        with open(vid_path, "w", encoding="utf-8") as f:
            f.write(ad.video_script)
        print(f"🎬 Saved video script → {vid_path}")
