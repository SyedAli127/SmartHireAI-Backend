from langgraph.graph import START, StateGraph, END
from typing import Dict, Any
from datetime import datetime
import uuid, os
import concurrent.futures

from Shared.prompts import (
    ad_generation_extraction_prompt,
    ad_generation_brief_prompt,
    ad_text_prompt,
    ad_image_script_prompt,
)
from Shared.config import (
    ad_generation_extract_llm,
    ad_generation_brief_llm,
    ad_text_llm_a,
    ad_text_llm_b,
    ad_image_script_llm,
)
from Shared.schemas import EssentialsSchema, PlatformBriefList, AdTextSchema, CombinedAdSchema
from Shared.state import JDCreationState
from services.image_generation_service import generate_ad_image
from services import s3_service


class AD_Generation:
    def __init__(self, checkpointer):
        self.llm_extract = ad_generation_extract_llm.with_structured_output(EssentialsSchema)
        self.llm_brief = ad_generation_brief_llm.with_structured_output(PlatformBriefList)
        self.llm_text_a = ad_text_llm_a.with_structured_output(AdTextSchema)
        self.llm_text_b = ad_text_llm_b.with_structured_output(AdTextSchema)

        graph = StateGraph(JDCreationState)

        graph.add_node("extract_essentials", self.extract_essentials_node)
        graph.add_node("briefs", self.briefs_node)
        graph.add_node("text_gen", self.text_gen_node)
        graph.add_node("pause_text", self.pause_text_node)
        graph.add_node("image_script_gen", self.image_script_gen_node)
        graph.add_node("pause_image_script", self.pause_image_script_node)
        graph.add_node("image_gen", self.image_gen_node)
        graph.add_node("pause_image", self.pause_image_node)
        graph.add_node("job_form", self.job_form_node)
        graph.add_node("finalize", self.finalize_node)

        graph.add_conditional_edges(START, self._phase_router)

        graph.add_edge("extract_essentials", "briefs")
        graph.add_edge("briefs", "text_gen")
        graph.add_edge("text_gen", "pause_text")
        graph.add_edge("pause_text", END)

        graph.add_edge("image_script_gen", "pause_image_script")
        graph.add_edge("pause_image_script", END)

        graph.add_edge("image_gen", "pause_image")
        graph.add_edge("pause_image", END)

        graph.add_edge("job_form", "finalize")
        graph.add_edge("finalize", END)

        self.adgen_workflow = graph.compile()

    def run(self, state: JDCreationState):
        return self.adgen_workflow.invoke(state)

    # ── Phase router ─────────────────────────────────────────────────────────

    def _phase_router(self, state: JDCreationState) -> str:
        # Phase 4: image approved → finalize
        if state.get("ad_image_approved") is True:
            return "job_form"

        # Phase 3 retry: image rejected with feedback → regenerate
        if state.get("image_feedback") and state.get("approved_image_script"):
            return "image_gen"

        # Phase 3: image script approved but no image yet → generate
        if state.get("approved_image_script") and not state.get("generated_image_url"):
            return "image_gen"

        # Phase 3 pause: image exists, awaiting HR decision
        if state.get("generated_image_url") and state.get("ad_image_approved") is None:
            return "pause_image"

        # Phase 2 retry: image script rejected → regenerate
        if state.get("image_script_feedback") and state.get("approved_texts"):
            return "image_script_gen"

        # Phase 2: texts approved, no image scripts yet → generate scripts
        if state.get("approved_texts") and not state.get("image_script_variants"):
            return "image_script_gen"

        # Phase 2 pause: scripts generated, awaiting HR decision
        if (
            state.get("image_script_variants")
            and not state.get("approved_image_script")
            and not state.get("image_script_feedback")
        ):
            return "pause_image_script"

        # Phase 1 retry: text rejected → regenerate (briefs already in state)
        if state.get("ad_text_feedback") and state.get("platform_briefs"):
            return "text_gen"

        # Phase 1 pause: texts generated, awaiting HR decision
        if (
            state.get("text_variants")
            and not state.get("approved_texts")
            and not state.get("ad_text_feedback")
        ):
            return "pause_text"

        # Phase 1 start: nothing yet
        return "extract_essentials"

    # ── Phase 1: per-platform text generation ────────────────────────────────

    def extract_essentials_node(self, state: JDCreationState):
        msgs = ad_generation_extraction_prompt.format_messages(jd=state["final_jd"])
        result = self.llm_extract.invoke(msgs)
        return {"essentials": result}

    def briefs_node(self, state: JDCreationState):
        msgs = ad_generation_brief_prompt.format_messages(
            essentials=state["essentials"],
            platforms=", ".join(state["platforms"]),
        )
        result = self.llm_brief.invoke(msgs)
        return {"platform_briefs": result.briefs}

    def text_gen_node(self, state: JDCreationState):
        essentials = state["essentials"]
        jd = state["final_jd"]
        briefs = state["platform_briefs"]
        feedback = state.get("ad_text_feedback") or "None"

        def _gen_a(brief) -> AdTextSchema:
            msgs = ad_text_prompt.format_messages(
                platform=brief.platform,
                brief=brief,
                essentials=essentials,
                jd=jd,
                focus="SKILLS — lead with the candidate's craft, technical mastery, and professional achievement",
                feedback=feedback,
            )
            return self.llm_text_a.invoke(msgs)

        def _gen_b(brief) -> AdTextSchema:
            msgs = ad_text_prompt.format_messages(
                platform=brief.platform,
                brief=brief,
                essentials=essentials,
                jd=jd,
                focus="BENEFITS — lead with growth, culture, mission, impact, and what makes this role worth joining",
                feedback=feedback,
            )
            return self.llm_text_b.invoke(msgs)

        text_variants: Dict[str, Any] = {}

        with concurrent.futures.ThreadPoolExecutor() as pool:
            futures_a = {b.platform: pool.submit(_gen_a, b) for b in briefs}
            futures_b = {b.platform: pool.submit(_gen_b, b) for b in briefs}

        for brief in briefs:
            p = brief.platform
            result_a = futures_a[p].result()
            result_b = futures_b[p].result()
            text_variants[p] = {
                "A": result_a.model_dump() if hasattr(result_a, "model_dump") else result_a,
                "B": result_b.model_dump() if hasattr(result_b, "model_dump") else result_b,
            }
            print(f"[TEXT GEN] {p} — A & B variants generated")

        return {
            "text_variants": text_variants,
            "ad_text_feedback": None,
            # reset downstream so re-approval is required if text is regenerated
            "approved_texts": None,
            "image_script_variants": None,
            "approved_image_script": None,
            "generated_image_url": None,
            "ad_image_approved": None,
        }

    def pause_text_node(self, _state: JDCreationState):
        return {"orchestration_status": "AD_TEXT_AWAIT_HR"}

    # ── Phase 2: global image script generation (2 style variants) ───────────

    def image_script_gen_node(self, state: JDCreationState):
        essentials = state["essentials"]
        approved_texts = state.get("approved_texts") or {}
        feedback = state.get("image_script_feedback") or "None"

        # Build a representative ad sample from the first approved platform
        sample_platform = next(iter(approved_texts), None)
        ad_sample = ""
        if sample_platform:
            t = approved_texts[sample_platform]
            ad_sample = (
                f"Headline: {t.get('headline', '')}\n"
                f"Text: {t.get('text', '')}\n"
                f"CTA: {t.get('cta', '')}"
            )

        def _gen_script(style: str) -> str:
            msgs = ad_image_script_prompt.format_messages(
                style=style,
                essentials=essentials,
                ad_sample=ad_sample,
                feedback=feedback,
            )
            result = ad_image_script_llm.invoke(msgs)
            return result.content if hasattr(result, "content") else str(result)

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future_pro = pool.submit(_gen_script, "PROFESSIONAL — clean, corporate, polished")
            future_cre = pool.submit(_gen_script, "CREATIVE — bold, vibrant, editorial")

        image_script_variants = {
            "professional": future_pro.result(),
            "creative": future_cre.result(),
        }
        print("[IMAGE SCRIPT GEN] Professional & Creative variants generated")

        return {
            "image_script_variants": image_script_variants,
            "image_script_feedback": None,
            # reset downstream
            "approved_image_script": None,
            "generated_image_url": None,
            "ad_image_approved": None,
        }

    def pause_image_script_node(self, state: JDCreationState):
        return {"orchestration_status": "AD_IMAGE_SCRIPT_AWAIT_HR"}

    # ── Phase 3: image generation + final approval ───────────────────────────

    def image_gen_node(self, state: JDCreationState):
        image_script = state["approved_image_script"]
        title = state["title"]
        safe_title = title.replace(" ", "_")
        folder_name = f"{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = f"Generated_Ads/{folder_name}"

        # Build text overlay from the first approved platform's ad
        approved_texts = state.get("approved_texts") or {}
        sample_platform = next(iter(approved_texts), None)
        approved_ad_sample: dict = {}
        if sample_platform:
            t = approved_texts[sample_platform]
            approved_ad_sample = {
                "headline": t.get("headline", ""),
                "text":     t.get("text", ""),
                "cta":      t.get("cta", ""),
                "hashtags": t.get("hashtags", []),
            }

        print(f"[IMAGE GEN] Generating shared ad image...")
        generated_image_url = generate_ad_image(
            image_prompt=image_script,
            output_dir=output_dir,
            base_name="ad_image",
            approved_ad=approved_ad_sample,
            platform="Instagram",
        )
        print(f"[IMAGE GEN] → {generated_image_url}")

        return {
            "generated_image_url": generated_image_url,
            "image_feedback": None,
            "ad_image_approved": None,
            "output_dir": output_dir,
        }

    def pause_image_node(self, state: JDCreationState):
        return {"orchestration_status": "AD_IMAGE_AWAIT_HR"}

    # ── Phase 4: finalize & upload ───────────────────────────────────────────

    def job_form_node(self, state: JDCreationState):
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        apply_id = state.get("job_id") or state.get("thread_id", "unknown")
        return {"job_form_link": f"{frontend_url}/apply/{apply_id}"}

    def finalize_node(self, state: JDCreationState):
        approved_texts = state.get("approved_texts") or {}
        image_url = state.get("generated_image_url")
        image_script = state.get("approved_image_script", "")
        output_dir = state.get("output_dir") or (
            f"Generated_Ads/{state['title'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        approved_ads = []
        for platform, text_data in approved_texts.items():
            variant = text_data.get("_selected_variant", "A")
            ad_entry = CombinedAdSchema(
                platform=platform,
                variant=variant,
                headline=text_data.get("headline", ""),
                text=text_data.get("text", ""),
                cta=text_data.get("cta", ""),
                hashtags=text_data.get("hashtags", []),
                image_prompt=image_script,
                generated_image_path=image_url,
            )

            # Upload text file to S3
            base_name = f"{platform}_{variant}"
            ad_text_content = (
                f"Platform: {platform}\n"
                f"Variant: {variant}\n"
                f"Headline: {ad_entry.headline}\n\n"
                f"{ad_entry.text}\n\n"
                f"CTA: {ad_entry.cta}\n"
                f"Hashtags: {' '.join(ad_entry.hashtags)}\n"
                f"Apply Link: {state.get('job_form_link', '')}\n"
            )
            if image_url:
                ad_text_content += f"Generated Image: {image_url}\n"

            s3_service.upload_text(ad_text_content, f"{output_dir}/{base_name}.txt")

            if image_script:
                s3_service.upload_text(image_script, f"{output_dir}/{base_name}_image_script.txt")

            approved_ads.append(ad_entry)
            print(f"[FINALIZE] {platform} → {output_dir}/{base_name}.txt")

        return {
            "approved_ads": approved_ads,
            "output_dir": output_dir,
            "temp_image_path": image_url,
            "ad_pack_id": str(uuid.uuid4()),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "orchestration_status": "AD_Generation_Agent",
        }
