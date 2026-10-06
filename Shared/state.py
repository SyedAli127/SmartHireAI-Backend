from typing import TypedDict, Optional, Annotated, Any, Literal, List, Dict, NotRequired
import operator
from Shared.schemas import EssentialsSchema,PlatformBriefSchema,AdVariantSchema,CombinedAdSchema


class JDCreationState(TypedDict):
    title: str
    department: Optional[str]
    seniority: Optional[str]
    location: Optional[str]
    employment_type: Optional[str]
    hr_notes: Optional[str]
    review_feedback: Optional[str]

    outline: str
    candidates: Annotated[List[Dict[str, Any]], operator.add]

    judgements: Annotated[Dict[str, Dict[str, Any]], operator.or_]
    winner: Dict[str, Any]

    pass_threshold: float
    passes_threshold: bool

    status: Literal["running","needs_review","done","failed",'await_hr']
    hr_approved: Optional[bool]
    final_jd: Optional[str]

    platforms: List[Literal["LinkedIn", "Facebook", "Rozee.pk", "Indeed","Instagram","X"]]
    essentials: Optional[EssentialsSchema]
    platform_briefs: Optional[List[PlatformBriefSchema]]
    ad_variants: Optional[List[AdVariantSchema]]
    checks: Optional[dict]
    ad_pack_id: Optional[str]
    created_at: Optional[str]

    combined_ads: Optional[List[CombinedAdSchema]]
    output_dir: Optional[str]

    approved_ads: Optional[List[CombinedAdSchema]]
    posting_urls: Annotated[Dict[str, str], operator.or_]

    # --- Ad pipeline stages ---
    orchestration_status: Literal[
        'JD_Gen_Agent',
        'AD_Generation_Agent',
        'AD_Posting_Agent',
        'AD_AWAIT_HR',
        'AD_TEXT_AWAIT_HR',
        'AD_IMAGE_SCRIPT_AWAIT_HR',
        'AD_IMAGE_AWAIT_HR',
        # --- Full hiring pipeline stages (set after ad posting) ---
        'COLLECTING_APPLICATIONS',
        'SCREENING',
        'INTERVIEWS_DISPATCHED',
        'INTERVIEWS_COMPLETE',
        'OFFER_SENT',
        'PIPELINE_COMPLETE',
    ]
    thread_id: str
    iteration: int
    job_form_link: Optional[str]

    # Pipeline tracking fields populated by HiringPipeline
    job_id: NotRequired[Optional[str]]
    correlation_id: NotRequired[Optional[str]]
    min_applications_threshold: NotRequired[int]
    shortlist_top_n: NotRequired[int]

    skip_hr_wait: NotRequired[bool]
    require_ad_approval: NotRequired[bool]
    ad_approved: NotRequired[bool]
    generated_image_paths: Optional[dict]

    # --- New ad generation fields (phase-based) ---
    # Phase 1: per-platform text variants {platform: {"A": {headline,text,cta,hashtags}, "B": {...}}}
    text_variants: NotRequired[Optional[Dict[str, Any]]]
    approved_texts: NotRequired[Optional[Dict[str, Any]]]   # {platform: selected_variant_dict}
    ad_text_feedback: NotRequired[Optional[str]]

    # Phase 2: image script variants {"A": "prompt...", "B": "prompt..."}
    image_script_variants: NotRequired[Optional[Dict[str, str]]]
    approved_image_script: NotRequired[Optional[str]]
    image_script_feedback: NotRequired[Optional[str]]

    # Phase 3: generated image
    generated_image_url: NotRequired[Optional[str]]
    ad_image_approved: NotRequired[Optional[bool]]
    image_feedback: NotRequired[Optional[str]]

    temp_image_path: NotRequired[Optional[str]]



