from typing import Optional,Literal,List
from pydantic import BaseModel, field_validator

class EssentialsSchema(BaseModel):
    role: str
    location: str
    skills: List[str]
    benefits: List[str]
    impact: str


class AdTextSchema(BaseModel):
    headline: str
    text: str
    cta: str
    hashtags: List[str]

class PlatformBriefSchema(BaseModel):
    platform: str
    tone: str
    length: str
    cta_style: str
    hashtags_required: bool
    extra_assets: List[str]

class PlatformBriefList(BaseModel):
    briefs: List[PlatformBriefSchema]

# class AdVariantSchema(BaseModel):
    platform: str
    variant: Literal["A", "B"]
    headline: str
    text: str
    cta: str
    hashtags: List[str]
    image_prompt: Optional[str] = None
    alt_text: Optional[str] = None
    video_script: Optional[str] = None

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

    @field_validator("image_prompt")
    @classmethod
    def image_prompt_must_be_detailed(cls, v):
        if v is not None and len(v.split()) < 30:
            raise ValueError(
                f"image_prompt is too short ({len(v.split())} words). "
                f"Must be at least 30 words with scene, lighting, style, and subject details."
            )
        return v


class AdVariantList(BaseModel):
    ads: List[AdVariantSchema]


class CombinedAdSchema(BaseModel):
    platform: str
    variant: str
    headline: Optional[str] = None
    text: Optional[str] = None
    cta: Optional[str] = None
    hashtags: List[str] = []
    image_prompt: Optional[str] = None
    video_script: Optional[str] = None
    text_file: Optional[str] = None   # path to saved text file
    image_file: Optional[str] = None  # path to saved image prompt
    video_file: Optional[str] = None  # path to saved video script
    generated_image_path: Optional[str] = None  # ← add this



class PostResultSchema(BaseModel):
    platform: str
    status: Literal["success", "error", "skipped"]
    post_id: Optional[str] = None
    url: Optional[str] = None
    details: Optional[str] = None

class AdPostSummary(BaseModel):
    total_platforms: int
    successful: int
    failed: int
    results: List[PostResultSchema]
