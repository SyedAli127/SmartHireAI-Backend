import os, uuid, base64, textwrap, io
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from services import s3_service

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_ad_image(
    image_prompt: str,
    output_dir: str,
    base_name: str,
    approved_ad: dict,
    platform: str = "Instagram"
) -> str:
    hardcoded_url = os.getenv("AD_IMAGE_URL")
    if hardcoded_url:
        print(f"   [IMAGE GEN] Using hardcoded image URL for {platform}: {hardcoded_url}")
        return hardcoded_url

    try:
        # ── 1. Set size per platform ──────────────────────────────
        sizes = {
            "Instagram": "1024x1024",
            "Facebook":  "1792x1024",
            "X":         "1792x1024",
        }
        size = sizes.get(platform, "1024x1024")

        # ── 2. Generate background via DALL-E 3 ───────────────────
        print(f"   [IMAGE GEN] Generating background for {platform}...")
        response = client.images.generate(
            model="dall-e-3",
            prompt=image_prompt + " (no text, no words, no letters in the image)",
            size=size,
            quality="hd",
            response_format="b64_json",
            n=1
        )
        img_bytes = base64.b64decode(response.data[0].b64_json)

        # ── 3. Open image and set up drawing ──────────────────────
        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        W, H = img.size
        draw = ImageDraw.Draw(img)

        # ── 4. Dark gradient overlay at bottom half ───────────────
        gradient = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        grad_draw = ImageDraw.Draw(gradient)

        for i in range(H // 2, H):
            alpha = int(210 * ((i - H // 2) / (H // 2)))
            grad_draw.line([(0, i), (W, i)], fill=(0, 0, 0, alpha))

        img = Image.alpha_composite(img, gradient)
        draw = ImageDraw.Draw(img)

        # ── 5. Load fonts ─────────────────────────────────────────
        def load_font(size):
            for font_name in ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "arial.ttf"]:
                try:
                    return ImageFont.truetype(font_name, size)
                except Exception:
                    continue
            return ImageFont.load_default()

        font_headline = load_font(52 if W > 1024 else 44)
        font_body     = load_font(30 if W > 1024 else 26)
        font_cta      = load_font(34 if W > 1024 else 28)
        font_hashtag  = load_font(22)

        # ── 6. Extract ad content ─────────────────────────────────
        headline = approved_ad.get("headline", "").upper()
        body     = approved_ad.get("text", "")
        cta      = approved_ad.get("cta", "")
        hashtags = "  ".join(approved_ad.get("hashtags", [])[:4])

        # ── 7. Draw text from bottom up ───────────────────────────
        pad = 50
        y   = H - 50

        if hashtags:
            y -= 30
            draw.text((pad, y), hashtags, font=font_hashtag, fill=(180, 180, 180, 230))
            y -= 45

        if cta:
            draw.rectangle([(pad, y - 4), (pad + 6, y + 38)], fill=(255, 200, 0, 255))
            draw.text((pad + 18, y), cta, font=font_cta, fill=(255, 215, 0, 255))
            y -= 55

        if body:
            wrap_width = 55 if W > 1024 else 42
            lines = textwrap.wrap(body, width=wrap_width)
            for line in reversed(lines[:3]):
                draw.text((pad, y), line, font=font_body, fill=(230, 230, 230, 255))
                y -= 38
            y -= 10

        if headline:
            wrap_width_h = 38 if W > 1024 else 28
            h_lines = textwrap.wrap(headline, width=wrap_width_h)
            for line in reversed(h_lines[:2]):
                draw.text((pad + 2, y + 2), line, font=font_headline, fill=(0, 0, 0, 180))
                draw.text((pad, y), line, font=font_headline, fill=(255, 255, 255, 255))
                y -= 60

        # ── 8. Upload final image to S3 ───────────────────────────
        final = img.convert("RGB")
        buf = io.BytesIO()
        final.save(buf, format="PNG", quality=95)
        buf.seek(0)
        s3_key = f"{output_dir}/{base_name}_generated.png"
        s3_url = s3_service.upload_fileobj(buf, s3_key, "image/png")

        print(f"   [IMAGE GEN] ✅ Final ad image → {s3_url}")
        return s3_url

    except Exception as e:
        print(f"   [IMAGE GEN] ❌ Error: {e}")
        return None
