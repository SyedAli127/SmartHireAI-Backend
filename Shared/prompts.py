from langchain_core.prompts import ChatPromptTemplate

jd_outline_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior HR policy specialist. Produce a structured Job Description OUTLINE using two sources:\n"
     "1. POLICY PASSAGES — these govern compliance, hiring process, approvals, benefits, and legal requirements. Follow them strictly.\n"
     "2. YOUR PROFESSIONAL HR KNOWLEDGE — use this to fill role-specific content (responsibilities, qualifications, skills) "
     "that the policy does not cover. Write concrete, realistic content appropriate for the role, seniority, and department.\n\n"
     "CRITICAL — HR FEEDBACK: If HR Feedback is provided, it contains MANDATORY changes requested by the hiring manager. "
     "You MUST apply every change exactly as specified. These override policy passages and your own judgement. "
     "Do not ignore, soften, or partially apply any requested change.\n\n"
     "Only use [POLICY-GAP] for data that is genuinely org-specific and unknowable from outside the organisation "
     "(e.g. exact salary bands, internal system names, unless feedback specifies them). Never use [POLICY-GAP] for role responsibilities, "
     "qualifications, or skills — always write those from your HR knowledge.\n"
     "Return ONLY the outline text."),
    ("human",
     "Intake\n"
     "- Title: {title}\n"
     "- Department: {department}\n"
     "- Seniority: {seniority}\n"
     "- Location: {location}\n"
     "- Employment Type: {employment_type}\n"
     "- HR Notes: {hr_notes}\n"
     "- HR Feedback(If Available): {review_feedback}\n\n"
     "Policy passages\n<<<\n{policy_passages}\n>>>\n\n"
     "Produce a hierarchical OUTLINE with numbered sections:\n"
     "1. Role Summary (1–2 bullets)\n"
     "2. Key Responsibilities (6–10 bullets, outcome-focused, non-overlapping, role-specific)\n"
     "3. Must-Have Qualifications (concrete, role-appropriate)\n"
     "4. Nice-to-Have Qualifications\n"
     "5. Skills & Tools (grouped logically, specific to the role)\n"
     "6. Hiring Process (stages & approvals per policy)\n"
     "7. Compensation & Benefits (policy phrasing; use [POLICY-GAP] only for exact salary figures)\n"
     "8. Location & Work Type\n"
     "9. Legal/Compliance Notes (required statements per policy)\n\n"
     "Constraints:\n"
     "- Sections 2–5 must be fully written using your HR knowledge — no [POLICY-GAP] allowed there.\n"
     "- Sections 6, 7, 9 follow the policy; use [POLICY-GAP] only where the policy is truly silent on org-specific data.\n"
     "- Be specific and concise.\n")
])

jd_creation_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior HR writer. Expand the OUTLINE into a complete, professional Job Description. "
     "Follow the OUTLINE strictly. No fluff; keep it crisp and usable by HR.\n\n"
     "CRITICAL — HR FEEDBACK: If HR Feedback is provided, it contains MANDATORY changes requested by the hiring manager. "
     "You MUST apply every change exactly as specified before writing the JD. These override the outline. "
     "Do not ignore, soften, or partially apply any requested change.\n\n"
     "If the OUTLINE contains [POLICY-GAP] in Responsibilities, Qualifications, or Skills & Tools, "
     "replace it with concrete, realistic content appropriate for the role — never carry [POLICY-GAP] "
     "into those sections of the final JD. Only preserve [POLICY-GAP] for genuinely org-specific "
     "unknowables such as exact salary figures or internal system names (unless feedback specifies them)."),
    ("human",
     "OUTLINE\n<<<\n{outline}\n>>>\n"
     "HR Feedback (MANDATORY — apply all changes exactly): {review_feedback}\n\n"

     "Write the full Job Description with these sections (exact order & headings):\n"
     "- Role Summary (short paragraph)\n"
     "- Responsibilities (6–10 bullets, outcome-oriented, non-overlapping)\n"
     "- Must-Have Qualifications\n"
     "- Nice-to-Have Qualifications\n"
     "- Skills & Tools\n"
     "- Hiring Process\n"
     "- Compensation & Benefits\n"
     "- Location & Work Type\n"
     "- Legal/Compliance Notes\n\n"
     "Constraints:\n"
     "- Responsibilities, Qualifications, and Skills must be fully written — no [POLICY-GAP].\n"
     "- Avoid redundancy and vague phrasing.\n"
     "- Return ONLY the JD text, no extra commentary.")
])

jd_judge_scores_prompt = ChatPromptTemplate.from_messages([
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
             "OUTLINE\n<<<\n{outline}\n>>>\n"
             "- HR Feedback(If Available): {review_feedback}\n\n"

             "CANDIDATES:\n"
             "- JD_A (model_id: {model_id_A})\n<<<\n{JD_A}\n>>>\n\n"
             "- JD_B (model_id: {model_id_B})\n<<<\n{JD_B}\n>>>\n\n"
             "- JD_C (model_id: {model_id_C})\n<<<\n{JD_C}\n>>>\n")
        ])


ad_generation_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant that extracts structured essentials from a Job Description (JD). "
     "Respond ONLY with valid JSON."),
    ("human", "JD:\n{jd}\n\nExtract the essentials now.")
])

ad_generation_brief_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a recruitment marketing strategist. "
     "Respond ONLY with compact JSON. Be extremely concise — every field must be short."),
    ("human",
     "JD essentials:\n{essentials}\n\n"
     "Platforms to target: {platforms}\n\n"
     "For each platform output ONE brief object with these keys (keep ALL values short):\n"
     "- platform: platform name (1-2 words)\n"
     "- tone: tone descriptor (1-3 words, e.g. 'professional', 'casual and bold')\n"
     "- length: 'short', 'medium', or 'long'\n"
     "- cta_style: CTA style (3-6 words, e.g. 'direct link', 'question hook')\n"
     "- hashtags_required: true or false\n"
     "- extra_assets: list of 1-3 short asset names (e.g. ['banner image', 'logo'])\n\n"
     "Output a JSON object with key 'briefs' containing the array. No explanation.")
])

# ad_generation_prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "You are a recruitment ad generator. "
#      "Always create TWO variants:\n"
#      "- Variant A: Skills/tech-focused hook\n"
#      "- Variant B: Benefits/impact-focused hook\n"
#      "Respond ONLY with valid JSON array of ads."),
#     ("human",
#      "Essentials:\n{essentials}\n\n"
#      "Platform Brief:\n{brief}\n\n"
#      "JD reference:\n{jd}\n\n"
#      "Now write the ads.")
# ])

ad_generation_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert recruitment ad copywriter and creative director.\n\n"
     "Always create TWO variants for the given platform:\n"
     "- Variant A: Skills/tech-focused hook (speaks to the candidate's expertise)\n"
     "- Variant B: Benefits/impact-focused hook (speaks to growth, culture, mission)\n\n"
     "For EACH variant you MUST provide all of these fields:\n"
     "- platform: the platform name (from the brief)\n"
     "- variant: 'A' or 'B'\n"
     "- headline: punchy and attention-grabbing, max 10 words\n"
     "- text: main body copy, 2-4 sentences, specific and engaging\n"
     "- cta: clear call-to-action, max 8 words\n"
     "- hashtags: list of 4-6 relevant hashtags\n"
     "- image_prompt: REQUIRED. A detailed visual description for AI image generation, MINIMUM 60 words. "
     "You MUST describe ALL of the following: (1) subject/people and their appearance, "
     "(2) environment/setting, (3) mood and lighting, (4) color palette, "
     "(5) composition and camera angle, (6) visual style (photorealistic/illustrated/cinematic), "
     "(7) any relevant tech/objects in the scene. "
     "BAD example: 'A professional office setting with diverse team'. "
     "GOOD example: 'A focused South Asian female software engineer in her late 20s reviewing "
     "Python code on three ultra-wide monitors, modern open-plan office with floor-to-ceiling "
     "windows, warm golden hour lighting, muted blue and white color palette, shallow depth of "
     "field close-up shot, photorealistic DSLR quality, mechanical keyboard and coffee cup on desk'\n"
     "- video_script: a 30-second video script for the ad (3-4 scenes with voiceover text)\n\n"
     "Respond ONLY with a valid JSON object matching the AdVariantList schema. "
     "No markdown, no explanation, no extra text."),
    ("human",
     "Job Essentials:\n{essentials}\n\n"
     "Platform Brief:\n{brief}\n\n"
     "Full Job Description:\n{jd}\n\n"
     "Generate two ad variants. The image_prompt must be vivid, detailed, and at least 60 words. "
     "Tailor the visuals to represent the role, team culture, and the platform's visual style.")
])


# ── New phase-based ad generation prompts ────────────────────────────────────

ad_text_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert recruitment ad copywriter. Your job is to write compelling ad copy "
     "for a specific platform with a specific creative focus.\n\n"
     "Focus direction:\n"
     "- SKILLS: lead with the candidate's craft, technical mastery, and professional achievement\n"
     "- BENEFITS: lead with growth, culture, mission, impact, and what makes this role worth joining\n\n"
     "CRITICAL — FEEDBACK: If feedback is provided, it contains MANDATORY revision instructions. "
     "Apply every point exactly. Do not ignore or soften any requested change.\n\n"
     "Return ONLY a valid JSON object with exactly these keys:\n"
     "  headline  — punchy and attention-grabbing, max 10 words\n"
     "  text      — main body copy, 2-4 sentences, specific and engaging\n"
     "  cta       — clear call-to-action, max 8 words\n"
     "  hashtags  — list of 4-6 relevant hashtags\n\n"
     "No markdown, no explanation, no extra keys."),
    ("human",
     "Platform: {platform}\n"
     "Platform Brief:\n{brief}\n\n"
     "Job Essentials:\n{essentials}\n\n"
     "Full Job Description:\n{jd}\n\n"
     "Creative Focus: {focus}\n\n"
     "Feedback (MANDATORY — apply all points exactly): {feedback}\n\n"
     "Write the ad copy now.")
])


ad_image_script_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a visual creative director specialising in recruitment imagery. "
     "Your job is to write a detailed AI image generation prompt for a job ad.\n\n"
     "The image will be used across all platforms so it must be platform-agnostic and "
     "universally appealing. It must represent the role, team culture, and work environment.\n\n"
     "Style direction:\n"
     "- PROFESSIONAL: clean, corporate, polished — suits LinkedIn / formal platforms\n"
     "- CREATIVE: bold, vibrant, editorial — suits Instagram / Facebook / X\n\n"
     "CRITICAL — FEEDBACK: If feedback is provided, it contains MANDATORY revision instructions. "
     "Apply every point exactly as specified.\n\n"
     "Your image prompt MUST be at least 60 words and describe ALL of:\n"
     "  1. Subject/people and their appearance (age, gender presentation, attire, expression)\n"
     "  2. Environment/setting (workspace, outdoors, studio, etc.)\n"
     "  3. Mood and lighting (golden hour, studio soft-box, natural diffused, etc.)\n"
     "  4. Color palette (2-3 dominant colors)\n"
     "  5. Composition and camera angle (wide shot, close-up, over-the-shoulder, etc.)\n"
     "  6. Visual style (photorealistic DSLR, cinematic, illustrated, flat design, etc.)\n"
     "  7. Relevant objects or props that signal the role\n\n"
     "Return ONLY a plain string — the image prompt text. No JSON, no explanation."),
    ("human",
     "Style direction: {style}\n\n"
     "Job Essentials:\n{essentials}\n\n"
     "Sample approved ad text (for tone reference):\n{ad_sample}\n\n"
     "Feedback (MANDATORY — apply all points exactly): {feedback}\n\n"
     "Write the image generation prompt now.")
])