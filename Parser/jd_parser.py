import json
import re
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import os
import docx
from PyPDF2 import PdfReader

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

JD_FIELDS = ["title", "department", "seniority", "location", "employment_type", "hr_notes"]

def read_file(file_path: str) -> str:
    path = Path(file_path)
    text = ""
    if path.suffix.lower() == ".txt":
        text = path.read_text()
    elif path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    elif path.suffix.lower() == ".docx":
        doc = docx.Document(str(path))
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        raise ValueError("Unsupported file type. Use .txt, .pdf, or .docx")
    return text.strip()

def parse_jd_input(text: str) -> dict:
    """
    Only extracts what the LLM can find from text.
    Does NOT fill missing fields by itself.
    """
    prompt = f"""
Extract the following Job Description fields from the text below.
Only extract what is explicitly mentioned. If something is missing, leave it empty.

Fields:
- title
- department
- seniority
- location
- employment_type
- hr_notes

Text: \"\"\"{text}\"\"\"

Return output strictly as JSON matching this format:
{{"title": "...", "department": "...", "seniority": "...", "location": "...", "employment_type": "...", "hr_notes": "..."}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = response.choices[0].message.content
    try:
        jd_data = json.loads(content)
    except:
        jd_data = json.loads(re.search(r"\{.*\}", content, re.DOTALL).group())
    return jd_data

def agentic_jd_creator(file_path: str = None):
    jd_data = {field: "" for field in JD_FIELDS}

    # Step 1: Extract whatever is explicitly present
    if file_path:
        print(f"Reading file: {file_path}")
        text = read_file(file_path)
        parsed = parse_jd_input(text)
        jd_data.update({k: v for k, v in parsed.items() if v})
    else:
        user_input = input("Enter JD info: ").strip()
        parsed = parse_jd_input(user_input)
        jd_data.update({k: v for k, v in parsed.items() if v})

    # Step 2: Ask user for any missing fields
    #for field in JD_FIELDS:
        #while not jd_data[field]:
            #value = input(f"Please provide {field.replace('_', ' ')}: ").strip()
            #if value:
                #jd_data[field] = value

    # Step 3: Show final JD
    print("\n✅ JD parameters ready:")
    print(json.dumps(jd_data, indent=2))
    return jd_data

# Example usage
if __name__ == "__main__":
    jd_params = agentic_jd_creator("hr_job_brief.pdf")
    #jd_params = agentic_jd_creator("hr_job_brief_vague.pdf")