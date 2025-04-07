from fastapi import FastAPI, UploadFile, File, Form
import fitz  # for PDF parsing
import re
from phi.agent import Agent
from phi.model.huggingface import HuggingFaceChat

app = FastAPI()

# ✅ Initialize Agents
resume_ner_agent = Agent(
    name="resume_extractor",
    model=HuggingFaceChat(id="meta-llama/Meta-Llama-3-8B-Instruct"),
    markdown=True
)

resume_score_agent = Agent(
    name="resume_scorer",
    model=HuggingFaceChat(id="meta-llama/Meta-Llama-3-8B-Instruct"),
    markdown=True
)

# ✅ Function: Extract PDF text
def extract_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# ✅ Function: Clean content before `content_type=`
def extract_before_content_type(text):
    match = re.search(r'(.*?)content_type=', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text  # fallback to full text if no match

@app.post("/analyze_resume/")
async def analyze_resume(file: UploadFile = File(...), job: str = Form(...)):
    pdf_bytes = await file.read()
    raw_text = extract_text(pdf_bytes)
    clean_text = extract_before_content_type(raw_text)

    # 🧠 NER Prompt
    ner_prompt = f"""[INSERT YOUR FULL RESUME EXTRACTION PROMPT HERE]\n\n{clean_text[:2500]}"""
    ner_details = resume_ner_agent.run(ner_prompt)

    # 🧠 Resume Scoring Prompt
    score_prompt = f"""
Evaluate this resume for the following job profile.

🔹 Job Profile:
{job}

🔹 Extracted Resume Info:
{ner_details}

Output a score from 0 to 100 and a one-paragraph explanation.
"""
    score_result = resume_score_agent.run(score_prompt)

    return {
        "ner_details": ner_details,
        "resume_score": score_result
    }
