from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from openai import OpenAI
import time
import json

# === Ollama Configuration ===
VISION_MODEL = "puyangwang/medgemma-27b-it:q8"
OLLAMA_BASE_URL = "http://172.20.52.98:11434/v1"
DUMMY_API_KEY = "none"

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=DUMMY_API_KEY)
app = FastAPI(title="Medical AI Diagnosis", docs_url=None, redoc_url=None)

# Load static HTML once
with open("index.html", "r", encoding="utf-8") as f:
    BASE_HTML = f.read()

@app.get("/", response_class=HTMLResponse)
async def serve_form():
    return HTMLResponse(content=BASE_HTML)

@app.post("/", response_class=HTMLResponse)
async def diagnose(prompt: str = Form(...)):
    messages = [
        {"role": "system", "content": "You are an expert medical AI designed to assist doctors, radiologists, and healthcare professionals. Provide clear, evidence-based clinical answers."},
        {"role": "system", "content": "You are required to give the response in proper formatting in English Language."},
        {"role": "user", "content": prompt}
    ]

    try:
        start = time.time()
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            max_tokens=512,
            temperature=0.1
        )
        duration = round(time.time() - start, 2)
        result = response.choices[0].message.content.strip()

        # Just return plain response text (will be processed on frontend)
        return HTMLResponse(content=result)

    except Exception as e:
        return HTMLResponse(content=f"⚠️ Error: {str(e)}")

# Optional: Add a JSON API endpoint for better integration
@app.post("/api/chat")
async def chat_api(prompt: str = Form(...)):
    messages = [
        {"role": "system", "content": "You are an expert medical AI designed to assist doctors, radiologists, and healthcare professionals. Provide clear, evidence-based clinical answers."},
        {"role": "system", "content": "You are required to give the response in proper formatting in English Language."},
        {"role": "user", "content": prompt}
    ]

    try:
        start = time.time()
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            max_tokens=512,
            temperature=0.1
        )
        duration = round(time.time() - start, 2)
        result = response.choices[0].message.content.strip()

        return JSONResponse({
            "response": result,
            "processing_time": duration,
            "status": "success"
        })

    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "status": "error"
        }, status_code=500)