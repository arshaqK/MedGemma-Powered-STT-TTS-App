# main.py
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from openai import OpenAI
import uuid
import time

VISION_MODEL = "puyangwang/medgemma-27b-it:q8"
OLLAMA_BASE_URL = "http://172.20.52.98:11434/v1"
DUMMY_API_KEY = "none"

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=DUMMY_API_KEY)
app = FastAPI(title="Medical AI Diagnosis", docs_url=None, redoc_url=None)

with open("index.html", "r", encoding="utf-8") as f:
    BASE_HTML = f.read()

@app.get("/", response_class=HTMLResponse)
async def serve_form():
    return HTMLResponse(content=BASE_HTML)

@app.post("/", response_class=HTMLResponse)
async def diagnose(prompt: str = Form(...)):
    messages = [
        {"role": "system", "content": "You are an expert medical AI for clinical diagnostics."},
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
        html = f'<div class="response-content">{result}</div><div class="response-meta">Time: {duration}s</div>'
        return HTMLResponse(content=html)
    except Exception as e:
        return HTMLResponse(content=f'<div class="response-content">⚠️ {str(e)}</div>')

@app.post("/api/tts")
async def tts_api(prompt: str = Form(...)):
    try:
        fake_audio_path = f"/tmp/tts_{uuid.uuid4().hex}.wav"
        with open(fake_audio_path, "wb") as f:
            f.write(b"")  # Replace with actual MetaTTS audio data
        return FileResponse(fake_audio_path, media_type="audio/wav", filename="speech.wav")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
