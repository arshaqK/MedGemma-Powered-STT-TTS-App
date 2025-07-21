# main.py
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from openai import OpenAI
import uuid
import time
import json
import os

VISION_MODEL = "puyangwang/medgemma-27b-it:q8"
OLLAMA_BASE_URL = "http://172.20.52.98:11434/v1"
DUMMY_API_KEY = "none"

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=DUMMY_API_KEY)
app = FastAPI(title="Medical AI Diagnosis", docs_url=None, redoc_url=None)

# Read HTML file
try:
    with open("index.html", "r", encoding="utf-8") as f:
        BASE_HTML = f.read()
except FileNotFoundError:
    BASE_HTML = "<h1>index.html not found</h1>"

@app.get("/", response_class=HTMLResponse)
async def serve_form():
    return HTMLResponse(content=BASE_HTML)

@app.post("/api/chat")
async def chat_api(prompt: str = Form(...)):
    """API endpoint for chat functionality"""
    messages = [
        {"role": "system", "content": "You are an expert medical AI for clinical diagnostics. Provide clear, helpful medical information while emphasizing that your responses should not replace professional medical advice."},
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
            "success": True,
            "response": result,
            "duration": duration
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/api/tts")
async def tts_api(text: str = Form(...)):
    """Text-to-speech API endpoint"""
    try:
        # Create a temporary audio file (placeholder for actual TTS implementation)
        fake_audio_path = f"/tmp/tts_{uuid.uuid4().hex}.wav"
        
        # For now, create an empty WAV file
        # In a real implementation, you would use a TTS service like:
        # - Google Text-to-Speech
        # - Amazon Polly
        # - Azure Speech Services
        # - Local TTS engines like espeak or festival
        
        with open(fake_audio_path, "wb") as f:
            # Minimal WAV header for a silent audio file
            wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22\x56\x00\x00\x44\xAC\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00'
            f.write(wav_header)
            f.write(b'\x00' * 2048)  # Silent audio data
        
        return FileResponse(
            fake_audio_path, 
            media_type="audio/wav", 
            filename="speech.wav",
            headers={"Cache-Control": "no-cache"}
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Legacy endpoint for backward compatibility
@app.post("/", response_class=HTMLResponse)
async def diagnose(prompt: str = Form(...)):
    """Legacy endpoint - redirects to new API"""
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
