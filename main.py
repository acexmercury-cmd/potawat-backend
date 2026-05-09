from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
import os

app = FastAPI()

# Allow your Lovable frontend to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://potawatomi-asr.lovable.app",
        "http://localhost:5173",  # for local dev
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper once at startup. "base" is fastest; "small" is more accurate.
model = whisper.load_model("base")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if file.size and file.size > 25 * 1024 * 1024:
        raise HTTPException(413, "File too large (25MB max)")

    # Whisper needs a file path, so write to temp
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path)
        return {"text": result["text"], "language": result.get("language")}
    finally:
        os.unlink(tmp_path)  # clean up — never stored
