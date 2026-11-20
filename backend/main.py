from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.recognition import SignRecognizer

app = FastAPI(
    title="SIGNBRIDGE API",
    description="Indian Sign Language static-sign recognition prototype",
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

recognizer = SignRecognizer()


@app.get("/")
def home():
    return {"project": "SIGNBRIDGE", "status": "online", "message": "ISL recognition prototype ready"}


@app.get("/health")
def health():
    return {
        "engine": "SIGNBRIDGE",
        "status": "healthy",
        "model_loaded": recognizer.model is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/signs")
def available_signs():
    labels = recognizer.labels
    return {
        "language": "Indian Sign Language",
        "recognition_status": "prototype_ready" if labels else "collecting_data",
        "supported_signs": labels,
        "message": "Static-sign prototype model loaded." if labels else "Hand tracking is available; collect verified samples for at least two signs and train the model.",
    }


@app.post("/api/recognize")
async def recognize_sign(image: UploadFile = File(...)):
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image frame.")
    content = await image.read()
    if not content or len(content) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image is empty or larger than 8 MB.")
    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")
    return recognizer.predict(decoded)


@app.on_event("shutdown")
def shutdown_event():
    recognizer.close()
