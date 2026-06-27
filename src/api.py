import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse
import shutil
import uuid

from src.face_detector import FaceDetector
from src.skin_tone import SkinToneAnalyzer
from src.skin_condition import SkinConditionAnalyzer
from src.recommender import SkinRecommender

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

detector           = FaceDetector()
tone_analyzer      = SkinToneAnalyzer()
condition_analyzer = SkinConditionAnalyzer()
recommender        = SkinRecommender()

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # Save uploaded image temporarily
        temp_path = f"test_images/temp_{uuid.uuid4().hex}.jpg"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run analysis
        image, landmarks = detector.get_landmarks(temp_path)

        if landmarks is None:
            os.remove(temp_path)
            return JSONResponse({"error": "No face detected. Please upload a clear selfie."})

        rois       = detector.get_cheek_roi(image, landmarks)
        tone       = tone_analyzer.analyze(rois)
        conditions = condition_analyzer.analyze(image, landmarks)

        skin_profile    = {"tone": tone, "conditions": conditions}
        recommendations = recommender.recommend(skin_profile)

        # Clean up temp file
        os.remove(temp_path)

        return JSONResponse({
            "tone":            tone,
            "conditions":      conditions["overall"],
            "recommendations": recommendations
        })

    except Exception as e:
        return JSONResponse({"error": str(e)})