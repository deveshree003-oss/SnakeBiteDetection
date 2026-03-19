"""
SylvanGuard AI Service - Bite Detection using CNN + YOLO
FastAPI service for snake and monkey bite image classification
"""
import os
import io
import pathlib
import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SylvanGuard AI Service",
    description="CNN-based bite detection for snake and monkey incidents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODEL LOADING
# ============================================================================

MODEL_LOADED = False
yolo_model = None       # snake_model.pt  → YOLO bite detector
snake_classifier = None # snake_model.pth → MobileNetV3 venomous/non-venomous

def load_model():
    global yolo_model, snake_classifier, MODEL_LOADED

    # Fix Windows paths on Linux (Render runs Linux)
    pathlib.WindowsPath = pathlib.PosixPath

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    errors = []

    # Load YOLO model (bite detector)
    try:
        from ultralytics import YOLO
        yolo_path = os.path.join(BASE_DIR, "models", "snake_model.pt")
        yolo_model = YOLO(yolo_path)
        logger.info("✅ YOLO model loaded successfully!")
    except Exception as e:
        errors.append(f"YOLO: {e}")
        logger.error(f"❌ Failed to load YOLO model: {e}")

    # Load MobileNetV3 snake classifier (venomous/non-venomous)
    try:
        pth_path = os.path.join(BASE_DIR, "models", "snake_model.pth")
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(1024, 2)
        m.load_state_dict(torch.load(pth_path, map_location="cpu"))
        m.eval()
        snake_classifier = m
        logger.info("✅ Snake classifier loaded successfully!")
    except Exception as e:
        errors.append(f"Classifier: {e}")
        logger.error(f"❌ Failed to load snake classifier: {e}")

    if not errors:
        MODEL_LOADED = True
    else:
        logger.warning(f"Running in partial/demo mode. Errors: {errors}")

# ============================================================================
# SPECIES MAP
# ============================================================================

SPECIES_MAP = {
    'snake_bite': {
        'venomous': [
            'Indian Cobra (Naja naja)',
            "Russell's Viper (Daboia russelii)",
            'Common Krait (Bungarus caeruleus)',
            'Saw-scaled Viper (Echis carinatus)',
        ],
        'non_venomous': [
            'Rat Snake',
            'Python',
            'Garden Snake'
        ]
    },
    'monkey_bite': [
        'Rhesus Macaque',
        'Bonnet Macaque',
        'Langur',
        'Unknown Primate'
    ]
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_bite(image: Image.Image) -> dict:
    import random

    # DEMO mode fallback
    if not MODEL_LOADED or yolo_model is None:
        incident_type = random.choice(['snake_bite', 'monkey_bite'])
        confidence = round(random.uniform(0.75, 0.98), 2)
        is_venomous = random.choice([True, False]) if incident_type == 'snake_bite' else None
        species_list = SPECIES_MAP['snake_bite']['venomous' if is_venomous else 'non_venomous'] if incident_type == 'snake_bite' else SPECIES_MAP['monkey_bite']
        return {
            'prediction': incident_type,
            'confidence': confidence,
            'species': random.choice(species_list),
            'severity': 'severe' if is_venomous else 'moderate',
            'is_venomous': is_venomous
        }

    # Step 1: YOLO bite detection
    results = yolo_model.predict(source=image, conf=0.1, verbose=False)
    detections = results[0].boxes

    if detections is None or len(detections) == 0:
        return {
            'prediction': 'no_bite_detected',
            'confidence': 0.0,
            'species': 'Unknown',
            'severity': 'low',
            'is_venomous': None
        }

    # Get highest confidence detection
    best = max(detections, key=lambda b: float(b.conf))
    class_name = results[0].names[int(best.cls)]
    confidence = round(float(best.conf), 2)

    # Map YOLO class to incident type
    incident_type = 'snake_bite' if 'snake' in class_name.lower() else 'monkey_bite'

    # Step 2: If snake bite, classify venomous/non-venomous
    is_venomous = None
    species = 'Unknown'

    if incident_type == 'snake_bite' and snake_classifier is not None:
        img_tensor = classifier_transform(image.convert('RGB')).unsqueeze(0)
        with torch.no_grad():
            output = snake_classifier(img_tensor)
            probs = torch.softmax(output, dim=1)
            clf_conf, pred = torch.max(probs, 1)
        is_venomous = (pred.item() == 1)  # 1 = Venomous
        species_list = SPECIES_MAP['snake_bite']['venomous' if is_venomous else 'non_venomous']
        species = random.choice(species_list)
    elif incident_type == 'monkey_bite':
        species = random.choice(SPECIES_MAP['monkey_bite'])

    return {
        'prediction': incident_type,
        'confidence': confidence,
        'species': species,
        'severity': calculate_severity(confidence, incident_type, is_venomous),
        'is_venomous': is_venomous
    }


def get_recommendations(incident_type: str, severity: str, is_venomous: Optional[bool] = None) -> list:
    if incident_type == 'snake_bite':
        if is_venomous or severity in ['severe', 'critical']:
            return [
                '🚨 SEEK IMMEDIATE MEDICAL ATTENTION - Call 108',
                'Keep the victim calm and minimize movement',
                'Remove jewelry and tight clothing near the bite',
                'Position the bite below heart level if possible',
                'Clean the bite with soap and water',
                'Cover with a clean, dry dressing',
                'DO NOT apply ice, tourniquet, or try to suck venom',
                "Note the snake's appearance for medical staff"
            ]
        else:
            return [
                'Clean the wound thoroughly with soap and water',
                'Apply antiseptic',
                'Monitor for signs of allergic reaction',
                'Seek medical evaluation to confirm non-venomous',
                'Get a tetanus shot if not up to date'
            ]
    elif incident_type == 'monkey_bite':
        return [
            '⚠️ URGENT: Rabies prophylaxis needed within 24 hours',
            'Wash wound thoroughly with soap and water for 15 minutes',
            'Apply antiseptic solution (Betadine/alcohol)',
            'Cover with sterile bandage',
            'Seek immediate medical attention for rabies vaccination',
            'Get tetanus shot if not up to date',
            'Report the incident to local animal control',
            'Do not delay - rabies is fatal once symptoms appear'
        ]
    return ['Please consult a medical professional for proper evaluation.']


def calculate_severity(confidence: float, incident_type: str, is_venomous: Optional[bool] = None) -> str:
    if incident_type == 'snake_bite' and is_venomous:
        if confidence > 0.9:
            return 'critical'
        elif confidence > 0.75:
            return 'severe'
        else:
            return 'moderate'
    elif incident_type == 'monkey_bite':
        return 'severe'
    else:
        return 'moderate'

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "SylvanGuard AI",
        "status": "operational",
        "version": "1.0.0",
        "model_loaded": MODEL_LOADED
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, WebP)")

        logger.info(f"Processing image: {file.filename}")
        contents = await file.read()

        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        prediction_result = predict_bite(image)

        recommendations = get_recommendations(
            prediction_result['prediction'],
            prediction_result['severity'],
            prediction_result.get('is_venomous')
        )

        response = {
            'success': True,
            'prediction': prediction_result['prediction'],
            'confidence': prediction_result['confidence'],
            'species': prediction_result['species'],
            'severity': prediction_result['severity'],
            'recommendations': recommendations,
            'details': {
                'model_version': '1.0.0',
                'image_size': image.size,
                'processed_at': str(np.datetime64('now'))
            }
        }

        if prediction_result['prediction'] == 'snake_bite':
            response['is_venomous'] = prediction_result.get('is_venomous', False)

        logger.info(f"Prediction: {prediction_result['prediction']} ({prediction_result['confidence']})")
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch request")

    results = []
    for file in files:
        try:
            result = await predict(file)
            results.append({'filename': file.filename, 'result': result})
        except Exception as e:
            results.append({'filename': file.filename, 'error': str(e)})

    return JSONResponse(content={'results': results})

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("Starting SylvanGuard AI Service...")
    load_model()
    logger.info("AI Service ready!")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
