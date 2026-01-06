import os
import io
import gc
import sys
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, Dict, Tuple, Any, List
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from PIL import Image
import numpy as np

# Ensure local directory is in path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    from models import ChestXrayAIModel
    from preprocess import MedicalImagePreprocessor, ImageModality
except ImportError as e:
    logger.critical(f"❌ critical module import failed: {e}")
    raise

load_dotenv()

# --- CONFIGURATION ---
# Force CPU for Render stability unless explicitly told otherwise
DEVICE = "cpu" 
MONGODB_URI = os.getenv("MONGODB_URI")
MODEL_PATH = os.getenv("MODEL_PATH", "model.pth")

# Global instances
app_state = {
    "mongo_client": None,
    "db": None,
    "ai_model": None,
    "preprocessor": None
}

# --- LIGHTWEIGHT VALIDATOR ---
class MedicalImageValidator:
    """Fast validation to filter out non-medical images without crashing RAM"""
    
    @staticmethod
    def validate(image_bytes: bytes, expected_type: str) -> Dict[str, Any]:
        errors = []
        warnings = []
        
        # 1. Size Check
        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > 15: # Render limit safety
            errors.append(f"File too large ({size_mb:.1f}MB). Max 15MB allowed.")

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                # 2. Basic Dimensions
                w, h = img.size
                if w < 128 or h < 128:
                    errors.append(f"Resolution too low ({w}x{h}).")
                
                # 3. Grayscale Heuristic (X-Rays/CTs are usually grayscale)
                # We check a small sample to save CPU
                sample = np.array(img.resize((32, 32)).convert("RGB"))
                rgb_diff = np.mean(np.std(sample, axis=2))
                is_grayscale = rgb_diff < 15
                
                if expected_type == "xray" and not is_grayscale:
                    warnings.append("Image contains significant color; check if this is a standard X-ray.")

                return {
                    "is_valid": len(errors) == 0,
                    "is_medical_guess": is_grayscale or (w == h),
                    "errors": errors,
                    "warnings": warnings,
                    "metadata": {"width": w, "height": h, "grayscale": is_grayscale}
                }
        except Exception as e:
            return {"is_valid": False, "errors": [f"Corrupt image: {str(e)}"], "warnings": []}

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and RAM-friendly initialization"""
    logger.info(f"🚀 Starting MedAI on {DEVICE}...")
    
    # 1. DB Init
    if MONGODB_URI:
        try:
            app_state["mongo_client"] = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            app_state["db"] = app_state["mongo_client"].get_default_database()
            await app_state["mongo_client"].admin.command('ping')
            logger.info("✅ MongoDB Connected")
        except Exception as e:
            logger.warning(f"⚠️ DB Optional: {e}")

    # 2. Model Init (The big memory consumer)
    try:
        app_state["ai_model"] = ChestXrayAIModel(model_path=MODEL_PATH, device=DEVICE)
        app_state["preprocessor"] = MedicalImagePreprocessor()
        logger.info("✅ AI Engine Loaded")
    except Exception as e:
        logger.error(f"❌ Model Load Failed: {e}")
    
    yield
    
    # Cleanup
    if app_state["mongo_client"]:
        app_state["mongo_client"].close()
    app_state.clear()
    gc.collect()

# --- API SETUP ---
app = FastAPI(title="MedAI API", version="1.5.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ROUTES ---

@app.post("/diagnostics/process")
async def process_diagnostics(
    file: UploadFile = File(...),
    type: str = Query("xray"),
    bypass_validation: bool = Query(False)
):
    """Full medical AI diagnostic pipeline"""
    if not app_state["ai_model"]:
        raise HTTPException(503, "AI Engine not initialized")

    img_data = await file.read()
    
    # 1. Validate
    if not bypass_validation:
        v = MedicalImageValidator.validate(img_data, type)
        if not v["is_valid"]:
            raise HTTPException(400, {"message": "Validation failed", "errors": v["errors"]})

    try:
        # 2. Preprocess
        tensor, meta = app_state["preprocessor"].process(img_data)
        
        # 3. Inference
        # Force garbage collection before heavy lifting
        gc.collect() 
        result = app_state["ai_model"].predict(tensor)
        
        # 4. Enrich Result
        result["file_info"] = {"filename": file.filename, "size_kb": len(img_data)/1024}
        result["preprocessing"] = meta

        # 5. Async Audit (Don't wait for it to finish)
        if app_state["db"] is not None:
            audit_entry = {**result, "timestamp": datetime.now(timezone.utc)}
            app_state["db"].diagnostics.insert_one(audit_entry)

        return result

    except Exception as e:
        logger.error(f"Pipeline Error: {e}")
        raise HTTPException(500, f"Processing Error: {str(e)}")
    finally:
        # Clean up memory immediately
        del img_data
        gc.collect()

@app.get("/health")
async def health():
    return {
        "status": "online",
        "device": DEVICE,
        "model_loaded": app_state["ai_model"] is not None,
        "db_connected": app_state["db"] is not None,
        "memory_usage_approx": f"{torch.cuda.memory_allocated() if DEVICE == 'cuda' else 'N/A'}"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
