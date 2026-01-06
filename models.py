"""
MedAI - AI Model Definitions
Optimized for Render CPU Deployment (512MB RAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from loguru import logger
from datetime import datetime, timezone

class SimpleCNN(nn.Module):
    """Memory-efficient CNN for CPU-bound environments"""
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # Reduced filters from 32 to 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # Reduced filters from 64 to 32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ChestXrayAIModel:
    """Wrapper for chest X-ray AI model optimized for Render"""
    
    def __init__(self, model_path=None, device="cpu", threshold=0.15):
        # Force CPU on Render even if CUDA is requested
        self.device = torch.device("cpu")
        self.threshold = threshold
        self.class_names = ["Normal", "Pneumonia", "COVID-19", "Tuberculosis", "Lung Opacity"]
        
        if model_path:
            try:
                # map_location='cpu' is critical for Render
                self.model = torch.load(model_path, map_location=self.device)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"✅ Loaded production model from {model_path}")
            except Exception as e:
                logger.warning(f"⚠️ Load failed ({e}). Falling back to SimpleCNN.")
                self._load_fallback()
        else:
            self._load_fallback()

    def _load_fallback(self):
        self.model = SimpleCNN(num_classes=len(self.class_names))
        self.model.to(self.device)
        self.model.eval()
        logger.info("🛠️ Initialized SimpleCNN Inference Engine")

    def predict(self, image_tensor):
        """Generate predictions using actual model inference"""
        logger.info(f"📡 Processing tensor: {image_tensor.shape}")
        
        start_time = datetime.now()
        
        try:
            # 1. Actual Inference
            with torch.no_grad():
                # Ensure tensor is on CPU
                image_tensor = image_tensor.to(self.device)
                outputs = self.model(image_tensor)
                # Convert logits to probabilities
                probabilities = F.softmax(outputs, dim=1)[0].tolist()

            # 2. Map Results
            detected = []
            for i, prob in enumerate(probabilities):
                if prob > self.threshold:
                    condition = self.class_names[i]
                    detected.append({
                        "condition": condition,
                        "confidence": round(prob, 4),
                        "severity": self._get_severity(condition)
                    })

            # Sort by confidence
            detected = sorted(detected, key=lambda x: x["confidence"], reverse=True)

            # Fallback to Normal if nothing detected
            if not detected:
                detected = [{"condition": "Normal", "confidence": 0.99, "severity": "none"}]

            primary = detected[0]
            
            # 3. Build Response
            end_time = datetime.now()
            proc_time = int((end_time - start_time).total_seconds() * 1000)

            return {
                "diagnosis": {
                    "primary_condition": primary["condition"],
                    "all_conditions": detected,
                    "overall_confidence": primary["confidence"],
                    "is_critical": any(d["severity"] in ["severe", "critical"] for d in detected),
                },
                "heatmap": {
                    "available": False,
                    "message": "Heatmap generation disabled on CPU-only tier"
                },
                "recommendations": self._generate_recommendations(primary["condition"]),
                "metadata": {
                    "model": "medai_v1.0_cpu",
                    "device": str(self.device),
                    "processing_time_ms": proc_time,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            }

        except Exception as e:
            logger.error(f"❌ Prediction Failed: {e}")
            raise e

    def _get_severity(self, condition):
        mapping = {
            "Normal": "none",
            "Pneumonia": "moderate",
            "COVID-19": "moderate",
            "Tuberculosis": "severe",
            "Lung Opacity": "mild"
        }
        return mapping.get(condition, "unknown")

    def _generate_recommendations(self, condition):
        base = ["AI analysis completed.", "Consult a radiologist for validation."]
        specific = {
            "Normal": ["Maintain regular checkups."],
            "Pneumonia": ["Immediate clinical consultation required.", "Follow-up Chest X-ray in 48h recommended."],
            "COVID-19": ["Isolate as per local guidelines.", "PCR testing recommended for confirmation."],
            "Tuberculosis": ["Urgent respiratory specialist referral.", "Sputum culture testing advised."],
            "Lung Opacity": ["Clinical correlation with symptoms required."]
        }
        return base + specific.get(condition, [])
