import io
import warnings
from typing import Tuple, Dict, Union, Any, List
from enum import Enum

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageEnhance
from loguru import logger

warnings.filterwarnings("ignore")

class ImageModality(Enum):
    XRAY = "xray"
    CT = "ct"
    MRI = "mri"

# =============================================================================
# OPTIMIZED MEDICAL ENHANCER (Low Memory)
# =============================================================================

class MedicalImageEnhancer:
    @staticmethod
    def normalize_medical_intensity(image: Image.Image) -> Image.Image:
        """
        Memory-efficient histogram-based normalization. 
        Replaces heavy NumPy percentile math with Pillow LUT (Look-Up Table).
        """
        # Convert to Grayscale to calculate stats efficiently
        gray = image.convert("L")
        hist = gray.histogram()
        total_pixels = sum(hist)
        
        # Find 2nd and 98th percentile to avoid outlier noise
        low_threshold, high_threshold = total_pixels * 0.02, total_pixels * 0.98
        
        low_val, high_val = 0, 255
        current_sum = 0
        for i, count in enumerate(hist):
            current_sum += count
            if current_sum >= low_threshold:
                low_val = i
                break
        
        current_sum = 0
        for i, count in enumerate(hist):
            current_sum += count
            if current_sum >= high_threshold:
                high_val = i
                break

        # Fast Point Transform: scale values between low_val and high_val to 0-255
        diff = high_val - low_val if high_val > low_val else 1
        lut = [max(0, min(255, int((i - low_val) * 255 / diff))) for i in range(256)]
        
        # Apply to all channels
        return image.point(lut * (3 if image.mode == "RGB" else 1))

    @staticmethod
    def enhance_details(image: Image.Image) -> Image.Image:
        """Sharpen medical features like lung textures"""
        image = ImageEnhance.Contrast(image).enhance(1.15)
        image = ImageEnhance.Sharpness(image).enhance(1.2)
        return image

# =============================================================================
# OPTIMIZED PREPROCESSOR
# =============================================================================

class MedicalImagePreprocessor:
    def __init__(self, modality: str = "xray"):
        self.modality = modality.lower()
        # Standard Medical Normalization Stats (ImageNet compatible)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.enhancer = MedicalImageEnhancer()

    def get_transform(self, image: Image.Image):
        """Adaptive transform to prevent distortion of medical images"""
        width, height = image.size
        
        # Use BICUBIC for high quality upscaling if image is small
        interpolation = transforms.InterpolationMode.BICUBIC
        
        return transforms.Compose([
            # Resize smallest side to 224 while maintaining aspect ratio
            transforms.Resize(224, interpolation=interpolation),
            # Center crop ensures the core diagnostic area is captured
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def process(self, image_bytes: bytes) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Main processing pipeline for Render"""
        try:
            # 1. Load Image
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            orig_size = image.size
            logger.info(f"Processing {self.modality} image: {orig_size}")

            # 2. Analyze & Enhance (Heuristic based on brightness)
            # Use extreme value check to see if image is washed out
            extrema = image.convert("L").getextrema()
            is_low_contrast = (extrema[1] - extrema[0]) < 80
            
            if is_low_contrast:
                logger.info("Enhancing low-contrast medical scan...")
                image = self.enhancer.normalize_intensity(image)
                image = self.enhancer.enhance_details(image)

            # 3. Transform to Tensor
            transform = self.get_transform(image)
            tensor = transform(image).unsqueeze(0) # Add batch dimension [1, 3, 224, 224]

            metadata = {
                "original_size": f"{orig_size[0]}x{orig_size[1]}",
                "modality": self.modality,
                "was_enhanced": is_low_contrast,
                "tensor_shape": list(tensor.shape)
            }

            return tensor, metadata

        except Exception as e:
            logger.error(f"Preprocessing Critical Error: {str(e)}")
            raise

# =============================================================================
# FAST API COMPATIBLE EXPORT
# =============================================================================

def preprocess_image(image_bytes: bytes, modality: str = "xray"):
    """Single entry point for main.py"""
    preprocessor = MedicalImagePreprocessor(modality)
    return preprocessor.process(image_bytes)
