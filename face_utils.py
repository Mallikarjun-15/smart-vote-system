import io
import pickle
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from deepface import DeepFace
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("DeepFace library is required for biometric utilities.") from exc


# -------------------------------------------------------------------
# DeepFace configuration: Use OpenCV backend ONLY (RetinaFace breaks)
# -------------------------------------------------------------------
DEEPFACE_MODEL = "Facenet512"
DEEPFACE_BACKEND = "opencv"


# Cache model to avoid re-loading on Streamlit Cloud
_face_model = None


def get_face_model():
    global _face_model
    if _face_model is None:
        _face_model = DeepFace.build_model(DEEPFACE_MODEL)
    return _face_model


# ------------------------------
# Utility Helpers
# ------------------------------
def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_image_bytes(image_bytes: bytes, destination: Path) -> None:
    ensure_dir(destination)
    with destination.open("wb") as file:
        file.write(image_bytes)


def image_bytes_to_array(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    return embedding if norm == 0 else embedding / norm


# -------------------------------------------------------------------
# Face Embedding Extraction
# -------------------------------------------------------------------
def generate_face_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    image_array = image_bytes_to_array(image_bytes)

    try:
        reps = DeepFace.represent(
            img_path=image_array,
            model_name=DEEPFACE_MODEL,
            model=get_face_model(),
            detector_backend=DEEPFACE_BACKEND,
            enforce_detection=False,
        )
    except Exception:
        return None

    # reps can be a list or dict
    embedding = None
    if isinstance(reps, list) and reps:
        embedding = reps[0].get("embedding")
    elif isinstance(reps, dict):
        embedding = reps.get("embedding")

    if embedding is None:
        return None

    return normalize_embedding(np.array(embedding, dtype=np.float32))


# -------------------------------------------------------------------
# Embedding persistence helpers
# -------------------------------------------------------------------
def embedding_to_blob(embedding: Optional[np.ndarray]) -> Optional[bytes]:
    return None if embedding is None else pickle.dumps(embedding.astype(np.float32))


def blob_to_embedding(blob: Optional[bytes]) -> Optional[np.ndarray]:
    if not blob:
        return None
    emb = pickle.loads(blob)
    return normalize_embedding(emb)


# -------------------------------------------------------------------
# Face verification (distance threshold)
# -------------------------------------------------------------------
def verify_face(
    registered_embedding: Optional[np.ndarray],
    live_embedding: Optional[np.ndarray],
    threshold: float = 0.8,
) -> Tuple[bool, float]:
    if registered_embedding is None or live_embedding is None:
        return False, float("inf")

    reg = normalize_embedding(registered_embedding)
    live = normalize_embedding(live_embedding)
    distance = np.linalg.norm(reg - live)

    return distance <= threshold, float(distance)


# -------------------------------------------------------------------
# Basic Liveness Detection (OpenCV-based)
# -------------------------------------------------------------------
def perform_basic_liveness_check(image_bytes: bytes, var_threshold: float = 45.0) -> Tuple[bool, float]:
    """
    Simple Laplacian variance sharpness measure.
    Higher variance → more real texture → less likely spoof.
    """
    arr = image_bytes_to_array(image_bytes)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance >= var_threshold, float(variance)


# -------------------------------------------------------------------
# OPTIONAL: Lightweight anti-spoofing wrapper
# (DeepFace anti_spoofing disabled due to RetinaFace incompatibility)
# -------------------------------------------------------------------
def deepface_liveness_check(_: bytes) -> Tuple[bool, str]:
    return True, "DeepFace anti-spoof disabled (using basic liveness)."
