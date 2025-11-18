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


DEEPFACE_MODEL = "Facenet512"
DEEPFACE_BACKEND = "opencv"


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
    if not norm:
        return embedding
    return embedding / norm


def generate_face_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    image_array = image_bytes_to_array(image_bytes)
    try:
        # DeepFace returns a list of representations; grab the first embedding.
        representations = DeepFace.represent(
            img_path=image_array,
            model_name=DEEPFACE_MODEL,
            detector_backend=DEEPFACE_BACKEND,
            enforce_detection=False,
        )
    except Exception:
        return None

    if isinstance(representations, list) and representations:
        embedding = representations[0].get("embedding")
    elif isinstance(representations, dict):
        embedding = representations.get("embedding")
    else:
        embedding = None

    if embedding is None:
        return None
    return normalize_embedding(np.array(embedding, dtype=np.float32))


def embedding_to_blob(embedding: Optional[np.ndarray]) -> Optional[bytes]:
    if embedding is None:
        return None
    return pickle.dumps(embedding.astype(np.float32))


def blob_to_embedding(blob: Optional[bytes]) -> Optional[np.ndarray]:
    if not blob:
        return None
    embedding = pickle.loads(blob)
    return normalize_embedding(embedding)


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
    return distance <= threshold, distance


def perform_basic_liveness_check(image_bytes: bytes, variance_threshold: float = 50.0) -> Tuple[bool, float]:
    """
    Simple texture & blur based heuristic. Higher Laplacian variance indicates a real 3D face vs flat media.
    """
    image_array = image_bytes_to_array(image_bytes)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance >= variance_threshold, variance


def deepface_liveness_check(image_bytes: bytes) -> Tuple[bool, str]:
    """
    Optional DeepFace anti-spoofing hook. Returns success flag and message.
    """
    np_image = image_bytes_to_array(image_bytes)
    try:
        analysis = DeepFace.analyze(
            img_path=np_image,
            actions=[],
            anti_spoofing=True,
            detector_backend="opencv",
            enforce_detection=False,
        )
        if isinstance(analysis, list):
            analysis = analysis[0]
        spoofing = analysis.get("anti_spoofing", {})
        is_real = spoofing.get("real_face", True)
        confidence = spoofing.get("confidence", 1.0)
        message = f"DeepFace anti-spoof confidence={confidence:.2f}"
        return bool(is_real), message
    except Exception as exc:  # pragma: no cover - best effort safeguard
        return False, f"DeepFace anti-spoof failed: {exc}"

