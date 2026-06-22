"""Adapters for local trained EyeRec feature models."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_ROOTS = {
    "eye_color": REPO_ROOT / "train_eyeColor",
    "eye_shape": REPO_ROOT / "train_eyeShape",
    "face_shape": REPO_ROOT / "train_faceShape",
    "pupil_distance": REPO_ROOT / "train_pupilDistance",
}

for root in MODEL_ROOTS.values():
    src_root = root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from train_eye_color.datasets import build_eval_transforms as build_eye_color_transforms
from train_eye_color.model import build_model as build_eye_color_model
from train_eye_shape.datasets import build_eval_transforms as build_eye_shape_transforms
from train_eye_shape.model import build_model as build_eye_shape_model
from train_face_shape.datasets import build_eval_transforms as build_face_shape_transforms
from train_face_shape.model import build_model as build_face_shape_model
from train_pupil_distance.datasets import build_eval_transforms as build_pupil_distance_transforms
from train_pupil_distance.model import build_model as build_pupil_distance_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def pil_to_cv2(image: Image.Image):
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv2_to_pil(image) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


class OpenCVCropper:
    def __init__(self) -> None:
        self.eye_detector = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml"))
        self.face_detector = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )

    def crop_face(self, image: Image.Image, padding: float = 0.35) -> Image.Image:
        cv_image = pil_to_cv2(image)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(48, 48))
        if len(faces) == 0:
            return image

        x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
        image_height, image_width = cv_image.shape[:2]
        pad_x = int(width * padding)
        pad_y = int(height * padding)
        left = clamp(x - pad_x, 0, image_width)
        top = clamp(y - pad_y, 0, image_height)
        right = clamp(x + width + pad_x, 0, image_width)
        bottom = clamp(y + height + pad_y, 0, image_height)
        return cv2_to_pil(cv_image[top:bottom, left:right])

    def crop_eyes(self, image: Image.Image, padding: float = 0.85, normalize_side: bool = False) -> list[Image.Image]:
        cv_image = pil_to_cv2(image)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(24, 24))
        if len(eyes) == 0:
            return [image]

        image_height, image_width = cv_image.shape[:2]
        candidates = sorted(eyes, key=lambda eye: eye[2] * eye[3], reverse=True)[:2]
        crops: list[Image.Image] = []
        for x, y, width, height in candidates:
            pad_x = int(width * padding)
            pad_y = int(height * padding)
            left = clamp(x - pad_x, 0, image_width)
            top = clamp(y - pad_y, 0, image_height)
            right = clamp(x + width + pad_x, 0, image_width)
            bottom = clamp(y + height + pad_y, 0, image_height)
            crop = cv_image[top:bottom, left:right]
            if normalize_side and (x + width / 2) < (image_width / 2):
                crop = cv2.flip(crop, 1)
            crops.append(cv2_to_pil(crop))
        return crops or [image]


class LocalClassifier:
    def __init__(
        self,
        checkpoint_path: Path,
        build_model: Callable,
        build_transforms: Callable,
        image_selector: Callable[[Image.Image], list[Image.Image]],
        label_aliases: Optional[dict[str, str]] = None,
        probability_adjustments: Optional[dict[str, float]] = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.build_model = build_model
        self.build_transforms = build_transforms
        self.image_selector = image_selector
        self.label_aliases = label_aliases or {}
        self.probability_adjustments = probability_adjustments or {}
        self.device = get_device()
        self._model = None
        self._transform = None
        self._index_to_label = None

    def _load(self) -> None:
        if self._model is not None:
            return
        checkpoint = load_checkpoint(self.checkpoint_path, self.device)
        label_to_index = checkpoint["label_to_index"]
        self._index_to_label = {index: label for label, index in label_to_index.items()}
        config = checkpoint.get("config", {})
        model_name = str(config.get("model_name", "mobilenet_v3_small"))
        image_size = int(config.get("image_size", 224))
        model = self.build_model(
            num_classes=len(label_to_index),
            model_name=model_name,
            use_pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        self._model = model
        self._transform = self.build_transforms(image_size)

    def predict(self, image: Image.Image) -> tuple[str, float]:
        self._load()
        assert self._model is not None
        assert self._transform is not None
        assert self._index_to_label is not None

        probabilities = []
        with torch.no_grad():
            for crop in self.image_selector(image):
                tensor = self._transform(crop).unsqueeze(0).to(self.device)
                outputs = self._model(tensor)
                probabilities.append(torch.softmax(outputs, dim=1).squeeze(0))
        average_probabilities = torch.stack(probabilities).mean(dim=0)
        if self.probability_adjustments:
            weights = torch.ones_like(average_probabilities)
            for index, label in self._index_to_label.items():
                weights[index] = self.probability_adjustments.get(label, 1.0)
            average_probabilities = average_probabilities * weights
            average_probabilities = average_probabilities / average_probabilities.sum()
        confidence, prediction = torch.max(average_probabilities, dim=0)
        label = self._index_to_label[int(prediction.item())]
        return self.label_aliases.get(label, label), float(confidence.item())


class LocalRegressor:
    def __init__(
        self,
        checkpoint_path: Path,
        build_model: Callable,
        build_transforms: Callable,
        image_selector: Callable[[Image.Image], Image.Image],
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.build_model = build_model
        self.build_transforms = build_transforms
        self.image_selector = image_selector
        self.device = get_device()
        self._model = None
        self._transform = None

    def _load(self) -> None:
        if self._model is not None:
            return
        checkpoint = load_checkpoint(self.checkpoint_path, self.device)
        config = checkpoint.get("config", {})
        model_name = str(config.get("model_name", "mobilenet_v3_small"))
        image_size = int(config.get("image_size", 160))
        model = self.build_model(model_name=model_name, use_pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        self._model = model
        self._transform = self.build_transforms(image_size)

    def predict(self, image: Image.Image) -> tuple[float, float]:
        self._load()
        assert self._model is not None
        assert self._transform is not None

        selected_image = self.image_selector(image)
        tensor = self._transform(selected_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = float(self._model(tensor).squeeze().item())
        confidence = max(0.0, min(1.0, 1.0 - abs(value - 0.25) / 0.10))
        return value, confidence


cropper = OpenCVCropper()


def multi_context_eye_crops(image: Image.Image, normalize_side: bool) -> list[Image.Image]:
    crops: list[Image.Image] = []
    for padding in (0.70, 0.95, 1.15):
        crops.extend(cropper.crop_eyes(image, padding=padding, normalize_side=normalize_side))
    return crops

eye_color_classifier = LocalClassifier(
    checkpoint_path=MODEL_ROOTS["eye_color"] / "artifacts" / "exported" / "eye_color_model.pt",
    build_model=build_eye_color_model,
    build_transforms=build_eye_color_transforms,
    image_selector=lambda image: cropper.crop_eyes(image, padding=0.45, normalize_side=False),
    label_aliases={"grey": "gray"},
    probability_adjustments={
        "brown": 1.35,
        "hazel": 1.08,
        "grey": 0.98,
        "green": 0.90,
        "blue": 0.88,
    },
)

eye_shape_classifier = LocalClassifier(
    checkpoint_path=MODEL_ROOTS["eye_shape"] / "artifacts" / "exported" / "eye_shape_model.pt",
    build_model=build_eye_shape_model,
    build_transforms=build_eye_shape_transforms,
    image_selector=lambda image: multi_context_eye_crops(image, normalize_side=True),
)

face_shape_classifier = LocalClassifier(
    checkpoint_path=MODEL_ROOTS["face_shape"] / "artifacts" / "exported" / "face_shape_model.pt",
    build_model=build_face_shape_model,
    build_transforms=build_face_shape_transforms,
    image_selector=lambda image: [cropper.crop_face(image, padding=0.35)],
)

pupil_distance_regressor = LocalRegressor(
    checkpoint_path=MODEL_ROOTS["pupil_distance"] / "artifacts" / "exported" / "pupil_distance_model.pt",
    build_model=build_pupil_distance_model,
    build_transforms=build_pupil_distance_transforms,
    image_selector=lambda image: image,
)
