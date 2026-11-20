"""SIGNBRIDGE static-sign recognition helpers.

The classifier is only a prototype for verified, static signs. It does not
translate continuous ISL or signed sentences.
"""
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "sign_model.joblib"


class SignRecognizer:
    def __init__(self):
        self.model = None
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.65,
        )
        self.load_model()

    def load_model(self):
        if MODEL_PATH.exists():
            try:
                self.model = joblib.load(MODEL_PATH)
            except Exception as exc:
                print(f"Could not load SIGNBRIDGE model: {exc}")
                self.model = None

    @property
    def labels(self):
        if self.model is None:
            return []
        return [str(label) for label in self.model.classes_]

    def extract_features(self, image_bgr):
        image_bgr = cv2.flip(image_bgr, 1)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        left = [0.0] * 63
        right = [0.0] * 63
        detected = []

        if result.multi_hand_landmarks:
            for index, hand_landmarks in enumerate(result.multi_hand_landmarks):
                handedness = "Unknown"
                if result.multi_handedness:
                    handedness = result.multi_handedness[index].classification[0].label
                values = []
                for point in hand_landmarks.landmark:
                    values.extend([float(point.x), float(point.y), float(point.z)])
                if handedness == "Left":
                    left = values
                elif handedness == "Right":
                    right = values
                detected.append(handedness)

        return np.asarray([left + right], dtype=np.float32), detected

    def predict(self, image_bgr):
        features, detected = self.extract_features(image_bgr)
        if not detected:
            return {
                "status": "no_hand",
                "message": "No hand detected. Place your hand clearly in the camera view.",
                "hands_detected": 0,
                "prediction": None,
                "confidence": None,
            }
        if self.model is None:
            return {
                "status": "model_not_trained",
                "message": "Hand detected. The recognition model has not been trained yet.",
                "hands_detected": len(detected),
                "prediction": None,
                "confidence": None,
            }

        probabilities = self.model.predict_proba(features)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        label = str(self.model.classes_[best_index])
        if confidence < 0.65:
            return {
                "status": "uncertain",
                "message": "Hand detected, but the sign is uncertain. Try again.",
                "hands_detected": len(detected),
                "prediction": None,
                "confidence": round(confidence, 4),
            }
        return {
            "status": "recognized",
            "message": "Prototype prediction only; verify against an ISL reference.",
            "hands_detected": len(detected),
            "prediction": label,
            "confidence": round(confidence, 4),
        }

    def close(self):
        self.hands.close()
