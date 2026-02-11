import argparse
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from .gauss_pd import GaussPDConfig, GaussPDExtractor
from .features import extract_symmetry_invariant_features
from .model import get_resnet
from .preprocess import Preprocessor
from .utils import imread_any


@dataclass(frozen=True)
class InferenceConfig:
    image_size: int = 224
    chirality_threshold: float = 0.2
    include_features: bool = False
    feature_resize: int = 256


class KnotRecognizer:
    def __init__(
        self,
        model,
        idx_to_class,
        device,
        config: Optional[InferenceConfig] = None,
        preprocessor: Optional[Preprocessor] = None,
        gauss_extractor: Optional[GaussPDExtractor] = None,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.idx_to_class = idx_to_class
        self.device = device
        self.config = config or InferenceConfig()
        self.transform = T.Compose(
            [T.Resize((self.config.image_size, self.config.image_size)), T.ToTensor()]
        )
        self.preprocessor = preprocessor or Preprocessor()
        self.gauss_extractor = gauss_extractor or GaussPDExtractor(GaussPDConfig())

    @classmethod
    def from_checkpoint(
        cls,
        path,
        device: Optional[torch.device] = None,
        config: Optional[InferenceConfig] = None,
        preprocessor: Optional[Preprocessor] = None,
        gauss_extractor: Optional[GaussPDExtractor] = None,
    ):
        device = _resolve_device(device)
        model, idx_to_class = _load_model_from_checkpoint(path, device)
        return cls(
            model,
            idx_to_class,
            device,
            config=config,
            preprocessor=preprocessor,
            gauss_extractor=gauss_extractor,
        )

    @staticmethod
    def _chirality_from_scores(
        same_score: float, flip_score: float, threshold: float
    ) -> Tuple[str, float]:
        confidence = float(same_score - flip_score)
        chirality = "ambiguous"
        if confidence > threshold:
            chirality = "right-handed (prediction favors original)"
        elif confidence < -threshold:
            chirality = "left-handed (prediction favors mirrored)"
        return chirality, confidence

    def _predict_probs(self, img: Image.Image) -> np.ndarray:
        x = self.transform(img).unsqueeze(0).to(self.device)
        out = self.model(x)
        return torch.softmax(out, dim=1).cpu().numpy()[0]

    def _maybe_features(self, img: Image.Image):
        if not self.config.include_features:
            return None
        feats = extract_symmetry_invariant_features(
            np.array(img), resize_to=self.config.feature_resize
        )
        return feats.tolist()

    @torch.inference_mode()
    def predict(self, img_path, mapping_csv: Optional[str] = None):
        img = Image.fromarray(imread_any(img_path))
        return self.predict_image(img, mapping_csv=mapping_csv)

    @torch.inference_mode()
    def predict_image(self, img: Image.Image, mapping_csv: Optional[str] = None):
        probs = self._predict_probs(img)
        pred = int(np.argmax(probs))
        label = self.idx_to_class[pred]

        pd_code, gauss_code = _lookup_mapping(label, mapping_csv)

        skel, _ = self.preprocessor.run(np.array(img))
        gauss_auto, pd_auto = self.gauss_extractor.extract(skel)
        features = self._maybe_features(img)

        probs_flip = self._predict_probs(img.transpose(Image.FLIP_LEFT_RIGHT))
        chirality, chirality_confidence = self._chirality_from_scores(
            probs[pred], probs_flip[pred], self.config.chirality_threshold
        )

        return {
            "predicted_label": label,
            "pred_prob": float(probs[pred]),
            "mapping_pd": pd_code,
            "mapping_gauss": gauss_code,
            "auto_gauss": gauss_auto,
            "auto_pd": pd_auto,
            "features": features,
            "chirality": chirality,
            "chirality_confidence": chirality_confidence,
        }


def load_checkpoint(path, device="cpu"):
    device = _resolve_device(None, str(device))
    return _load_model_from_checkpoint(path, device)


def infer_image(
    img_path,
    checkpoint,
    mapping_csv=None,
    config: Optional[InferenceConfig] = None,
    device: Optional[str] = None,
):
    recognizer = KnotRecognizer.from_checkpoint(
        checkpoint, config=config, device=_resolve_device(None, device)
    )
    return recognizer.predict(img_path, mapping_csv=mapping_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mapping', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--features', action='store_true')
    args = parser.parse_args()
    config = InferenceConfig(include_features=args.features)
    res = infer_image(
        args.image,
        args.checkpoint,
        args.mapping,
        config=config,
        device=args.device,
    )

    import json
    print(json.dumps(_json_safe(res), indent=2, default=str))


def _resolve_device(device: Optional[torch.device], device_str: Optional[str] = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def _load_model_from_checkpoint(path: str, device: torch.device):
    ck = torch.load(path, map_location=device)
    class_to_idx = ck.get("class_to_idx")
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    model = get_resnet(num_classes, pretrained=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, idx_to_class


@lru_cache(maxsize=4)
def _load_mapping_df(mapping_csv: str) -> pd.DataFrame:
    return pd.read_csv(mapping_csv)


def _lookup_mapping(label: str, mapping_csv: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if mapping_csv is None:
        return None, None
    df = _load_mapping_df(mapping_csv)
    row = df[df["label"] == label]
    if len(row) == 0:
        return None, None
    return row.iloc[0].get("pd_code"), row.iloc[0].get("gauss_code")


def _json_safe(obj):
    if isinstance(obj, dict):
        safe = {}
        for k, v in obj.items():
            if not isinstance(k, (str, int, float, bool)) and k is not None:
                k = str(k)
            safe[k] = _json_safe(v)
        return safe
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


if __name__ == '__main__':
    main()
