import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from .gauss_pd import GaussPDConfig, GaussPDExtractor
from .model import get_resnet
from .preprocess import Preprocessor
from .utils import imread_any


@dataclass(frozen=True)
class InferenceConfig:
    image_size: int = 224
    chirality_threshold: float = 0.2


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
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location=device)
        class_to_idx = ck.get("class_to_idx")
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)
        model = get_resnet(num_classes, pretrained=False)
        model.load_state_dict(ck["model_state"])
        model.eval()
        return cls(
            model,
            idx_to_class,
            device,
            config=config,
            preprocessor=preprocessor,
            gauss_extractor=gauss_extractor,
        )

    @torch.inference_mode()
    def predict(self, img_path, mapping_csv: Optional[str] = None):
        img = Image.fromarray(imread_any(img_path))
        x = self.transform(img).unsqueeze(0).to(self.device)

        out = self.model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        label = self.idx_to_class[pred]

        pd_code = None
        gauss_code = None
        if mapping_csv is not None:
            df = pd.read_csv(mapping_csv)
            row = df[df["label"] == label]
            if len(row) > 0:
                pd_code = row.iloc[0].get("pd_code")
                gauss_code = row.iloc[0].get("gauss_code")

        skel, _ = self.preprocessor.run(np.array(img))
        gauss_auto, pd_auto = self.gauss_extractor.extract(skel)

        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        x2 = self.transform(img_flip).unsqueeze(0).to(self.device)
        out2 = self.model(x2)
        probs2 = torch.softmax(out2, dim=1).cpu().numpy()[0]

        same_score = probs[pred]
        flip_score_same = probs2[pred]
        chirality_confidence = float(same_score - flip_score_same)

        chirality = "ambiguous"
        if chirality_confidence > self.config.chirality_threshold:
            chirality = "right-handed (prediction favors original)"
        elif chirality_confidence < -self.config.chirality_threshold:
            chirality = "left-handed (prediction favors mirrored)"

        return {
            "predicted_label": label,
            "pred_prob": float(probs[pred]),
            "mapping_pd": pd_code,
            "mapping_gauss": gauss_code,
            "auto_gauss": gauss_auto,
            "auto_pd": pd_auto,
            "chirality": chirality,
            "chirality_confidence": chirality_confidence,
        }


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    class_to_idx = ck.get("class_to_idx")
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    model = get_resnet(num_classes, pretrained=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, idx_to_class


def infer_image(img_path, checkpoint, mapping_csv=None):
    recognizer = KnotRecognizer.from_checkpoint(checkpoint)
    return recognizer.predict(img_path, mapping_csv=mapping_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mapping', default=None)
    args = parser.parse_args()
    res = infer_image(args.image, args.checkpoint, args.mapping)

    import json
    print(json.dumps(res, indent=2, default=str))


if __name__ == '__main__':
    main()
