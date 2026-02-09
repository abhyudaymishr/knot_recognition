import argparse
import torch
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import numpy as np

from .model import get_resnet
from .preprocess import preprocess_for_skeleton
from .gauss_pd import extract_gauss_code
from .utils import imread_any


def load_checkpoint(path, device='cpu'):
    ck = torch.load(path, map_location=device)
    class_to_idx = ck.get('class_to_idx')
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    model = get_resnet(num_classes, pretrained=False)
    model.load_state_dict(ck['model_state'])
    model.eval()
    return model, idx_to_class


def infer_image(img_path, checkpoint, mapping_csv=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, idx_to_class = load_checkpoint(checkpoint, device)
    
    # load image
    img = Image.fromarray(imread_any(img_path))
    ts = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    x = ts(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    label = idx_to_class[pred]

    # mapping lookup
    pd_code = None
    gauss_code = None
    if mapping_csv is not None:
        df = pd.read_csv(mapping_csv)
        row = df[df['label'] == label]
        if len(row) > 0:
            pd_code = row.iloc[0].get('pd_code')
            gauss_code = row.iloc[0].get('gauss_code')

    
    skel, gray = preprocess_for_skeleton(np.array(img))
    gauss_auto, pd_auto = extract_gauss_code(skel)

    
    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    x2 = ts(img_flip).unsqueeze(0).to(device)
    with torch.no_grad():
        out2 = model(x2)
        probs2 = torch.softmax(out2, dim=1).cpu().numpy()[0]

    same_score = probs[pred]
    flip_score_same = probs2[pred]
    chirality_confidence = float(same_score - flip_score_same)

    chirality = 'ambiguous'
    if chirality_confidence > 0.2:
        chirality = 'right-handed (prediction favors original)'
    elif chirality_confidence < -0.2:
        chirality = 'left-handed (prediction favors mirrored)'

    return {
        'predicted_label': label,
        'pred_prob': float(probs[pred]),
        'mapping_pd': pd_code,
        'mapping_gauss': gauss_code,
        'auto_gauss': gauss_auto,
        'auto_pd': pd_auto,
        'chirality': chirality,
        'chirality_confidence': chirality_confidence,
    }


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
