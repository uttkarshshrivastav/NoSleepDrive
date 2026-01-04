# inference/eye_infer.py

import torch
import torch.nn.functional as F
from torchvision import transforms
from models.architectures import EyeModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_eye_model = None

_eye_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_eye_model(checkpoint_path):
    global _eye_model

    if _eye_model is not None:
        return _eye_model

    model = EyeModel(num_classes=2)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    _eye_model = model
    return model


def infer_eye(roi, model):
    # infering the roi and giving out the probability of eye closed 
    if roi is None:
        return None

    # transforming the roi in desired mean and standard deviation 
    x = _eye_transform(roi)
    
    # correcting the dimentions
    x = x.unsqueeze(0)
    
    # moving to device
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    prob_closed = probs[1].item()
    prob_open = probs[0].item()

    return {
        "prob": prob_closed,
        "conf": max(prob_closed, prob_open)
    }