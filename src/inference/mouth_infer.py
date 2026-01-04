import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

from models.architectures import YawnModel


# select device once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# global model holder  
_yawn_model = None


def load_yawn_model(checkpoint_path):
    global _yawn_model

    # if already loaded, reuse
    if _yawn_model is not None:
        return _yawn_model

    # create model architecture
    model = YawnModel(num_classes=2)
    # load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # move model to device and set eval mode
    model.to(device)
    model.eval()
    # cache model so it loads only once
    _yawn_model = model
    return _yawn_model


# preprocessing must match training
_mouth_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def infer_yawn(roi, model):
    """
    roi: np.ndarray (H, W, 3) or None
    model: loaded YawnModel
    """
        # if mouth ROI not detected
    if roi is None:
        return None

    # preprocess ROI to tensor to batch dimension
    x = _mouth_transform(roi).unsqueeze(0).to(device)


    # forward pass (no gradients)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]


    prob_yawn = probs[1].item()
    prob_no_yawn = probs[0].item()


    return {
        "pred": int(prob_yawn >= 0.5),
        "prob": prob_yawn,
        "conf": max(prob_yawn, prob_no_yawn)
    }

