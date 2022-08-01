import numpy as np
from PIL import Image
import torch
import clip


class Model:
    def __init__(self) -> None:
        # initialise model
        self.model, self.preprocess = clip.load("ViT-B/32")
        self.model.cuda().eval()

    def run_model(self, fp_image, caption) -> float:
        img = Image.open(fp_image).convert("RGB")
    
        image_input = torch.tensor(np.stack([self.preprocess(img)])).cuda()
        text_tokens = clip.tokenize([caption]).cuda()

        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float()
            text_features = self.model.encode_text(text_tokens).float()

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features @ image_features.T

        return similarity.cpu().numpy()[0][0]
