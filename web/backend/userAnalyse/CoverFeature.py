import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNeXt50_32X4D_Weights, resnext50_32x4d


class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = ResNeXt50_32X4D_Weights.DEFAULT
        self.base_model = resnext50_32x4d(weights=self.weights)
        self.model = nn.Sequential(
            *list(self.base_model.children())[:-2],
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.model.to(self.device)
        self.transforms = self.weights.transforms()
        self.model.eval()

    def preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        image = image.unsqueeze(0).to(self.device)
        return image

    def forward(self, input_batch):
        with torch.no_grad():
            output = self.model(input_batch)
        return output
