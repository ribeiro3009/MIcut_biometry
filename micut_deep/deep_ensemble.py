import warnings
import os
import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.exceptions import InconsistentVersionWarning
from torchvision import models

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

class RegressionHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_layer = nn.Sequential(nn.Linear(in_features, out_features))
    def forward(self, x):
        return self.out_layer(x)


class AFQAModel(nn.Module):
    def __init__(self, outputs, fcn=256):
        super().__init__()
        self.outputs = outputs
        self.fcn = fcn
        self.encoder = models.densenet121(weights="DEFAULT")
        self.encoder.features[0].weight = nn.Parameter(
            torch.sum(self.encoder.features[0].weight, dim=1, keepdim=True)
        )
        self.encoder.classifier = nn.Sequential(
            nn.Linear(1024, self.fcn), nn.LeakyReLU(),
        )
        self.output = RegressionHead(self.fcn, self.outputs)
    def forward(self, x):
        features_vector = self.encoder(x)
        out = self.output(features_vector)
        return out
    def load_weights(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])


class DeepEnsemble:
    MODEL_NAMES = ["vfq", "nfq", "lqm", "mor"]

    def __init__(self, resources_dir: str, device: str = "cpu"):
        self.device = device
        self.imsize = 512

        pca_path = os.path.join(resources_dir, "pca_fusion_model.pkl")
        model_path = os.path.join(resources_dir, "model_densenet121.pt")
        if not os.path.exists(pca_path) or not os.path.exists(model_path):
            raise FileNotFoundError("Missing resources in resources_dir (pca_fusion_model.pkl / model_densenet121.pt)")

        with open(pca_path, "rb") as handle:
            self.pca_coeffs = pickle.load(handle)

        self.deep_model = AFQAModel(outputs=4, fcn=512).to(device)
        self.deep_model.load_weights(model_path)

    def predict_ensemble(self, input_image):
        if len(input_image.shape) != 2 or input_image.dtype != np.uint8:
            raise TypeError("The input image is expected to be a 2D array in 8 bit grayscale color.")
        x = cv2.resize(input_image, (self.imsize, self.imsize), interpolation=cv2.INTER_NEAREST)
        x = x.astype(np.float32) / 255
        x = np.expand_dims(x, 0)
        x = np.expand_dims(x, 0)
        pred_labels = self.deep_model(torch.from_numpy(x).to(self.device))
        predictions = pred_labels.squeeze(1).detach().cpu().numpy()[0]
        ensemble_predictions = {}
        for model, prediction in zip(self.MODEL_NAMES, predictions):
            ensemble_predictions[model] = prediction
        return ensemble_predictions

    def fusion(self, ensemble_predictions):
        vfq = int(ensemble_predictions["vfq"]) 
        nfq = int(ensemble_predictions["nfq"]) 
        lqm = int(ensemble_predictions["lqm"]) 
        mor = int(ensemble_predictions["mor"]) 
        pca_transform = ((self.pca_coeffs["model"].transform([[nfq, vfq, lqm, mor]]) - self.pca_coeffs["min"]) / (
                    self.pca_coeffs["max"] - self.pca_coeffs["min"]))[0][0]
        fusion_quality = int(np.clip(pca_transform, a_min=0, a_max=100) * 100)
        return fusion_quality