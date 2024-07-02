# Test with new image
import os
from typing import Union
from PIL import Image
import torch
from preprocessing import transform


CLASS_NAMES = ['Cat', 'Dog']


def predict(image_path: Union[str, os.PathLike], model: torch.nn.Module):
    image = Image.open(image_path).convert('RGB')
    # Put the image in a batch of size 1
    # to match the size expected by the model
    image = transform(image).unsqueeze(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    # Map the prediction to the class label
    predicted_class = CLASS_NAMES[preds.item()]
    return predicted_class
