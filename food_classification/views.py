import io
import os
import json

import torch as torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
from django.conf import settings
from googlesearch import search

model = torchvision.models.resnet18(pretrained=True)
model_path = os.path.join(settings.STATIC_ROOT, "vietnam_food.pth")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model.load_state_dict(torch.load(model_path))
model.eval()
# load mapping of ImageNet index to human-readable label (from staticfiles directory)
# run "python manage.py collectstatic" to ensure all static files are copied to the STATICFILES_DIRS
json_path = os.path.join(settings.STATIC_ROOT, "imagenet_class_index.json")
imagenet_mapping = json.load(open(json_path))


def transform_image(image_bytes):
    """
    Transform image into required DenseNet format: 224x224 with 3 RGB channels and normalized.
    Return the corresponding tensor.
    """
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.6189, 0.5372, 0.3989], [0.2598, 0.2522, 0.2834])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    class_name, human_label = imagenet_mapping[predicted_idx]
    return human_label
import base64
from django.shortcuts import render
from .forms import ImageUploadForm

def index(request):
    image_uri = None
    predicted_label = None
    google_result = None

    if request.method == 'POST':
        # in case of POST: get the uploaded image from the form and process it
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # retrieve the uploaded image and convert it to bytes (for PyTorch)
            image = form.cleaned_data['image']
            image_bytes = image.file.read()
            # convert and pass the image as base64 string to avoid storing it to DB or filesystem
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)

            # get predicted label with previously implemented PyTorch function
            try:
                predicted_label = get_prediction(image_bytes)
                google_result = list(search(predicted_label, num=10, stop=10))
            except RuntimeError as re:
                print(re)

    else:
        # in case of GET: simply show the empty form for uploading images
        form = ImageUploadForm()

    # pass the form, image URI, and predicted label to the template to be rendered
    context = {
        'form': form,
        'image_uri': image_uri,
        'predicted_label': predicted_label,
        'google_result': google_result
    }
    return render(request, 'index.html', context)