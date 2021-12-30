import catalogue
from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_cors import cross_origin
from ..services.cv.fireForestService import FireForestService
import json

from PIL import Image
import base64
import io
# import numpy as np
# import torch


cv = Blueprint('cv', __name__)

#generate_code_service = GenerateCodeService()
fire_forest_service = FireForestService()

@cv.route('/fire_forest', methods=['POST'])
def generate_code():
    imageData = request.get_json()["imageData"]
    imageData = imageData.replace("data:image/png;base64", "")
    imageData = imageData.replace("data:image/jpg;base64", "")
    imageData = imageData.replace("data:image/jpeg;base64", "")
    base64_decoded = base64.b64decode(imageData)
    image = Image.open(io.BytesIO(base64_decoded))

    result = fire_forest_service.generate(image)
    # result = "0"
    return dict(string=result)

