from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image
import io

router = APIRouter()

# Load model once at startup
model = load_model('waste_classifier_models.h5')
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def predict_image(img: Image.Image):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    probs = {cls: float(f"{prob:.4f}") for cls, prob in zip(class_names, prediction)}

    return predicted_class, confidence, probs

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        pred_class, confidence, probs = predict_image(img)
        return JSONResponse(content={
            "prediction": pred_class,
            "confidence": f"{confidence:.2f}%",
            "probabilities": probs
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
