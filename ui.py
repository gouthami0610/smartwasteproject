'''import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model('waste_classifier_models.h5')

#model=tf.models.load_model('waste_classifier_models.h5')
class_names=['cardboard','glass','metal','paper','plastic','trash']

def predict_waste(img):
    img=img.resize((224,224))
    img_array=image.img_to_array(img)
    img_array=np.expand_dims(img_array,axis=0)
    img_array=preprocess_input(img_array)

    #predict
    prediction=model.predict(img_array)[0]
    prediction_class=class_names[np.argmax(prediction)]
    confidence=float(np.max(prediction))*100
    probs={cls:float(prob) for cls,prob in zip(class_names,prediction)}
    return prediction_class,confidence,probs


with gr.Blocks() as demo:
    gr.Markdown("waste classification")
    with gr.Row():
        img_input=gr.Image(type='pil',lable='upload an image')
        with gr.Column():
            out_class=gr.Textbox(lable='predicted class')
            out_conf=gr.Number(lable='confidence level')
            out_prob=gr.Lable(lable='class probability')
    btn=gr.Button("classifier")
    btn.clixck(fn=predict_waste,inputs=img_input,
               outputs=[out_class,out_conf,out_prob])
demo.launch()'''

import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('waste_classifier_models.h5')

# Define class labels
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Prediction function
def predict_waste(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)[0]
    prediction_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    probs = {cls: float(prob) for cls, prob in zip(class_names, prediction)}
    
    return prediction_class, confidence, probs

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## 🗑️ Smart Waste Classification")
    
    with gr.Row():
        img_input = gr.Image(type='pil', label='Upload an Image')
        
        with gr.Column():
            out_class = gr.Textbox(label='Predicted Class')
            out_conf = gr.Number(label='Confidence Level (%)')
            out_prob = gr.Label(label='Class Probabilities')

    btn = gr.Button("Classify")
    btn.click(fn=predict_waste, inputs=img_input, outputs=[out_class, out_conf, out_prob])

# Launch the app
demo.launch()


