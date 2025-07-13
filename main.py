from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tensorflow as tf
from transformers import AutoTokenizer, TFBertModel
import numpy as np

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = tf.keras.models.load_model("bert_classification_model", custom_objects={"TFBertModel": TFBertModel})

# Define label names
label_name = ["Art and Music", "Food", "History", "Manufacturing", "Science and Technology", "Travel"]

# Pydantic model for request body
class TextInput(BaseModel):
    texts: list[str]

def predict_custom_text(texts, model, tokenizer, max_len=128, label_name=None):
    inputs = tokenizer(
        texts,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

    input_dict = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask']
    }

    probs = model.predict(input_dict)
    pred_labels = tf.argmax(probs, axis=1).numpy()

    predictions = []
    for i, text in enumerate(texts):
        pred_dict = {
            'Text': text[:50] + "..." if len(text) > 50 else text,
            'Predicted Class': label_name[pred_labels[i]],
            'Probabilities': {label: float(prob) for label, prob in zip(label_name, probs[i])}
        }
        predictions.append(pred_dict)

    return pred_labels, probs, predictions

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(text_input: TextInput):
    texts = text_input.texts
    pred_labels, pred_probs, predictions = predict_custom_text(texts, model, tokenizer, max_len=128, label_name=label_name)
    return {"predictions": predictions}