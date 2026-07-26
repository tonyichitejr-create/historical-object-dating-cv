from fastapi import FastAPI, UploadFile

import numpy as np

from tensorflow.keras.models import load_model

from preprocessing.preprocess import preprocess_image

app = FastAPI()

model = load_model("models/resnet.keras")

classes = [

    "1800-1850",

    "1851-1900",

    "1901-1950",

    "1951-2000"

]


@app.post("/predict")

async def predict(file: UploadFile):

    contents = await file.read()

    with open("temp.jpg","wb") as f:

        f.write(contents)

    image = preprocess_image("temp.jpg")

    prediction = model.predict(np.expand_dims(image,0))[0]

    index = np.argmax(prediction)

    return {

        "prediction": classes[index],

        "confidence": float(prediction[index])

    }
