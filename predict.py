import sys
import numpy as np

from tensorflow.keras.models import load_model

from preprocessing.preprocess import preprocess_image

CLASSES=[

    "1800-1850",

    "1851-1900",

    "1901-1950",

    "1951-2000"

]

model=load_model("models/resnet.keras")

image=preprocess_image(sys.argv[1])

prediction=model.predict(np.expand_dims(image,0))[0]

index=np.argmax(prediction)

print()

print("Prediction:",CLASSES[index])

print()

print("Confidence:",prediction[index])
