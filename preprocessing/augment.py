from tensorflow.keras.preprocessing.image import ImageDataGenerator

generator = ImageDataGenerator(

    rotation_range=20,

    horizontal_flip=True,

    zoom_range=0.20,

    brightness_range=[0.8,1.2],

    fill_mode="nearest"

)
