from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import *

from tensorflow.keras.optimizers import Adam


def build_cnn(num_classes=4):

    model = Sequential()

    model.add(Conv2D(

        32,

        (3,3),

        activation="relu",

        input_shape=(224,224,3)

    ))

    model.add(MaxPooling2D())

    model.add(Conv2D(64,(3,3),activation="relu"))

    model.add(MaxPooling2D())

    model.add(Conv2D(128,(3,3),activation="relu"))

    model.add(MaxPooling2D())

    model.add(Conv2D(256,(3,3),activation="relu"))

    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(512,activation="relu"))

    model.add(Dropout(.5))

    model.add(Dense(num_classes,activation="softmax"))

    model.compile(

        optimizer=Adam(0.001),

        loss="categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model
