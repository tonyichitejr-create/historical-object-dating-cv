from tensorflow.keras.applications import ResNet50

from tensorflow.keras.layers import *

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam


def build_resnet(classes=4):

    base = ResNet50(

        include_top=False,

        weights="imagenet",

        input_shape=(224,224,3)

    )

    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)

    x = Dense(512,activation="relu")(x)

    x = Dropout(.5)(x)

    outputs = Dense(classes,activation="softmax")(x)

    model = Model(base.input,outputs)

    model.compile(

        optimizer=Adam(0.001),

        loss="categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model
