import os

from tensorflow.keras.callbacks import *

from preprocessing.dataset import create_generators

from models.cnn import build_cnn

from models.resnet50 import build_resnet


class Trainer:

    def __init__(self,config):

        self.config=config

    def callbacks(self):

        return [

            EarlyStopping(

                patience=8,

                restore_best_weights=True

            ),

            ReduceLROnPlateau(

                factor=0.3,

                patience=3

            ),

            ModelCheckpoint(

                filepath="checkpoints/model.keras",

                save_best_only=True

            ),

            TensorBoard(

                log_dir="logs"

            )

        ]

    def train_cnn(self):

        train,val,test=create_generators(

            self.config["dataset"]["train"],

            self.config["dataset"]["validation"],

            self.config["dataset"]["test"]

        )

        model=build_cnn()

        history=model.fit(

            train,

            validation_data=val,

            epochs=self.config["epochs"],

            callbacks=self.callbacks()

        )

        model.save("models/cnn.keras")

        return history

    def train_resnet(self):

        train,val,test=create_generators(

            self.config["dataset"]["train"],

            self.config["dataset"]["validation"],

            self.config["dataset"]["test"]

        )

        model=build_resnet()

        history=model.fit(

            train,

            validation_data=val,

            epochs=self.config["epochs"],

            callbacks=self.callbacks()

        )

        model.save("models/resnet.keras")

        return history
