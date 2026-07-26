import yaml

from training.trainer import Trainer


with open("configs/config.yaml") as f:

    config=yaml.safe_load(f)

trainer=Trainer(config)

trainer.train_cnn()

trainer.train_resnet()
