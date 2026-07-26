from models.cnn import build_cnn
from models.resnet50 import build_resnet

def test_cnn():

    model=build_cnn()

    assert model.output_shape[-1]==4

def test_resnet():

    model=build_resnet()

    assert model.output_shape[-1]==4
