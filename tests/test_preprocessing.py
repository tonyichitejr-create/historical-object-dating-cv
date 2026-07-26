from preprocessing.preprocess import preprocess_image

def test_image():

    img = preprocess_image("sample.jpg")

    assert img.shape == (224,224,3)
