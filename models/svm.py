import cv2
import numpy as np
import joblib

from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

IMAGE_SIZE = (224,224)


def extract_features(image):

    image = cv2.resize(image, IMAGE_SIZE)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray],[0],None,[64],[0,256])

    hist = hist.flatten()

    hist /= np.sum(hist)

    return hist


def load_dataset(directory):

    X = []
    y = []

    classes = sorted(Path(directory).iterdir())

    for label, folder in enumerate(classes):

        for image_path in folder.glob("*"):

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            X.append(extract_features(image))
            y.append(label)

    return np.array(X), np.array(y)


def train(train_dir,test_dir):

    X_train,y_train = load_dataset(train_dir)
    X_test,y_test = load_dataset(test_dir)

    model = SVC(kernel="rbf",probability=True)

    model.fit(X_train,y_train)

    predictions = model.predict(X_test)

    print("Accuracy:",accuracy_score(y_test,predictions))

    joblib.dump(model,"models/svm.pkl")
