import cv2
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

IMAGE_SIZE=(224,224)


def features(image):

    image=cv2.resize(image,IMAGE_SIZE)

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    hist=cv2.calcHist([gray],[0],None,[64],[0,256])

    hist=hist.flatten()

    hist/=np.sum(hist)

    return hist


def load(directory):

    X=[]
    y=[]

    folders=sorted(Path(directory).iterdir())

    for label,folder in enumerate(folders):

        for img in folder.glob("*"):

            image=cv2.imread(str(img))

            if image is None:
                continue

            X.append(features(image))

            y.append(label)

    return np.array(X),np.array(y)


def train(train_dir,test_dir):

    X_train,y_train=load(train_dir)

    X_test,y_test=load(test_dir)

    model=RandomForestClassifier(

        n_estimators=300,

        random_state=42

    )

    model.fit(X_train,y_train)

    predictions=model.predict(X_test)

    print(accuracy_score(y_test,predictions))

    joblib.dump(model,"models/random_forest.pkl")
