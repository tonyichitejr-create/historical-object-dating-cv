import numpy as np

from sklearn.metrics import *

import matplotlib.pyplot as plt


def evaluate(model,test_generator):

    probabilities=model.predict(test_generator)

    predictions=np.argmax(probabilities,axis=1)

    labels=test_generator.classes

    print()

    print(classification_report(labels,predictions))

    print()

    print("Accuracy:",accuracy_score(labels,predictions))

    print("Precision:",precision_score(labels,predictions,average="weighted"))

    print("Recall:",recall_score(labels,predictions,average="weighted"))

    print("F1:",f1_score(labels,predictions,average="weighted"))

    cm=confusion_matrix(labels,predictions)

    plt.imshow(cm)

    plt.colorbar()

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.show()
