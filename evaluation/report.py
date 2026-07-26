import pandas as pd

from sklearn.metrics import classification_report


def save_report(y_true, y_pred):

    report = classification_report(

        y_true,

        y_pred,

        output_dict=True

    )

    df = pd.DataFrame(report).transpose()

    df.to_csv("evaluation/report.csv")
