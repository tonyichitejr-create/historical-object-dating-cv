import os
import requests
from pathlib import Path
from tqdm import tqdm

DATASETS = {
    "smithsonian": "https://api.si.edu/openaccess/api/v1.0/",
    "loc": "https://www.loc.gov/",
    "europeana": "https://api.europeana.eu/"
}


def download_file(url, output):

    response = requests.get(url, stream=True)

    total = int(response.headers.get("content-length", 0))

    with open(output, "wb") as file:

        for chunk in tqdm(
            response.iter_content(1024),
            total=total // 1024,
            unit="KB"
        ):
            file.write(chunk)


def create_structure():

    periods = [
        "1800_1850",
        "1851_1900",
        "1901_1950",
        "1951_2000"
    ]

    for split in ["train", "validation", "test"]:

        for period in periods:

            Path(f"data/{split}/{period}").mkdir(
                parents=True,
                exist_ok=True
            )


if __name__ == "__main__":

    create_structure()

    print("Dataset folders created.")
