import json
import time

import pandas as pd
from summarizers import Summarizers

from Recap import config
from Recap import model_serve
from Recap.utils import tools


def new_model_serve(texts):
    pred = []
    summ = Summarizers(device="cuda")
    for text in texts:
        pred.append(summ(text))
    return pred


def load_json():
    path = f"C:/Users/AG94866/OneDrive - Anthem/Documents/Python Scripts/thehiveproject/src/ml/scripts/Amit/Summarizer API Package/Summarizer/dataset/{config.TEST_JSON}"
    with open(path) as file:
        json_data = json.load(file)
    return json_data


def api_integration(test_input):
    return model_serve(test_input)


if __name__ == "__main__":
    st = time.time()
    df = pd.read_csv(
        "C:/Users/AG94866/OneDrive - Anthem/Documents/Python Scripts/thehiveproject/src/ml/scripts/Amit/Summarizer API Package/Summarizer/dataset/3rd_person_pov_eval.csv"
    )
    text = df[config.text_col].values
    TEST_INPUT = load_json()
    output = api_integration(TEST_INPUT)
    print(output)
    # summ1 = []
    # for key in output[config.response_column]:
    #     summ1.append(key[config.summary_col])
    # summ2 = new_model_serve(text)

    # for i in range(len(text)):
    #     print(text[i])
    #     print(summ1[i])
    #     print(summ2[i])
    #     print("-" * 75)
    #     print()

    print(f"Time taken {time.time() - st}")
