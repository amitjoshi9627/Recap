from typing import List
from Recap import Summarizers

from Recap import config
from Recap import model_serve


def serve(text_list: List[str]) -> List[str]:
    """
    Model Serve method

    Arguments:
        text_list: List of texts to be summarised

    Returns:
        List of summaries
    """
    pred = []
    summ = Summarizers(device=config.DEVICE)
    for text in text_list:
        pred.append(summ(text))
    return pred
