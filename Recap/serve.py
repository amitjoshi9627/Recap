import os
import sys
from typing import Dict, List, Union

import numpy as np
from torch.utils.data.dataloader import DataLoader

from Recap import config, tools
from Recap.constants import ServingKeys
from Recap.dataset import SummarizerDataset
from Recap.engine import SummarizerEngine
from Recap.model import SummarizerBackbone


def model_serve(data: Union[Dict, List]) -> Dict:
    """
    Model serving function

    Parameters:
        data: Input texts/ JSON for prediction from the API request (format - list, JSON).

    Returns:
        Response from the Summarizer serving component.

    """
    # Loading backbone model
    backbone_model = SummarizerBackbone(model_name=config.BASE_FINETUNED_MODEL)

    if isinstance(data, (list, np.ndarray)):

        # Initializing the Summarizer Dataset
        dataset = SummarizerDataset(
            model=backbone_model, text_list=data, mode=ServingKeys.SERVE.value
        )

    else:
        dataset = SummarizerDataset(
            model=backbone_model, json_data=data, mode=ServingKeys.SERVE.value
        )

    # Initialize the Data-loader.
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Run the Summarizer Engine
    summarization_engine = SummarizerEngine(backbone_model, dataset, data_loader)

    # Getting the response
    response = summarization_engine.serve(save_result=False)
    return response
