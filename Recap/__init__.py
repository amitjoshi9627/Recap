import os
import sys
from typing import Dict, Union, List

import numpy as np
from torch.utils.data.dataloader import DataLoader

from Recap import config
from Recap.dataset import SummarizerDataset
from Recap.engine import SummarizerEngine
from Recap.model import SummarizerBackbone
from Recap import tools

sys.path.insert(1, os.getcwd())


def model_serve(data: Union[Dict, List]) -> Dict:
    """
    Model serving function

    Parameters:
        data: Input texts/ JSON for prediction from the API request (format - list, JSON).

    Returns:
        Response from the Summarizer serving component.

    """
    if isinstance(data, (list, np.ndarray)):
        data = tools.list_to_json(data)

    # Loading backbone model
    backbone_model = SummarizerBackbone(model_name=config.BASE_FINETUNED_MODEL)

    # Initializing the Summarizer Dataset
    dataset = SummarizerDataset(model=backbone_model, json_data=data, is_train=False)

    # Initialize the Data-loader.
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Run the Summarizer Engine
    summarization_engine = SummarizerEngine(backbone_model, dataset, data_loader)

    # Getting the response
    response = summarization_engine.serve(save_result=False)
    return response
