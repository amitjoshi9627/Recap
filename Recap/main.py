"""
Highest level python script for the Summarizer. This script can be used to run the package in the
standalone mode which enables the Summarizer to be trained, evaluated, served and tested agnostic of the
application interface code.

"""
import argparse
import logging
import os
import sys
from typing import Dict, Optional

from torch.utils.data.dataloader import DataLoader

from Recap import config
from Recap.constants import ServingKeys, TestingKeys, TrainingKeys
from Recap.dataset import SummarizerDataset
from Recap.engine import SummarizerEngine
from Recap.model import SummarizerBackbone
from Recap import tools

# Appending the Package path to the system PATH variable
sys.path.append("../")

# Adding the StreamHandler to print the logfile output to the stderr stream.
logging.getLogger().addHandler(logging.StreamHandler())
logging.basicConfig(
    filename=os.path.join(config.OUTPUT_LOG, config.LOG_FILE),
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
)


def arg_parser():
    """
    ArgumentParser for parsing of the various arguments passed

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="train, eval, and serve mode to run the summarizer engine",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--file_path",
        help="Input data file path for serving the model",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--text",
        help="text to be summarised",
        type=str,
        required=False,
    )
    args = parser.parse_args()

    return args


def network(mode: str, json_data: Optional[Dict] = None) -> SummarizerEngine:
    """
    Function to initialise the Summarizer engine for using the model in train, eval or serve mode

    Parameters:
        mode: {`train`, `eval`, `serve`} mode of Summarizer model
        json_data: input JSON for the model

    Returns:
        Summarizer Engine class which contains the method train, evaluate and serve for the Summarizer model

    """
    # Initialize Model Creator Class, load the model architecture and the weights.

    if mode == TrainingKeys.TRAIN.value:
        shuffle = True
        model_name = config.SUMMARIZATION_MODEL
    else:
        shuffle = False
        model_name = config.BASE_FINETUNED_MODEL

    backbone_model = SummarizerBackbone(model_name)

    # Initializing the dataset
    dataset = SummarizerDataset(backbone_model, json_data=json_data, mode=mode)

    # Initialize the Data-loader.
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle)

    # Run the Sentiment Engine.
    model_engine = SummarizerEngine(
        model=backbone_model,
        dataset=dataset,
        data_loader=data_loader,
    )
    return model_engine


def train(func_test: bool = False) -> None:
    """
    Wrapper function for Training of the Summarizer model engine

    Parameters:
        func_test: True for model's package testing

    Returns:
        None

    """
    model_engine = network(TrainingKeys.TRAIN.value, func_test=func_test)
    model_engine.train(func_test=func_test)

    return None


def evaluate(func_test: bool = False) -> None:
    """
    Wrapper function for Evaluation of the Summarizer model engine

    Parameters:
        func_test: True for model's package testing

    Returns:
        None

    """
    model_engine = network(TestingKeys.EVAL.value, func_test=func_test)
    model_engine.evaluate()

    return None


def serve(json_data: Optional[Dict] = None, save_result: bool = False):
    """
    Wrapper function for serving of the Summarizer model engine

    Parameters:
        json_data: Input JSON for serving
        save_result: whether to save result locally or not (True for functional testing)

    Returns:
        API response for the input JSON

    """
    model_engine = network(ServingKeys.serve.value, json_data=json_data)
    response = model_engine.serve(save_result=save_result)
    return response


def package_test(test_mode: str) -> None:
    """
    Package test for functional testing of the Summarizer package

    Parameters:
        test_mode:{`all`, `train`, `test`, `eval`} mode of the functional testing

    Returns:
        None

    """
    if test_mode == TestingKeys.ALL.value:

        logging.debug("Testing all the components of the package..")

        train(func_test=True)
        logging.debug("Training Component test passed..")

        evaluate()
        logging.debug("Evaluation Component test passed..")

        serve(save_result=True)
        logging.debug("Serving component test passed..")

    elif test_mode == TrainingKeys.TRAIN.value:

        logging.debug("Testing Training component..")

        train(func_test=True)

        logging.debug("Training component test passed..")

    elif test_mode == ServingKeys.SERVE.value:

        logging.debug("Testing Serving component")

        serve(save_result=True)

        logging.debug("Serving component test passed..")

    elif test_mode == TestingKeys.EVAL.value:

        logging.debug("Testing Evaluation component")

        evaluate()

        logging.debug("Evaluating Component passed..")

    else:
        logging.exception(
            "Invalid argument for func_test. Argument should be from {`all`, `train`, `test`, `eval`}"
        )
    return None


def main() -> None:
    """
    Main function for the Summarizer package

    """
    cmd_args = arg_parser()
    if cmd_args.mode == TrainingKeys.TRAIN.value:

        #  Training the model
        logging.debug("Initializing Summarizer model for training..")

        model_engine = network(cmd_args.mode)
        model_engine.train()

    elif cmd_args.mode == TestingKeys.EVAL.value:

        # Evaluating the model
        logging.debug("Initializing Summarizer model for Evaluation..")

        model_engine = network(cmd_args.mode)
        model_engine.evaluate()

    elif cmd_args.mode == ServingKeys.SERVE.value:

        # Serving process of the model
        logging.debug("Initializing Summarizer model for Serving..")

        if cmd_args.file_path:
            json_data = tools.load_json(cmd_args.file_path)
        else:
            json_data = None

        model_engine = network(cmd_args.mode, json_data)
        response = model_engine.serve()

        logging.debug(f"The response from the Model : {response}")

    else:
        logging.exception("Invalid Argument mode argument passed..", exc_info=True)

    return None


if __name__ == "__main__":
    main()
