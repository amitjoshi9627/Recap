import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Union

from jsonschema import validate
from torch import tensor
from torch.utils.data.dataset import Dataset

from Recap import config, tools
from Recap.constants import SchemaKeys, TrainingKeys
from Recap.schema import input_schema

logging.basicConfig(
    filename=os.path.join(config.OUTPUT_LOG, config.LOG_FILE),
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
)


class SummarizerDataset(Dataset):
    """
    SummarizerDataset class - to load the dataset used the __getitem__ fashion supported by the Pytorch.
    The loader supports the JSON, list and the csv format for parsing the input to the network.

    Attributes:
        model: SummarizerBackbone Model used for tokenization method
        json_data: Input JSON data for training / serving
        text_list: list of texts for training / serving
        summary_list: list of summaries corresponding to text_list. Mandatory to pass this if `is_train` is True
        mode: {`train`, `serve`} mode of running the model

    """

    def __init__(
        self,
        model,
        json_data: Optional[Dict] = None,
        text_list: Optional[List] = None,
        summary_list: Optional[List] = None,
        mode: str = TrainingKeys.TRAIN.value,
    ) -> None:

        super().__init__()

        if mode == TrainingKeys.RETRAIN.value:
            # Code block for retraining of model from feedback loop [TBD].
            raise NotImplementedError("Retraining part of the model is yet to be implemented")

        self.texts = text_list or []
        self.summaries = summary_list or []

        # if JSON is provided
        if json_data is not None:

            self.validate_json(json_data)

            logging.debug("Loading Json file for serving...")

            self.texts = self._extract_texts_from_json(json_data=json_data)

            if mode == TrainingKeys.TRAIN.value:
                self.summaries = self._extract_summaries_from_json(json_data=json_data)

        # Loading data for training of the model from sample file in local_assets
        if not self.texts:
            logging.debug("Loading training CSV file ...")

            train_data_file = config.TRAINING_DATA
            self.data = tools.load_csv(train_data_file)
            self.texts = self.data[config.text_col].values
            self.summaries = self.data[config.summary_col].values

        self.mode = mode
        self.json_data = json_data
        self.input_length = config.INPUT_LENGTH  # Max length for input texts
        self.output_length = config.OUTPUT_LENGTH  # max length for output summaries
        self.model = model

    def validate_json(self, json_data: dict):
        validate(json_data, schema=input_schema)

    def get_index_item(self, index: int) -> Tuple[bool, Dict[str, tensor]]:
        """Returns tokenized texts and labels(if mode is train) for the given index"""

        return self.__getitem__(index)

    def __getitem__(self, index):

        # Keyword Arguments (Kwargs) for input text to be used by Tokenizer
        src_kwargs = {
            "add_special_tokens": True,
            "padding": "max_length",
            "return_tensors": "pt",
            "max_length": self.input_length,
            "truncation": True,
        }

        # Keyword Arguments (Kwargs) for target summaries to be used by Tokenizer
        tgt_kwargs = {
            "add_special_tokens": True,
            "padding": "max_length",
            "return_tensors": "pt",
            "max_length": self.output_length,
            "truncation": True,
        }

        text = self.texts[index]
        input_ = self.clean_text("summarize:  " + text)  # Preprocessing the input texts

        source = self.model.tokenize(input_, **src_kwargs)  # Tokenizing the input texts
        source_ids = source["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()

        labels = []
        target_mask = []

        encodings = {
            "input_ids": source_ids,
            "attention_mask": src_mask,
        }

        if self.mode == TrainingKeys.TRAIN.value or self.mode == TrainingKeys.RETRAIN.value:
            # Preprocessing and tokenizing the summaries when mode is training

            target_ = self.clean_text(self.summaries[index])  # Preprocessing the summaries
            targets = self.model.tokenize(target_, **tgt_kwargs)  # Tokenizing the summaries

            labels = targets["input_ids"].squeeze()
            target_mask = targets["attention_mask"].squeeze()
            labels[labels[:] == self.model.tokenizer.pad_token_id] = -100  # Padding the labels

            encodings["labels"] = labels
            encodings["decoder_attention_mask"] = target_mask
        else:
            encodings["text"] = self.texts[index]

        return encodings

    def clean_text(self, text: str) -> str:
        """
        Base method for applying a list of methods for preprocessing of input texts

        """
        text = self._remove_extra_spaces(text)
        text = self._de_emojify(text)

        return text

    def _de_emojify(self, text: str) -> str:
        """
        Removing emojis from texts

        Parameters:
            text: String

        Returns:
            string: text with emojis removed

        """
        regex_pattern = re.compile(
            pattern="["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE,
        )
        return regex_pattern.sub(r"", text)

    def _remove_extra_spaces(self, text: str) -> str:
        """

        Parameters:
            text: input text for removing extra spaces from text

        Returns:
            text with extra spaces removed
        """
        return re.sub(" +", " ", text)

    def _extract_texts_from_json(self, json_data: Dict) -> List:
        """
        Extracting text from JSON

        Parameters:
            json_data (dict): JSON for the extraction of texts

        Returns:
            List of texts extracted

        """
        text_list = []
        keys_list = json_data[SchemaKeys.INPUT_TEXTS.value]

        for key in keys_list:
            text_list.append(key[SchemaKeys.TEXT.value])

        return text_list

    def _extract_summaries_from_json(self, json_data: Dict) -> List:
        """
        Extracting Summaries from JSON

        Parameters:
            json_data (dict): JSON for the extraction of Summaries

        Returns:
            List of summaries extracted
        """
        summary_list = []
        keys_list = json_data[SchemaKeys.INPUT_TEXTS.value]

        for key in keys_list:
            summary_list.append(key[SchemaKeys.SUMMARY.value])

        return summary_list

    def get_response(
        self,
        predictions: List[str],
        json_data: Dict[str, Union[list, str]],
        status: str,
    ) -> Dict[str, Union[list, str]]:
        """
        Returns Response JSON

        Parameters:
            predictions: list of predictions/ summaries
            json_data: Input JSON
            status: Success or fail/ failure

        Returns:
            Response JSON
        """

        num_samples = len(json_data[SchemaKeys.INPUT_TEXTS.value])

        json_data[SchemaKeys.STATUS.value] = status
        json_data[SchemaKeys.MODEL.value] = config.MODEL_NAME

        for ind in range(num_samples):
            json_data[SchemaKeys.INPUT_TEXTS.value][ind][SchemaKeys.SUMMARY.value] = predictions[
                ind
            ]

        return json_data

    def __len__(self):
        return len(self.texts)
