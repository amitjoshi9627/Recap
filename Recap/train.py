import logging
import os
import time
from typing import Dict, List, Union

from rouge_score import rouge_scorer
from torch.utils.data.dataloader import DataLoader

from Recap import config
from Recap.model import SummarizerBackbone
from Recap import tools

logging.basicConfig(
    filename=os.path.join(config.OUTPUT_LOG, config.LOG_FILE),
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
)


def train(model: SummarizerBackbone, train_data: DataLoader) -> SummarizerBackbone:
    """

    Function to train the T5 model

    Parameters:
        model: Summarizer backbone (T5 model)
        train_data: Data loader/ batcher for training data

    Returns:
        Fine tuned T5 model

    """

    t0 = time.time()
    device = model.device
    model.to(device)
    model.train()

    for epoch in range(config.NUM_EPOCHS):

        logging.info(f"[{model.model_name}] Epoch {epoch + 1} Started..")

        len_dataset = len(train_data)
        interval = max(len_dataset // config.NUM_LOGGING, 1)

        for index, batch in enumerate(train_data):
            if (index + 1) % interval == 0:
                logging.info(
                    f"[{model.model_name}] Epoch {epoch + 1} Iteration {index + 1} Running.."
                )
            model.optimizer.zero_grad()

            # Loading Data into GPU/ CPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            decoder_attention_mask = batch["decoder_attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )
            loss = outputs[0]
            loss.backward()
            model.optimizer.step()

            del batch  # Clearing batch_data from memory

    logging.debug(
        f"[{model.model_name}] Training took {round(time.time() - t0, 2)} seconds"
    )

    return model


def train_model(model: SummarizerBackbone, train_data: Union[DataLoader, List]) -> None:
    """
    Wrapper function for training of Summarizer model

    Parameters:
        model: Summarizer Backbone (T5 model) for training
        train_data: Input dataset for training

    Returns:
        None

    """

    finetuned_model = train(model, train_data)

    # Saving the Base finetuned model
    model_name = config.BASE_FINETUNED_MODEL
    tools.save_model(model_name, finetuned_model.model, finetuned_model.tokenizer)

    return None


def evaluate(
    model: SummarizerBackbone,
    x_test: Union[DataLoader, List],
    y_test: Union[DataLoader, List],
) -> Dict[str, float]:
    """
    Wrapper function for model evaluation

    Parameters:
        model: Model used for evaluation
        x_test: Evaluation data
        y_test: summaries corresponding to evaluation data

    Returns:
        Dictionary of evaluation scores

    """
    machine_summary = model.predict(x_test)  # Getting the prediction
    gold_summary = y_test.copy()
    scores = _evaluate_summary(gold_summary, machine_summary)

    return scores


def _evaluate_summary(gold_summary: List, machine_summary: List) -> Dict[str, float]:
    """
    Getting Rogue Scores for predicted summary and actual summary

    Parameters:
        gold_summary: list of actual summaries
        machine_summary: list of predicted summaries

    Returns:
        Dictionary of evaluation scores
    """

    if len(gold_summary) != len(machine_summary) or len(gold_summary) == 0:
        logging.exception("Length of machine summary is not equal to actual summary...")
        raise IndexError("Length of machine summary is not equal to actual summary.")

    precision_list = []
    recall_list = []
    fmeasure_list = []

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"], use_stemmer=True
    )  # scorer class for calculating rogue scores

    for summ1, summ2 in zip(gold_summary, machine_summary):
        scores = scorer.score(summ1, summ2)
        precision_list.append(scores["rouge1"].precision)
        recall_list.append(scores["rouge1"].recall)
        fmeasure_list.append(scores["rouge1"].fmeasure)

    # Finding Average scores for precision, Recall and f measure
    avg_precision = sum(precision_list) / len(precision_list)
    avg_recall = sum(recall_list) / len(recall_list)
    avg_fmeasure = sum(fmeasure_list) / len(fmeasure_list)
    scores = {
        "Average Precision": avg_precision,
        "Average Recall": avg_recall,
        "Average F Measure": avg_fmeasure,
    }
    logging.info(f"Scores: {scores}")
    return scores
