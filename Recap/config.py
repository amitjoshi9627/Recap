import os
from datetime import datetime

# File Directories
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)  # Base Directory of Classifier Module
INPUT_DIR = os.path.join(BASE_DIR, "local_assets")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
OUTPUT_LOG = os.path.join(OUTPUT_DIR, "run_logs")  # Output directory for saving Logs
OUTPUT_RESULTS = os.path.join(
    OUTPUT_DIR, "test_outputs"
)  # Output directory for saving output JSONs

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
if not os.path.exists(OUTPUT_LOG):
    os.mkdir(OUTPUT_LOG)
if not os.path.exists(OUTPUT_RESULTS):
    os.mkdir(OUTPUT_RESULTS)

TEST_JSON = "test.json"  # For batch Json put "batch_test.json" here
OUT_JSON = "output.json"
LOG_FILE = datetime.now().strftime("log_%H-%M-%d-%m-%Y.log")

# File and column names
TRAINING_DATA = "3rd_person_pov_train.csv"
summary_col = "summary"
text_col = "transcript"
response_column = "response"
chunk_id = "chunk_id"

SUMMARIZATION_MODEL = "t5-small"  # pretrained T5 model name
FINETUNED_MODEL = "T5 Test Model"  # Model name for functional testing
BASE_FINETUNED_MODEL = "Fine Tuned T5"  # Base fine tuned model
PARAPHRASING_MODEL = "tuner007/pegasus_paraphrase"  # Pegasus paraphrasing model name

# Model configuration
BATCH_SIZE = 4
INPUT_LENGTH = 512  # Max Input text length
OUTPUT_LENGTH = 150  # Max output summary length
NUM_SENTENCES = 1  # Number of sentences to be returned by Pegasus Paraphraser Model
NUM_BEAMS = 4
TEMPERATURE = 1.5
NUM_EPOCHS = 10  # Number of Epochs
LEARNING_RATE = 2e-5
# SUMM_THRESHOLD = 0  # Number of characters less than which the input text is itself the Summary
DEVICE = "cuda"  # cuda or cpu

NUM_LOGGING = 20  # Number of Logging information in training iterations
