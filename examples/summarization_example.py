from jsonschema import validate
from torch.utils.data.dataloader import DataLoader

from Recap import config
from Recap.constants import ServingKeys
from Recap.dataset import SummarizerDataset
from Recap.engine import SummarizerEngine
from Recap.model import SummarizerBackbone
from Recap.schema import input_schema
from Recap.tools import load_json

model_name = config.BASE_FINETUNED_MODEL

# Serve mode
json_data = load_json(path="examples/assets/example.json")

validate(json_data, schema=input_schema)

# Initializing the dataset
backbone_model = SummarizerBackbone(model_name)

dataset = SummarizerDataset(backbone_model, json_data=json_data, mode=ServingKeys.SERVE.value)

# Initialize the Data-loader.
data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# Run the Sentiment Engine.
model_engine = SummarizerEngine(
    model=backbone_model,
    dataset=dataset,
    data_loader=data_loader,
)

response = model_engine.serve(save_result=False)
