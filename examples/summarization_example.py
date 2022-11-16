from torch.utils.data.dataloader import DataLoader

from Recap import config
from Recap.constants import ServingKeys
from Recap.dataset import SummarizerDataset
from Recap.engine import SummarizerEngine
from Recap.model import SummarizerBackbone
from Recap.tools import load_json

"""
Passing JSON to model
"""

model_name = config.BASE_FINETUNED_MODEL

# Serve mode
json_data = load_json(path="examples/assets/example.json")

# Initializing the dataset
backbone_model = SummarizerBackbone(model_name)

dataset = SummarizerDataset(
    backbone_model, json_data=json_data, mode=ServingKeys.SERVE.value
)

# Initialize the Data-loader.
data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# Run the Sentiment Engine.
model_engine = SummarizerEngine(
    model=backbone_model,
    dataset=dataset,
    data_loader=data_loader,
)

response = model_engine.serve(save_result=False)


"""
Passing a list of texts to model
"""

model_name = config.BASE_FINETUNED_MODEL


# Initializing the dataset
backbone_model = SummarizerBackbone(model_name)

text_list = [
    "The black mamba is a species of highly venomous snake belonging to the family Elapidae. It is native to parts of sub-Saharan Africa. First formally described by Albert Günther in 1864, it is the second-longest venomous snake after the king cobra; mature specimens generally exceed 2 m and commonly grow to 3 m.",
    "A volcano is an opening or rupture in the earth’s surface that allows magma (hot liquid and semi-liquid rock), volcanic ash and gases to escape. They are generally found where tectonic plates come together or separate but they can also occur in the middle of plates due to volcanic hotspots. A volcanic eruption is when lava and gas are released from a volcano—sometimes explosively. The most dangerous type of eruption is called a 'glowing avalanche' which is when freshly erupted magma flows down the sides of a volcano.",
]

dataset = SummarizerDataset(
    backbone_model, text_list=text_list, mode=ServingKeys.SERVE.value
)

# Initialize the Data-loader.
data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# Run the Sentiment Engine.
model_engine = SummarizerEngine(
    model=backbone_model,
    dataset=dataset,
    data_loader=data_loader,
)

response = model_engine.serve(save_result=False)
