# Hi, I'm Recap! (A Summarizer) 👋

I am a module known for summarizing a given sentence/ sentences of texts and I am cooool! 🧊

**What really is a Summarization ?** : `Text summarization in NLP is the process of summarizing (or shortening long texts) the information in large texts for quicker consumption.`

## Here is an example 👇

**Original**
So how do you go about identifying your strengths and weaknesses, and analyzing the opportunities and threats that flow from them? SWOT Analysis is a useful technique that helps you to do this.

What makes SWOT especially powerful is that, with a little thought, it can help you to uncover opportunities that you would not otherwise have spotted. And by understanding your weaknesses, you can manage and eliminate threats that might otherwise hurt your ability to move forward in your role.

If you look at yourself using the SWOT framework, you can start to separate yourself from your peers, and further develop the specialized talents and abilities that you need in order to advance your career and to help you achieve your personal goals.

**Summary**
SWOT Analysis is a technique that helps you identify strengths, weakness, opportunities, and threats. Understanding and managing these factors helps you to develop the abilities you need to achieve your goals and progress in your career.

## How to use me ? 💁

```javascript
from Recap import model_serve

response = model_serve(test_input)
```

## Want to test me ? 🧐

No issues! check me, run the following command. Let me give you some dependency issues

```
python -m Recap.main --mode=package_test --func_test=all --file_path=path/to/file.json [optional]
```

**Here are the the different options for package test**:

- _func_test_ = "all" -> runs test for all the components.
- _func_test_ = "train" -> runs the test for training component only.
- _func_test_ = "serve" -> runs the test for serving component only.
- _func_test_ = "eval" -> runs the test for evaluation component only.

#### Here is my Pluggable Component ⚓

This mode of running the package, showcase the capability to be able to plug in the train, eval and serving
component of this module into API integration or MLOPs engine.

- _mode_="train" - run the training component of the package.
- _mode_="serve" - run the serving component of the package.
- _mode_="eval" - run the evaluation component of the package.

`python -m Recap.main --mode=serve`

### Do the Parameters Setup

**Network Hyperparameters**

- **DEVICE** - "cpu" or "gpu", depending upon the hardware supported by your local machine / your choice
- **BATCH_SIZE:** Number of samples in a single batch.
- **NUM_EPOCHS:** Number of epochs for training the model.
- **LEARNING_RATE:** Tuning parameter in the optimization algorithm that determines the step size at each iteration.
- **INPUT_LENGTH:** Max length of the input texts
- **OUTPUT_LENGTH:** Max length of the output summaries

**SUMM_THRESHOLD:** Length threshold for generating summary or not.

## Something related to installation 🔨

Dude! Install the dependencies.

```sh
pip install -r requirements.txt
```

## How to fuel me ? ⛽

Provide me with some data for training inside dataset/ folder and don't forget to change the _config.py_ for the location of the file.

## Train me How? 🚅

**You have no idea. Really? Why don't I give you a hint.**

```sh
python -m Recap.main --mode=train
```

That was more than a hint. But it's okay

## QNA Time!!! 📣

**Q1. What kind of model is used in this package ?**
**Ans.** [T5 model](https://huggingface.co/transformers/model_doc/t5.html#t5forconditionalgeneration) (Text-To-Text Transfer Transformer) is used as a base model by me.

**Q2. What do you mean by base model ? Are you not using it directly ?**
**Ans.** Good question! I am using the T5 model as a base model, so that I can be fine tune it according to specific data. That way I get to know some basic terminologies from your data and I can create my summaries catering to the need of your domain.

**Q3. What is this T5 model ?**
**Ans.** [T5](https://huggingface.co/transformers/model_doc/t5.html) (Text-To-Text Transfer Transformer) is an encoder-decoder model and converts all NLP problems into a text-to-text format, which means the input and output are both text strings. T5 is an extremely large new neural network model that is trained on a mixture of unlabeled text (the authors’ huge new C4 collection of English web text) and labeled data from popular natural language processing tasks, then fine-tuned individually for each of the tasks that they authors aim to solve. It works quite well, setting the state of the art on many of the most prominent text classification tasks for English, and several additional question-answering and summarization tasks as well. Isn't it interesting ??

**Q4. What are the different kind of format supported for input data ?**
**Ans.** Currently I support JSON and list format as an input for the API.

**Q5. What if I want to see the logs of what is happening ?**
**Ans.** That too has been taken care of. Whenever the process is completed, you can go to results/run_logs/ to see the logs.

**Q6. Are you a human being ?**
**Ans.** So a human being is a person (man, woman, child) from the _homo sapiens_ species.. Wait. What ?? 😕

## My Roadmap✈️

- Training on 3rd person POV summaries
- Adding the retraining part

## This Guy made me 🦸‍♂️

- Amit Joshi _aka_ A.J
