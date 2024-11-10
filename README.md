# Question Generation Model with T5

This repository contains a machine learning model for Question Generation using the T5 (Text-to-Text Transfer Transformer) architecture. The model is fine-tuned on a custom dataset to generate questions based on a given context and answer.

## Project Overview

The model takes a context and a corresponding answer as input and generates a question related to that answer. This can be useful for educational tools, chatbots, and various NLP applications where question-answering is important.

### Key Features:
- Fine-tuning T5 for question generation.
- Supports training and inference with a custom dataset.
- Trained model can generate multiple question variations.
- Integrated with Gradio for a simple user interface for deployment.

## Installation

1. **Clone this repository:**
    ```bash
    git clone https://github.com/yourusername/question-generation-t5.git
    cd question-generation-t5
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

The `requirements.txt` file includes libraries like:
- `transformers`
- `torch`
- `pytorch-lightning`
- `gradio`
- and other necessary libraries for training and inference.

## Dataset

This model was fine-tuned using the SQuAD (Stanford Question Answering Dataset), which is a popular dataset used for question-answering tasks.

- **Training Dataset:** `train_squad.parquet`
- **Validation Dataset:** `validation_squad.parquet`

Ensure that your dataset is in the same format for seamless integration.

## Usage

### 1. Fine-tuning the Model

To train the model on your own dataset, run the following script:

```bash
python train_model.py
```
### 2. Making Predictions (Inference)
Once the model is trained, you can generate questions using the trained model. Here's how you can do that:

```bash
from transformers import T5ForConditionalGeneration, T5Tokenizer

###### Load the trained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5_trained_model')
tokenizer = T5Tokenizer.from_pretrained('t5_tokenizer')

###### Example input text
context = "President Donald Trump said and predicted that some states would reopen this month."
answer = "Donald Trump"
text = f"context: {context} answer: {answer}"

###### Encode the input text
encoding = tokenizer.encode_plus(text, max_length=512, padding='max_length', truncation=True, return_tensors="pt")

###### Generate the question
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]
beam_outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=72, num_beams=5, num_return_sequences=2)

#Decode the generated question(s)
for beam_output in beam_outputs:
    question = tokenizer.decode(beam_output, skip_special_tokens=True)
    print(question)
```
### 3. Gradio Interface
You can also use Gradio for a simple web-based interface to interact with the model:

```bash
import gradio as gr

def generate_question(context, answer):
    text = f"context: {context} answer: {answer}"
    encoding = tokenizer.encode_plus(text, max_length=512, padding='max_length', truncation=True, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    beam_outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=72, num_beams=5, num_return_sequences=2)
    
    return [tokenizer.decode(beam_output, skip_special_tokens=True) for beam_output in beam_outputs]

iface = gr.Interface(fn=generate_question, inputs=["text", "text"], outputs="text")
iface.launch()
```
### 4. Deploying the Model
The model and tokenizer can be deployed using platforms like Hugging Face Spaces or any cloud service (AWS, GCP, etc.) that supports deploying machine learning models.

### File Structure
```bash
├── train_model.py           # Script for fine-tuning the T5 model
├── inference.py             # Script for making predictions
├── gradio_demo.py           # Gradio interface for demo
├── t5_tokenizer/            # Directory for the saved tokenizer
├── t5_trained_model/        # Directory for the saved model
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
```
## Acknowledgements
The model is based on the T5 architecture from Hugging Face.
The dataset used for fine-tuning is the SQuAD dataset.
