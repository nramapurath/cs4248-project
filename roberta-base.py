
# Import libraries
from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset
import json
import torch

# Load JSON data
def load_squad_data(file_path):
    with open(file_path, "r") as file:
        squad_data = json.load(file)["data"]
        
    contexts = []
    questions = []
    answers = []
    for group in squad_data:
        for paragraph in group["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                if qa["answers"]:
                    answer = qa["answers"][0]
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    return {"context": contexts, "question": questions, "answers": answers}

# Load your training and validation data
train_data = load_squad_data("train-v1.1.json")
val_data = load_squad_data("dev-v1.1.json")

# Convert to Hugging Face dataset format
train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)

# Load BERT tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Preprocess data
def preprocess(example):
    inputs = tokenizer(
        example["question"],
        example["context"],
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True
    )
    
    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []
    
    # Process single answer start and text
    start_char = example["answers"]["answer_start"]
    end_char = start_char + len(example["answers"]["text"])
    
    # Find token positions for the answer span
    token_start_index = 0
    token_end_index = len(offset_mapping) - 1

    while token_start_index < len(offset_mapping) and offset_mapping[token_start_index][0] <= start_char:
        token_start_index += 1
    while token_end_index >= 0 and offset_mapping[token_end_index][1] >= end_char:
        token_end_index -= 1

    start_positions.append(token_start_index - 1)
    end_positions.append(token_end_index + 1)

    inputs["start_positions"] = torch.tensor(start_positions)
    inputs["end_positions"] = torch.tensor(end_positions)
    return inputs


# Apply preprocessing to the datasets
train_dataset = train_dataset.map(preprocess, remove_columns=["context", "question", "answers"])
val_dataset = val_dataset.map(preprocess, remove_columns=["context", "question", "answers"])

# Set up the Trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Evaluation results:", results)
