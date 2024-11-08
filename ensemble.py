from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Load the tokenizers and models
bert_model_name = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
bert_finetuned_model_name = "Ashaduzzaman/bert-finetuned-squad"
roberta_model_name = "deepset/roberta-large-squad2"

# Load the tokenizers
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_finetuned_tokenizer = AutoTokenizer.from_pretrained(bert_finetuned_model_name)
roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)

# Load the models
bert_model = AutoModelForQuestionAnswering.from_pretrained(bert_model_name)
bert_finetuned_model = AutoModelForQuestionAnswering.from_pretrained(bert_finetuned_model_name)
roberta_model = AutoModelForQuestionAnswering.from_pretrained(roberta_model_name)

from datasets import load_dataset

# Load the SQuAD dataset
dataset = load_dataset("squad")

# Preprocessing function
def preprocess_function(examples, tokenizer):
    # Tokenize the input
    return tokenizer(examples["question"], examples["context"], truncation=True, padding=True, max_length=512)

# Preprocess the dataset for each model's tokenizer
bert_train = dataset["train"].map(lambda x: preprocess_function(x, bert_tokenizer), batched=True)
bert_finetuned_train = dataset["train"].map(lambda x: preprocess_function(x, bert_finetuned_tokenizer), batched=True)
roberta_train = dataset["train"].map(lambda x: preprocess_function(x, roberta_tokenizer), batched=True)

import torch

def get_predictions(model, tokenizer, dataset, batch_size=8):
    model.eval()
    predictions = []
    for batch in torch.utils.data.DataLoader(dataset, batch_size=batch_size):
        inputs = {key: val.to(model.device) for key, val in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        start_indexes = torch.argmax(start_scores, dim=-1)
        end_indexes = torch.argmax(end_scores, dim=-1)
        
        # Convert token indices to text
        for i in range(len(start_indexes)):
            start_idx = start_indexes[i].item()
            end_idx = end_indexes[i].item()
            answer_tokens = batch['input_ids'][i][start_idx:end_idx+1]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            predictions.append(answer)
    
    return predictions


# Get predictions from each model
bert_predictions = get_predictions(bert_model, bert_tokenizer, bert_train)
bert_finetuned_predictions = get_predictions(bert_finetuned_model, bert_finetuned_tokenizer, bert_finetuned_train)
roberta_predictions = get_predictions(roberta_model, roberta_tokenizer, roberta_train)

from collections import Counter

# Simple majority voting for answer selection
def ensemble_predictions(*model_predictions):
    final_predictions = []
    for answers in zip(*model_predictions):
        # Get the most common answer
        common_answer = Counter(answers).most_common(1)[0][0]
        final_predictions.append(common_answer)
    return final_predictions

# Get the final ensemble predictions
ensemble_preds = ensemble_predictions(bert_predictions, bert_finetuned_predictions, roberta_predictions)

from datasets import load_metric

# Load the metric
metric = load_metric("squad")

# Evaluate the ensemble predictions
results = metric.compute(predictions=ensemble_preds, references=dataset["validation"]["answers"])

# Print the evaluation results
print(results)
