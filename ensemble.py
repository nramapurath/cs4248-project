from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch
import json

# Load the tokenizers and models
# bert_model_name = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
# bert_finetuned_model_name = "distilbert/distilbert-base-cased-distilled-squad"
# roberta_model_name = "best_model"
# roberta_tokenizer_name = "best_tokenizer"

albert_model_path = "albert/albert_model/best_albert_model"
albert_tokenizer_path = "albert/albert_model/best_albert_tokenizer"

bert_model_path = "bert/bert_model/best_model"
bert_tokenizer_path = "bert/bert_model/best_tokenizer"

distilbert_model_path = "distilbert/distilbert_model/best_distilbert_model"
distilbert_tokenizer_path = "distilbert/distilbert_model/best_distilbert_tokenizer"

roberta_model_path = "roberta/roberta_finetuned/best_roberta_model"
roberta_tokenizer_path = "roberta/roberta_finetuned/best_roberta_tokenizer"

xlnet_model_path = "xlnet/xlnet_model/best_xlnet_model"
xlnet_tokenizer_path = "xlnet/xlnet_model/best_xlnet_tokenizer"


model_paths = [albert_model_path, bert_model_path, distilbert_model_path, roberta_model_path, xlnet_model_path]
tokenizer_paths = [albert_tokenizer_path, bert_tokenizer_path, distilbert_tokenizer_path, roberta_tokenizer_path, xlnet_tokenizer_path]
model_weights = [1, 1, 1, 2, 2]

models = [AutoModelForQuestionAnswering.from_pretrained(path) for path in model_paths]
tokenizers = [AutoTokenizer.from_pretrained(path) for path in tokenizer_paths]

# Prepare the pipeline for question answering for each model
pipelines = [pipeline("question-answering", model=model, tokenizer=tokenizer, batch_size=16, device=0) for model, tokenizer in zip(models, tokenizers)]

# Define a function to get top 3 answer spans with scores from each model
def get_top_spans(pipeline, context, question, top_k=3):
    results = pipeline({"question": question, "context": context}, top_k=top_k)
    return [(result['answer'], result['score']) for result in results]

# Implement weighted averaging of confidence scores
def weighted_answer_aggregation(context, question):
    # Dictionary to store answers and their cumulative scores
    answer_scores = {}
    answer_counts = {}

    # Get top 3 spans from each model
    for i, pipe in enumerate(pipelines):
        top_spans = get_top_spans(pipe, context, question)

        model_weight = model_weights[i]
        
        # Aggregate scores for each unique answer span
        for answer, score in top_spans:
            model_weighted_score = score * model_weight
            if answer in answer_scores:
                answer_scores[answer] += model_weighted_score
                answer_counts[answer] += 1  
            else:
                answer_scores[answer] = model_weighted_score   # Initialize score for a new answer
                answer_counts[answer] = 1

    # Compute average score for each answer
    averaged_scores = {answer: answer_scores[answer] / answer_counts[answer] for answer in answer_scores}

    # Select the answer with the highest averaged score
    best_answer = max(averaged_scores, key=averaged_scores.get)
    best_score = averaged_scores[best_answer]

    return best_answer, best_score

# Load JSON data
def load_squad_data(file_path):
    with open(file_path, "r") as file:
        squad_data = json.load(file)["data"]
        
    contexts = []
    questions = []
    question_ids = []
    for group in squad_data:
        for paragraph in group["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                qid = qa["id"]
                contexts.append(context)
                questions.append(question)
                question_ids.append(qid)
    return {"context": contexts, "question": questions, "id": question_ids}

# Prepare the validation data
val_data = load_squad_data("data/dev-v1.1.json")

# Run the ensemble on each question in dev data
predictions = {}
for idx, (context, question, qid) in enumerate(zip(val_data["context"], val_data["question"], val_data["id"])):
    best_answer, best_score = weighted_answer_aggregation(context=context, question=question)
    predictions[qid] = best_answer

    if idx % 50 == 0:
        print(f"{idx} done")


# Save predictions to a JSON file in the required format
with open("predictions_weighted_ensemble.json", "w") as outfile:
    json.dump(predictions, outfile, indent=4)
