from transformers import BertTokenizerFast, BertForQuestionAnswering, pipeline
import json

# Load the best trained model and tokenizer
model = BertForQuestionAnswering.from_pretrained("best_model")
tokenizer = BertTokenizerFast.from_pretrained("best_tokenizer")

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
val_data = load_squad_data("dev-v1.1.json")

# Prepare the pipeline for question answering
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Apply the model on the dev dataset and store predictions
predictions = {}
predictions_full = {}
for idx, (context, question, qid) in enumerate(zip(val_data["context"], val_data["question"], val_data["id"])):
    answers = qa_pipeline({"question": question, "context": context}, top_k=3)
    predictions[qid] = answers[0]["answer"]  # Store the answer using question ID as key
    predictions_full[qid] = answers

    if idx % 50 == 0:
        print(f"{idx} predicted")

# Save predictions to a JSON file in the required format
with open("bertfinetuned-predictions.json", "w") as outfile:
    json.dump(predictions, outfile, indent=4)

with open("bertfinetuned-predictions_full.json", "w") as outfile:
    json.dump(predictions_full, outfile, indent=4)
print("Predictions saved to predictions.json and predictions_full.json")