from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import json

# Load the tokenizers and models
bert_model_name = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
bert_finetuned_model_name = "bert/bertfinetuned_model_tokenizer/best_bertfinetuned_model"
bert_finetuned_tokenizer_name = "bert/bertfinetuned_model_tokenizer/best_best_bertfinetuned_tokenizer"
roberta_model_name = "roberta/roberta_model_tokenizer/best_roberta_model"
roberta_tokenizer_name = "roberta/roberta_model_tokenizer/best_roberta_tokenizer"
xlnet_model_name = "xlnet/best_xlnet_model"
xlnet_tokenizer_name = "xlnet/best_xlnet_tokenizer"
albert_model_name = "albert/best_albert_model"
albert_tokenizer_name = "albert/best_albert_tokenizer"
distilbert_model_name = "distilbert/best_distilbert_model"
distilbert_tokenizer_name = "distilbert/best_distilbert_tokenizer"

model_paths = [
    # bert_model_name,
    bert_finetuned_model_name,
    roberta_model_name,
    xlnet_model_name,
    albert_model_name,
    distilbert_model_name,
]
tokenizer_paths = [
    # bert_model_name,
    bert_finetuned_model_name,
    roberta_tokenizer_name,
    xlnet_tokenizer_name,
    albert_tokenizer_name,
    distilbert_tokenizer_name,
]

if __name__ == "__main__":

    models = [AutoModelForQuestionAnswering.from_pretrained(path) for path in model_paths]
    tokenizers = [AutoTokenizer.from_pretrained(path) for path in tokenizer_paths]

    # Prepare the pipeline for question answering for each model
    pipelines = [
        pipeline("question-answering", model=model, tokenizer=tokenizer, device=0)
        for model, tokenizer in zip(models, tokenizers)
    ]


    # Define a function to get top 3 answer spans with scores from each model
    def get_top_spans(pipeline, context, question, top_k=3):
        results = pipeline(question=question, context=context, top_k=top_k)
        return [(result["answer"], result["score"]) for result in results]


    # Implement weighted averaging of confidence scores
    def weighted_answer_aggregation(context, question):
        # Dictionary to store answers and their cumulative scores
        answer_scores = {}
        answer_counts = {}

        # Get top 3 spans from each model
        for pipe in pipelines:
            top_spans = get_top_spans(pipe, context, question)

            # Aggregate scores for each unique answer span
            for answer, score in top_spans:
                if answer in answer_scores:
                    answer_scores[answer] += score
                    answer_counts[answer] += 1
                else:
                    answer_scores[answer] = score  # Initialize score for a new answer
                    answer_counts[answer] = 1

        # Compute average score for each answer
        averaged_scores = {
            answer: answer_scores[answer] / answer_counts[answer]
            for answer in answer_scores
        }

        # Select the answer with the highest averaged score
        best_answer = max(averaged_scores, key=lambda k: averaged_scores[k])
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
    for idx, (context, question, qid) in enumerate(
        zip(val_data["context"], val_data["question"], val_data["id"])
    ):
        best_answer, best_score = weighted_answer_aggregation(
            context=context, question=question
        )
        predictions[qid] = best_answer

        if idx % 50 == 0:
            print(f"{idx} done")


    # Save predictions to a JSON file in the required format
    with open("predictions_ensemble.json", "w") as outfile:
        json.dump(predictions, outfile, indent=4)
