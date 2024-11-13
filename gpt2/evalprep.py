from transformers import GPT2ForQuestionAnswering, GPT2TokenizerFast, pipeline
import json

MODEL_NAME = "best_gpt2_model"
TOKENIZER_NAME = "best_gpt2_tokenizer"
DATASET_NAME = "dev-v1.1.json"

# Load the best trained model and tokenizer from disk (run this after training)
model = GPT2ForQuestionAnswering.from_pretrained(MODEL_NAME)
tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_NAME)


# Load data from the SQuAD dataset
def load_squad_data(file_path) -> dict[str, list[str] | list[dict]]:
    with open(file_path, "r") as file:
        squad_data = json.load(file)["data"]

    contexts = []
    questions = []
    question_ids = []
    model_answers = []
    for group in squad_data:
        for paragraph in group["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                qid = qa["id"]
                curr_model_answers: list[dict] = qa["answers"]

                # Map for multiple answers
                model_answer_map = {}
                # Flatten
                for model_answer in curr_model_answers:
                    new_answer_start = model_answer["answer_start"]
                    new_text = model_answer["text"]
                    new_len = len(new_text)
                    if new_answer_start not in model_answer_map:
                        model_answer_map[new_answer_start] = (new_len, new_text)
                    else:
                        # Compare
                        old_model_answer_len, _old_text = model_answer_map[
                            new_answer_start
                        ]
                        if new_len > old_model_answer_len:
                            model_answer_map[new_answer_start] = (new_len, new_text)
                print("Successfully flattened answers", len(model_answer_map))

                model_answers.extend(
                    [
                        {
                            "answer_start": k,
                            "len": v[0],
                            "text": v[1],
                        }
                        for k, v in model_answer_map.items()
                    ]
                )
                contexts.append(context)
                questions.append(question)
                question_ids.append(qid)

    return {
        "context": contexts,
        "question": questions,
        "id": question_ids,
        "model_answers": model_answers,
    }


if __name__ == "__main__":
    # Prepare the validation data
    val_data = load_squad_data(DATASET_NAME)

    # Prepare the pipeline for question answering
    qa_pipeline = pipeline(
        "question-answering", model=model, tokenizer=tokenizer, device=0
    )

    # Apply the model on the dev dataset and store predictions
    predictions = {}
    predictions_full = {}
    imperfect_matches = []
    for idx, (context, question, qid, model_answers) in enumerate(
        zip(
            val_data["context"],
            val_data["question"],
            val_data["id"],
            val_data["model_answers"],
        )
    ):
        answers = qa_pipeline({"question": question, "context": context}, top_k=5) or []

        # Assert that answers is an iterable
        if not isinstance(answers, list):
            print(
                f"[ERROR] answers is not a list for question {qid}. Answers {answers}"
            )
            break

        # Store answer metrics and see what common errors are made
        for idx, answer in enumerate(answers):
            if not answer:
                continue
            new_answer_start = answer["start"]
            new_answer_end = answer["end"]
            new_len_answer = new_answer_end - new_answer_start
            new_answer_text = answer["answer"]
            for model_answer in model_answers:
                # Compare lengths if same start
                assert type(model_answer) == dict, "Should have model answer as dict"
                if model_answer["answer_start"] == new_answer_start:
                    if model_answer["len"] == new_len_answer:
                        # Correct match
                        if idx != 0:
                            imperfect_matches.append(
                                {
                                    "qid": qid,
                                    "question": question,
                                    "context": context,
                                    "model_answer_text": model_answer["text"],
                                    "answer": new_answer_text,
                                    "reason": "Correct match but not top answer",
                                }
                            )
                        break
                imperfect_matches.append(
                    {
                        "qid": qid,
                        "question": question,
                        "context": context,
                        "model_answer_text": model_answer["text"],
                        "answer": new_answer_text,
                        "reason": "Imperfect match",
                    }
                )

        predictions[qid] = answers[0][
            "answer"
        ]  # Store the answer using question ID as key
        predictions_full[qid] = answers

        if idx % 50 == 0:
            print(f"{idx} predicted")

    # Save predictions to a JSON file in the required format
    with open("predictions.json", "w") as outfile:
        json.dump(predictions, outfile, indent=4)

    with open("predictions_full.json", "w") as outfile:
        json.dump(predictions_full, outfile, indent=4)

    with open("imperfect_matches.json", "w") as outfile:
        json.dump(imperfect_matches, outfile, indent=4)
    print("Predictions saved to predictions.json and predictions_full.json")
    print("Imperfect matches saved to imperfect_matches.json")
