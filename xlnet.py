from transformers import (
    XLNetTokenizerFast,
    XLNetForQuestionAnsweringSimple,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import json
import torch
import optuna
import dotenv

dotenv.load_dotenv()
MODEL_NAME = "xlnet/xlnet-base-cased"
TOKENIZER_NAME = "xlnet/xlnet-base-cased"
TRAIN_DATASET_NAME = "train-v1.1-liter.json"
VAL_DATASET_NAME = "dev-v1.1.json"
OUTPUT_MODEL_NAME = "best_xlnet_model"
OUTPUT_TOKENIZER_NAME = "best_xlnet_tokenizer"
DEVICE = "cpu"


# Load SQuAD data as JSON
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
                    # TODO: Handle multiple answers
                    answer = qa["answers"][
                        0
                    ]  # Only use the first answer for simplicity
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    return {"context": contexts, "question": questions, "answers": answers}


if __name__ == "__main__":

    device = torch.device(DEVICE)

    # Convert data to Hugging Face Dataset format
    train_data = load_squad_data(TRAIN_DATASET_NAME)
    val_data = load_squad_data(VAL_DATASET_NAME)
    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)

    # Load BERT tokenizer and model
    model_name = MODEL_NAME
    tokenizer: XLNetTokenizerFast = XLNetTokenizerFast.from_pretrained(
        model_name,
    )  # Don't use AutoTokenizer to specify exact tokenizer
    model: PreTrainedModel = XLNetForQuestionAnsweringSimple.from_pretrained(model_name)

    # Preprocess data
    def preprocess(example):
        inputs = tokenizer(
            example["question"],
            example["context"],
            max_length=512,
            truncation="only_second",
            padding="max_length",
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")
        start_char = example["answers"]["answer_start"]
        end_char = start_char + len(example["answers"]["text"])
        token_start_index = 0
        token_end_index = len(offset_mapping) - 1

        for i, (start, end) in enumerate(offset_mapping):
            if start <= start_char < end:
                token_start_index = i
                break

        for i, (start, end) in enumerate(offset_mapping):
            if start < end_char <= end:
                token_end_index = i
                break

        if token_start_index > token_end_index:
            token_start_index, token_end_index = 0, 0

        inputs["start_positions"] = torch.tensor([token_start_index])
        inputs["end_positions"] = torch.tensor([token_end_index])
        return inputs

    # Apply preprocessing directly to datasets
    train_dataset = train_dataset.map(
        preprocess, remove_columns=["context", "question", "answers"]
    )
    val_dataset = val_dataset.map(
        preprocess, remove_columns=["context", "question", "answers"]
    )

    # Hyperparameter tuning function using Optuna
    def model_training(trial: optuna.Trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        batch_size = trial.suggest_categorical("batch_size", [2, 4])
        num_train_epochs = trial.suggest_int("num_train_epochs", 2, 4)

        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            save_total_limit=1,
            save_strategy="epoch",
            dataloader_num_workers=4,
            fp16=False,
            gradient_accumulation_steps=4,
            gradient_checkpointing=False,  # XLNet does not support gradient checkpointing
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()
        eval_results = trainer.evaluate()

        return eval_results["eval_loss"]

    # Run hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(model_training, n_trials=3)

    # Best hyperparameters
    print("Best hyperparameters found:", study.best_params)

    # Save the best model and tokenizer
    model.save_pretrained(OUTPUT_MODEL_NAME)
    tokenizer.save_pretrained(OUTPUT_TOKENIZER_NAME)

    # # Load and apply the model later
    # model = XLNetForQuestionAnswering.from_pretrained("best_xlnet_model")
    # tokenizer = XLNetTokenizerFast.from_pretrained("best_xlnet_tokenizer")
