from transformers import (
    AutoTokenizer,
    RobertaForQuestionAnswering,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import json
import torch
import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_intermediate_values,
    plot_param_importances,
)
from matplotlib import pyplot as plt
import os
import dotenv

dotenv.load_dotenv()


def plot_files(study: optuna.Study):
    dirname = "plots"
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    filename_prefix = "roberta_plot"

    # Optimization history
    opt_history_filename = f"{filename_prefix}_opt_history.png"
    filepath_opt_history = os.path.join(dirname, opt_history_filename)
    opt_hist_ax = plot_optimization_history(study)
    plt.savefig(filepath_opt_history)

    # Intermediate values
    intermediate_values_filename = f"{filename_prefix}_intermediate_values.png"
    filepath_intermediate_values = os.path.join(dirname, intermediate_values_filename)
    intermediate_values_ax = plot_intermediate_values(study)
    plt.savefig(filepath_intermediate_values)

    # Hyperparameter importances
    param_importances_filename = f"{filename_prefix}_param_importances.png"
    filepath_param_importances = os.path.join(dirname, param_importances_filename)
    param_importances_ax = plot_param_importances(
        study,
        target=lambda t: t.duration.total_seconds() if t.duration else 0,
        target_name="duration",
    )
    plt.savefig(filepath_param_importances)


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
                question = qa["question"].lstrip()
                if qa["answers"]:
                    answer = qa["answers"][0]
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    return {"context": contexts, "question": questions, "answers": answers}


# Preprocess data
def preprocess(example):
    inputs = tokenizer(
        example["question"],
        example["context"],
        max_length=384,
        truncation="only_second" if len(example["context"]) > 384 else False,
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


# Hyperparameter tuning function using Optuna
def model_training(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
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
        dataloader_num_workers=2,
        fp16=True,  # disabled as it causes error on my machine, it was set to true originally.
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
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


if __name__ == "__main__":
    torch.device("cuda")
    # Convert data to Hugging Face Dataset format
    train_data = load_squad_data("../data/train-v1.1.json")
    val_data = load_squad_data("../data/dev-v1.1.json")
    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)

    # Load BERT tokenizer and model
    model_name = "FacebookAI/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = RobertaForQuestionAnswering.from_pretrained(model_name)

    # Apply preprocessing directly to datasets
    train_dataset = train_dataset.map(
        preprocess, remove_columns=["context", "question", "answers"]
    )
    val_dataset = val_dataset.map(
        preprocess, remove_columns=["context", "question", "answers"]
    )

    # Run hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(model_training, n_trials=3)

    op_history_plot = plot_optimization_history(study)
    optuna.visualization.plot_param_importances(
        study, target=lambda t: t.duration.total_seconds(), target_name="duration"
    )

    # Best hyperparameters
    print("Best hyperparameters found:", study.best_params)
    # Plot training graphs
    plot_files(study)

    # Save the best model and tokenizer
    model.save_pretrained("best_roberta_model")
    tokenizer.save_pretrained("best_roberta_tokenizer")

    # Load and apply the model later
    model = RobertaForQuestionAnswering.from_pretrained("best_roberta_model")
    tokenizer = AutoTokenizer.from_pretrained("best_roberta_tokenizer")
