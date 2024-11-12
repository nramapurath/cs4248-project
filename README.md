
# BERT Question Answering Model Training and Evaluation

This project demonstrates how to fine-tune a BERT model on the SQuAD dataset for question answering tasks. It includes two main scripts:
- `base-bert.py`: Prepares the dataset, performs hyperparameter tuning, and trains the BERT model using Hugging Face's `Trainer` API.
- `evalprep.py`: Loads the trained model and tokenizer for evaluating and generating predictions on new data.

## Table of Contents

- [Setup](#setup)
- [Running the code](#Running-the-code)
- [Data Preparation with `base-bert.py`](#data-preparation-with-base-bertpy)
- [Training and Hyperparameter Tuning with `base-bert.py`](#training-and-hyperparameter-tuning-with-base-bertpy)
- [Evaluating the Model with `evalprep.py`](#evaluating-the-model-with-evalpreppy)
- [Generating Predictions with `evalprep.py`](#generating-predictions-with-evalpreppy)
- [Saving and Loading the Model](#saving-and-loading-the-model)

## Setup

1. Clone the repository and navigate to the project directory:

    ```bash
    git clone https://github.com/nramapurath/cs4248-project.git
    cd cs4248-project
    ```

2. Install the required dependencies:

    ```bash
    pip install torch transformers datasets optuna
    ```
## Running the code
1. Run the following command in the present working directory to train base bert on train data set.

    ```bash
    python base-bert.py
    ```
2. Run the following command to get the output from apply the model on the dev data set.
    ```bash
    python evalprep.py
    ```
3. Finally run the following command to get the evaluation metrics.
    ```bash
    python evaluate-v2.0.py dev-v1.1.json bertfinetuned-predictions.json -o eval_results.json -v
    ```

## Data Preparation with `base-bert.py`

1. **Loading and Formatting Data**: The `bert.py` script loads the SQuAD JSON files using the `load_squad_data` function, extracting contexts, questions, and answers to prepare for fine-tuning BERT.

    ```python
    train_data = load_squad_data("train-v1.1.json")
    val_data = load_squad_data("dev-v1.1.json")
    ```

2. **Preprocessing**: The data is tokenized and processed to prepare for training by converting character positions of answers into token positions.

    ```python
    train_dataset = train_dataset.map(preprocess, remove_columns=["context", "question", "answers"])
    val_dataset = val_dataset.map(preprocess, remove_columns=["context", "question", "answers"])
    ```

## Training and Hyperparameter Tuning with `base-bert.py`

1. **Defining the Training Function**: The script uses an `optuna` study to search for the best hyperparameters (`learning_rate`, `batch_size`, `num_train_epochs`) for fine-tuning BERT. 

2. **Running the Training**: After setting up `TrainingArguments`, `Trainer` is used to fine-tune the BERT model on the SQuAD dataset.

    ```python
    study = optuna.create_study(direction="minimize")
    study.optimize(model_training, n_trials=3)
    ```

3. **Best Hyperparameters**: Once the study completes, the best hyperparameters are displayed, allowing you to adjust your training setup for future experiments.

## Evaluating the Model with `evalprep.py`

1. **Loading the Model**: After training in `bert.py`, the best model and tokenizer are saved and can be reloaded in `evalprep.py` for generating predictions.

    ```python
    model = BertForQuestionAnswering.from_pretrained("best_model")
    tokenizer = BertTokenizerFast.from_pretrained("best_tokenizer")
    ```

2. **Preparing the Validation Data**: The `evalprep.py` script loads the validation dataset (`dev-v1.1.json`) for generating predictions.

## Generating Predictions with `evalprep.py`

1. **Running Inference**: Using the Hugging Face `pipeline` API, the model predicts answers for each question in the validation dataset. The results are stored in a JSON file.

    ```python
    with open("predictions.json", "w") as outfile:
        json.dump(predictions, outfile, indent=4)
    ```

## Saving and Loading the Model

To reuse the model, save it after training:

```python
model.save_pretrained("best_model")
tokenizer.save_pretrained("best_tokenizer")
```

Then load the model and tokenizer as needed for further evaluations or predictions.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Hugging Face Transformers Library](https://github.com/huggingface/transformers)
- [Optuna Hyperparameter Optimization Framework](https://github.com/optuna/optuna)
- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
