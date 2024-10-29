# cs4248-project

Exploring finetuned models for extractive question answering using SQuAD

## Setup

The following setup guide follows and elaborates on steps listed [here](https://huggingface.co/docs/transformers/installation)

Assuming clean setup with python, pip and git already installed on your machine:

### 1. Clone the repo

### 2. Set up virtual environment in python

`python -m venv .env`

### 3. Activate virtual environment

To install and use packages, we will need to activate the virtual environment as follows:

- For Windows:
  `.env/Scripts/activate`
  (NOTE: remember to prefix with `source ` if running the commmand from bash or other unix-like terminal)

- For Linux and MacOS:
  `source .env/bin/activate`

If you wish to exit the venv, just enter `deactivate`.

### 4. Install necessary packages:

- Install numpy: `pip install numpy`

- Install pytorch (find your appropriate installation version and index-url [here](https://pytorch.org/get-started/locally/) ): e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu118`

- Install hugging-face transformers library: `pip install transformers`

### 5. Bringing it all together

If all goes well, the following command should download a pretrained model, perform sentiment analysis, and print out the sentiment label and score:
`python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"`
Output:
`[{'label': 'POSITIVE', 'score': 0.9998704195022583}]`

## Running the project:

idk
