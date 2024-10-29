'''
Python file to demo reading dataset, first context and first question and obtaining the result from roberta-base
'''

from transformers import pipeline
import json
import torch

model = "deepset/roberta-base-squad2"
device = "cuda" if torch.cuda.is_available() else "cpu"

roberta_base = pipeline(model=model, device=device)

with open("dev-v1.1.json", "r") as read_file:
    dataset = json.load(read_file)['data']

example = dataset[0]['paragraphs'][0]
context = example['context']
question = example['qas'][0]['question']

res = roberta_base(question=question, context=context)

print("True answer: " + str(example['qas'][0]['answers'][0]))
print("Model output: " + str(res))
