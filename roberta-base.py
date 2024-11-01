import json
import logging
import tensorflow as tf
from transformers import TFBertForQuestionAnswering, BertTokenizer
from transformers import create_optimizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load SQuAD training dataset
logger.info("Loading the SQuAD training dataset...")
with open('train-v1.1.json') as f:
    train_data = json.load(f)
logger.info("Training dataset loaded successfully.")

# Prepare training dataset
def prepare_data(data):
    logger.info("Preparing data for training...")
    questions = []
    contexts = []
    start_positions = []
    end_positions = []
    
    for item in data['data']:
        for paragraph in item['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                questions.append(qa['question'])
                contexts.append(context)
                if qa['answers']:
                    start = qa['answers'][0]['answer_start']
                    text = qa['answers'][0]['text']
                    start_positions.append(start)
                    end_positions.append(start + len(text) - 1)
                else:
                    start_positions.append(0)  # Default value if no answer
                    end_positions.append(0)    # Default value if no answer

    logger.info(f"Data preparation complete. Total questions: {len(questions)}")
    return questions, contexts, start_positions, end_positions

# Prepare the training data
questions, contexts, start_positions, end_positions = prepare_data(train_data)

# Tokenization
logger.info("Initializing the tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
logger.info("Tokenizer initialized successfully.")

# Tokenizing the input for training
logger.info("Tokenizing the input data...")
inputs = tokenizer(questions, contexts, truncation=True, padding=True, return_tensors='tf')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
logger.info("Input data tokenized successfully.")

# Create the training dataset
logger.info("Creating the training dataset...")
train_dataset = tf.data.Dataset.from_tensor_slices(
    {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
).map(lambda x: (x, (start_positions, end_positions)))  # Map the labels separately

train_dataset = train_dataset.shuffle(len(questions)).batch(16)
logger.info("Training dataset created successfully.")

# Load the model
logger.info("Loading the BERT model for question answering...")
model = TFBertForQuestionAnswering.from_pretrained('bert-base-uncased')
logger.info("Model loaded successfully.")

# Use mixed precision if you have a compatible GPU
if tf.config.list_physical_devices('GPU'):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    logger.info("Mixed precision enabled for GPU.")

# Optimizer and compile the model
num_train_steps = len(questions) // 16 * 3  # Total steps for 3 epochs
logger.info("Creating optimizer and compiling the model...")
optimizer, schedule = create_optimizer(init_lr=3e-5, num_warmup_steps=0, num_train_steps=num_train_steps)
model.compile(optimizer=optimizer, loss=[SparseCategoricalCrossentropy(from_logits=True)])
logger.info("Model compiled successfully.")

# Training the model
try:
    logger.info("Starting training...")
    model.fit(train_dataset, epochs=3)
    logger.info("Training completed successfully.")
except Exception as e:
    logger.error(f"Error during training: {e}")

# Save the model
logger.info("Saving the fine-tuned model...")
model.save_pretrained("fine-tuned-bert")
tokenizer.save_pretrained("fine-tuned-bert")
logger.info("Model saved successfully.")

# Load SQuAD development dataset for evaluation
logger.info("Loading the SQuAD development dataset...")
with open('dev-v1.1.json') as f:
    dev_data = json.load(f)
logger.info("Development dataset loaded successfully.")

# Prepare evaluation data
dev_questions, dev_contexts, dev_start_positions, dev_end_positions = prepare_data(dev_data)

# Tokenizing the input for evaluation
logger.info("Tokenizing the evaluation input data...")
dev_inputs = tokenizer(dev_questions, dev_contexts, truncation=True, padding=True, return_tensors='tf')
dev_input_ids = dev_inputs['input_ids']
dev_attention_mask = dev_inputs['attention_mask']
logger.info("Evaluation input data tokenized successfully.")

# Create the evaluation dataset
logger.info("Creating the evaluation dataset...")
dev_dataset = tf.data.Dataset.from_tensor_slices(
    {
        'input_ids': dev_input_ids,
        'attention_mask': dev_attention_mask
    }
).batch(16)
logger.info("Evaluation dataset created successfully.")

# Evaluate the model
logger.info("Starting evaluation on the development dataset...")
predictions = model.predict(dev_dataset)
logger.info("Evaluation completed successfully.")

# Extract start and end logits
start_logits = predictions.start_logits
end_logits = predictions.end_logits

# Process predictions
def extract_answers(questions, contexts, start_logits, end_logits):
    logger.info("Extracting answers from logits...")
    predicted_answers = []
    
    for i in range(len(questions)):
        start_index = tf.argmax(start_logits[i]).numpy()
        end_index = tf.argmax(end_logits[i]).numpy()
        
        if start_index <= end_index < len(contexts[i]):
            answer = contexts[i][start_index:end_index + 1]
        else:
            answer = ""
        
        predicted_answers.append(answer)
    
    logger.info("Answers extracted successfully.")
    return predicted_answers

# Get predicted answers
predicted_answers = extract_answers(dev_questions, dev_contexts, start_logits, end_logits)

# Save predicted answers to a file (optional)
with open('predicted_answers.json', 'w') as f:
    json.dump(predicted_answers, f)
logger.info("Predicted answers saved successfully.")