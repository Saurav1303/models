# Install dependencies if not already installed
# pip install transformers datasets torch scikit-learn fastapi uvicorn

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from fastapi import FastAPI
import torch

### 1. Load and Prepare Dataset
# Load the Yelp polarity dataset (or use your custom dataset)
dataset = load_dataset('yelp_polarity')

# Add a neutral class to the dataset (for illustration, here we use a synthetic approach)
# For real-world data, you can have a neutral class in your labeled data
def add_neutral_label(example):
    if example['label'] == 1:
        example['label'] = 2  # Positive -> 2
    elif example['label'] == 0:
        example['label'] = 0  # Negative -> 0
    return example

# Map neutral label to the middle reviews in the dataset (e.g., reviews of average length/score)
dataset = dataset.map(add_neutral_label)

### 2. Tokenization Function
# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Rename label column and format dataset for PyTorch
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

### 3. Load Pre-trained Model (3 labels: Negative, Neutral, Positive)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

### 4. Define Metrics for Evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')  # 'macro' for multi-class
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

### 5. Training Arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    evaluation_strategy="epoch",     # Evaluate model every epoch
    save_strategy="epoch",           # Save model every epoch
    learning_rate=2e-5,              # Learning rate
    per_device_train_batch_size=16,  # Training batch size
    per_device_eval_batch_size=16,   # Evaluation batch size
    num_train_epochs=3,              # Number of epochs
    weight_decay=0.01,               # Weight decay
    logging_dir='./logs',            # Directory for logging
    logging_steps=10,
)

### 6. Initialize Trainer
trainer = Trainer(
    model=model,                         # BERT model
    args=training_args,                  # Training arguments
    train_dataset=tokenized_dataset['train'],  # Training dataset
    eval_dataset=tokenized_dataset['test'],    # Evaluation dataset
    compute_metrics=compute_metrics,          # Evaluation metrics
)

### 7. Train the Model
trainer.train()

### 8. Evaluate the Model
results = trainer.evaluate()
print(f"Evaluation results: {results}")

### 9. Save the Fine-tuned Model and Tokenizer
model.save_pretrained("./fine-tuned-bert-sentiment")
tokenizer.save_pretrained("./fine-tuned-bert-sentiment")

### 10. Deploy with FastAPI

# Create FastAPI application
app = FastAPI()

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained('./fine-tuned-bert-sentiment')
tokenizer = BertTokenizer.from_pretrained('./fine-tuned-bert-sentiment')

# Create the prediction route
@app.post("/predict/")
async def predict_sentiment(review: str):
    inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    sentiment = torch.argmax(outputs.logits, dim=1).item()

    # Map sentiment to human-readable labels
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    return {"sentiment": sentiment_map[sentiment]}

# To run the FastAPI server, use the following command:
# uvicorn app:app --reload

