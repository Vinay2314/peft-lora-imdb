# PEFT LoRA IMDb Sentiment Classification

## Overview
This project implements **Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation)** on the IMDb dataset for sentiment classification. The goal is to fine-tune a **lightweight BERT model** efficiently using **LoRA** while reducing computational costs and memory usage.

## Features
- **IMDb Sentiment Classification** using `bert-tiny`
- **LoRA-based Fine-Tuning** for efficiency
- **Lightweight Model Training** on CPU
- **Dataset Reduction** for quick testing

---

## Setup and Installation
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/Vinay2314/peft-lora-imdb.git
cd peft-lora-imdb
```

### 2️⃣ Create and Activate Virtual Environment
```sh
python -m venv venv  # Create virtual environment
# Activate venv:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

---

## Model Implementation
### 1️⃣ Load IMDb Dataset
The dataset is loaded using the `datasets` library and reduced to **2000 samples** for faster training.
```python
from datasets import load_dataset
raw_datasets = load_dataset("imdb")
small_train_dataset = raw_datasets["train"].select(range(2000))
small_test_dataset = raw_datasets["test"].select(range(2000))
```

### 2️⃣ Tokenization using `bert-tiny`
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_train = small_train_dataset.map(tokenize_function, batched=True)
tokenized_test = small_test_dataset.map(tokenize_function, batched=True)
```

### 3️⃣ Load Pretrained Model
```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
```

### 4️⃣ Apply PEFT LoRA Configuration
```python
from peft import get_peft_model, LoraConfig
peft_config = LoraConfig(
    task_type="SEQ_CLS", r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "value"]
)
peft_model = get_peft_model(model, peft_config)
```

### 5️⃣ Training the Model
```python
from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
    output_dir="./results", num_train_epochs=2, per_device_train_batch_size=2,
    evaluation_strategy="epoch", save_strategy="epoch",
    logging_dir="./logs", logging_steps=10, fp16=True
)
trainer = Trainer(
    model=peft_model, args=training_args,
    train_dataset=tokenized_train, eval_dataset=tokenized_test,
)
trainer.train()
```

---

## Running the Model
To start training, run:
```sh
python peft_train.py  # Replace with your script filename if needed
```

---

## GitHub Repository Setup
### 1️⃣ Initialize Git and Push to GitHub
```sh
git init
git add .
git commit -m "Initial commit - PEFT LoRA IMDb project"
git branch -M main
git remote add origin https://github.com/Vinay2314/peft-lora-imdb.git
git push -u origin main
```

---

## Conclusion
This project demonstrates how to efficiently fine-tune a transformer model for sentiment classification using PEFT LoRA, reducing the computational cost while achieving effective results. 🚀

🔗 **GitHub Repository:** [peft-lora-imdb](https://github.com/Vinay2314/peft-lora-imdb)
