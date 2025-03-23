from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig

# Load dataset
raw_datasets = load_dataset("imdb")

# Reduce dataset to 5000 samples for faster training
small_train_dataset = raw_datasets["train"].select(range(5000))
small_test_dataset = raw_datasets["test"].select(range(5000))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print("Tokenizer loaded.")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize dataset
tokenized_train = small_train_dataset.map(tokenize_function, batched=True)
tokenized_test = small_test_dataset.map(tokenize_function, batched=True)
print("Dataset loaded and tokenized.")

# Load base model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
print("Model loaded on CPU.")

# Apply PEFT LoRA configuration
peft_config = LoraConfig(
    task_type="SEQ_CLS",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"]  # Adjusted target modules for DistilBERT
)
peft_model = get_peft_model(model, peft_config)
print("PEFT model applied.")

# Training arguments optimized for 16GB RAM
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Balanced batch size for memory efficiency
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    fp16=False,  # Disabled mixed precision since CPU does not support FP16
    gradient_checkpointing=True,  # Saves memory during training
)

# Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

# Train model
trainer.train()
