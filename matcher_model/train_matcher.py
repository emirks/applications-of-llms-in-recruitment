from matcher_model.training.trainer import (
    load_labeled_pairs, evaluate_model, train_cross_encoder, StatementPairDataset, StatementPairCollator
)
from sklearn.model_selection import train_test_split
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)

def main():
    logger.info("Starting matcher model training pipeline")
    
    # Load labeled pairs
    logger.info("\nLoading labeled statement pairs...")
    pairs = load_labeled_pairs("matcher_dataset/statement_pairs_labeled")
    logger.info(f"Loaded {len(pairs)} labeled pairs")
    
    # Log initial dataset statistics
    categories = [p['label_details']['category'] for p in pairs]
    logger.info("\nInitial Dataset Statistics:")
    logger.info(f"Category Distribution: {Counter(categories)}")
    
    # Initialize model and tokenizer
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    # Create balanced dataset first
    dataset = StatementPairDataset(pairs, tokenizer=None, undersample=True)
    balanced_pairs = dataset.pairs
    logger.info("\nAfter undersampling:")
    logger.info(f"Total balanced pairs: {len(balanced_pairs)}")
    logger.info(f"Category Distribution: {Counter([p['label_details']['category'] for p in balanced_pairs])}")
    
    # Split balanced dataset into train/val/test
    train_val, test = train_test_split(
        balanced_pairs, 
        test_size=0.2, 
        random_state=42,
        stratify=[p['label_details']['category'] for p in balanced_pairs]
    )
    train, val = train_test_split(
        train_val, 
        test_size=0.2, 
        random_state=42,
        stratify=[p['label_details']['category'] for p in train_val]
    )
    
    logger.info("\nFinal split sizes:")
    logger.info(f"Train: {len(train)} pairs")
    logger.info(f"Validation: {len(val)} pairs")
    logger.info(f"Test: {len(test)} pairs")
    
    # Create datasets (without undersampling since data is already balanced)
    train_dataset = StatementPairDataset(train, tokenizer, undersample=False)
    val_dataset = StatementPairDataset(val, tokenizer, undersample=False)
    test_dataset = StatementPairDataset(test, tokenizer, undersample=False)
    
    # Evaluate base model
    logger.info("Evaluating base model...")
    base_metrics = evaluate_model(base_model, tokenizer, test_dataset)
    logger.info(f"Base model metrics: {base_metrics}")
    
    # Train model
    logger.info("Training model...")
    training_args = TrainingArguments(
        output_dir="matcher_model/trained_models",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Create custom collator
    data_collator = StatementPairCollator(tokenizer)
    
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    trainer.train()
    
    # Evaluate trained model
    logger.info("Evaluating trained model...")
    trained_metrics = evaluate_model(trainer.model, tokenizer, test_dataset)
    logger.info(f"Trained model metrics: {trained_metrics}")
    
    # Save the trained model
    trainer.save_model("matcher_model/trained_models/final")
    tokenizer.save_pretrained("matcher_model/trained_models/final")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()