from typing import List, Dict, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sentence_transformers import CrossEncoder as SentenceCrossEncoder
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from tqdm import tqdm
from collections import defaultdict
import random
from statement_pair_labeler.response_model import QualificationCategory

logger = logging.getLogger(__name__)

class StatementPairDataset(Dataset):
    def __init__(self, pairs: List[Dict], tokenizer, undersample=True):
        if undersample:
            self.pairs = self._undersample_pairs(pairs)
        else:
            self.pairs = pairs
        self.tokenizer = tokenizer

    def _undersample_pairs(self, pairs: List[Dict]) -> List[Dict]:
        """Undersample all classes to match the size of the smallest class"""
        # Group pairs by category
        category_pairs = defaultdict(list)
        for pair in pairs:
            category = pair['label_details']['category']
            category_pairs[category].append(pair)
        
        # Find size of smallest category
        min_size = min(len(pairs) for pairs in category_pairs.values())
        logger.info(f"Undersampling all categories to {min_size} pairs")
        
        # Undersample each category
        balanced_pairs = []
        for category in QualificationCategory:
            if category.value in category_pairs:
                sampled_pairs = random.sample(category_pairs[category.value], min_size)
                balanced_pairs.extend(sampled_pairs)
                logger.info(f"{category.value}: {len(category_pairs[category.value])} -> {min_size}")
        
        random.shuffle(balanced_pairs)
        return balanced_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return [pair['jd_statement'], pair['skill_statement']], pair['label_details']['score']

def load_labeled_pairs(data_dir: str) -> List[Dict]:
    pairs = []
    data_path = Path(data_dir)
    
    for json_file in data_path.glob('*.json'):
        with open(json_file, 'r') as f:
            pair = json.load(f)
            # Add filename to the pair data for reference
            pair['_filename'] = json_file.name
            pairs.append(pair)
    
    return pairs

def normalize_score(score: float) -> float:
    """Normalize a score to [0, 1] range using sigmoid normalization"""
    return 1.0 / (1.0 + np.exp(-score))

def normalize_scores(scores: Union[List[float], float]) -> Union[List[float], float]:
    """Normalize a single score or list of scores"""
    if isinstance(scores, list):
        return [normalize_score(score) for score in scores]
    return normalize_score(scores)

def evaluate_model(model, tokenizer, test_dataset: StatementPairDataset) -> Dict[str, float]:
    logger.info("Starting model evaluation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    predictions = []
    true_labels = []
    
    cross_encoder = SentenceCrossEncoder(tokenizer.name_or_path, device=device)
    cross_encoder.model = model
    
    # Create a single progress bar for all pairs
    progress_bar = tqdm(range(len(test_dataset)), desc="Evaluating pairs", leave=True)
    
    with torch.no_grad():
        for i in progress_bar:
            # Get pair from dataset
            texts, true_score = test_dataset[i]
            
            raw_pred = cross_encoder.predict([texts], show_progress_bar=False)[0]
            pred = normalize_score(raw_pred)
            
            predictions.append(pred)
            true_labels.append(true_score)
            
            # Update progress bar description occasionally
            if i < 5 or i % 100 == 0:
                progress_bar.set_postfix({
                    'pred': f'{pred:.3f}',
                    'true': f'{true_score:.3f}'
                })
    
    # Calculate metrics
    mse = mean_squared_error(true_labels, predictions)
    mae = mean_absolute_error(true_labels, predictions)
    rmse = np.sqrt(mse)
    
    # Log final statistics
    pred_array = np.array(predictions)
    logger.info("\nPrediction Distribution:")
    logger.info(f"Mean: {pred_array.mean():.3f}")
    logger.info(f"Std: {pred_array.std():.3f}")
    logger.info(f"Min: {pred_array.min():.3f}")
    logger.info(f"Max: {pred_array.max():.3f}")
    
    return {'mse': mse, 'mae': mae, 'rmse': rmse}

def train_cross_encoder(
    model_name: str,
    train_pairs: List[Dict],
    val_pairs: List[Dict],
    output_dir: str,
    batch_size: int = 8,
    num_epochs: int = 3
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model = model.to(device)
    
    train_dataset = StatementPairDataset(train_pairs, tokenizer)
    val_dataset = StatementPairDataset(val_pairs, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        logging_first_step=True,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            texts, labels = inputs
            outputs = model(**self.tokenizer(
                texts[0], texts[1],
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device))
            
            # Apply sigmoid to model outputs before computing loss
            predictions = torch.sigmoid(outputs.logits).squeeze()
            loss = torch.nn.functional.mse_loss(
                predictions,
                labels.to(device)
            )
            
            return (loss, outputs) if return_outputs else loss
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Saving final model...")
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    
    return model, tokenizer

class StatementPairCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Unzip the features (texts, labels)
        texts, labels = zip(*features)
        
        # Unzip the texts (jd_statements, skill_statements)
        jd_statements, skill_statements = zip(*texts)
        
        # Tokenize the texts
        encoded = self.tokenizer(
            list(jd_statements),
            list(skill_statements),
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.float)
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'token_type_ids': encoded.get('token_type_ids', None),
            'labels': labels
        }