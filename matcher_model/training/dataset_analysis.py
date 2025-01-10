from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import logging
from pathlib import Path
import seaborn as sns
from statement_pair_labeler.response_model import QualificationCategory

logger = logging.getLogger(__name__)

def analyze_labeled_dataset(pairs: List[Dict]) -> Dict:
    """Analyze the health of labeled statement pairs dataset"""
    
    # Score distribution
    scores = [p['label_details']['score'] for p in pairs]
    categories = [p['label_details']['category'] for p in pairs]
    
    # Text length statistics
    jd_lengths = [len(p['jd_statement'].split()) for p in pairs]
    skill_lengths = [len(p['skill_statement'].split()) for p in pairs]
    
    # Calculate statistics
    stats = {
        'total_pairs': len(pairs),
        'score_stats': {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': min(scores),
            'max': max(scores),
            'quartiles': np.percentile(scores, [25, 50, 75])
        },
        'category_dist': Counter(categories),
        'text_length_stats': {
            'jd_mean': np.mean(jd_lengths),
            'jd_std': np.std(jd_lengths),
            'skill_mean': np.mean(skill_lengths),
            'skill_std': np.std(skill_lengths)
        }
    }
    
    # Check for potential issues
    issues = []
    
    # Check score distribution
    if stats['score_stats']['std'] < 0.1:
        issues.append("Low variance in scores - might need more diverse examples")
    
    # Check category balance
    category_counts = stats['category_dist']
    total = sum(category_counts.values())
    for cat in QualificationCategory:
        if cat.value not in category_counts or category_counts[cat.value] / total < 0.1:
            issues.append(f"Underrepresented category: {cat.value}")
    
    # Check for very short or identical statements
    for pair_file in pairs:
        # Extract pair number from filename (e.g., "pair_095845.json" -> "095845")
        pair_id = Path(pair_file['_filename']).stem.split('_')[1]
        
        jd_words = len(pair_file['jd_statement'].split())
        skill_words = len(pair_file['skill_statement'].split())
        
        if jd_words < 3:
            issues.append(f"Very short JD statement in pair_{pair_id}.json: '{pair_file['jd_statement']}'")
        if skill_words < 3:
            issues.append(f"Very short skill statement in pair_{pair_id}.json: '{pair_file['skill_statement']}'")
        if pair_file['jd_statement'] == pair_file['skill_statement']:
            issues.append(f"Identical statements in pair_{pair_id}.json")
    
    stats['issues'] = issues
    
    # Plot distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    sns.histplot(scores, bins=20)
    plt.title('Score Distribution')
    plt.xlabel('Score')
    
    plt.subplot(132)
    sns.barplot(x=list(stats['category_dist'].keys()), 
                y=list(stats['category_dist'].values()))
    plt.title('Category Distribution')
    plt.xticks(rotation=45)
    
    plt.subplot(133)
    sns.kdeplot(jd_lengths, label='JD Statements')
    sns.kdeplot(skill_lengths, label='Skill Statements')
    plt.title('Statement Length Distribution')
    plt.xlabel('Word Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png')
    
    return stats 