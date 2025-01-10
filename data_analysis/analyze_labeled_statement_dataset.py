from matcher_model.training.dataset_analysis import analyze_labeled_dataset
from matcher_model.training.trainer import load_labeled_pairs
import logging
from pathlib import Path
import json
from tabulate import tabulate
import numpy as np

logger = logging.getLogger(__name__)

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'  # Simplified format for readability
    )
    
    logger.info("Starting labeled dataset analysis...")
    
    # Load dataset
    data_dir = "matcher_dataset/statement_pairs_labeled"
    pairs = load_labeled_pairs(data_dir)
    logger.info(f"Loaded {len(pairs)} pairs from {data_dir}")
    
    # Run analysis
    stats = analyze_labeled_dataset(pairs)
    
    # Print results in a clean format
    logger.info("\n=== Dataset Analysis Report ===\n")
    
    # Basic stats
    logger.info(f"Total pairs: {stats['total_pairs']}")
    
    # Score distribution
    logger.info("\nScore Distribution:")
    score_table = []
    for k, v in stats['score_stats'].items():
        if isinstance(v, np.ndarray):
            # Handle numpy arrays (like quartiles)
            formatted_value = ", ".join([f"{x:.3f}" for x in v])
        else:
            # Handle single values
            formatted_value = f"{v:.3f}" if isinstance(v, (float, np.float32, np.float64)) else str(v)
        score_table.append([k, formatted_value])

    logger.info(tabulate(score_table, headers=['Metric', 'Value'], tablefmt='simple'))
    
    # Category distribution
    logger.info("\nCategory Distribution:")
    cat_table = [[cat, count, f"{count/stats['total_pairs']*100:.1f}%"] 
                 for cat, count in stats['category_dist'].items()]
    logger.info(tabulate(cat_table, headers=['Category', 'Count', 'Percentage'], tablefmt='simple'))
    
    # Text length statistics
    logger.info("\nText Length Statistics:")
    length_table = [
        ['JD Statements', f"{stats['text_length_stats']['jd_mean']:.1f}", 
         f"{stats['text_length_stats']['jd_std']:.1f}"],
        ['Skill Statements', f"{stats['text_length_stats']['skill_mean']:.1f}", 
         f"{stats['text_length_stats']['skill_std']:.1f}"]
    ]
    logger.info(tabulate(length_table, headers=['Type', 'Mean Words', 'Std Dev'], tablefmt='simple'))
    
    # Issues
    if stats['issues']:
        logger.info("\nPotential Issues:")
        for issue in stats['issues']:
            logger.info(f"⚠️  {issue}")
    
    # Save plots
    plots_dir = Path("analysis_results")
    plots_dir.mkdir(exist_ok=True)
    
    # Save statistics to JSON
    with open(plots_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    
    logger.info(f"\nAnalysis complete. Results saved to {plots_dir}")
    logger.info("Plots saved as 'dataset_analysis.png'")

if __name__ == "__main__":
    main() 