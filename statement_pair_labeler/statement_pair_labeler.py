import yaml
import importlib
import json
import argparse
import os
import logging
from typing import Dict, Any, List
from tqdm import tqdm

from .labelers import BaseLabeler
from .response_model import LabeledPair, LabeledPairsData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_LABELER_NAME = "dummy_labeler"  # Will be replaced with actual labeler

class StatementPairLabeler:
    def __init__(self, labeler_name=DEFAULT_LABELER_NAME):
        self.labeler_name = labeler_name
        logger.info(f"Initializing StatementPairLabeler with labeler: {self.labeler_name}")
        self.config = self.read_labeler_yaml()
        self.labeler = self.create_labeler_from_config(self.config)

    def read_labeler_yaml(self):
        yaml_path = f"statement_pair_labeler/yaml_configs/{self.labeler_name}.yaml"
        logger.info(f"Reading labeler configuration from {yaml_path}")
        try:
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
                logger.info(f"Labeler configuration loaded successfully")
                return config
        except Exception as e:
            logger.error(f"Failed to read labeler YAML: {e}")
            raise

    def create_labeler_from_config(self, config) -> BaseLabeler:
        logger.info("Creating labeler from configuration")
        try:
            labeler_module = importlib.import_module(
                f"statement_pair_labeler.labelers.{config['labeler'].split('.')[0]}"
            )
            LabelerClass = getattr(labeler_module, config['labeler'].split('.')[1])
            labeler = LabelerClass()
            logger.info("Labeler created successfully")
            return labeler
        except Exception as e:
            logger.error(f"Failed to create labeler from configuration: {e}")
            raise

    def label_pairs(self, input_file: str, output_file: str = None):
        logger.info(f"Loading pairs from {input_file}")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read input file: {e}")
            raise

        labeled_pairs = []
        logger.info("Labeling pairs...")
        for pair in tqdm(data['pairs'], desc="Labeling pairs"):
            label = self.labeler.label_pair(
                pair['jd_statement'],
                pair['skill_statement'],
                pair['metadata']
            )
            
            labeled_pair = LabeledPair(
                jd_statement=pair['jd_statement'],
                skill_statement=pair['skill_statement'],
                label=label,
                metadata=pair['metadata']
            )
            labeled_pairs.append(labeled_pair)

        # Create output data
        labeled_data = LabeledPairsData(
            total_pairs=len(labeled_pairs),
            sampling_params=data.get('sampling_params', {}),
            labeled_pairs=[pair.dict() for pair in labeled_pairs]
        )

        # Save to file if output_file is specified
        if output_file:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Saving labeled pairs to {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(labeled_data.dict(), f, indent=2)

        return labeled_data

def main():
    parser = argparse.ArgumentParser(description='Label statement pairs')
    parser.add_argument('input_file', type=str, help='Path to the statement pairs JSON file')
    parser.add_argument('--output_file', type=str, help='Path to save labeled pairs')
    parser.add_argument('--labeler', type=str, default=DEFAULT_LABELER_NAME, 
                       help='Name of the labeler to use')
    args = parser.parse_args()

    labeler = StatementPairLabeler(args.labeler)
    labeled_data = labeler.label_pairs(args.input_file, args.output_file)
    
    if not args.output_file:
        print(json.dumps(labeled_data.dict(), indent=2))

if __name__ == "__main__":
    main() 