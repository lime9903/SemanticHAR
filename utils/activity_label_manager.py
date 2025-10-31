"""
Activity Label Manager for Cross-Domain HAR
Handles activity label mapping between different datasets
"""
import json
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
import os

@dataclass
class ActivityMapping:
    """Activity mapping configuration"""
    unified_label: str
    dataset_labels: Dict[str, List[str]]
    category: str
    description: str

class ActivityLabelManager:
    """
    Manages activity labels across different datasets for cross-domain HAR
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Activity Label Manager
        
        Args:
            config_path: Path to activity mapping configuration file
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), '..', 'configs', 'activity_mapping.json'
        )
        
        # Load activity mappings
        self.activity_mappings = self._load_activity_mappings()
        
        # Build reverse mappings for quick lookup
        self._build_reverse_mappings()
        
        # Unified activity vocabulary
        self.unified_vocab = self._build_unified_vocab()
        
    def _load_activity_mappings(self) -> Dict[str, ActivityMapping]:
        """Load activity mappings from configuration file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            # Default configuration
            config = self._get_default_mapping_config()
            # Save default configuration
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        
        mappings = {}
        for mapping_data in config['mappings']:
            mapping = ActivityMapping(
                unified_label=mapping_data['unified_label'],
                dataset_labels=mapping_data['dataset_labels'],
                category=mapping_data['category'],
                description=mapping_data['description']
            )
            mappings[mapping.unified_label] = mapping
            
        return mappings
    
    def _get_default_mapping_config(self) -> Dict:
        """Get default activity mapping configuration"""
        return {
            "version": "1.0",
            "description": "Cross-domain activity mapping for HAR datasets",
            "mappings": [
                {
                    "unified_label": "sleeping",
                    "category": "rest",
                    "description": "Sleeping or resting activities",
                    "dataset_labels": {
                        "UCI_ADL_home_a": ["Sleeping"],
                        "UCI_ADL_home_b": ["Sleeping"],
                        "MARBLE": []
                    }
                },
                {
                    "unified_label": "personal_care",
                    "category": "hygiene",
                    "description": "Personal hygiene and care activities",
                    "dataset_labels": {
                        "UCI_ADL_home_a": ["Toileting", "Showering", "Grooming"],
                        "UCI_ADL_home_b": ["Toileting", "Showering", "Grooming"],
                        "MARBLE": ["Washing_dishes", "Taking_medicines"]
                    }
                },
                {
                    "unified_label": "eating",
                    "category": "nutrition",
                    "description": "Food preparation and consumption activities",
                    "dataset_labels": {
                        "UCI_ADL_home_a": ["Breakfast", "Lunch", "Dinner", "Snack"],
                        "UCI_ADL_home_b": ["Breakfast", "Lunch", "Dinner", "Snack"],
                        "MARBLE": ["Eating", "Preparing_cold_meal", "Cooking", "Setting_up_table", "Clearing_table"]
                    }
                },
                {
                    "unified_label": "leisure",
                    "category": "entertainment",
                    "description": "Leisure and entertainment activities",
                    "dataset_labels": {
                        "UCI_ADL_home_a": ["Spare_Time/TV"],
                        "UCI_ADL_home_b": ["Spare_Time/TV"],
                        "MARBLE": ["Watching_tv", "Using_pc"]
                    }
                },
                {
                    "unified_label": "mobility",
                    "category": "movement",
                    "description": "Movement and transportation activities",
                    "dataset_labels": {
                        "UCI_ADL_home_a": ["Leaving"],
                        "UCI_ADL_home_b": ["Leaving"],
                        "MARBLE": ["Leaving_home", "Entering_home"]
                    }
                },
                {
                    "unified_label": "communication",
                    "category": "social",
                    "description": "Communication and social activities",
                    "dataset_labels": {
                        "UCI_ADL_home_a": [],
                        "UCI_ADL_home_b": [],
                        "MARBLE": ["Answering_phone", "Making_phone_call"]
                    }
                }
            ]
        }
    
    def _build_reverse_mappings(self):
        """Build reverse mappings for quick lookup"""
        self.dataset_to_unified = {}
        self.unified_to_idx = {}
        
        for unified_label, mapping in self.activity_mappings.items():
            # Map dataset-specific labels to unified labels
            for dataset, labels in mapping.dataset_labels.items():
                if dataset not in self.dataset_to_unified:
                    self.dataset_to_unified[dataset] = {}
                
                for label in labels:
                    self.dataset_to_unified[dataset][label] = unified_label
        
        # Build unified label to index mapping
        for idx, unified_label in enumerate(sorted(self.activity_mappings.keys())):
            self.unified_to_idx[unified_label] = idx
    
    def _build_unified_vocab(self) -> List[str]:
        """Build unified vocabulary"""
        return sorted(self.activity_mappings.keys())
    
    def map_to_unified(self, activity_label: str, dataset: str) -> Optional[str]:
        """
        Map dataset-specific activity label to unified label
        
        Args:
            activity_label: Dataset-specific activity label
            dataset: Dataset name
            
        Returns:
            Unified activity label or None if not found
        """
        if dataset in self.dataset_to_unified:
            return self.dataset_to_unified[dataset].get(activity_label)
        return None
    
    def get_unified_index(self, unified_label: str) -> Optional[int]:
        """
        Get index for unified activity label
        
        Args:
            unified_label: Unified activity label
            
        Returns:
            Index or None if not found
        """
        return self.unified_to_idx.get(unified_label)
    
    def get_dataset_activities(self, dataset: str) -> List[str]:
        """
        Get all activity labels for a specific dataset
        
        Args:
            dataset: Dataset name
            
        Returns:
            List of activity labels
        """
        activities = []
        for mapping in self.activity_mappings.values():
            if dataset in mapping.dataset_labels:
                activities.extend(mapping.dataset_labels[dataset])
        return sorted(list(set(activities)))
    
    def get_unified_activities(self) -> List[str]:
        """
        Get all unified activity labels
        
        Returns:
            List of unified activity labels
        """
        return self.unified_vocab.copy()
    
    def get_category(self, unified_label: str) -> Optional[str]:
        """
        Get category for unified activity label
        
        Args:
            unified_label: Unified activity label
            
        Returns:
            Category or None if not found
        """
        if unified_label in self.activity_mappings:
            return self.activity_mappings[unified_label].category
        return None
    
    def add_custom_mapping(self, unified_label: str, dataset: str, 
                          dataset_labels: List[str], category: str = "custom",
                          description: str = ""):
        """
        Add custom activity mapping
        
        Args:
            unified_label: Unified activity label
            dataset: Dataset name
            dataset_labels: List of dataset-specific labels
            category: Activity category
            description: Description of the activity
        """
        if unified_label not in self.activity_mappings:
            self.activity_mappings[unified_label] = ActivityMapping(
                unified_label=unified_label,
                dataset_labels={},
                category=category,
                description=description
            )
        
        self.activity_mappings[unified_label].dataset_labels[dataset] = dataset_labels
        
        # Rebuild reverse mappings
        self._build_reverse_mappings()
        self.unified_vocab = self._build_unified_vocab()
        
        # Save updated configuration
        self._save_configuration()
    
    def _save_configuration(self):
        """Save current configuration to file"""
        config = {
            "version": "1.0",
            "description": "Cross-domain activity mapping for HAR datasets",
            "mappings": []
        }
        
        for mapping in self.activity_mappings.values():
            config["mappings"].append({
                "unified_label": mapping.unified_label,
                "category": mapping.category,
                "description": mapping.description,
                "dataset_labels": mapping.dataset_labels
            })
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def get_mapping_info(self) -> Dict:
        """
        Get comprehensive mapping information
        
        Returns:
            Dictionary containing mapping information
        """
        return {
            "unified_vocab": self.unified_vocab,
            "unified_to_idx": self.unified_to_idx,
            "dataset_to_unified": self.dataset_to_unified,
            "mappings": {
                label: {
                    "category": mapping.category,
                    "description": mapping.description,
                    "dataset_labels": mapping.dataset_labels
                }
                for label, mapping in self.activity_mappings.items()
            }
        }
    
    def validate_mapping(self, dataset: str, activity_label: str) -> bool:
        """
        Validate if activity label exists for dataset
        
        Args:
            dataset: Dataset name
            activity_label: Activity label to validate
            
        Returns:
            True if valid, False otherwise
        """
        return activity_label in self.get_dataset_activities(dataset)
    
    def get_unknown_activities(self, dataset: str, activity_labels: List[str]) -> List[str]:
        """
        Get activities that are not mapped for a dataset
        
        Args:
            dataset: Dataset name
            activity_labels: List of activity labels to check
            
        Returns:
            List of unmapped activity labels
        """
        known_activities = set(self.get_dataset_activities(dataset))
        return [label for label in activity_labels if label not in known_activities]


# Global instance for easy access
activity_manager = ActivityLabelManager()


