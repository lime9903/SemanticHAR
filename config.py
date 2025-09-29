"""
SemanticHAR System Configuration File
"""
import os
import torch
from dataclasses import dataclass

@dataclass
class SemanticHARConfig:
    # Data Path
    marble_data_path: str = "/workspace/semantic/data/MARBLE"
    uci_adl_data_path: str = "/workspace/semantic/data/UCI ADL Binary Dataset"
    
    # Data
    dataset_name: str = "UCI_ADL"
    window_size_seconds: int = 60
    overlap_ratio: float = 0.8
    max_windows_per_home: int = 100
    max_activity_interpretations: int = 20

    # Model
    text_encoder_model: str = "bert-base-uncased"
    sensor_encoder_hidden_dim: int = 512  # Reduced from 768
    sensor_encoder_layers: int = 3        # Reduced from 6
    sensor_encoder_heads: int = 4         # Reduced from 8
    max_sequence_length: int = 256        # Reduced from 512
    
    # Training
    batch_size: int = 16
    learning_rate: float = 1e-5
    num_epochs: int = 100
    temperature: float = 0.07
    early_stopping: bool = True
    patience: int = 10
    weight_decay: float = 0.05  # Increased weight decay for better regularization
    gradient_clip_norm: float = 0.5  # Gradient clipping for stability
    
    # LLM
    llm_model: str = "gpt-3.5-turbo"
    max_tokens: int = 800
    temperature_llm: float = 0.7  # consistent: 0.2, creative: 0.7
    
    # Iterative Regeneration
    max_iterations: int = 5
    cluster_threshold: float = 0.1
    outlier_ratio: float = 0.2

    # Text Encoder
    alpha: float = 0.5  # contrastive learning weight
    beta: float = 0.3   # reconstruction weight
    text_encoder_model: str = "bert-base-uncased"
    text_encoder_hidden_dim: int = 768
    text_encoder_layers: int = 6
    text_encoder_heads: int = 8
    text_encoder_dropout: float = 0.1
    text_encoder_attention_dropout: float = 0.1

    # Sensor Encoder
    sensor_encoder_input_dim: int = 9
    sensor_encoder_output_dim: int = 768
    sensor_encoder_layers: int = 6
    sensor_encoder_heads: int = 8
    sensor_encoder_dropout: float = 0.1
    sensor_encoder_attention_dropout: float = 0.1

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output Path
    output_dir: str = "/workspace/semantic/outputs"
    model_save_path: str = "/workspace/semantic/models"
    
    # Logging
    log_interval: int = 10
    save_interval: int = 1000
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # run 'export OPENAI_API_KEY=your-api-key' in terminal
