"""
SemanticHAR System Configuration File
"""
import os
import torch
from dataclasses import dataclass

@dataclass
class SemanticHARConfig:
    # Environment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_evaluation: bool = False
    
    # Directory Path
    root_dir: str = "/workspace/semantic"
    data_dir: str = os.path.join(root_dir, "data")  # TODO: You need to download the dataset and put it in this directory
    output_dir: str = os.path.join(root_dir, "outputs")
    model_dir: str = os.path.join(root_dir, "checkpoints")
    
    # Data Path
    marble_data_path: str = os.path.join(data_dir, "MARBLE")
    uci_adl_data_path: str = os.path.join(data_dir, "UCI ADL Binary Dataset")
    windows_file: str = os.path.join(output_dir, "windows.json")
    semantic_interpretations_file: str = os.path.join(output_dir, "semantic_interpretations.json")
    matched_windows_file: str = os.path.join(output_dir, "matched_windows.json")

    # Data
    window_size_seconds: int = 60
    overlap_ratio: float = 0.8
    use_event_based: bool = True
    source_dataset: str = "UCI_ADL_home_b"  # TODO: choices (UCI_ADL_home_a, UCI_ADL_home_b, MARBLE)
    target_dataset: str = "UCI_ADL_home_a"
    source_train_ratio: float = 0.8  # 80% for train, 20% for val
    
    # LLM
    llm_model: str = "gpt-3.5-turbo"
    max_tokens: int = 250
    temperature_llm: float = 0.7  # consistent: 0.2, creative: 0.7
    max_sequence_length: int = 256

    # Iterative Regeneration
    max_iterations: int = 5
    cluster_threshold: float = 0.1
    outlier_ratio: float = 0.2

    # General Training Parameters
    early_stopping: bool = True

    # Text Encoder
    text_encoder_temperature: float = 0.07
    text_encoder_patience: int = 15
    text_encoder_gradient_clip_norm: float = 0.5
    text_encoder_batch_size: int = 16
    text_encoder_learning_rate: float = 1e-5
    text_encoder_num_epochs: int = 100
    text_encoder_model: str = "bert-base-uncased"
    text_encoder_weight_decay: float = 0.01
    alpha: float = 0.7  # contrastive learning weight
    beta: float = 0.3   # reconstruction weight
    text_encoder_hidden_dim: int = 768
    text_encoder_layers: int = 6
    text_encoder_heads: int = 8
    text_encoder_dropout: float = 0.1
    text_encoder_attention_dropout: float = 0.1

    # Sensor Encoder
    sensor_encoder_temperature: float = 0.1
    sensor_encoder_patience: int = 10
    sensor_encoder_gradient_clip_norm: float = 1.0
    sensor_encoder_batch_size: int = 32
    sensor_encoder_learning_rate: float = 1e-4
    sensor_encoder_num_epochs: int = 100
    sensor_encoder_weight_decay: float = 0.01
    sensor_encoder_input_dim: int = 9
    sensor_encoder_hidden_dim: int = 768  # Match BERT output dimension
    sensor_encoder_output_dim: int = 768
    sensor_encoder_layers: int = 3
    sensor_encoder_heads: int = 4
    sensor_encoder_dropout: float = 0.1
    sensor_encoder_attention_dropout: float = 0.1

    # Logging
    log_interval: int = 10
    save_interval: int = 1000
    
    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # run 'export OPENAI_API_KEY=your-api-key' in terminal
