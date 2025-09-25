"""
Transformer-based sensor encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
from config import SemanticHARConfig

class PositionalEncoding(nn.Module):
    """Positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class SensorEncoder(nn.Module):
    """Transformer-based sensor encoder"""
    
    def __init__(self, config: SemanticHARConfig):
        super(SensorEncoder, self).__init__()
        self.config = config
        
        # input dimension (Ambient sensor feature dimension)
        self.input_dim = config.sensor_encoder_input_dim
        self.hidden_dim = config.sensor_encoder_hidden_dim
        
        # input projection
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.input_layer_norm = nn.LayerNorm(self.hidden_dim)
        
        # positional encoding
        self.positional_encoding = PositionalEncoding(self.hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config.sensor_encoder_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.sensor_encoder_layers
        )
        
        # output projection
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_layer_norm = nn.LayerNorm(self.hidden_dim)
        
        # global average pooling
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        
        # dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensor_data: (batch_size, sequence_length, sensor_dim)
        Returns:
            embeddings: (batch_size, hidden_dim)
        """
        batch_size, seq_len, sensor_dim = sensor_data.shape
        
        # input projection
        x = self.input_projection(sensor_data)  # (batch_size, seq_len, hidden_dim)
        x = self.input_layer_norm(x)
        x = self.dropout(x)
        
        # positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_dim)
        
        # global average pooling
        pooled = self.global_pooling(encoded.transpose(1, 2))  # (batch_size, hidden_dim, 1)
        pooled = pooled.squeeze(-1)  # (batch_size, hidden_dim)
        
        # output projection
        output = self.output_projection(pooled)
        output = self.output_layer_norm(output)
        output = self.dropout(output)
        
        return output

class SensorEncoderTrainer:
    """Sensor encoder trainer"""
    
    def __init__(self, config: SemanticHARConfig, text_encoder):
        self.config = config
        self.device = torch.device(config.device)
        
        # initialize models
        self.sensor_encoder = SensorEncoder(config).to(self.device)
        self.text_encoder = text_encoder  # trained text encoder
        
        # optimizer
        self.optimizer = torch.optim.AdamW(
            self.sensor_encoder.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
    def train_step(self, sensor_data: torch.Tensor, 
                   sensor_interpretations: List[str]) -> Dict[str, float]:
        """training step"""
        
        self.optimizer.zero_grad()
        
        # sensor data encoding
        sensor_embeddings = self.sensor_encoder(sensor_data)
        
        # text encoding (fixed text encoder)
        with torch.no_grad():
            text_embeddings = self.text_encoder(sensor_interpretations)
        
        # contrastive learning loss
        contrastive_loss = self._compute_contrastive_loss(
            sensor_embeddings, text_embeddings
        )
        
        # backpropagation
        contrastive_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.sensor_encoder.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'contrastive_loss': contrastive_loss.item()
        }
    
    def _compute_contrastive_loss(self, sensor_embeddings: torch.Tensor, 
                                 text_embeddings: torch.Tensor) -> torch.Tensor:
        """compute contrastive learning loss"""
        # 정규화
        sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        
        # similarity matrix
        similarity_matrix = torch.matmul(sensor_embeddings, text_embeddings.T) / self.config.temperature
        
        # batch size
        batch_size = sensor_embeddings.size(0)
        
        # labels (diagonal is the answer)
        labels = torch.arange(batch_size, device=sensor_embeddings.device)
        
        # symmetric loss
        loss_s2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2s = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_s2t + loss_t2s) / 2
    
    def save_model(self, path: str):
        """save model"""
        torch.save({
            'sensor_encoder': self.sensor_encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, path)
    
    def load_model(self, path: str):
        """load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.sensor_encoder.load_state_dict(checkpoint['sensor_encoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

class MultiScaleSensorEncoder(nn.Module):
    """multi-scale sensor encoder (improved version)"""
    
    def __init__(self, config: SemanticHARConfig):
        super(MultiScaleSensorEncoder, self).__init__()
        self.config = config
        
        # input dimension (Ambient sensor feature dimension)
        self.input_dim = config.sensor_encoder_input_dim
        self.hidden_dim = config.sensor_encoder_hidden_dim
        
        # multi-scale convolution
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(self.input_dim, self.hidden_dim // 4, kernel_size=3, padding=1),
            nn.Conv1d(self.input_dim, self.hidden_dim // 4, kernel_size=5, padding=2),
            nn.Conv1d(self.input_dim, self.hidden_dim // 4, kernel_size=7, padding=3),
            nn.Conv1d(self.input_dim, self.hidden_dim // 4, kernel_size=9, padding=4),
        ])
        
        # convolution after normalization
        self.conv_layer_norm = nn.LayerNorm(self.hidden_dim)
        
        # input projection
        self.input_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.input_layer_norm = nn.LayerNorm(self.hidden_dim)
        
        # positional encoding
        self.positional_encoding = PositionalEncoding(self.hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config.sensor_encoder_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.sensor_encoder_layers
        )
        
        # attention pooling
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=config.sensor_encoder_heads,
            batch_first=True
        )
        
        # output projection
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_layer_norm = nn.LayerNorm(self.hidden_dim)
        
        # dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensor_data: (batch_size, sequence_length, sensor_dim)
        Returns:
            embeddings: (batch_size, hidden_dim)
        """
        batch_size, seq_len, sensor_dim = sensor_data.shape
        
        # multi-scale convolution
        conv_outputs = []
        for conv_layer in self.conv_layers:
            # (batch_size, sensor_dim, seq_len) -> (batch_size, hidden_dim//4, seq_len)
            conv_out = conv_layer(sensor_data.transpose(1, 2))
            conv_outputs.append(conv_out)
        
        # convolution output combination
        conv_combined = torch.cat(conv_outputs, dim=1)  # (batch_size, hidden_dim, seq_len)
        conv_combined = conv_combined.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
        conv_combined = self.conv_layer_norm(conv_combined)
        conv_combined = self.dropout(conv_combined)
        
        # input projection
        x = self.input_projection(conv_combined)
        x = self.input_layer_norm(x)
        x = self.dropout(x)
        
        # positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_dim)
        
        # attention pooling
        # use average embedding as query
        query = encoded.mean(dim=1, keepdim=True)  # (batch_size, 1, hidden_dim)
        pooled, _ = self.attention_pooling(query, encoded, encoded)  # (batch_size, 1, hidden_dim)
        pooled = pooled.squeeze(1)  # (batch_size, hidden_dim)
        
        # output projection
        output = self.output_projection(pooled)
        output = self.output_layer_norm(output)
        output = self.dropout(output)
        
        return output

class SensorEncoderWithAttention(nn.Module):
    """sensor encoder with attention mechanism"""
    
    def __init__(self, config: SemanticHARConfig):
        super(SensorEncoderWithAttention, self).__init__()
        self.config = config
        
        # input dimension (Ambient sensor feature dimension)
        self.input_dim = config.sensor_encoder_input_dim
        self.hidden_dim = config.sensor_encoder_hidden_dim
        
        # sensor-specific encoder
        self.accelerometer_encoder = nn.Linear(3, self.hidden_dim // 3)
        self.gyroscope_encoder = nn.Linear(3, self.hidden_dim // 3)
        self.magnetometer_encoder = nn.Linear(3, self.hidden_dim // 3)
        
        # sensor fusion
        self.sensor_fusion = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fusion_layer_norm = nn.LayerNorm(self.hidden_dim)
        
        # positional encoding
        self.positional_encoding = PositionalEncoding(self.hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config.sensor_encoder_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.sensor_encoder_layers
        )
        
        # sensor-specific attention
        self.sensor_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=config.sensor_encoder_heads,
            batch_first=True
        )
        
        # output projection
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_layer_norm = nn.LayerNorm(self.hidden_dim)
        
        # dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensor_data: (batch_size, sequence_length, 9) - [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z]
        Returns:
            embeddings: (batch_size, hidden_dim)
        """
        batch_size, seq_len, _ = sensor_data.shape
        
        # sensor data separation
        acc_data = sensor_data[:, :, :3]  # accelerometer
        gyro_data = sensor_data[:, :, 3:6]  # gyroscope
        mag_data = sensor_data[:, :, 6:9]  # magnetometer
        
        # 센서별 인코딩
        acc_encoded = self.accelerometer_encoder(acc_data)  # (batch_size, seq_len, hidden_dim//3)
        gyro_encoded = self.gyroscope_encoder(gyro_data)
        mag_encoded = self.magnetometer_encoder(mag_data)
        
        # 센서 융합
        fused = torch.cat([acc_encoded, gyro_encoded, mag_encoded], dim=-1)  # (batch_size, seq_len, hidden_dim)
        fused = self.sensor_fusion(fused)
        fused = self.fusion_layer_norm(fused)
        fused = self.dropout(fused)
        
        # 위치 인코딩 추가
        fused = fused.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        fused = self.positional_encoding(fused)
        fused = fused.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
        
        # Transformer 인코더
        encoded = self.transformer_encoder(fused)  # (batch_size, seq_len, hidden_dim)
        
        # 센서별 어텐션
        # 쿼리로 평균 임베딩 사용
        query = encoded.mean(dim=1, keepdim=True)  # (batch_size, 1, hidden_dim)
        attended, attention_weights = self.sensor_attention(query, encoded, encoded)
        attended = attended.squeeze(1)  # (batch_size, hidden_dim)
        
        # 출력 프로젝션
        output = self.output_projection(attended)
        output = self.output_layer_norm(output)
        output = self.dropout(output)
        
        return output, attention_weights
