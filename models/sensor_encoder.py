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
    """Transformer-based sensor encoder for ambient/environmental sensors"""
    
    def __init__(self, config: SemanticHARConfig):
        super(SensorEncoder, self).__init__()
        self.config = config
        
        # Hidden dimension
        self.hidden_dim = config.sensor_encoder_hidden_dim
        
        # Sensor type vocabulary (환경 센서 타입)
        self.sensor_types = ['PIR', 'Magnetic', 'Flush', 'Pressure', 'Electric', 'UNK']
        self.sensor_type_to_idx = {s: i for i, s in enumerate(self.sensor_types)}
        
        # Sensor type embedding
        self.sensor_type_embedding = nn.Embedding(
            num_embeddings=len(self.sensor_types),
            embedding_dim=self.hidden_dim // 4
        )
        
        # Location embedding (common locations in ambient sensor data)
        # Dynamic vocabulary - will be updated during training
        self.location_vocab = ['UNK']
        self.location_to_idx = {'UNK': 0}
        self.location_embedding = nn.Embedding(
            num_embeddings=100,  # Reserve space for locations
            embedding_dim=self.hidden_dim // 4,
            padding_idx=0
        )
        
        # Place embedding (rooms)
        self.place_vocab = ['UNK']
        self.place_to_idx = {'UNK': 0}
        self.place_embedding = nn.Embedding(
            num_embeddings=50,  # Reserve space for places
            embedding_dim=self.hidden_dim // 4,
            padding_idx=0
        )
        
        # Temporal feature projection (duration, time_since_last_event)
        self.temporal_projection = nn.Linear(2, self.hidden_dim // 4)
        
        # Event feature fusion
        self.event_fusion = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.event_layer_norm = nn.LayerNorm(self.hidden_dim)
        
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
        
        # Attention pooling
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
        
    def _update_vocab(self, locations: List[str], places: List[str]):
        """동적으로 위치/장소 vocabulary 업데이트"""
        for loc in locations:
            if loc not in self.location_to_idx:
                idx = len(self.location_vocab)
                self.location_vocab.append(loc)
                self.location_to_idx[loc] = idx
        
        for place in places:
            if place not in self.place_to_idx:
                idx = len(self.place_vocab)
                self.place_vocab.append(place)
                self.place_to_idx[place] = idx
    
    def forward(self, sensor_events: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            sensor_events: Dictionary containing:
                - sensor_types: (batch_size, max_events) - sensor type indices
                - locations: (batch_size, max_events) - location indices
                - places: (batch_size, max_events) - place indices
                - durations: (batch_size, max_events) - event duration (normalized)
                - time_deltas: (batch_size, max_events) - time since last event (normalized)
                - mask: (batch_size, max_events) - padding mask
        Returns:
            embeddings: (batch_size, hidden_dim)
        """
        batch_size = sensor_events['sensor_types'].shape[0]
        max_events = sensor_events['sensor_types'].shape[1]
        
        # Embed sensor types
        type_embeds = self.sensor_type_embedding(sensor_events['sensor_types'])  # (B, E, H/4)
        
        # Embed locations
        location_embeds = self.location_embedding(sensor_events['locations'])  # (B, E, H/4)
        
        # Embed places
        place_embeds = self.place_embedding(sensor_events['places'])  # (B, E, H/4)
        
        # Project temporal features
        temporal_features = torch.stack([
            sensor_events['durations'],
            sensor_events['time_deltas']
        ], dim=-1)  # (B, E, 2)
        temporal_embeds = self.temporal_projection(temporal_features)  # (B, E, H/4)
        
        # Concatenate all embeddings
        event_embeds = torch.cat([
            type_embeds,
            location_embeds,
            place_embeds,
            temporal_embeds
        ], dim=-1)  # (B, E, H)
        
        # Fuse event features
        x = self.event_fusion(event_embeds)
        x = self.event_layer_norm(x)
        x = self.dropout(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (E, B, H)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # (B, E, H)
        
        # Create attention mask for padding
        # mask is True for padding positions
        src_key_padding_mask = sensor_events['mask']  # (B, E)
        
        # Transformer encoder
        encoded = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )  # (B, E, H)
        
        # Attention pooling (평균 임베딩을 쿼리로 사용)
        # Compute mean only over non-padded positions
        mask_expanded = (~src_key_padding_mask).unsqueeze(-1).float()  # (B, E, 1)
        masked_encoded = encoded * mask_expanded
        sum_encoded = masked_encoded.sum(dim=1)  # (B, H)
        count = mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
        query = (sum_encoded / count).unsqueeze(1)  # (B, 1, H)
        
        pooled, _ = self.attention_pooling(
            query, encoded, encoded,
            key_padding_mask=src_key_padding_mask
        )  # (B, 1, H)
        pooled = pooled.squeeze(1)  # (B, H)
        
        # Output projection
        output = self.output_projection(pooled)
        output = self.output_layer_norm(output)
        output = self.dropout(output)
        
        return output

class SensorEncoderTrainer:
    """Sensor encoder trainer for Phase 2 training"""
    
    def __init__(self, config: SemanticHARConfig, text_encoder):
        self.config = config
        self.device = torch.device(config.device)
        
        # initialize models
        self.sensor_encoder = SensorEncoder(config).to(self.device)
        self.text_encoder = text_encoder  # trained text encoder (frozen)
        
        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # optimizer (only for sensor encoder)
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


class SensorEncoderDataset:
    """Dataset for sensor encoder training (Phase 2) - Ambient sensor events"""
    
    def __init__(self, sensor_events_list: List[Dict], 
                 sensor_interpretations: List[str],
                 activities: List[str],
                 max_events: int = 50):
        """
        Args:
            sensor_events_list: List of sensor event dictionaries
            sensor_interpretations: List of corresponding text interpretations
            activities: List of activity labels
            max_events: Maximum number of events per window (for padding)
        """
        # Filter out None values and ensure data consistency
        valid_indices = []
        for i, (events, interpretation) in enumerate(zip(sensor_events_list, sensor_interpretations)):
            if events is not None and interpretation is not None:
                valid_indices.append(i)
        
        self.sensor_events_list = [sensor_events_list[i] for i in valid_indices]
        self.sensor_interpretations = [sensor_interpretations[i] for i in valid_indices]
        self.activities = [activities[i] for i in valid_indices]
        self.max_events = max_events
        
        print(f"✓ SensorEncoderDataset created with {len(self.sensor_events_list)} valid samples")
    
    def __len__(self):
        return len(self.sensor_events_list)
    
    def __getitem__(self, idx):
        return {
            'sensor_events': self.sensor_events_list[idx],
            'sensor_interpretation': self.sensor_interpretations[idx],
            'activity': self.activities[idx]
        }


class SensorEncoderPhase2Trainer:
    """Phase 2 trainer for sensor encoder using text encoder interpretations"""
    
    def __init__(self, config: SemanticHARConfig, text_encoder):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize sensor encoder
        self.sensor_encoder = SensorEncoder(config).to(self.device)
        self.text_encoder = text_encoder  # Pre-trained text encoder (frozen)
        
        # Freeze text encoder parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Optimizer for sensor encoder only
        self.optimizer = torch.optim.AdamW(
            self.sensor_encoder.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        print(f"✓ SensorEncoderPhase2Trainer initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Text encoder frozen: {not any(p.requires_grad for p in self.text_encoder.parameters())}")
        print(f"   - Sensor encoder trainable: {any(p.requires_grad for p in self.sensor_encoder.parameters())}")
    
    def train_step(self, sensor_events: Dict[str, torch.Tensor], 
                   sensor_interpretations: List[str]) -> Dict[str, float]:
        """Single training step for Phase 2"""
        
        self.optimizer.zero_grad()
        
        # Sensor data encoding (trainable)
        sensor_embeddings = self.sensor_encoder(sensor_events)
        
        # Text encoding (frozen text encoder)
        with torch.no_grad():
            text_embeddings = self.text_encoder(sensor_interpretations)
        
        # Contrastive learning loss
        contrastive_loss = self._compute_contrastive_loss(
            sensor_embeddings, text_embeddings
        )
        
        # Backpropagation
        contrastive_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.sensor_encoder.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'contrastive_loss': contrastive_loss.item()
        }
    
    def _compute_contrastive_loss(self, sensor_embeddings: torch.Tensor, 
                                 text_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute contrastive learning loss between sensor and text embeddings"""
        # Normalize embeddings
        sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        
        # Similarity matrix
        similarity_matrix = torch.matmul(sensor_embeddings, text_embeddings.T) / self.config.temperature
        
        # Batch size
        batch_size = sensor_embeddings.size(0)
        
        # Labels (diagonal is the correct match)
        labels = torch.arange(batch_size, device=sensor_embeddings.device)
        
        # Symmetric loss
        loss_s2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2s = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_s2t + loss_t2s) / 2
    
    def train_with_interpretations(self, interpretations_file: str,
                                 sensor_data_file: str = None,
                                 num_epochs: int = 50,
                                 batch_size: int = 16,
                                 early_stopping: bool = True,
                                 patience: int = 10) -> 'SensorEncoder':
        """Train sensor encoder using text encoder interpretations"""
        
        print("=" * 60)
        print("Phase 2: Sensor Encoder Training")
        print("=" * 60)
        print("Training sensor encoder to match text encoder interpretations...")
        
        # Load training data
        train_dataset = self._prepare_training_data(interpretations_file, splits=['train'])
        val_dataset = self._prepare_training_data(interpretations_file, splits=['val'])
        
        if len(train_dataset) == 0:
            raise ValueError("No training data found")
        
        print(f"Training dataset: {len(train_dataset)} samples")
        print(f"Validation dataset: {len(val_dataset)} samples")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {num_epochs}")
        
        # Create data loaders with custom collate function
        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                     collate_fn=self._collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                   collate_fn=self._collate_fn)
        
        # Training tracking
        train_losses = []
        val_losses = []
        best_loss = float('inf')
        patience_counter = 0
        
        print(f"\nStarting training with improved monitoring...")
        print("=" * 50)
        
        for epoch in range(num_epochs):
            # Training phase
            self.sensor_encoder.train()
            epoch_losses = []
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 30)
            
            for batch_idx, batch in enumerate(train_dataloader):
                try:
                    # Get batch data
                    sensor_events = batch['sensor_events']
                    sensor_interpretations = batch['sensor_interpretation']
                    
                    # Move to device
                    for key in sensor_events:
                        if isinstance(sensor_events[key], torch.Tensor):
                            sensor_events[key] = sensor_events[key].to(self.device)
                    
                    # Training step
                    losses = self.train_step(sensor_events, sensor_interpretations)
                    epoch_losses.append(losses['contrastive_loss'])
                    
                    # Progress indicator
                    if (batch_idx + 1) % 5 == 0:
                        current_loss = np.mean(epoch_losses[-5:])
                        print(f"  Batch {batch_idx+1}/{len(train_dataloader)} - Loss: {current_loss:.4f}")
                    
                except Exception as e:
                    print(f"⨺ Training batch error: {e}")
                    continue
            
            if epoch_losses:
                avg_train_loss = np.mean(epoch_losses)
                train_losses.append(avg_train_loss)
                print(f"✓ Training Loss: {avg_train_loss:.4f}")
                
                # Validation phase with detailed monitoring
                if early_stopping and len(val_dataset) > 0:
                    print(f"  Validating on {len(val_dataset)} samples...")
                    val_loss = self._validate(val_dataloader)
                    
                    if val_loss is not None:
                        val_losses.append(val_loss)
                        print(f"✓ Validation Loss: {val_loss:.4f}")
                        
                        # Learning rate monitoring
                        current_lr = self.optimizer.param_groups[0]['lr']
                        print(f"  Learning Rate: {current_lr:.6f}")
                        
                        # Loss improvement analysis
                        if len(val_losses) > 1:
                            loss_change = val_losses[-1] - val_losses[-2]
                            print(f"  Loss Change: {loss_change:+.4f}")
                            
                            if loss_change > 0:
                                print(f"  ⚠️  Loss increased by {loss_change:.4f}")
                            else:
                                print(f"  ✓ Loss decreased by {abs(loss_change):.4f}")
                        
                        # Early stopping logic with detailed feedback
                        if val_loss < best_loss:
                            improvement = best_loss - val_loss
                            best_loss = val_loss
                            patience_counter = 0
                            
                            print(f"  🎯 New best validation loss: {best_loss:.4f} (improvement: {improvement:.4f})")
                            
                            # Save best model
                            import os
                            os.makedirs("checkpoints", exist_ok=True)
                            best_model_path = "checkpoints/sensor_encoder_best.pth"
                            torch.save(self.sensor_encoder.state_dict(), best_model_path)
                            print(f"  💾 Best model saved: {best_model_path}")
                        else:
                            patience_counter += 1
                            print(f"  ⏳ No improvement ({patience_counter}/{patience})")
                            
                            if patience_counter >= patience:
                                print(f"  🛑 Early stopping triggered at epoch {epoch+1}")
                                print(f"  📊 Best validation loss: {best_loss:.4f}")
                                break
                    else:
                        print(f"  ⨺ Validation failed - no valid batches")
                else:
                    print(f"  ⏭️  Skipping validation (early_stopping={early_stopping}, val_samples={len(val_dataset)})")
                
                # Update learning rate
                self.scheduler.step()
                
                # Training progress summary
                if len(train_losses) > 1:
                    train_improvement = train_losses[-2] - train_losses[-1]
                    print(f"  📈 Training improvement: {train_improvement:+.4f}")
            else:
                print(f"⨺ Epoch {epoch+1}/{num_epochs} - No valid training batches")
        
        print("\n" + "=" * 60)
        print("Sensor Encoder Phase 2 Training Completed!")
        print("=" * 60)
        
        # Training summary
        if train_losses:
            print(f"📊 Training Summary:")
            print(f"   Initial Loss: {train_losses[0]:.4f}")
            print(f"   Final Loss:   {train_losses[-1]:.4f}")
            print(f"   Total Improvement: {train_losses[0] - train_losses[-1]:.4f}")
            
            if val_losses:
                print(f"   Best Validation Loss: {best_loss:.4f}")
                print(f"   Validation Improvement: {val_losses[0] - best_loss:.4f}")
        
        # Create training curves visualization
        if len(train_losses) > 1:
            try:
                from models.sensor_encoder import SensorEncoderEvaluator
                evaluator = SensorEncoderEvaluator(self.config, self.sensor_encoder, self.text_encoder)
                evaluator.create_training_curves(train_losses, val_losses)
            except Exception as e:
                print(f"⨺ Could not create training curves: {e}")
        
        # Save final model
        import os
        os.makedirs("checkpoints", exist_ok=True)
        final_model_path = "checkpoints/sensor_encoder_trained.pth"
        torch.save(self.sensor_encoder.state_dict(), final_model_path)
        print(f"✓ Final sensor encoder saved: {final_model_path}")
        
        return self.sensor_encoder
    
    def _prepare_training_data(self, interpretations_file: str, splits: List[str]) -> SensorEncoderDataset:
        """Prepare training data for sensor encoder from actual ambient sensor data"""
        import json
        import os
        
        print(f"Loading sensor encoder training data from {interpretations_file}...")
        
        # Load interpretations
        with open(interpretations_file, 'r', encoding='utf-8') as f:
            interpretations_data = json.load(f)
        
        # Load time windows with actual sensor events
        time_windows_file = os.path.join(os.path.dirname(interpretations_file), 'time_windows.json')
        if not os.path.exists(time_windows_file):
            raise FileNotFoundError(f"Time windows file not found: {time_windows_file}")
        
        with open(time_windows_file, 'r', encoding='utf-8') as f:
            time_windows_data = json.load(f)
        
        sensor_events_list = []
        sensor_interpretations = []
        activities = []
        
        # Extract data from specified splits
        for home_id in interpretations_data['sensor_interpretations']:
            if home_id not in time_windows_data['time_windows']:
                print(f"⚠️ Warning: {home_id} not found in time_windows.json")
                continue
            
            for split in interpretations_data['sensor_interpretations'][home_id]:
                if split not in splits:
                    continue
                
                # Get windows for this split
                interpretation_windows = interpretations_data['sensor_interpretations'][home_id][split]
                time_windows = time_windows_data['time_windows'][home_id][split]
                
                # Create a mapping from window_id to time_window
                window_id_to_time_window = {}
                for tw in time_windows:
                    window_id = tw['window_id']
                    window_id_to_time_window[window_id] = tw
                
                # Process each interpretation window
                for window_key, window_data in interpretation_windows.items():
                    if 'interpretation' not in window_data or 'error' in window_data:
                        continue
                    
                    # Find corresponding time window
                    # window_key format: "window_1", need to find matching window_id
                    matching_time_window = None
                    for window_id, tw in window_id_to_time_window.items():
                        # Match based on activity and window number
                        if window_key in window_id or f"_{window_key.split('_')[1]}" in window_id:
                            matching_time_window = tw
                            break
                    
                    if matching_time_window is None:
                        # Try to match by index
                        try:
                            window_idx = int(window_key.split('_')[1]) - 1
                            if 0 <= window_idx < len(time_windows):
                                matching_time_window = time_windows[window_idx]
                        except (IndexError, ValueError):
                            continue
                    
                    if matching_time_window is None:
                        continue
                    
                    # Extract sensor events from time window
                    sensor_events = self._extract_sensor_events(matching_time_window)
                    
                    if sensor_events is not None:
                        sensor_events_list.append(sensor_events)
                        sensor_interpretations.append(window_data['interpretation'])
                        activities.append(window_data['activity'])
        
        print(f"✓ Loaded {len(sensor_events_list)} samples for splits: {splits}")
        
        return SensorEncoderDataset(sensor_events_list, sensor_interpretations, activities)
    
    def _extract_sensor_events(self, time_window: Dict) -> Optional[Dict]:
        """Extract sensor events from a time window"""
        try:
            # Get sensor event details
            sensor_types = time_window.get('sensor_types', [])
            locations = time_window.get('locations', [])
            places = time_window.get('places', [])
            
            # Get event details if available
            events = time_window.get('events', [])
            
            if not sensor_types:
                return None
            
            # If no detailed events, create from summary
            if not events:
                num_events = len(sensor_types)
                events = []
                for i in range(num_events):
                    events.append({
                        'sensor_type': sensor_types[i] if i < len(sensor_types) else 'UNK',
                        'location': locations[i] if i < len(locations) else 'UNK',
                        'place': places[i] if i < len(places) else 'UNK',
                        'duration': 1.0,  # Default duration
                        'time_delta': 1.0 if i > 0 else 0.0  # Default time delta
                    })
            
            return {
                'events': events,
                'activity': time_window.get('activity', 'Unknown')
            }
        except Exception as e:
            print(f"⚠️ Error extracting sensor events: {e}")
            return None
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Collate function for batching sensor events"""
        max_events = 50  # Maximum number of events per window
        
        sensor_events_batch = []
        interpretations = []
        activities = []
        
        for item in batch:
            sensor_events_batch.append(item['sensor_events'])
            interpretations.append(item['sensor_interpretation'])
            activities.append(item['activity'])
        
        # Process sensor events into tensors
        batch_size = len(sensor_events_batch)
        
        # Initialize tensors
        sensor_types_tensor = torch.zeros(batch_size, max_events, dtype=torch.long)
        locations_tensor = torch.zeros(batch_size, max_events, dtype=torch.long)
        places_tensor = torch.zeros(batch_size, max_events, dtype=torch.long)
        durations_tensor = torch.zeros(batch_size, max_events, dtype=torch.float)
        time_deltas_tensor = torch.zeros(batch_size, max_events, dtype=torch.float)
        mask_tensor = torch.ones(batch_size, max_events, dtype=torch.bool)  # True = padding
        
        # Collect all unique locations and places for vocabulary update
        all_locations = []
        all_places = []
        
        for i, sensor_events in enumerate(sensor_events_batch):
            events = sensor_events['events']
            num_events = min(len(events), max_events)
            
            for j, event in enumerate(events[:num_events]):
                # Sensor type
                sensor_type = event.get('sensor_type', 'UNK')
                sensor_types_tensor[i, j] = self.sensor_encoder.sensor_type_to_idx.get(sensor_type, 
                    self.sensor_encoder.sensor_type_to_idx['UNK'])
                
                # Location
                location = event.get('location', 'UNK')
                all_locations.append(location)
                locations_tensor[i, j] = self.sensor_encoder.location_to_idx.get(location, 0)
                
                # Place
                place = event.get('place', 'UNK')
                all_places.append(place)
                places_tensor[i, j] = self.sensor_encoder.place_to_idx.get(place, 0)
                
                # Temporal features
                durations_tensor[i, j] = event.get('duration', 1.0)
                time_deltas_tensor[i, j] = event.get('time_delta', 0.0)
                
                # Mark as non-padding
                mask_tensor[i, j] = False
        
        # Update vocabulary
        self.sensor_encoder._update_vocab(all_locations, all_places)
        
        # Re-map locations and places with updated vocabulary
        for i, sensor_events in enumerate(sensor_events_batch):
            events = sensor_events['events']
            num_events = min(len(events), max_events)
            
            for j, event in enumerate(events[:num_events]):
                location = event.get('location', 'UNK')
                locations_tensor[i, j] = self.sensor_encoder.location_to_idx.get(location, 0)
                
                place = event.get('place', 'UNK')
                places_tensor[i, j] = self.sensor_encoder.place_to_idx.get(place, 0)
        
        # Normalize temporal features
        if durations_tensor.sum() > 0:
            durations_tensor = torch.log1p(durations_tensor)  # Log normalization
            durations_tensor = durations_tensor / (durations_tensor.max() + 1e-8)
        
        if time_deltas_tensor.sum() > 0:
            time_deltas_tensor = torch.log1p(time_deltas_tensor)
            time_deltas_tensor = time_deltas_tensor / (time_deltas_tensor.max() + 1e-8)
        
        return {
            'sensor_events': {
                'sensor_types': sensor_types_tensor,
                'locations': locations_tensor,
                'places': places_tensor,
                'durations': durations_tensor,
                'time_deltas': time_deltas_tensor,
                'mask': mask_tensor
            },
            'sensor_interpretation': interpretations,
            'activity': activities
        }
    
    def _validate(self, val_dataloader) -> float:
        """Validation step"""
        self.sensor_encoder.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                try:
                    # Move sensor events to device
                    sensor_events = {}
                    for key, value in batch['sensor_events'].items():
                        if isinstance(value, torch.Tensor):
                            sensor_events[key] = value.to(self.device)
                        else:
                            sensor_events[key] = value
                    
                    sensor_interpretations = batch['sensor_interpretation']
                    
                    # Forward pass
                    sensor_embeddings = self.sensor_encoder(sensor_events)
                    text_embeddings = self.text_encoder(sensor_interpretations)
                    
                    # Compute loss
                    loss = self._compute_contrastive_loss(sensor_embeddings, text_embeddings)
                    val_losses.append(loss.item())
                    
                except Exception as e:
                    print(f"⨺ Validation batch error: {e}")
                    continue
        
        self.sensor_encoder.train()
        
        if val_losses:
            return np.mean(val_losses)
        else:
            return None
    
    def save_model(self, path: str):
        """Save sensor encoder model"""
        torch.save({
            'sensor_encoder': self.sensor_encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, path)
    
    def load_model(self, path: str):
        """Load sensor encoder model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.sensor_encoder.load_state_dict(checkpoint['sensor_encoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])


class SensorEncoderEvaluator:
    """Sensor encoder evaluation and visualization"""
    
    def __init__(self, config: SemanticHARConfig, sensor_encoder, text_encoder):
        self.config = config
        self.device = torch.device(config.device)
        self.sensor_encoder = sensor_encoder
        self.text_encoder = text_encoder
        
        # Set models to eval mode
        self.sensor_encoder.eval()
        self.text_encoder.eval()
        
        print(f"✓ SensorEncoderEvaluator initialized")
    
    def evaluate_alignment_quality(self, sensor_events_list: List[Dict], 
                                  sensor_interpretations: List[str],
                                  trainer) -> Dict[str, float]:
        """Evaluate sensor-text alignment quality"""
        print("⨠ Evaluating sensor-text alignment quality...")
        
        with torch.no_grad():
            # Get sensor embeddings
            sensor_embeddings_list = []
            for sensor_events in sensor_events_list:
                # Create a mini-batch of size 1
                batch = trainer._collate_fn([{'sensor_events': sensor_events, 
                                             'sensor_interpretation': '', 
                                             'activity': ''}])
                
                # Move to device
                sensor_events_batch = {}
                for key in batch['sensor_events']:
                    if isinstance(batch['sensor_events'][key], torch.Tensor):
                        sensor_events_batch[key] = batch['sensor_events'][key].to(self.device)
                    else:
                        sensor_events_batch[key] = batch['sensor_events'][key]
                
                embedding = self.sensor_encoder(sensor_events_batch)
                sensor_embeddings_list.append(embedding)
            
            sensor_embeddings = torch.cat(sensor_embeddings_list, dim=0)
            
            # Get text embeddings
            text_embeddings = self.text_encoder(sensor_interpretations)
            
            # Normalize embeddings
            sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
            
            # Calculate similarity matrix
            similarity_matrix = torch.matmul(sensor_embeddings, text_embeddings.T)
            
            # Calculate metrics
            diagonal_similarities = torch.diag(similarity_matrix)
            max_similarities, max_indices = torch.max(similarity_matrix, dim=1)
            
            # Accuracy (correct matches)
            correct_predictions = (max_indices == torch.arange(len(sensor_interpretations), device=self.device)).float()
            accuracy = correct_predictions.mean().item()
            
            # Average correct pair similarity
            avg_correct_similarity = diagonal_similarities.mean().item()
            
            # Average incorrect similarity
            off_diagonal_mask = ~torch.eye(len(similarity_matrix), device=self.device).bool()
            off_diagonal_similarities = similarity_matrix[off_diagonal_mask]
            avg_incorrect_similarity = off_diagonal_similarities.mean().item()
            
            margin = avg_correct_similarity - avg_incorrect_similarity
        
        return {
            'accuracy': accuracy,
            'avg_correct_similarity': avg_correct_similarity,
            'avg_incorrect_similarity': avg_incorrect_similarity,
            'margin': margin,
            'diagonal_similarities': diagonal_similarities.detach().cpu().numpy(),
            'similarity_matrix': similarity_matrix.detach().cpu().numpy()
        }
    
    def visualize_embeddings(self, sensor_events_list: List[Dict], 
                            sensor_interpretations: List[str],
                            activities: List[str],
                            trainer,
                            save_path: str = "outputs/sensor_encoder_embeddings.png"):
        """Visualize sensor and text embeddings"""
        print("⨠ Visualizing sensor encoder embeddings...")
        
        with torch.no_grad():
            # Get sensor embeddings
            sensor_embeddings_list = []
            for sensor_events in sensor_events_list:
                # Create a mini-batch of size 1
                batch = trainer._collate_fn([{'sensor_events': sensor_events, 
                                             'sensor_interpretation': '', 
                                             'activity': ''}])
                
                # Move to device
                sensor_events_batch = {}
                for key in batch['sensor_events']:
                    if isinstance(batch['sensor_events'][key], torch.Tensor):
                        sensor_events_batch[key] = batch['sensor_events'][key].to(self.device)
                    else:
                        sensor_events_batch[key] = batch['sensor_events'][key]
                
                embedding = self.sensor_encoder(sensor_events_batch)
                sensor_embeddings_list.append(embedding)
            
            sensor_embeddings = torch.cat(sensor_embeddings_list, dim=0)
            text_embeddings = self.text_encoder(sensor_interpretations)
        
        # t-SNE visualization
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        # Combine embeddings
        all_embeddings = torch.cat([sensor_embeddings, text_embeddings], dim=0)
        all_embeddings_np = all_embeddings.detach().cpu().numpy()
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings_np)-1))
        embeddings_2d = tsne.fit_transform(all_embeddings_np)
        
        # Plot
        plt.figure(figsize=(15, 10))
        
        # Sensor embeddings
        sensor_2d = embeddings_2d[:len(sensor_embeddings)]
        scatter1 = plt.scatter(sensor_2d[:, 0], sensor_2d[:, 1], 
                              c='blue', alpha=0.6, s=50, label='Sensor Embeddings')
        
        # Text embeddings
        text_2d = embeddings_2d[len(sensor_embeddings):]
        scatter2 = plt.scatter(text_2d[:, 0], text_2d[:, 1], 
                              c='red', alpha=0.6, s=50, label='Text Embeddings')
        
        # Draw connections between matched pairs
        for i in range(len(sensor_embeddings)):
            plt.plot([sensor_2d[i, 0], text_2d[i, 0]], 
                    [sensor_2d[i, 1], text_2d[i, 1]], 
                    'k--', alpha=0.3, linewidth=0.5)
        
        plt.title('Sensor Encoder Embeddings Visualization')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Embedding visualization saved: {save_path}")
    
    def analyze_activity_performance(self, sensor_events_list: List[Dict], 
                                   sensor_interpretations: List[str],
                                   activities: List[str],
                                   trainer) -> Dict:
        """Analyze performance by activity type"""
        print("⨠ Analyzing performance by activity...")
        
        with torch.no_grad():
            # Get embeddings
            sensor_embeddings_list = []
            for sensor_events in sensor_events_list:
                # Create a mini-batch of size 1
                batch = trainer._collate_fn([{'sensor_events': sensor_events, 
                                             'sensor_interpretation': '', 
                                             'activity': ''}])
                
                # Move to device
                sensor_events_batch = {}
                for key in batch['sensor_events']:
                    if isinstance(batch['sensor_events'][key], torch.Tensor):
                        sensor_events_batch[key] = batch['sensor_events'][key].to(self.device)
                    else:
                        sensor_events_batch[key] = batch['sensor_events'][key]
                
                embedding = self.sensor_encoder(sensor_events_batch)
                sensor_embeddings_list.append(embedding)
            
            sensor_embeddings = torch.cat(sensor_embeddings_list, dim=0)
            text_embeddings = self.text_encoder(sensor_interpretations)
            
            # Normalize
            sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
            
            # Calculate similarities
            similarities = torch.sum(sensor_embeddings * text_embeddings, dim=1)
            
            # Per-activity analysis
            activity_stats = {}
            unique_activities = list(set(activities))
            
            for activity in unique_activities:
                activity_indices = [i for i, a in enumerate(activities) if a == activity]
                if len(activity_indices) > 0:
                    activity_similarities = similarities[activity_indices]
                    activity_stats[activity] = {
                        'count': len(activity_indices),
                        'avg_similarity': activity_similarities.mean().item(),
                        'std_similarity': activity_similarities.std().item(),
                        'min_similarity': activity_similarities.min().item(),
                        'max_similarity': activity_similarities.max().item()
                    }
        
        return {
            'overall_avg_similarity': similarities.mean().item(),
            'overall_std_similarity': similarities.std().item(),
            'activity_stats': activity_stats
        }
    
    def create_training_curves(self, train_losses: List[float], 
                             val_losses: List[float] = None,
                             save_path: str = "outputs/sensor_encoder_training_curves.png"):
        """Create training curves visualization"""
        print("⨠ Creating training curves...")
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        
        if val_losses:
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.title('Sensor Encoder Training Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add annotations for best validation loss
        if val_losses:
            best_epoch = np.argmin(val_losses) + 1
            best_loss = min(val_losses)
            plt.annotate(f'Best Val Loss: {best_loss:.4f} at Epoch {best_epoch}',
                        xy=(best_epoch, best_loss), xytext=(best_epoch + 5, best_loss + 0.1),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
        
        # Save
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training curves saved: {save_path}")
    
    def comprehensive_evaluation(self, interpretations_file: str, 
                               output_dir: str = "outputs") -> Dict:
        """Comprehensive evaluation of sensor encoder"""
        print("⨠ Starting comprehensive sensor encoder evaluation...")
        
        # Load test data using trainer's data preparation method
        import json
        import os
        
        # Create a temporary trainer instance to use its data preparation methods
        from models.sensor_encoder import SensorEncoderPhase2Trainer
        temp_trainer = SensorEncoderPhase2Trainer(self.config, self.text_encoder)
        temp_trainer.sensor_encoder = self.sensor_encoder
        
        # Load test dataset
        test_dataset = temp_trainer._prepare_training_data(interpretations_file, splits=['test'])
        
        # Limit data for evaluation
        max_samples = min(50, len(test_dataset))
        
        # Extract samples
        sensor_events_list = []
        sensor_interpretations = []
        activities = []
        
        for i in range(max_samples):
            sample = test_dataset[i]
            sensor_events_list.append(sample['sensor_events'])
            sensor_interpretations.append(sample['sensor_interpretation'])
            activities.append(sample['activity'])
        
        print(f"Evaluating on {len(sensor_events_list)} test samples")
        
        # 1. Alignment quality evaluation
        alignment_results = self.evaluate_alignment_quality(
            sensor_events_list, sensor_interpretations, temp_trainer
        )
        
        # 2. Activity performance analysis
        activity_results = self.analyze_activity_performance(
            sensor_events_list, sensor_interpretations, activities, temp_trainer
        )
        
        # 3. Visualizations
        self.visualize_embeddings(
            sensor_events_list, sensor_interpretations, activities, temp_trainer,
            os.path.join(output_dir, "sensor_encoder_embeddings.png")
        )
        
        # 4. Comprehensive results
        results = {
            'alignment_quality': alignment_results,
            'activity_performance': activity_results,
            'evaluation_summary': {
                'accuracy': alignment_results['accuracy'],
                'margin': alignment_results['margin'],
                'overall_similarity': activity_results['overall_avg_similarity'],
                'test_samples': len(sensor_events_list)
            }
        }
        
        # Save results
        import os
        results_file = os.path.join(output_dir, "sensor_encoder_evaluation_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✓ Evaluation results saved: {results_file}")
        
        # Print results
        self.print_evaluation_results(results)
        
        return results
    
    def print_evaluation_results(self, results: Dict):
        """Print evaluation results in formatted table"""
        print("\n" + "="*80)
        print("SENSOR ENCODER EVALUATION RESULTS")
        print("="*80)
        
        summary = results['evaluation_summary']
        alignment = results['alignment_quality']
        activity = results['activity_performance']
        
        print(f"\nMAIN METRICS:")
        print(f"   Accuracy:                    {summary['accuracy']:.3f}")
        print(f"   Similarity Margin:          {summary['margin']:.3f}")
        print(f"   Overall Similarity:         {summary['overall_similarity']:.3f}")
        print(f"   Test Samples:              {summary['test_samples']}")
        
        print(f"\nALIGNMENT QUALITY:")
        print(f"   Correct Pair Similarity:    {alignment['avg_correct_similarity']:.3f}")
        print(f"   Incorrect Pair Similarity: {alignment['avg_incorrect_similarity']:.3f}")
        print(f"   Similarity Margin:         {alignment['margin']:.3f}")
        
        print(f"\nACTIVITY-SPECIFIC PERFORMANCE:")
        activity_stats = activity['activity_stats']
        for activity, stats in activity_stats.items():
            print(f"   {activity:15}: {stats['avg_similarity']:.3f} ± {stats['std_similarity']:.3f} (n={stats['count']})")
        
        # Quality assessment
        print(f"\nQUALITY ASSESSMENT:")
        accuracy = summary['accuracy']
        margin = summary['margin']
        overall_sim = summary['overall_similarity']
        
        if accuracy > 0.8 and margin > 0.5 and overall_sim > 0.7:
            print("   EXCELLENT: Sensor encoder training is very successful!")
        elif accuracy > 0.6 and margin > 0.3 and overall_sim > 0.5:
            print("   GOOD: Sensor encoder training is successful.")
        elif accuracy > 0.4 and margin > 0.2 and overall_sim > 0.3:
            print("   FAIR: Sensor encoder training shows some improvement.")
        else:
            print("   POOR: Sensor encoder training needs more work.")
        
        print("="*80)
