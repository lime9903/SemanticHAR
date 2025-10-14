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


class SensorEncoderDataset:
    """Dataset for sensor encoder training with text encoder - Ambient sensor events"""
    
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


class SensorEncoderTrainer:
    """Sensor encoder trainer using text encoder for contrastive learning"""
    
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
        
        print(f"✓ SensorEncoderTrainer initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Text encoder frozen: {not any(p.requires_grad for p in self.text_encoder.parameters())}")
        print(f"   - Sensor encoder trainable: {any(p.requires_grad for p in self.sensor_encoder.parameters())}")
    
    def train_step(self, sensor_events: Dict[str, torch.Tensor], 
                   sensor_interpretations: List[str]) -> Dict[str, float]:
        """Single training step using contrastive learning with text encoder"""
        
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
        print("Sensor Encoder Training")
        print("=" * 60)
        print("Training sensor encoder to align with text encoder interpretations...")
        print("⚠️  Using home_b TEST data for sensor encoder training")
        print("    (home_b used for sensor encoder, home_a reserved for final inference)")
        
        # Use home_b test data (different home from text encoder)
        all_data = self._prepare_training_data(interpretations_file, splits=['test'], home_filter=['home_b'])
        
        # Split data into train/val for sensor encoder (90/10 split)
        total_samples = len(all_data)
        train_size = int(total_samples * 0.90)
        
        # Shuffle indices for better generalization
        import random
        indices = list(range(total_samples))
        random.seed(42)  # For reproducibility
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create train/val datasets
        train_dataset = self._create_subset_dataset(all_data, train_indices)
        val_dataset = self._create_subset_dataset(all_data, val_indices)
        
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
        print("Sensor Encoder Training Completed!")
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
    
    def _prepare_training_data(self, interpretations_file: str, splits: List[str], 
                               home_filter: Optional[List[str]] = None) -> SensorEncoderDataset:
        """Prepare training data for sensor encoder from actual ambient sensor data"""
        import json
        import os
        import pandas as pd
        from dataloader.data_loader import load_sensor_data
        
        print(f"Loading sensor encoder training data from {interpretations_file}...")
        if home_filter:
            print(f"  Filtering for homes: {home_filter}")
        
        # Load interpretations
        with open(interpretations_file, 'r', encoding='utf-8') as f:
            interpretations_data = json.load(f)
        
        # Load actual raw sensor data
        print("  Loading raw sensor data from dataset...")
        sensor_data_dict = load_sensor_data(
            self.config, 
            self.config.dataset_name,
            window_size_seconds=self.config.window_size_seconds,
            overlap_ratio=self.config.overlap_ratio
        )
        
        sensor_events_list = []
        sensor_interpretations = []
        activities = []
        
        # Extract data from specified splits
        for home_id in interpretations_data['sensor_interpretations']:
            # Apply home filter if specified
            if home_filter and home_id not in home_filter:
                continue
                
            if home_id not in sensor_data_dict:
                print(f"⚠️ Warning: {home_id} not found in sensor data")
                continue
            
            for split in interpretations_data['sensor_interpretations'][home_id]:
                if split not in splits:
                    continue
                
                # Get windows for this split
                interpretation_windows = interpretations_data['sensor_interpretations'][home_id][split]
                raw_windows = sensor_data_dict[home_id].get(split, [])
                
                # Process each interpretation window
                for window_key, window_data in interpretation_windows.items():
                    if 'interpretation' not in window_data or 'error' in window_data:
                        continue
                    
                    # Match by index
                    try:
                        window_idx = int(window_key.split('_')[1]) - 1
                        if 0 <= window_idx < len(raw_windows):
                            raw_window_df = raw_windows[window_idx]
                            
                            # Extract sensor events from DataFrame
                            sensor_events = self._extract_sensor_events_from_df(raw_window_df)
                            
                            if sensor_events is not None and len(sensor_events['events']) > 0:
                                sensor_events_list.append(sensor_events)
                            sensor_interpretations.append(window_data['interpretation'])
                            activities.append(window_data['activity'])
                    except (IndexError, ValueError, Exception) as e:
                        continue
        
        print(f"✓ Loaded {len(sensor_events_list)} samples for splits: {splits}")
        
        return SensorEncoderDataset(sensor_events_list, sensor_interpretations, activities)
    
    def _create_subset_dataset(self, dataset: SensorEncoderDataset, indices: List[int]) -> SensorEncoderDataset:
        """Create a subset dataset from indices"""
        subset_events = [dataset.sensor_events_list[i] for i in indices]
        subset_interpretations = [dataset.sensor_interpretations[i] for i in indices]
        subset_activities = [dataset.activities[i] for i in indices]
        
        return SensorEncoderDataset(subset_events, subset_interpretations, subset_activities)
    
    def _extract_sensor_events_from_df(self, window_df) -> Optional[Dict]:
        """Extract sensor events from DataFrame with actual temporal information"""
        import pandas as pd
        
        try:
            if window_df is None or len(window_df) == 0:
                return None
            
            # Convert to DataFrame if it's a list
            if isinstance(window_df, list):
                if len(window_df) == 0:
                    return None
                window_df = pd.DataFrame(window_df)
            
            # Sort by start_time
            if 'start_time' in window_df.columns:
                window_df = window_df.sort_values('start_time')
            
            events = []
            prev_time = None
            
            for idx, row in window_df.iterrows():
                # Extract sensor information
                sensor_type = str(row.get('type', 'UNK')).strip()
                location = str(row.get('location', 'UNK')).strip()
                place = str(row.get('place', 'UNK')).strip()
                
                # Calculate duration
                try:
                    if pd.notna(row.get('start_time')) and pd.notna(row.get('end_time')):
                        start_time = pd.to_datetime(row['start_time'])
                        end_time = pd.to_datetime(row['end_time'])
                        duration = (end_time - start_time).total_seconds()
                        
                        # Calculate time since last event
                        if prev_time is not None:
                            time_delta = (start_time - prev_time).total_seconds()
                        else:
                            time_delta = 0.0
                        
                        prev_time = end_time
                    else:
                        duration = 1.0
                        time_delta = 1.0 if len(events) > 0 else 0.0
                except Exception:
                    duration = 1.0
                    time_delta = 1.0 if len(events) > 0 else 0.0
                
                # Ensure positive values
                duration = max(0.1, abs(duration))
                time_delta = max(0.0, abs(time_delta))
                
                events.append({
                    'sensor_type': sensor_type,
                    'location': location,
                    'place': place,
                    'duration': duration,
                    'time_delta': time_delta
                })
            
            if len(events) == 0:
                return None
            
            return {
                'events': events,
                'activity': 'Unknown'
            }
            
        except Exception as e:
            print(f"⚠️ Error extracting sensor events from DataFrame: {e}")
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
        from models.sensor_encoder import SensorEncoderTrainer
        temp_trainer = SensorEncoderTrainer(self.config, self.text_encoder)
        temp_trainer.sensor_encoder = self.sensor_encoder
        
        # Load home_a data for final evaluation
        # home_a is completely unseen by sensor encoder
        print("⚠️  Using home_a ALL data for evaluation (completely unseen environment)")
        test_dataset = temp_trainer._prepare_training_data(
            interpretations_file, 
            splits=['train', 'val', 'test'],
            home_filter=['home_a']
        )
        
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


class SensorEncoderInference:
    """Inference engine for sensor encoder to predict activities from raw sensor data"""
    
    def __init__(self, config: SemanticHARConfig, sensor_encoder, text_encoder):
        self.config = config
        self.device = torch.device(config.device)
        self.sensor_encoder = sensor_encoder
        self.text_encoder = text_encoder
        
        # Set models to eval mode
        self.sensor_encoder.eval()
        self.text_encoder.eval()
        
        print(f"✓ SensorEncoderInference initialized")
    
    def predict_activities(self, interpretations_file: str, 
                          output_dir: str = "outputs") -> Dict:
        """
        Predict activities from raw sensor data using trained models
        
        Args:
            interpretations_file: Path to semantic interpretations file
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing predictions and evaluation metrics
        """
        print("\n" + "=" * 80)
        print("SENSOR ENCODER INFERENCE - Activity Prediction")
        print("=" * 80)
        print("Using home_a ALL data (completely unseen environment by sensor encoder)")
        
        # Load activity label interpretations (from text encoder training)
        activity_interpretations = self._load_activity_interpretations(interpretations_file)
        
        # Generate activity label embeddings
        print("\nGenerating activity label embeddings...")
        activity_embeddings = self._generate_activity_embeddings(activity_interpretations)
        
        # Load test sensor data (train split - unseen by both models)
        print("\nLoading test sensor data from TRAIN split...")
        test_data = self._load_test_data(interpretations_file)
        
        if len(test_data['sensor_events']) == 0:
            print("⨺ No test data found!")
            return None
        
        print(f"✓ Loaded {len(test_data['sensor_events'])} test samples")
        
        # Predict activities
        print("\nPredicting activities...")
        predictions = self._predict_batch(
            test_data['sensor_events'],
            activity_embeddings,
            list(activity_interpretations.keys())
        )
        
        # Evaluate predictions
        print("\nEvaluating predictions...")
        evaluation_results = self._evaluate_predictions(
            predictions,
            test_data['true_activities'],
            list(activity_interpretations.keys())
        )
        
        # Save results
        self._save_results(evaluation_results, output_dir)
        
        # Print results
        self._print_results(evaluation_results)
        
        return evaluation_results
    
    def _load_activity_interpretations(self, interpretations_file: str) -> Dict[str, str]:
        """Load activity label interpretations"""
        import json
        
        with open(interpretations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        activity_interpretations = {}
        for activity, activity_data in data.get('activity_interpretations', {}).items():
            if 'interpretation' in activity_data and 'error' not in activity_data:
                activity_interpretations[activity] = activity_data['interpretation']
        
        print(f"✓ Loaded {len(activity_interpretations)} activity label interpretations")
        return activity_interpretations
    
    def _generate_activity_embeddings(self, activity_interpretations: Dict[str, str]) -> torch.Tensor:
        """Generate embeddings for all activity labels"""
        with torch.no_grad():
            interpretations = list(activity_interpretations.values())
            embeddings = self.text_encoder(interpretations)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def _load_test_data(self, interpretations_file: str) -> Dict:
        """Load test data from home_a (completely unseen by sensor encoder)"""
        from models.sensor_encoder import SensorEncoderTrainer
        
        # Create temporary trainer to use data loading methods
        temp_trainer = SensorEncoderTrainer(self.config, self.text_encoder)
        temp_trainer.sensor_encoder = self.sensor_encoder
        
        # Load home_a all splits (completely unseen by sensor encoder)
        dataset = temp_trainer._prepare_training_data(
            interpretations_file, 
            splits=['train', 'val', 'test'],
            home_filter=['home_a']
        )
        
        return {
            'sensor_events': dataset.sensor_events_list,
            'true_activities': dataset.activities,
            'interpretations': dataset.sensor_interpretations
        }
    
    def _predict_batch(self, sensor_events_list: List[Dict], 
                      activity_embeddings: torch.Tensor,
                      activity_labels: List[str]) -> List[str]:
        """Predict activities for a batch of sensor events"""
        predictions = []
        
        # Create temporary trainer for collate function
        temp_trainer = SensorEncoderTrainer(self.config, self.text_encoder)
        temp_trainer.sensor_encoder = self.sensor_encoder
        
        with torch.no_grad():
            for i, sensor_events in enumerate(sensor_events_list):
                # Create mini-batch
                batch = temp_trainer._collate_fn([{
                    'sensor_events': sensor_events,
                    'sensor_interpretation': '',
                    'activity': ''
                }])
                
                # Move to device
                sensor_events_batch = {}
                for key in batch['sensor_events']:
                    if isinstance(batch['sensor_events'][key], torch.Tensor):
                        sensor_events_batch[key] = batch['sensor_events'][key].to(self.device)
                    else:
                        sensor_events_batch[key] = batch['sensor_events'][key]
                
                # Generate sensor embedding
                sensor_embedding = self.sensor_encoder(sensor_events_batch)
                sensor_embedding = F.normalize(sensor_embedding, p=2, dim=1)
                
                # Compute similarity with all activity embeddings
                similarities = torch.matmul(sensor_embedding, activity_embeddings.T)
                
                # Get most similar activity
                pred_idx = similarities.argmax(dim=1).item()
                predicted_activity = activity_labels[pred_idx]
                predictions.append(predicted_activity)
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(sensor_events_list)} samples...")
        
        return predictions
    
    def _evaluate_predictions(self, predictions: List[str], 
                             true_labels: List[str],
                             activity_labels: List[str]) -> Dict:
        """Evaluate prediction results"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = \
            precision_recall_fscore_support(
                true_labels, predictions, labels=activity_labels, zero_division=0
            )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_labels, predictions, labels=activity_labels)
        
        # Per-class accuracy
        per_class_metrics = {}
        for i, activity in enumerate(activity_labels):
            per_class_metrics[activity] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i]),
                'support': int(support[i])
            }
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'per_class_metrics': per_class_metrics,
            'activity_labels': activity_labels,
            'total_samples': len(predictions),
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    def _save_results(self, results: Dict, output_dir: str):
        """Save inference results and visualizations"""
        import json
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results JSON
        results_file = os.path.join(output_dir, "inference_results.json")
        
        # Prepare JSON-serializable results
        json_results = {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'total_samples': results['total_samples'],
            'confusion_matrix': results['confusion_matrix'],
            'per_class_metrics': results['per_class_metrics'],
            'activity_labels': results['activity_labels']
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved to: {results_file}")
        
        # Create confusion matrix heatmap
        self._plot_confusion_matrix(
            results['confusion_matrix'],
            results['activity_labels'],
            os.path.join(output_dir, "inference_confusion_matrix.png")
        )
    
    def _plot_confusion_matrix(self, conf_matrix: List, 
                               activity_labels: List[str],
                               save_path: str):
        """Plot confusion matrix heatmap"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=activity_labels,
            yticklabels=activity_labels,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title('Inference Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Activity', fontsize=12)
        plt.ylabel('True Activity', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    def _print_results(self, results: Dict):
        """Print evaluation results"""
        print("\n" + "=" * 80)
        print("INFERENCE RESULTS")
        print("=" * 80)
        
        print(f"\nOVERALL METRICS:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1']:.4f}")
        print(f"  Total Samples: {results['total_samples']}")
        
        print(f"\nPER-CLASS METRICS:")
        print("-" * 80)
        print(f"{'Activity':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-" * 80)
        
        for activity, metrics in results['per_class_metrics'].items():
            if metrics['support'] > 0:  # Only show activities with samples
                print(f"{activity:<20} {metrics['precision']:>10.4f} "
                      f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} "
                      f"{metrics['support']:>10}")
        
        print("=" * 80)
        
        # Quality assessment
        accuracy = results['accuracy']
        f1 = results['f1']
        
        print(f"\nQUALITY ASSESSMENT:")
        if accuracy > 0.7 and f1 > 0.7:
            print("   EXCELLENT: The model performs very well on unseen data!")
        elif accuracy > 0.5 and f1 > 0.5:
            print("   GOOD: The model shows reasonable performance.")
        elif accuracy > 0.3 and f1 > 0.3:
            print("   FAIR: The model has learned some patterns but needs improvement.")
        else:
            print("   POOR: The model needs more training or better features.")
        
        print("=" * 80)
