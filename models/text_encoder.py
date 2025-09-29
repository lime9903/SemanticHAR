"""
BERT-based text encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from transformers import BertModel, BertTokenizer
except ImportError:
    BertModel = None
    BertTokenizer = None
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import json
import os
from tqdm import tqdm
from config import SemanticHARConfig
            

class TextEncoder(nn.Module):
    """BERT-based text encoder"""
    
    def __init__(self, config: SemanticHARConfig):
        super(TextEncoder, self).__init__()
        self.config = config
        
        if BertModel is None or BertTokenizer is None:
            raise ImportError("transformers library is not installed. run pip install transformers.")
        
        self.bert = BertModel.from_pretrained(config.text_encoder_model)
        self.tokenizer = BertTokenizer.from_pretrained(config.text_encoder_model)
        
        self.hidden_dim = self.bert.config.hidden_size
        self.output_dim = config.sensor_encoder_hidden_dim
        
        self.projection = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(0.2)  # Increased dropout for better regularization
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Convert text to embedding"""
        if not texts:
            return torch.empty(0, self.output_dim, device=next(self.parameters()).device)
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(next(self.parameters()).device)
        attention_mask = encoded['attention_mask'].to(next(self.parameters()).device)
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        pooled_output = outputs.pooler_output
        projected = self.projection(pooled_output)
        projected = self.dropout(projected)
        projected = self.layer_norm(projected)
        
        return projected
    

class TextDecoder(nn.Module):
    """decoder for text reconstruction"""
    
    def __init__(self, config: SemanticHARConfig, vocab_size: int = 30522):
        super(TextDecoder, self).__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.sensor_encoder_hidden_dim,
            nhead=config.sensor_encoder_heads,
            dim_feedforward=config.sensor_encoder_hidden_dim * 2,  # Reduced from *4
            dropout=0.2,  # Increased dropout
            batch_first=True
        )
        
        # Simplified decoder with fewer layers
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=config.sensor_encoder_layers
        )
        
        # Additional regularization layers
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(config.sensor_encoder_hidden_dim)
        
        self.output_projection = nn.Linear(
            config.sensor_encoder_hidden_dim, 
            vocab_size
        )
        
        self.embedding = nn.Embedding(vocab_size, config.sensor_encoder_hidden_dim)
        self.positional_encoding = nn.Parameter(
            torch.randn(1000, config.sensor_encoder_hidden_dim)
        )
        
    def forward(self, memory: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Text reconstruction"""
        batch_size, seq_len = target_ids.shape
        
        target_embeddings = self.embedding(target_ids)
        positions = torch.arange(seq_len, device=target_ids.device)
        target_embeddings = target_embeddings + self.positional_encoding[:seq_len].unsqueeze(0)
        
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(target_ids.device)
        
        decoder_output = self.transformer_decoder(
            target_embeddings,
            memory.unsqueeze(1),
            tgt_mask=tgt_mask
        )
        
        # Additional regularization
        decoder_output = self.dropout(decoder_output)
        decoder_output = self.layer_norm(decoder_output)
        
        output = self.output_projection(decoder_output)
        return output
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate mask for autoregressive generation"""
        mask = torch.triu(torch.ones(sz, sz)) == 1
        mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class ContrastiveLearningModule(nn.Module):
    """contrastive learning module"""
    
    def __init__(self, config: SemanticHARConfig):
        super(ContrastiveLearningModule, self).__init__()
        self.config = config
        self.temperature = config.temperature
        
    def alignment_loss(self, sensor_embeddings: torch.Tensor, 
                      activity_embeddings: torch.Tensor) -> torch.Tensor:
        """Alignment loss between sensor embeddings and activity embeddings"""
        sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=1)
        activity_embeddings = F.normalize(activity_embeddings, p=2, dim=1)
        
        similarity_matrix = torch.matmul(sensor_embeddings, activity_embeddings.T) / self.temperature
        
        batch_size = sensor_embeddings.size(0)
        labels = torch.arange(batch_size, device=sensor_embeddings.device)
        
        loss_s2a = F.cross_entropy(similarity_matrix, labels)
        loss_a2s = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_s2a + loss_a2s) / 2
    
    def category_contrastive_loss(self, embeddings: torch.Tensor, 
                                 categories: torch.Tensor) -> torch.Tensor:
        """category-based contrastive learning loss"""
        # normalization
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # create category mask
        category_mask = (categories.unsqueeze(0) == categories.unsqueeze(1)).float()
        
        # separate positive and negative pairs
        positive_mask = category_mask - torch.eye(category_mask.size(0), device=category_mask.device)
        negative_mask = 1 - category_mask
        
        # calculate loss
        positive_similarities = similarity_matrix * positive_mask
        negative_similarities = similarity_matrix * negative_mask
        
        # softmax loss
        logits = positive_similarities - negative_similarities
        loss = -torch.log(torch.sigmoid(logits) + 1e-8).mean()
        
        return loss
    
    def activity_contrastive_loss(self, embeddings: torch.Tensor, 
                                 activities: torch.Tensor) -> torch.Tensor:
        """activity-based contrastive learning loss"""
        # normalization
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # create activity mask
        activity_mask = (activities.unsqueeze(0) == activities.unsqueeze(1)).float()
        
        # separate positive and negative pairs
        positive_mask = activity_mask - torch.eye(activity_mask.size(0), device=activity_mask.device)
        negative_mask = 1 - activity_mask
        
        # calculate loss
        positive_similarities = similarity_matrix * positive_mask
        negative_similarities = similarity_matrix * negative_mask
        
        # softmax loss
        logits = positive_similarities - negative_similarities
        loss = -torch.log(torch.sigmoid(logits) + 1e-8).mean()
        
        return loss

class TextEncoderTrainer:
    """text encoder trainer"""
    
    def __init__(self, config: SemanticHARConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print(f"⨠ TextEncoderTrainer using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.text_encoder = TextEncoder(config).to(self.device)
        self.text_decoder = TextDecoder(config).to(self.device)
        self.contrastive_module = ContrastiveLearningModule(config).to(self.device)
        
        print(f"⨠ Models moved to {self.device}")
        
        self.optimizer = torch.optim.AdamW(
            list(self.text_encoder.parameters()) + 
            list(self.text_decoder.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
    def train_step(self, sensor_interpretations: List[str], 
                   activity_interpretations: List[str],
                   sensor_categories: torch.Tensor,
                   activity_categories: torch.Tensor,
                   activity_labels: torch.Tensor) -> Dict[str, float]:
        """training step"""
        
        self.optimizer.zero_grad()
        
        sensor_categories = sensor_categories.to(self.device)
        activity_categories = activity_categories.to(self.device)
        activity_labels = activity_labels.to(self.device)
        
        sensor_embeddings = self.text_encoder(sensor_interpretations)
        activity_embeddings = self.text_encoder(activity_interpretations)
        
        alignment_loss = self.contrastive_module.alignment_loss(
            sensor_embeddings, activity_embeddings
        )
        
        sensor_category_loss = self.contrastive_module.category_contrastive_loss(
            sensor_embeddings, sensor_categories
        )
        
        activity_category_loss = self.contrastive_module.category_contrastive_loss(
            activity_embeddings, activity_categories
        )
        
        activity_contrastive_loss = self.contrastive_module.activity_contrastive_loss(
            activity_embeddings, activity_labels
        )
        
        reconstruction_loss = self._compute_reconstruction_loss(
            sensor_embeddings, sensor_interpretations
        )
        
        # Activity-side reconstruction loss (Sl -> H -> g_de -> Sl_hat)
        activity_reconstruction_loss = self._compute_reconstruction_loss(
            activity_embeddings, activity_interpretations
        )
        
        # Combine reconstruction losses
        total_reconstruction_loss = reconstruction_loss + activity_reconstruction_loss
        
        total_loss = (
            alignment_loss + 
            self.config.alpha * (sensor_category_loss + activity_category_loss + activity_contrastive_loss) +
            self.config.beta * total_reconstruction_loss
        )
        
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            list(self.text_encoder.parameters()) + list(self.text_decoder.parameters()),
            max_norm=self.config.gradient_clip_norm
        )
        
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'alignment_loss': alignment_loss.item(),
            'sensor_category_loss': sensor_category_loss.item(),
            'activity_category_loss': activity_category_loss.item(),
            'activity_contrastive_loss': activity_contrastive_loss.item(),
            'reconstruction_loss': total_reconstruction_loss.item()
        }
    
    def _validation_step(self, sensor_interpretations: List[str], 
                        activity_interpretations: List[str],
                        activity_categories: torch.Tensor,
                        activity_indices: torch.Tensor) -> torch.Tensor:
        """Validation step without gradient computation - computes full loss"""
        sensor_embeddings = self.text_encoder(sensor_interpretations)
        activity_embeddings = self.text_encoder(activity_interpretations)
        
        # Alignment loss
        alignment_loss = self.contrastive_module.alignment_loss(
            sensor_embeddings, activity_embeddings
        )
        
        # Category contrastive loss
        sensor_category_loss = self.contrastive_module.category_contrastive_loss(
            sensor_embeddings, activity_categories
        )
        
        activity_category_loss = self.contrastive_module.category_contrastive_loss(
            activity_embeddings, activity_categories
        )
        
        # Activity contrastive loss
        activity_contrastive_loss = self.contrastive_module.activity_contrastive_loss(
            activity_embeddings, activity_indices
        )
        
        # Reconstruction loss
        reconstruction_loss = self._compute_reconstruction_loss(
            sensor_embeddings, sensor_interpretations
        )
        
        # Activity-side reconstruction loss
        activity_reconstruction_loss = self._compute_reconstruction_loss(
            activity_embeddings, activity_interpretations
        )
        
        total_reconstruction_loss = reconstruction_loss + activity_reconstruction_loss
        
        # Total loss (same as training)
        total_loss = (
            alignment_loss + 
            self.config.alpha * (sensor_category_loss + activity_category_loss + activity_contrastive_loss) +
            self.config.beta * total_reconstruction_loss
        )
        
        return total_loss
    
    def _compute_reconstruction_loss(self, embeddings: torch.Tensor, 
                                   texts: List[str]) -> torch.Tensor:
        """reconstruction loss calculation using TextDecoder"""
        try:
            # 1. Use TextDecoder to reconstruct text from embeddings
            batch_size = embeddings.size(0)
            device = embeddings.device
            
            # Create target sequences from texts using tokenizer
            target_sequences = []
            max_length = 0
            
            for text in texts:
                # Tokenize text
                tokens = self.text_encoder.tokenizer.encode(
                    text, 
                    add_special_tokens=True, 
                    max_length=self.config.max_sequence_length,
                    truncation=True
                )
                target_sequences.append(tokens)
                max_length = max(max_length, len(tokens))
            
            # Pad sequences to same length
            padded_sequences = []
            for tokens in target_sequences:
                padded = tokens + [self.text_encoder.tokenizer.pad_token_id] * (max_length - len(tokens))
                padded_sequences.append(padded[:max_length])  # Ensure exact length
            
            target_tensor = torch.tensor(padded_sequences, device=device)
            
            # 2. Use TextDecoder to reconstruct from embeddings
            # Create input sequence (shifted by one for autoregressive generation)
            input_sequences = target_tensor[:, :-1]  # Remove last token
            target_sequences = target_tensor[:, 1:]   # Remove first token
            
            # Decode using TextDecoder
            decoder_output = self.text_decoder(embeddings, input_sequences)
            
            # 3. Use language model loss (cross-entropy) for reconstruction
            # Reshape for cross-entropy loss
            vocab_size = decoder_output.size(-1)
            decoder_output_flat = decoder_output.reshape(-1, vocab_size)
            target_flat = target_sequences.reshape(-1)
            
            # Compute cross-entropy loss
            reconstruction_loss = F.cross_entropy(
                decoder_output_flat, 
                target_flat, 
                ignore_index=self.text_encoder.tokenizer.pad_token_id
            )
            
            return reconstruction_loss
            
        except Exception as e:
            # Fallback to simple regularization if reconstruction fails
            print(f"⨺ Warning: Reconstruction loss computation failed: {e}")
            print("  Falling back to regularization loss")
            
            # Simple regularization loss as fallback
            regularization_loss = torch.mean(torch.norm(embeddings, p=2, dim=1))
            
            # Additional consistency loss
            if len(embeddings) > 1:
                normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
                similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T)
                diversity_loss = -torch.mean(torch.log(torch.abs(similarity_matrix) + 1e-8))
                total_loss = regularization_loss + 0.1 * diversity_loss
            else:
                total_loss = regularization_loss
                
            return total_loss
    
    def save_model(self, path: str):
        """save model"""
        torch.save({
            'text_encoder': self.text_encoder.state_dict(),
            'text_decoder': self.text_decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, path)
    
    def load_model(self, path: str):
        """load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.text_encoder.load_state_dict(checkpoint['text_encoder'])
        self.text_decoder.load_state_dict(checkpoint['text_decoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
    
    def train_with_interpretations(self, interpretations_file: str,
                                 num_epochs: int = 10,
                                 batch_size: int = 8,
                                 early_stopping: bool = False,
                                 patience: int = 10) -> 'TextEncoder':
        """Train text encoder with semantic interpretations"""

        print("=" * 50)
        print("Starting Text Encoder Training with Semantic Interpretations")
        print("=" * 50)
        
        # Prepare data (train and validation) - JSON loaded only once
        train_dataset = prepare_training_data(interpretations_file, splits=['train'])
        val_dataset = prepare_training_data(interpretations_file, splits=['val'])
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Batch size: {batch_size}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Early stopping: {early_stopping}")
        if early_stopping:
            print(f"Patience: {patience} epochs")
            print(f"Validation batches: {len(val_dataloader)}")
        
        print(f"\nTraining Text Encoder...")
        print("=" * 40)
        
        # Early stopping variables initialization
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            self.text_encoder.train()
            epoch_losses = []
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
                try:
                    # Batch data preparation
                    sensor_interpretations = batch['sensor_interpretation']
                    activity_interpretations = batch['activity_interpretation']
                    activities = batch['activity']
                    activity_categories = batch['activity_category'].clone().detach().to(self.device)
                    activity_indices = batch['activity_idx'].clone().detach().to(self.device)
                    
                    # Training step
                    losses = self.train_step(
                        sensor_interpretations=sensor_interpretations,
                        activity_interpretations=activity_interpretations,
                        sensor_categories=activity_categories,
                        activity_categories=activity_categories,
                        activity_labels=activity_indices
                    )
                    
                    epoch_losses.append(losses['total_loss'])
                    
                except Exception as e:
                    print(f"⨺ Error in batch {batch_idx}: {e}")
                    continue
            
            if epoch_losses:
                avg_train_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch+1}/{num_epochs} - Average Train Loss: {avg_train_loss:.4f}")
                
                # Validation phase for early stopping
                if early_stopping:
                    print(f"  ⨠ Validating on {len(val_dataset)} validation samples...")
                    self.text_encoder.eval()
                    val_losses = []
                    
                    with torch.no_grad():
                        # Use tqdm to show progress for validation
                        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="  Validation", leave=False)):
                            try:
                                sensor_interpretations = batch['sensor_interpretation']
                                activity_interpretations = batch['activity_interpretation']
                                activities = batch['activity']
                                
                                # Move to device
                                activity_categories = batch['activity_category'].clone().detach().to(self.device)
                                activity_indices = batch['activity_idx'].clone().detach().to(self.device)
                                
                                # Forward pass (validation) - no gradient computation
                                loss = self._validation_step(
                                    sensor_interpretations, 
                                    activity_interpretations, 
                                    activity_categories, 
                                    activity_indices
                                )
                                val_losses.append(loss.item())
                                
                            except Exception as e:
                                print(f"⨺ Validation batch {batch_idx} error: {e}")
                                continue
                    
                    if val_losses:
                        avg_val_loss = np.mean(val_losses)
                        std_val_loss = np.std(val_losses)
                        print(f"   Validation Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f} (from {len(val_losses)} batches)")
                        
                        # Update learning rate scheduler
                        self.scheduler.step(avg_val_loss)
                        
                        # Early stopping logic based on validation loss
                        if avg_val_loss < best_loss:
                            best_loss = avg_val_loss
                            patience_counter = 0
                            print(f"  → New best validation loss: {best_loss:.4f}")
                            
                            # Save best model
                            checkpoint_dir = "checkpoints"
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            best_model_path = os.path.join(checkpoint_dir, "text_encoder_best.pth")
                            torch.save(self.text_encoder.state_dict(), best_model_path)
                            print(f"   Best model saved to: {best_model_path}")
                            
                        else:
                            patience_counter += 1
                            print(f"   No improvement ({patience_counter}/{patience})")
                            if patience_counter >= patience:
                                print(f"   Early stopping triggered at epoch {epoch+1}")
                                print(f"   Best validation loss: {best_loss:.4f}")
                                break
                    else:
                        print("  ⨺ No valid validation batches")
                    
                    # Return to training mode for next epoch
                    self.text_encoder.train()
            else:
                print(f"⨺ Epoch {epoch+1}/{num_epochs} - No valid training batches processed")
        
        print("\nText Encoder Training Completed!")
        
        # Save final model
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if early_stopping:
            # Use best model if early stopping was used
            best_model_path = os.path.join(checkpoint_dir, "text_encoder_best.pth")
            if os.path.exists(best_model_path):
                print(f"✓ Using best model (validation loss: {best_loss:.4f})")
                # Rename best model to final model
                final_model_path = os.path.join(checkpoint_dir, "text_encoder_trained.pth")
                os.rename(best_model_path, final_model_path)
                print(f"✓ Best model renamed to final model: {final_model_path}")
            else:
                print("⨺ Best model not found, saving current model")
                final_model_path = os.path.join(checkpoint_dir, "text_encoder_trained.pth")
                torch.save(self.text_encoder.state_dict(), final_model_path)
                print(f"✓ Final text encoder saved to: {final_model_path}")
        else:
            # Save current model directly
            final_model_path = os.path.join(checkpoint_dir, "text_encoder_trained.pth")
            torch.save(self.text_encoder.state_dict(), final_model_path)
            print(f"✓ Final text encoder saved to: {final_model_path}")
        
        return self.text_encoder


# Training utilities
class InterpretationDataset(Dataset):
    """Dataset for semantic interpretations"""
    
    def __init__(self, sensor_interpretations: List[str], 
                 activity_interpretations: List[str],
                 activities: List[str]):
        # Filter out None values
        valid_indices = []
        for i, (sensor, activity) in enumerate(zip(sensor_interpretations, activity_interpretations)):
            if sensor is not None and activity is not None:
                valid_indices.append(i)
        
        self.sensor_interpretations = [sensor_interpretations[i] for i in valid_indices]
        self.activity_interpretations = [activity_interpretations[i] for i in valid_indices]
        self.activities = [activities[i] for i in valid_indices]
        
        # Activity to category mapping
        self.activity_categories = {
            'Sleeping': 0,
            'Toileting': 1, 
            'Showering': 1,
            'Breakfast': 2,
            'Lunch': 2,
            'Dinner': 2,
            'Grooming': 1,
            'Spare_Time/TV': 3,
            'Leaving': 4,
            'Snack': 2
        }
        
        # Activity to index mapping
        # True labels: Leaving, Toileting, Showering, Sleeping, Breakfast, Lunch, Dinner, Snack, Spare_Time/TV, Grooming
        self.activity_to_idx = {
            'Sleeping': 0,
            'Toileting': 1,
            'Showering': 2,
            'Breakfast': 3,
            'Lunch': 4,
            'Dinner': 5,
            'Grooming': 6,
            'Spare_Time/TV': 7,
            'Leaving': 8,
            'Snack': 9
        }
    
    def __len__(self):
        return len(self.sensor_interpretations)
    
    def __getitem__(self, idx):
        return {
            'sensor_interpretation': self.sensor_interpretations[idx],
            'activity_interpretation': self.activity_interpretations[idx],
            'activity': self.activities[idx],
            'activity_category': self.activity_categories.get(self.activities[idx], 0),
            'activity_idx': self.activity_to_idx.get(self.activities[idx], 0)
        }


# Global cache for loaded data
_loaded_data_cache = {}

def load_interpretations_from_json(json_file: str) -> Tuple[List[Dict], List[str]]:
    """Load interpretations from JSON file - returns all data with split info (cached)"""
    
    # Check cache first
    if json_file in _loaded_data_cache:
        print(f"Using cached data for {json_file}")
        return _loaded_data_cache[json_file]
    
    print(f"Loading data from {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_sensor_data = []
    activity_interpretations = []
    
    # Load all sensor data with split information
    for home_id in data['sensor_interpretations']:
        for split in data['sensor_interpretations'][home_id]:
            for window_id, window_data in data['sensor_interpretations'][home_id][split].items():
                if 'interpretation' in window_data and 'error' not in window_data:
                    all_sensor_data.append({
                        'interpretation': window_data['interpretation'],
                        'activity': window_data['activity'],
                        'split': split,
                        'home_id': home_id,
                        'window_id': window_id
                    })
    
    # Activity interpretations load
    for activity, activity_data in data['activity_interpretations'].items():
        if 'interpretation' in activity_data and 'error' not in activity_data:
            activity_interpretations.append(activity_data['interpretation'])
    
    print(f"Loaded {len(all_sensor_data)} total sensor interpretations")
    print(f"Loaded {len(activity_interpretations)} activity interpretations")
    
    # Cache the data
    _loaded_data_cache[json_file] = (all_sensor_data, activity_interpretations)
    
    return all_sensor_data, activity_interpretations


def prepare_training_data(interpretations_file: str, splits: List[str] = ['train']) -> InterpretationDataset:
    """Prepare training data for specific splits (train/val/test)
    
    Args:
        interpretations_file: Path to JSON file
        splits: List of splits to include (e.g., ['train'], ['val'], ['train', 'val'], ['test'])
    """
    print(f"Loading interpretations from: {interpretations_file} for splits: {splits}")
    
    # Load all data and filter by splits
    all_sensor_data, activity_interpretations = load_interpretations_from_json(interpretations_file)
    
    # Filter data by splits
    split_data = [item for item in all_sensor_data if item['split'] in splits]
    
    if not split_data:
        raise ValueError(f"No valid data found for splits {splits} in the JSON file")
    
    # Extract data for the dataset
    sensor_interpretations = [item['interpretation'] for item in split_data]
    activities = [item['activity'] for item in split_data]
    
    if len(sensor_interpretations) == 0:
        raise ValueError("⨺ No valid sensor interpretations found in the JSON file")
    
    if len(activity_interpretations) == 0:
        raise ValueError("⨺ No valid activity interpretations found in the JSON file")
    
    # Activity interpretations match with sensor interpretations
    activity_to_interpretation = {}
    for i, activity in enumerate(activities):
        if activity not in activity_to_interpretation:
            # Find the interpretation for the corresponding activity
            for act_interpretation in activity_interpretations:
                if act_interpretation and (activity in act_interpretation or activity.lower() in act_interpretation.lower()):
                    activity_to_interpretation[activity] = act_interpretation
                    break
    
    # Matched activity interpretations
    matched_activity_interpretations = []
    for activity in activities:
        if activity in activity_to_interpretation:
            matched_activity_interpretations.append(activity_to_interpretation[activity])
        else:
            # Fallback: Use the first activity interpretation
            if activity_interpretations and activity_interpretations[0]:
                matched_activity_interpretations.append(activity_interpretations[0])
            else:
                # Final fallback: Use a default interpretation
                matched_activity_interpretations.append(f"Activity: {activity}")
    
    print(f"Prepared {len(sensor_interpretations)} sensor-activity interpretation pairs")
    
    return InterpretationDataset(
        sensor_interpretations=sensor_interpretations,
        activity_interpretations=matched_activity_interpretations,
        activities=activities
    )


def test_text_encoder(text_encoder: TextEncoder, test_interpretations: List[str]):
    """Text encoder test"""
    print("\nTesting Text Encoder...")
    print("=" * 30)
    
    text_encoder.eval()
    
    with torch.no_grad():
        for i, interpretation in enumerate(test_interpretations[:3]):  # First 3 only
            try:
                embedding = text_encoder.encode_single(interpretation)
                print(f"Interpretation {i+1}:")
                print(f"  Text: {interpretation[:100]}...")
                print(f"  Embedding shape: {embedding.shape}")
                print(f"  Embedding norm: {torch.norm(embedding).item():.4f}")
                print()
            except Exception as e:
                print(f"⨺ Error encoding interpretation {i+1}: {e}")


class TextEncoderEvaluator:
    """Text Encoder training validation class"""
    
    def __init__(self, config: SemanticHARConfig, text_encoder: Optional[TextEncoder] = None, model_path: Optional[str] = None):
        self.config = config
        self.device = torch.device(config.device)
        
        # Model loading
        if text_encoder is not None:
            self.text_encoder = text_encoder
            self.text_decoder = TextDecoder(config).to(self.device)
            print(f"✓ TextEncoder object is used directly.")
        elif model_path and os.path.exists(model_path):
            self.text_encoder = TextEncoder(config).to(self.device)
            self.text_decoder = TextDecoder(config).to(self.device)
            self.load_model(model_path)
            print(f"✓ Model loaded: {model_path}")
        else:
            self.text_encoder = TextEncoder(config).to(self.device)
            self.text_decoder = TextDecoder(config).to(self.device)
            print("⨺ Model file not found. Randomly initialized model is used.")
        
        # Evaluation mode
        self.text_encoder.eval()
        self.text_decoder.eval()
    
    def load_model(self, model_path: str):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'text_encoder' in checkpoint:
            self.text_encoder.load_state_dict(checkpoint['text_encoder'])
        else:
            self.text_encoder.load_state_dict(checkpoint)
    
    def evaluate_alignment_quality(self, sensor_interpretations: List[str], 
                                 activity_interpretations: List[str]) -> Dict[str, float]:
        """Sensor-Activity alignment quality evaluation"""
        print("⨠ Sensor-Activity alignment quality evaluation...")
        
        # Data length matching
        min_length = min(len(sensor_interpretations), len(activity_interpretations))
        sensor_interpretations = sensor_interpretations[:min_length]
        activity_interpretations = activity_interpretations[:min_length]
        
        with torch.no_grad():
            # Embedding generation
            sensor_embeddings = self.text_encoder(sensor_interpretations)
            activity_embeddings = self.text_encoder(activity_interpretations)
            
            # Normalization
            sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=1)
            activity_embeddings = F.normalize(activity_embeddings, p=2, dim=1)
            
            # Similarity matrix calculation
            similarity_matrix = torch.matmul(sensor_embeddings, activity_embeddings.T)
            
            # Diagonal elements (correct pairs) similarity
            diagonal_similarities = torch.diag(similarity_matrix)
            
            # Maximum similarity in each row (check if correct is the highest)
            max_similarities, max_indices = torch.max(similarity_matrix, dim=1)
            
            # Accuracy calculation
            correct_predictions = (max_indices == torch.arange(len(sensor_interpretations), device=self.device)).float()
            accuracy = correct_predictions.mean().item()
            
            # Average correct pair similarity
            avg_correct_similarity = diagonal_similarities.mean().item()
            
            # Difference in similarity between correct and incorrect
            off_diagonal_similarities = similarity_matrix - torch.diag(diagonal_similarities)
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
    
    def evaluate_reconstruction_quality(self, texts: List[str]) -> Dict[str, float]:
        """Reconstruction quality evaluation"""
        print("⨠ Reconstruction quality evaluation...")
        
        # Embedding generation
        embeddings = self.text_encoder(texts)
        
        reconstruction_losses = []
        reconstruction_accuracies = []
        
        for i, text in enumerate(texts):
            try:
                # Tokenization
                tokens = self.text_encoder.tokenizer.encode(
                    text, 
                    add_special_tokens=True, 
                    max_length=self.config.max_sequence_length,
                    truncation=True
                )
                
                # Reconstruction attempt
                input_tokens = torch.tensor([tokens[:-1]], device=self.device)  # Last token excluded
                target_tokens = torch.tensor([tokens[1:]], device=self.device)   # First token excluded
                
                # Reconstruction
                decoder_output = self.text_decoder(embeddings[i:i+1], input_tokens)
                
                # Loss calculation
                loss = F.cross_entropy(
                    decoder_output.reshape(-1, decoder_output.size(-1)),
                    target_tokens.reshape(-1),
                    ignore_index=self.text_encoder.tokenizer.pad_token_id
                )
                
                reconstruction_losses.append(loss.item())
                
                # Accuracy calculation (simple version)
                predicted_tokens = torch.argmax(decoder_output, dim=-1)
                accuracy = (predicted_tokens == target_tokens).float().mean().item()
                reconstruction_accuracies.append(accuracy)
                
            except Exception as e:
                print(f"Reconstruction error (text {i}): {e}")
                reconstruction_losses.append(float('inf'))
                reconstruction_accuracies.append(0.0)
        
        return {
            'avg_reconstruction_loss': np.mean(reconstruction_losses),
            'avg_reconstruction_accuracy': np.mean(reconstruction_accuracies),
            'reconstruction_losses': reconstruction_losses,
            'reconstruction_accuracies': reconstruction_accuracies
        }
    
    def visualize_embeddings(self, sensor_interpretations: List[str], 
                           activity_interpretations: List[str],
                           activities: List[str],
                           save_path: str = "outputs/embedding_visualization.png"):
        """Embedding visualization"""
        print("⨠ Embedding visualization generating...")
        
        # Data length matching
        min_length = min(len(sensor_interpretations), len(activity_interpretations))
        sensor_interpretations = sensor_interpretations[:min_length]
        activity_interpretations = activity_interpretations[:min_length]
        activities = activities[:min_length]
        
        with torch.no_grad():
            # Embedding generation
            sensor_embeddings = self.text_encoder(sensor_interpretations)
            activity_embeddings = self.text_encoder(activity_interpretations)
        
        # t-SNE dimension reduction
        all_embeddings = torch.cat([sensor_embeddings, activity_embeddings], dim=0)
        all_embeddings_np = all_embeddings.detach().cpu().numpy()
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings_np)-1))
        embeddings_2d = tsne.fit_transform(all_embeddings_np)
        
        # Visualization
        plt.figure(figsize=(15, 10))
        
        # Sensor embeddings
        sensor_2d = embeddings_2d[:len(sensor_interpretations)]
        scatter1 = plt.scatter(sensor_2d[:, 0], sensor_2d[:, 1], 
                              c='blue', alpha=0.6, s=50, label='Sensor Interpretations')
        
        # Activity embeddings
        activity_2d = embeddings_2d[len(sensor_interpretations):]
        scatter2 = plt.scatter(activity_2d[:, 0], activity_2d[:, 1], 
                              c='red', alpha=0.6, s=50, label='Activity Interpretations')
        
        for i in range(len(sensor_interpretations)):
            plt.plot([sensor_2d[i, 0], activity_2d[i, 0]], 
                    [sensor_2d[i, 1], activity_2d[i, 1]], 
                    'k--', alpha=0.3, linewidth=0.5)
        
        plt.title('Text Encoder Embeddings Visualization')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved: {save_path}")
    
    def evaluate_similarity_matrix(self, sensor_interpretations: List[str], 
                                 activity_interpretations: List[str],
                                 save_path: str = "outputs/similarity_matrix.png"):
        """Similarity matrix visualization"""
        print("⨠ Similarity matrix visualization generating...")
        
        # Data length matching
        min_length = min(len(sensor_interpretations), len(activity_interpretations))
        sensor_interpretations = sensor_interpretations[:min_length]
        activity_interpretations = activity_interpretations[:min_length]
        
        with torch.no_grad():
            # Embedding generation
            sensor_embeddings = self.text_encoder(sensor_interpretations)
            activity_embeddings = self.text_encoder(activity_interpretations)
            
            # Normalization
            sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=1)
            activity_embeddings = F.normalize(activity_embeddings, p=2, dim=1)
            
            # Similarity matrix calculation
            similarity_matrix = torch.matmul(sensor_embeddings, activity_embeddings.T)
            similarity_matrix_np = similarity_matrix.detach().cpu().numpy()
        
        # Visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix_np, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='RdYlBu_r',
                   center=0,
                   square=True)
        
        plt.title('Sensor-Activity Similarity Matrix')
        plt.xlabel('Activity Interpretations')
        plt.ylabel('Sensor Interpretations')
        
        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Similarity matrix saved: {save_path}")
    
    def comprehensive_evaluation(self, interpretations_file: str, 
                              output_dir: str = "outputs") -> Dict:
        """Comprehensive evaluation"""
        print("⨠ Text Encoder comprehensive evaluation starting...")
        
        # Data loading
        with open(interpretations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Sensor interpretations extraction
        sensor_interpretations = []
        activity_interpretations = []
        activities = []
        
        for home_id, home_data in data['sensor_interpretations'].items():
            for split, windows in home_data.items():
                if split == 'train':  # train data only
                    for window_id, window_data in windows.items():
                        if 'interpretation' in window_data:
                            sensor_interpretations.append(window_data['interpretation'])
                            activities.append(window_data['activity'])
        
        # Activity interpretations extraction
        for activity, interpretation_data in data.get('activity_interpretations', {}).items():
            if 'interpretation' in interpretation_data:
                activity_interpretations.append(interpretation_data['interpretation'])
        
        # Data limit (for evaluation)
        max_samples = min(50, len(sensor_interpretations), len(activity_interpretations))
        sensor_interpretations = sensor_interpretations[:max_samples]
        activities = activities[:max_samples]
        activity_interpretations = activity_interpretations[:max_samples]
        
        print(f" Evaluation data: {len(sensor_interpretations)} sensors, {len(activity_interpretations)} activities")
        
        # 1. Alignment quality evaluation
        alignment_results = self.evaluate_alignment_quality(
            sensor_interpretations, activity_interpretations
        )
        
        # 2. Reconstruction quality evaluation
        reconstruction_results = self.evaluate_reconstruction_quality(
            sensor_interpretations[:10]  # Reconstruction is only tested for part
        )
        
        # 3. Visualization
        self.visualize_embeddings(
            sensor_interpretations, activity_interpretations, activities,
            os.path.join(output_dir, "embedding_visualization.png")
        )
        
        self.evaluate_similarity_matrix(
            sensor_interpretations, activity_interpretations,
            os.path.join(output_dir, "similarity_matrix.png")
        )
        
        # Comprehensive results
        results = {
            'alignment_quality': alignment_results,
            'reconstruction_quality': reconstruction_results,
            'evaluation_summary': {
                'accuracy': alignment_results['accuracy'],
                'margin': alignment_results['margin'],
                'reconstruction_loss': reconstruction_results['avg_reconstruction_loss'],
                'reconstruction_accuracy': reconstruction_results['avg_reconstruction_accuracy']
            }
        }
        
        # Results saving (numpy arrays to list)
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        results_serializable = convert_numpy_types(results)
        
        results_file = os.path.join(output_dir, "text_encoder_evaluation_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Evaluation results saved: {results_file}")
        
        return results
