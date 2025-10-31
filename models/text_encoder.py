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
        
        self.hidden_dim = self.bert.config.hidden_size  # 768
        self.output_dim = config.sensor_encoder_hidden_dim  # 768
        
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        self.dropout = nn.Dropout(0.1)
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
            d_model=config.text_encoder_hidden_dim,
            nhead=config.text_encoder_heads,
            dim_feedforward=config.text_encoder_hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=4
        )

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(config.text_encoder_hidden_dim)
        
        self.output_projection = nn.Sequential(
            nn.Linear(config.text_encoder_hidden_dim, config.text_encoder_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.text_encoder_hidden_dim * 2, vocab_size)
        )
        
        self.embedding = nn.Embedding(vocab_size, config.text_encoder_hidden_dim)
        self.positional_encoding = nn.Parameter(
            torch.randn(1000, config.text_encoder_hidden_dim) * 0.1
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        
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
        self.temperature = config.text_encoder_temperature
        
    def alignment_loss(self, sensor_embeddings: torch.Tensor, 
                      activity_embeddings: torch.Tensor) -> torch.Tensor:
        """Alignment loss between sensor embeddings and activity embeddings"""
        sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=1)
        activity_embeddings = F.normalize(activity_embeddings, p=2, dim=1)
        
        similarity_matrix = torch.matmul(sensor_embeddings, activity_embeddings.T) / self.config.text_encoder_temperature
        
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
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.config.text_encoder_temperature
        
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
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.config.text_encoder_temperature
        
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
        
        print(f"TextEncoderTrainer using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.text_encoder = TextEncoder(config).to(self.device)
        self.text_decoder = TextDecoder(config).to(self.device)
        self.contrastive_module = ContrastiveLearningModule(config).to(self.device)
        
        print(f"✓ Models moved to {self.device}")
        
        self.optimizer = torch.optim.AdamW(
            list(self.text_encoder.parameters()) + 
            list(self.text_decoder.parameters()),
            lr=config.text_encoder_learning_rate,
            weight_decay=config.text_encoder_weight_decay
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
        
        total_loss = (
            alignment_loss + 
            self.config.alpha * (sensor_category_loss + activity_category_loss + activity_contrastive_loss) +
            self.config.beta * reconstruction_loss
        )
        
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            list(self.text_encoder.parameters()) + list(self.text_decoder.parameters()),
            max_norm=self.config.text_encoder_gradient_clip_norm
        )
        
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'alignment_loss': alignment_loss.item(),
            'sensor_category_loss': sensor_category_loss.item(),
            'activity_category_loss': activity_category_loss.item(),
            'activity_contrastive_loss': activity_contrastive_loss.item(),
            'reconstruction_loss': reconstruction_loss.item()
        }
    
    def _validation_step(self, sensor_interpretations: List[str], 
                        activity_interpretations: List[str],
                        activity_categories: torch.Tensor,
                        activity_indices: torch.Tensor) -> torch.Tensor:
        """Validation step without gradient computation"""
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
        
        # Total loss (same as training)
        total_loss = (
            alignment_loss + 
            self.config.alpha * (sensor_category_loss + activity_category_loss + activity_contrastive_loss) +
            self.config.beta * reconstruction_loss
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
            
            # Compute cross-entropy loss with label smoothing
            reconstruction_loss = F.cross_entropy(
                decoder_output_flat, 
                target_flat, 
                ignore_index=self.text_encoder.tokenizer.pad_token_id,
                label_smoothing=0.1  # Label smoothing for better generalization
            )
            
            return reconstruction_loss
            
        except Exception as e:
            # Fallback to simple regularization if reconstruction fails
            print(f"✗ Warning: Reconstruction loss computation failed: {e}")
            print("  Falling back to regularization loss")
            
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
    
    def train_text_encoder(self, interpretations_file: str, num_epochs: int = 10,
              batch_size: int = 8, early_stopping: bool = False, patience: int = 10) -> Tuple[TextEncoder, TextDecoder]:
        """Train text encoder with semantic interpretations"""
        
        print(f"\nUsing {self.config.source_dataset} train and val splits\n")

        train_dataset = prepare_training_data(interpretations_file, splits=['train'], 
                                            dataset_name=self.config.source_dataset)
        val_dataset = prepare_training_data(interpretations_file, splits=['val'], 
                                         dataset_name=self.config.source_dataset)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"  - Train dataset size: {len(train_dataset)}")
        print(f"  - Validation dataset size: {len(val_dataset)}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Number of epochs: {num_epochs}")
        print(f"  - Early stopping: {early_stopping}")
        if early_stopping:
            print(f"  - Patience: {patience} epochs")
            print(f"  - Validation batches: {len(val_dataloader)}")
        
        print(f"\nTraining Text Encoder...")
        print("-" * 40)
        
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
                    print(f"✗ Error in batch {batch_idx}: {e}")
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
                                print(f"✗ Validation batch {batch_idx} error: {e}")
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
                        print("  ✗ No valid validation batches")
                    
                    # Return to training mode for next epoch
                    self.text_encoder.train()
            else:
                print(f"✗ Epoch {epoch+1}/{num_epochs} - No valid training batches processed")
        
        print("\nText Encoder Training Completed!")
        
        # Save final model
        checkpoint_dir = self.config.model_dir
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
                print("✗ Best model not found, saving current model")
                final_model_path = os.path.join(checkpoint_dir, "text_encoder_trained.pth")
                torch.save(self.text_encoder.state_dict(), final_model_path)
                print(f"✓ Final text encoder saved to: {final_model_path}")
        else:
            # Save current model directly
            final_model_path = os.path.join(checkpoint_dir, "text_encoder_trained.pth")
            torch.save(self.text_encoder.state_dict(), final_model_path)
            print(f"✓ Final text encoder saved to: {final_model_path}")
        
        return self.text_encoder, self.text_decoder


class InterpretationDataset(Dataset):
    """Dataset for semantic interpretations with cross-domain activity support"""
    
    def __init__(self, sensor_interpretations: List[str], 
                 activity_interpretations: List[str],
                 activities: List[str],
                 dataset_name: str = None,
                 activity_manager=None):
        # Import here to avoid circular imports
        from utils.activity_label_manager import activity_manager as default_manager
        
        # Filter out None values
        valid_indices = []
        for i, (sensor, activity) in enumerate(zip(sensor_interpretations, activity_interpretations)):
            if sensor is not None and activity is not None:
                valid_indices.append(i)
        
        self.sensor_interpretations = [sensor_interpretations[i] for i in valid_indices]
        self.activity_interpretations = [activity_interpretations[i] for i in valid_indices]
        self.activities = [activities[i] for i in valid_indices]
        self.dataset_name = dataset_name
        self.activity_manager = activity_manager or default_manager
        
        # Build unified activity mappings
        self._build_unified_mappings()
    
    def _build_unified_mappings(self):
        """Build unified activity mappings using ActivityLabelManager"""
        # Get unique activities in this dataset
        unique_activities = list(set(self.activities))
        
        # Map each activity to unified label
        self.activity_to_unified = {}
        self.activity_to_idx = {}
        self.activity_categories = {}
        
        # Create category to index mapping
        category_to_idx = {}
        category_idx = 0
        
        # Track unmapped activities
        unmapped_activities = []
        
        for activity in unique_activities:
            if self.dataset_name:
                unified_label = self.activity_manager.map_to_unified(activity, self.dataset_name)
                if unified_label:
                    self.activity_to_unified[activity] = unified_label
                    self.activity_to_idx[activity] = self.activity_manager.get_unified_index(unified_label)
                    category = self.activity_manager.get_category(unified_label)
                    
                    # Convert category to index
                    if category not in category_to_idx:
                        category_to_idx[category] = category_idx
                        category_idx += 1
                    self.activity_categories[activity] = category_to_idx[category]
                else:
                    unmapped_activities.append(activity)
                    # Use fallback mapping
                    self.activity_to_unified[activity] = 'unknown'
                    self.activity_to_idx[activity] = len(self.activity_manager.get_unified_activities())
                    
                    # Add unknown category
                    if 'unknown' not in category_to_idx:
                        category_to_idx['unknown'] = category_idx
                        category_idx += 1
                    self.activity_categories[activity] = category_to_idx['unknown']
            else:
                # Fallback to original mapping if no dataset specified
                self.activity_to_unified[activity] = activity
                self.activity_to_idx[activity] = len(self.activity_to_idx)
                
                # Add unknown category
                if 'unknown' not in category_to_idx:
                    category_to_idx['unknown'] = category_idx
                    category_idx += 1
                self.activity_categories[activity] = category_to_idx['unknown']
        
        if unmapped_activities:
            print(f"Warning: Found unmapped activities in {self.dataset_name}: {unmapped_activities}")
            print("Consider adding these activities to the activity mapping configuration.")
    
    def __len__(self):
        return len(self.sensor_interpretations)
    
    def __getitem__(self, idx):
        activity = self.activities[idx]
        return {
            'sensor_interpretation': self.sensor_interpretations[idx],
            'activity_interpretation': self.activity_interpretations[idx],
            'activity': activity,
            'unified_activity': self.activity_to_unified.get(activity, 'unknown'),
            'activity_category': torch.tensor(self.activity_categories.get(activity, 'unknown'), dtype=torch.long),
            'activity_idx': torch.tensor(self.activity_to_idx.get(activity, 0), dtype=torch.long)
        }
    
    def get_unified_vocab_size(self) -> int:
        """Get the size of unified vocabulary"""
        return len(self.activity_manager.get_unified_activities()) + 1  # +1 for unknown
    
    def get_activity_mapping_info(self) -> Dict:
        """Get activity mapping information for debugging"""
        return {
            'dataset_name': self.dataset_name,
            'unique_activities': list(set(self.activities)),
            'activity_to_unified': self.activity_to_unified,
            'activity_to_idx': self.activity_to_idx,
            'unified_vocab_size': self.get_unified_vocab_size()
        }


# Global cache for loaded data
_loaded_data_cache = {}

def load_interpretations_from_json(json_file: str) -> Tuple[List[Dict], Dict[str, str]]:
    """Load interpretations from JSON file"""
    
    # Check cache first
    if json_file in _loaded_data_cache:
        print(f"Using cached data for {json_file}")
        return _loaded_data_cache[json_file]
    
    print(f"Loading data from {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sensor_interpretations = []
    activity_interpretations = {}
    
    # Load all sensor data with split information
    for split in data['sensor_interpretations']:
        for window_id, window_data in data['sensor_interpretations'][split].items():
            if isinstance(window_data, dict) and 'interpretation' in window_data and 'error' not in window_data:
                sensor_interpretations.append({
                    'interpretation': window_data['interpretation'],
                    'activity': window_data['activity'],
                    'split': split,
                    'window_id': window_id
                })
    
    # Activity interpretations load as dictionary
    for activity, activity_data in data['activity_interpretations'].items():
        if 'interpretation' in activity_data and 'error' not in activity_data:
            activity_interpretations[activity] = activity_data['interpretation']
    
    print(f"Loaded {len(sensor_interpretations)} total sensor interpretations")
    print(f"Loaded {len(activity_interpretations)} activity interpretations")
    
    # Cache the data
    _loaded_data_cache[json_file] = (sensor_interpretations, activity_interpretations)
    
    return sensor_interpretations, activity_interpretations


def prepare_training_data(interpretations_file: str, splits: List[str] = ['train'], 
                         dataset_name: str = None) -> InterpretationDataset:
    """Prepare training data for specific splits (train/val/test)
    
    Args:
        interpretations_file: Path to JSON file
        splits: List of splits to include (e.g., ['train'], ['val'])
        dataset_name: Name of the dataset for activity mapping
    """
    print(f"Loading interpretations from: {interpretations_file} for splits: {splits}")
    
    # Load all data and filter by splits
    all_sensor_data, activity_interpretations = load_interpretations_from_json(interpretations_file)
    
    # Filter data by splits
    split_data = [item for item in all_sensor_data if item['split'] in splits]
    
    if not split_data:
        raise ValueError(f"✗ No valid data found for splits {splits} in the JSON file")
    
    # Extract data for the dataset
    sensor_interpretations = [item['interpretation'] for item in split_data]
    activities = [item['activity'] for item in split_data]
    
    if len(sensor_interpretations) == 0:
        raise ValueError("✗ No valid sensor interpretations found in the JSON file")
    
    if len(activity_interpretations) == 0:
        raise ValueError("✗ No valid activity interpretations found in the JSON file")
    
    # Activity interpretations match with sensor interpretations
    activity_to_interpretation = {}
    for i, activity in enumerate(activities):
        if activity not in activity_to_interpretation:
            if activity in activity_interpretations:
                activity_to_interpretation[activity] = activity_interpretations[activity]
    
    # Matched activity interpretations
    matched_activity_interpretations = []
    for activity in activities:
        if activity in activity_to_interpretation:
            matched_activity_interpretations.append(activity_to_interpretation[activity])
    
    print(f"Prepared {len(sensor_interpretations)} sensor-activity interpretation pairs")

    return InterpretationDataset(
        sensor_interpretations=sensor_interpretations,
        activity_interpretations=matched_activity_interpretations,
        activities=activities,
        dataset_name=dataset_name
    )


class TextEncoderEvaluator:
    """Text Encoder training validation class"""
    
    def __init__(self, config: SemanticHARConfig, text_encoder: TextEncoder, text_decoder: TextDecoder):
        self.config = config
        self.device = torch.device(config.device)
        
        # Use trained text encoder/decoder
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder
        
        # Evaluation mode
        self.text_encoder.eval()
        self.text_decoder.eval()
    
    def evaluate_alignment_quality(self, sensor_interpretations: List[str],
                                   activity_interpretations: List[str]) -> Dict[str, float]:
        """Sensor-Activity alignment quality evaluation"""
        
        with torch.no_grad():
            # Embedding generation
            sensor_embeddings = self.text_encoder(sensor_interpretations)
            activity_embeddings = self.text_encoder(activity_interpretations)
            
            sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=1)
            activity_embeddings = F.normalize(activity_embeddings, p=2, dim=1)
            
            similarity_matrix = torch.matmul(sensor_embeddings, activity_embeddings.T)
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
    
    def evaluate_reconstruction_quality(self, sensor_interpretations: List[str]) -> Dict[str, float]:
        """Reconstruction quality evaluation with realistic autoregressive generation"""
        
        sensor_interpretations = sensor_interpretations[:3]
        sensor_embeddings = self.text_encoder(sensor_interpretations)
        
        reconstruction_losses = []
        reconstruction_accuracies_teacher_forcing = []
        reconstruction_accuracies_autoregressive = []
        bleu_scores = []
        
        for i, original_text in enumerate(sensor_interpretations):
            try:
                tokens = self.text_encoder.tokenizer.encode(
                    original_text, 
                    add_special_tokens=True, 
                    max_length=self.config.max_sequence_length,
                    truncation=True
                )
                
                # 1. Teacher Forcing Evaluation (for loss calculation)
                input_tokens = torch.tensor([tokens[:-1]], device=self.device)
                target_tokens = torch.tensor([tokens[1:]], device=self.device)
                
                decoder_output = self.text_decoder(sensor_embeddings[i:i+1], input_tokens)
                
                # Loss calculation (teacher forcing)
                loss = F.cross_entropy(
                    decoder_output.reshape(-1, decoder_output.size(-1)),
                    target_tokens.reshape(-1),
                    ignore_index=self.text_encoder.tokenizer.pad_token_id
                )
                reconstruction_losses.append(loss.item())
                
                # Teacher forcing accuracy
                predicted_tokens = torch.argmax(decoder_output, dim=-1)
                tf_accuracy = (predicted_tokens == target_tokens).float().mean().item()
                reconstruction_accuracies_teacher_forcing.append(tf_accuracy)
                
                # 2. Autoregressive Generation
                generated_tokens = self._generate_autoregressive(sensor_embeddings[i:i+1], max_length=len(tokens))
                
                # Autoregressive accuracy (token-level)
                if len(generated_tokens) > 1:
                    # Compare with target tokens (excluding [CLS])
                    target_compare = target_tokens[0][:len(generated_tokens)-1]
                    gen_compare = torch.tensor(generated_tokens[1:], device=self.device)
                    
                    min_len = min(len(target_compare), len(gen_compare))
                    if min_len > 0:
                        ar_accuracy = (target_compare[:min_len] == gen_compare[:min_len]).float().mean().item()
                        reconstruction_accuracies_autoregressive.append(ar_accuracy)
                    else:
                        reconstruction_accuracies_autoregressive.append(0.0)
                else:
                    reconstruction_accuracies_autoregressive.append(0.0)
                
                # 3. BLEU Score (semantic similarity)
                try:
                    generated_text = self.text_encoder.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    original_text = self.text_encoder.tokenizer.decode(tokens, skip_special_tokens=True)
                    
                    def calculate_bleu_score(reference: str, candidate: str) -> float:
                        """Calculate BLEU-like score between reference and candidate text"""
                        ref_words = reference.lower().split()
                        cand_words = candidate.lower().split()
                        
                        if not ref_words or not cand_words:
                            return 0.0
                        
                        # 1-gram precision
                        ref_1grams = {}
                        for word in ref_words:
                            ref_1grams[word] = ref_1grams.get(word, 0) + 1
                        
                        cand_1grams = {}
                        for word in cand_words:
                            cand_1grams[word] = cand_1grams.get(word, 0) + 1
                        
                        # Count matches
                        matches = 0
                        for word, count in cand_1grams.items():
                            matches += min(count, ref_1grams.get(word, 0))
                        
                        precision = matches / len(cand_words) if cand_words else 0.0
                        
                        # Brevity penalty
                        if len(cand_words) < len(ref_words):
                            brevity_penalty = np.exp(1 - len(ref_words) / len(cand_words))
                        else:
                            brevity_penalty = 1.0
                        
                        return brevity_penalty * precision
                    
                    bleu_score = calculate_bleu_score(original_text, generated_text)
                    bleu_scores.append(bleu_score)
                    
                except Exception as e:
                    print(f"BLEU calculation error (text {i}): {e}")
                    bleu_scores.append(0.0)
                
            except Exception as e:
                print(f"Reconstruction error (text {i}): {e}")

                if len(reconstruction_losses) == i:  # Only add if not already added
                    reconstruction_losses.append(float('inf'))
                    reconstruction_accuracies_teacher_forcing.append(0.0)
                    reconstruction_accuracies_autoregressive.append(0.0)
                    bleu_scores.append(0.0)
        
        return {
            'avg_reconstruction_loss': np.mean(reconstruction_losses),
            'avg_reconstruction_accuracy_teacher_forcing': np.mean(reconstruction_accuracies_teacher_forcing),
            'avg_reconstruction_accuracy_autoregressive': np.mean(reconstruction_accuracies_autoregressive),
            'avg_bleu_score': np.mean(bleu_scores),
            'reconstruction_losses': reconstruction_losses,
            'teacher_forcing_accuracies': reconstruction_accuracies_teacher_forcing,
            'autoregressive_accuracies': reconstruction_accuracies_autoregressive,
            'bleu_scores': bleu_scores
        }
    
    def _generate_autoregressive(self, embedding: torch.Tensor, max_length: int = 50) -> List[int]:
        """Generate text autoregressively from embedding"""
        self.text_decoder.eval()
        
        with torch.no_grad():
            try:
                # Start with [CLS] token
                cls_token_id = self.text_encoder.tokenizer.cls_token_id
                sep_token_id = self.text_encoder.tokenizer.sep_token_id
                pad_token_id = self.text_encoder.tokenizer.pad_token_id
                
                if cls_token_id is None:
                    cls_token_id = self.text_encoder.tokenizer.bos_token_id
                if sep_token_id is None:
                    sep_token_id = self.text_encoder.tokenizer.eos_token_id
                
                generated_tokens = [cls_token_id]
                
                for _ in range(max_length - 1):
                    # Current input sequence
                    input_tokens = torch.tensor([generated_tokens], device=embedding.device)
                    
                    # Get decoder output
                    decoder_output = self.text_decoder(embedding, input_tokens)
                    
                    # Get next token prediction (last position)
                    next_token_logits = decoder_output[0, -1, :]
                    
                    # Sample next token with temperature (better than greedy)
                    temperature = 0.8  # Slightly random for diversity
                    next_token_probs = F.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(next_token_probs, 1).item()
                    
                    # Stop if [SEP] token is generated or if it's a pad token
                    if next_token == sep_token_id or next_token == pad_token_id:
                        break
                        
                    generated_tokens.append(next_token)
                
                # Ensure [SEP] token at the end if not already present
                if sep_token_id is not None and generated_tokens[-1] != sep_token_id:
                    generated_tokens.append(sep_token_id)
                
            except Exception as e:
                print(f"Autoregressive generation error: {e}")
                # Return minimal valid sequence
                generated_tokens = [cls_token_id, sep_token_id] if cls_token_id and sep_token_id else [0, 1]
        
        self.text_decoder.train()
        return generated_tokens
    
    def evaluate_similarity_matrix(self, sensor_interpretations: List[str],
                                 activity_interpretations: List[str],
                                 max_length: int = 50,
                                 save_path: str = "outputs/similarity_matrix.png"):
        """Similarity matrix visualization"""
        
        # Ensure max_length is an integer
        max_length = int(max_length) if max_length is not None else 50
        
        sensor_interpretations = sensor_interpretations[:max_length]
        activity_interpretations = activity_interpretations[:max_length]

        with torch.no_grad():
            sensor_embeddings = self.text_encoder(sensor_interpretations)
            activity_embeddings = self.text_encoder(activity_interpretations)
            
            sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=1)
            activity_embeddings = F.normalize(activity_embeddings, p=2, dim=1)

            similarity_matrix = torch.matmul(sensor_embeddings, activity_embeddings.T)
            similarity_matrix_np = similarity_matrix.detach().cpu().numpy()
        
        # Visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix_np, cmap='RdYlBu_r', center=0, square=True)
        
        plt.title('Sensor-Activity Similarity Matrix')
        plt.xlabel('Activity Interpretations')
        plt.ylabel('Sensor Interpretations')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Similarity matrix saved: {save_path}")
    
    def analyze_pairwise_similarities(self, sensor_interpretations: List[str], 
                                    activity_interpretations: List[str],
                                    activities: List[str]) -> Dict:
        """Analyze pair-wise similarities between matched sensor-activity pairs"""
        
        with torch.no_grad():
            sensor_embeddings = self.text_encoder(sensor_interpretations)
            activity_embeddings = self.text_encoder(activity_interpretations)
            
            sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=1)
            activity_embeddings = F.normalize(activity_embeddings, p=2, dim=1)
            
            correct_pair_similarities = torch.sum(sensor_embeddings * activity_embeddings, dim=1)
            similarity_matrix = torch.matmul(sensor_embeddings, activity_embeddings.T)
            mask = torch.eye(len(sensor_embeddings), device=self.device).bool()
            off_diagonal_similarities = similarity_matrix[~mask].view(len(sensor_embeddings), -1)
            
            # Statistics
            avg_correct_similarity = correct_pair_similarities.mean().item()
            std_correct_similarity = correct_pair_similarities.std().item()
            avg_incorrect_similarity = off_diagonal_similarities.mean().item()
            std_incorrect_similarity = off_diagonal_similarities.std().item()
            
            # Per-activity analysis
            activity_stats = {}
            unique_activities = list(set(activities))
            for activity in unique_activities:
                activity_indices = [i for i, a in enumerate(activities) if a == activity]
                if len(activity_indices) > 1:
                    activity_similarities = correct_pair_similarities[activity_indices]
                    activity_stats[activity] = {
                        'count': len(activity_indices),
                        'avg_similarity': activity_similarities.mean().item(),
                        'std_similarity': activity_similarities.std().item()
                    }
        
        return {
            'avg_correct_pair_similarity': avg_correct_similarity,
            'std_correct_pair_similarity': std_correct_similarity,
            'avg_incorrect_pair_similarity': avg_incorrect_similarity,
            'std_incorrect_pair_similarity': std_incorrect_similarity,
            'similarity_margin': avg_correct_similarity - avg_incorrect_similarity,
            'activity_stats': activity_stats,
            'correct_pair_similarities': correct_pair_similarities.detach().cpu().numpy().tolist()
        }
    
    def visualize_embeddings(self, sensor_interpretations: List[str],
                            activity_interpretations: List[str],
                            activities: List[str],
                            save_path: str = "outputs/embedding_visualization.png"):
        """3D Embedding visualization with activity-based coloring"""
        
        # Create unique activity interpretations for visualization only
        unique_activities = list(set(activities))
        unique_activity_interpretations = [activity_interpretations[activities.index(a)] for a in unique_activities]

        with torch.no_grad():
            sensor_embeddings = self.text_encoder(sensor_interpretations)
            activity_embeddings = self.text_encoder(unique_activity_interpretations)

            sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=1)
            activity_embeddings = F.normalize(activity_embeddings, p=2, dim=1)

        all_embeddings = torch.cat([sensor_embeddings, activity_embeddings], dim=0)
        all_embeddings_np = all_embeddings.detach().cpu().numpy()
        
        tsne = TSNE(n_components=3, random_state=42, perplexity=8)
        embeddings_3d = tsne.fit_transform(all_embeddings_np)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_activities)))
        activity_to_color = dict(zip(unique_activities, colors))
        
        fig = plt.figure(figsize=(20, 15))
        ax1 = fig.add_subplot(221, projection='3d')
        
        sensor_3d = embeddings_3d[:len(sensor_interpretations)]
        activity_3d = embeddings_3d[len(sensor_interpretations):]

        for i, activity in enumerate(activities):
            color = activity_to_color[activity]
            ax1.scatter(sensor_3d[i, 0], sensor_3d[i, 1], sensor_3d[i, 2],
                        c=[color], alpha=0.75, s=50, edgecolors='k', linewidth=0.3)

        for i, activity in enumerate(unique_activities):
            color = activity_to_color[activity]
            ax1.scatter(activity_3d[i, 0], activity_3d[i, 1], activity_3d[i, 2],
                        c=[color], alpha=0.95, s=120, marker='^', edgecolors='k', linewidth=0.8)
        
        ax1.set_title('3D Text Encoder Embeddings')
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')
        ax1.set_zlabel('t-SNE Dimension 3')
        ax1.grid(True, alpha=0.2)

        # XY plane
        ax2 = fig.add_subplot(222)
        for i, activity in enumerate(activities):
            color = activity_to_color[activity]
            ax2.scatter(sensor_3d[i, 0], sensor_3d[i, 1], 
                    c=[color], alpha=0.7, s=60, edgecolors='k', linewidth=0.3)
            
        for i, activity in enumerate(unique_activities):
            color = activity_to_color[activity]
            ax2.scatter(activity_3d[i, 0], activity_3d[i, 1], 
                    c=[color], alpha=0.9, s=100, marker='^', edgecolors='k', linewidth=1)
            
        ax2.set_title('XY Plane Projection')
        ax2.set_xlabel('t-SNE Dimension 1')
        ax2.set_ylabel('t-SNE Dimension 2')
        ax2.grid(True, alpha=0.3)
        
        # XZ plane
        ax3 = fig.add_subplot(223)
        for activity in unique_activities:
            activity_indices = [i for i, a in enumerate(activities) if a == activity]
            if activity_indices:
                color = activity_to_color[activity]
                sensor_points = sensor_3d[activity_indices]
                ax3.scatter(sensor_points[:, 0], sensor_points[:, 2], 
                           c=[color], alpha=0.7, s=60, edgecolors='k', linewidth=0.3)
        
        for i, activity in enumerate(unique_activities):
            color = activity_to_color[activity]
            ax3.scatter(activity_3d[i, 0], activity_3d[i, 2], 
                       c=[color], alpha=0.9, s=100, marker='^', edgecolors='k', linewidth=1)
        
        ax3.set_title('XZ Plane Projection')
        ax3.set_xlabel('t-SNE Dimension 1')
        ax3.set_ylabel('t-SNE Dimension 3')
        ax3.grid(True, alpha=0.3)
        
        # YZ plane
        ax4 = fig.add_subplot(224)
        for activity in unique_activities:
            activity_indices = [i for i, a in enumerate(activities) if a == activity]
            if activity_indices:
                color = activity_to_color[activity]
                sensor_points = sensor_3d[activity_indices]
                ax4.scatter(sensor_points[:, 1], sensor_points[:, 2], 
                           c=[color], alpha=0.7, s=60, edgecolors='k', linewidth=0.3)
        
        for i, activity in enumerate(unique_activities):
            color = activity_to_color[activity]
            ax4.scatter(activity_3d[i, 1], activity_3d[i, 2], 
                       c=[color], alpha=0.9, s=100, marker='^', edgecolors='k', linewidth=1)
        
        ax4.set_title('YZ Plane Projection')
        ax4.set_xlabel('t-SNE Dimension 2')
        ax4.set_ylabel('t-SNE Dimension 3')
        ax4.grid(True, alpha=0.3)
        
        # Legend
        handles = []
        for activity in unique_activities:
            color = activity_to_color[activity]
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=color, markeredgecolor='k',
                                    markersize=8, label=activity))

        handles.append(plt.Line2D([0], [0], marker='o', color='k', label='Sensor', markersize=6))
        handles.append(plt.Line2D([0], [0], marker='^', color='k', label='Activity', markersize=6))
        
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False, fontsize=9)

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 3D Embedding visualization saved: {save_path}")

    def analyze_category_similarities(self, sensor_interpretations: List[str], 
                                    activity_interpretations: List[str],
                                    activities: List[str]) -> Dict:
        """Analyze similarities within and across activity categories"""
        
        # Activity to category mapping (same as in dataset)
        activity_categories = {
            'Sleeping': 0,
            'Toileting': 1, 
            'Showering': 1,
            'Breakfast': 2,
            'Lunch': 2,
            'Dinner': 2,
            'Grooming': 1,
            'Spare_Time/TV': 3,
            'Leaving': 4,
            'Snack': 3
        }
        
        with torch.no_grad():
            sensor_embeddings = self.text_encoder(sensor_interpretations)
            activity_embeddings = self.text_encoder(activity_interpretations)
            
            sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=1)
            activity_embeddings = F.normalize(activity_embeddings, p=2, dim=1)
            
            # Get categories for each sample
            categories = torch.tensor([activity_categories.get(activity, 0) for activity in activities], device=self.device)
            
            # Calculate similarities within same categories
            same_category_similarities = []
            different_category_similarities = []
            
            for i in range(len(sensor_embeddings)):
                for j in range(i + 1, len(sensor_embeddings)):
                    sim = torch.sum(sensor_embeddings[i] * activity_embeddings[j])
                    if categories[i] == categories[j]:
                        same_category_similarities.append(sim.item())
                    else:
                        different_category_similarities.append(sim.item())
            
            # Statistics
            avg_same_category = np.mean(same_category_similarities) if same_category_similarities else 0
            std_same_category = np.std(same_category_similarities) if same_category_similarities else 0
            avg_different_category = np.mean(different_category_similarities) if different_category_similarities else 0
            std_different_category = np.std(different_category_similarities) if different_category_similarities else 0
            
            # Per-category analysis
            category_stats = {}
            for category_id in range(5):  # 0-4 categories
                category_indices = [i for i, cat in enumerate(categories) if cat == category_id]
                if len(category_indices) > 1:
                    category_activities = [activities[i] for i in category_indices]
                    category_similarities = []
                    for i in category_indices:
                        for j in category_indices:
                            if i != j:
                                sim = torch.sum(sensor_embeddings[i] * activity_embeddings[j])
                                category_similarities.append(sim.item())
                    
                    category_stats[f'category_{category_id}'] = {
                        'activities': category_activities,
                        'count': len(category_indices),
                        'avg_similarity': np.mean(category_similarities) if category_similarities else 0,
                        'std_similarity': np.std(category_similarities) if category_similarities else 0
                    }
        
        return {
            'avg_same_category_similarity': avg_same_category,
            'std_same_category_similarity': std_same_category,
            'avg_different_category_similarity': avg_different_category,
            'std_different_category_similarity': std_different_category,
            'category_margin': avg_same_category - avg_different_category,
            'category_stats': category_stats,
            'same_category_count': len(same_category_similarities),
            'different_category_count': len(different_category_similarities)
        }
    
    def create_detailed_visualizations(self, sensor_interpretations: List[str], 
                                     activity_interpretations: List[str],
                                     activities: List[str], output_dir: str):
        """Create detailed visualizations for analysis"""
        
        # Activity to category mapping
        activity_categories = {
            'Sleeping': 0,
            'Toileting': 1, 
            'Showering': 1,
            'Breakfast': 2,
            'Lunch': 2,
            'Dinner': 2,
            'Grooming': 1,
            'Spare_Time/TV': 3,
            'Leaving': 4,
            'Snack': 3
        }
        
        with torch.no_grad():
            # Generate embeddings
            sensor_embeddings = self.text_encoder(sensor_interpretations)
            activity_embeddings = self.text_encoder(activity_interpretations)
            
            # Normalize embeddings
            sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=1)
            activity_embeddings = F.normalize(activity_embeddings, p=2, dim=1)
            
            # Calculate similarities
            similarity_matrix = torch.matmul(sensor_embeddings, activity_embeddings.T)
            
            # 1. Pair-wise similarity distribution
            correct_similarities = torch.diag(similarity_matrix)
            off_diagonal_mask = ~torch.eye(len(similarity_matrix), device=self.device).bool()
            incorrect_similarities = similarity_matrix[off_diagonal_mask]
            
            plt.figure(figsize=(12, 8))
            plt.hist(correct_similarities.detach().cpu().numpy(), bins=20, alpha=0.7, 
                    label='Correct Pairs', color='green', density=True)
            plt.hist(incorrect_similarities.detach().cpu().numpy(), bins=20, alpha=0.7, 
                    label='Incorrect Pairs', color='red', density=True)
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Density')
            plt.title('Distribution of Pair-wise Similarities')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "pair_similarity_distribution.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Category-based similarity analysis
            categories = [activity_categories.get(activity, 0) for activity in activities]
            category_colors = ['blue', 'green', 'red', 'orange', 'purple']
            
            plt.figure(figsize=(15, 10))
            
            # Create subplots for each category
            for category_id in range(5):
                plt.subplot(2, 3, category_id + 1)
                category_indices = [i for i, cat in enumerate(categories) if cat == category_id]
                
                if len(category_indices) > 1:
                    category_similarities = []
                    for i in category_indices:
                        for j in category_indices:
                            if i != j:
                                category_similarities.append(similarity_matrix[i, j].item())
                    
                    if category_similarities:
                        plt.hist(category_similarities, bins=10, alpha=0.7, 
                                color=category_colors[category_id], density=True)
                        plt.title(f'Category {category_id} Similarities')
                        plt.xlabel('Similarity')
                        plt.ylabel('Density')
                        plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, "category_similarity_analysis.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Category similarity analysis saved: {save_path}")
            
            # 3. Activity-specific similarity heatmap
            unique_activities = list(set(activities))
            activity_matrix = np.zeros((len(unique_activities), len(unique_activities)))
            
            for i, act1 in enumerate(unique_activities):
                for j, act2 in enumerate(unique_activities):
                    indices1 = [k for k, a in enumerate(activities) if a == act1]
                    indices2 = [k for k, a in enumerate(activities) if a == act2]
                    
                    if indices1 and indices2:
                        similarities = []
                        for idx1 in indices1:
                            for idx2 in indices2:
                                similarities.append(similarity_matrix[idx1, idx2].item())
                        activity_matrix[i, j] = np.mean(similarities)
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(activity_matrix, 
                       xticklabels=unique_activities, 
                       yticklabels=unique_activities,
                       annot=True, fmt='.3f', cmap='RdYlBu_r', center=0)
            plt.title('Activity-to-Activity Similarity Matrix')
            plt.xlabel('Activity Interpretations')
            plt.ylabel('Sensor Interpretations')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            save_path = os.path.join(output_dir, "activity_similarity_heatmap.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Activity similarity heatmap saved: {save_path}")
    
    def comprehensive_evaluation(self, interpretations_file: str, output_dir: str = "outputs", 
                               dataset_name: str = None) -> Dict:
        """Comprehensive evaluation with cross-domain support"""

        # Data loading
        with open(interpretations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sensor_interpretations = []
        activity_interpretation = {}
        activities = []
        
        print(f"✓ Using {self.config.source_dataset} val split for evaluation")
        dataset_name = dataset_name or self.config.source_dataset

        if data['sensor_interpretations']['val']:
            for _, window_data in data['sensor_interpretations']['val'].items():
                if 'interpretation' in window_data:
                    sensor_interpretations.append(window_data['interpretation'])
                    activities.append(window_data['activity'])
        
        # Activity interpretations extraction and proper matching
        for activity, interpretation_data in data.get('activity_interpretations', {}).items():
            if 'interpretation' in interpretation_data:
                activity_interpretation[activity] = interpretation_data['interpretation']
        
        # Match sensor interpretations with their corresponding activity interpretations
        matched_activity_interpretations = []
        for activity in activities:
            if activity in activity_interpretation:
                matched_activity_interpretations.append(activity_interpretation[activity])
            else:
                raise ValueError("✗ No matched sensor-activity interpretation pair")

        max_samples = min(80, len(sensor_interpretations))
        print(f"  Using {max_samples} samples in total for evaluation of {len(sensor_interpretations)} samples")

        sensor_interpretations = sensor_interpretations[:max_samples]
        activities = activities[:max_samples]
        matched_activity_interpretations = matched_activity_interpretations[:max_samples]

        print(f" Evaluation data: {len(sensor_interpretations)} sensors, {len(matched_activity_interpretations)} activities")
        
        # 1. Alignment quality evaluation
        alignment_results = self.evaluate_alignment_quality(
            sensor_interpretations, matched_activity_interpretations
        )
        
        # 2. Reconstruction quality evaluation
        reconstruction_results = self.evaluate_reconstruction_quality(
            sensor_interpretations
        )
        
        # 3. Visualization
        self.visualize_embeddings(
            sensor_interpretations, matched_activity_interpretations, activities,
            os.path.join(output_dir, "embedding_visualization.png")
        )
        
        self.evaluate_similarity_matrix(
            sensor_interpretations, matched_activity_interpretations,
            max_length=50,
            save_path=os.path.join(output_dir, "similarity_matrix.png")
        )
        
        # 4. Detailed pair-wise and category analysis
        pair_analysis = self.analyze_pairwise_similarities(
            sensor_interpretations, matched_activity_interpretations, activities
        )
        
        category_analysis = self.analyze_category_similarities(
            sensor_interpretations, matched_activity_interpretations, activities
        )
        
        # 5. Enhanced visualizations
        self.create_detailed_visualizations(
            sensor_interpretations, matched_activity_interpretations, activities,
            output_dir
        )
        
        # Comprehensive results
        results = {
            'alignment_quality': alignment_results,
            'reconstruction_quality': reconstruction_results,
            'pair_analysis': pair_analysis,
            'category_analysis': category_analysis,
            'evaluation_summary': {
                'accuracy': alignment_results['accuracy'],
                'margin': alignment_results['margin'],
                'reconstruction_loss': reconstruction_results['avg_reconstruction_loss'],
                'reconstruction_accuracy': reconstruction_results['avg_reconstruction_accuracy_teacher_forcing'],
                'reconstruction_accuracy_ar': reconstruction_results['avg_reconstruction_accuracy_autoregressive'],
                'bleu_score': reconstruction_results['avg_bleu_score'],
                'avg_pair_similarity': pair_analysis['avg_correct_pair_similarity'],
                'avg_category_similarity': category_analysis['avg_same_category_similarity']
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
        
        print(f"  ✓ Evaluation results saved: {results_file}")
        
        # Print detailed results table
        self.print_evaluation_results(results)
        
        return results
    
    def print_evaluation_results(self, results: Dict):
        """Print evaluation results in a formatted table"""
        
        # Main metrics
        summary = results['evaluation_summary']
        print(f"\nMAIN METRICS:")
        print(f"   Accuracy:                    {summary['accuracy']:.3f}")
        print(f"   Similarity Margin:          {summary['margin']:.3f}")
        print(f"   Reconstruction Loss:        {summary['reconstruction_loss']:.3f}")
        print(f"   Reconstruction Accuracy (TF): {summary['reconstruction_accuracy']:.3f}")
        print(f"   Reconstruction Accuracy (AR): {summary.get('reconstruction_accuracy_ar', 0.0):.3f}")
        print(f"   BLEU Score:                 {summary.get('bleu_score', 0.0):.3f}")
        print(f"   Avg Pair Similarity:        {summary['avg_pair_similarity']:.3f}")
        print(f"   Avg Category Similarity:    {summary['avg_category_similarity']:.3f}")
        
        # Pair-wise analysis
        pair_analysis = results['pair_analysis']
        print(f"\nPAIR-WISE ANALYSIS:")
        print(f"   Correct Pairs Similarity:   {pair_analysis['avg_correct_pair_similarity']:.3f} ± {pair_analysis['std_correct_pair_similarity']:.3f}")
        print(f"   Incorrect Pairs Similarity: {pair_analysis['avg_incorrect_pair_similarity']:.3f} ± {pair_analysis['std_incorrect_pair_similarity']:.3f}")
        print(f"   Similarity Margin:          {pair_analysis['similarity_margin']:.3f}")
        
        # Activity-specific results
        print(f"\nACTIVITY-SPECIFIC RESULTS:")
        activity_stats = pair_analysis['activity_stats']
        for activity, stats in activity_stats.items():
            print(f"   {activity:15}: {stats['avg_similarity']:.3f} ± {stats['std_similarity']:.3f} (n={stats['count']})")
        
        # Category analysis
        category_analysis = results['category_analysis']
        print(f"\nCATEGORY ANALYSIS:")
        print(f"   Same Category Similarity:   {category_analysis['avg_same_category_similarity']:.3f} ± {category_analysis['std_same_category_similarity']:.3f}")
        print(f"   Different Category Similarity: {category_analysis['avg_different_category_similarity']:.3f} ± {category_analysis['std_different_category_similarity']:.3f}")
        print(f"   Category Margin:            {category_analysis['category_margin']:.3f}")
        
        # Category-specific results
        print(f"\nCATEGORY-SPECIFIC RESULTS:")
        category_stats = category_analysis['category_stats']
        category_names = {
            'category_0': 'Sleeping',
            'category_1': 'Bathroom (Toileting, Showering, Grooming)', 
            'category_2': 'Meals (Breakfast, Lunch, Dinner)',
            'category_3': 'Entertainment (Spare_Time/TV, Snack)',
            'category_4': 'Leaving'
        }
        
        for category_key, stats in category_stats.items():
            category_name = category_names.get(category_key, category_key)
            activities_str = ', '.join(stats['activities'][:3])  # Show first 3 activities
            if len(stats['activities']) > 3:
                activities_str += f" (+{len(stats['activities'])-3} more)"
            print(f"   {category_name:25}: {stats['avg_similarity']:.3f} ± {stats['std_similarity']:.3f} ({activities_str})")
        
        # Reconstruction analysis
        print(f"\nRECONSTRUCTION ANALYSIS:")
        print(f"   Teacher Forcing Accuracy:   {summary['reconstruction_accuracy']:.3f} (training-like, optimistic)")
        print(f"   Autoregressive Accuracy:    {summary['reconstruction_accuracy_ar']:.3f} (realistic, pessimistic)")
        print(f"   BLEU Score (semantic):      {summary['bleu_score']:.3f} (word overlap)")
