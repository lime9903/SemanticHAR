"""
BERT-based text encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
try:
    from transformers import BertModel, BertTokenizer
except ImportError:
    BertModel = None
    BertTokenizer = None
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
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
        
        # BERT model loading
        if BertModel is None or BertTokenizer is None:
            raise ImportError("transformers library is not installed. run pip install transformers.")
        
        self.bert = BertModel.from_pretrained(config.text_encoder_model)
        self.tokenizer = BertTokenizer.from_pretrained(config.text_encoder_model)
        
        # output dimension
        self.hidden_dim = self.bert.config.hidden_size
        self.output_dim = config.sensor_encoder_hidden_dim
        
        # projection layer
        self.projection = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(0.1)
        
        # normalization
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
    def forward(self, texts: List[str]) -> torch.Tensor:
        """convert text to embedding"""
        # tokenization
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_tensors='pt'
        )
        
        # move to device
        input_ids = encoded['input_ids'].to(next(self.parameters()).device)
        attention_mask = encoded['attention_mask'].to(next(self.parameters()).device)
        
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # use [CLS] token embedding
        pooled_output = outputs.pooler_output
        
        # projection and normalization
        projected = self.projection(pooled_output)
        projected = self.dropout(projected)
        projected = self.layer_norm(projected)
        
        return projected
    
    def encode_single(self, text: str) -> torch.Tensor:
        """single text encoding"""
        return self.forward([text])

class TextDecoder(nn.Module):
    """decoder for text reconstruction"""
    
    def __init__(self, config: SemanticHARConfig, vocab_size: int = 30522):
        super(TextDecoder, self).__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.sensor_encoder_hidden_dim,
            nhead=config.sensor_encoder_heads,
            dim_feedforward=config.sensor_encoder_hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=config.sensor_encoder_layers
        )
        
        # output layer
        self.output_projection = nn.Linear(
            config.sensor_encoder_hidden_dim, 
            vocab_size
        )
        
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, config.sensor_encoder_hidden_dim)
        self.positional_encoding = nn.Parameter(
            torch.randn(1000, config.sensor_encoder_hidden_dim)
        )
        
    def forward(self, memory: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """text reconstruction"""
        batch_size, seq_len = target_ids.shape
        
        # target embedding
        target_embeddings = self.embedding(target_ids)
        
        # add positional encoding
        positions = torch.arange(seq_len, device=target_ids.device)
        target_embeddings = target_embeddings + self.positional_encoding[:seq_len].unsqueeze(0)
        
        # generate mask (autoregressive generation)
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(target_ids.device)
        
        # transformer decoder
        decoder_output = self.transformer_decoder(
            target_embeddings,
            memory.unsqueeze(1),  # use memory as query
            tgt_mask=tgt_mask
        )
        
        # output projection
        output = self.output_projection(decoder_output)
        
        return output
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """mask for autoregressive generation"""
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
        """alignment loss between sensor embeddings and activity embeddings"""
        # normalization
        sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=1)
        activity_embeddings = F.normalize(activity_embeddings, p=2, dim=1)
        
        # calculate similarity
        similarity_matrix = torch.matmul(sensor_embeddings, activity_embeddings.T) / self.temperature
        
        # batch size
        batch_size = sensor_embeddings.size(0)
        
        # labels (diagonal is the answer)
        labels = torch.arange(batch_size, device=sensor_embeddings.device)
        
        # symmetric loss
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
        
        # initialize models
        self.text_encoder = TextEncoder(config).to(self.device)
        self.text_decoder = TextDecoder(config).to(self.device)
        self.contrastive_module = ContrastiveLearningModule(config).to(self.device)
        
        print(f"⨠ Models moved to {self.device}")
        
        # optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.text_encoder.parameters()) + 
            list(self.text_decoder.parameters()),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
    def train_step(self, sensor_interpretations: List[str], 
                   activity_interpretations: List[str],
                   sensor_categories: torch.Tensor,
                   activity_categories: torch.Tensor,
                   activity_labels: torch.Tensor) -> Dict[str, float]:
        """training step"""
        
        self.optimizer.zero_grad()
        
        # Move tensors to device
        sensor_categories = sensor_categories.to(self.device)
        activity_categories = activity_categories.to(self.device)
        activity_labels = activity_labels.to(self.device)
        
        # text encoding
        sensor_embeddings = self.text_encoder(sensor_interpretations)
        activity_embeddings = self.text_encoder(activity_interpretations)
        
        # alignment loss
        alignment_loss = self.contrastive_module.alignment_loss(
            sensor_embeddings, activity_embeddings
        )
        
        # category contrastive loss
        sensor_category_loss = self.contrastive_module.category_contrastive_loss(
            sensor_embeddings, sensor_categories
        )
        
        activity_category_loss = self.contrastive_module.category_contrastive_loss(
            activity_embeddings, activity_categories
        )
        
        # activity contrastive loss
        activity_contrastive_loss = self.contrastive_module.activity_contrastive_loss(
            activity_embeddings, activity_labels
        )
        
        # reconstruction loss
        reconstruction_loss = self._compute_reconstruction_loss(
            sensor_embeddings, sensor_interpretations
        )
        
        # total loss
        total_loss = (
            alignment_loss + 
            self.config.alpha * (sensor_category_loss + activity_category_loss + activity_contrastive_loss) +
            self.config.beta * reconstruction_loss
        )
        
        # backpropagation
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.text_encoder.parameters()) + list(self.text_decoder.parameters()),
            max_norm=1.0
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
        from torch.utils.data import DataLoader
        import numpy as np
        from tqdm import tqdm
        
        # Use local prepare_training_data function
        
        print("=" * 50)
        print("Starting Text Encoder Training with Semantic Interpretations")
        print("=" * 50)
        
        # Prepare data (train only for training)
        dataset = prepare_training_data(interpretations_file, use_validation=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Batch size: {batch_size}")
        print(f"Number of epochs: {num_epochs}")
        
        print(f"\nTraining Text Encoder...")
        print("=" * 40)
        
        # Early stopping variables initialization
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
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
                avg_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
                
                # Early stopping logic
                if early_stopping:
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        patience_counter = 0
                        print(f"  → New best loss: {best_loss:.4f}")
                    else:
                        patience_counter += 1
                        print(f"  → No improvement ({patience_counter}/{patience})")
                        if patience_counter >= patience:
                            print(f"Early stopping triggered at epoch {epoch+1}")
                            break
            else:
                print(f"⨺ Epoch {epoch+1}/{num_epochs} - No valid batches processed")
        
        print("\nText Encoder Training Completed!")
        
        # Validation evaluation
        if early_stopping:
            print("\nEvaluating on validation set...")
            try:
                val_dataset = prepare_training_data(interpretations_file, use_validation=True)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                val_losses = []
                self.text_encoder.eval()
                
                for batch in val_dataloader:
                    try:
                        sensor_interpretations = batch['sensor_interpretation']
                        activity_interpretations = batch['activity_interpretation']
                        activity_categories = batch['activity_category'].clone().detach().to(self.device)
                        activity_indices = batch['activity_idx'].clone().detach().to(self.device)
                        
                        with torch.no_grad():
                            # Simple validation: just encode and compute similarity
                            sensor_embeddings = self.text_encoder(sensor_interpretations)
                            activity_embeddings = self.text_encoder(activity_interpretations)
                            
                            # Compute cosine similarity as a simple validation metric
                            similarity = torch.cosine_similarity(sensor_embeddings, activity_embeddings, dim=1)
                            val_loss = 1.0 - similarity.mean()  # Convert similarity to loss
                            val_losses.append(val_loss.item())
                            
                    except Exception as e:
                        print(f"⨺ Validation batch error: {e}")
                        continue
                
                if val_losses:
                    avg_val_loss = np.mean(val_losses)
                    print(f"Validation Loss: {avg_val_loss:.4f}")
                else:
                    print("⨺ No valid validation batches")
                    
            except Exception as e:
                print(f"⨺ Validation evaluation failed: {e}")
        
        # Model saving
        import os
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        model_path = os.path.join(checkpoint_dir, "text_encoder_trained.pth")
        torch.save(self.text_encoder.state_dict(), model_path)
        print(f"✓ Text encoder saved to: {model_path}")
        
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


def load_interpretations_from_json(json_file: str, use_validation: bool = False) -> Tuple[List[str], List[str], List[str]]:
    """Load interpretations from JSON file"""
    import json
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sensor_interpretations = []
    activity_interpretations = []
    activities = []
    
    # Sensor interpretations load (nested structure handling)
    for home_id in data['sensor_interpretations']:
        for split in data['sensor_interpretations'][home_id]:
            # 학습 시에는 train만 사용, 검증 시에는 val도 사용
            if split == 'train' or (use_validation and split == 'val'):
                for window_id, window_data in data['sensor_interpretations'][home_id][split].items():
                    if 'interpretation' in window_data and 'error' not in window_data:
                        sensor_interpretations.append(window_data['interpretation'])
                        activities.append(window_data['activity'])
    
    # Activity interpretations load
    for activity, activity_data in data['activity_interpretations'].items():
        if 'interpretation' in activity_data and 'error' not in activity_data:
            activity_interpretations.append(activity_data['interpretation'])
    
    print(f"Loaded {len(sensor_interpretations)} sensor interpretations")
    print(f"Loaded {len(activity_interpretations)} activity interpretations")
    print(f"Unique activities: {list(set(activities))}")
    
    return sensor_interpretations, activity_interpretations, activities


def prepare_training_data(interpretations_file: str, use_validation: bool = False) -> InterpretationDataset:
    """Prepare training data"""
    print(f"Loading interpretations from: {interpretations_file}")
    print(f"Using validation data: {use_validation}")
    
    # Interpretations load
    sensor_interpretations, activity_interpretations, activities = load_interpretations_from_json(interpretations_file, use_validation)
    
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
    """Text Encoder 학습 검증 클래스"""
    
    def __init__(self, config: SemanticHARConfig, text_encoder: Optional[TextEncoder] = None, model_path: Optional[str] = None):
        self.config = config
        self.device = torch.device(config.device)
        
        # 모델 로드
        if text_encoder is not None:
            self.text_encoder = text_encoder
            self.text_decoder = TextDecoder(config).to(self.device)
            print(f"✅ TextEncoder 객체를 직접 사용합니다.")
        elif model_path and os.path.exists(model_path):
            self.text_encoder = TextEncoder(config).to(self.device)
            self.text_decoder = TextDecoder(config).to(self.device)
            self.load_model(model_path)
            print(f"✅ 모델 로드 완료: {model_path}")
        else:
            self.text_encoder = TextEncoder(config).to(self.device)
            self.text_decoder = TextDecoder(config).to(self.device)
            print("⚠️  모델 파일이 없습니다. 랜덤 초기화된 모델을 사용합니다.")
    
    def load_model(self, model_path: str):
        """학습된 모델 로드"""
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'text_encoder' in checkpoint:
            self.text_encoder.load_state_dict(checkpoint['text_encoder'])
        else:
            self.text_encoder.load_state_dict(checkpoint)
    
    def evaluate_alignment_quality(self, sensor_interpretations: List[str], 
                                 activity_interpretations: List[str]) -> Dict[str, float]:
        """Sensor-Activity 정렬 품질 평가"""
        print("🔍 Sensor-Activity 정렬 품질 평가 중...")
        
        # 임베딩 생성
        sensor_embeddings = self.text_encoder(sensor_interpretations)
        activity_embeddings = self.text_encoder(activity_interpretations)
        
        # 정규화
        sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=1)
        activity_embeddings = F.normalize(activity_embeddings, p=2, dim=1)
        
        # 유사도 행렬 계산
        similarity_matrix = torch.matmul(sensor_embeddings, activity_embeddings.T)
        
        # 대각선 요소 (정답 쌍)의 유사도
        diagonal_similarities = torch.diag(similarity_matrix)
        
        # 각 행에서 최고 유사도 (정답이 최고인지 확인)
        max_similarities, max_indices = torch.max(similarity_matrix, dim=1)
        
        # 정답률 계산
        correct_predictions = (max_indices == torch.arange(len(sensor_interpretations), device=self.device)).float()
        accuracy = correct_predictions.mean().item()
        
        # 평균 정답 쌍 유사도
        avg_correct_similarity = diagonal_similarities.mean().item()
        
        # 정답과 비정답 간 유사도 차이
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
        """재구성 품질 평가"""
        print("🔍 재구성 품질 평가 중...")
        
        # 임베딩 생성
        embeddings = self.text_encoder(texts)
        
        reconstruction_losses = []
        reconstruction_accuracies = []
        
        for i, text in enumerate(texts):
            try:
                # 토큰화
                tokens = self.text_encoder.tokenizer.encode(
                    text, 
                    add_special_tokens=True, 
                    max_length=self.config.max_sequence_length,
                    truncation=True
                )
                
                # 재구성 시도
                input_tokens = torch.tensor([tokens[:-1]], device=self.device)  # 마지막 토큰 제외
                target_tokens = torch.tensor([tokens[1:]], device=self.device)   # 첫 토큰 제외
                
                # 재구성
                decoder_output = self.text_decoder(embeddings[i:i+1], input_tokens)
                
                # 손실 계산
                loss = F.cross_entropy(
                    decoder_output.reshape(-1, decoder_output.size(-1)),
                    target_tokens.reshape(-1),
                    ignore_index=self.text_encoder.tokenizer.pad_token_id
                )
                
                reconstruction_losses.append(loss.item())
                
                # 정확도 계산 (간단한 버전)
                predicted_tokens = torch.argmax(decoder_output, dim=-1)
                accuracy = (predicted_tokens == target_tokens).float().mean().item()
                reconstruction_accuracies.append(accuracy)
                
            except Exception as e:
                print(f"재구성 오류 (텍스트 {i}): {e}")
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
        """임베딩 시각화"""
        print("📊 임베딩 시각화 생성 중...")
        
        # 임베딩 생성
        sensor_embeddings = self.text_encoder(sensor_interpretations)
        activity_embeddings = self.text_encoder(activity_interpretations)
        
        # t-SNE로 차원 축소
        all_embeddings = torch.cat([sensor_embeddings, activity_embeddings], dim=0)
        all_embeddings_np = all_embeddings.detach().cpu().numpy()
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings_np)-1))
        embeddings_2d = tsne.fit_transform(all_embeddings_np)
        
        # 시각화
        plt.figure(figsize=(15, 10))
        
        # Sensor embeddings
        sensor_2d = embeddings_2d[:len(sensor_interpretations)]
        scatter1 = plt.scatter(sensor_2d[:, 0], sensor_2d[:, 1], 
                              c='blue', alpha=0.6, s=50, label='Sensor Interpretations')
        
        # Activity embeddings
        activity_2d = embeddings_2d[len(sensor_interpretations):]
        scatter2 = plt.scatter(activity_2d[:, 0], activity_2d[:, 1], 
                              c='red', alpha=0.6, s=50, label='Activity Interpretations')
        
        # 연결선 그리기 (정답 쌍)
        for i in range(len(sensor_interpretations)):
            plt.plot([sensor_2d[i, 0], activity_2d[i, 0]], 
                    [sensor_2d[i, 1], activity_2d[i, 1]], 
                    'k--', alpha=0.3, linewidth=0.5)
        
        plt.title('Text Encoder Embeddings Visualization')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 저장
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 시각화 저장 완료: {save_path}")
    
    def evaluate_similarity_matrix(self, sensor_interpretations: List[str], 
                                 activity_interpretations: List[str],
                                 save_path: str = "outputs/similarity_matrix.png"):
        """유사도 행렬 시각화"""
        print("📊 유사도 행렬 시각화 생성 중...")
        
        # 임베딩 생성
        sensor_embeddings = self.text_encoder(sensor_interpretations)
        activity_embeddings = self.text_encoder(activity_interpretations)
        
        # 정규화
        sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=1)
        activity_embeddings = F.normalize(activity_embeddings, p=2, dim=1)
        
        # 유사도 행렬 계산
        similarity_matrix = torch.matmul(sensor_embeddings, activity_embeddings.T)
        similarity_matrix_np = similarity_matrix.detach().cpu().numpy()
        
        # 시각화
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
        
        # 저장
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 유사도 행렬 저장 완료: {save_path}")
    
    def comprehensive_evaluation(self, interpretations_file: str, 
                              output_dir: str = "outputs") -> Dict:
        """종합 평가"""
        print("🚀 Text Encoder 종합 평가 시작...")
        
        # 데이터 로드
        with open(interpretations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Sensor interpretations 추출
        sensor_interpretations = []
        activity_interpretations = []
        activities = []
        
        for home_id, home_data in data['sensor_interpretations'].items():
            for split, windows in home_data.items():
                if split == 'train':  # train 데이터만 사용
                    for window_id, window_data in windows.items():
                        if 'interpretation' in window_data:
                            sensor_interpretations.append(window_data['interpretation'])
                            activities.append(window_data['activity'])
        
        # Activity interpretations 추출
        for activity, interpretation_data in data.get('activity_interpretations', {}).items():
            if 'interpretation' in interpretation_data:
                activity_interpretations.append(interpretation_data['interpretation'])
        
        # 데이터 수 제한 (평가용)
        max_samples = min(50, len(sensor_interpretations))
        sensor_interpretations = sensor_interpretations[:max_samples]
        activities = activities[:max_samples]
        
        # Activity interpretations도 매칭
        activity_interpretations = activity_interpretations[:max_samples]
        
        print(f"📊 평가 데이터: {len(sensor_interpretations)}개 sensor, {len(activity_interpretations)}개 activity")
        
        # 1. 정렬 품질 평가
        alignment_results = self.evaluate_alignment_quality(
            sensor_interpretations, activity_interpretations
        )
        
        # 2. 재구성 품질 평가
        reconstruction_results = self.evaluate_reconstruction_quality(
            sensor_interpretations[:10]  # 재구성은 일부만 테스트
        )
        
        # 3. 시각화
        self.visualize_embeddings(
            sensor_interpretations, activity_interpretations, activities,
            os.path.join(output_dir, "embedding_visualization.png")
        )
        
        self.evaluate_similarity_matrix(
            sensor_interpretations, activity_interpretations,
            os.path.join(output_dir, "similarity_matrix.png")
        )
        
        # 결과 종합
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
        
        # 결과 저장 (numpy arrays를 리스트로 변환)
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
        
        print(f"✅ 평가 결과 저장: {results_file}")
        
        return results
