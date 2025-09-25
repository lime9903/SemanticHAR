"""
Text Encoder 학습 검증을 위한 평가 모듈
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Optional
import json
import os
from tqdm import tqdm

from models.text_encoder import TextEncoder, TextDecoder, TextEncoderTrainer
from config import LanHARConfig

class TextEncoderEvaluator:
    """Text Encoder 학습 검증 클래스"""
    
    def __init__(self, config: LanHARConfig, text_encoder: Optional[TextEncoder] = None, model_path: Optional[str] = None):
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

def test_text_encoder_evaluation():
    """Text Encoder 평가 테스트"""
    from config import LanHARConfig
    
    config = LanHARConfig()
    
    # 모델 경로 확인
    model_path = "checkpoints/text_encoder_trained.pth"
    if not os.path.exists(model_path):
        print("⚠️  학습된 모델이 없습니다. 랜덤 초기화된 모델로 테스트합니다.")
        model_path = None
    
    # 평가기 초기화
    evaluator = TextEncoderEvaluator(config, model_path)
    
    # Interpretations 파일 경로
    interpretations_file = "outputs/batch_semantic_interpretations_20250922_091130.json"
    
    if os.path.exists(interpretations_file):
        # 종합 평가 실행
        results = evaluator.comprehensive_evaluation(interpretations_file)
        
        # 결과 출력
        print("\n" + "="*50)
        print("📊 TEXT ENCODER 평가 결과")
        print("="*50)
        print(f"정확도: {results['evaluation_summary']['accuracy']:.3f}")
        print(f"마진: {results['evaluation_summary']['margin']:.3f}")
        print(f"재구성 손실: {results['evaluation_summary']['reconstruction_loss']:.3f}")
        print(f"재구성 정확도: {results['evaluation_summary']['reconstruction_accuracy']:.3f}")
        print("="*50)
        
    else:
        print(f"❌ Interpretations 파일을 찾을 수 없습니다: {interpretations_file}")

if __name__ == "__main__":
    test_text_encoder_evaluation()
