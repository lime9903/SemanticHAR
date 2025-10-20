#!/usr/bin/env python3
"""
SemanticHAR System Main Execution File
"""
import os
import sys
import argparse
import json
import torch

# Add project root to Python path
sys.path.append('/workspace/semantic')

from config import SemanticHARConfig, OPENAI_API_KEY
from dataloader.data_loader import load_sensor_data
from llm.semantic_generator import SemanticGenerator
from models.text_encoder import TextEncoder, TextEncoderTrainer, TextEncoderEvaluator
from models.sensor_encoder import SensorEncoderTrainer, SensorEncoderEvaluator, SensorEncoderInference


def setup_environment():
    """Setup environment"""
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("GPU not available, using CPU")
        print("   Training will be slower on CPU")
    
    # Check OpenAI API key
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY is not set.")
        print("   To use LLM features, set the API key.")
        print("   You can set it by running: export OPENAI_API_KEY='your-api-key-here'")
    else:
        print(f"OpenAI API key: {OPENAI_API_KEY[:8]}...")
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    print("Output directories created")


def train_lanhar(config: SemanticHARConfig):
    """SemanticHAR System Training Pipeline"""
    print('\n' + "=" * 60)
    print("SemanticHAR System Training Pipeline")
    print("=" * 60)
    
    # Initialize model instances
    text_encoder = None
    sensor_encoder = None
    
    # Step 1: Load sensor data
    print('\n' + "-" * 60)
    print("Step 1: Load raw sensor data")
    print("-" * 60)
    
    print(f"Source dataset: {config.source_dataset}")
    print(f"Target dataset: {config.target_dataset}")

    if os.path.exists(config.time_windows_file):
        print(f"Loading time windows from: {config.time_windows_file}")
        try:
            # Load time windows from JSON
            with open(config.time_windows_file, 'r') as f:
                time_windows_data = json.load(f)
            
            sensor_data = {
                'train': {},
                'val': {},
                'test': {}
            }
            
            for split in ['train', 'val', 'test']:
                if split in time_windows_data['time_windows']:
                    sensor_data[split] = time_windows_data['time_windows'][split]
            
            print(f"✓ Time windows loaded from JSON")
            
        except Exception as e:
            print(f"  ✗ Failed to load time windows from JSON: {e}")
            print("  Falling back to loading sensor data...")
            try:
                sensor_data = load_sensor_data(config)
                print("✓ Sensor data loaded successfully")

            except Exception as e2:
                print(f"✗ Failed to load sensor data: {e2}")
                return None, None
    else:
        try:
            sensor_data = load_sensor_data(config)
            print(f"✓ Sensor data loaded successfully")
            
        except Exception as e:
            print(f"✗ Failed to load sensor data: {e}")
            return None, None
    
    # Step 2: Generate semantic interpretations
    print('\n' + "-" * 60)
    print("Step 2: Generate semantic interpretations")
    print("-" * 60)

    if os.path.exists(config.semantic_interpretations_file):
        print(f"Loading semantic interpretations from: {config.semantic_interpretations_file}")
        
        # Verify the file contains valid interpretations
        try:
            with open(config.semantic_interpretations_file, 'r') as f:
                interpretations_data = json.load(f)
            
            # Count valid interpretations
            sensor_count = 0
            for home_id in interpretations_data.get('sensor_interpretations', {}):
                for split in interpretations_data['sensor_interpretations'][home_id]:
                    for window_id, window_data in interpretations_data['sensor_interpretations'][home_id][split].items():
                        if 'interpretation' in window_data and 'error' not in window_data:
                            sensor_count += 1
            
            activity_count = len([k for k, v in interpretations_data.get('activity_interpretations', {}).items() 
                                if 'interpretation' in v and 'error' not in v])
            
            print(f"✓ Loaded {sensor_count} sensor interpretations and {activity_count} activity interpretations")
            
        except Exception as e:
            print(f"✗ Failed to load semantic interpretations from JSON: {e}")
            print("  Falling back to generating new interpretations...")
            # Generate new interpretations using SemanticGenerator
            generator = SemanticGenerator(config)
            interpretations_file = generator.generate_interpretations(
                source_dataset=config.source_dataset,
                target_dataset=config.target_dataset,
                window_size_seconds=config.window_size_seconds,
                overlap_ratio=config.overlap_ratio
            )
            if not interpretations_file:
                print("✗ Cannot proceed without semantic interpretations")
                return None, None
    else:
        # Generate new interpretations using SemanticGenerator
        generator = SemanticGenerator(config)
        interpretations_file = generator.generate_interpretations(
            source_dataset=config.source_dataset,
            target_dataset=config.target_dataset,
            window_size_seconds=config.window_size_seconds,
            overlap_ratio=config.overlap_ratio
        )
        if not interpretations_file:
            print("✗ Cannot proceed without semantic interpretations")
            return None, None
    
    # Step 3: Train or load text encoder
    print("\n" + "-" * 60)
    print("Step 3-1: Text Encoder Training")
    print("-" * 60)
    
    try:
        # Check if trained text encoder exists
        text_encoder_checkpoint = "checkpoints/text_encoder_trained.pth"
        
        # Initialize TextEncoderTrainer
        text_encoder_trainer = TextEncoderTrainer(config)
        
        if os.path.exists(text_encoder_checkpoint):
            print(f"✓ Trained text encoder found: {text_encoder_checkpoint}")
            print("  Loading existing model weights...")
            
            # Load trained model
            text_encoder_trainer.text_encoder.load_state_dict(
                torch.load(text_encoder_checkpoint, map_location=config.device)
            )
            text_encoder = text_encoder_trainer.text_encoder
            text_encoder.eval()
            
            print("✓ Text encoder weights loaded successfully!")
            print(f"  Model location: {text_encoder_checkpoint}")
            
        else:
            print("✗ Trained text encoder not found.")
            print(f"  Expected location: {text_encoder_checkpoint}")
            print("  Starting new training...")
            
            # Train text encoder
            text_encoder = text_encoder_trainer.train_text_encoder(
                interpretations_file=config.semantic_interpretations_file,
                num_epochs=config.text_encoder_num_epochs,
                batch_size=config.text_encoder_batch_size,
                early_stopping=config.early_stopping,
                patience=config.patience
            )
        
        if not text_encoder:
            print("✗ Cannot proceed without trained text encoder")
            return None, None
            
    except Exception as e:
        print(f"✗ Error during text encoder loading/training: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # Evaluate text encoder
    print("\n" + "-" * 60)
    print("Step 3-2: Text Encoder Evaluation")
    print("-" * 60)
    
    try:
        evaluator = TextEncoderEvaluator(config, text_encoder=text_encoder)
        evaluation_results = evaluator.comprehensive_evaluation(config.semantic_interpretations_file)

    except Exception as e:
        print(f"✗ Error during text encoder evaluation: {e}")
        import traceback
        traceback.print_exc()

    
    # Step 4: Train sensor encoder
    print("\n" + "-" * 60)
    print("Step 4-1: Sensor Encoder Training")
    print("-" * 60)

    try:
        # Check if trained sensor encoder exists
        sensor_encoder_checkpoint = "checkpoints/sensor_encoder_trained.pth"
        
        # Initialize sensor encoder trainer
        sensor_trainer = SensorEncoderTrainer(config, text_encoder)
        
        if os.path.exists(sensor_encoder_checkpoint):
            print(f"✓ Trained sensor encoder found: {sensor_encoder_checkpoint}")
            print("  Loading existing model weights...")
            
            # Load trained model
            sensor_trainer.sensor_encoder.load_state_dict(
                torch.load(sensor_encoder_checkpoint, map_location=config.device)
            )
            sensor_encoder = sensor_trainer.sensor_encoder
            sensor_encoder.eval()
            
            print("✓ Sensor encoder weights loaded successfully!")
            print(f"  Model location: {sensor_encoder_checkpoint}")
            
        else:
            print("✗  Trained sensor encoder not found.")
            print(f"  Expected location: {sensor_encoder_checkpoint}")
            print("  Starting new training...")
            
            # Train sensor encoder
            sensor_encoder = sensor_trainer.train_with_interpretations(
                interpretations_file=config.semantic_interpretations_file,
                num_epochs=config.sensor_encoder_num_epochs,
                batch_size=config.sensor_encoder_batch_size,
                early_stopping=config.early_stopping,
                patience=config.patience
            )
        
        if not sensor_encoder:
            print("✗ Cannot proceed without trained sensor encoder")
                    
    except Exception as e:
        print(f"✗ Error during sensor encoder loading/training: {e}")
        import traceback
        traceback.print_exc()

    if sensor_encoder:
        print("✓ Sensor encoder training completed successfully!")
        
        # Evaluate sensor encoder
        print("\n" + "-" * 60)
        print("Step 4-2: Sensor Encoder Evaluation")
        print("-" * 60)
        
        try:
            sensor_evaluator = SensorEncoderEvaluator(config, sensor_encoder, text_encoder)
            sensor_evaluation_results = sensor_evaluator.comprehensive_evaluation(interpretations_file)
            
            if sensor_evaluation_results:
                print("✓ Sensor encoder evaluation completed successfully!")
            else:
                print("✗ Sensor encoder evaluation failed!")
                
        except Exception as e:
            print(f"✗ Error during sensor encoder evaluation: {e}")
            import traceback
            traceback.print_exc()
            sensor_evaluation_results = None
    else:
        print("✗ Training sensor encoder failed!")
        sensor_evaluation_results = None
    
    print("\nSemanticHAR System Training Pipeline Completed!")
    print("=" * 60)
    
    return text_encoder, sensor_encoder


def main():
    """Main function"""
    print("=" * 60)
    print("SemanticHAR System - Human Activity Recognition")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description='SemanticHAR System')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'inference', 'generate'],
                       help='Execution mode')
    parser.add_argument('--source_dataset', type=str, default='UCI_ADL_home_b',
                       choices=['UCI_ADL_home_b', 'UCI_ADL_home_a', 'MARBLE'],
                       help='Source dataset')
    parser.add_argument('--target_dataset', type=str, default='UCI_ADL_home_a',
                       choices=['UCI_ADL_home_b', 'UCI_ADL_home_a', 'MARBLE'],
                       help='Target dataset')
    parser.add_argument('--window_size', type=int, default=60,
                       help='Time window size in seconds')
    parser.add_argument('--overlap', type=float, default=0.8,
                       help='Window overlap ratio')
    parser.add_argument('--api_key', type=str, default=None,
                       help='OpenAI API key (optional, can use OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Load configuration
    config = SemanticHARConfig()
    
    # Override configuration with command line arguments
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
    if args.window_size:
        config.window_size_seconds = args.window_size
    if args.overlap:
        config.overlap_ratio = args.overlap
    if args.source_dataset:
        config.source_dataset = args.source_dataset
    if args.target_dataset:
        config.target_dataset = args.target_dataset

    print(f"Configuration:")
    print(f"  - Source Dataset: {config.source_dataset}")
    print(f"  - Target Dataset: {config.target_dataset}")
    print(f"  - Window size: {config.window_size_seconds}s")
    print(f"  - Overlap: {config.overlap_ratio}")
    print(f"  - Text Encoder Batch size: {config.text_encoder_batch_size}")
    print(f"  - Text Encoder Epochs: {config.text_encoder_num_epochs}")
    print(f"  - Text Encoder Learning rate: {config.text_encoder_learning_rate}")
    print(f"  - Sensor Encoder Batch size: {config.sensor_encoder_batch_size}")
    print(f"  - Sensor Encoder Epochs: {config.sensor_encoder_num_epochs}")
    print(f"  - Sensor Encoder Learning rate: {config.sensor_encoder_learning_rate}")
    
    if args.mode == 'train':
        text_encoder, sensor_encoder = train_lanhar(config)
        
        if text_encoder and sensor_encoder:
            print("\n✓ Both text encoder and sensor encoder training completed successfully!")
            print("✓ Text encoder and sensor encoder are ready for inference!")
            
            # Run inference automatically after successful training
            print("\n" + "=" * 60)
            print("Running Inference on Unseen Data")
            print("=" * 60)
            
            try:
                inference_engine = SensorEncoderInference(config, sensor_encoder, text_encoder)
                interpretations_file = "outputs/semantic_interpretations.json"
                
                if os.path.exists(interpretations_file):
                    inference_results = inference_engine.predict_activities(interpretations_file)
                    
                    if inference_results:
                        print("\n✓ Inference completed successfully!")
                        print(f"  Final Accuracy: {inference_results['accuracy']:.4f}")
                        print(f"  Final F1-Score: {inference_results['f1']:.4f}")
                else:
                    print("✗ Semantic interpretations file not found. Skipping inference.")
                    
            except Exception as e:
                print(f"✗ Error during inference: {e}")
                import traceback
                traceback.print_exc()
                
        elif text_encoder:
            print("\n✓ Text encoder training completed successfully!")
            print("✗ Training sensor encoder failed!")
        else:
            print("\n✗ Training failed!")

    elif args.mode == 'inference':
        print("\n" + "-" * 60)
        print("Only Inference Mode")
        print("-" * 60)
        
        try:
            # Load trained models
            text_encoder_path = "checkpoints/text_encoder_trained.pth"
            sensor_encoder_path = "checkpoints/sensor_encoder_trained.pth"
            interpretations_file = "outputs/semantic_interpretations.json"
            
            if not os.path.exists(text_encoder_path):
                print("✗ Trained text encoder not found. Please run 'train' mode first.")
                return
            
            if not os.path.exists(sensor_encoder_path):
                print("✗ Trained sensor encoder not found. Please run 'train' mode first.")
                return
            
            if not os.path.exists(interpretations_file):
                print("✗ Semantic interpretations file not found. Please run 'train' or 'generate' mode first.")
                return
            
            # Load text encoder
            print("Loading text encoder...")
            text_encoder = TextEncoder(config).to(config.device)
            text_encoder.load_state_dict(torch.load(text_encoder_path, map_location=config.device))
            text_encoder.eval()
            print("✓ Text encoder loaded")
            
            # Load sensor encoder
            print("Loading sensor encoder...")
            from models.sensor_encoder import SensorEncoder
            sensor_encoder = SensorEncoder(config).to(config.device)
            sensor_encoder.load_state_dict(torch.load(sensor_encoder_path, map_location=config.device))
            sensor_encoder.eval()
            print("✓ Sensor encoder loaded")
            
            # Run inference
            inference_engine = SensorEncoderInference(config, sensor_encoder, text_encoder)
            inference_results = inference_engine.predict_activities(interpretations_file)
            
            if inference_results:
                print("\n✓ Inference completed successfully!")
                print(f"  Final Accuracy: {inference_results['accuracy']:.4f}")
                print(f"  Final F1-Score: {inference_results['f1']:.4f}")
            else:
                print("✗ Inference failed!")
                
        except Exception as e:
            print(f"✗ Error during inference: {e}")
            import traceback
            traceback.print_exc()

    elif args.mode == 'generate':
        print("\n" + "-" * 60)
        print("Only Generate Mode")
        print("-" * 60)
        
        try:
            generator = SemanticGenerator(config)

            if os.path.exists(config.semantic_interpretations_file):
                print(f"Loading semantic interpretations from: {config.semantic_interpretations_file}")
                interpretations_data = generator.load_interpretations(config.semantic_interpretations_file)
            else:
                print(f"Generating semantic interpretations to: {config.semantic_interpretations_file}")
                
                interpretations_file = generator.generate_interpretations(
                    dataset_name=config.dataset_name,
                    window_size_seconds=config.window_size_seconds,
                    overlap_ratio=config.overlap_ratio,
                    output_file=config.semantic_interpretations_file
                )
            
            if config.semantic_interpretations_file:
                print(f"✓ Semantic interpretations generated: {config.semantic_interpretations_file}")
                summary = generator.get_interpretations_summary(config.semantic_interpretations_file)
            else:
                print("✗ Failed to generate semantic interpretations")
                
        except Exception as e:
            print(f"✗ Error generating interpretations: {e}")
    
    print("\nExecution completed!")


if __name__ == "__main__":
    main()
