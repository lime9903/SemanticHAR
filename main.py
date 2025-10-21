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
from models.sensor_encoder import SensorEncoder, SensorEncoderTrainer, SensorEncoderEvaluator, SensorEncoderInference


def setup_environment():
    """Check CUDA availability and OpenAI API key"""
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  - CUDA Version: {torch.version.cuda}")
    else:
        print("GPU not available, using CPU: training will be slower on CPU")
    
    # Check OpenAI API key
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY is not set.")
        print("   To use LLM features, set the API key.")
        print("   You can set it by running: export OPENAI_API_KEY='your-api-key-here'")
    else:
        print(f"OpenAI API key: {OPENAI_API_KEY[:8]}...")


def train_lanhar(config: SemanticHARConfig):
    """
    SemanticHAR System Training Pipeline
    Step 1: load sensor data windows from JSON
    Step 2: generate semantic interpretations
    Step 3: train text encoder
    Step 4: evaluate text encoder
    Step 5: train sensor encoder
    Step 6: evaluate sensor encoder

    Args:
        config: SemanticHARConfig
        
    Returns:
        text_encoder: trained TextEncoder
        sensor_encoder: trained SensorEncoder
    """
    text_encoder = None
    sensor_encoder = None
    
    # Step 1: Load sensor data windows
    print('\n' + "-" * 60)
    print("Step 1: Load sensor data windows")
    print("-" * 60)
    
    print(f"Source dataset: {config.source_dataset}")
    print(f"Target dataset: {config.target_dataset}")

    if os.path.exists(config.windows_file):
        print(f"Loading windows from: {config.windows_file}")
        try:
            # Load windows from JSON
            with open(config.windows_file, 'r') as f:
                windows_data = json.load(f)
            
            generation_info = windows_data['generation_info']
            windows = windows_data['windows']
            
            print(f"✓ Windows loaded from JSON")
            print(f"  - Generated at: {generation_info['timestamp']}")
            print(f"  - Source dataset: {generation_info['source_dataset']}")
            print(f"  - Target dataset: {generation_info['target_dataset']}")
            print(f"  - Generation strategy: {generation_info['generation_strategy']}")
            print(f"  - Total windows: {sum(len(windows[split]) for split in windows)}")

        except Exception as e:
            print(f"  ✗ Failed to load windows from JSON: {e}")
            print("  Falling back to loading sensor data...")
            try:
                windows = load_sensor_data(config, use_event_based=config.use_event_based)
                print(f"✓ Windows loaded successfully with {sum(len(windows[split]) for split in windows)} windows")

            except Exception as e2:
                print(f"✗ Failed to load sensor data windows: {e2}")
                return None, None
    else:
        try:
            windows = load_sensor_data(config, use_event_based=config.use_event_based)
            print(f"✓ Windows loaded successfully with {sum(len(windows[split]) for split in windows)} windows")

        except Exception as e:
            print(f"✗ Failed to load sensor data windows: {e}")
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
            for split in interpretations_data['sensor_interpretations']:
                for window_id, window_data in interpretations_data['sensor_interpretations'][split].items():
                        if 'interpretation' in window_data and 'error' not in window_data:
                            sensor_count += 1
            
            activity_count = len([k for k, v in interpretations_data.get('activity_interpretations', {}).items() 
                                if 'interpretation' in v and 'error' not in v])
            
            print(f"✓ Loaded {sensor_count} sensor interpretations and {activity_count} activity interpretations")
            
        except Exception as e:
            print(f"✗ Failed to load semantic interpretations from JSON: {e}")
            print("  Falling back to generating new interpretations...")

            generator = SemanticGenerator(config)
            interpretations_file = generator.generate_interpretations(config)
            if not interpretations_file:
                print("✗ Failed to generate semantic interpretations")
                return None, None
    else:
        # Generate new interpretations
        generator = SemanticGenerator(config)
        interpretations_file = generator.generate_interpretations(config)
        if not interpretations_file:
            print("✗ Failed to generate semantic interpretations")
            return None, None
    
    # Step 3: Train or load text encoder
    print("\n" + "-" * 60)
    print("Step 3-1: Train text encoder")
    print("-" * 60)
    
    try:
        text_encoder_checkpoint = os.path.join(config.model_dir, "text_encoder_trained.pth")
        
        # Initialize text encoder trainer
        text_encoder_trainer = TextEncoderTrainer(config)

        if os.path.exists(text_encoder_checkpoint):
            print(f"✓ Text encoder checkpoint found: {text_encoder_checkpoint}")
            print("  Loading existing model weights...")
            
            # Load text encoder checkpoint
            text_encoder_trainer.text_encoder.load_state_dict(
                torch.load(text_encoder_checkpoint, map_location=config.device)
            )
            text_encoder = text_encoder_trainer.text_encoder
            text_decoder = text_encoder_trainer.text_decoder
            text_encoder.eval()
            text_decoder.eval()
            
            print("✓ Text encoder checkpoint loaded successfully!")
            print(f"  Model location: {text_encoder_checkpoint}")
            
        else:
            print("✗ Text encoder checkpoint not found.")
            print(f"  Expected location: {text_encoder_checkpoint}")
            print("  Starting text encoder training...")
            
            # Train text encoder
            text_encoder, text_decoder = text_encoder_trainer.train_text_encoder(
                interpretations_file=config.semantic_interpretations_file,
                num_epochs=config.text_encoder_num_epochs,
                batch_size=config.text_encoder_batch_size,
                early_stopping=config.early_stopping,
                patience=config.patience
            )
        
        if not text_encoder or not text_decoder:
            print("✗ Failed to train text encoder/decoder")
            return None, None
            
    except Exception as e:
        print(f"✗ Error during text encoder loading/training: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # Evaluate text encoder
    print("\n" + "-" * 60)
    print("Step 3-2: Evaluate text encoder")
    print("-" * 60)
    
    if config.use_evaluation:
        try:
            evaluator = TextEncoderEvaluator(config, text_encoder=text_encoder, text_decoder=text_decoder)
            evaluation_results = evaluator.comprehensive_evaluation(config.semantic_interpretations_file)

        except Exception as e:
            print(f"✗ Error during text encoder evaluation: {e}")
            import traceback
            traceback.print_exc()

    
    # Step 4: Train or load sensor encoder
    print("\n" + "-" * 60)
    print("Step 4-1: Train sensor encoder")
    print("-" * 60)

    try:
        sensor_encoder_checkpoint = os.path.join(config.model_dir, "sensor_encoder_trained.pth")
        
        # Initialize sensor encoder trainer
        sensor_trainer = SensorEncoderTrainer(config, text_encoder)
        
        if os.path.exists(sensor_encoder_checkpoint):
            print(f"✓ Sensor encoder checkpoint found: {sensor_encoder_checkpoint}")
            print("  Loading existing model weights...")
            
            # Load sensor encoder checkpoint
            sensor_trainer.sensor_encoder.load_state_dict(
                torch.load(sensor_encoder_checkpoint, map_location=config.device)
            )
            sensor_encoder = sensor_trainer.sensor_encoder
            sensor_encoder.eval()
            
            print("✓ Sensor encoder checkpoint loaded successfully!")
            print(f"  Model location: {sensor_encoder_checkpoint}")
            
        else:
            print("✗  Sensor encoder checkpoint not found.")
            print(f"  Expected location: {sensor_encoder_checkpoint}")
            print("  Starting sensor encoder training...")
            
            # Train sensor encoder
            sensor_encoder = sensor_trainer.train_with_interpretations(
                windows_file=config.windows_file,
                interpretations_file=config.semantic_interpretations_file,
                num_epochs=config.sensor_encoder_num_epochs,
                batch_size=config.sensor_encoder_batch_size,
                early_stopping=config.early_stopping,
                patience=config.patience
            )
        
        if not sensor_encoder:
            print("✗ Failed to train sensor encoder")
                    
    except Exception as e:
        print(f"✗ Error during sensor encoder loading/training: {e}")
        import traceback
        traceback.print_exc()

    if sensor_encoder:
        print("✓ Sensor encoder trained successfully!")
        
        if config.use_evaluation:
            # Evaluate sensor encoder
            print("\n" + "-" * 60)
            print("Step 4-2: Evaluate sensor encoder")
            print("-" * 60)
            
            try:
                sensor_evaluator = SensorEncoderEvaluator(config, sensor_encoder, text_encoder)
                sensor_evaluation_results = sensor_evaluator.comprehensive_evaluation(interpretations_file)
                
                if sensor_evaluation_results:
                    print("✓ Sensor encoder evaluated successfully!")
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
    
    print("\nSemanticHAR system training pipeline completed!")
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
    setup_environment()
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
    print(f"  - Source dataset: {config.source_dataset}")
    print(f"  - Target dataset: {config.target_dataset}")
    print(f"  - Generation strategy: {'event-based' if config.use_event_based else 'time-based'}")
    print(f"  - Window size: {config.window_size_seconds}s")
    print(f"  - Overlap: {config.overlap_ratio}")
    print(f"  - Text encoder batch size: {config.text_encoder_batch_size}")
    print(f"  - Text encoder epochs: {config.text_encoder_num_epochs}")
    print(f"  - Text encoder learning rate: {config.text_encoder_learning_rate}")
    print(f"  - Sensor encoder batch size: {config.sensor_encoder_batch_size}")
    print(f"  - Sensor encoder epochs: {config.sensor_encoder_num_epochs}")
    print(f"  - Sensor encoder learning rate: {config.sensor_encoder_learning_rate}")
    
    if args.mode == 'train':
        text_encoder, sensor_encoder = train_lanhar(config)
        
        if text_encoder and sensor_encoder:
            print("\n✓ Both text encoder and sensor encoder trained successfully!")
            print("✓ Text encoder and sensor encoder are ready for inference!")
            
            # Run inference automatically after successful training
            print("\n" + "=" * 60)
            print("Running Inference on Unseen Data")
            print("=" * 60)
            
            try:
                inference_engine = SensorEncoderInference(config, sensor_encoder, text_encoder)
                interpretations_file = config.semantic_interpretations_file
                
                if os.path.exists(interpretations_file):
                    inference_results = inference_engine.predict_activities(interpretations_file)
                    
                    if inference_results:
                        print("\n✓ Inference successful!")
                        print(f"  - Final accuracy: {inference_results['accuracy']:.4f}")
                        print(f"  - Final F1-score: {inference_results['f1']:.4f}")
                else:
                    print("✗ Semantic interpretations not found. Skipping inference.")
                    
            except Exception as e:
                print(f"✗ Error during inference: {e}")
                import traceback
                traceback.print_exc()
                
        elif text_encoder:
            print("\n✓ Text encoder trained successfully!")
            print("✗ Failed to train sensor encoder!")
        else:
            print("\n✗ Failed to train text encoder!")

    elif args.mode == 'inference':
        print("\n" + "-" * 60)
        print("Only inference mode")
        print("-" * 60)
        
        try:
            # Load trained models
            text_encoder_path = os.path.join(config.model_dir, "text_encoder_trained.pth")
            sensor_encoder_path = os.path.join(config.model_dir, "sensor_encoder_trained.pth")
            interpretations_file = config.semantic_interpretations_file
            
            if not os.path.exists(text_encoder_path):
                print("✗ Text encoder checkpoint not found. Please run 'train' mode first.")
                return
            
            if not os.path.exists(sensor_encoder_path):
                print("✗ Sensor encoder checkpoint not found. Please run 'train' mode first.")
                return
            
            if not os.path.exists(interpretations_file):
                print("✗ Semantic interpretations not found. Please run 'train' or 'generate' mode first.")
                return
            
            # Load text encoder
            print("Loading text encoder...")
            text_encoder = TextEncoder(config).to(config.device)
            text_encoder.load_state_dict(torch.load(text_encoder_path, map_location=config.device))
            text_encoder.eval()
            print("✓ Text encoder loaded")
            
            # Load sensor encoder
            print("Loading sensor encoder...")
            sensor_encoder = SensorEncoder(config).to(config.device)
            sensor_encoder.load_state_dict(torch.load(sensor_encoder_path, map_location=config.device))
            sensor_encoder.eval()
            print("✓ Sensor encoder loaded")
            
            # Run inference
            inference_engine = SensorEncoderInference(config, sensor_encoder, text_encoder)
            inference_results = inference_engine.predict_activities(interpretations_file)
            
            if inference_results:
                print("\n✓ Inference successful!")
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
