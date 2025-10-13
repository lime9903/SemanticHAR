#!/usr/bin/env python3
"""
SemanticHAR System Main Execution File
"""
import os
import sys
import argparse
import json
import torch
import numpy as np
from typing import Dict, List
from datetime import datetime

# Add project root to Python path
sys.path.append('/workspace/semantic')

from config import SemanticHARConfig, OPENAI_API_KEY
from dataloader.data_loader import load_sensor_data
from llm.semantic_generator import SemanticGenerator
from models.text_encoder import TextEncoder, TextEncoderTrainer, TextEncoderEvaluator
from models.sensor_encoder import SensorEncoderPhase2Trainer, SensorEncoderEvaluator


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


def save_time_windows(sensor_data: Dict, config: SemanticHARConfig) -> str:
    """Save time windows to JSON file"""
    print("=" * 50)
    print("Step 1: Save Time Windows")
    print("=" * 50)
    
    # Prepare time windows data for saving
    time_windows_data = {
        'generation_info': {
            'timestamp': datetime.now().isoformat(),
            'dataset': config.dataset_name,
            'window_size_seconds': config.window_size_seconds,
            'overlap_ratio': config.overlap_ratio
        },
        'time_windows': {}
    }
    
    for home_id in ['home_a', 'home_b']:
        time_windows_data['time_windows'][home_id] = {
            'train': [],
            'val': [],
            'test': []
        }
        
        for split in ['train', 'val', 'test']:
            windows = sensor_data[home_id][split]
            for i, window in enumerate(windows):
                window_info = {
                    'window_id': f'{home_id}_{split}_{i+1}',
                    'activity': window['activity'].iloc[0] if 'activity' in window.columns else 'Unknown',
                    'sensor_events': len(window),
                    'window_duration': window['window_duration'].iloc[0] if 'window_duration' in window.columns else 60,
                    'window_start': window['window_start'].iloc[0] if 'window_start' in window.columns else 'N/A',
                    'window_end': window['window_end'].iloc[0] if 'window_end' in window.columns else 'N/A',
                    'sensor_types': window['sensor_type'].unique().tolist() if 'sensor_type' in window.columns else [],
                    'locations': window['sensor_location'].unique().tolist() if 'sensor_location' in window.columns else [],
                    'places': window['sensor_place'].unique().tolist() if 'sensor_place' in window.columns else []
                }
                time_windows_data['time_windows'][home_id][split].append(window_info)
    
    # Save time windows
    time_windows_file = f"outputs/time_windows.json"
    
    with open(time_windows_file, 'w', encoding='utf-8') as f:
        json.dump(time_windows_data, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"Time windows saved to: {time_windows_file}")
    print(f"Home A train: {len(sensor_data['home_a']['train'])} windows")
    print(f"Home A val: {len(sensor_data['home_a']['val'])} windows")
    print(f"Home A test: {len(sensor_data['home_a']['test'])} windows")
    print(f"Home B train: {len(sensor_data['home_b']['train'])} windows")
    print(f"Home B val: {len(sensor_data['home_b']['val'])} windows")
    print(f"Home B test: {len(sensor_data['home_b']['test'])} windows")
    
    return time_windows_file


def train_lanhar(config: SemanticHARConfig, max_windows_per_home: int = None, max_activity_interpretations: int = 20):
    """SemanticHAR System Training Pipeline"""
    print('\n' + "=" * 60)
    print("SemanticHAR System Training Pipeline")
    print("=" * 60)
    
    # Initialize model instances
    text_encoder = None
    sensor_encoder = None
    
    # Step 1: Load sensor data
    print(f"\nLoading {config.dataset_name} sensor data...")
    time_windows_file = "outputs/time_windows.json"

    if os.path.exists(time_windows_file):
        print(f"Loading time windows from: {time_windows_file}")
        try:
            # Load time windows from JSON
            with open(time_windows_file, 'r') as f:
                time_windows_data = json.load(f)
            
            # Convert JSON data to sensor_data format
            sensor_data = {}
            for home_id in ['home_a', 'home_b']:
                sensor_data[home_id] = {}
                for split in ['train', 'val', 'test']:
                    windows = time_windows_data['time_windows'][home_id][split]
                    # Convert window info back to DataFrame format (simplified)
                    sensor_data[home_id][split] = windows
            
            print(f"✓ Time windows loaded from JSON successfully")
            print(f"   - Home A: {len(sensor_data['home_a']['train'])} train, {len(sensor_data['home_a']['val'])} val, {len(sensor_data['home_a']['test'])} test")
            print(f"   - Home B: {len(sensor_data['home_b']['train'])} train, {len(sensor_data['home_b']['val'])} val, {len(sensor_data['home_b']['test'])} test")
            
        except Exception as e:
            print(f"  ⨺ Failed to load time windows from JSON: {e}")
            print("  Falling back to loading sensor data from scratch...")
            try:
                sensor_data = load_sensor_data(
                    config=config,
                    dataset_name=config.dataset_name,
                    window_size_seconds=config.window_size_seconds,
                    overlap_ratio=config.overlap_ratio
                )
                print("✓ Sensor data loaded successfully")
                print(f"   - Home A: {len(sensor_data['home_a']['train'])} train, {len(sensor_data['home_a']['val'])} val, {len(sensor_data['home_a']['test'])} test")
                print(f"   - Home B: {len(sensor_data['home_b']['train'])} train, {len(sensor_data['home_b']['val'])} val, {len(sensor_data['home_b']['test'])} test")
            except Exception as e2:
                print(f"⨺ Failed to load sensor data: {e2}")
                return None, None
    else:
        try:
            sensor_data = load_sensor_data(
                config=config,
                dataset_name=config.dataset_name,
                window_size_seconds=config.window_size_seconds,
                overlap_ratio=config.overlap_ratio
            )
            print(f"✓ Sensor data loaded successfully: {time_windows_file}")
            print(f"   - Home A: {len(sensor_data['home_a']['train'])} train, {len(sensor_data['home_a']['val'])} val, {len(sensor_data['home_a']['test'])} test")
            print(f"   - Home B: {len(sensor_data['home_b']['train'])} train, {len(sensor_data['home_b']['val'])} val, {len(sensor_data['home_b']['test'])} test")
            
            # Save time windows
            time_windows_file = save_time_windows(sensor_data, config)
            
        except Exception as e:
            print(f"⨺ Failed to load sensor data: {e}")
            return None, None
    
    # Step 2: Generate semantic interpretations
    interpretations_file = "outputs/semantic_interpretations.json"
    if os.path.exists(interpretations_file):
        print(f"Loading semantic interpretations from: {interpretations_file}")
        
        # Verify the file contains valid interpretations
        try:
            with open(interpretations_file, 'r') as f:
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
            print(f"⨺ Failed to load semantic interpretations from JSON: {e}")
            print("  Falling back to generating new interpretations...")
            # Generate new interpretations using SemanticGenerator
            generator = SemanticGenerator(config)
            interpretations_file = generator.generate_interpretations(
                dataset_name=config.dataset_name,
                window_size_seconds=config.window_size_seconds,
                overlap_ratio=config.overlap_ratio,
                max_windows_per_home=config.max_windows_per_home,
                max_activity_interpretations=config.max_activity_interpretations,
                splits=['train', 'val', 'test']
            )
            if not interpretations_file:
                print("⨺ Cannot proceed without semantic interpretations")
                return None, None
    else:
        # Generate new interpretations using SemanticGenerator
        generator = SemanticGenerator(config)
        interpretations_file = generator.generate_interpretations(
            dataset_name=config.dataset_name,
            window_size_seconds=config.window_size_seconds,
            overlap_ratio=config.overlap_ratio,
            max_windows_per_home=config.max_windows_per_home,
            max_activity_interpretations=config.max_activity_interpretations,
            splits=['train', 'val', 'test']
        )
        if not interpretations_file:
            print("⨺ Cannot proceed without semantic interpretations")
            return None, None
    
    # Step 3: Train or load text encoder
    print("\n" + "=" * 60)
    print("Step 3: Text Encoder Training")
    print("=" * 60)
    
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
            print("⨺ Trained text encoder not found.")
            print(f"  Expected location: {text_encoder_checkpoint}")
            print("  Starting new training...")
            
            # Train text encoder
            text_encoder = text_encoder_trainer.train_with_interpretations(
                interpretations_file=interpretations_file,
                num_epochs=config.num_epochs,
                batch_size=config.batch_size,
                early_stopping=config.early_stopping,
                patience=config.patience
            )
        
        if not text_encoder:
            print("⨺ Cannot proceed without trained text encoder")
            return None, None
            
    except Exception as e:
        print(f"⨺ Error during text encoder loading/training: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # Evaluate text encoder
    print("\n" + "=" * 60)
    print("Text Encoder Evaluation")
    print("=" * 60)
    
    try:
        evaluator = TextEncoderEvaluator(config, text_encoder=text_encoder)
        evaluation_results = evaluator.comprehensive_evaluation(interpretations_file)

    except Exception as e:
        print(f"⨺ Error during text encoder evaluation: {e}")
        import traceback
        traceback.print_exc()

    
    # Step 4: Train sensor encoder
    print("\n" + "=" * 60)
    print("Step 4: Sensor Encoder Training")
    print("=" * 60)

    try:
        # Check if trained sensor encoder exists
        sensor_encoder_checkpoint = "checkpoints/sensor_encoder_trained.pth"
        
        # Initialize Phase 2 trainer
        sensor_trainer = SensorEncoderPhase2Trainer(config, text_encoder)
        
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
            print("⨺  Trained sensor encoder not found.")
            print(f"  Expected location: {sensor_encoder_checkpoint}")
            print("  Starting new training...")
            
            # Train sensor encoder
            sensor_encoder = sensor_trainer.train_with_interpretations(
                interpretations_file=interpretations_file,
                num_epochs=config.num_epochs,
                batch_size=config.batch_size,
                early_stopping=config.early_stopping,
                patience=config.patience
            )
        
        if not sensor_encoder:
            print("⨺ Cannot proceed without trained sensor encoder")
                    
    except Exception as e:
        print(f"⨺ Error during sensor encoder loading/training: {e}")
        import traceback
        traceback.print_exc()

    if sensor_encoder:
        print("✓ Sensor encoder training completed successfully!")
        
        # Evaluate sensor encoder
        print("\n" + "=" * 60)
        print("Sensor Encoder Evaluation")
        print("=" * 60)
        
        try:
            sensor_evaluator = SensorEncoderEvaluator(config, sensor_encoder, text_encoder)
            sensor_evaluation_results = sensor_evaluator.comprehensive_evaluation(interpretations_file)
            
            if sensor_evaluation_results:
                print("✓ Sensor encoder evaluation completed successfully!")
            else:
                print("⨺ Sensor encoder evaluation failed!")
                
        except Exception as e:
            print(f"⨺ Error during sensor encoder evaluation: {e}")
            import traceback
            traceback.print_exc()
            sensor_evaluation_results = None
    else:
        print("⨺ Training sensor encoder failed!")
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
                       choices=['train', 'train_sensor_encoder', 'evaluate_sensor', 'generate', 'evaluate'],
                       help='Execution mode')
    parser.add_argument('--dataset', type=str, default='UCI_ADL',
                       choices=['UCI_ADL', 'MARBLE'],
                       help='Dataset to use')
    parser.add_argument('--window_size', type=int, default=60,
                       help='Time window size in seconds')
    parser.add_argument('--overlap', type=float, default=0.8,
                       help='Window overlap ratio')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--api_key', type=str, default=None,
                       help='OpenAI API key (optional, can use OPENAI_API_KEY env var)')
    parser.add_argument('--max_windows', type=int, default=10000,
                       help='Maximum number of windows per home for processing')
    parser.add_argument('--max_activities', type=int, default=20,
                       help='Maximum number of activities for interpretation')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Load configuration
    config = SemanticHARConfig()
    
    # Override configuration with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.max_windows:
        config.max_windows_per_home = args.max_windows
    if args.max_activities:
        config.max_activity_interpretations = args.max_activities
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
    if args.dataset:
        config.dataset_name = args.dataset
    if args.window_size:
        config.window_size_seconds = args.window_size
    if args.overlap:
        config.overlap_ratio = args.overlap

    print(f"Configuration:")
    print(f"  - Dataset: {config.dataset_name}")
    print(f"  - Window size: {config.window_size_seconds}s")
    print(f"  - Overlap: {config.overlap_ratio}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Max windows per home: {config.max_windows_per_home}")
    print(f"  - Max activity interpretations: {config.max_activity_interpretations}")
    
    if args.mode == 'train':
        text_encoder, sensor_encoder = train_lanhar(config, args.max_windows, args.max_activities)
        
        if text_encoder and sensor_encoder:
            print("\n✓ Both text encoder and sensor encoder training completed successfully!")
            print("✓ Text encoder and sensor encoder are ready for inference!")
        elif text_encoder:
            print("\n✓ Text encoder training completed successfully!")
            print("⨺ Training sensor encoder failed!")
        else:
            print("\n✗ Training failed!")

    elif args.mode == 'train_sensor_encoder':
        # Phase 2 only: Train sensor encoder using pre-trained text encoder
        print("\nStep 4: Training sensor encoder only...")
        try:
            # Check if text encoder exists
            text_encoder_path = "checkpoints/text_encoder_trained.pth"
            interpretations_file = "outputs/semantic_interpretations.json"
            
            if not os.path.exists(text_encoder_path):
                print("⨺ Pre-trained text encoder not found. Please run 'train' mode first.")
                return
            
            if not os.path.exists(interpretations_file):
                print("⨺ Semantic interpretations file not found. Please run 'generate' mode first.")
                return
            
            # Load pre-trained text encoder
            from models.text_encoder import TextEncoder
            text_encoder = TextEncoder(config)
            text_encoder.load_state_dict(torch.load(text_encoder_path, map_location=config.device))
            text_encoder.to(config.device)
            text_encoder.eval()
            
            print("✓ Pre-trained text encoder loaded successfully")
            
            # Train sensor encoder
            sensor_encoder = train_phase2_sensor_encoder(config, text_encoder, interpretations_file)
            
            if sensor_encoder:
                print("✓ Training sensor encoder completed successfully!")
            else:
                print("✗ Training sensor encoder failed!")
                
        except Exception as e:
            print(f"⨺ Error during training sensor encoder: {e}")
            import traceback
            traceback.print_exc()

    elif args.mode == 'evaluate_sensor':
        # Evaluate sensor encoder only
        print("\nEvaluating Sensor Encoder...")
        try:
            # Check if models exist
            text_encoder_path = "checkpoints/text_encoder_trained.pth"
            sensor_encoder_path = "checkpoints/sensor_encoder_trained.pth"
            interpretations_file = "outputs/semantic_interpretations.json"
            
            if not os.path.exists(text_encoder_path):
                print("⨺ Pre-trained text encoder not found. Please run 'train' mode first.")
                return
            
            if not os.path.exists(sensor_encoder_path):
                print("⨺ Trained sensor encoder not found. Please run 'train' or 'train_sensor_encoder' mode first.")
                return
            
            if not os.path.exists(interpretations_file):
                print("⨺ Semantic interpretations file not found. Please run 'generate' mode first.")
                return
            
            # Load models
            from models.text_encoder import TextEncoder
            from models.sensor_encoder import SensorEncoder
            
            text_encoder = TextEncoder(config)
            text_encoder.load_state_dict(torch.load(text_encoder_path, map_location=config.device))
            text_encoder.to(config.device)
            text_encoder.eval()
            
            sensor_encoder = SensorEncoder(config)
            sensor_encoder.load_state_dict(torch.load(sensor_encoder_path, map_location=config.device))
            sensor_encoder.to(config.device)
            sensor_encoder.eval()
            
            print("✓ Models loaded successfully")
            
            # Evaluate sensor encoder
            sensor_evaluator = SensorEncoderEvaluator(config, sensor_encoder, text_encoder)
            results = sensor_evaluator.comprehensive_evaluation(interpretations_file)
            
            if results:
                print("✓ Sensor encoder evaluation completed successfully!")
            else:
                print("✗ Sensor encoder evaluation failed!")
                
        except Exception as e:
            print(f"⨺ Error during sensor encoder evaluation: {e}")
            import traceback
            traceback.print_exc()

    elif args.mode == 'generate':
        # Generate semantic interpretations only
        print("\nGenerating semantic interpretations...")
        try:
            generator = SemanticGenerator(config)

            if os.path.exists("outputs/semantic_interpretations.json"):
                print("Loading semantic interpretations from existing JSON file...")
                interpretations_file = "outputs/semantic_interpretations.json"
                interpretations_data = generator.load_interpretations(interpretations_file)
            else:
                print("Generating semantic interpretations...")
                
                interpretations_file = generator.generate_interpretations(
                    dataset_name=config.dataset_name,
                    window_size_seconds=config.window_size_seconds,
                    overlap_ratio=config.overlap_ratio,
                    max_windows_per_home=args.max_windows,
                    max_activity_interpretations=args.max_activities,
                    splits=['train', 'val', 'test']
                )
            
            if interpretations_file:
                print(f"✓ Semantic interpretations generated: {interpretations_file}")
                summary = generator.get_interpretations_summary(interpretations_file)
            else:
                print("✗ Failed to generate semantic interpretations")
                
        except Exception as e:
            print(f"⨺ Error generating interpretations: {e}")
        
    elif args.mode == 'evaluate':
        # Evaluate text encoder
        print("\nEvaluating Text Encoder...")
        try:
            from models.text_encoder import TextEncoderEvaluator
            
            # Check model path
            model_path = "checkpoints/text_encoder_trained.pth"
            interpretations_file = "outputs/semantic_interpretations.json"
            
            if not os.path.exists(model_path):
                print("⨺  Trained model is not found. Please train the model first in 'train' mode.")
                return
            
            if not os.path.exists(interpretations_file):
                print("⨺  Semantic interpretations file is not found. Please generate interpretations first.")
                return
            
            # Load text encoder
            text_encoder = TextEncoder(config).to(config.device)
            text_encoder.load_state_dict(torch.load(model_path, map_location=config.device))
            text_encoder.eval()
            
            # Evaluation execution
            evaluator = TextEncoderEvaluator(config, text_encoder=text_encoder)
            results = evaluator.comprehensive_evaluation(interpretations_file)
            
            # Result output
            print("\n" + "="*60)
            print("Text Encoder Evaluation Results")
            print("="*60)
            print(f"- Accuracy: {results['evaluation_summary']['accuracy']:.3f}")
            print(f"- Margin: {results['evaluation_summary']['margin']:.3f}")
            print(f"- Reconstruction Loss: {results['evaluation_summary']['reconstruction_loss']:.3f}")
            print(f"- Reconstruction Accuracy: {results['evaluation_summary']['reconstruction_accuracy']:.3f}")
            print("="*60)
            
            # Quality judgment
            accuracy = results['evaluation_summary']['accuracy']
            margin = results['evaluation_summary']['margin']
            
            if accuracy > 0.8 and margin > 0.5:
                print("✓ Text Encoder training is very well done!")
            elif accuracy > 0.6 and margin > 0.3:
                print("⨠ Text Encoder training is good.")
            else:
                print("⨺  Text Encoder training is insufficient. Please train more epochs.")
                
        except Exception as e:
            print(f"⨺ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nExecution completed!")


if __name__ == "__main__":
    main()