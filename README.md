# SemanticHAR: LLM-based Human Activity Recognition

This project is an implementation of the paper "LanHAR: A Novel System for Cross-Dataset Human Activity Recognition Using LLM-Generated Semantic Interpretations". It uses LLMs to generate semantic interpretations of sensor data and performs cross-dataset human activity recognition.

## 🚀 Key Features

- **LLM-based Semantic Interpretation**: Generate natural language interpretations for sensor data and activity labels
- **Iterative Regeneration**: Quality improvement through clustering-based outlier detection and regeneration
- **BERT Text Encoder**: Convert semantic interpretations to embeddings
- **Transformer Sensor Encoder**: Map sensor data to language space
- **Contrastive Learning**: Align sensor and text embeddings
- **UCI ADL Dataset Support**: Activity recognition based on ambient sensor data


## 🛠️ Installation and Setup

### 1. Install Required Packages

```bash
pip install -r requirements.txt
```

### 2. OpenAI API Key Setup (Required)

An OpenAI API key is required to use LLM features:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or set it directly in `config.py`.

### 3. Data Preparation

Place the UCI ADL dataset in the `data/UCI ADL Binary Dataset/` directory.

## 🚀 Usage

### 1. Full Training Pipeline

```bash
python main.py --mode train --max_windows 100 --max_activities 10 --epochs 10
```

### 2. Generate Semantic Interpretations Only

```bash
python main.py --mode generate --max_windows 50 --max_activities 5
```

### 3. Text Encoder Evaluation

```bash
python main.py --mode evaluate
```

### 4. Advanced Options

```bash
python main.py --mode train \
    --dataset UCI_ADL \
    --window_size 60 \
    --overlap 0.8 \
    --max_windows 1000 \
    --max_activities 20 \
    --epochs 20 \
    --batch_size 64 \
    --learning_rate 1e-5
```

### 5. Command Line Arguments

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--mode` | Execution mode | `train` | `train`, `generate`, `evaluate` |
| `--dataset` | Dataset to use | `UCI_ADL` | `UCI_ADL`, `MARBLE` |
| `--window_size` | Time window size (seconds) | `60` | Any integer |
| `--overlap` | Window overlap ratio | `0.8` | 0.0-1.0 |
| `--batch_size` | Training batch size | `32` | Any integer |
| `--epochs` | Number of training epochs | `100` | Any integer |
| `--learning_rate` | Learning rate | `2e-5` | Any float |
| `--max_windows` | Max windows per home | `10000` | Any integer |
| `--max_activities` | Max activities for interpretation | `20` | Any integer |
| `--api_key` | OpenAI API key (optional) | `None` | Your API key |

## 📊 Supported Datasets

- **UCI ADL**: Ambient sensor data (PIR, Magnetic, Pressure, Flush, Electric sensors)
- **MARBLE**: IMU sensor data (accelerometer, gyroscope, magnetometer, barometer, smartphone)

## 🏗️ Project Structure

```
semantic/
├── main.py                     # Main execution script
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data/                      # Dataset directory
│   ├── MARBLE/                # MARBLE dataset (IMU sensors)
│   └── UCI ADL Binary Dataset/ # UCI ADL dataset (ambient sensors)
├── dataloader/                # Data loading modules
│   └── data_loader.py         # Dataset loading and preprocessing
├── llm/                       # LLM semantic generation
│   └── semantic_generator.py  # Semantic interpretation generation
├── models/                     # Model implementations
│   └── text_encoder.py        # Text encoder and trainer
├── outputs/                   # Generated outputs
│   ├── time_windows.json     # Generated time windows
│   └── semantic_interpretations.json # LLM interpretations
└── checkpoints/              # Model checkpoints
    ├── text_encoder.pth      # Trained text encoder
    └── text_decoder.pth      # Trained text decoder
```

## 🔧 System Architecture

### 1. Data Loading and Preprocessing (`dataloader/data_loader.py`)
- **UCI ADL**: Ambient sensor data parsing and preprocessing
- **MARBLE**: IMU sensor data loading and preprocessing
- Time window generation (default 60 seconds, 80% overlap)
- Train/Validation/Test split maintaining temporal order
- Data normalization and scaling

### 2. LLM Semantic Interpretation Generation (`llm/semantic_generator.py`)
- **OpenAI GPT Integration**: Semantic interpretation generation
- **4-Part Prompt Structure**: Data Introduction, Data Analysis, Knowledge, Task Introduction
- **3-Part Activity Prompt**: General Description, Ambient Sensor Patterns, Environmental Context
- **Window2Text Approach**: Subject-perspective descriptions from sensor events
- **Iterative Re-generation**: Quality control and filtering
- **Batch Processing**: Efficient processing of large datasets

### 3. Text Encoder Training (`models/text_encoder.py`)
- **BERT-based Text Encoder**: `bert-base-uncased` model
- **Contrastive Learning**: Alignment loss for sensor-activity matching
- **Reconstruction Loss**: TextDecoder for language model retention
- **Early Stopping**: Validation-based training termination
- **GPU Acceleration**: CUDA support for faster training

### 4. Model Evaluation (`models/text_encoder.py`)
- **TextEncoderEvaluator**: Comprehensive evaluation class
- **Alignment Quality**: Cosine similarity metrics
- **Reconstruction Quality**: Cross-entropy loss assessment
- **t-SNE Visualization**: High-dimensional embedding visualization
- **Similarity Matrix**: Heatmap analysis of embeddings

## 📈 Training Process

1. **Data Preparation**: Load dataset (UCI ADL or MARBLE) and generate time windows
2. **Semantic Interpretation Generation**: LLM-based interpretation of sensor data and activity labels
3. **Text Encoder Training**: BERT-based encoder with contrastive learning
4. **Model Evaluation**: Comprehensive evaluation with visualization
5. **Output Generation**: Save trained models and evaluation results

## 🎯 Key Features

### LLM Semantic Interpretation
- **Statistical Analysis**: Ambient sensor data characteristics
- **Window2Text Approach**: Subject-perspective natural language descriptions
- **4-Part Prompt Structure**: Systematic interpretation generation
- **Iterative Re-generation**: Quality control and filtering

### Contrastive Learning
- **Alignment Loss**: Sensor-activity embedding alignment
- **Category Contrastive Loss**: Category-wise grouping
- **Activity Contrastive Loss**: Activity-wise grouping
- **Reconstruction Loss**: TextDecoder for language model retention

### Evaluation and Visualization
- **Alignment Quality**: Accuracy and margin metrics
- **Reconstruction Quality**: Loss and accuracy assessment
- **t-SNE Visualization**: High-dimensional embedding visualization
- **Similarity Matrix**: Heatmap analysis of embeddings

## 📊 Performance Metrics

- **Alignment Accuracy**: Sensor-activity embedding alignment quality
- **Reconstruction Loss**: Text reconstruction quality
- **Similarity Matrix**: Inter-embedding similarity analysis
- **t-SNE Visualization**: High-dimensional embedding space visualization
- **GPU Utilization**: CUDA acceleration for faster training

## 🔍 Usage Examples

### Main Pipeline Execution

```bash
# Full training pipeline
python main.py --mode train --max_windows 100 --max_activities 10 --epochs 10

# Generate semantic interpretations only
python main.py --mode generate --max_windows 50 --max_activities 5

# Evaluation
python main.py --mode evaluate
```

### Individual Component Usage

```python
from llm.semantic_generator import SemanticGenerator
from models.text_encoder import TextEncoder, TextEncoderEvaluator

# LLM Generator
generator = SemanticGenerator()
interpretation = generator.generate_sensor_interpretation(sensor_data)

# Text Encoder
text_encoder = TextEncoder()
embeddings = text_encoder.encode_texts([interpretation])

# Evaluator
evaluator = TextEncoderEvaluator(config)
results = evaluator.comprehensive_evaluation()
```

## 🚨 Important Notes

1. **API Key**: Set `OPENAI_API_KEY` environment variable for LLM integration
2. **Memory**: Sufficient memory required for large datasets (recommend 16GB+)
3. **GPU**: CUDA-enabled GPU significantly improves training speed
4. **Data**: Datasets must be placed in the correct `data/` directory
5. **Dependencies**: Install all requirements with `pip install -r requirements.txt`

## 📁 Output Files

- `outputs/time_windows.json`: Generated time window data
- `outputs/semantic_interpretations.json`: LLM-generated semantic interpretations
- `outputs/text_encoder_evaluation.json`: Text encoder evaluation results
- `outputs/embedding_visualization.png`: t-SNE visualization
- `outputs/similarity_matrix.png`: Similarity matrix heatmap
- `checkpoints/text_encoder.pth`: Trained text encoder model
- `checkpoints/text_decoder.pth`: Trained text decoder model

## 🆘 Troubleshooting

### Common Issues

1. **CUDA Error**: Use CPU when GPU memory is insufficient
2. **API Error**: Check OpenAI API key and quota
3. **Memory Error**: Reduce batch size or use `--batch_size 16`
4. **Data Loading Error**: Check UCI ADL dataset path

### Test Dummy Data

When OpenAI API key is not available:

```bash
python create_dummy_interpretations.py
```

### Log Verification

```bash
# Run with detailed logging
python main.py --mode train --max_windows 10 --max_activities 3
```

## 🤝 Contributing

1. Create issues to report bugs or improvements.
2. Send pull requests to add new features.
3. Follow PEP 8 code style.

## 📄 License

This project is distributed under the MIT License.

## 📚 References

- LanHAR Paper (Original paper reference)
- BERT: Pre-training of Deep Bidirectional Transformers
- Attention Is All You Need (Transformer paper)

---

**Experience more accurate human activity recognition with the LanHAR system!** 🎯