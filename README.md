# LanHAR: LLM-based Human Activity Recognition

This project is an implementation of the paper "LanHAR: A Novel System for Cross-Dataset Human Activity Recognition Using LLM-Generated Semantic Interpretations". It uses LLMs to generate semantic interpretations of sensor data and performs cross-dataset human activity recognition.

## 🚀 Key Features

- **LLM-based Semantic Interpretation**: Generate natural language interpretations for sensor data and activity labels
- **Iterative Regeneration**: Quality improvement through clustering-based outlier detection and regeneration
- **BERT Text Encoder**: Convert semantic interpretations to embeddings
- **Transformer Sensor Encoder**: Map sensor data to language space
- **Contrastive Learning**: Align sensor and text embeddings
- **UCI ADL Dataset Support**: Activity recognition based on ambient sensor data

## 📁 Project Structure

```
semantic/
├── main.py                           # Main execution script
├── config.py                         # System configuration
├── create_dummy_interpretations.py   # Test dummy data generation
├── requirements.txt                  # Required packages
├── dataloader/
│   └── data_loader.py               # UCI ADL data loading and preprocessing
├── llm/
│   └── semantic_generator.py        # LLM-based semantic interpretation generation
├── models/
│   ├── text_encoder.py              # BERT-based text encoder (includes training logic)
│   └── sensor_encoder.py            # Transformer-based sensor encoder
├── evaluation/
│   └── text_encoder_evaluator.py    # Text encoder evaluation
├── outputs/                         # Generated results storage
├── checkpoints/                     # Model checkpoint storage
└── data/                           # Dataset storage
```

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
    --max_windows 1000 \
    --max_activities 20 \
    --epochs 20 \
    --batch_size 64 \
    --learning_rate 1e-5
```

## 📊 Supported Datasets

- **UCI ADL**: Ambient sensor data (PIR, Magnetic, Pressure, Flush, Electric sensors)
- **MARBLE**: IMU sensor data (accelerometer, gyroscope) - Future support planned

## 🔧 System Architecture

### 1. Data Loading and Preprocessing (`dataloader/data_loader.py`)
- UCI ADL data parsing and preprocessing
- Time window generation (default 60 seconds, 80% overlap)
- Train/Validation/Test split

### 2. LLM Semantic Interpretation Generation (`llm/semantic_generator.py`)
- Sensor data statistical analysis
- 4-stage prompt structure (Data Introduction, Data Analysis, Knowledge, Task Introduction)
- Activity label interpretation (General Description, Ambient Sensor Patterns, Environmental Context)
- Quality improvement through iterative regeneration

### 3. Text Encoder (BERT) (`models/text_encoder.py`)
- Convert semantic interpretations to embeddings
- Sensor-activity alignment through contrastive learning
- Maintain language model characteristics through reconstruction tasks

### 4. Sensor Encoder (Transformer) (`models/sensor_encoder.py`)
- Map sensor data to language space
- Positional encoding and multi-head attention
- Global average pooling

### 5. Evaluation and Visualization (`evaluation/text_encoder_evaluator.py`)
- Alignment quality evaluation
- Reconstruction quality evaluation
- t-SNE visualization
- Similarity matrix heatmap

## 📈 Training Process

1. **Data Preparation**: Load UCI ADL dataset and generate time windows
2. **Semantic Interpretation Generation**: LLM-based interpretation of sensor data and activity labels
3. **Text Encoder Training**: BERT-based embedding learning (batch size 32)
4. **Evaluation**: Alignment quality, reconstruction quality, visualization
5. **Sensor Encoder Training**: Sensor-text alignment learning (future implementation)

## 🎯 Key Features

### LLM Semantic Interpretation
- Statistical analysis of ambient sensor data characteristics
- Subject-perspective natural language description generation using Window2Text approach
- Systematic interpretation generation through 4-stage prompt structure
- Quality improvement through iterative regeneration

### Contrastive Learning
- Alignment between sensor and activity interpretations
- Alignment Loss: Sensor-activity embedding alignment
- Category Contrastive Loss: Category-wise grouping
- Activity Contrastive Loss: Activity-wise grouping
- Reconstruction Loss: Text reconstruction

### Evaluation and Visualization
- Alignment quality evaluation (accuracy, margin)
- Reconstruction quality evaluation (loss, accuracy)
- Embedding visualization through t-SNE
- Similarity matrix heatmap

## 📊 Performance Metrics

- **Alignment Accuracy**: Sensor-activity embedding alignment quality
- **Reconstruction Loss**: Text reconstruction quality
- **Similarity Matrix**: Inter-embedding similarity analysis
- **Visualization**: Embedding space visualization through t-SNE

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
from models.text_encoder import TextEncoder
from evaluation.text_encoder_evaluator import TextEncoderEvaluator

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

1. **API Key**: A valid API key is required to use OpenAI API.
2. **Memory**: Sufficient memory is required for large datasets.
3. **GPU**: Using CUDA-enabled GPU significantly improves training speed.
4. **Data**: UCI ADL dataset must be placed in the correct path.

## 📁 Output Files

- `outputs/time_windows.json`: Generated time window data
- `outputs/semantic_interpretations.json`: LLM-generated semantic interpretations
- `outputs/text_encoder_evaluation.json`: Text encoder evaluation results
- `outputs/embedding_visualization.png`: t-SNE visualization
- `outputs/similarity_matrix.png`: Similarity matrix heatmap
- `checkpoints/`: Trained model checkpoints

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