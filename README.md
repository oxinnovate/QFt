# QFT (Quick Framework Transfer) 

A rapid model architecture adaptation framework for large language models, enabling smooth transitions between any model architectures.


## Installation

### Prerequisites
- Python 3.10
- PyTorch 2.3.1+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/oxinnovate/QFt
cd QFt
```

2. Install PyTorch with CUDA support:
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

3. Install accelerate:
```bash
pip install accelerate==1.8.1
```

4. Install the modified transformers library:
```bash
cd transformers-qft
pip install .
cd ..
```

5. Install other dependencies:
```bash
pip install numpy
```

6. Download the base model and data:
```bash
# The script expects Qwen2.5-1.5B-Instruct at:
# Qwen/Qwen2.5-1.5B-Instruct from Modelscope or HuggingFace
# Data for training/validation is also available on HuggingFace datasets:
# oxinnovate/company_iqa_for_qft
```

## Usage

### Basic Usage

Run the main learning script:

```bash
python qf_train.py
```
after training, you can test with 
```bash
python qf_validate.py
```

## Models and Data

### Pre-trained Models
- **Base Model**: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) - Available on HuggingFace and ModelScope
- **Trained Model**: [oxinnovate/QF2-1.5B-instruct](https://huggingface.co/oxinnovate/QF2-1.5B-instruct) - Our QFT-trained model

### Training Data
- **Dataset**: [oxinnovate/company_iqa_for_qft](https://huggingface.co/datasets/oxinnovate/company_iqa_for_qft) - Available on HuggingFace Datasets

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{qf_learning_2025,
  title={QF: Quick Feedforward AI Model Training without Gradient Back Propagation},
  author={Feng Qi},
  year={2025},
  cite={https://arxiv.org/abs/2507.04300}
}
@misc{qf_learning_2025,
  title={QF2: Quick Firing Model Weight Updates},
  author={Feng Qi},
  year={2025},
  cite={}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 