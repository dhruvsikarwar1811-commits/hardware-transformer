# 🤖 Hardware-Accelerated Transformer

Multi-Head Self-Attention mechanism implemented from scratch in pure NumPy with INT8 quantization and attention visualization.

## ✨ Features
- Scaled dot-product attention from scratch
- Multi-head attention (2/4/8 heads)
- INT8 quantization (8x memory reduction)
- Attention heatmap visualization
- Inference time benchmarking

## 📊 Results
| Config | Inference Time |
|--------|---------------|
| d=32, h=2, seq=8 | 0.064ms |
| d=64, h=4, seq=16 | 0.166ms |
| d=128, h=8, seq=32 | 0.644ms |

INT8: 8x memory reduction (32KB → 4KB)

## 🛠️ Tech Stack
Pure NumPy, Matplotlib, Seaborn

## ▶️ Quick Start
```bash
conda activate transformer-hw
python src/attention.py
python src/visualize.py
