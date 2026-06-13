import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from attention import MultiHeadAttention, scaled_dot_product_attention, softmax

def plot_attention_heatmap():
    np.random.seed(42)
    tokens = ["The", "cat", "sat", "on", "mat", "and", "slept", "well"]
    seq_len = len(tokens)
    d_model = 64

    mha = MultiHeadAttention(d_model=d_model, num_heads=4)
    x = np.random.randn(1, seq_len, d_model)
    _, attn_weights = mha.forward(x)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Multi-Head Attention Heatmaps\nHardware-Accelerated Transformer',
                 fontsize=14, fontweight='bold')

    for head in range(4):
        ax = axes[head // 2][head % 2]
        weights = attn_weights[0, head]
        sns.heatmap(weights, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=tokens, yticklabels=tokens,
                    ax=ax, cbar=True, vmin=0, vmax=1)
        ax.set_title(f'Head {head+1}', fontweight='bold')
        ax.set_xlabel('Key'); ax.set_ylabel('Query')

    plt.tight_layout()
    plt.savefig('visualizations/attention_heatmap.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/attention_heatmap.png")

def plot_benchmark():
    configs = [(32,2,8), (64,4,16), (128,8,32), (256,8,64)]
    means = []
    labels = []
    for d_model, heads, seq_len in configs:
        from attention import MultiHeadAttention
        import time
        mha = MultiHeadAttention(d_model, heads)
        x = np.random.randn(1, seq_len, d_model)
        times = []
        for _ in range(50):
            s = time.perf_counter()
            mha.forward(x)
            times.append((time.perf_counter()-s)*1000)
        means.append(np.mean(times[5:]))
        labels.append(f'd={d_model}\nh={heads}\nseq={seq_len}')

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, means, color=['#0ea5e9','#7c3aed','#10b981','#f59e0b'], width=0.5)
    ax.set_title('Inference Time vs Model Size\nHardware-Accelerated Transformer', fontweight='bold')
    ax.set_ylabel('Mean Inference Time (ms)')
    ax.set_xlabel('Configuration')
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}ms', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/benchmark.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/benchmark.png")

if __name__ == '__main__':
    plot_attention_heatmap()
    plot_benchmark()
    print("All visualizations saved!")
