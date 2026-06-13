import numpy as np
import time

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask * -1e9
    weights = softmax(scores)
    output = np.matmul(weights, V)
    return output, weights

def quantize_int8(x):
    scale = np.max(np.abs(x)) / 127.0
    return np.clip(np.round(x / scale), -128, 127).astype(np.int8), scale

class MultiHeadAttention:
    def __init__(self, d_model=64, num_heads=4):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        np.random.seed(42)
        self.W_Q = np.random.randn(d_model, d_model) * 0.1
        self.W_K = np.random.randn(d_model, d_model) * 0.1
        self.W_V = np.random.randn(d_model, d_model) * 0.1
        self.W_O = np.random.randn(d_model, d_model) * 0.1

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3).reshape(batch_size * self.num_heads, -1, self.d_k)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        Q = np.matmul(x, self.W_Q)
        K = np.matmul(x, self.W_K)
        V = np.matmul(x, self.W_V)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        attn_out, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = attn_out.reshape(batch_size, self.num_heads, seq_len, self.d_k)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        output = np.matmul(attn_out, self.W_O)
        return output, attn_weights.reshape(batch_size, self.num_heads, seq_len, seq_len)

def benchmark(d_model=64, num_heads=4, seq_len=16, runs=100):
    mha = MultiHeadAttention(d_model, num_heads)
    x = np.random.randn(1, seq_len, d_model)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        mha.forward(x)
        times.append((time.perf_counter() - start) * 1000)
    return np.mean(times[10:]), np.min(times), np.max(times)

if __name__ == '__main__':
    print("=" * 50)
    print(" Hardware-Accelerated Transformer Engine")
    print(" Pure NumPy | Multi-Head Attention")
    print("=" * 50)

    configs = [(32,2,8), (64,4,16), (128,8,32)]
    for d_model, heads, seq_len in configs:
        mean_t, min_t, max_t = benchmark(d_model, heads, seq_len)
        print(f"\nd_model={d_model}, heads={heads}, seq_len={seq_len}")
        print(f"  Mean: {mean_t:.3f}ms | Min: {min_t:.3f}ms | Max: {max_t:.3f}ms")

    # INT8 Quantization demo
    print("\n--- INT8 Quantization ---")
    mha = MultiHeadAttention(64, 4)
    wq_int8, scale = quantize_int8(mha.W_Q)
    print(f"W_Q original: float32, shape={mha.W_Q.shape}")
    print(f"W_Q quantized: int8, scale={scale:.6f}")
    print(f"Memory: {mha.W_Q.nbytes}B → {wq_int8.nbytes}B ({mha.W_Q.nbytes//wq_int8.nbytes}x reduction)")
    print("\nDone!")
