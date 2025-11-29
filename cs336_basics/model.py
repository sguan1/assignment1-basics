from typing import Optional
import torch
import math
from einops import einsum, rearrange, reduce

class Linear(torch.nn.Module):

    def __init__(self, in_features: int, out_features:int, device=None, dtype=None):
        """Construct a
        linear transformation module. This function should accept the following parameters:
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters """
        super().__init__()

        mean = 0.0
        std = math.sqrt( 2.0 / (in_features + out_features))
        lower = -3 * std
        upper = 3 * std

        w = torch.empty((out_features, in_features), device = device, dtype = torch.float32)
        torch.nn.init.trunc_normal_(w, mean = mean, std = std, a = lower, b = upper)

        self.weight = torch.nn.Parameter(w)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the
        input. """
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
    
class EmbeddingModule(torch.nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim:int, device=None, dtype=None):
        """Construct an embedding module. This function should accept the following parameters:
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters"""

        super().__init__()

        w = torch.empty((num_embeddings, embedding_dim), device=device, dtype=torch.float32)
        torch.nn.init.trunc_normal_(w, mean=0, std = 1, a = -3, b = 3)

        self.weight = torch.nn.Parameter(w)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs."""
        return self.weight[token_ids]
    
class Rmsnorm(torch.nn.Module):
    
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        if not torch.is_floating_point(torch.tensor(0, dtype=dtype)):
            dtype = torch.float32
        w = torch.ones(d_model, device = device, dtype = dtype)
        self.weight = torch.nn.Parameter(w)
        self.eps = eps

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(reduce(x ** 2, "... d-> ... 1", "mean") + self.eps)
        result= self.weight * x / rms
        return result.to(in_dtype)

def SiLU(x:torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class Swiglu(torch.nn.Module):

    def __init__(self, d_model:int, d_ff:int, device = None, dtype = None):

        super().__init__()

        self.w1 = Linear(d_model, d_ff, device=device, dtype = dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype = dtype)       

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.w2((SiLU(self.w1(x)) * self.w3(x)))
    
class RotaryPositionalEmbedding(torch.nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """Construct the
        RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        assert d_k % 2 == 0, f"d_k must be even, got {d_k}"

        self._precompute_rotations(device)
    
    def _precompute_rotations(self, device: Optional[torch.device]):
        positions = torch.arange(self.max_seq_len, dtype=torch.float32, device= device)
        dim_indices = torch.arange(self.d_k // 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (self.theta ** (2 * dim_indices / self.d_k))
        angles = torch.outer(positions, inv_freq)

        cos_values = torch.cos(angles)
        sin_values = torch.sin(angles)

        self.register_buffer('cos_cached', cos_values, persistent=False)
        self.register_buffer('sin_cached', sin_values, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor: 
        """Process
        an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note
        that you should tolerate x with an arbitrary number of batch dimensions. You should assume
        that the token positions are a tensor"""

        seq_len = x.shape[-2]

        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        result = torch.empty_like(x)
        result[..., 0::2] = rotated_x1
        result[..., 1::2] = rotated_x2

        return result
    


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim= dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp/x_exp.sum(dim = dim, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
  d_k = Q.shape[-1]
  attention = softmax(torch.where(mask,einsum(Q, K, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") / math.sqrt(d_k), float("-inf")), dim = -1)
  return einsum(attention, V, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype = None):
        super().__init__()
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.num_heads = num_heads
        self.q_proj= Linear(d_model, num_heads * self.d_k , device, dtype)
        self.k_proj= Linear(d_model, num_heads * self.d_k , device, dtype)
        self.v_proj= Linear(d_model, num_heads * self.d_v , device, dtype)
        self.output_proj = Linear(num_heads * self.d_v, d_model, device, dtype)

    def forward(self, x: torch.Tensor, rope: RotaryPositionalEmbedding | None = None,
                token_positions: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        Q = rearrange(self.q_proj(x), "batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim", num_heads = self.num_heads)
        K = rearrange(self.k_proj(x), "batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim", num_heads = self.num_heads)
        V = rearrange(self.v_proj(x), "batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim", num_heads = self.num_heads)

        if rope is not None:
            if token_positions is  None:
                token_positions = torch.arange(seq_len, device = x.device)
            Q = rope(Q, token_positions)
            K = rope(K, token_positions)

        mask = ~torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)

        return self.output_proj(rearrange(scaled_dot_product_attention(Q,K,V,mask), "batch num_heads seq_len head_dim -> batch seq_len (num_heads head_dim)"))
    

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RotaryPositionalEmbedding | None = None, device=None, dtype = None):
        super().__init__()
        self.rope = rope
        self.ln1 = Rmsnorm(d_model, device=device,dtype= dtype)
        self.attn = MultiheadSelfAttention(d_model, num_heads, device, dtype)
        self.ln2 = Rmsnorm(d_model, device=device, dtype = dtype)
        self.ffn = Swiglu(d_model,d_ff, device, dtype)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), self.rope) 
        return x + self.ffn(self.ln2(x))
    
class TransformerLm(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, vocab_size: int, context_length: int, num_layers: int, rope_theta:float, device=None, dtype = None):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = EmbeddingModule(vocab_size, d_model, device, dtype)
        self.rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length, device)
        self.layers = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, self.rope, device, dtype) for _ in range(num_layers)])
        self.ln_final = Rmsnorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        x = self.lm_head(x)
        return x