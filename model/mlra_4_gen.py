import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from flash_attn_interface import flash_attn_func, flash_attn_with_kvcache
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().to(x.dtype)
            self.sin_cached = freqs.sin().to(x.dtype)
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # (batch, seq_len, n_heads, head_dim)
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1).type_as(x)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank

        self.n_latent_head = 4
        assert config.kv_lora_rank % self.n_latent_head == 0
        self.kv_latent_dim = self.kv_lora_rank // self.n_latent_head
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        self.q_a_proj = nn.Linear(self.n_embd, self.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.n_head * self.q_head_dim, bias=False)
        self.q_scale = (self.n_embd / self.q_lora_rank) ** 0.5

        self.kv_a_proj_with_mqa = nn.Linear(self.n_embd, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorms = nn.ModuleList([RMSNorm(self.kv_latent_dim) for _ in range(self.n_latent_head)])
        self.kv_b_proj = nn.Parameter(
            torch.empty(self.n_latent_head, self.kv_latent_dim, self.n_head * 2 * self.qk_nope_head_dim)
        )
        self.kv_scale = (self.n_embd / self.kv_latent_dim) ** 0.5
        self.attn_output_scale = 1 / (self.n_latent_head ** 0.5)

        self.c_proj = nn.Linear(self.n_head * self.qk_nope_head_dim, self.n_embd, bias=False)
        self.rotary = Rotary(self.qk_rope_head_dim)

        self.k_cache: torch.Tensor = None
        self.v_caches: list = [None] * self.n_latent_head
        self.cache_length: int = 0

        self.register_buffer('W_UK', None)
        self.register_buffer('W_UV', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.q_a_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.kv_a_proj_with_mqa.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.q_b_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.kv_b_proj, mean=0.0, std=0.02)
        nn.init.zeros_(self.c_proj.weight)

    def reset_cache(self):
        self.k_cache = None
        self.v_caches = [None] * self.n_latent_head
        self.cache_length = 0

    def _precompute_absorption(self):
        # kv_b_proj.weight: [n_latent_head, kv_latent_dim, n_head * 2 * qk_nope_head_dim]
        w = self.kv_b_proj.view(
            self.n_latent_head, self.kv_latent_dim, self.n_head, 2 * self.qk_nope_head_dim
        ).permute(0, 2, 3, 1).contiguous()

        self.W_UK = w[:, :, :self.qk_nope_head_dim, :].contiguous()
        # W_UK: [n_latent_head, n_head, qk_nope_head_dim, kv_latent_dim]
        self.W_UV = w[:, :, self.qk_nope_head_dim:, :].contiguous().transpose(2, 3)
        # W_UV: [n_latent_head, n_head, kv_latent_dim, qk_nope_head_dim]

    def forward(self, x):
        B, T, C = x.size()
        device = x.device
        is_prefill = self.cache_length == 0

        # ==================== Process Q ====================
        compressed_q = self.q_a_layernorm(self.q_a_proj(x)) * self.q_scale
        q = self.q_b_proj(compressed_q)
        q = q.view(B, T, self.n_head, self.q_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # ==================== Process KV ====================
        all_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = torch.split(all_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(B, T, 1, self.qk_rope_head_dim)  # [B, T, 1, qk_rope_head_dim]
        kv_latents = torch.split(compressed_kv, self.kv_latent_dim, dim=-1)
        normed_kv_latents = []
        for i, latent in enumerate(kv_latents):
            normed_kv_latents.append(self.kv_a_layernorms[i](latent).unsqueeze(2).to(k_pe.dtype) * self.kv_scale)  # [B, T, 1， kv_latent_dim]

        self.cache_length += T
        for i in range(self.n_latent_head):
            if self.v_caches[i] is None:
                self.v_caches[i] = normed_kv_latents[i]
            else:
                self.v_caches[i] = torch.cat([self.v_caches[i], normed_kv_latents[i]], dim=1)

        # ==================== Apply RoPE ====================
        cos, sin = self.rotary(self.v_caches[0])
        q_pe = apply_rotary_emb(q_pe, cos[:, -T:], sin[:, -T:])
        k_pe = apply_rotary_emb(k_pe, cos[:, -T:], sin[:, -T:])

        if self.k_cache is None:
            self.k_cache = k_pe
        else:
            self.k_cache = torch.cat([self.k_cache, k_pe], dim=1)

        # ==================== Prefill ====================
        if is_prefill:
            y = 0
            for i in range(self.n_latent_head):
                q_absorbed_i = torch.einsum("bthp,hpq->bthq", q_nope, self.W_UK[i])  # [B, T, H, kv_latent_dim]
                z_i = flash_attn_func(q=q_pe, k=k_pe, v=normed_kv_latents[i], qv=q_absorbed_i, causal=True)
                y_i = torch.einsum("bthq,hqp->bthp", z_i, self.W_UV[i]) * self.attn_output_scale
                y = y + y_i

        # ==================== Decode ====================
        else:
            cache_seqlens = torch.tensor([self.cache_length] * B, device=device, dtype=torch.int)
            y = 0
            for i in range(self.n_latent_head):
                q_absorbed_i = torch.einsum("bthp,hpq->bthq", q_nope, self.W_UK[i])
                z_i = flash_attn_with_kvcache(
                    q=q_pe,
                    k_cache=self.k_cache,
                    v_cache=self.v_caches[i],
                    qv=q_absorbed_i,
                    cache_seqlens=cache_seqlens,
                    causal=True,
                )
                y_i = torch.einsum("bthq,hqp->bthp", z_i, self.W_UV[i]) * self.attn_output_scale
                y = y + y_i

        # ==================== Output ====================
        out = y.reshape(B, T, -1)
        out = self.c_proj(out)
        return out

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Split the linear projection into two parts for SwiGLU
        self.c_fc1 = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        
        # Output projection
        self.c_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.c_fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.c_fc2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.c_proj.weight)

    def forward(self, x):
        # Apply the first linear layer to produce two projections
        x1 = self.c_fc1(x)
        x2 = self.c_fc2(x)

        # Apply the SwiGLU gating: SILU on one projection, and gate with the other
        x = F.silu(x1) * x2
        
        # Apply the final output projection
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.n_embd)
        self.post_attention_layernorm = RMSNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig(PretrainedConfig):
    model_type = "gpt2"  
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 13
    n_embd : int = 768
    intermediate_size : int = 3072
    block_size: int = 1024  # Maximum sequence length
    bias: bool = False  # Use bias in all linear layers
    dropout: float = 0.0  # Dropout rate
    q_lora_rank: int = 512
    qk_rope_head_dim: int = 32
    kv_lora_rank: int = 256
    qk_nope_head_dim: int = 64
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
class GPT(PreTrainedModel):
    config_class = GPTConfig
    base_model_prefix = "gpt2"
    supports_gradient_checkpointing = True

    def __init__(self, config):
        # if self is not a subclass of PreTrinedModel, then we need to call super().__init__()
        # else we can just call super().__init__(config) to handle the config argument
        if not isinstance(self, PreTrainedModel):
            super().__init__()
        else:
            super().__init__(config)
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.layernorm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.02)

    def reset_cache(self):
        for block in self.transformer.h:
            block.attn.reset_cache()

    def forward(self, input_ids, **kwargs):
        # forward the GPT model itself
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            x = self.transformer.wte(input_ids) # token embeddings of shape (b, t, n_embd)
            for block in self.transformer.h:
                x = block(x)
            x = self.layernorm(x)
        logits = self.lm_head(x).float()
        return CausalLMOutputWithPast(logits=logits)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def generate(self, *args, **kwargs):
        self.reset_cache()
        return super().generate(*args, **kwargs)
        
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        # return n_params
        return n_params

    def save_pretrained(self, save_directory):
        self.config.save_pretrained(save_directory)
        super().save_pretrained(save_directory, safe_serialization=False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = super().from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)

        for block in model.transformer.h:
            block.attn._precompute_absorption()
        return model