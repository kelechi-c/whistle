from dataclasses import dataclass

@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    sliding_window: int | None = None
    use_sliding_window: bool = False
    use_cache: bool = True
    tie_word_embeddings: bool = True
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int | None = None

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


TINY_CONFIG = Qwen3Config(
    vocab_size=32000,
    hidden_size=64,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=1024,
    sliding_window=None,
    use_sliding_window=False,
    tie_word_embeddings=False,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
)
