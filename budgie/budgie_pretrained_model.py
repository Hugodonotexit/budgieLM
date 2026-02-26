from torch import nn
from transformers.modeling_utils import PreTrainedModel

from .budgie_config import BudgieConfig

class BudgiePreTrainedModel(PreTrainedModel):
    config_class = BudgieConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding) or (
            hasattr(module, "weight")
            and isinstance(module.weight, nn.Parameter)
            and getattr(module.weight, "ndim", 0) == 2
            and hasattr(module, "num_embeddings")
            and hasattr(module, "embedding_dim")
        ):
            module.weight.data.normal_(mean=0.0, std=std)
            padding_idx = getattr(module, "padding_idx", None)
            if padding_idx is not None:
                module.weight.data[int(padding_idx)].zero_()


__all__ = ["BudgiePreTrainedModel"]
