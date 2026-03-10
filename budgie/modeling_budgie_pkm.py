"""Product Key Memory (PKM) implementation for Budgie models.

Based on "Large Product Key Memory for Pretrained Language Models" 
(https://arxiv.org/abs/2010.03881).

PKM augments feed-forward networks with efficient memory lookup mechanisms,
increasing model capacity with minimal computational overhead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ProductKeyMemory(nn.Module):
    """Product Key Memory module with slot selection and value retrieval.
    
    Uses learned embeddings as memory keys and retrieves values based on
    cosine similarity to the input projection.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_product_keys: int = 8,  # Number of key tables
        product_key_size: int = 32,  # Dimension of each key
        value_size: int = 128,  # Size of memory values
        shared_memory: Optional[nn.Module] = None,  # Optional shared memory embeddings
    ):
        """Initialize ProductKeyMemory.
        
        Args:
            input_size: Input feature dimension
            output_size: Output feature dimension
            num_product_keys: Number of independent key tables (m in paper)
            product_key_size: Dimension of each key vector
            value_size: Dimension of memory values
            shared_memory: Optional pre-initialized memory module for sharing
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_product_keys = num_product_keys
        self.product_key_size = product_key_size
        self.value_size = value_size
        self.num_values = output_size // value_size if output_size % value_size == 0 else output_size // value_size + 1
        
        # Projection to product key space
        self.key_projections = nn.ModuleList([
            nn.Linear(input_size, product_key_size, bias=False)
            for _ in range(num_product_keys)
        ])
        
        # Memory values - use shared_memory if provided for weight initialization
        if shared_memory is not None:
            self.memory_values = shared_memory
        else:
            # Initialize memory values with learned embeddings
            # Each slot contains a value vector of size value_size
            total_slots = (output_size // value_size + (output_size % value_size > 0))
            self.memory_values = nn.Embedding(
                num_embeddings=total_slots,
                embedding_dim=value_size,
                padding_idx=None
            )
        
        # Product key slot tables: (num_slots, product_key_size)
        # Each key table is a set of slot embeddings
        num_slots = output_size // value_size + (output_size % value_size > 0)
        self.key_slot_tables = nn.ParameterList([
            nn.Parameter(torch.randn(num_slots, product_key_size) / math.sqrt(product_key_size))
            for _ in range(num_product_keys)
        ])
        
        # Output projection to combine memory with input
        self.output_proj = nn.Linear(value_size + input_size, output_size, bias=False)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters with proper scaling."""
        for key_proj in self.key_projections:
            nn.init.xavier_uniform_(key_proj.weight)
        
        for table in self.key_slot_tables:
            nn.init.normal_(table, std=1.0 / math.sqrt(self.product_key_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of ProductKeyMemory.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            Memory-augmented output of shape (batch_size, seq_len, output_size)
        """
        batch_size, seq_len, input_size = x.shape
        
        # Project input to product key space for each table
        # List of (batch_size, seq_len, product_key_size)
        key_projections = [proj(x) for proj in self.key_projections]
        
        # Select memory slots by finding nearest key in each table
        # For each position, compute cosine similarity to all slot keys in each table
        memory_outputs = []
        
        for i, key_proj in enumerate(key_projections):
            # key_proj shape: (batch_size, seq_len, product_key_size)
            # self.key_slot_tables[i] shape: (num_slots, product_key_size)
            
            # Compute cosine similarity between key_proj and slot keys
            # Using einsum for efficiency: (b,s,d) @ (n,d)^T -> (b,s,n)
            key_proj_normalized = F.normalize(key_proj, dim=-1, p=2)
            slot_keys_normalized = F.normalize(self.key_slot_tables[i], dim=-1, p=2)
            
            # Similarity scores: (batch_size, seq_len, num_slots)
            similarities = torch.matmul(key_proj_normalized, slot_keys_normalized.t())
            
            # Select best matching slot for each position (hard selection)
            selected_slots = torch.argmax(similarities, dim=-1)  # (batch_size, seq_len)
            
            # Retrieve values from memory
            # Flatten for indexing
            flat_slots = selected_slots.view(-1)  # (batch_size * seq_len,)
            
            # Get memory values: (batch_size * seq_len, value_size)
            if hasattr(self.memory_values, 'weight'):
                # nn.Embedding case
                values = self.memory_values(flat_slots)
            else:
                # Custom memory module
                values = self.memory_values(flat_slots)
            
            # Reshape back
            values = values.view(batch_size, seq_len, self.value_size)
            memory_outputs.append(values)
        
        # Combine memory outputs (concatenate across product keys or average)
        # Use average pooling across product keys
        memory = torch.stack(memory_outputs, dim=-1).mean(dim=-1)  # (batch_size, seq_len, value_size)
        
        # Concatenate memory with input and project to output size
        combined = torch.cat([memory, x], dim=-1)  # (batch_size, seq_len, value_size + input_size)
        output = self.output_proj(combined)  # (batch_size, seq_len, output_size)
        
        return output


class PKMFFNBlock(nn.Module):
    """PKM-augmented FFN block: RMSNorm → PKM → output.
    
    This replaces the standard MLP in SWA layers while maintaining
    the residual connection structure.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_product_keys: int = 8,
        product_key_size: int = 32,
        value_size: int = 128,
        mlp_ratio: float = 3.75,
        dropout: float = 0.0,
    ):
        """Initialize PKM FFN block.
        
        Args:
            hidden_size: Model dimension
            num_product_keys: Number of product key tables
            product_key_size: Dimension of each product key
            value_size: Size of memory values
            mlp_ratio: Hidden dimension ratio for expansion
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        # PKM module
        self.pkm = ProductKeyMemory(
            input_size=hidden_size,
            output_size=hidden_size,
            num_product_keys=num_product_keys,
            product_key_size=product_key_size,
            value_size=value_size,
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with PKM augmentation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Apply PKM
        output = self.pkm(x)
        output = self.dropout(output)
        return output


class SWABlockWithPKM(nn.Module):
    """SWA block with PKM-augmented FFN: RMSNorm → SWA → RMSNorm → PKM-FFN → residual.
    
    This is a specialized version of LlamaDecoderLayer that:
    1. Removes the standard MLP
    2. Replaces it with PKM-FFN
    3. Maintains the same interface as the original layer
    """
    
    def __init__(
        self,
        config,
        layer_idx: int,
        attn_implementation: str = "gla_sliding",
        num_product_keys: int = 8,
        product_key_size: int = 32,
        value_size: int = 128,
        enable_tiny_conv: bool = False,
        enable_gsm: bool = False,
        is_bridge_layer: bool = False,
    ):
        """Initialize SWA block with PKM.
        
        Args:
            config: Model configuration
            layer_idx: Layer index
            attn_implementation: Attention implementation to use
            num_product_keys: Number of product keys for PKM
            product_key_size: Dimension of each product key
            value_size: Size of PKM memory values
            enable_tiny_conv: Whether to include tiny convolution
            enable_gsm: Whether to include GSM token mixer
            is_bridge_layer: Whether this is a bridge layer
        """
        super().__init__()
        # Import here to avoid circular imports
        from .modeling_budgie_GLA import (
            LLAMA_ATTENTION_CLASSES,
            BudgieCausalDepthwiseConv1d,
            budgie_make_rmsnorm,
            GSM,
        )
        
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        # Attention
        impl = attn_implementation
        self.self_attn = LLAMA_ATTENTION_CLASSES[impl](config=config, layer_idx=layer_idx)
        
        # Layer norms (RMSNorm)
        self.input_layernorm = budgie_make_rmsnorm(config, config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = budgie_make_rmsnorm(config, config.hidden_size, config.rms_norm_eps)
        
        # PKM-FFN block replaces standard MLP
        self.pkm_ffn = PKMFFNBlock(
            hidden_size=config.hidden_size,
            num_product_keys=num_product_keys,
            product_key_size=product_key_size,
            value_size=value_size,
            dropout=float(getattr(config, "attention_dropout", 0.01)),
        )
        
        # Optional tiny convolution
        if enable_tiny_conv:
            kernel_size = int(getattr(config, "tiny_conv_kernel_size", 3))
            bias = bool(getattr(config, "tiny_conv_bias", False))
            init_zero = bool(getattr(config, "tiny_conv_init_zero", True))
            use_causal_conv1d = bool(getattr(config, "use_causal_conv1d", True))
            self.tiny_conv = BudgieCausalDepthwiseConv1d(
                config.hidden_size,
                kernel_size=kernel_size,
                bias=bias,
                init_zero=init_zero,
                use_causal_conv1d=use_causal_conv1d,
            )
        else:
            self.tiny_conv = None
        
        # Optional GSM token mixer
        if enable_gsm:
            self.gsm = GSM(
                d_model=config.hidden_size,
                max_seq_len=int(getattr(config, "max_position_embeddings", 0)),
                n_groups=int(getattr(config, "gsm_n_groups", 8)),
                gate_rank=int(getattr(config, "gsm_gate_rank", 32)),
                alpha=float(getattr(config, "gsm_alpha", 0.5)),
                dropout=float(getattr(config, "gsm_dropout", 0.0)),
                use_triton=bool(getattr(config, "gsm_use_triton", True)),
                w_init_scale=float(getattr(config, "gsm_w_init_scale", 0.02)),
                rms_eps=float(getattr(config, "gsm_rms_eps", 1e-6)),
            )
        else:
            self.gsm = None
        
        self.bridge_mixer_fusion = bool(is_bridge_layer and self.gsm is not None and impl == "gla_landmark")
        if self.bridge_mixer_fusion:
            self.bridge_alpha_proj = nn.Linear(config.hidden_size, 1, bias=True)
            self.bridge_out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            nn.init.zeros_(self.bridge_alpha_proj.weight)
            nn.init.zeros_(self.bridge_alpha_proj.bias)
            nn.init.eye_(self.bridge_out_proj.weight)
        else:
            self.bridge_alpha_proj = None
            self.bridge_out_proj = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_gate: Optional[torch.Tensor] = None,
        mlp_gate: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """Forward pass with SWA + PKM-FFN structure.
        
        Structure: RMSNorm → SWA → RMSNorm → PKM-FFN → residual
        """
        residual = hidden_states
        
        # First residual: input norm -> attention
        hidden_states = self.input_layernorm(hidden_states)
        
        if self.bridge_mixer_fusion:
            z = hidden_states
            if self.tiny_conv is not None:
                z = self.tiny_conv(
                    z,
                    use_cache=bool(use_cache),
                    past_key_value=past_key_value,
                    layer_idx=kwargs.get("layer_idx", self.layer_idx),
                    attention_mask_2d=kwargs.get("attention_mask_2d"),
                )
        else:
            if self.tiny_conv is not None:
                hidden_states = hidden_states + self.tiny_conv(
                    hidden_states,
                    use_cache=bool(use_cache),
                    past_key_value=past_key_value,
                    layer_idx=kwargs.get("layer_idx", self.layer_idx),
                    attention_mask_2d=kwargs.get("attention_mask_2d"),
                )
            z = hidden_states
        
        # Self Attention
        attn_output, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=z,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        if self.bridge_mixer_fusion:
            y_gsm = self.gsm(z)
            alpha = torch.sigmoid(self.bridge_alpha_proj(z.mean(dim=1))).view(z.shape[0], 1, 1)
            attn_output = (alpha * attn_output) + ((1.0 - alpha) * y_gsm)
            attn_output = self.bridge_out_proj(attn_output)
        elif self.gsm is not None:
            attn_output = attn_output + self.gsm(z)
        
        if attn_gate is None:
            attn_gate = 1.0
        hidden_states = residual + (attn_output * attn_gate)
        
        # Second residual: post-attention norm -> PKM-FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.pkm_ffn(hidden_states)
        
        if mlp_gate is None:
            mlp_gate = 1.0
        hidden_states = residual + (hidden_states * mlp_gate)
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs
