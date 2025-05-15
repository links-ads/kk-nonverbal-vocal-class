"""
    Part of the code was referenced from SUPERB: https://github.com/s3prl/s3prl
    and https://github.com/wngh1187/IPET/blob/main/Speechcommands_V2/W2V2/models/W2V2.py
"""
import torch
import loralib as lora
import transformers.models.wav2vec2.modeling_wav2vec2 as w2v2

from it_multimodal_er.configs import ModelConfig
from ..adapters import Adapter
from torch import nn


class Wav2Vec2EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        """
        
        """
        super().__init__()
        self.attention = w2v2.Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = w2v2.Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.config = config
        
        if self.config.finetune_method == "embedding_prompt" or self.config.finetune_method == "combined":
            self.embed_prompt = nn.Parameter(torch.randn([1, self.config.embedding_prompt_dim, 768]))
            nn.init.xavier_uniform_(self.embed_prompt)
            
        if self.config.finetune_method == "lora" or self.config.finetune_method == "combined":
            self.feed_forward.intermediate_dense = lora.Linear(config.hidden_size, config.intermediate_size, r=config.lora_rank)
            self.feed_forward.output_dense = lora.Linear(config.intermediate_size, config.hidden_size, r=config.lora_rank)
            
        if self.config.finetune_method == "adapter" or self.config.finetune_method == "adapter_l" or self.config.finetune_method == "combined":
            self.adapter = Adapter(
                config,
                dropout=0.1,
                d_model=config.hidden_size,
                bottleneck=config.adapter_hidden_dim,
                adapter_scalar=0.1
            )

    # TODO: Use strategy pattern for different finetuning methods
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        if self.config.finetune_method == "embedding_prompt" or self.config.finetune_method == "combined":
            hidden_states = torch.cat((self.embed_prompt.repeat(hidden_states.size(0), 1, 1), hidden_states), dim=1)
        attn_residual = hidden_states
        
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        # Adapter
        if self.config.finetune_method == "adapter":
            adapt_h = self.adapter(hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)

        # Adapter
        if self.config.finetune_method == "adapter":
            hidden_states = hidden_states+ adapt_h
        if self.config.finetune_method == "adapter_l" or self.config.finetune_method == "combined":
            hidden_states = hidden_states + self.adapter(hidden_states)
            
        hidden_states = self.final_layer_norm(hidden_states)
        if self.config.finetune_method == "embedding_prompt" or self.config.finetune_method == "combined":
            hidden_states = hidden_states[:, self.config.embedding_prompt_dim:, :]

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
        return outputs