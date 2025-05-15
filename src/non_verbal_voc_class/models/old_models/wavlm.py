# part of the code was referenced from SUPERB: https://github.com/s3prl/s3prl
# and https://github.com/wngh1187/IPET/blob/main/Speechcommands_V2/W2V2/models/W2V2.py
import os
import pdb
import copy
import torch
import argparse
import numpy as np
# import loralib as lora
import transformers.models.wav2vec2.modeling_wav2vec2 as w2v2
import transformers.models.wavlm.modeling_wavlm as wavlm

from functools import lru_cache
# from torchaudio.compliance import kaldi

from torch import nn
# from src.it_peft_ser.models.adapter import Adapter
from collections import OrderedDict
from typing import Optional, Callable, Union
from torch.nn import functional as F
from torch.nn.functional import normalize
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor, AutoProcessor, WavLMModel, WhisperModel, AutoFeatureExtractor

class WavLMEncoderLayer(nn.Module):
    def __init__(self, config, has_relative_position_bias: bool = True):
        super().__init__()
        self.attention = wavlm.WavLMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = wavlm.WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.config = config
        
        if self.config.finetune_method == "embedding_prompt" or self.config.finetune_method == "combined":
            self.embed_prompt = nn.Parameter(torch.randn([1, self.config.embedding_prompt_dim, 768]))
            nn.init.xavier_uniform_(self.embed_prompt)
        if self.config.finetune_method == "lora" or self.config.finetune_method == "combined":
            self.feed_forward.intermediate_dense    = lora.Linear(config.hidden_size, config.intermediate_size, r=config.lora_rank)
            self.feed_forward.output_dense          = lora.Linear(config.intermediate_size, config.hidden_size, r=config.lora_rank)
            
        if self.config.finetune_method == "adapter" or self.config.finetune_method == "adapter_l" or self.config.finetune_method == "combined":
            self.adapter = Adapter(
                config,
                dropout=0.1,
                d_model=config.hidden_size,
                bottleneck=config.adapter_hidden_dim,
                adapter_scalar=0.1
            )

    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False, index=0):
        if self.config.finetune_method == "embedding_prompt" or self.config.finetune_method == "combined":
            hidden_states = torch.cat((self.embed_prompt.repeat(hidden_states.size(0), 1, 1), hidden_states), dim=1)
        
        attn_residual = hidden_states
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            index=index,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        
        # Adapter
        if self.config.finetune_method == "adapter":
            adapt_h = self.adapter(hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        if self.config.finetune_method == "adapter":
            hidden_states = hidden_states + adapt_h
        if self.config.finetune_method == "adapter_l" or self.config.finetune_method == "combined": 
            hidden_states = hidden_states + self.adapter(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        if self.config.finetune_method == "embedding_prompt" or self.config.finetune_method == "combined":
            hidden_states = hidden_states[:, self.config.embedding_prompt_dim:, :]
        outputs = (hidden_states, position_bias)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
   
class WavLMWrapper(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(WavLMWrapper, self).__init__()
        config.finetune_method = 'finetune'

        # 0. Asserts
        assert config.finetune_method in ["adapter", "lora", "finetune"], "finetune method not available"
        assert config.model_type == 'wavlm', "model type must be wavlm"
        assert config.output_hidden_states == True, "The upstream model must return all hidden states"

        # 1. Load the config
        self.config = config

        # 2. Set the max audio length and feature length
        # self.max_raw_length = torch.tensor(self.config.max_duration * self.config.sampling_rate) # (e.g., 4 * 16000 = 64000)
        # self.max_feat_length = self.get_feat_extract_output_lengths(self.max_raw_length) # (e.g., 200) considering the subsampling of factor two in the conv layers

        # 3. We Load the model first with weights
        self.config = config
        self.backbone_model = WavLMModel.from_pretrained(
            config._name_or_path,
            output_hidden_states=config.output_hidden_states,
        )

        state_dict = self.backbone_model.state_dict()
        # 4. Read the model config
        self.model_config = self.backbone_model.config
        self.model_config.finetune_method        = config.finetune_method
        # self.model_config.adapter_hidden_dim     = config.adapter_hidden_dim
        # self.model_config.embedding_prompt_dim   = config.embedding_prompt_dim
        # self.model_config.lora_rank              = config.lora_rank
        
        # 5. Config encoder layers with adapter or embedding prompt
        # pdb.set_trace()
        self.backbone_model.encoder.layers = nn.ModuleList(
            [WavLMEncoderLayer(self.model_config, has_relative_position_bias=(i == 0)) for i in range(self.model_config.num_hidden_layers)]
        )
        # 6. Load the weights back
        msg = self.backbone_model.load_state_dict(state_dict, strict=False)
        # 7. Freeze the weights
        if self.config.finetune_method == "adapter" or self.config.finetune_method == "adapter_l" or self.config.finetune_method == "embedding_prompt" or self.config.finetune_method == "finetune" or self.config.finetune_method == "lora" or self.config.finetune_method == "combined":
            for name, p in self.backbone_model.named_parameters():
                if name in msg.missing_keys: p.requires_grad = True
                else: p.requires_grad = False
        self.finetune_method = self.config.finetune_method
        
    def forward(self,
                input_features: torch.Tensor,
                # attention_mask: Optional[torch.Tensor] = None,
                length=None):
        """
        args:
        - input_features: Tensor, preprocessed (AutoFeatureExtractor) input features (batch_size, seq_len)
        - length: Tensor, length of the audio arrays
        - attention_mask: Tensor, attention mask for the transformer encoder
        It returns a tuple of hidden states of the transformer encoder + the input hidden states.
        """

        # 1. feature extraction and projections
        with torch.no_grad():
            # Original
            hidden_states = self.backbone_model.feature_extractor(input_features)
            hidden_states = hidden_states.transpose(1, 2) # New version of huggingface
            hidden_states, _ = self.backbone_model.feature_projection(hidden_states) # New version of huggingface
            
            # Adapted for attention mask
            # extract_features = self.backbone_model.feature_extractor(input_features)
            # extract_features = extract_features.transpose(1, 2)

            # if attention_mask is not None:
            #     # compute reduced attention_mask corresponding to feature vectors
            #     attention_mask = self.backbone_model._get_feature_vector_attention_mask(
            #         extract_features.shape[1], attention_mask, add_adapter=False
            #     )

            # hidden_states, _ = self.backbone_model.feature_projection(extract_features)
            # hidden_states = self.backbone_model._mask_hidden_states(
            #     hidden_states, mask_time_indices=None, attention_mask=attention_mask
            # )
        
        # 2. get length and mask
        if length is not None:
            length = self.get_feat_extract_output_lengths(length.detach().cpu())
            # length = length.cuda()
    
        # 3. transformer encoding features
        hidden_states = self.backbone_model.encoder(
            hidden_states,
            # attention_mask=attention_mask,
            output_hidden_states=self.config.output_hidden_states
        ).hidden_states
        
        return {'encoder_hidden_states': hidden_states, 'length': length}
        
    # From huggingface
    def get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1
        for kernel_size, stride in zip(self.backbone_model.config.conv_kernel, self.backbone_model.config.conv_stride):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length


def prepare_mask(length, shape, dtype):
    # Modified from huggingface
    mask = torch.zeros(
        shape, dtype=dtype
    )
    # these two operations makes sure that all values
    # before the output lengths indices are attended to
    mask[(torch.arange(mask.shape[0]), length.cpu() - 1)] = 1
    mask = mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return mask

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='emo2vec finetune experiments')
    parser.add_argument(
        '--finetune_method',
        default='none',
        type=str,
        help='finetune method: adapter, embedding prompt, input prompt'
    )
    
    parser.add_argument(
        '--adapter_hidden_dim',
        default=128,
        type=int,
        help='adapter dimension'
    )
    
    parser.add_argument(
        '--embedding_prompt_dim',
        default=5,
        type=int,
        help='adapter dimension'
    )
    
    args = parser.parse_args()
    model = WavLMWrapper(args)
    data = torch.zeros([1, 16000])
    output = model(data)
    print(output.shape)