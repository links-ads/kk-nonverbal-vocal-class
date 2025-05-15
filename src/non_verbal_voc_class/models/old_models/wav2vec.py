# part of the code was referenced from SUPERB: https://github.com/s3prl/s3prl
# and https://github.com/wngh1187/IPET/blob/main/Speechcommands_V2/W2V2/models/W2V2.py
import torch
import argparse
import transformers.models.wav2vec2.modeling_wav2vec2 as w2v2

from torch import nn
from transformers import Wav2Vec2Model

class Wav2Vec2EncoderLayer(nn.Module):
    def __init__(
        self,
        config,
        i
    ):
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
        self.i = i

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        if self.config.finetune_method == "embedding_prompt" or self.config.finetune_method == "combined":
            hidden_states = torch.cat((self.embed_prompt.repeat(hidden_states.size(0), 1, 1), hidden_states), dim=1)
        attn_residual = hidden_states
        
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class Wav2VecWrapper(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(Wav2VecWrapper, self).__init__()
        config.finetune_method = 'finetune'

        # 0. Asserts
        assert config.model_type == 'wav2vec2', "model type must be wav2vec2"
        assert config.output_hidden_states == True, "The upstream model must return all hidden states"

        # 1. We Load the model first with weights
        self.config = config

        self.backbone_model = Wav2Vec2Model.from_pretrained(
            config._name_or_path,
            output_hidden_states=config.output_hidden_states,
        )
        state_dict = self.backbone_model.state_dict()
        # 2. Read the model config
        self.model_config = self.backbone_model.config
        self.model_config.finetune_method        = config.finetune_method

        # 3. Config encoder layers with adapter or embedding prompt
        self.backbone_model.encoder.layers = nn.ModuleList([Wav2Vec2EncoderLayer(self.model_config, i) for i in range(self.model_config.num_hidden_layers)])

        # 4. Load the weights back
        msg = self.backbone_model.load_state_dict(state_dict, strict=False)

        # 5. Freeze the weights
        for name, p in self.backbone_model.named_parameters():
            if name in msg.missing_keys: p.requires_grad = True
            else: p.requires_grad = False

        self.finetune_method = self.config.finetune_method
        
    def forward(self,
                input_features: torch.Tensor,
                # attention_mask: Optional[torch.Tensor] = None,
                length: torch.Tensor = None,
            ):
        """
        args:
        - input_features: Tensor, preprocessed (AutoFeatureExtractor) input features (batch_size, seq_len)
        - length: Tensor, length of the audio arrays
        It returns a tuple of hidden states of the transformer encoder + the input hidden states.
        """
        # 1. feature extraction and projections
        with torch.no_grad():
            hidden_states = self.backbone_model.feature_extractor(input_features)
            hidden_states = hidden_states.transpose(1, 2) # New version of huggingface
            hidden_states, _ = self.backbone_model.feature_projection(hidden_states) # New version of huggingface
        
        # 2. get length and mask
        if length is not None:
            length = self.get_feat_extract_output_lengths(length.detach().cpu())
            # length = length.cuda()
            
        # 3. transformer encoding features
        hidden_states = self.backbone_model.encoder(
            hidden_states,
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
    model = Wav2VecWrapper(args)
    data = torch.zeros([1, 16000])
    output = model(data)
    print(output.shape)