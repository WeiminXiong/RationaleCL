from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import (
    List, 
    Dict, 
    Any
)

import torch

class CLT5(T5ForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, *args, **kwargs):
        head_entity_offset = tail_entity_offset = None
        if 'head_entity_offset' in kwargs:
            head_entity_offset = kwargs['head_entity_offset']
            del kwargs['head_entity_offset']
        if 'tail_entity_offset' in kwargs:
            tail_entity_offset = kwargs['tail_entity_offset']
            del kwargs['tail_entity_offset']
            
        output = super().forward(*args, **kwargs)
        encoder_last_hidden_state = output['encoder_last_hidden_state']
        if head_entity_offset is not None and tail_entity_offset is not None:
            head_entity_hidden_state = torch.stack([encoder_last_hidden_state[idx, head_entity_offset[idx, 0]:head_entity_offset[idx, 1]].mean(dim=0) for idx in range(encoder_last_hidden_state.shape[0])])
            tail_entity_hidden_state = torch.stack([encoder_last_hidden_state[idx, tail_entity_offset[idx, 0]:tail_entity_offset[idx, 1]].mean(dim=0) for idx in range(encoder_last_hidden_state.shape[0])])
            hidden_states = torch.cat([head_entity_hidden_state, tail_entity_hidden_state], dim=-1)
            output.hidden_states = hidden_states
        
        return output
    