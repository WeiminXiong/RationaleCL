from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase
)
from dataclasses import dataclass, field
from typing import Union, Optional, List, Dict, Any
from transformers.utils import PaddingStrategy
from copy import deepcopy

@dataclass
class default_collator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_ = deepcopy(features)
        qta_input_ids = [{"input_ids":item.pop("qta_input_ids"), 'labels':item.pop("labels"), 'head_entity_offset': item.pop("head_entity_offset"),
                        'tail_entity_offset': item.pop("tail_entity_offset")} for item in features_]
        
        qta_input_ids = self.tokenizer.pad(
            qta_input_ids,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        qta_output_ids = [{"input_ids":item.pop("qta_output_ids")} for item in features_]
        qta_output_ids = self.tokenizer.pad(
            qta_output_ids,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )['input_ids']
        qta_output_ids[qta_output_ids==self.tokenizer.pad_token_id] = -100
        
        qtr_input_ids = [{"input_ids":item.pop("qtr_input_ids")} for item in features_]
        qtr_input_ids = self.tokenizer.pad(
            qtr_input_ids,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        qtr_output_ids = [{"input_ids":item.pop("qtr_output_ids")} for item in features_]
        qtr_output_ids = self.tokenizer.pad(
            qtr_output_ids,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )['input_ids']
        qtr_output_ids[qtr_output_ids==self.tokenizer.pad_token_id] = -100
        
        qti_input_ids = [{"input_ids":item.pop("qti_input_ids")} for item in features_]
        qti_input_ids = self.tokenizer.pad(
            qti_input_ids,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        qti_output_ids = [{"input_ids":item.pop("qti_output_ids")} for item in features_]
        qti_output_ids = self.tokenizer.pad(
            qti_output_ids,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )['input_ids']
        qti_output_ids[qti_output_ids==self.tokenizer.pad_token_id] = -100
        
        if 'head_entity_offset' in qta_input_ids:
            batch = {'qta_input_ids': qta_input_ids['input_ids'],
                    'qta_attention_mask': qta_input_ids['attention_mask'],
                    'qta_output_ids': qta_output_ids,
                    'qtr_input_ids': qtr_input_ids['input_ids'],
                    'qtr_attention_mask': qtr_input_ids['attention_mask'],
                    'qtr_output_ids': qtr_output_ids,
                    'qti_input_ids': qti_input_ids['input_ids'],
                    'qti_attention_mask': qti_input_ids['attention_mask'],
                    'qti_output_ids': qti_output_ids,
                    'labels': qta_input_ids['labels'],
                    'head_entity_offset': qta_input_ids['head_entity_offset'],
                    'tail_entity_offset': qta_input_ids['tail_entity_offset']}
        
        return batch