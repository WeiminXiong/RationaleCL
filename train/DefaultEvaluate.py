import logging
import random
from typing import Optional, Union, Callable, Tuple, Dict, List
from types import SimpleNamespace
from tqdm import tqdm

from sklearn.metrics import classification_report, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import (
    PreTrainedModel,
    DataCollator,
    PreTrainedTokenizer,
    LogitsProcessor,
    LogitsProcessorList
)

from utils import confusion_matrix_view

logger = logging.getLogger(__name__)


class MyProcessor(LogitsProcessor):
    def __init__(self, seen_labels: List[int]):
        super().__init__()
        self.seen_labels = seen_labels

    def __call__(self, input_ids, scores):
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[:, self.seen_labels] = 0
        scores = torch.masked_fill(scores, mask, -1e9)
        return scores

@torch.no_grad()
def default_evaluate(
        model: Union[PreTrainedModel, nn.Module] = None,
        args: SimpleNamespace = None,
        data_collator: Optional[DataCollator] = None,
        eval_dataset: Optional[Dataset] = None,
        seen_labels: Optional[List[int]] = None,
        verbose: bool = True,
        tokenizer: PreTrainedTokenizer = None,
        **kwargs,
):
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    len_dataloader = len(eval_dataloader)
    num_examples = len_dataloader * args.eval_batch_size

    logger.info("***** Running evaluating *****")
    logger.info(f"  Num examples = {num_examples}")
    logger.info(f"  Eval batch size = {args.eval_batch_size}")

    progress_bar = tqdm(range(len_dataloader))

    golds = []
    preds = []

    out_of_scope = 0
    target_names = [args.id2label[label] for label in seen_labels]
    constrained_token_ids = get_token_ids(tokenizer=tokenizer, seen_labels=target_names)
    my_processor = MyProcessor(seen_labels=constrained_token_ids)
    my_processor_list = LogitsProcessorList([my_processor])
    
    model.eval()
    for step, inputs in enumerate(eval_dataloader):
        labels = inputs['labels']
        input_ids = inputs['qta_input_ids'].to(model.device)
        attention_mask = inputs['qta_attention_mask'].to(model.device)
        
        # greedy decoding
        output_sequences = model.generate(
            input_ids,
            logits_processor=my_processor_list,
            attention_mask = attention_mask,
            max_length = 100
        )
        
        pred = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        
        pred_rel_t = []
        for p_ in pred:
            p = p_.split(':')[-1]
            p = p.strip(' .,\n')
            if p in args.label2id:
                if args.label2id[p] in seen_labels:
                    pred_rel_t.append(seen_labels.index(args.label2id[p]))
                    continue
            pred_rel_t.append(random.randint(0, len(seen_labels)-1))
            print(f"out of scope: {p}")
            out_of_scope+=1
        gold_rel_t = [seen_labels.index(label) for label in labels]
        
        golds.extend(gold_rel_t)
        preds.extend(pred_rel_t)

        
        progress_bar.update(1)
    
    progress_bar.close()


    logger.info(f"Illegal number = {out_of_scope}")
    micro_f1 = f1_score(golds, preds, average='micro')
    logger.info("Micro F1 {}".format(micro_f1))

    details = None
    if verbose:
        details = classification_report(golds, preds, labels=range(len(seen_labels)), target_names=target_names,
                                         zero_division=0, output_dict=True)
        logger.info(
            '\n' + classification_report(golds, preds, labels=range(len(seen_labels)), target_names=target_names,
                                         zero_division=0))
        logger.info(f"confusion matrix\n{confusion_matrix_view(golds, preds, target_names, logger)}")

    return micro_f1, details


def get_token_ids(tokenizer, seen_labels):
    prefix = "The answer is: "
    token_ids = set()
    for label in seen_labels:
        text = prefix+label+'.'
        input_ids = tokenizer.encode(text)
        for ids in input_ids:
            token_ids.add(ids)
    return list(token_ids)