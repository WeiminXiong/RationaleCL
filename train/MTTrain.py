import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, Callable, Tuple, Dict, List
import hydra
from omegaconf import DictConfig
from types import SimpleNamespace
from pathlib import Path
from functools import partial
from copy import deepcopy
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from torch.utils.tensorboard import SummaryWriter


from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    DataCollator,
)


logger = logging.getLogger(__name__)

    

def MT_train(
        model: Union[PreTrainedModel, nn.Module] = None,
        args: SimpleNamespace = None,
        num_train_epochs: int = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        optimizer: torch.optim.Optimizer = None,
        tokenizer: PreTrainedTokenizerBase = None,
        writer: SummaryWriter=None,
        prefix: str=None,
        qtr: bool=False,
        qti: bool=False,
):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    len_dataloader = len(train_dataloader)
    num_examples = len_dataloader * args.train_batch_size
    max_steps = len_dataloader * num_train_epochs

    if optimizer is None: 
        parameters = [{'params': model.parameters(), 'lr': args.learning_rate}]
        optimizer = AdamW(parameters) 

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Train batch size = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {max_steps}")

    progress_bar = tqdm(range(max_steps))
    global_step = 0
    for epoch in range(num_train_epochs):
        model.train()
        optimizer.zero_grad()
        loss_rec = []
        for step, inputs in enumerate(train_dataloader):
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
                
            qta_inputs = {'input_ids': inputs['qta_input_ids'],
                          'attention_mask': inputs['qta_attention_mask'],
                          'labels': inputs['qta_output_ids']}
            qta_outputs = model(**qta_inputs)
            qta_loss = qta_outputs["loss"] if isinstance(qta_outputs, dict) else qta_outputs[0]
            
            if qtr and not qti:
                qtr_inputs = {'input_ids': inputs['qtr_input_ids'],
                              'attention_mask': inputs['qtr_attention_mask'],
                              'labels': inputs['qtr_output_ids']}
                qtr_outputs = model(**qtr_inputs)
                qtr_loss = qtr_outputs["loss"] if isinstance(qtr_outputs, dict) else qtr_outputs[0]
                loss = qta_loss*args.alpha + qtr_loss*(1-args.alpha)
            if qti and not qtr:
                qti_inputs = {'input_ids': inputs['qti_input_ids'],
                              'attention_mask': inputs['qti_attention_mask'],
                              'labels': inputs['qti_output_ids']}
                qti_outputs = model(**qti_inputs)
                qti_loss = qti_outputs["loss"] if isinstance(qti_outputs, dict) else qti_outputs[0]
                loss = qta_loss*args.alpha + qti_loss*(1-args.alpha)
            if qti and qtr:
                qtr_inputs = {'input_ids': inputs['qtr_input_ids'],
                              'attention_mask': inputs['qtr_attention_mask'],
                              'labels': inputs['qtr_output_ids']}
                qtr_outputs = model(**qtr_inputs)
                qtr_loss = qtr_outputs["loss"] if isinstance(qtr_outputs, dict) else qtr_outputs[0]
                qti_inputs = {'input_ids': inputs['qti_input_ids'],
                              'attention_mask': inputs['qti_attention_mask'],
                              'labels': inputs['qti_output_ids']}
                qti_outputs = model(**qti_inputs)
                qti_loss = qti_outputs["loss"] if isinstance(qti_outputs, dict) else qti_outputs[0]
                loss = qta_loss*args.alpha + (qtr_loss*args.beta+qti_loss*(1-args.beta))*(1-args.alpha)
            if not qti and not qtr:
                loss = qta_loss
            
            loss = loss/args.accumulation_steps
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            progress_bar.update(1)
            loss_rec.append(loss.item())
            if (step+1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if (step+1) % args.report_freq == 0:
                progress_bar.set_postfix({"Loss": np.array(loss_rec).mean()*args.accumulation_steps})
                loss_rec = []
            writer.add_scalar(tag=prefix+'loss', scalar_value=loss*args.accumulation_steps, global_step=global_step)
            global_step = global_step+1

    progress_bar.close()
