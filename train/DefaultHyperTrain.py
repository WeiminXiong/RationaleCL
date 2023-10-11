import logging
import os
from typing import Optional, Union, Callable, Tuple, Dict, List
from types import SimpleNamespace
from pathlib import Path
from functools import partial
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from evaluate import load

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollator,
    set_seed,
)

from data import BaseData, BaseDataset
from .MTTrain import MT_train
from .DefaultEvaluate import default_evaluate
from .DefaultCollator import default_collator
from utils import select_exemplars, compute_forgetting_rate, reform_data

from model import (
    CLT5,
)


task_to_additional_special_tokens = {
    "RelationExtraction": ["[E11]", "[E12]", "[E21]", "[E22]"]
}

name2model = {
    "CLT5": CLT5,
}


logger = logging.getLogger(__name__)


def default_hyper_train(
            args: SimpleNamespace = None,
            data_collator: Optional[DataCollator] = None,
            writer: SummaryWriter = None
):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
        model_max_length = 512
    )

    default_data_collator = default_collator(tokenizer)

    stage1_data_collator = default_data_collator
    stage2_data_collator = default_data_collator
    evaluate_data_collator = default_data_collator

    stage1_train = MT_train
    stage2_train = MT_train

    logger.info(args)
    evaluate = default_evaluate

    all_cur_metric_rec = []
    all_total_metric_rec = []
    forgetting_rate_rec = []

    for cur_round in range(args.num_exp_rounds):
        set_seed(args.seed + cur_round * 100)
        model_save_path = os.path.join(args.output_dir, f'round{cur_round}')
        os.makedirs(model_save_path, exist_ok=True)
        pt_data = torch.load(f'sampled_data/{args.dataset_name}_rationale/round{cur_round}.pt')
        data = BaseData(args, pt_data)
        args.id2label = data.id2label
        args.label2id = data.label2id
        meta_features = data.features
        num_labels = len(args.id2label)
        label_list = args.id2label

        task_seq = np.array([args.label2id[label] if label !=-1 else -1 for label in pt_data['class_sequence']])
        task_seq = task_seq.reshape((args.num_tasks, args.class_per_task))
        
        temp = {}
        for key, value in meta_features.items():
            temp[args.label2id[key]] = value
        meta_features = temp
        
        task_seq = task_seq.tolist()

        ModelForContinualLearning = name2model[args.model_name]

        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            classifier_dropout=args.classifier_dropout,
            label2id=args.label2id,
            id2label=args.id2label,
        )
        config.global_args = args
        model = ModelForContinualLearning.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
        model.args = args
        model.resize_token_embeddings(config.vocab_size)
          
        model.to(args.device)

        cur_metric_rec = []
        total_metric_rec = []
        detail_metric_rec = []
        
        similarity_dict = {}

        memory_data = {}

        for task_id in range(args.num_tasks):
            new_memory_data = {}
            prefix = f'Round_{cur_round}_Task_{task_id}_'
            cur_labels = deepcopy(task_seq[task_id])
            cur_target_names = [label_list[tmp_label] for tmp_label in cur_labels]
            seen_labels = np.array(task_seq[:task_id + 1]).flatten().tolist()
            before_labels = np.array(task_seq[:task_id]).flatten().tolist()
            
            for label in cur_labels:
                similarity_dict[label] = []

            logger.info(f"***** Task-{task_id + 1} *****")
            logger.info(f"Current classes: {' | '.join(cur_target_names)}")

            stage1_train_data = data.filter(cur_labels, 'train')
            stage1_train_dataset = BaseDataset(stage1_train_data)

            stage1_train(
                model,
                args,
                num_train_epochs=args.stage1_epochs,
                data_collator=stage1_data_collator,
                train_dataset=stage1_train_dataset,
                tokenizer=tokenizer,
                writer=writer,
                prefix = prefix,
                qtr=args.qtr,
                qti=args.qti
            )
            
            logger.info("***** Collect Exemplars *****")
            for label in tqdm(cur_labels):
                tmp_train_data = data.filter(label, 'train')
                memory_data[label] = select_exemplars(model, args, evaluate_data_collator, tmp_train_data, tokenizer)
            
            for label in seen_labels:
                new_memory_data[label] = memory_data[label]

            if args.use_contrastive_rationale:
                logger.info("***** Update Constrastive Rationale *****")
                new_memory_data, similarity_dict = reform_data(args, new_memory_data, tokenizer, before_labels, cur_labels, similarity_dict, meta_features)
                memory_data = new_memory_data
                
            torch.save(new_memory_data, os.path.join(args.output_dir, f'round{cur_round}_memory_data.pt'))
            
            if task_id !=0 and args.stage2_epochs > 0:
                stage2_train_dataset = BaseDataset(new_memory_data)
                
                stage2_train(
                    model,
                    args,
                    num_train_epochs=args.stage2_epochs,
                    data_collator=stage2_data_collator,
                    train_dataset=stage2_train_dataset,
                    tokenizer=tokenizer,
                    writer=writer,
                    prefix = prefix,
                    qtr=args.qtr,
                    qti=args.qti
                )

            torch.save(model.state_dict(), os.path.join(model_save_path, f'model_{task_id}.pt'))
            
            cur_test_data = data.filter(cur_labels, 'test')
            history_test_data = data.filter(seen_labels, 'test')

            cur_test_dataset = BaseDataset(cur_test_data)
            history_test_dataset = BaseDataset(history_test_data)

            cur_metric, _ = evaluate(
                model,
                args,
                data_collator=evaluate_data_collator,
                eval_dataset=cur_test_dataset,
                cur_train_data=data,
                memory_data=memory_data,
                cur_labels=cur_labels,
                seen_labels=cur_labels,
                verbose=False,
                tokenizer=tokenizer
            )

            total_metric, detail_metric = evaluate(
                model,
                args,
                data_collator=evaluate_data_collator,
                eval_dataset=history_test_dataset,
                cur_train_data=data,
                memory_data=memory_data,
                cur_labels=cur_labels,
                seen_labels=seen_labels,
                verbose=True,
                tokenizer=tokenizer
            )

            cur_metric_rec.append(cur_metric * 100)
            total_metric_rec.append(total_metric * 100)
            detail_metric_rec.append(detail_metric)

            logger.info(f"***** Round-{cur_round + 1} Task-{task_id + 1} *****")
            logger.info(f"History test metrics: {' '.join([str(round(metric, 3)) for metric in total_metric_rec])}")
            logger.info(f"Current test metrics: {' '.join([str(round(metric, 3)) for metric in cur_metric_rec])}")

        all_cur_metric_rec.append(cur_metric_rec)
        all_total_metric_rec.append(total_metric_rec)
        forgetting_rate_rec.append(compute_forgetting_rate(detail_metric_rec, task_seq, args.id2label, mode='task'))



    all_cur_metric_rec = np.array(all_cur_metric_rec).mean(axis=0).tolist()
    all_total_metric_rec = np.array(all_total_metric_rec).mean(axis=0).tolist()
    forgetting_rate_rec = np.mean(forgetting_rate_rec)


    logger.info(f"***** Experiment over *****")
    logger.info(f"Average history test metrics: {' '.join([str(round(metric, 3)) for metric in all_total_metric_rec])}")
    logger.info(f"Average current test metrics: {' '.join([str(round(metric, 3)) for metric in all_cur_metric_rec])}")
    logger.info(f"Average forgetting rate: {forgetting_rate_rec}")



