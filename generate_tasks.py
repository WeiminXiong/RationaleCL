import random
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from types import SimpleNamespace

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    set_seed,
)


from data import (
    data_sampler,
)


def generate_data_fewrel_tacred(args: SimpleNamespace = None):
    tokenizer = AutoTokenizer.from_pretrained('t5-base', 
                                              use_fast=args.use_fast_tokenizer, model_max_length = 512)
    train_data = None
    val_data = None
    test_data = None
    for i in range(5):
        random.seed(args.seed + i * 100)
        torch.manual_seed(args.seed + i*100)
        torch.cuda.manual_seed(args.seed+i*100)
        # sampler setup
        sampler = data_sampler(config=args, seed=args.seed + i * 100, tokenizer=tokenizer)
        if test_data is None:
            train_data = sampler.training_dataset
            val_data = sampler.valid_dataset
            test_data = sampler.test_dataset

   
        class_sequence = []
        for _, (_, _, _, current_relations, _, _) in enumerate(sampler):
            class_sequence.extend(current_relations)
        
        print(f"Round {i}: {class_sequence}")
       
        round_data = {
            'train_data': train_data,
            'val_data':  val_data,
            'test_data': test_data,
            'class_sequence': class_sequence
        }

        if not os.path.exists(f'./sampled_data/{args.dataset_name}_rationale'):
            os.system(f'mkdir -p ./sampled_data/{args.dataset_name}_rationale')
        torch.save(obj=round_data, f=f'./sampled_data/{args.dataset_name}_rationale/round{i}.pt')


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg:DictConfig):
    args = OmegaConf.create()
    args = OmegaConf.merge(args, cfg.task_args, cfg.model_args, cfg.training_args)
    args = SimpleNamespace(**args)
    
    if args.dataset_name == "FewRel" or "TACRED" in args.dataset_name:
        generate_data_fewrel_tacred(args=args)
    
if __name__=="__main__":
    main()