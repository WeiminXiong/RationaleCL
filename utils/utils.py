import numpy as np
from texttable import Texttable
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

import logging
from typing import Optional, Union, Callable, Tuple, Dict, List, Any
from types import SimpleNamespace
import random
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import requests

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    DataCollator,
)

logger = logging.getLogger(__name__)


def confusion_matrix_view(true_label, pred_label, labels, logger):
    cf_matrix = confusion_matrix(true_label, pred_label)

    table = Texttable()
    table.add_row([" "] + [i[:8] for i in labels])
    table.set_max_width(2000)
    for idx, r in enumerate(cf_matrix):
        table.add_row([labels[idx][:8]] + [str(i) for i in cf_matrix[idx]])
    return table.draw()

def compute_forgetting_rate(detail_metrics, task_seq, id2label, mode='task'):
    label_metric_rec = {}
    for metric in detail_metrics:
        for label in metric:
            if label not in id2label:
                continue
            if label not in label_metric_rec:
                label_metric_rec[label] = []
            label_metric_rec[label].append(metric[label]['recall'] * 100)

    if mode == 'label':
        fr_per_label = []
        for label in label_metric_rec:
            if len(label_metric_rec) == 1:
                continue
            fr_tmp = max(label_metric_rec[label]) - label_metric_rec[label][-1]
            fr_per_label.append(fr_tmp)
        return np.mean(fr_per_label)
    elif mode == 'task':
        task_fr_rec = []
        for task in task_seq[:-1]:
            cur_task_metric = [label_metric_rec[id2label[label]] for label in task]
            cur_task_metric = np.array(cur_task_metric)
            cur_task_metric = np.mean(cur_task_metric, axis=0)
            cur_task_fr = np.max(cur_task_metric) - cur_task_metric[-1]
            task_fr_rec.append(cur_task_fr)
        return np.mean(task_fr_rec)


@torch.no_grad()
def get_hidden_states(
        model: Union[PreTrainedModel, nn.Module] = None,
        args: SimpleNamespace = None,
        data_collator: Optional[DataCollator] = None,
        eval_dataset: Optional[Dataset] = None,
        shuffle: bool = False,
        return_type: Optional[str] = 'np',
        return_labels: bool = False,
        tokenizer: PreTrainedTokenizerBase = None
):
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=shuffle,
        collate_fn=data_collator
    )

    len_dataloader = len(eval_dataloader)

    hidden_states = []
    label_list = []
    model.eval()
    for step, inputs in enumerate(eval_dataloader):
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        labels = inputs['labels']
        inputs = {'input_ids':inputs['qta_input_ids'], 'attention_mask':inputs['qta_attention_mask'], 'labels':inputs['qta_output_ids'], 'head_entity_offset': inputs['head_entity_offset'], 'tail_entity_offset': inputs['tail_entity_offset']}
        outputs = model(**inputs)
        hidden_state = outputs.hidden_states
        hidden_states.append(hidden_state)
        label_list.append(labels)

    hidden_states = torch.cat(hidden_states, dim=0)
    label_list = torch.cat(label_list, dim=0).to(hidden_states.device)
    if return_type == 'np':
        hidden_states = hidden_states.cpu().numpy()
        label_list = label_list.cpu().numpy()

    if return_labels:
        return hidden_states, label_list
    else:
        return hidden_states

def select_exemplars(
        model: Union[PreTrainedModel, nn.Module] = None,
        args: SimpleNamespace = None,
        data_collator: Optional[DataCollator] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: PreTrainedTokenizerBase=None,
        memory_size: int = None
):
    if memory_size is None:
        memory_size = args.memory_size
    features = get_hidden_states(model, args, data_collator, eval_dataset, return_type='np', tokenizer=tokenizer)
    distances = KMeans(n_clusters=memory_size, random_state=0).fit_transform(features)
    exemplars = []
    for k in range(memory_size):
        exemplar_idx = np.argmin(distances[:, k])
        exemplars.append(deepcopy(eval_dataset[exemplar_idx]))
    return exemplars


def reform_data(args, data, tokenizer, before_labels, cur_labels, similarity_dict, meta_features):
    new_data = {}
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    new_similarity_dict = deepcopy(similarity_dict)
    for label_1 in before_labels:
        for label_2 in cur_labels:
            similarity = cos(meta_features[label_1], meta_features[label_2])
            if similarity > args.threshold:
                new_similarity_dict[label_1].append(label_2)
                new_similarity_dict[label_2].append(label_1)
    for i in range(len(cur_labels)-1):
        for j in range(i+1, len(cur_labels)):
            label_1 = cur_labels[i]
            label_2 = cur_labels[j]
            similarity = cos(meta_features[label_1], meta_features[label_2])
            if similarity > args.threshold:
                new_similarity_dict[label_1].append(label_2)
                new_similarity_dict[label_2].append(label_1)
    
    for label in tqdm(data.keys()):
        cur_label_data = data[label]
        if sorted(similarity_dict[label])==sorted(new_similarity_dict[label]):
            new_data[label] = cur_label_data
        else:
            cur_label_data = generate_rationale(cur_label_data, tokenizer, new_similarity_dict[label], args.id2label)
            new_data[label] = cur_label_data
               
    return new_data, new_similarity_dict

def generate_rationale(data, tokenizer, similar_type, id2label):
    new_data = []
    label = data[0]['relation']
    option = generate_option_and_answer(similar_type, label, id2label, type='option')
    answer = generate_option_and_answer(similar_type, label, id2label, type='answer')
    for item in data:
        new_item = deepcopy(item)
        head_entity_offset = item['head_entity_offset']
        tail_entity_offset = item['tail_entity_offset']
        qta_input_ids = item['qta_input_ids']
        head_entity = tokenizer.decode(qta_input_ids[head_entity_offset[0]: head_entity_offset[1]], skip_special_tokens=True)
        tail_entity = tokenizer.decode(qta_input_ids[tail_entity_offset[0]: tail_entity_offset[1]], skip_special_tokens=True)
        tokens = item['tokens']
        start_index = tokens.index('sentence')
        while tokens[start_index+1] != "\"":
            start_index = tokens.index('sentence', start_index+1)
        start_index += 2
        sentence_tokens = tokens[start_index:-1]
        sentence = " ".join(sentence_tokens)
        prompt = f"Classify the relation between \"{head_entity}\" and \"{tail_entity}\" in this sentence:\n\n{sentence}\n\nOptions: {option}\n\nThe relation between \"{head_entity}\" and \"{tail_entity}\" is \"{label}\"{answer}"
        response = send_request(prompt)
        while True:
            if response.status_code==200:
                dict = json.loads(response.text)
                if 'choices' not in dict.keys():
                    response = send_request(prompt)
                    continue
                else:
                    predict = dict['choices'][0]['message']['content']
                    break
            else:
                response = send_request(prompt)
        
        rationale = predict.strip(' .,\n')
        qta_input = tokenizer.decode(qta_input_ids, skip_special_tokens=True)
        qtr_output = f'{rationale}. Therefore, the answer is : {label}.'
        new_item['qtr_output_ids'] = tokenizer.encode(qtr_output, return_tensors='pt')[0]
        qti_input = qta_input.replace('qta Q', 'qti Q')
        qti_input = qti_input + " Rationale: " +rationale
        new_item['qti_input_ids'] = tokenizer.encode(qti_input, return_tensors='pt')[0]     
        new_data.append(new_item)
    
    return new_data
        
def generate_option_and_answer(similar_type, label, id2label, type='option'):
    if type=="option":
        option = f"OPTIONS:\n- {label}"
        for similar_label in similar_type:
            option+=f"\n- {id2label[similar_label]}"
        return option
    else:
        answer = ", instead of "
        type = similar_type[0]
        answer+=f"\"{id2label[type]}\""
        if len(similar_type)>1:
            for type in similar_type:
                answer+=f" or \"{id2label[type]}\""
        answer+=".\n\nPlease explain why."
        return answer

def send_request(prompt):
    dict = {'model':"gpt-3.5-turbo",
        'messages' : [{'role': 'user', 'content':prompt}],
        'temperature':0,
        'top_p' : 1,
        'max_tokens': 400,
        'frequency_penalty':0,
        'presence_penalty':0,}

    # input your openai api key here
    headers = {
        'Authorization': 'Bearer $OPENAI_API_KEY',
        'Content-Type': 'application/json',
    }

    data = json.dumps(dict)
    while True:
        try:
            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, data=data)
        except Exception:
            continue
        else:
            break
    return response