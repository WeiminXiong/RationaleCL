from copy import deepcopy
import random
import numpy as np
import json
import os
class data_sampler(object):

    def __init__(self, config=None, seed=None, tokenizer=None):

        self.config = config
        self.entity_markers = ["[E11]", "[E12]", "[E21]", "[E22]"]
        self.tokenizer = tokenizer

        # read relation data
        self.id2rel, self.rel2id, self.rel2name = self._read_relations()
        self.config.num_of_relation = len(self.rel2id)
        # random sampling
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed) 
            
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

        # regenerate data
        self.training_dataset, self.valid_dataset, self.test_dataset = self._read_data()

        # generate the task number
        self.batch = 0
        self.task_length = len(self.shuffle_index) // self.config.class_per_task  # 每一轮任务进入几个新关系
        self.lb_id2train_id = list(range(len(self.shuffle_index)))

        # record relations
        self.seen_relations = []
        self.history_test_data = {}

    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

    def __iter__(self):
        return self

    def __len__(self):
        return self.task_length

    def __next__(self):

        if self.batch == self.task_length:
            self.batch == 0
            raise StopIteration()

        indexs = self.shuffle_index[
                 self.config.class_per_task * self.batch: self.config.class_per_task * (self.batch + 1)]  # 每个任务出现的id
        self.batch += 1

        current_relations = []
        cur_training_data = {}
        cur_valid_data = {}
        cur_test_data = {}

        for index in indexs:
            if index == -1:
                current_relations.append(-1)
                self.seen_relations.append(-1)
                continue
            current_relations.append(self.id2rel[index])
            self.seen_relations.append(self.id2rel[index])

            cur_training_data[self.id2rel[index]] = self.training_dataset[index]
            cur_valid_data[self.id2rel[index]] = self.valid_dataset[index]
            cur_test_data[self.id2rel[index]] = self.test_dataset[index]
            self.history_test_data[self.id2rel[index]] = self.test_dataset[index]

        return cur_training_data, cur_valid_data, cur_test_data, current_relations, self.history_test_data, self.seen_relations

    def _read_data(self):
        '''
        :param file: the input sample file
        :return: samples for the model: [relation label, text]
        '''
        if self.config.dataset_name == 'FewRel' or 'TACRED' in self.config.dataset_name:
            data = json.load(open(os.path.join(self.config.data_path, self.config.dataset_name, 'data_with_rationale.json')))

        train_dataset = [[] for i in range(self.config.num_of_relation)]
        val_dataset = [[] for i in range(self.config.num_of_relation)]
        test_dataset = [[] for i in range(self.config.num_of_relation)]
        self.id2sent = {}
        
        for j, relation in enumerate(data.keys()):
            rel_samples = data[relation]
            if self.seed != None:
                random.seed(self.seed)
            random.shuffle(rel_samples)
            relation = self.rel2name[relation]
            count = 0
            count1 = 0
            for i, sample in enumerate(rel_samples):
                tokenized_sample = {}
                tokenized_sample['tokens'] = sample['tokens']
                tokenized_sample['relation'] = self.rel2name[sample['relation']]
                tokenized_sample['labels'] = self.rel2id[tokenized_sample['relation']]
                rationale = sample['rationale']
                
                subject_start_pos = tokenized_sample['tokens'].index('[E11]')
                subject_end_pos = tokenized_sample['tokens'].index('[E12]')
                object_start_pos = tokenized_sample['tokens'].index('[E21]')
                object_end_pos = tokenized_sample['tokens'].index('[E22]')

                subject = ' '.join(tokenized_sample['tokens'][subject_start_pos+1:subject_end_pos])
                object = ' '.join(tokenized_sample['tokens'][object_start_pos+1:object_end_pos])
                    
                qta_input_tokens, head_entity_offset, tail_entity_offset = self.edit_sentence(subject, object, sample['tokens'], type='qta')
                
                offset_char_list = []
                cur_pos = 0
                for token in qta_input_tokens:
                    offset_char_list.append((cur_pos, cur_pos + len(token)))
                    cur_pos += len(token) + 1
                head_entity_offset = [offset_char_list[head_entity_offset[0]][0], offset_char_list[head_entity_offset[1]-1][1]]
                tail_entity_offset = [offset_char_list[tail_entity_offset[0]][0], offset_char_list[tail_entity_offset[1]-1][1]]
                
                tokenized_sample['tokens'] = qta_input_tokens
                result = self.tokenizer(' '.join(qta_input_tokens), return_tensors = 'pt', return_offsets_mapping=True)
                tokenized_sample['qta_input_ids'] = result.input_ids[0]
                offset_mapping = result.offset_mapping
                head_start_pos = head_end_pos = tail_start_pos = tail_end_pos = -1
                for j, item in enumerate(offset_mapping[0]):
                    if item[0] ==0 and item[1]==0:
                        continue
                    if item[0] == head_entity_offset[0]:
                        head_start_pos = j
                    if item[0] == tail_entity_offset[0]:
                        tail_start_pos = j
                    if item[1] == head_entity_offset[1]:
                        head_end_pos = j+1
                    if item[1] == tail_entity_offset[1]:
                        tail_end_pos = j+1
                assert head_start_pos != -1 and head_end_pos != -1 and tail_start_pos != -1 and tail_end_pos != -1
                assert head_start_pos < head_end_pos and tail_start_pos < tail_end_pos
                tokenized_sample['head_entity_offset'] = [head_start_pos, head_end_pos]
                tokenized_sample['tail_entity_offset'] = [tail_start_pos, tail_end_pos]
                
                qta_output_tokens = self.edit_target(tokenized_sample['relation'])
                tokenized_sample['qta_output_ids'] = self.tokenizer.encode(' '.join(qta_output_tokens), return_tensors = 'pt')[0]
                
                qtr_input_tokens, _, _ = self.edit_sentence(subject, object, sample['tokens'], type='qtr')
                tokenized_sample['qtr_input_ids'] = self.tokenizer.encode(' '.join(qtr_input_tokens), return_tensors= 'pt')[0]
                qtr_output_tokens = self.edit_target(tokenized_sample['relation'], rationale)
                tokenized_sample['qtr_output_ids'] = self.tokenizer.encode(' '.join(qtr_output_tokens), return_tensors = 'pt')[0]
                
                qti_input_tokens, _, _ = self.edit_sentence(subject, object, sample['tokens'], type='qti', rationale=rationale)
                tokenized_sample['qti_input_ids'] = self.tokenizer.encode(' '.join(qti_input_tokens), return_tensors= 'pt')[0]
                qti_output_tokens= self.edit_target(tokenized_sample['relation'])
                tokenized_sample['qti_output_ids'] = self.tokenizer.encode(' '.join(qti_output_tokens), return_tensors = 'pt')[0]

                
                if self.config.dataset_name == 'FewRel':
                    if i < 420:
                        train_dataset[self.rel2id[relation]].append(tokenized_sample)
                    elif i < 420 + 140:
                        val_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        test_dataset[self.rel2id[relation]].append(tokenized_sample)
                else:
                    if i < len(rel_samples) // 5 and count < 40:
                        count += 1
                        test_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        count1 += 1
                        train_dataset[self.rel2id[relation]].append(tokenized_sample)
                        if count1 >= 320:  # 一个关系最多320个样本
                            break
                    
        
        return train_dataset, val_dataset, test_dataset


    def _read_relations(self):
        '''
        :param file: input relation file
        :return:  a list of relations, and a mapping from relations to their ids.
        '''
        if self.config.dataset_name == 'FewRel':
            rel2name = json.load(open(os.path.join(self.config.data_path, self.config.dataset_name, 'rel2name.json')))
            id2rel = json.load(open(os.path.join(self.config.data_path, self.config.dataset_name, 'id2label.json')))
            id2rel = [rel2name[rel] for rel in id2rel]
            rel2id = {}
            for i, x in enumerate(id2rel):
                rel2id[x] = i
            return id2rel, rel2id, rel2name
        
        if self.config.dataset_name == 'TACRED':
            rel2name = open(os.path.join(self.config.data_path, self.config.dataset_name, 'rel2name.txt')).readlines()
            id2rel = json.load(open(os.path.join(self.config.data_path, self.config.dataset_name, 'id2rel.json')))
            rel2name = {key: value.strip() for (key,value) in zip(id2rel, rel2name)}
            id2rel = [rel2name[rel] for rel in id2rel]
            rel2id = {}
            for i, x in enumerate(id2rel):
                rel2id[x] = i
            return id2rel, rel2id, rel2name

    def get_id2sent(self):
        return self.id2sent
    
    def edit_sentence(self, head_entity, tail_entity, tokens, type, rationale=None):
        if type=='qta':
            prefix = f'qta Q: Given the subject entity \" {head_entity} \" and object entity \" {tail_entity} \" , what is the relation type between them in sentence \" '
        elif type=='qtr':
            prefix = f'qtr Q: Given the subject entity \" {head_entity} \" and object entity \" {tail_entity} \" , what is the relation type between them in sentence \" '            
        elif type=='qti':
            prefix = f'qti Q: Given the subject entity \" {head_entity} \" and object entity \" {tail_entity} \" , what is the relation type between them in sentence \" '
        special_tokens = ['[E11]', '[E12]', '[E21]', '[E22]']
        for special_token in special_tokens:
            if special_token in tokens:
                tokens.remove(special_token)
        new_tokens = prefix.split() + tokens + [" \" ? "]
        head_entity_offset = tail_entity_offset = None
        if type=='qta':
            head_entity_offset = (len('qta Q: Given the subject entity \"'.split()),
                                  len('qta Q: Given the subject entity \"'.split())+len(head_entity.split()))
            tail_entity_offset = (len('qta Q: Given the subject entity \"'.split())+len(head_entity.split())+len('\" and object entity \"'.split()),
                                  len('qta Q: Given the subject entity \"'.split())+len(head_entity.split())+len('\" and object entity \"'.split())+len(tail_entity.split()))
        if type=='qti':
            new_tokens = new_tokens+['Rationale:']+rationale.split()
        
        return new_tokens, head_entity_offset, tail_entity_offset

    def edit_target(self, answer, rationale=None):
        if rationale is None:
            prefix = f'The answer is: {answer}.'
        else:
            prefix = f'{rationale}. Therefore, the answer is: {answer}.'
        new_tokens = prefix.strip().split()
        return new_tokens