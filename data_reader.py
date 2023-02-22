# !/usr/bin/env python
# coding=utf-8
"""
author: yonas
"""
import argparse

import torch
import numpy as np
from pathlib import Path
from transformers import BertTokenizer, AutoTokenizer, RobertaTokenizer
import datautils as utils
from datautils import NerExample, Any2Id
import time, copy
import ipdb
from types import MethodType

try:
    from prefetch_generator import BackgroundGenerator  # prefetch-generator


    class DataLoaderX(torch.utils.data.DataLoader):
        def __iter__(self):
            return BackgroundGenerator(super().__iter__())
except:
    pass


def is_whitespace(char):
    if char in [' ', '\n', '\t', '\r']:
        return True
    return False


class CharTokenizer(AutoTokenizer):  # 为了适配robert-large 的vocab.json
# class CharTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super(CharTokenizer, self).__init__(*args, **kwargs)
        self.fast_get_vocab = self.vocab

    def char_tokenize(self, text, **kwargs):
        """tokenize by char"""
        token_list = []
        for c in text:
            if c in self.vocab:
                token_list.append(c)
            elif is_whitespace(c):
                token_list.append('[unused1]')
            else:
                token_list.append(self.unk_token)
        return token_list


def get_char_tokenize_fn(tokenizer):
    vocab = tokenizer.vocab  # AutoTokenizer.vocab会非常耗时，先一次获取?

    # vocab = tokenizer.get_vocab()
    def char_tokenize_fn(text):
        token_list = []
        # time0 = time.time()
        for c in text:
            if c in vocab:
                token_list.append(c)
            elif is_whitespace(c):
                token_list.append('[unused1]')
            else:
                token_list.append(tokenizer.unk_token)
        # print('a', time.time() - time0)
        return token_list

    return char_tokenize_fn
# index = self._tokenizer.token_to_id(token)
# if index is None:
#     return self.unk_token_id

class NerDataReader:
    def __init__(self, tokenizer_path, max_len, ent_file_or_ent_lst, loss_type=None, args=None):
        self.tokenizer_path = tokenizer_path
        if 'roberta' in tokenizer_path.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_prefix_space=True)  # JAPAN -> "ĠJ", "AP", "AN"
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.char_tokenize_fn = get_char_tokenize_fn(self.tokenizer)  # used to handle ZH
        self.max_len = max_len
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        self.pad_token = self.tokenizer.pad_token
        self.sep_id = self.tokenizer.sep_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.pad_id = self.tokenizer.pad_token_id

        self.loss_type = loss_type

        self.char2id = Any2Id(exist_dict=self.tokenizer.vocab)

        # if is softmax, should add 'O'
        if isinstance(ent_file_or_ent_lst, list):
            self.ent2id = Any2Id(exist_dict={e: i for i, e in enumerate(ent_file_or_ent_lst)})
        else:
            self.ent2id = Any2Id.from_file(ent_file_or_ent_lst, use_line_no=True)

        # tag2id = {'[PAD]': 0, 'O': 1}
        tag2id = {'O': 0}  # use O as pad simultaneously
        for ent in self.ent2id:
            if ent not in tag2id:
                tag2id[f'B-{ent}'] = len(tag2id)
                tag2id[f'I-{ent}'] = len(tag2id)

        self.tag2id = Any2Id(exist_dict=tag2id)

        self.id2char = self.char2id.get_reverse()
        self.id2ent = self.ent2id.get_reverse()
        self.id2tag = self.tag2id.get_reverse()

        # self.args = args
        self.args = argparse.Namespace()
        self.args.pretrain_mode = 'feature_based'
        self.args.pretrain_mode = 'fine_tuning'
        self.args.use_refine_mask = False

    def post_process(self, exm: NerExample, lang='ENG', train=True, arch='seq', loss_type='sigmoid'):
        if not hasattr(exm, 'train_cache'):
            if lang == 'ENG':  # ENG means having sub_tokens
                input_ids = self.tokenizer.convert_tokens_to_ids(exm.bert_tok_char_lst)
            elif lang == 'ZH':  # split by each char
                input_ids = self.tokenizer.convert_tokens_to_ids(self.char_tokenize_fn(exm.char_lst))

            input_ids = [self.cls_id] + input_ids + [self.sep_id]
            exm.train_cache = dict(input_ids=input_ids, len=len(input_ids))
            if lang == 'ENG':
                exm.train_cache.update(ori_len=len(exm.char_lst), ori_2_tok=exm.ori_2_tok)
            if train:
                if arch == 'seq':
                    tag_lst = NerExample.to_tag_lst(exm.char_lst, exm.ent_dct)
                    tag_ids = [self.tag2id[tag] for tag in tag_lst]
                    exm.train_cache.update(tag_ids=tag_ids)

                elif arch == 'span':
                    assert loss_type in ['sigmoid', 'softmax']
                    ent_size = len(self.ent2id)
                    span_ner_tgt_lst = exm.get_span_level_ner_tgt_lst(neg_symbol='O')
                    if loss_type == 'sigmoid':
                        # use one-hot
                        # 一般torch的向量都是float()而默认的numpy则是doble(float64)
                        span_tgt_onehot = np.zeros([len(span_ner_tgt_lst), ent_size], dtype='float32')  # [num_spans, ent]
                        for i, tag in enumerate(span_ner_tgt_lst):
                            if tag != 'O' and tag in self.ent2id:
                                span_tgt_onehot[i][self.ent2id[tag]] = 1.
                        exm.train_cache.update(span_tgt=span_tgt_onehot)
                    elif loss_type == 'softmax':
                        span_tgt = [self.ent2id[e] for e in span_ner_tgt_lst]
                        exm.train_cache.update(span_tgt=span_tgt)
                else:
                    raise NotImplementedError

        # other setting
        if hasattr(exm, 'distilled_span_ner_pred_lst'):
            num_spans, so_far_ent_size = exm.distilled_span_ner_pred_lst.shape
            distilled_span_ner_tgt_lst = copy.deepcopy(exm.train_cache['span_tgt'])  # [num_spans, ent]
            distilled_span_ner_tgt_lst[:, :so_far_ent_size] = exm.distilled_span_ner_pred_lst
            exm.train_cache['distilled_span_tgt'] = distilled_span_ner_tgt_lst
            delattr(exm, 'distilled_span_ner_pred_lst')

        if hasattr(exm, 'distilled_task_ent_output'):
            exm.train_cache['distilled_task_ent_output'] = exm.distilled_task_ent_output
        else:
            if 'distilled_task_ent_output' in exm.train_cache:
                exm.train_cache.pop('distilled_task_ent_output')
        # ipdb.set_trace()
        return dict(ner_exm=exm, **exm.train_cache)

    def get_batcher_fn(self, gpu=False, device=None, arch='span'):

        def tensorize(array, dtype='int'):
            if isinstance(array, np.ndarray):
                ret = torch.from_numpy(array)
            elif isinstance(array, torch.Tensor):
                ret = array
            else:  # list
                #  Creating a tensor from a list of numpy.ndarrays is extremely slow.
                #  Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
                if dtype == 'int':
                    ret = torch.LongTensor(array)
                elif dtype == 'float':
                    ret = torch.FloatTensor(array)
                elif dtype == 'double':
                    ret = torch.DoubleTensor(array)
                else:
                    raise NotImplementedError
            if gpu:
                if device is not None:
                    ret = ret.to(device)
                else:
                    ret = ret.cuda()
            return ret

        def span_batcher(batch_e):
            max_len = max(e['len'] for e in batch_e)  # length after bert tokenized, i.e. the longer sub-word
            batch_input_ids = []
            batch_bert_token_type_ids = []
            batch_bert_attention_mask = []
            batch_seq_len = []
            batch_span_tgt_lst = []  # list of [num_spans, ent]
            batch_ner_exm = []

            batch_ori_seq_len = []
            batch_ori_2_tok = []
            if 'ori_len' in batch_e[0]:  # ori_len is the raw len, especially in ENG using tokenizer to split into longer subtokens
                ori_max_len = max(e['ori_len'] for e in batch_e)  # length before bert tokenized: shorter list

            if self.args.use_refine_mask:
                batch_refine_mask = np.zeros([len(batch_e), ori_max_len, ori_max_len])  # 0113
            if self.args.pretrain_mode == 'feature_based':
                batch_input_pts = []  # container for feature-based pt

            batch_span_tgt_lst_distilled = []  # list of [num_spans, ent]

            for bdx, e in enumerate(batch_e):
                batch_seq_len.append(e['len'] - 2)  # 去除cls和sep后的长度
                batch_input_ids.append(e['input_ids'] + [self.pad_id] * (max_len - e['len']))
                batch_bert_token_type_ids.append([0] * max_len)  # seg0
                batch_bert_attention_mask.append([1] * e['len'] + [0] * (max_len - e['len']))
                batch_ner_exm.append(e['ner_exm'])

                if 'ori_len' in batch_e[0]:  # ENG
                    batch_ori_seq_len.append(e['ori_len'])
                    batch_ori_2_tok.append(e['ori_2_tok'] + [0] * (ori_max_len - e['ori_len']))

                if self.args.pretrain_mode == 'feature_based':  # feature-based pt
                    if hasattr(e['ner_exm'], 'pt'):
                        assert e['ner_exm'].pt.shape[0] == e['ori_len']
                        batch_input_pts.append(e['ner_exm'].pt)

                if 'span_tgt' in e:
                    batch_span_tgt_lst.append(tensorize(e['span_tgt']))  # list of [num_spans, ent]
                    if self.args.use_refine_mask:
                        batch_refine_mask[bdx, :e['ori_len'], :e['ori_len']] = e['ner_exm'].refine_mask  # 0113

                if 'distilled_span_tgt' in e:
                    batch_span_tgt_lst_distilled.append(tensorize(e['distilled_span_tgt']))  # list of [num_spans, ent]

            if 'ori_len' not in batch_e[0]:  # ZH
                batch_ori_seq_len = batch_seq_len  # 方便兼容ZH时也能使用ori_seq_len

            if self.args.pretrain_mode == 'feature_based':
                batch_input_pts = torch.nn.utils.rnn.pad_sequence(batch_input_pts, batch_first=True, padding_value=0.)  # [b,len,1024]

            if batch_span_tgt_lst:
                batch_span_tgt = torch.cat(batch_span_tgt_lst, dim=0)  # [bsz*num_spans, ent]
            else:
                batch_span_tgt = None

            if batch_span_tgt_lst_distilled:
                batch_span_tgt_distilled = torch.cat(batch_span_tgt_lst_distilled, dim=0)
            else:
                batch_span_tgt_distilled = None

            return {
                'input_ids': tensorize(batch_input_ids),
                'bert_token_type_ids': tensorize(batch_bert_token_type_ids),
                'bert_attention_mask': tensorize(batch_bert_attention_mask),
                'seq_len': tensorize(batch_seq_len),
                'batch_ner_exm': batch_ner_exm,

                'ori_seq_len': tensorize(batch_ori_seq_len),
                'batch_ori_2_tok': tensorize(batch_ori_2_tok),

                'batch_span_tgt': batch_span_tgt,
                'batch_span_tgt_lst': batch_span_tgt_lst,

                'batch_input_pts': tensorize(batch_input_pts) if self.args.pretrain_mode == 'feature_based' else None,

                'batch_refine_mask': tensorize(batch_refine_mask) if self.args.use_refine_mask else None,

                'batch_span_tgt_distilled': batch_span_tgt_distilled,
                'batch_span_tgt_lst_distilled': batch_span_tgt_lst_distilled
            }

        def seq_batcher(batch_e):
            max_len = max(e['len'] for e in batch_e)
            batch_input_ids = []
            batch_bert_token_type_ids = []
            batch_bert_attention_mask = []
            batch_seq_len = []
            batch_tag_ids = []
            batch_ner_exm = []

            batch_ori_seq_len = []
            batch_ori_2_tok = []
            if 'ori_len' in batch_e[0]:  # ori_len is the raw len, especially in ENG using tokenizer to split into longer subtokens
                ori_max_len = max(e['ori_len'] for e in batch_e)  # length before bert tokenized: shorter list

            if self.args.pretrain_mode == 'feature_based':
                batch_input_pts = []  # container for feature-based pt

            batch_distilled_task_ent_output = []

            for e in batch_e:
                batch_seq_len.append(e['len'] - 2)
                batch_input_ids.append(e['input_ids'] + [self.pad_id] * (max_len - e['len']))
                batch_bert_token_type_ids.append([0] * max_len)  # seg0
                batch_bert_attention_mask.append([1] * e['len'] + [0] * (max_len - e['len']))
                batch_ner_exm.append(e['ner_exm'])

                if 'ori_len' in batch_e[0]:  # ENG
                    batch_ori_seq_len.append(e['ori_len'])
                    batch_ori_2_tok.append(e['ori_2_tok'] + [0] * (ori_max_len - e['ori_len']))

                if self.args.pretrain_mode == 'feature_based':  # feature-based pt
                    if hasattr(e['ner_exm'], 'pt'):
                        assert e['ner_exm'].pt.shape[0] == e['ori_len']
                    batch_input_pts.append(e['ner_exm'].pt)

                if 'tag_ids' in e:
                    batch_tag_ids.append(tensorize(e['tag_ids']))  # list of [len]

                if 'distilled_task_ent_output' in e:
                    batch_distilled_task_ent_output.append(tensorize(e['distilled_task_ent_output']))  # list of [len,ent]

            if 'ori_len' not in batch_e[0]:  # ZH
                batch_ori_seq_len = batch_seq_len  # 方便兼容ZH时也能使用ori_seq_len

            if self.args.pretrain_mode == 'feature_based':
                batch_input_pts = torch.nn.utils.rnn.pad_sequence(batch_input_pts, batch_first=True, padding_value=0.)  # [b,len,1024]

            if batch_tag_ids:
                if '[PAD]' in self.tag2id:
                    padding_value = self.tag2id['[PAD]']  # 补PAD 当tag2id有pad时
                else:
                    padding_value = self.tag2id['O']  # 补O
                batch_tag_ids = torch.nn.utils.rnn.pad_sequence(batch_tag_ids, batch_first=True, padding_value=padding_value)  # [b,len]  # 补O
            else:
                batch_tag_ids = None

            if batch_distilled_task_ent_output:
                batch_distilled_task_ent_output = torch.nn.utils.rnn.pad_sequence(batch_distilled_task_ent_output, batch_first=True, padding_value=0.)
            else:
                batch_distilled_task_ent_output = None

            return {
                'input_ids': tensorize(batch_input_ids),
                'bert_token_type_ids': tensorize(batch_bert_token_type_ids),
                'bert_attention_mask': tensorize(batch_bert_attention_mask),
                'seq_len': tensorize(batch_seq_len),
                'batch_ner_exm': batch_ner_exm,

                'ori_seq_len': tensorize(batch_ori_seq_len),
                'batch_ori_2_tok': tensorize(batch_ori_2_tok),

                'batch_tag_ids': batch_tag_ids,

                'batch_input_pts': tensorize(batch_input_pts) if self.args.pretrain_mode == 'feature_based' else None,

                'batch_distilled_task_ent_output': batch_distilled_task_ent_output,
            }

        return {'span': span_batcher,
                'seq': seq_batcher,
                }.get(arch, None)

    def build_dataset(self, data_source, lang='ENG', arch='span', loss_type=None):
        """构造数据集"""
        if isinstance(data_source, (str, Path)):
            exm_lst = NerExample.load_from_jsonl(data_source)
        else:
            exm_lst = data_source

        if lang == 'ENG':
            for exm in exm_lst:
                if not hasattr(exm, 'ori_2_tok') or not hasattr(exm, 'bert_tok_char_lst'):
                    exm.update_to_bert_tokenize(self.tokenizer, is_split_into_words=True)
        for i, exm in enumerate(exm_lst):
            if hasattr(exm, 'bert_tok_char_lst'):
                if len(exm.bert_tok_char_lst) > self.max_len - 2:
                    print(f'[index:{i}] find one exception example due to bert_tok_char_lst longer then max_len({self.max_len})')
                    exm.truncate_by_bert_tok_char_lst(max_size=self.max_len - 2, direction='tail')
                    # print(f'strip one example due to bert_tok_char_lst longer then max_len({max_len})')
                    # continue
            else:
                exm.truncate(max_size=self.max_len - 2, direction='tail')

        if loss_type is None:
            loss_type = self.loss_type
        return LazyDataset(exm_lst, self.post_process,
                           post_process_args=dict(lang=lang, train=True, arch=arch, loss_type=loss_type)
                           )


# class LazyDataset(DataLoaderX):
class LazyDataset(torch.utils.data.Dataset):
    """LazyDataset"""

    def __init__(self, instances, post_process_fn, post_process_args):
        self.instances = instances
        self.post_process_fn = post_process_fn
        self.post_process_args = post_process_args

    def __getitem__(self, idx):
        """Get the instance with index idx"""
        return self.post_process_fn(self.instances[idx], **self.post_process_args)  # 在DataLoader的时候才对输入进行处理(wrapper) 所以叫Lazy

    def __len__(self):
        return len(self.instances)

    def __str__(self):
        return f"<LazyDataset> Num:{len(self)}"

    def __repr__(self):
        return str(self)
