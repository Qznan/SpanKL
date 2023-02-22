import sys, random, copy, math
import numpy as np
import torch
import torch.nn as nn
from typing import *
from transformers import BertConfig, BertModel, AdamW, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
import ipdb
import logging

logger = logging.getLogger(__name__)

# Below 2 functions are for task embedding, not used in this published paper.
def gumbel_sigmoid_oldbug(logits, tau=2 / 3, hard=True, use_gumbel=True, generator=None):
    """gumbel-sigmoid estimator"""
    # tau = 1
    # tau = 0.1
    # tau = 1/3
    # tau=1/3
    # 从指数分布中抽样 exponential_() 能被seed控制
    if use_gumbel:
        gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_(generator=generator).log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.sigmoid()
    else:  # 不要gumbel采样了
        y_soft = (logits / tau).sigmoid()
    if hard:
        # Straight through.
        y_hard = (y_soft > 0.5).float()
        ret = y_hard - y_soft.detach() + y_soft  # 用了hard之后激活是0的logits梯度也不会得到梯度
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret  # if hard = True then binary else continuous


# bernoully_logits = torch.stack([gate_logit, -gate_logit], dim=0)
# gate = F.gumbel_softmax(bernoully_logits, tau=2/3, hard=True, dim=0)[0]
def gumbel_sigmoid(logits, tau=2 / 3, hard=True, use_gumbel=True, dim=-1, generator=None):
    tau = 2 / 3
    # sigmoid to softmax
    # print(logits.shape)
    # print((logits>=0.).float().sum(-1))
    # print(logits.sigmoid().mean(-1) * 768)
    # print(logits.sigmoid().mean(-1))
    # print(torch.sum(torch.relu(logits))/torch.sum(logits>=0.))
    # print(-torch.sum(torch.relu(-logits))/torch.sum(-logits>=0.))
    logits = torch.stack([logits, torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)], dim=-1)

    if use_gumbel:
        gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_(generator=generator).log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        # gumbels = (logits/tau + gumbels)  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)
    else:
        y_soft = (logits / tau).softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    ret = ret[..., 0]
    # print(ret.sum(-1))
    return ret


def transpose_for_scores(x, num_heads, head_size):
    """ split head """
    # x: [bat,len,totalhid]
    new_x_shape = x.size()[:-1] + (num_heads, head_size)  # [bat,len,num_ent,hid]
    # new_x_shape = x.size()[:-1] + (self.num_ent*2 + 1, self.hidden_size_per_ent)  # [bat,len,num_ent,hid]
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)  # [bat,num_ent,len,hid]


def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    """ mask 句子非pad部分为 1"""
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1)
    if lengths.is_cuda:
        row_vector = row_vector.cuda()
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask.type(dtype)
    return mask


def count_params(model_or_params: Union[torch.nn.Module, torch.nn.Parameter, List[torch.nn.Parameter]],
                 return_trainable=True, verbose=True):
    """
    NOTE: `nn.Module.parameters()` return a `Generator` which can only been iterated ONCE.
    Hence, `model_or_params` passed-in must be a `List` of parameters (which can be iterated multiple times).
    """
    if isinstance(model_or_params, torch.nn.Module):
        model_or_params = list(model_or_params.parameters())
    elif isinstance(model_or_params, torch.nn.Parameter):
        model_or_params = [model_or_params]
    elif not isinstance(model_or_params, list):
        raise TypeError("`model_or_params` is neither a `torch.nn.Module` nor a list of `torch.nn.Parameter`, "
                        "`model_or_params` should NOT be a `Generator`. ")

    num_trainable = sum(p.numel() for p in model_or_params if p.requires_grad)
    num_frozen = sum(p.numel() for p in model_or_params if not p.requires_grad)

    if verbose:
        logger.info(f"The model has {num_trainable + num_frozen:,} parameters, "
                    f"in which {num_trainable:,} are trainable and {num_frozen:,} are frozen.")

    if return_trainable:
        return num_trainable
    else:
        return num_trainable + num_frozen


def check_param_groups(model: torch.nn.Module, param_groups: list, verbose=True):
    # grouped_params_set = set()
    # for d in param_groups:
    #     for p in d['params']:
    #         grouped_params_set.add(id(p))
    # # assert grouped_params_set == set([id(p) for p in model.parameters()])
    # is_equal = (grouped_params_set == set([id(p) for p in model.parameters()]))

    num_grouped_params = sum(count_params(group['params'], verbose=False) for group in param_groups)
    num_model_params = count_params(model, verbose=False)
    is_equal = (num_grouped_params == num_model_params)

    if verbose:
        if is_equal:
            logger.info(f"Grouped parameters ({num_grouped_params:,}) == Model parameters ({num_model_params:,})")
        else:
            logger.warning(f"Grouped parameters ({num_grouped_params:,}) != Model parameters ({num_model_params:,})")
    return is_equal


class NerModel(torch.nn.Module):
    def save_model(self, path, info=''):
        torch.save({
            'state_dict': self.state_dict(),
            'opt': self.opt.state_dict(),
        }, path)
        logger.info(f'[{info}] Saved Model: {path}')

    def load_model(self, path, info='', **kwargs):
        if hasattr(self.args, 'device'):
            map_location = self.args.device
        else:
            map_location = None
        dct = torch.load(path, **kwargs, map_location=map_location)
        self.load_state_dict(dct['state_dict'])
        self.opt.load_state_dict(dct['opt'])
        logger.info(f'[{info}] Loaded Model: {path}')

    def init_opt(self):
        if hasattr(self, 'grouped_params'):
            params = self.grouped_params
        else:
            params = self.parameters()
        self.opt = torch.optim.AdamW(params, lr=self.lr)  # default weight_decay=1e-2
        # self.opt = AdamW(params, lr=self.lr)  # Transformer impl. default weight_decay=0.

    def init_lrs(self, num_step_per_epo=None, epo=None, num_warmup_steps=None):
        if epo is None:
            epo = self.args.num_epochs
        if num_step_per_epo is None:
            # num_step_per_epo = (num_training_instancs - 1) // self.args.batch_size + 1
            num_step_per_epo = 1234
        num_training_steps = num_step_per_epo * epo
        if num_warmup_steps is None:
            ratio = 0.1
            num_warmup_steps = ratio * num_training_steps
        # print(num_training_instancs, epo, ratio, num_step_per_epo, num_training_steps)
        self.lrs = get_cosine_schedule_with_warmup(self.opt, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        # self.lrs = get_constant_schedule_with_warmup(self.opt, num_warmup_steps=num_warmup_steps)


class BaselineExtend(NerModel):
    def __init__(self, args, loader):
        super(BaselineExtend, self).__init__()
        self.args = args
        self.loader = loader
        self.num_tasks = loader.num_tasks
        self.num_ents_per_task = loader.num_ents_per_task
        self.grad_clip = None
        if args.corpus == 'onto':
            self.grad_clip = 1.0
        if args.corpus == 'fewnerd':
            self.grad_clip = 5.0
        self.use_schedual = True
        self.use_bert = args.pretrain_mode == 'fine_tuning'

        if self.use_bert:
            self.dropout_layer = nn.Dropout(p=args.enc_dropout)
            self.bert_conf = BertConfig.from_pretrained(args.bert_model_dir)
            self.bert_layer = BertModel.from_pretrained(args.bert_model_dir)
            encoder_dim = self.bert_conf.hidden_size
            # self.lr = 5e-5
            # self.lr = 1e-4
            self.lr = self.args.bert_lr  # 5e-5
        else:
            self.bilstm_layer = nn.LSTM(
                input_size=1024,
                hidden_size=512,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
                dropout=0
            )
            encoder_dim = 1024
            self.lr = self.args.lr  # 1e-3

        num_total_ents = sum(self.num_ents_per_task)

        # Extend NER
        self.ent_layer = torch.nn.Linear(encoder_dim, 1 + num_total_ents * 2)  # B I for each ent, O for each task  ExtendNER
        # print('total_params:', sum(p.numel() for p in self.parameters()))
        # print(*[n for n, p in self.named_parameters()], sep='\n')

        self.task_offset_lst = []
        offset_s = 1
        offset_e = None
        for num_ents in self.num_ents_per_task:
            if not self.task_offset_lst:
                offset_e = offset_s + num_ents * 2
                self.task_offset_lst.append([offset_s, offset_e])
            else:
                offset_s = offset_e
                offset_e = offset_s + num_ents * 2
                self.task_offset_lst.append([offset_s, offset_e])

        # self.task_offset_lst = [[1,5], [5, 9]]  # e.g. 2ent per task  ent2id: [O, B-e1, I-e1, B-e2, I-e2, B-e3, I-e3, B_e4, I-e4]
        print('task_offset_lst (task offset in ent_layer dim)', self.task_offset_lst)  # e.g. fewnerd [[0, 15], [15, 32], [32, 53], [53, 78], [78, 97], [97, 114], [114, 127], [127, 140]]
        # print('offset_split (num_dim in ent_layer per task)', self.offset_split)  # e.g. fewnerd   # [15, 32, 53, 78, 97, 114, 127]
        # print('taskid2tagid_range', self.taskid2tagid_range)  # e.g. fewnerd {0: [1, 14], 1: [15, 30], 2: [31, 50], 3: [51, 74], 4: [75, 92], 5: [93, 108], 6: [109, 120], 7: [121, 132]}

        self.ce_loss_layer = nn.CrossEntropyLoss(reduction='none')
        count_params(self)

        if self.use_bert:
            fast_lr = self.args.lr  # 1e-3
            no_decay = ['bias', 'LayerNorm.weight']
            p1 = [p for n, p in self.named_parameters() if n == 'task_embed']
            entlayer_p_weight = [p for n, p in self.named_parameters() if 'ent_layer' in n and 'weight' in n]
            entlayer_p_bias = [p for n, p in self.named_parameters() if 'ent_layer' in n and 'bias' in n]
            p2 = [p for n, p in self.named_parameters() if n != 'task_embed' and 'ent_layer' not in n and any(nd in n for nd in no_decay)]
            p3 = [p for n, p in self.named_parameters() if n != 'task_embed' and 'ent_layer' not in n and not any(nd in n for nd in no_decay)]
            self.grouped_params = [
                {'params': entlayer_p_weight, 'lr': fast_lr},
                {'params': entlayer_p_bias, 'weight_decay': 0.0, 'lr': fast_lr},
                {'params': p2, 'weight_decay': 0.0},
                {'params': p3},
            ]
            if p1:  # using task emb
                self.grouped_params = [{'params': p1, 'weight_decay': 0.0, 'lr': fast_lr}] + self.grouped_params

            check_param_groups(self, self.grouped_params)

        self.init_opt()
        if self.use_schedual:
            self.init_lrs()

    def encode(self, inputs_dct):
        batch_input_pts = inputs_dct['batch_input_pts']  # [bsz, len, 1024]
        seq_len = inputs_dct['ori_seq_len']

        if self.use_bert:
            bert_outputs = self.bert_layer(input_ids=inputs_dct['input_ids'],
                                           token_type_ids=inputs_dct['bert_token_type_ids'],
                                           attention_mask=inputs_dct['bert_attention_mask'],
                                           output_hidden_states=True,
                                           )
            seq_len_lst = seq_len.tolist()
            bert_out = bert_outputs.last_hidden_state
            # 去除bert_output[CLS]和[SEP]
            bert_out_lst = [t for t in bert_out]  # split along batch
            for i, t in enumerate(bert_out_lst):  # iter along batch
                # tensor [len, hid]
                bert_out_lst[i] = torch.cat([t[1: 1 + seq_len_lst[i]], t[2 + seq_len_lst[i]:]], 0)
            bert_out = torch.stack(bert_out_lst, 0)  # stack along batch

            batch_ori_2_tok = inputs_dct['batch_ori_2_tok']
            if batch_ori_2_tok.shape[0]:  # 只取子词的第一个字  ENG
                bert_out_lst = [t for t in bert_out]
                for bdx, t in enumerate(bert_out_lst):
                    ori_2_tok = batch_ori_2_tok[bdx]
                    bert_out_lst[bdx] = bert_out_lst[bdx][ori_2_tok]
                bert_out = torch.stack(bert_out_lst, 0)

            bert_out = self.dropout_layer(bert_out)  # don't forget
            encode_output = bert_out

        else:  # bilstm
            pack_embed = torch.nn.utils.rnn.pack_padded_sequence(batch_input_pts, seq_len.cpu(), batch_first=True, enforce_sorted=False)
            pack_out, _ = self.bilstm_layer(pack_embed)
            rnn_output_x, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_out, batch_first=True)  # [bat,len,hid]
            encode_output = rnn_output_x

        # ffn
        encode_output = self.ent_layer(encode_output)
        self.batch_tag_tensor = encode_output
        return encode_output

    def decode(self, ent_output, seq_len, task_id):
        offset_s, offset_e = self.task_offset_lst[task_id]

        mask = sequence_mask(seq_len, dtype=torch.uint8)
        # decode_ids = self.crf_layer.decode(ent_output, mask)

        ent_output_prob = ent_output[:, :, :offset_e].softmax(-1)

        id2tag = self.loader.datareader.id2tag
        curr_task_id2tag = {i: t for i, t in id2tag.items() if i < offset_e}
        curr_task_tag2id = {t: i for i, t in curr_task_id2tag.items()}
        curr_task_num_tags = len(curr_task_id2tag)
        trans_mask = get_BIO_transitions_mask(curr_task_tag2id, curr_task_id2tag)
        trans_mask = torch.tensor(1 - trans_mask).to(ent_output_prob.device)
        num_valid_for_prev_tag = trans_mask.sum(-1)
        trans_mask = trans_mask / num_valid_for_prev_tag
        start_transitions = torch.full([curr_task_num_tags], 1 / curr_task_num_tags).to(ent_output_prob.device)
        end_transitions = torch.full([curr_task_num_tags], 1 / curr_task_num_tags).to(ent_output_prob.device)

        decode_ids = viterbi_decode(ent_output_prob, mask, start_transitions, end_transitions, trans_mask)
        return decode_ids

    def calc_ce_loss(self, target, predict, seq_len_mask, ce_mask):
        # target [b,l]  predict [b,l,tag]
        ce_loss = self.ce_loss_layer(predict.transpose(1, 2), target)  # [b,ent,l] [b,l] -> [b,l]
        ce_loss = ce_loss * seq_len_mask
        ce_loss = ce_loss * ce_mask
        # return span_loss。mean()  # [b,l]

        num_ce = torch.sum(torch.logical_and(seq_len_mask, ce_mask))
        if num_ce == 0.:
            ce_loss = 0.
        else:
            ce_loss = ce_loss.sum() / num_ce
        # print(num_ce, ce_loss)
        return ce_loss  # [b,l]

    def calc_kl_loss(self, predict, target, seq_len_mask, kl_mask, ofs, ofe):
        # kl_loss = torch.nn.functional.kl_div(log_kl_pred.double(), target.double(), reduction='none')  # [num_spans, ent, 2]
        # target = target / 2
        # print('target', target)
        # print('predict', predict)
        # target_prob = target.softmax(dim=-1)
        # print('target_prob', target.softmax(dim=-1))
        # print('predic_prob', predict.softmax(dim=-1))
        # curr_ent_dim = ofe - ofs
        # pad_tenosr = torch.zeros_like(target_prob)[:, :, :curr_ent_dim]
        # target_prob = torch.cat([target_prob, pad_tenosr], dim=-1)

        # kl_loss = torch.nn.functional.kl_div(predict.softmax(dim=-1).log(), target_prob, reduction='none')  # [b,l,dim]
        kl_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(predict, dim=-1),
            torch.nn.functional.log_softmax(target, dim=-1),
            log_target=True,
            reduction='none')  # [b,l,dim]

        kl_loss = kl_loss.sum(-1)  # [b,l]
        kl_loss = kl_loss * seq_len_mask
        kl_loss = kl_loss * kl_mask

        num_kl = torch.sum(torch.logical_and(seq_len_mask, kl_mask))
        if num_kl == 0.:
            kl_loss = 0.
        else:
            kl_loss = kl_loss.sum() / num_kl
        # print(num_kl, kl_loss)
        # print(seq_len_mask.sum())
        return kl_loss  # [b,l]

    def calc_loss(self, ce_target, kl_target, ent_output, seq_len, curr_task_id):
        # task_ent_output  # :截至当前任务的 0:offset_e
        seq_len_mask = sequence_mask(seq_len)  # b,l
        ofs, ofe = self.task_offset_lst[curr_task_id]  # 当前任务的offset是
        if curr_task_id == 0:  # 第一个任务
            ce_mask = seq_len_mask  # ce_mask [b,l] 哪些是要计算ce_loss的 第一个任务所有都要。
            predict = ent_output[:, :, :ofe]
            ce_loss = self.calc_ce_loss(ce_target, predict, seq_len_mask, ce_mask)
            kl_loss = torch.tensor(0.)

        else:  # 后续任务
            ce_mask = torch.logical_and(ce_target >= ofs, ce_target < ofe).float()  # 之后的任务当前token属于新任务的ent时要
            predict = ent_output[:, :, :ofe]
            ce_loss = self.calc_ce_loss(ce_target, predict, seq_len_mask, ce_mask)

            kl_mask = 1. - ce_mask
            # kl_predict = ent_output[:, :, :ofs]
            kl_predict = ent_output[:, :, :ofe]  # 用 小值来补新标签对应的B和I
            # print(kl_target.shape)
            kl_target = torch.nn.functional.pad(kl_target, (0, ofe - ofs), mode='constant', value=-1e8)
            # print(kl_target.shape)
            # print(kl_predict.shape)
            # ipdb.set_trace()
            kl_loss = self.calc_kl_loss(kl_predict, kl_target, seq_len_mask, kl_mask, ofs, ofe)

        return ce_loss, kl_loss

    def runloss(self, inputs_dct, task_id):
        seq_len = inputs_dct['ori_seq_len']
        batch_tag_ids = inputs_dct['batch_tag_ids']  # [b,l]

        batch_distilled_task_ent_output = inputs_dct['batch_distilled_task_ent_output']

        ent_output = self.encode(inputs_dct)

        # 过滤其他任务的ent
        ofs, ofe = self.task_offset_lst[task_id]  # 当前任务的offset是
        curr_task_batch_tag_ids = batch_tag_ids.masked_fill(torch.logical_or(batch_tag_ids >= ofe, batch_tag_ids < ofs), 0.)  # [b,l]
        # print(inputs_dct['batch_ner_exm'][0])
        # print(batch_tag_ids[0])
        # print(curr_task_batch_tag_ids[0])
        # ipdb.set_trace()
        ce_loss, kl_loss = self.calc_loss(curr_task_batch_tag_ids, batch_distilled_task_ent_output, ent_output, seq_len, task_id)

        self.ce_loss = ce_loss
        # self.ce_loss = ce_loss.mean()
        self.kl_loss = kl_loss
        # self.kl_loss = kl_loss.mean()

        self.total_loss = self.ce_loss + self.kl_loss

        self.opt.zero_grad()
        self.total_loss.backward()

        if self.grad_clip is None:
            self.total_norm = 0
        else:
            self.total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        self.opt.step()

        if self.use_schedual:
            self.lrs.step()

        self.curr_lr = self.opt.param_groups[0]['lr']

        return (float(self.total_loss),
                float(self.ce_loss),
                float(self.kl_loss),
                )

    def run_loss_non_cl(self, inputs_dct, task_id):
        seq_len = inputs_dct['ori_seq_len']
        batch_tag_ids = inputs_dct['batch_tag_ids']  # [b,l]

        batch_distilled_task_ent_output = inputs_dct['batch_distilled_task_ent_output']

        ent_output = self.encode(inputs_dct)

        # 过滤后面任务的ent
        ofs, ofe = self.task_offset_lst[task_id]  # 当前任务的offset是
        curr_task_batch_tag_ids = batch_tag_ids.masked_fill(batch_tag_ids >= ofe, 0.)  # [b,l]
        seq_len_mask = sequence_mask(seq_len)  # b,l

        ce_mask = seq_len_mask  # ce_mask [b,l] 哪些是要计算ce_loss的 第一个任务所有都要。
        predict = ent_output[:, :, :ofe]
        self.ce_loss = self.calc_ce_loss(curr_task_batch_tag_ids, predict, seq_len_mask, ce_mask)

        self.kl_loss = 0.

        self.total_loss = self.ce_loss + self.kl_loss

        self.opt.zero_grad()
        self.total_loss.backward()

        if self.grad_clip is None:
            self.total_norm = 0
        else:
            self.total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        self.opt.step()

        if self.use_schedual:
            self.lrs.step()

        self.curr_lr = self.opt.param_groups[0]['lr']

        return (float(self.total_loss),
                float(self.ce_loss),
                float(self.kl_loss),
                )


class BaselineAdd(NerModel):
    def __init__(self, args, loader):
        super(BaselineAdd, self).__init__()
        self.args = args
        self.loader = loader
        self.num_tasks = loader.num_tasks
        self.num_ents_per_task = loader.num_ents_per_task
        self.grad_clip = None
        if args.corpus == 'onto':
            self.grad_clip = 1.0
        if args.corpus == 'fewnerd':
            self.grad_clip = 5.0
        self.use_schedual = True
        self.use_bert = args.pretrain_mode == 'fine_tuning'
        if self.use_bert:
            self.dropout_layer = nn.Dropout(p=args.enc_dropout)
            self.bert_conf = BertConfig.from_pretrained(args.bert_model_dir)
            self.bert_layer = BertModel.from_pretrained(args.bert_model_dir)
            encoder_dim = self.bert_conf.hidden_size
            # self.lr = 5e-5
            # self.lr = 1e-4
            self.lr = self.args.bert_lr  # 5e-5
        else:
            self.bilstm_layer = nn.LSTM(
                input_size=1024,
                hidden_size=512,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
                dropout=0
            )
            encoder_dim = 1024
            self.lr = self.args.lr  # 1e-3

        num_total_ents = sum(self.num_ents_per_task)

        # Add NER
        self.ent_layer = torch.nn.Linear(encoder_dim, self.num_tasks + num_total_ents * 2)  # B I for each ent, O for each task AddNER
        # print('total_params:', sum(p.numel() for p in self.parameters()))
        # print(*[n for n, p in self.named_parameters()], sep='\n')

        self.task_offset_lst = []  # the dim position in ent_layer
        offset_s = 0
        offset_e = None
        for num_ents in self.num_ents_per_task:
            if not self.task_offset_lst:
                offset_e = 1 + num_ents * 2
                self.task_offset_lst.append([offset_s, offset_e])
            else:
                offset_s = offset_e
                offset_e = offset_e + (1 + num_ents * 2)
                self.task_offset_lst.append([offset_s, offset_e])

        self.offset_split = [e[0] for e in self.task_offset_lst][1:]  # 用以np.split
        # np.split(tag_logtis, offset_split, axis=-1) split each task of ent_layer [O,B,I][O,B,I]...

        self.taskid2tagid_range = {}  # the BIO-tag id in tag2id dict, use to process the original input label.
        for task_id in range(self.num_tasks):
            start_ent = self.loader.entity_task_lst[task_id][0]
            end_ent = self.loader.entity_task_lst[task_id][-1]
            self.taskid2tagid_range[task_id] = [self.loader.datareader.tag2id[f'B-{start_ent}'],
                                                self.loader.datareader.tag2id[f'I-{end_ent}']
                                                ]
        print('task_offset_lst (task offset in ent_layer dim)', self.task_offset_lst)  # e.g. fewnerd [[0, 15], [15, 32], [32, 53], [53, 78], [78, 97], [97, 114], [114, 127], [127, 140]]
        print('offset_split (num_dim in ent_layer per task)', self.offset_split)  # e.g. fewnerd   # [15, 32, 53, 78, 97, 114, 127]
        print('taskid2tagid_range', self.taskid2tagid_range)  # e.g. fewnerd {0: [1, 14], 1: [15, 30], 2: [31, 50], 3: [51, 74], 4: [75, 92], 5: [93, 108], 6: [109, 120], 7: [121, 132]}

        # e.g. onto
        # task_offset_lst [[0, 3], [3, 6], [6, 9], [9, 12], [12, 15], [15, 18]]
        # offset_split [3, 6, 9, 12, 15]
        # taskid2tagid_range {0: [1, 2], 1: [3, 4], 2: [5, 6], 3: [7, 8], 4: [9, 10], 5: [11, 12]}

        # each task dummy id2tag
        self.tid_id2tag_lst = []
        for tid, ents in enumerate(loader.entity_task_lst):
            id2tag = {0: 'O'}
            for ent in ents:
                id2tag[len(id2tag)] = f'B-{ent}'
                id2tag[len(id2tag)] = f'I-{ent}'
            self.tid_id2tag_lst.append(id2tag)

        self.ce_loss_layer = nn.CrossEntropyLoss(reduction='none')
        count_params(self)

        if self.use_bert:
            fast_lr = self.args.lr  # 1e-3
            no_decay = ['bias', 'LayerNorm.weight']
            p1 = [p for n, p in self.named_parameters() if n == 'task_embed']
            entlayer_p_weight = [p for n, p in self.named_parameters() if 'ent_layer' in n and 'weight' in n]
            entlayer_p_bias = [p for n, p in self.named_parameters() if 'ent_layer' in n and 'bias' in n]
            p2 = [p for n, p in self.named_parameters() if n != 'task_embed' and 'ent_layer' not in n and any(nd in n for nd in no_decay)]
            p3 = [p for n, p in self.named_parameters() if n != 'task_embed' and 'ent_layer' not in n and not any(nd in n for nd in no_decay)]
            self.grouped_params = [
                {'params': entlayer_p_weight, 'lr': fast_lr},
                {'params': entlayer_p_bias, 'weight_decay': 0.0, 'lr': fast_lr},
                {'params': p2, 'weight_decay': 0.0},
                {'params': p3},
            ]
            if p1:  # using task emb
                self.grouped_params = [{'params': p1, 'weight_decay': 0.0, 'lr': fast_lr}] + self.grouped_params

            check_param_groups(self, self.grouped_params)

        self.init_opt()
        if self.use_schedual:
            self.init_lrs()

    def encode(self, inputs_dct):
        batch_input_pts = inputs_dct['batch_input_pts']  # [bsz, len, 1024]
        seq_len = inputs_dct['ori_seq_len']

        if self.use_bert:
            bert_outputs = self.bert_layer(input_ids=inputs_dct['input_ids'],
                                           token_type_ids=inputs_dct['bert_token_type_ids'],
                                           attention_mask=inputs_dct['bert_attention_mask'],
                                           output_hidden_states=True,
                                           )
            seq_len_lst = seq_len.tolist()
            bert_out = bert_outputs.last_hidden_state
            # 去除bert_output[CLS]和[SEP]
            bert_out_lst = [t for t in bert_out]  # split along batch
            for i, t in enumerate(bert_out_lst):  # iter along batch
                # tensor [len, hid]
                bert_out_lst[i] = torch.cat([t[1: 1 + seq_len_lst[i]], t[2 + seq_len_lst[i]:]], 0)
            bert_out = torch.stack(bert_out_lst, 0)  # stack along batch

            batch_ori_2_tok = inputs_dct['batch_ori_2_tok']
            if batch_ori_2_tok.shape[0]:  # 只取子词的第一个字  ENG
                bert_out_lst = [t for t in bert_out]
                for bdx, t in enumerate(bert_out_lst):
                    ori_2_tok = batch_ori_2_tok[bdx]
                    bert_out_lst[bdx] = bert_out_lst[bdx][ori_2_tok]
                bert_out = torch.stack(bert_out_lst, 0)

            bert_out = self.dropout_layer(bert_out)  # don't forget
            encode_output = bert_out

        else:  # bilstm
            pack_embed = torch.nn.utils.rnn.pack_padded_sequence(batch_input_pts, seq_len.cpu(), batch_first=True, enforce_sorted=False)
            pack_out, _ = self.bilstm_layer(pack_embed)
            rnn_output_x, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_out, batch_first=True)  # [bat,len,hid]
            encode_output = rnn_output_x

        # ffn
        encode_output = self.ent_layer(encode_output)  # [b,l,t]
        self.batch_tag_tensor = encode_output
        return encode_output

    def decode(self, ent_output, seq_len, task_id):
        offset_s, offset_e = self.task_offset_lst[task_id]

        mask = sequence_mask(seq_len, dtype=torch.uint8)
        # decode_ids = self.crf_layer.decode(ent_output, mask)

        ent_output_prob = ent_output[:, :, :offset_e].softmax(-1)

        id2tag = self.loader.datareader.id2tag
        curr_task_id2tag = {i: t for i, t in id2tag.items() if i < offset_e}
        curr_task_tag2id = {t: i for i, t in curr_task_id2tag.items()}
        curr_task_num_tags = len(curr_task_id2tag)
        trans_mask = get_BIO_transitions_mask(curr_task_tag2id, curr_task_id2tag)
        trans_mask = torch.tensor(1 - trans_mask).to(ent_output_prob.device)
        num_valid_for_prev_tag = trans_mask.sum(-1)
        trans_mask = trans_mask / num_valid_for_prev_tag
        start_transitions = torch.full([curr_task_num_tags], 1 / curr_task_num_tags).to(ent_output_prob.device)
        end_transitions = torch.full([curr_task_num_tags], 1 / curr_task_num_tags).to(ent_output_prob.device)

        decode_ids = viterbi_decode(ent_output_prob, mask, start_transitions, end_transitions, trans_mask)
        return decode_ids

    def decode_one_task(self, ent_output, seq_len, task_id):
        offset_s, offset_e = self.task_offset_lst[task_id]

        mask = sequence_mask(seq_len, dtype=torch.uint8)
        # decode_ids = self.crf_layer.decode(ent_output, mask)

        ent_output_prob = ent_output[:, :, offset_s:offset_e].softmax(-1)

        curr_task_id2tag = self.tid_id2tag_lst[task_id]
        curr_task_tag2id = {t: i for i, t in curr_task_id2tag.items()}
        curr_task_num_tags = len(curr_task_id2tag)
        trans_mask = get_BIO_transitions_mask(curr_task_tag2id, curr_task_id2tag)
        trans_mask = torch.tensor(1 - trans_mask).to(ent_output_prob.device)
        num_valid_for_prev_tag = trans_mask.sum(-1)
        trans_mask = trans_mask / num_valid_for_prev_tag
        start_transitions = torch.full([curr_task_num_tags], 1 / curr_task_num_tags).to(ent_output_prob.device)
        end_transitions = torch.full([curr_task_num_tags], 1 / curr_task_num_tags).to(ent_output_prob.device)

        decode_ids = viterbi_decode(ent_output_prob, mask, start_transitions, end_transitions, trans_mask)
        return decode_ids

    def calc_ce_loss(self, target, predict, seq_len_mask, ce_mask):
        # target [b,l]  predict [b,l,ent]
        ce_loss = self.ce_loss_layer(predict.transpose(1, 2), target)  # [b,ent,l] [b,l] -> [b,l]
        ce_loss = ce_loss * seq_len_mask
        ce_loss = ce_loss * ce_mask
        # return span_loss。mean()  # [b,l]
        return ce_loss  # [b,l]

    def calc_kl_loss(self, predict, target, seq_len_mask, kl_mask):
        # kl_loss = torch.nn.functional.kl_div(log_kl_pred.double(), target.double(), reduction='none')  # [num_spans, ent, 2]
        # target = target / 2.  # temperature
        target_prob = target.softmax(dim=-1)
        # curr_ent_dim = ofe - ofs
        # pad_tenosr = torch.zeros_like(target_prob)[:, :, :curr_ent_dim]
        # target_prob = torch.cat([target_prob, pad_tenosr], dim=-1)

        kl_loss = torch.nn.functional.kl_div(predict.softmax(dim=-1).log(), target_prob, reduction='none')  # [b,l,dim]
        kl_loss = kl_loss.sum(-1)  # [b,l]
        kl_loss = kl_loss * seq_len_mask
        kl_loss = kl_loss * kl_mask

        return kl_loss  # [b,l]

    def runloss(self, inputs_dct, task_id):
        seq_len = inputs_dct['ori_seq_len']
        batch_tag_ids = inputs_dct['batch_tag_ids']  # [b,l]
        seq_len_mask = sequence_mask(seq_len)

        batch_distilled_task_ent_output = inputs_dct['batch_distilled_task_ent_output']

        ent_output = self.encode(inputs_dct)

        # 过滤其他任务的ent
        ofs, ofe = self.task_offset_lst[task_id]  # 当前任务的offset是
        tagid_s, tagid_e = self.taskid2tagid_range[task_id]

        need_to_change_mask = torch.logical_and(batch_tag_ids >= tagid_s, batch_tag_ids <= tagid_e)

        curr_task_batch_tag_ids = batch_tag_ids + need_to_change_mask.int() * (-ofs + task_id)

        curr_task_batch_tag_ids = curr_task_batch_tag_ids.masked_fill(torch.logical_not(need_to_change_mask), 0.)  # [b,l]
        self.ce_loss = self.calc_ce_loss(curr_task_batch_tag_ids, ent_output[:, :, ofs:ofe], seq_len_mask, seq_len_mask)
        self.ce_loss = self.ce_loss.mean()
        # self.ce_loss = self.ce_loss.mean(-1)
        # self.ce_loss = self.ce_loss.mean(-1)

        if task_id > 0:
            all_task_kl_loss = []
            for tid in range(task_id):
                ofs, ofe = self.task_offset_lst[tid]
                task_kl_loss = self.calc_kl_loss(ent_output[:, :, ofs:ofe],
                                                 batch_distilled_task_ent_output[:, :, ofs:ofe],
                                                 seq_len_mask, kl_mask=seq_len_mask)  # [b,l]
                task_kl_loss = torch.mean(task_kl_loss)
                all_task_kl_loss.append(task_kl_loss)
            self.kl_loss = sum(all_task_kl_loss) / len(all_task_kl_loss)
        else:
            self.kl_loss = 0

        self.total_loss = self.ce_loss + self.kl_loss

        self.opt.zero_grad()
        self.total_loss.backward()

        if self.grad_clip is None:
            self.total_norm = 0
        else:
            self.total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        self.opt.step()

        if self.use_schedual:
            self.lrs.step()

        self.curr_lr = self.opt.param_groups[0]['lr']

        return (float(self.total_loss),
                float(self.ce_loss),
                float(self.kl_loss),
                )

    def run_loss_non_cl(self, inputs_dct, task_id):
        seq_len = inputs_dct['ori_seq_len']
        batch_tag_ids = inputs_dct['batch_tag_ids']  # [b,l]
        seq_len_mask = sequence_mask(seq_len)
        batch_distilled_task_ent_output = inputs_dct['batch_distilled_task_ent_output']
        ent_output = self.encode(inputs_dct)

        ce_loss_lst = []
        for tid in range(task_id + 1):
            # 过滤其他任务的ent
            ofs, ofe = self.task_offset_lst[tid]  # 当前任务的offset是
            tagid_s, tagid_e = self.taskid2tagid_range[tid]

            need_to_change_mask = torch.logical_and(batch_tag_ids >= tagid_s, batch_tag_ids <= tagid_e)

            curr_task_batch_tag_ids = batch_tag_ids + need_to_change_mask.int() * (-ofs + tid)

            curr_task_batch_tag_ids = curr_task_batch_tag_ids.masked_fill(torch.logical_not(need_to_change_mask), 0.)  # [b,l]
            ce_loss = self.calc_ce_loss(curr_task_batch_tag_ids, ent_output[:, :, ofs:ofe], seq_len_mask, seq_len_mask)
            ce_loss = ce_loss.mean()
            # ce_loss = ce_loss.mean(-1)
            # ce_loss = ce_loss.mean(-1)
            ce_loss_lst.append(ce_loss)

        self.ce_loss = sum(ce_loss_lst) / len(ce_loss_lst)

        self.kl_loss = 0.

        self.total_loss = self.ce_loss + self.kl_loss

        self.opt.zero_grad()
        self.total_loss.backward()

        if self.grad_clip is None:
            self.total_norm = 0
        else:
            self.total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        self.opt.step()

        if self.use_schedual:
            self.lrs.step()

        self.curr_lr = self.opt.param_groups[0]['lr']

        return (float(self.total_loss),
                float(self.ce_loss),
                float(self.kl_loss),
                )


class SpanKL(NerModel):
    def __init__(self, args, loader):
        super(SpanKL, self).__init__()
        self.args = args
        self.loader = loader
        self.num_tasks = loader.num_tasks
        self.num_ents_per_task = loader.num_ents_per_task
        self.taskid2offset = loader.tid2offset  # 每个任务不同个ent
        self.hidden_size_per_ent = 50
        self.grad_clip = None
        if args.corpus == 'onto':
            self.grad_clip = 1.0
        if args.corpus == 'fewnerd':
            self.grad_clip = 5.0
        self.use_schedual = True
        self.gumbel_generator = torch.Generator(device=args.device)
        self.gumbel_generator.manual_seed(self.args.seed)
        # self.ent_size = self.num_ent = self.loader.ent_dim  # total ent dim across all task
        self.ep = None
        self.use_slr = False  # not used in the published paper

        self.use_bert = args.pretrain_mode == 'fine_tuning'
        if self.use_bert:
            self.dropout_layer = nn.Dropout(p=args.enc_dropout)
            self.bert_conf = BertConfig.from_pretrained(args.bert_model_dir)
            self.bert_layer = BertModel.from_pretrained(args.bert_model_dir)
            encoder_dim = self.bert_conf.hidden_size
            # self.lr = 5e-5
            # self.lr = 1e-4
            self.lr = self.args.bert_lr

            # self.extra_dense = torch.nn.Linear(encoder_dim, encoder_dim)
        else:
            self.bilstm_layer = nn.LSTM(
                input_size=1024,
                hidden_size=512,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
                dropout=0
            )
            encoder_dim = 1024
            self.lr = self.args.lr  # 1e-3

        if self.use_slr:
            self.slr_layer = torch.nn.Linear(encoder_dim, 50 * 2)

        if self.args.use_task_embed:
            # self.task_embed = torch.nn.Embedding(self.num_tasks, 1024)
            self.task_embed = torch.nn.Parameter(torch.Tensor(self.num_tasks, encoder_dim))
            # torch.nn.init.uniform_(self.task_embed.data, -1., 1.)
            torch.nn.init.normal_(self.task_embed.data, mean=0., std=1.)
            # torch.nn.init.normal_(self.task_embed.data, mean=0., std=0.25)  # 不行 会导致更少的门开
            # torch.nn.init.constant_(self.task_embed.data, 0)
            # self.task_embed.data = self.task_embed.data.abs()  # 门全部先开着

            self.gate_tensor_lst = [None] * self.num_tasks  # 用来存放当前前向传播时经过gumble_softmax采样得到的0-1binary vector

        # task dense layer
        self.output_dim_per_task_lst = [2 * self.hidden_size_per_ent * num_ents for num_ents in self.num_ents_per_task]
        # self.task_layers = [nn.Linear(encoder_dim, output_dim_per_task) for output_dim_per_task in self.output_dim_per_task_lst]  # one task one independent layer
        # 这样才能cuda()生效
        self.task_layers = nn.ModuleList([torch.nn.Linear(encoder_dim, output_dim_per_task) for output_dim_per_task in self.output_dim_per_task_lst])  # one task one independent layer
        for layer in self.task_layers:
            torch.nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

        self.bce_loss_layer = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss_layer = torch.nn.MSELoss(reduction='none')
        count_params(self)

        # print(*[n for n, p in self.named_parameters()], sep='\n')
        # ipdb.set_trace()

        if self.use_bert:
            """1"""
            no_decay = ['bias', 'LayerNorm.weight']
            p1 = [p for n, p in self.named_parameters() if n == 'task_embed']
            spanlayer_p_weight = [p for n, p in self.named_parameters() if 'task_layers' in n and 'weight' in n]
            spanlayer_p_bias = [p for n, p in self.named_parameters() if 'task_layers' in n and 'bias' in n]
            p2 = [p for n, p in self.named_parameters() if n != 'task_embed' and 'task_layers' not in n and any(nd in n for nd in no_decay)]
            p3 = [p for n, p in self.named_parameters() if n != 'task_embed' and 'task_layers' not in n and not any(nd in n for nd in no_decay)]
            self.grouped_params = [
                {'params': spanlayer_p_weight, 'lr': 1e-3},
                {'params': spanlayer_p_bias, 'weight_decay': 0.0, 'lr': 1e-3},
                {'params': p2, 'weight_decay': 0.0},
                {'params': p3},
            ]
            if p1:  # using task emb
                self.grouped_params = [{'params': p1, 'weight_decay': 0.0, 'lr': 1e-3}] + self.grouped_params
            """2"""
            # no_decay = ['bias', 'LayerNorm.weight']
            # p1 = [p for n, p in self.named_parameters() if n == 'task_embed']
            # spanlayer_p_weight = [p for n, p in self.named_parameters() if 'task_layers' in n and 'weight' in n]
            # spanlayer_p_bias = [p for n, p in self.named_parameters() if 'task_layers' in n and 'bias' in n]
            # extralayer_w = [p for n, p in self.named_parameters() if 'extra_dense' in n and 'weight' in n]
            # extralayer_b = [p for n, p in self.named_parameters() if 'extra_dense' in n and 'bias' in n]
            # p2 = [p for n, p in self.named_parameters() if n != 'task_embed' and 'task_layers' not in n and 'extra_dense' not in n and any(nd in n for nd in no_decay)]
            # p3 = [p for n, p in self.named_parameters() if n != 'task_embed' and 'task_layers' not in n and 'extra_dense' not in n and not any(nd in n for nd in no_decay)]
            # self.grouped_params = [
            #     {'params': spanlayer_p_weight, 'lr': 1e-3},
            #     {'params': spanlayer_p_bias, 'weight_decay': 0.0, 'lr': 1e-3},
            #     {'params': extralayer_w, 'lr': 1e-3},
            #     {'params': extralayer_b, 'weight_decay': 0.0, 'lr': 1e-3},
            #     {'params': p2, 'weight_decay': 0.0},
            #     {'params': p3},
            # ]
            # if p1:  # using task emb
            #     self.grouped_params = [{'params': p1, 'weight_decay': 0.0, 'lr': 1e-3}]+ self.grouped_params
            """ """
            check_param_groups(self, self.grouped_params)

        self.init_opt()
        if self.use_schedual:
            self.init_lrs()

        # ipdb.set_trace()

    def encoder_forward(self, inputs_dct):
        if self.use_bert:
            seq_len = inputs_dct['seq_len']  # seq_len [bat]
            bert_outputs = self.bert_layer(input_ids=inputs_dct['input_ids'],
                                           token_type_ids=inputs_dct['bert_token_type_ids'],
                                           attention_mask=inputs_dct['bert_attention_mask'],
                                           output_hidden_states=True,
                                           )
            seq_len_lst = seq_len.tolist()
            bert_out = bert_outputs.last_hidden_state
            # 去除bert_output[CLS]和[SEP]
            bert_out_lst = [t for t in bert_out]  # split along batch
            for i, t in enumerate(bert_out_lst):  # iter along batch
                # tensor [len, hid]
                bert_out_lst[i] = torch.cat([t[1: 1 + seq_len_lst[i]], t[2 + seq_len_lst[i]:]], 0)
            bert_out = torch.stack(bert_out_lst, 0)  # stack along batch

            batch_ori_2_tok = inputs_dct['batch_ori_2_tok']
            if batch_ori_2_tok.shape[0]:  # 只取子词的第一个字  ENG
                bert_out_lst = [t for t in bert_out]
                for bdx, t in enumerate(bert_out_lst):
                    ori_2_tok = batch_ori_2_tok[bdx]
                    bert_out_lst[bdx] = bert_out_lst[bdx][ori_2_tok]
                bert_out = torch.stack(bert_out_lst, 0)

            bert_out = self.dropout_layer(bert_out)  # don't forget

            # bert_out = torch.tanh(bert_out)
            # bert_out = self.extra_dense(bert_out)
            return bert_out
        else:
            batch_input_pts = inputs_dct['batch_input_pts']  # [bsz, len, 1024]
            seq_len = inputs_dct['ori_seq_len']
            pack_embed = torch.nn.utils.rnn.pack_padded_sequence(batch_input_pts, seq_len.cpu(), batch_first=True, enforce_sorted=False)
            pack_out, _ = self.bilstm_layer(pack_embed)
            rnn_output_x, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_out, batch_first=True)  # [bat,len,hid]
            return rnn_output_x

    def task_layer_forward(self, encoder_output, use_task_embed=True, use_gumbel_softmax=True, deterministic=False):
        output_per_task_lst = []  # list of [b,l,h~]
        for task_id in range(self.num_tasks):
            if use_task_embed:
                gate_logit = self.task_embed[task_id]  # [1024]
                # calc gate [emb_size]
                if use_gumbel_softmax:  # gate binary vector {0,1}
                    if deterministic:  # when are not training, which unable grad back propagation
                        gate = (gate_logit >= 0.).float()  # deterministic output
                    else:  # when training, which enable grad back propagation
                        # bernoully_logits = torch.stack([gate_logit, -gate_logit], dim=0)
                        # gate = F.gumbel_softmax(bernoully_logits, tau=2/3, hard=True, dim=0)[0]
                        gate = gumbel_sigmoid(gate_logit, hard=True, generator=self.gumbel_generator)  # random output
                        self.gate_tensor_lst[task_id] = gate
                else:
                    gate = torch.sigmoid(gate_logit)  # [0~1] prob

                input_per_task = encoder_output * gate[None, None, :]
            else:
                input_per_task = encoder_output

            output_per_task = self.task_layers[task_id](input_per_task)
            output_per_task_lst.append(output_per_task)
        output_per_task_lst = torch.cat(output_per_task_lst, dim=-1)  # [b,l,h+]
        return output_per_task_lst

    def task_layer_forward1(self, encoder_output, use_task_embed=True, use_gumbel_softmax=True, deterministic_tasks=None):
        output_per_task_lst = []  # list of [b,l,h~]
        for task_id in range(self.num_tasks):
            if use_task_embed:
                gate_logit = self.task_embed[task_id]  # [1024]
                # calc gate [emb_size]
                if use_gumbel_softmax:  # gate binary vector {0,1}
                    if task_id in deterministic_tasks:  # when are not training, which unable grad back propagation
                        gate = (gate_logit >= 0.).float()  # deterministic output
                    else:  # when training, which enable grad back propagation
                        # bernoully_logits = torch.stack([gate_logit, -gate_logit], dim=0)
                        # gate = F.gumbel_softmax(bernoully_logits, tau=2/3, hard=True, dim=0)[0]
                        gate = gumbel_sigmoid(gate_logit, hard=True, generator=self.gumbel_generator)  # random output
                        self.gate_tensor_lst[task_id] = gate
                    # ipdb.set_trace()
                else:
                    gate = torch.sigmoid(gate_logit)  # [0~1] prob

                input_per_task = encoder_output * gate[None, None, :]
            else:
                input_per_task = encoder_output

            output_per_task = self.task_layers[task_id](input_per_task)
            output_per_task_lst.append(output_per_task)
        output_per_task_lst = torch.cat(output_per_task_lst, dim=-1)  # [b,l,h+]
        return output_per_task_lst

    def task_layer_forward2(self, encoder_output, use_task_embed=True, use_gumbel_softmax=True, gumbel_tasks=None):
        output_per_task_lst = []  # list of [b,l,h~]
        for task_id in range(self.num_tasks):
            if use_task_embed:
                gate_logit = self.task_embed[task_id]  # [1024]
                # calc gate [emb_size]
                if use_gumbel_softmax:  # gate binary vector {0,1} with random sample. logits is distribution
                    if task_id in gumbel_tasks:  # when train specific task, which enable grad back propagation
                        gate = gumbel_sigmoid(gate_logit, hard=True, generator=self.gumbel_generator)  # random output
                        self.gate_tensor_lst[task_id] = gate
                        # gate = torch.ones_like(gate_logit)
                        # self.gate_tensor_lst[task_id] = gate_logit
                    else:
                        # 这种方式不会传梯度到gate_logit也就是task_embed去。
                        gate = (gate_logit >= 0.).float()  # deterministic output. when is not training, which unable grad back propagation
                        # if task_id == 0:
                        #     gate = (gate_logit > 0.).float()
                        # else:
                        #     gate = torch.ones_like(gate_logit)
                    # ipdb.set_trace()
                else:
                    gate = torch.sigmoid(gate_logit)  # [0~1] prob

                input_per_task = encoder_output * gate[None, None, :]
            else:
                input_per_task = encoder_output

            output_per_task = self.task_layers[task_id](input_per_task)
            output_per_task_lst.append(output_per_task)
        output_per_task_lst = torch.cat(output_per_task_lst, dim=-1)  # [b,l,h+]

        if self.use_slr:
            slr_output = self.slr_layer(encoder_output)  # [b,l,2h]
            link_start_hidden, link_end_hidden = torch.chunk(slr_output, 2, dim=-1)
            link_scores = calc_link_score(link_start_hidden, link_end_hidden)  # b,l-1
            pooling_type = 'softmin'
            logsumexp_temp = 0.3
            self.refined_scores = calc_refined_mat_tensor(link_scores, pooling_type=pooling_type, temp=logsumexp_temp)  # b,l,l,1

        return output_per_task_lst

    def task_layer_forward3(self, encoder_output, use_task_embed=True, use_gumbel_softmax=True, gumbel_tasks=None, ep=None):
        output_per_task_lst = []  # list of [b,l,h~]
        for task_id in range(self.num_tasks):
            if use_task_embed:
                gate_logit = self.task_embed[task_id]  # [1024]
                # calc gate [emb_size]
                if use_gumbel_softmax:  # gate binary vector {0,1} with random sample. logits is distribution
                    if task_id in gumbel_tasks:  # when train specific task, which enable grad back propagation
                        # if ep == 0:
                        #     gate = torch.ones_like(gate_logit)
                        #     self.gate_tensor_lst[task_id] = gate_logit
                        # elif ep == 1:
                        #     gate = gumbel_sigmoid(gate_logit, hard=False, use_gumbel=False, tau=1/3, generator=self.gumbel_generator)  # random output
                        #     self.gate_tensor_lst[task_id] = gate
                        # elif ep > 1:
                        #     gate = gumbel_sigmoid(gate_logit, hard=True, use_gumbel=True, tau=1/3, generator=self.gumbel_generator)  # random output
                        #     self.gate_tensor_lst[task_id] = gate
                        # if ep <= 6:
                        #     gate = gumbel_sigmoid(gate_logit, hard=True, generator=self.gumbel_generator)  # random output
                        #     self.gate_tensor_lst[task_id] = gate
                        # else:
                        #     gate = (gate_logit > 0.).float()
                        #     self.gate_tensor_lst[task_id] = gate
                        if ep is not None:
                            gate = (gate_logit >= 0.).float()
                            self.gate_tensor_lst[task_id] = gate

                    else:
                        # 这种方式不会传梯度到gate_logit也就是task_embed去。
                        gate = (gate_logit > 0.).float()  # deterministic output. when is not training, which unable grad back propagation
                    # ipdb.set_trace()
                else:
                    gate = torch.sigmoid(gate_logit)  # [0~1] prob

                input_per_task = encoder_output * gate[None, None, :]
            else:
                input_per_task = encoder_output

            output_per_task = self.task_layers[task_id](input_per_task)
            output_per_task_lst.append(output_per_task)
        output_per_task_lst = torch.cat(output_per_task_lst, dim=-1)  # [b,l,h+]
        return output_per_task_lst

    def span_matrix_forward(self, output, seq_len):
        bsz, length = output.shape[:2]
        # decode the output from output_per_task_lst = torch.cat(output_per_task_lst, dim=-1)
        ent_hid_lst = torch.split(output, self.output_dim_per_task_lst, dim=-1)
        start_hid_lst = [torch.chunk(t, 2, dim=-1)[0].reshape([bsz, length, self.hidden_size_per_ent, -1]) for t in ent_hid_lst]  # list of [b,l,hid,ent](for one task)
        end_hid_lst = [torch.chunk(t, 2, dim=-1)[1].reshape([bsz, length, self.hidden_size_per_ent, -1]) for t in ent_hid_lst]  # list of [b,l,hid,ent](for one task)
        start_hidden = torch.cat(start_hid_lst, dim=-1)  # b,l,h,e
        end_hidden = torch.cat(end_hid_lst, dim=-1)  # # b,l,h,e
        start_hidden = start_hidden.permute(0, 3, 1, 2)  # b,e,l,h
        end_hidden = end_hidden.permute(0, 3, 1, 2)  # b,e,l,h

        total_ent_size = start_hidden.shape[1]

        attention_scores = torch.matmul(start_hidden, end_hidden.transpose(-1, -2))  # [bat,num_ent,len,hid] * [bat,num_ent,hid,len] = [bat,num_ent,len,len]
        attention_scores = attention_scores / math.sqrt(self.hidden_size_per_ent)
        span_ner_mat_tensor = attention_scores.permute(0, 2, 3, 1)  # b,l,l,e
        if self.use_slr:
            span_ner_mat_tensor = span_ner_mat_tensor + self.refined_scores

        self.batch_span_tensor = span_ner_mat_tensor
        # 构造下三角mask 去除了pad和下三角区域
        len_mask = sequence_mask(seq_len)  # b,l
        matrix_mask = torch.logical_and(torch.unsqueeze(len_mask, 1), torch.unsqueeze(len_mask, 2))  # b,l,l  # 小正方形mask pad为0
        score_mat_mask = torch.triu(matrix_mask, diagonal=0)  # b,l,l  # 下三角0 上三角和对角线1
        span_ner_pred_lst = torch.masked_select(span_ner_mat_tensor, score_mat_mask[..., None])  # 只取True或1组成列表
        span_ner_pred_lst = span_ner_pred_lst.view(-1, total_ent_size)  # [*,ent]
        return span_ner_pred_lst

    def compute_offsets(self, task_id, mode='train'):
        ofs_s, ofs_e = self.taskid2offset[task_id]
        if mode == 'train':
            pass
        if mode == 'test':
            ofs_s = 0  # 从累计最开始的task算起
        return int(ofs_s), int(ofs_e)

    def calc_loss(self, batch_span_pred, batch_span_tgt):
        # pred  batch_span_pred [num_spans,ent]
        # label batch_span_tgt  [num_spans,ent]
        span_loss = self.bce_loss_layer(batch_span_pred, batch_span_tgt)  # [*,ent] [*,ent](target已是onehot) -> [*,ent]
        # ipdb.set_trace()
        # span_loss = torch.sum(span_loss, -1)  # [*] over ent
        span_loss = torch.mean(span_loss, -1)  # [*] over ent
        # span_loss = torch.mean(span_loss)  # 这样loss是0.00x 太小优化不了
        span_loss = torch.sum(span_loss, -1)  # [] over num_spans
        # span_loss = torch.mean(span_loss, -1)  # [] over num_spans
        return span_loss  # []

    def calc_f1(self, batch_span_pred, batch_span_tgt):
        # pred batch_span_pred [*,ent], label batch_span_tgt [*,ent]
        batch_span_pred = (batch_span_pred > 0).int()  # [*,ent] logits before sigmoid
        # calc f1
        num_pred = torch.sum(batch_span_pred)
        num_gold = torch.sum(batch_span_tgt)
        tp = torch.sum(batch_span_tgt * batch_span_pred)
        f1 = torch.tensor(1.) if num_gold == num_pred == 0 else 2 * tp / (num_gold + num_pred + 1e-12)
        return f1.item(), (num_gold.item(), num_pred.item(), tp.item())

    def calc_kl_loss(self, batch_target_lst_distilled, batch_predict_lst_need_distill, task_id):
        ofs_s, ofs_e = self.compute_offsets(task_id)
        bsz = len(batch_target_lst_distilled)
        # kl_losses = []
        batch_kl_loss = 0.
        for bdx in range(bsz):  # 只蒸馏当前的样本 忽略记忆库的
            pred_need_distill = batch_predict_lst_need_distill[bdx][:, :ofs_s]
            kl_pred = torch.stack([pred_need_distill, -pred_need_distill], dim=-1)  # [num_spans, ent, 2]
            log_kl_pred = torch.nn.functional.logsigmoid(kl_pred)

            kl_tgt = batch_target_lst_distilled[bdx][:, :ofs_s]

            # kl_tgt = torch.stack([kl_tgt, 1. - kl_tgt], dim=-1)  # [num_spans, ent, 2]  # kl_tgt为prob
            # kl_tgt_logits = -torch.log(1 / (tgt_prob + 1e-8) - 1 + 1e-8)  # inverse of sigmoid
            # kl_loss = torch.nn.functional.kl_div(log_kl_pred, kl_tgt, reduction='none')  # [num_spans, ent, 2]

            kl_tgt = kl_tgt / 1.  # temp=1
            kl_tgt_logit = torch.stack([kl_tgt, -kl_tgt], dim=-1)  # [num_spans, ent, 2]  # kl_tgt为logits
            log_kl_tgt = torch.nn.functional.logsigmoid(kl_tgt_logit)
            kl_loss = torch.nn.functional.kl_div(log_kl_pred, log_kl_tgt, reduction='none', log_target=True)  # [num_spans, ent, 2]

            kl_loss = torch.sum(kl_loss, -1)  # kl definition
            kl_loss = torch.mean(kl_loss, -1)  # over ent
            kl_loss = torch.sum(kl_loss, -1)  # over spans
            # kl_loss = torch.mean(kl_loss, -1)  # over spans

            # kl_losses.append(kl_loss)
            # self.kl_loss = sum(kl_losses) / len(kl_losses)
            batch_kl_loss += kl_loss

        # return torch.abs(batch_kl_loss / bsz)
        return batch_kl_loss / bsz
        # return torch.relu(batch_kl_loss / bsz)

    def calc_kl_loss_mse(self, batch_target_lst_distilled, batch_predict_lst_need_distill, task_id):
        ofs_s, ofs_e = self.compute_offsets(task_id)
        bsz = len(batch_target_lst_distilled)
        # kl_losses = []
        batch_kl_loss = 0.
        for bdx in range(bsz):  # 只蒸馏当前的样本 忽略记忆库的
            kl_tgt = batch_target_lst_distilled[bdx][:, :ofs_s]  # prob

            pred_need_distill = batch_predict_lst_need_distill[bdx][:, :ofs_s]
            mse_loss = self.mse_loss_layer(pred_need_distill.sigmoid(), kl_tgt)  # [num_spans, ent]

            mse_loss = torch.mean(mse_loss, -1)  # over ent
            mse_loss = torch.sum(mse_loss, -1)  # over spans
            # mse_loss = torch.mean(mse_loss, -1)  # over spans

            # kl_losses.append(kl_loss)
            # self.kl_loss = sum(kl_losses) / len(kl_losses)
            batch_kl_loss += mse_loss

        return batch_kl_loss / bsz

    def take_loss(self, task_id, batch_predict, batch_target, f1_meaner=None, bsz=None):
        """single task of example"""  # batch_predict,batch_target: [bsz*num_spans, ent]
        ofs_s, ofs_e = self.compute_offsets(task_id)
        loss = self.calc_loss(batch_predict[:, ofs_s:ofs_e], batch_target[:, ofs_s:ofs_e])  # 只计算对应task的头
        if f1_meaner is not None:
            f1, f1_detail = self.calc_f1(batch_predict[:, ofs_s:ofs_e], batch_target[:, ofs_s:ofs_e])
            f1_meaner.add(*f1_detail)
        if bsz is not None:
            loss = loss / bsz
        return loss

    def take_multitask_loss(self, batch_task_id, batch_predict_lst, batch_target_lst, f1_meaner=None):
        """multiple task of example"""  # batch_predict_lst,batch_target_lst: list of [num_spans, ent]
        losses = []
        for bdx, task_id in enumerate(batch_task_id):  # 以样本为单位计算loss 该batch有多个任务
            ofs_s, ofs_e = self.compute_offsets(task_id)
            span_loss = self.calc_loss(batch_predict_lst[bdx][:, ofs_s:ofs_e], batch_target_lst[bdx][:, ofs_s:ofs_e])  # 只计算对应task的头
            f1, f1_detail = self.calc_f1(batch_predict_lst[bdx][:, ofs_s:ofs_e], batch_target_lst[bdx][:, ofs_s:ofs_e])
            losses.append(span_loss)
            if f1_meaner is not None:
                f1_meaner.add(*f1_detail)
        loss = sum(losses) / len(losses)
        return loss

    def eval_forward(self, inputs_dct, task_id, mode='train'):
        # 用于eval
        seq_len = inputs_dct['ori_seq_len']
        encoder_output = self.encoder_forward(inputs_dct)
        task_layer_output = self.task_layer_forward2(encoder_output,
                                                     use_task_embed=self.args.use_task_embed, use_gumbel_softmax=self.args.use_gumbel_softmax,
                                                     gumbel_tasks=[])
        batch_span_pred = self.span_matrix_forward(task_layer_output, seq_len)  # [bsz*num_spans, ent]
        batch_predict_lst = torch.split(batch_span_pred, (seq_len * (seq_len + 1) / 2).int().tolist())  # 根据每个batch中的样本拆开 list of [num_spans, ent]

        ofs_s, ofs_e = self.compute_offsets(task_id, mode=mode)  # make sure we predict classes within the current task

        f1, detail_f1, span_loss, kl_loss = None, None, None, None
        if 'batch_span_tgt' in inputs_dct:  # if label had passed into
            batch_span_tgt = inputs_dct['batch_span_tgt']  # [bsz*num_spans, ent]
            f1, detail_f1 = self.calc_f1(batch_span_pred[:, ofs_s:ofs_e], batch_span_tgt[:, ofs_s:ofs_e])

            span_loss = self.take_loss(task_id, batch_span_pred, batch_span_tgt, bsz=len(seq_len))  # 默认是curr(train)
        if self.args.use_distill and task_id > 0 and inputs_dct.get('batch_span_tgt_lst_distilled', None):
            kl_loss = self.calc_kl_loss(inputs_dct['batch_span_tgt_lst_distilled'], batch_predict_lst, task_id)  # 默认是so_far-curr

        return batch_span_pred[:, ofs_s:ofs_e], f1, detail_f1, span_loss, kl_loss

    def forward(self, *args, **kwargs):
        return self.eval_forward(*args, **kwargs)

    def observe(self, inputs_dct, task_id, f1_meaner, ep=None):
        batch_exm = inputs_dct['batch_ner_exm']
        batch_length = inputs_dct['ori_seq_len']
        batch_input_pts = inputs_dct['batch_input_pts']

        batch_target = inputs_dct['batch_span_tgt']
        batch_target_lst = inputs_dct['batch_span_tgt_lst']  # list of onehot tensor [num_spans, ent]
        batch_target_lst_distilled = inputs_dct.get('batch_span_tgt_lst_distilled', None)  # list of sigmoided prob [num_spans, ent]
        bsz = batch_length.shape[0]

        self.total_loss = 0.
        self.span_loss = 0.
        self.sparse_loss = 0.
        self.kl_loss = 0.
        self.entropy_loss = 0.

        encoder_output = self.encoder_forward(inputs_dct)

        if ep is not None:
            task_layer_output = self.task_layer_forward3(encoder_output,
                                                         use_task_embed=self.args.use_task_embed, use_gumbel_softmax=self.args.use_gumbel_softmax,
                                                         gumbel_tasks=[task_id],
                                                         ep=ep)
        else:
            task_layer_output = self.task_layer_forward2(encoder_output,
                                                         use_task_embed=self.args.use_task_embed, use_gumbel_softmax=self.args.use_gumbel_softmax,
                                                         gumbel_tasks=[task_id],
                                                         )

        batch_predict = self.span_matrix_forward(task_layer_output, batch_length)  # [bsz*num_spans, ent]

        batch_predict_lst = torch.split(batch_predict, (batch_length * (batch_length + 1) / 2).int().tolist())  # 根据每个batch中的样本拆开 list of [num_spans, ent]

        # self.span_loss1 = self.take_multitask_loss([task_id] * bsz, batch_predict_lst, batch_target_lst, f1_meaner)
        self.span_loss = self.take_loss(task_id, batch_predict, batch_target, f1_meaner=f1_meaner, bsz=bsz)  # loss按batch平均才能跟kl对齐
        self.total_loss += self.span_loss
        # self.total_loss += self.span_loss / (self.span_loss.detach() + 1e-7)

        if self.args.use_distill and task_id > 0:
            batch_predict_lst_need_distill = batch_predict_lst
            self.kl_loss = self.calc_kl_loss(batch_target_lst_distilled, batch_predict_lst_need_distill, task_id)

            self.total_loss += self.kl_loss
            # self.total_loss += self.kl_loss / (self.kl_loss.detach() + 1e-7)

            # kl_coe = self.span_loss.detach() / (self.kl_loss.detach() + 1e-7)
            # self.kl_loss *= kl_coe
            # self.total_loss += self.kl_loss

        if self.args.use_task_embed:
            if self.args.use_gumbel_softmax:
                # curr_task_gate = self.gate_tensor_lst[task_id]  # [emb_dim] 0-1 binary
                curr_task_gate = torch.sigmoid(self.task_embed[task_id])  # [emb_dim]  0~1 prob
            else:
                curr_task_gate = torch.sigmoid(self.task_embed[task_id])  # [emb_dim]  0~1 prob  # 这样没有sparse_loss
            # print(curr_task_gate.sum(-1))
            # 这里norm_loss 如果值为1则有梯度1/1024, 如果值为0则有梯度为0,相当于不优化为0的门。即prob=0.49的门有可能越过成0.51.
            self.sparse_loss = torch.norm(curr_task_gate, p=1, dim=-1) / curr_task_gate.shape[0]  # L1范数 need / 1024dim
            # print('\ncurr_task_gate', curr_task_gate.sum(-1))
            # print('sparse_loss', self.sparse_loss)
            # task_embed_binary = (self.task_embed[task_id] > 0).float()
            # print('task_embed', task_embed_binary.sum(-1))
            # print('sparse_loss\n', torch.norm(task_embed_binary, p=1, dim=-1) / task_embed_binary.shape[0])

            # self.sparse_loss *= 0.5
            # if task_id > 0:
            #     self.sparse_loss *= 2
            # self.total_loss += self.sparse_loss
            # self.total_loss += self.sparse_loss / (self.sparse_loss.detach() + 1e-7)

            sparse_coe = self.total_loss.detach() / (self.sparse_loss.detach())
            # self.sparse_loss *= sparse_coe
            # self.total_loss += self.sparse_loss * sparse_coe
            # if self.ep and self.ep > 0:
            #     self.total_loss += self.sparse_loss * sparse_coe * 0.5
            self.total_loss += self.sparse_loss * sparse_coe * 0.5

            # self-entropy loss
            # task_prob = torch.sigmoid(self.task_embed[task_id])
            # entropy_loss = - task_prob * torch.log(task_prob) - (1. - task_prob) * torch.log(1. - task_prob)
            # self.entropy_loss = torch.mean(entropy_loss, -1)
            # entropy_coe = self.total_loss.detach() / (self.entropy_loss.detach())

            # self.total_loss += self.entropy_loss * entropy_coe * 0.5

            # 0602
            # self.sparse_loss *= self.total_loss.detach() / (self.sparse_loss.detach() + 1e-7)
            # self.total_loss += self.sparse_loss

        # if self.args.use_task_embed:
        #     if self.args.use_gumbel_softmax:
        #         s_l_lst = []
        #         for tid in range(task_id+1):
        #             curr_task_gate = self.gate_tensor_lst[tid]  # [emb_dim] 0-1 binary
        #             _sparse_loss = torch.norm(curr_task_gate, p=1, dim=0) / curr_task_gate.shape[0]  # L1范数 need / 1024dim
        #             s_l_lst.append(_sparse_loss)
        #         self.sparse_loss = sum(s_l_lst) / len(s_l_lst)
        #     else:
        #         curr_task_gate = torch.sigmoid(self.task_embed[task_id])  # [emb_dim]  0~1 prob
        #     self.total_loss += self.sparse_loss / (self.sparse_loss.detach() + 1e-7)

        self.opt.zero_grad()
        self.total_loss.backward()
        # if task_id > 0:
        #     ipdb.set_trace()

        # ori_task_embed = copy.deepcopy(self.net.task_embed)
        # ipdb.set_trace()
        if self.grad_clip is None:
            self.total_norm = 0
        else:
            self.total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        self.opt.step()
        # if ep is not None:
        #     if ep <= 6:
        #         self.lrs.step()
        #     else:
        #         pass
        if self.use_schedual:
            self.lrs.step()
        # self.curr_lr = self.lrs.get_last_lr()[0]
        self.curr_lr = self.opt.param_groups[0]['lr']

        return (float(self.total_loss),
                float(self.span_loss),
                float(self.sparse_loss),
                float(self.kl_loss),
                )

    def take_alltask_loss(self, batch_predict, batch_target, f1_meaner=None, bsz=None):
        """single task of example"""  # batch_predict,batch_target: [bsz*num_spans, ent]
        loss = self.calc_loss(batch_predict, batch_target)  # 只计算对应task的头
        if f1_meaner is not None:
            f1, f1_detail = self.calc_f1(batch_predict, batch_target)
            f1_meaner.add(*f1_detail)
        if bsz is not None:
            loss = loss / bsz
        return loss

    def observe_all(self, inputs_dct, f1_meaner, ep=None):
        batch_exm = inputs_dct['batch_ner_exm']
        batch_length = inputs_dct['ori_seq_len']
        batch_input_pts = inputs_dct['batch_input_pts']

        batch_target = inputs_dct['batch_span_tgt']
        batch_target_lst = inputs_dct['batch_span_tgt_lst']  # list of onehot tensor [num_spans, ent]
        batch_target_lst_distilled = inputs_dct.get('batch_span_tgt_lst_distilled', None)  # list of sigmoided prob [num_spans, ent]
        bsz = batch_length.shape[0]

        self.opt.zero_grad()

        self.total_loss = 0.
        self.span_loss = 0.
        self.sparse_loss = 0.
        self.kl_loss = 0.
        self.entropy_loss = 0.

        encoder_output = self.encoder_forward(inputs_dct)
        task_layer_output = self.task_layer_forward2(encoder_output,
                                                     use_task_embed=self.args.use_task_embed, use_gumbel_softmax=self.args.use_gumbel_softmax,
                                                     gumbel_tasks=list(range(self.num_tasks)),
                                                     )

        batch_predict = self.span_matrix_forward(task_layer_output, batch_length)  # [bsz*num_spans, ent]

        batch_predict_lst = torch.split(batch_predict, (batch_length * (batch_length + 1) / 2).int().tolist())  # 根据每个batch中的样本拆开 list of [num_spans, ent]

        self.span_loss = self.take_alltask_loss(batch_predict, batch_target, f1_meaner=f1_meaner, bsz=bsz)  # 这样loss不能按batch平均
        self.total_loss += self.span_loss
        # self.total_loss += self.span_loss / (self.span_loss.detach() + 1e-7)

        if self.args.use_task_embed:
            for task_id in range(self.num_tasks):
                if self.args.use_gumbel_softmax:
                    curr_task_gate = self.gate_tensor_lst[task_id]  # [emb_dim] 0-1 binary
                else:
                    curr_task_gate = torch.sigmoid(self.task_embed[task_id])  # [emb_dim]  0~1 prob  # 这样没有sparse_loss
                sparse_loss = torch.norm(curr_task_gate, p=1, dim=-1) / curr_task_gate.shape[0]  # L1范数 need / 1024dim
                self.sparse_loss += sparse_loss
            self.sparse_loss /= self.num_tasks
            self.total_loss += self.sparse_loss
            # self.total_loss += self.sparse_loss / (self.sparse_loss.detach() + 1e-7)

        self.opt.zero_grad()
        self.total_loss.backward()

        if self.grad_clip is None:
            self.total_norm = 0
        else:
            self.total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        self.opt.step()
        if self.use_schedual:
            self.lrs.step()
        # self.curr_lr = self.lrs.get_last_lr()[0]
        self.curr_lr = self.opt.param_groups[0]['lr']

        return (float(self.total_loss),
                float(self.span_loss),
                float(self.sparse_loss),
                float(self.kl_loss),
                )

    def take_so_far_task_loss(self, task_id, batch_predict, batch_target, f1_meaner=None, bsz=None):
        """single task of example"""  # batch_predict,batch_target: [bsz*num_spans, ent]
        ofs_s, ofs_e = self.compute_offsets(task_id, mode='test')
        loss = self.calc_loss(batch_predict[:, ofs_s:ofs_e], batch_target[:, ofs_s:ofs_e])  # 只计算对应task的头
        if f1_meaner is not None:
            f1, f1_detail = self.calc_f1(batch_predict[:, ofs_s:ofs_e], batch_target[:, ofs_s:ofs_e])
            f1_meaner.add(*f1_detail)
        if bsz is not None:
            loss = loss / bsz
        return loss

    def observe_non_cl(self, inputs_dct, task_id, f1_meaner):
        batch_exm = inputs_dct['batch_ner_exm']
        batch_length = inputs_dct['ori_seq_len']
        batch_input_pts = inputs_dct['batch_input_pts']

        batch_target = inputs_dct['batch_span_tgt']
        batch_target_lst = inputs_dct['batch_span_tgt_lst']  # list of onehot tensor [num_spans, ent]
        batch_target_lst_distilled = inputs_dct.get('batch_span_tgt_lst_distilled', None)  # list of sigmoided prob [num_spans, ent]
        bsz = batch_length.shape[0]

        self.opt.zero_grad()

        self.total_loss = 0.
        self.span_loss = 0.
        self.sparse_loss = 0.
        self.kl_loss = 0.
        self.entropy_loss = 0.

        encoder_output = self.encoder_forward(inputs_dct)
        task_layer_output = self.task_layer_forward2(encoder_output,
                                                     use_task_embed=self.args.use_task_embed, use_gumbel_softmax=self.args.use_gumbel_softmax,
                                                     gumbel_tasks=list(range(self.num_tasks)),
                                                     )

        batch_predict = self.span_matrix_forward(task_layer_output, batch_length)  # [bsz*num_spans, ent]

        batch_predict_lst = torch.split(batch_predict, (batch_length * (batch_length + 1) / 2).int().tolist())  # 根据每个batch中的样本拆开 list of [num_spans, ent]

        self.span_loss = self.take_so_far_task_loss(task_id, batch_predict, batch_target, f1_meaner=f1_meaner, bsz=bsz)  # 这样loss不能按batch平均
        self.total_loss += self.span_loss
        # self.total_loss += self.span_loss / (self.span_loss.detach() + 1e-7)

        if self.args.use_task_embed:
            for task_id in range(self.num_tasks):
                if self.args.use_gumbel_softmax:
                    curr_task_gate = self.gate_tensor_lst[task_id]  # [emb_dim] 0-1 binary
                else:
                    curr_task_gate = torch.sigmoid(self.task_embed[task_id])  # [emb_dim]  0~1 prob  # 这样没有sparse_loss
                sparse_loss = torch.norm(curr_task_gate, p=1, dim=-1) / curr_task_gate.shape[0]  # L1范数 need / 1024dim
                self.sparse_loss += sparse_loss
            self.sparse_loss /= self.num_tasks
            self.total_loss += self.sparse_loss
            # self.total_loss += self.sparse_loss / (self.sparse_loss.detach() + 1e-7)

        self.opt.zero_grad()
        self.total_loss.backward()

        if self.grad_clip is None:
            self.total_norm = 0
        else:
            self.total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        self.opt.step()
        if self.use_schedual:
            self.lrs.step()
        # self.curr_lr = self.lrs.get_last_lr()[0]
        self.curr_lr = self.opt.param_groups[0]['lr']

        return (float(self.total_loss),
                float(self.span_loss),
                float(self.sparse_loss),
                float(self.kl_loss),
                )


def get_BIO_transitions_mask(tag2id, id2tag):
    num_tags = len(tag2id)
    all_tag_lst = list(tag2id.keys())
    trans_mask = np.ones([num_tags, num_tags], dtype=np.int)  # 默认全部要mask(不通=1) 不mask则为0
    O_tag_id = tag2id['O']
    trans_mask[O_tag_id, :] = 0  # O能转为任何
    trans_mask[:, O_tag_id] = 0  # 任何都能转为O
    for prev_tag_id in range(num_tags):
        prev_tag = id2tag[prev_tag_id]
        if prev_tag.startswith('B-'):  # B- 只能跟自己的I 和 任何其他B- # TODO 允不允许B1B1I1？
            prev_tag_name = prev_tag.replace('B-', '')
            trans_mask[prev_tag_id][tag2id[f'I-{prev_tag_name}']] = 0
            for rear_tag in all_tag_lst:
                # if rear_tag.startswith('B-') and rear_tag != prev_tag:  # 不允许B1B1I1
                if rear_tag.startswith('B-'):  # 允许B1B1I1
                    trans_mask[prev_tag_id][tag2id[rear_tag]] = 0
        if prev_tag.startswith('I-'):  # 只能跟自己的I 和任何其他B-
            prev_tag_name = prev_tag.replace('I-', '')
            trans_mask[prev_tag_id][tag2id[f'I-{prev_tag_name}']] = 0
            for rear_tag in all_tag_lst:
                if rear_tag.startswith('B-'):
                    trans_mask[prev_tag_id][tag2id[rear_tag]] = 0
    return trans_mask


def viterbi_decode(emissions: torch.FloatTensor,
                   mask: torch.ByteTensor,
                   start_transitions,
                   end_transitions,
                   transitions) -> List[List[int]]:
    # emissions: (batch_size,seq_length,num_tags)
    # mask: (batch_size,seq_length)
    # start_transitions end_transitions [num_tags]
    # transitions [num_tags, num_tags]

    emissions = emissions.transpose(0, 1)
    mask = mask.transpose(0, 1)

    assert emissions.dim() == 3 and mask.dim() == 2
    assert emissions.shape[:2] == mask.shape
    assert emissions.size(2) == start_transitions.size(0) == end_transitions.size(0) == transitions.size(0)
    assert mask[0].all()

    seq_length, batch_size = mask.shape

    # Start transition and first emission
    # shape: (batch_size, num_tags)
    score = start_transitions + emissions[0]
    history = []

    # score is a tensor of size (batch_size, num_tags) where for every batch,
    # value at column j stores the score of the best tag sequence so far that ends
    # with tag j
    # history saves where the best tags candidate transitioned from; this is used
    # when we trace back the best tag sequence

    # Viterbi algorithm recursive case: we compute the score of the best tag sequence
    # for every possible next tag
    for i in range(1, seq_length):
        # Broadcast viterbi score for every possible next tag
        # shape: (batch_size, num_tags, 1)
        broadcast_score = score.unsqueeze(2)

        # Broadcast emission score for every possible current tag
        # shape: (batch_size, 1, num_tags)
        broadcast_emission = emissions[i].unsqueeze(1)

        # Compute the score tensor of size (batch_size, num_tags, num_tags) where
        # for each sample, entry at row i and column j stores the score of the best
        # tag sequence so far that ends with transitioning from tag i to tag j and emitting
        # shape: (batch_size, num_tags, num_tags)
        next_score = broadcast_score + start_transitions + broadcast_emission

        # Find the maximum score over all possible current tag
        # shape: (batch_size, num_tags)
        next_score, indices = next_score.max(dim=1)

        # Set score to the next score if this timestep is valid (mask == 1)
        # and save the index that produces the next score
        # shape: (batch_size, num_tags)
        score = torch.where(mask[i].unsqueeze(1), next_score, score)
        history.append(indices)

    # End transition score
    # shape: (batch_size, num_tags)
    score += end_transitions

    # Now, compute the best path for each sample

    # shape: (batch_size,)
    seq_ends = mask.long().sum(dim=0) - 1
    best_tags_list = []

    for idx in range(batch_size):
        # Find the tag which maximizes the score at the last timestep; this is our best tag
        # for the last timestep
        _, best_last_tag = score[idx].max(dim=0)
        best_tags = [best_last_tag.item()]

        # We trace back where the best last tag comes from, append that to our best tag
        # sequence, and trace it back again, and so on
        for hist in reversed(history[:seq_ends[idx]]):
            best_last_tag = hist[idx][best_tags[-1]]
            best_tags.append(best_last_tag.item())

        # Reverse the order because we start from the last timestep
        best_tags.reverse()
        best_tags_list.append(best_tags)

    return best_tags_list

# Below functions are for SLR (another reseach), not used in this published paper.
def calc_link_score(link_start_hidden, link_end_hidden, fast_impl=True):
    # link_start_hidden [b,l,h]
    # link_end_hidden [b,l,h]
    # return link_dot_prod_scores [b,l-1]
    hidden_size = link_start_hidden.shape[-1]

    if fast_impl:
        # link score 快速计算方式 直接移位相乘再相加(点积)
        link_dot_prod_scores = link_start_hidden[:, :-1, :] * link_end_hidden[:, 1:, :]  # b,l-1,h
        link_dot_prod_scores = torch.sum(link_dot_prod_scores, dim=-1)  # b,l-1
        link_dot_prod_scores = link_dot_prod_scores / hidden_size ** 0.5  # b,l-1
    else:
        # link score 普通计算方式 通过计算矩阵后取对角线 有大量非对角线的无用计算
        link_dot_prod_scores = torch.matmul(link_start_hidden, link_end_hidden.transpose(-1, -2))  # b,l,l
        link_dot_prod_scores = link_dot_prod_scores / hidden_size ** 0.5  # b,e,l,l
        link_dot_prod_scores = torch.diagonal(link_dot_prod_scores, offset=1, dim1=-2, dim2=-1)  # b,l-1

    return link_dot_prod_scores  # b,l-1
    # return torch.relu(link_dot_prod_scores)  # b,l-1


def calc_refined_mat_tensor(link_scores, pooling_type, temp=1):
    # link_scores [b,l-1]
    # span_ner_mat_tensor [b,l,l,e]
    if pooling_type == 'softmin':
        mask_matrix = aggregate_mask_by_reduce(link_scores, mode='min', use_soft=True, temp=temp)[..., None]  # b,l-1,l-1,1
    elif pooling_type == 'min':
        mask_matrix = aggregate_mask_by_reduce(link_scores, mode='min', use_soft=False)[..., None]  # b,l-1,l-1,1
    elif pooling_type == 'softmax':
        mask_matrix = aggregate_mask_by_reduce(link_scores, mode='max', use_soft=True, temp=temp)[..., None]  # b,l-1,l-1,1
    elif pooling_type == 'max':
        mask_matrix = aggregate_mask_by_reduce(link_scores, mode='max', use_soft=False)[..., None]  # b,l-1,l-1,1
    elif pooling_type == 'mean':
        mask_matrix = aggregate_mask_by_cum(link_scores, mean=True)[..., None]  # b,l-1,l-1,1
    elif pooling_type == 'sum':
        mask_matrix = aggregate_mask_by_cum(link_scores, mean=False)[..., None]  # b,l-1,l-1,1
    else:
        raise NotImplementedError
    final_mask = torch.nn.functional.pad(mask_matrix, pad=(0, 0, 1, 0, 0, 1), mode="constant", value=0)  # b,l,l,1  # 长宽增加1对齐

    return final_mask


def aggregate_mask_by_reduce(tensor1, mode='max', use_soft=True, temp=1):
    """目前在用"""
    # tensor [batch,len]
    # e.g. [[1,2,3]]
    batch_size, length = tensor1.shape

    diag_t = torch.diag_embed(tensor1, offset=0)  # [b,l,l]
    """diag_t
    [1., 0., 0.]
    [0., 2., 0.]
    [0., 0., 3.]
    """

    cum_t = torch.cumsum(diag_t, dim=-1)  # [b,l,l]
    """cum_t
    [1., 1., 1.]
    [0., 2., 2.]
    [0., 0., 3.]
    """

    cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
    """cum_t
    [0., 0., 3.]
    [0., 2., 2.]
    [1., 1., 1.]
    """

    if mode in ['max', 'min']:
        triu_mask = torch.triu(torch.ones([length, length]), diagonal=1).to(tensor1.device)[None, ...]  # 1,l,l
        """triu_mask
        [0., 1., 1.]
        [0., 0., 1.]
        [0., 0., 0.]
        """
        inv_triu_mask = torch.flip(triu_mask, dims=[-1])
        """inv_triu_mask
        [1., 1., 0.]
        [1., 0., 0.]
        [0., 0., 0.]
        """
        if mode == 'max':
            inv_triu_mask = inv_triu_mask * -1e12
            cum_t = cum_t + inv_triu_mask
            """cum_t
            [-inf., -inf., 3.]
            [-inf., 2., 2.]
            [1., 1., 1.]
            """

            if use_soft:
                cum_t = torch.logcumsumexp(cum_t / temp, dim=-2) * temp  # [b,l,l]
            else:
                cum_t, _ = torch.cummax(cum_t, dim=-2)  # [b,l,l]
            """cum_t 0: denote -inf
            [max0., max0., max3.]
            [max0+0., max2+0., max2+3.]
            [max1+0+0., max1+2+0., max1+2+3.]
            """

            cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
            """cum_t
            [max1., max1+2., max1+2+3.]
            [max0., max2., max2+3.]
            [max0., max0., max3.]
            """

            cum_t = torch.triu(cum_t, diagonal=0)  # [b,l,l]
            """cum_t
            [max1., max1+2., max1+2+3.]
            [0., max2., max2+3.]
            [0., 0., max3.]
            """


        elif mode == 'min':
            inv_triu_mask = inv_triu_mask * 1e12
            cum_t = cum_t + inv_triu_mask
            """cum_t
            [inf., inf., 3.]
            [inf., 2., 2.]
            [1., 1., 1.]
            """

            if use_soft:
                cum_t = torch.logcumsumexp(-cum_t / temp, dim=-2) * temp  # [b,l,l]
                cum_t = - cum_t
            else:
                cum_t, _ = torch.cummin(cum_t, dim=-2)  # [b,l,l]
            """cum_t 0: denote inf
            [min0., min0., min3.]
            [min0+0., min2+0., min2+3.]
            [min1+0+0., min1+2+0., min1+2+3.]
            """

            cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
            """cum_t
            [min1., min1+2., min1+2+3.]
            [min0., min2., min2+3.]
            [min0., min0., min3.]
            """

            cum_t = torch.triu(cum_t, diagonal=0)  # [b,l,l]
            """cum_t
            [min1., min1+2., min1+2+3.]
            [0., min2., min2+3.]
            [0., 0., min3.]
            """

    return cum_t


def aggregate_mask_by_cum(tensor1, mean=True):
    # tensor [batch,len]
    # e.g. [[1,2,3]]
    batch_size, length = tensor1.shape

    # diag_mask = torch.diag_embed(torch.ones([length]), offset=0)  # [l,l]
    # diag_mask = diag_mask[None, ..., None]  # [1,l,l,1]
    # torch.diag_embed(tensor1, )

    diag_t = torch.diag_embed(tensor1, offset=0)  # [b,l,l]
    """diag_t
    [1., 0., 0.]
    [0., 2., 0.]
    [0., 0., 3.]
    """

    cum_t = torch.cumsum(diag_t, dim=-1)  # [b,l,l]
    """cum_t
    [1., 1., 1.]
    [0., 2., 2.]
    [0., 0., 3.]
    """

    cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
    """cum_t
    [0., 0., 3.]
    [0., 2., 2.]
    [1., 1., 1.]
    """

    cum_t = torch.cumsum(cum_t, dim=-2)  # [b,l,l]
    """cum_t
    [0., 0., 3.]
    [0., 2., 2+3.]
    [1., 1+2., 1+2+3.]
    """

    cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
    """cum_t
    [1., 1+2., 1+2+3.]
    [0., 2., 2+3.]
    [0., 0., 3.]
    """
    sum_t = cum_t

    """构造相关mask矩阵"""
    ones_matrix = torch.ones(length, length).to(tensor1.device)
    triu_mask = torch.triu(ones_matrix, 0)[None, ...]  # 1,l,l  # 上三角包括对角线为1 其余为0
    ignore_mask = 1. - triu_mask

    if mean:
        # 求平均逻辑
        # 分母： 要除以来求平均
        # e.g. length=3
        heng = torch.arange(1, length + 1).to(tensor1.device)  # [1,2,3]
        heng = heng.unsqueeze(0).repeat((batch_size, 1))  # b,l
        heng = heng.unsqueeze(1).repeat((1, length, 1))  # b,l,l
        """
        [1,2,3]
        [1,2,3]
        [1,2,3]
        """
        shu = torch.arange(0, length).to(tensor1.device)  # [0,1,2]
        shu = shu.unsqueeze(0).repeat((batch_size, 1))  # b,l
        shu = shu.unsqueeze(1).repeat((1, length, 1))  # b,l,l
        shu = shu.transpose(1, 2)
        shu = - shu
        """
        [-0,-0,-0]
        [-1,-1,-1]
        [-2,-2,-2]
        """
        count = heng + shu  # 这里一开始竟然用了- --得正 日
        """
        [1,2,3]
        [0,1,2]
        [-1,0,1]  # 下三角会被mask掉不用管  Note:但是除以不能为0！
        """

        # 把下三角强制变为1 避免计算溢常 因为后面会mask 没关系
        count = count * triu_mask + ignore_mask

        sum_t = sum_t / count

    # 再把下三角强制变为0
    sum_t = sum_t * triu_mask
    return sum_t
