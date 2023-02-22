import sys, random, os
from typing import *
import ipdb
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
import numpy as np

import datautils as utils
from datautils import NerExample
from data_reader import NerDataReader


class SubsetSequentialSampler(Sampler[int]):
    r"""Samples elements `sequentially` from a given list of indices, without replacement.
    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


# curr_dir = os.path.dirname(__file__)
# parent_dir = os.path.dirname(curr_dir)
data_dir = 'data/'


def analyse_task_ent_dist():
    exm_file = data_dir + 'onto/train_task.jsonl'
    exm_lst = NerExample.load_from_jsonl(exm_file, token_deli=' ',
                                         external_attrs=['task_id', 'bert_tok_char_lst', 'ori_2_tok'])
    task_exms_dct = {}
    for exm in exm_lst:
        task_id = str(exm.task_id)
        task_exms_dct.setdefault(task_id, []).append(exm)
    for task_id in sorted(task_exms_dct):
        print(f'===task: {task_id}===')
        NerExample.stats(task_exms_dct[task_id])


def sort_by_idx(lst, ids=None):
    if ids is None:
        return lst
    sorted_lst = []
    for idx in ids:
        sorted_lst.append(lst[idx])
    return sorted_lst


class CL_Loader:
    def __init__(self, entity_task_lst,
                 split_seed=1,
                 max_len=512,
                 bert_model_dir='huggingface_model_resource/bert-base-cased'):
        self.bert_model_dir = bert_model_dir
        self.split_seed = split_seed
        self.max_len = max_len
        self.entity_task_lst = entity_task_lst
        self.num_tasks = len(self.entity_task_lst)
        self.tid2ents = {tid: ents for tid, ents in enumerate(self.entity_task_lst)}
        self.num_ents_per_task = [len(ents) for ents in self.entity_task_lst]
        self.ent2tid = {}
        for tid, ents in self.tid2ents.items():
            for ent in ents:
                self.ent2tid[ent] = tid

        self.entity_lst = sum(self.entity_task_lst, [])
        self.datareader = NerDataReader(self.bert_model_dir, self.max_len, ent_file_or_ent_lst=self.entity_lst)

        self.ent2id = self.datareader.ent2id
        self.tid2entids = {tid: [self.ent2id[ent] for ent in ents] for tid, ents in self.tid2ents.items()}
        self.tid2offset = {tid: [min(entids), max(entids) + 1] for tid, entids in self.tid2entids.items()}
        print('tid2offset', self.tid2offset)
        print('id2ent', self.datareader.id2ent)
        print('tid2entids', self.tid2entids)

    def init_data(self, datafiles=None, setup=None, bsz=14, test_bsz=64, arch='span', use_pt=False, gpu=True, quick_test=False):
        self.task_train_generator = torch.Generator()  # make sure no affect by model_init (teacher model) i.e. non_cl_task0 = cl_task0
        # to make sure it's the same e.g. train single task 6 above task 5 = train task 1-6  # 要先蒸馏消耗g 再训练也消耗g
        # self.task_train_generators = [torch.Generator() for _ in range(self.num_tasks)]
        # [g.manual_seed(i) for i, g in enumerate(self.task_train_generators)]
        # 当时出现的问题是 每轮测试还是最后一轮测试竟然影响模型的随机性，因为只要dataloader被迭代一次都会消耗一次全局种子的随机次数。

        # setup: split or filter
        self.bsz = bsz
        self.test_bsz = test_bsz
        self.arch = arch
        self.gpu = gpu
        if setup is None:
            setup = self.setup
        else:
            self.setup = setup
        if datafiles is None:
            datafiles = self.datafiles

        if not quick_test:
            """train"""
            self.train_exm_lst, self.train_tid2exmids = self.load_data_with_taskid(data_dir + datafiles['train'],
                                                                                   setup=setup, use_pt=use_pt)

        """dev"""
        self.dev_exm_lst, self.dev_tid2exmids = self.load_data_with_taskid(data_dir + datafiles['dev'],
                                                                           setup=setup, use_pt=use_pt)
        if quick_test:
            self.train_exm_lst = self.dev_exm_lst
            self.train_tid2exmids = self.dev_tid2exmids

        """test"""  # for Test Filter
        self.test_exm_lst, self.test_tid2exmids = self.load_data_with_taskid(data_dir + datafiles['test'],
                                                                             setup='filter', use_pt=use_pt)

        self.num_train = len(self.train_exm_lst)
        self.num_dev = len(self.dev_exm_lst)
        self.num_test = len(self.test_exm_lst)

        self.train_dataset = self.datareader.build_dataset(self.train_exm_lst, arch=self.arch, loss_type='sigmoid')
        self.dev_dataset = self.datareader.build_dataset(self.dev_exm_lst, arch=self.arch, loss_type='sigmoid')
        self.test_dataset = self.datareader.build_dataset(self.test_exm_lst, arch=self.arch, loss_type='sigmoid')
        # fewnerd self.test_dateset - 1 because 1 of test_exm_lst have max_len>510
        self.init_dataloaders()

    def init_dataloaders(self):
        """ initialize dataloaders for CL"""
        setup = self.setup
        gpu = self.gpu
        self.train_tasks_dataloaders = []  # CL Train Split or Filter
        for tid in range(self.num_tasks):
            exmids = sorted(self.train_tid2exmids[tid])
            print(f'task_id {tid} have {len(exmids)} train examples')
            dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                     batch_size=self.bsz,
                                                     sampler=SubsetRandomSampler(exmids,
                                                                                 generator=self.task_train_generator
                                                                                 ),
                                                     collate_fn=self.datareader.get_batcher_fn(gpu=gpu, arch=self.arch),
                                                     )
            self.train_tasks_dataloaders.append(dataloader)

        self.dev_tasks_dataloaders = []  # CL Dev Split or Filter
        for tid in range(self.num_tasks):
            exmids = sorted(self.dev_tid2exmids[tid])
            print(f'task_id {tid} have {len(exmids)} dev examples')
            dataloader = torch.utils.data.DataLoader(self.dev_dataset,
                                                     batch_size=self.test_bsz,
                                                     sampler=SubsetSequentialSampler(exmids),
                                                     collate_fn=self.datareader.get_batcher_fn(gpu=gpu, arch=self.arch),
                                                     )
            self.dev_tasks_dataloaders.append(dataloader)  # CL

        self.test_tasks_dataloaders_filtered = []  # Test Filter
        for tid in range(self.num_tasks):
            so_far_exmids = set()
            for i in range(0, tid + 1):
                so_far_exmids.update(self.test_tid2exmids[i])
            so_far_exmids = sorted(so_far_exmids)
            print(f'task_id {tid} have {len(so_far_exmids)} test filtered examples')
            dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                     batch_size=self.test_bsz,
                                                     sampler=SubsetSequentialSampler(so_far_exmids),
                                                     collate_fn=self.datareader.get_batcher_fn(gpu=gpu, arch=self.arch),
                                                     )
            self.test_tasks_dataloaders_filtered.append(dataloader)

        print(f'total {len(self.test_dataset)} test examples:')
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,  # Test All
                                                           batch_size=self.test_bsz,
                                                           shuffle=False,
                                                           collate_fn=self.datareader.get_batcher_fn(gpu=gpu, arch=self.arch),
                                                           )

        # experimental all tasks
        self.train_alltask_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                                    batch_size=self.bsz,
                                                                    shuffle=True,
                                                                    collate_fn=self.datareader.get_batcher_fn(gpu=gpu, arch=self.arch),
                                                                    # generator=self.task_train_generators[0]
                                                                    )
        # experimental all tasks
        self.dev_alltask_dataloader = torch.utils.data.DataLoader(self.dev_dataset,
                                                                  batch_size=self.test_bsz,
                                                                  shuffle=False,
                                                                  collate_fn=self.datareader.get_batcher_fn(gpu=gpu, arch=self.arch),
                                                                  )

        # non_CL Train
        self.so_far_train_tasks_dataloaders = [self.train_tasks_dataloaders[0]]  # need the first NonCL one align with CL first one 要第一个相当于与CL的对齐
        for tid in range(1, self.num_tasks):
            so_far_exmids = set()
            for i in range(0, tid + 1):
                so_far_exmids.update(self.train_tid2exmids[i])
            so_far_exmids = sorted(so_far_exmids)
            print(f'non_cl task_id {tid} have {len(so_far_exmids)} train examples')
            dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                     batch_size=self.bsz,
                                                     sampler=SubsetRandomSampler(so_far_exmids,
                                                                                 generator=self.task_train_generator
                                                                                 ),
                                                     collate_fn=self.datareader.get_batcher_fn(gpu=gpu, arch=self.arch),
                                                     )
            self.so_far_train_tasks_dataloaders.append(dataloader)
        # non_CL Dev
        self.so_far_dev_tasks_dataloaders = [self.dev_tasks_dataloaders[0]]
        for tid in range(1, self.num_tasks):
            so_far_exmids = set()
            for i in range(0, tid + 1):
                so_far_exmids.update(self.dev_tid2exmids[i])
            so_far_exmids = sorted(so_far_exmids)
            print(f'non_cl task_id {tid} have {len(so_far_exmids)} dev examples')
            dataloader = torch.utils.data.DataLoader(self.dev_dataset,
                                                     batch_size=self.bsz,
                                                     sampler=SubsetSequentialSampler(so_far_exmids),
                                                     collate_fn=self.datareader.get_batcher_fn(gpu=gpu, arch=self.arch),
                                                     )
            self.so_far_dev_tasks_dataloaders.append(dataloader)
        print(f'initialize CL dataloaders success, setup: {setup}')

    def reset_entity_task_lst(self, entity_task_lst):
        # when order in entity lst is permuted, we should recompute some mapping and offset for the model
        self.entity_task_lst = entity_task_lst
        self.num_tasks = len(self.entity_task_lst)
        self.tid2ents = {tid: ents for tid, ents in enumerate(self.entity_task_lst)}
        self.num_ents_per_task = [len(ents) for ents in self.entity_task_lst]
        self.ent2tid = {}
        for tid, ents in self.tid2ents.items():
            for ent in ents:
                self.ent2tid[ent] = tid

        self.entity_lst = sum(self.entity_task_lst, [])
        self.datareader = NerDataReader(self.bert_model_dir, self.max_len, ent_file_or_ent_lst=self.entity_lst)
        self.train_dataset = self.datareader.build_dataset(self.train_exm_lst, arch=self.arch, loss_type='sigmoid')
        self.dev_dataset = self.datareader.build_dataset(self.dev_exm_lst, arch=self.arch, loss_type='sigmoid')
        self.test_dataset = self.datareader.build_dataset(self.test_exm_lst, arch=self.arch, loss_type='sigmoid')
        self.ent2id = self.datareader.ent2id
        self.tid2entids = {tid: [self.ent2id[ent] for ent in ents] for tid, ents in self.tid2ents.items()}
        self.tid2offset = {tid: [min(entids), max(entids) + 1] for tid, entids in self.tid2entids.items()}
        print('tid2offset', self.tid2offset)
        print('id2ent', self.datareader.id2ent)
        print('tid2entids', self.tid2entids)

    def permute_task_order(self, tasks_sorted_ids=None):
        # permute the task order
        if tasks_sorted_ids is None:
            return
        print('start to change learning order for the specific permutation.')
        perm_entity_task_lst = sort_by_idx(self.entity_task_lst, tasks_sorted_ids)
        print('perm_entity_task_lst:', perm_entity_task_lst)
        self.reset_entity_task_lst(perm_entity_task_lst)
        perm_mapping = {tasks_sorted_ids[i]: i for i in tasks_sorted_ids}  # 3->0, 0->4
        print('perm_mapping', perm_mapping)
        perm_tid2emxids = {perm_mapping[tid]: exmids for tid, exmids in self.train_tid2exmids.items()}
        self.train_tid2exmids = perm_tid2emxids
        perm_tid2emxids = {perm_mapping[tid]: exmids for tid, exmids in self.dev_tid2exmids.items()}
        self.dev_tid2exmids = perm_tid2emxids
        perm_tid2emxids = {perm_mapping[tid]: exmids for tid, exmids in self.test_tid2exmids.items()}
        self.test_tid2exmids = perm_tid2emxids
        self.init_dataloaders()

    def load_data_with_taskid(self, exm_file, setup='split', split_seed=None, use_pt=False):
        tid2exmids = {tid: set() for tid in range(self.num_tasks)}
        if setup == 'filter':  # task contain all exm with the required entities, non negative
            exm_lst = NerExample.load_from_jsonl(exm_file, token_deli=' ',
                                                 external_attrs=['bert_tok_char_lst', 'ori_2_tok'])
            for exmid, exm in enumerate(exm_lst):
                exm.remove_ent_by_type(self.entity_lst, input_keep=True)
                for ent in exm.ent_dct:
                    tid2exmids[self.ent2tid[ent]].add(exmid)

        elif setup == 'split':  # task contain a set of exm and only contain the required entities, have negative

            if split_seed is None:
                split_seed = self.split_seed
            # Compare to train.jsonl, train_task.jsonl contain task_id as attr per exm by split
            task_emx_file = exm_file.replace('.jsonl', '_task.jsonl')
            if os.path.exists(task_emx_file):
                exm_lst = NerExample.load_from_jsonl(task_emx_file, token_deli=' ',
                                                     external_attrs=['task_id', 'bert_tok_char_lst', 'ori_2_tok'])
            else:
                exm_lst = NerExample.load_from_jsonl(exm_file, token_deli=' ',
                                                     external_attrs=['bert_tok_char_lst', 'ori_2_tok'])

                num_data = len(exm_lst)
                data_order = list(range(num_data))
                random.seed(split_seed)
                random.shuffle(data_order)

                num_per_task = num_data // self.num_tasks

                # 分好每个数据属于哪个task 然后仅保留这多个task的实体。具体每个task对应的实体在后面会自动mask
                # allocate the data into its predefined task
                for task_id in range(self.num_tasks):
                    if task_id == self.num_tasks - 1:  # in case exist the remain data 除不尽的放入最后一个task
                        exm_ids_per_task = data_order[task_id * num_per_task:]
                    else:
                        exm_ids_per_task = data_order[task_id * num_per_task: task_id * num_per_task + num_per_task]

                    for exm_id in exm_ids_per_task:
                        exm = exm_lst[exm_id]
                        # only need to keep entities used in all tasks, for ents in each task it will automaticly mask afterward (model calc_loss)
                        exm.remove_ent_by_type(self.entity_lst, input_keep=True)
                        exm.task_id = task_id
                NerExample.save_to_jsonl(exm_lst, task_emx_file,
                                         external_attrs=['task_id', 'bert_tok_char_lst', 'ori_2_tok'])

            for exmid, exm in enumerate(exm_lst):
                tid2exmids[exm.task_id].add(exmid)

        else:
            raise NotImplementedError

        if use_pt:
            print(f'{exm_file}.pt loading...')
            pt_lst = torch.load(f'{exm_file}.pt')
            print(f'{exm_file}.pt loaded!')
            assert len(exm_lst) == len(pt_lst)
            for exm, pt in zip(exm_lst, pt_lst):
                exm.pt = pt

        return exm_lst, tid2exmids

    def get_task_dataloader(self, mode='test', tid=None, ent=None):
        if tid is None and ent is None:
            return {
                'test': self.test_dataloader,
                'train': self.train_alltask_dataloader,
                'dev': self.dev_alltask_dataloader,
            }[mode]
        if tid is None:
            assert ent is not None
            tid = self.ent2tid[ent]

        if mode == 'train':
            return self.train_tasks_dataloaders[tid]
        if mode == 'dev':
            return self.dev_tasks_dataloaders[tid]

        raise NotImplementedError


onto_entity_task_lst = [
    ['ORG'],
    ['PERSON'],
    ['GPE'],
    ['DATE'],
    ['CARDINAL'],
    ['NORP'],
]


class Onto_Loader(CL_Loader):
    def __init__(self, setup, bert_model_dir='huggingface_model_resource/bert-base-cased'):
        super(Onto_Loader, self).__init__(
            bert_model_dir=bert_model_dir,
            entity_task_lst=onto_entity_task_lst
        )
        self.datafiles = {
            'train': 'onto/train.jsonl',
            'dev': 'onto/dev.jsonl',
            'test': 'onto/test.jsonl',
        }
        # self.setup = 'split'
        # self.setup = 'filter'
        self.setup = setup


fewnerd_entity_task_lst = [
    ['location-GPE', 'location-bodiesofwater', 'location-island', 'location-mountain', 'location-other', 'location-park', 'location-road/railway/highway/transit'],
    ['person-actor', 'person-artist/author', 'person-athlete', 'person-director', 'person-other', 'person-politician', 'person-scholar', 'person-soldier'],
    ['organization-company', 'organization-education', 'organization-government/governmentagency', 'organization-media/newspaper', 'organization-other', 'organization-politicalparty', 'organization-religion', 'organization-showorganization', 'organization-sportsleague',
     'organization-sportsteam'],
    ['other-astronomything', 'other-award', 'other-biologything', 'other-chemicalthing', 'other-currency', 'other-disease', 'other-educationaldegree', 'other-god', 'other-language', 'other-law', 'other-livingthing', 'other-medical'],
    ['product-airplane', 'product-car', 'product-food', 'product-game', 'product-other', 'product-ship', 'product-software', 'product-train', 'product-weapon'],
    ['building-airport', 'building-hospital', 'building-hotel', 'building-library', 'building-other', 'building-restaurant', 'building-sportsfacility', 'building-theater'],
    ['art-broadcastprogram', 'art-film', 'art-music', 'art-other', 'art-painting', 'art-writtenart'],
    ['event-attack/battle/war/militaryconflict', 'event-disaster', 'event-election', 'event-other', 'event-protest', 'event-sportsevent'],
]


class FewNERD_Loader(CL_Loader):
    def __init__(self, setup):
        super(FewNERD_Loader, self).__init__(
            entity_task_lst=fewnerd_entity_task_lst
        )
        self.datafiles = {
            'train': 'fewnerd/supervised/train.jsonl',
            'dev': 'fewnerd/supervised/dev.jsonl',
            'test': 'fewnerd/supervised/test.jsonl',
        }
        # self.setup = 'split'
        # self.setup = 'filter'
        self.setup = setup


if __name__ == '__main__':
    # analyse_task_ent_dist()
    # exit(0)

    # import datautils as utils
    #
    # utils.setup_seed(0, np, torch)
    # loader = Onto_Loader(setup='split')
    # loader.init_data(bsz=32, arch='span')
    #
    # train_dataloader1 = loader.get_task_dataloader(mode='train', tid=0)
    # dev_dataloader1 = loader.get_task_dataloader(mode='dev', tid=0)
    # for i, inputs_dct in enumerate(train_dataloader1):  # iter step
    #     print(inputs_dct['batch_ner_exm'][0])
    #     x = input()
    #     if x == 'break':
    #         break
    #
    # train_dataloader2 = loader.so_far_train_tasks_dataloaders[0]
    # dev_dataloader2 = loader.so_far_dev_tasks_dataloaders[0]
    # for i, inputs_dct in enumerate(train_dataloader2):  # iter step
    #     print(inputs_dct['batch_ner_exm'][0])
    #     x = input()
    #     if x == 'break':
    #         break

    exit(0)
