# !/usr/bin/env python
# coding=utf-8
"""
Miscellaneous utils for IO and NER.
Started from 2019
"""
import argparse
import os, re, glob, shutil, copy, time, datetime
from typing import *
import json
import random
from pathlib import Path

import ipdb
import numpy as np
import logging
import logging.handlers
from collections import Counter, defaultdict

try:
    import xlrd, xlwt, openpyxl
except ImportError:
    print('找不到xlrd/xlwt/openpyxl包 不能使用xls相关函数')

try:
    import paramiko
except ImportError:
    print('找不到paramiko包 不能使用remote相关函数')

try:
    import prettytable
    # import rich
    # from rich import table
    # from rich import print
    # from rich.pretty import pprint
except ImportError:
    print('找不到prettytable或rich包 不能使用prettytable相关函数')


def file2list(in_file, strip_nl=True, encoding='U8'):
    with open(in_file, 'r', encoding=encoding) as f:
        lines = [line.strip('\n') if strip_nl else line for line in f]
        print(f'read ok! filename: {in_file}, length: {len(lines)}')
    return lines


def file2txt(in_file, encoding='U8'):
    with open(in_file, 'r', encoding=encoding) as f:
        txt = f.read()
    return txt


def file2items(in_file, strip_nl=True, deli='\t', extract=None, filter_fn=None, encoding='U8'):
    # extract: [0,2] or (0,2) or '02'  # assume indices > 9 should not use str
    # filter_fn: lambda item: item[2] == 'Y'
    lines = file2list(in_file, strip_nl=strip_nl, encoding=encoding)
    items = [line.split(deli) for line in lines]
    if filter_fn is not None:
        items = list(filter(filter_fn, items))
        print(f'after filter, length: {len(lines)}')
    if extract is not None:
        assert isinstance(extract, (list, tuple, str)), 'invalid extract args'
        items = [[item[int(e)] for e in extract] for item in items]
    return items


# kv_ids: [0,1] or (0,1) or '01'  # assume indices > 9 should not use str
def file2dict(in_file, deli='\t', kv_order='01'):
    items = file2items(in_file, deli=deli)
    assert isinstance(kv_order, (list, tuple, str)) and len(kv_order) == 2, 'invalid kv_order args'
    k_idx = int(kv_order[0])
    v_idx = int(kv_order[1])
    return {item[k_idx]: item[v_idx] for item in items}


def file2nestlist(in_file, strip_nl=True, encoding='U8', seg_line=''):
    # l1,l2,seg_l,l3,l4,l5,seg_l -> [[l1,l2],[l3,l4,l5]]
    lst = file2list(in_file, strip_nl=strip_nl, encoding=encoding)
    out_lst_lst = []
    out_lst = []
    for line in lst:
        if line == seg_line:
            out_lst_lst.append(out_lst)
            out_lst = []
            continue
        out_lst.append(line)
    if out_lst:
        out_lst_lst.append(out_lst)
    return out_lst_lst


def seg_list(lst, is_seg_fn=None):
    # [l1,l2,seg,l3,l4,l5] -> [[l1,l2],[l3,l4,l5]]
    if is_seg_fn is None:
        is_seg_fn = lambda e: e == ''
    nest_lst = [[]]
    for e in lst:
        if is_seg_fn(e):
            nest_lst.append([])
            continue
        nest_lst[-1].append(e)
    return nest_lst


def make_sure_dir_exist(file_or_dir):
    file_or_dir = Path(file_or_dir)
    if not file_or_dir.parent.exists():
        file_or_dir.parent.mkdir(parents=True, exist_ok=True)
    return True


def np2py(obj):
    # np格式不支持json序列号,故转化为python数据类型
    if isinstance(obj, (int, float, bool, str)):
        return obj
    if isinstance(obj, dict):
        for k in obj:
            obj[k] = np2py(obj[k])
        return obj
    elif isinstance(obj, (list, tuple)):
        for i in range(len(obj)):
            obj[i] = np2py(obj[i])
        return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_json(obj, json_file, indent=2, verbose=True):
    make_sure_dir_exist(json_file)
    obj = np2py(obj)
    json.dump(obj, open(json_file, 'w', encoding='U8'),
              ensure_ascii=False, indent=indent)
    if verbose: print(f'save json file ok! {json_file}')


def load_json(json_file):
    with open(json_file, 'r', encoding='U8') as f:
        json_data = json.load(f)
    return json_data


def save_jsonl(obj_lst, jsonl_file, verbose=True):
    """write data by line with json"""
    make_sure_dir_exist(jsonl_file)
    with open(jsonl_file, 'w', encoding='U8') as f:
        for obj in obj_lst:
            obj = np2py(obj)
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    if verbose: print(f'save jsonl file ok! {jsonl_file}, length: {len(obj_lst)}')


def load_jsonl(jsonl_file):
    """read data by line with json"""
    with open(jsonl_file, 'r', encoding='U8') as f:
        return [json.loads(line.strip()) for line in f]


def stats_lst(lst, digtal=True):
    ret = {}
    if digtal and lst:
        lst.sort()
        num = len(lst)
        quarter = num // 4
        ret['1/4分位'] = lst[quarter:quarter + 1]
        ret['2/4分位'] = lst[quarter * 2:quarter * 2 + 1]
        ret['3/4分位'] = lst[quarter * 3:quarter * 3 + 1]

        ret['mean'] = np.mean(lst)
        ret['max'] = np.max(lst)
        ret['min'] = np.min(lst)
        ret['std'] = np.std(lst)

        counts = [(key, cnt, f'{cnt / num:.2%}') for key, cnt in Counter(lst).most_common()]
        if len(counts) > 10:
            brief_counts = counts[:5] + ['...'] + counts[-5:]
        else:
            brief_counts = counts
        ret['brief_counts'] = brief_counts
        return ret
    return {}


def list_dir_and_file(path):
    lsdir = os.listdir(path)
    dirs = [d for d in lsdir if os.path.isdir(os.path.join(path, d))]
    files = [f for f in lsdir if os.path.isfile(os.path.join(path, f))]
    return dirs, files


def list2file(lines, out_file, add_nl=True, deli='\t'):
    # 兼容
    if isinstance(lines, str):
        lines, out_file = out_file, lines
    assert len(lines) > 0, 'lines must be not None'
    with open(out_file, 'w', encoding='U8') as f:
        if isinstance(lines[0], (list, tuple)):
            lines = [deli.join(map(str, item)) for item in lines]
        # other: str, int, float, bool, obj will use f'{} to strify
        out_list = [f'{line}\n' if add_nl else f'{line}' for line in lines]
        f.writelines(out_list)
        print(f'save ok! filename: {out_file}, length: {len(out_list)}')


def freqs(lst):
    c = Counter(lst)
    return c.most_common()


def list2stats(in_lst, out_file=None):
    stats = freqs(in_lst)
    print(*stats, sep='\n')
    if out_file is not None:
        list2file(stats, out_file, deli='\t')


# sheet_ids: [0,1,3] default read all sheet
def xls2items(in_xls, start_row=1, sheet_ids=None):
    items = []
    xls = xlrd.open_workbook(in_xls)
    sheet_ids = list(range(xls.nsheets)) if sheet_ids is None else sheet_ids
    for sheet_id in sheet_ids:
        sheet = xls.sheet_by_index(sheet_id)
        nrows, ncols = sheet.nrows, sheet.ncols
        print(f'reading... sheet_id:{sheet_id} sheet_name:{sheet.name} rows:{nrows} cols:{ncols}')
        for i in range(start_row, nrows):
            items.append([sheet.cell_value(i, j) for j in range(ncols)])
    return items


# this only support old xls (nrows<65537)
def items2xls_old(items, out_xls=None, sheet_name=None, header=None, workbook=None, max_row_per_sheet=65537):
    workbook = xlwt.Workbook(encoding='utf-8') if workbook is None else workbook
    sheet_name = '1' if sheet_name is None else sheet_name
    num_sheet = 1
    worksheet = workbook.add_sheet(f'{sheet_name}_{num_sheet}')  # 创建一个sheet
    if header is not None:
        for j in range(len(header)):
            worksheet.write(0, j, header[j])
    row_ptr = 1 if header is not None else 0
    for item in items:
        if row_ptr + 1 > max_row_per_sheet:
            num_sheet += 1
            worksheet = workbook.add_sheet(f'{sheet_name}_{num_sheet}')
            if header is not None:
                for j in range(len(header)):
                    worksheet.write(0, j, header[j])
            row_ptr = 1 if header is not None else 0

        for j in range(len(item)):
            worksheet.write(row_ptr, j, item[j])
        row_ptr += 1
    if out_xls is not None:  # 如果为None表明调用者其实还有新的items要加到新的sheet要中，只想要返回的workbook对象
        workbook.save(out_xls)
        print(f'save ok! xlsname: {out_xls}, num_sheet: {num_sheet}')
    return workbook


def items2xls(items, out_xls=None, sheet_name=None, header=None, workbook=None, max_row_per_sheet=65537):
    if workbook is None:
        workbook = openpyxl.Workbook()  # create new workbook instance
        active_worksheet = workbook.active
        workbook.remove(active_worksheet)
    if sheet_name is None:
        sheet_name = 'sheet'
    num_sheet = 1
    worksheet = workbook.create_sheet(f'{sheet_name}_{num_sheet}' if len(items) > max_row_per_sheet else f'{sheet_name}')  # 创建一个sheet
    if header is not None:
        for j in range(len(header)):
            worksheet.cell(0 + 1, j + 1, header[j])  # cell x y 从1开始
    row_ptr = 1 if header is not None else 0
    for item in items:
        if row_ptr + 1 > max_row_per_sheet:
            num_sheet += 1
            worksheet = workbook.create_sheet(f'{sheet_name}_{num_sheet}')
            if header is not None:
                for j in range(len(header)):
                    worksheet.cell(0 + 1, j + 1, header[j])
            row_ptr = 1 if header is not None else 0

        for j in range(len(item)):
            worksheet.cell(row_ptr + 1, j + 1, item[j])
        row_ptr += 1
    if out_xls is not None:  # 如果为None表明调用者其实还有新的items要加到新的sheet表中，只想要返回的workbook对象
        workbook.save(out_xls)
        print(f'save ok! xlsname: {out_xls}, num_sheet: {num_sheet}')
    return workbook


def merge_file(file_list, out_file, shuffle=False):
    assert isinstance(file_list, (list, tuple))
    ret_lines = []
    for i, file in enumerate(file_list):
        lines = file2list(file, strip_nl=False)
        print(f'已读取第{i}个文件:{file}\t行数{len(lines)}')
        ret_lines.extend(lines)
    if shuffle:
        random.shuffle(ret_lines)
    list2file(out_file, ret_lines, add_nl=False)


def merge_file_by_pattern(pattern, out_file, shuffle=False):
    file_list = glob.glob(pattern)
    merge_file(file_list, out_file, shuffle)


# ratio: [18,1,1] or '18:1:1'
# num: 1000
def split_file(file_or_lines, num=None, ratio=None, files=None, shuffle=True, seed=None):
    assert num or ratio, 'invalid args: at least use num or ratio'
    if type(file_or_lines) == str:
        lines = file2list(file_or_lines, strip_nl=False)
    else:
        assert isinstance(file_or_lines, (list, tuple)), 'invalid args file_or_lines'
        lines = file_or_lines
    length = len(lines)
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(lines)
    if num:
        assert num < length, f'invalid args num: num:{num} should < filelen: {length}'
        lines1 = lines[:num]
        lines2 = lines[num:]
        if files:
            assert len(files) == 2
            list2file(files[0], lines1, add_nl=False)
            list2file(files[1], lines2, add_nl=False)
        return lines1, lines2
    if ratio:  # [6,2,2]
        if isinstance(ratio, str):
            ratio = list(map(int, ratio.split(':')))
        cumsum_ratio = np.cumsum(ratio)  # [6,8,10]
        sum_ratio = cumsum_ratio[-1]  # 10
        assert sum_ratio <= length, f'invalid args ratio: ratio:{ratio} should <= filelen: {length}'
        indices = [length * r // sum_ratio for r in cumsum_ratio]  # [6,8,11] if length=11
        indices = [0] + indices
        split_lines_lst = []
        for i in range(len(indices) - 1):
            split_lines_lst.append(lines[indices[i]:indices[i + 1]])
        if files:
            assert len(files) == len(split_lines_lst)
            for i, lines in enumerate(split_lines_lst):
                list2file(files[i], lines, add_nl=False)
        return split_lines_lst


def delete_file(file_or_dir, verbose=True):
    if os.path.exists(file_or_dir):
        if os.path.isfile(file_or_dir):  # 文件file
            os.remove(file_or_dir)
            if verbose:
                print(f'delete ok! file: {file_or_dir}')
            return True
        else:  # 目录dir
            for file_lst in os.walk(file_or_dir):
                for name in file_lst[2]:
                    os.remove(os.path.join(file_lst[0], name))
            shutil.rmtree(file_or_dir)
            if verbose:
                print(f'delete ok! dir: {file_or_dir}')
            return True
    else:
        print(f'delete false! file/dir not exists: {file_or_dir}')
        return False


def find_duplicates(in_lst):
    # duplicates = []
    # seen = set()
    # for item in in_lst:
    #     if item not in seen:
    #         seen.add(item)
    #     else:
    #         duplicates.append(item)
    # return duplicates
    c = Counter(in_lst)
    return [k for k, v in c.items() if v > 1]


def remove_duplicates_for_file(in_file, out_file=None, keep_sort=True):
    if not out_file:
        out_file = in_file
    lines = file2list(in_file)

    if keep_sort:
        out_lines = []
        tmp_set = set()
        for line in lines:
            if line not in tmp_set:
                out_lines.append(line)
                tmp_set.add(line)
    else:
        out_lines = list(set(lines))

    list2file(out_file, out_lines)


def remove_duplicates(in_list, keep_sort=True):
    lines = in_list

    if keep_sort:
        out_lines = []
        tmp_set = set()
        for line in lines:
            if line not in tmp_set:
                out_lines.append(line)
                tmp_set.add(line)
    else:
        out_lines = list(set(lines))

    return out_lines


def set_items(items, keep_order=False):
    items = [tuple(item) for item in items]  # need to be hashable i.e. tuple
    ret = []
    if keep_order:
        seen = set()
        for item in items:
            if item not in seen:
                seen.append(item)
                ret.append(list(item))
        return ret
    if not keep_order:
        ret = list(map(list, set(items)))
        return ret


def sort_items(items, sort_order):
    if isinstance(sort_order, str):
        sort_order = map(int, sort_order)
    for idx in reversed(sort_order):
        items.sort(key=lambda item: item[idx])
    return items


def check_overlap(list1, list2, verbose=False):
    set1 = set(list1)
    set2 = set(list2)
    count1 = Counter(list1)
    dupli1 = [k for k, v in count1.items() if v > 1]
    count2 = Counter(list2)
    dupli2 = [k for k, v in count2.items() if v > 1]

    print(f'原始长度:{len(list1)}\t去重长度{len(set1)}\t重复项{dupli1}')
    print(f'原始长度:{len(list2)}\t去重长度{len(set2)}\t重复项{dupli2}')

    union = sorted(set1 & set2)  # 变为list
    print(f'一样的数量: {len(union)}')
    if verbose or len(union) <= 30:
        print(*union, sep='\n', end='\n\n')
    else:
        print(*union[:30], sep='\n', end=f'\n ..more(total:{len(union)})\n\n')

    a = sorted(set1 - set2)  # 变为list
    print(f'前者多了: {len(a)}')
    if verbose or len(a) <= 30:
        print(*a, sep='\n', end='\n\n')
    else:
        print(*a[:30], sep='\n', end=f'\n ..more(total:{len(a)})\n\n')

    b = sorted(set2 - set1)  # 变为list
    print(f'后者多了: {len(b)}')
    if verbose or len(b) <= 30:
        print(*b, sep='\n', end='\n\n')
    else:
        print(*b[:30], sep='\n', end=f'\n ..more(total:{len(a)})\n\n')


def print_len(files):
    if not isinstance(files, list):
        files = [files]
    len_lst = []
    for file in files:
        with open(file, 'r', encoding='U8') as f:
            len_lst.append(len(f.readlines()))
    print(len_lst, f'总和: {sum(len_lst)}')


def f(object):
    """ 格式化 """
    if 'numpy' in str(type(object)) and len(object.shape) == 1:
        object = object.tolist()
    if isinstance(object, (list, tuple)):
        if len(object) == 0:
            return ''
        if isinstance(object[0], (int, float)):
            ret = list(map(lambda e: f'{e:.2f}', object))
            return str(ret)
    return str(object)


def get_first_value_of_dict(dct):
    # ret = list(dct.values())[0]  # slow but readable
    ret = next(iter(dct.values()))  # fast
    return ret


def print_total_param(pt_model):
    print('total_params:', sum(p.numel() for p in pt_model.parameters()))


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict2obj(dct):
    if not isinstance(dct, dict):
        return dct
    inst = Dict()
    for k, v in dct.items():
        inst[k] = dict2obj(v)
    return inst


class ImmutableDict(dict):
    def __hash__(self):
        return id(self)

    def _immutable(self, *args, **kws):
        raise TypeError('object is immutable')

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable
    pop = _immutable
    popitem = _immutable


class Any2Id():
    # fix 是否固定住字典不在增加
    def __init__(self, use_line_no=False, exist_dict=None, fix=True, counter=None):
        self.fix = fix
        self.use_line_no = use_line_no
        self.file = None
        self.counter = Counter() if not counter else counter
        self.any2id = {}  # 内部核心dict
        if exist_dict is not None:
            self.any2id.update(exist_dict)

        # for method_name in dict.__dict__:  # 除了下面显式定义外保证能以dict的各种方法操作Any2id
        #     setattr(self, method_name, getattr(self.any2id, method_name))

    def keys(self):
        return self.any2id.keys()

    def values(self):
        return self.any2id.values()

    def items(self):
        return self.any2id.items()

    def pop(self, key):
        return self.any2id.pop(key)

    def __getitem__(self, item):
        return self.any2id.__getitem__(item)

    def __setitem__(self, key, value):
        self.any2id.__setitem__(key, value)

    def __len__(self):
        return self.any2id.__len__()

    def __iter__(self):
        return self.any2id.__iter__()

    def __str__(self):
        string = self.any2id.__str__()
        return string[:200] + '...' if len(string) > 200 else string

    def __repr__(self):
        return f'<Any2Id> {str(self)}'

    def set_fix(self, fix):
        self.fix = fix

    def get(self, key, default=None, add=False):
        if not add:
            return self.any2id.get(key, default)
        else:
            # new_id = len(self.any2id)
            new_id = max(self.any2id.values()) + 1
            self.any2id[key] = new_id
            return new_id

    def get_reverse(self):
        return self.__class__(exist_dict={v: k for k, v in self.any2id.items()})

    def save(self, file=None, use_line_no=None, deli='\t'):
        use_line_no = self.use_line_no if use_line_no is None else use_line_no
        items = sorted(self.any2id.items(), key=lambda e: e[1])
        out_items = [item[0] for item in items] if use_line_no else items
        file = self.file if file is None else file
        list2file(out_items, file, deli=deli)
        print(f'词表文件生成成功: {file} {items[:5]}...')

    def load(self, file, use_line_no=None, deli='\t'):
        if use_line_no is None:
            use_line_no = self.use_line_no
        items = file2items(file, deli=deli)
        if use_line_no or len(items[0]) == 1:
            self.any2id = {item[0]: i for i, item in enumerate(items)}
        else:
            self.any2id = {item[0]: int(item[1]) for item in items}

    def to_count(self, any_lst):
        self.counter.update(any_lst)  # 维护一个计数器

    def reset_counter(self):
        self.counter = Counter()

    def rebuild_by_counter(self, restrict=None, min_freq=None, max_vocab_size=None):
        if not restrict:
            restrict = ['<pad>', '<unk>', '<eos>']
        freqs = self.counter.most_common()
        tokens_lst = restrict[:]
        curr_vocab_size = len(tokens_lst)
        for token, cnt in freqs:
            if min_freq and cnt < min_freq:
                break
            if max_vocab_size and curr_vocab_size >= max_vocab_size:
                break
            tokens_lst.append(token)
            curr_vocab_size += 1
        self.any2id = {token: i for i, token in enumerate(tokens_lst)}

    @classmethod
    def from_file(cls, file, use_line_no=False, deli='\t'):
        inst = cls(use_line_no=use_line_no)
        if os.path.exists(file):
            inst.load(file, deli=deli)
        else:
            # will return inst with empty any2id , e.g. boolean(inst) or len(inst) will return False
            print(f'vocab file: {file} not found, need to build and save later')
        inst.file = file
        return inst


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__') or not all([hasattr(x, '__len__') for x in sequences]):
        raise ValueError(f'sequences invalid: {sequences}')

    len_lst = [len(x) for x in sequences]

    if len(set(len_lst)) == 1:  # 长度均相等
        ret = np.array(sequences)
        if maxlen is not None:
            ret = ret[:, -maxlen:] if truncating == 'pre' else ret[:, :maxlen]
        return ret

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(len_lst)

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError(f'Truncating type not support')

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type not support')
    return x


def get_file_logger(logger_name, log_file='./qiznlp.log', level='DEBUG'):
    level = {
        'ERROR': logging.ERROR,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'NOTSET': logging.NOTSET,
    }.get(level, logging.DEBUG)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    fh = logging.handlers.TimedRotatingFileHandler(log_file, when="D", interval=1, backupCount=7, encoding="utf-8")
    fh.suffix = "%Y-%m-%d.log"  # 设置后缀名称，跟strftime的格式一样
    fh.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")  # _\d{2}-\d{2}
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    logger.propagate = False  # 取消传递，只有这样才不会传到logger_root然后又在控制台也打印一遍

    return logger


def suppress_tf_warning(tf):
    import warnings
    warnings.filterwarnings('ignore')
    tf.logging.set_verbosity(tf.logging.ERROR)


def flat_list(lst_in_lst):
    # lst_in_lst: 二维列表
    return sum(lst_in_lst, [])  # flatten


def split_list(lst, len_lst):
    # 按照len_lst对应的长度来来切分lst
    # batch_num_span = list(map(lambda x: int(x * (x + 1) / 2), batch_seq_len))
    # num_span_indices = np.cumsum(batch_num_span).tolist()
    # split_lst = [ner_span_lst_of_batch[i:j] for i, j in zip([0] + num_span_indices, num_span_indices)]
    assert sum(len_lst) == len(lst)
    lst_ = copy.deepcopy(lst)
    ret_lst = []
    for length in len_lst:
        ret_lst.append(lst_[:length])
        lst_ = lst_[length:]
    return ret_lst


def find_files(dir_name, pattern=None):
    dir_name = Path(dir_name)
    if pattern is None:
        files = dir_name.iterdir()
    else:
        files = list(dir_name.glob(pattern))  # pattern = '*.py'
    return files


def find_latest_file(dir_name, pattern=None):
    dir_name = Path(dir_name)
    if pattern is None:
        files = dir_name.iterdir()
    else:
        files = dir_name.glob(pattern)  # pattern = '*.py'
    files = sorted(files, key=lambda e: e.lstat().st_mtime, reverse=True)
    # files = [os.path.join(dir_name, f) for f in os.listdir(dir_name)]
    # files = sorted(glob.glob(dir_name+'/*.pth'), key=os.path.getmtime, reverse=True)
    if len(files) > 0:
        return str(files[0])
    else:
        print('not found!')
        return None


def del_if_exists(dir_or_path_name, pattern=None):
    dir_or_path_name = Path(dir_or_path_name)
    if not dir_or_path_name.exists():
        return
    if dir_or_path_name.is_dir():
        if pattern is None:
            files = dir_or_path_name.iterdir()
        else:
            files = dir_or_path_name.glob(pattern)  # pattern = '*.py'
        for file in files:
            file.unlink()
    else:
        dir_or_path_name.unlink()


def get_curr_time_str(pattern=None):
    if pattern is None:
        pattern = '%y_%m_%d_%H_%M'  # 21_09_12_11_12
        # pattern = '%y_%m_%d_%H_%M_%S'  # 21_09_12_11_12_34 for seconds
        # pattern = '%y_%m_%d_%H_%M_%S_%f'  # 21_09_12_11_12_34_123456 for microsecond微秒
    # time_str = time.strftime(pattern, time.localtime(time.time()))
    time_str = datetime.datetime.now().strftime(pattern)
    return time_str


def cal_time(since):
    now = time.time()
    s = now - since
    m = s // 60
    s -= m * 60
    return '%dm %ds' % (m, s)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_like_object():
    return argparse.Namespace()


def save_args_to_json_file(args, json_file, verbose=False):
    # args_json = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float))}  # only save certain type supported by json
    # def save_list(v):
    #     if isinstance(v, list):
    #         if not v or all(isinstance(v_, (str, int, float)) for v_ in v):
    #             return True
    #     return False
    # args_json = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float)) or save_list(v)}  # only save certain type supported by json
    args_json = {}
    for k, v in vars(args).items():
        if isinstance(v, (str, int, float)):  # json valid types
            args_json[k] = v
        elif isinstance(v, list):  # list of valid types
            if all(isinstance(v_, (str, int, float)) for v_ in v):
                args_json[k] = v
        elif isinstance(v, Path):  # Path object saved as string
            args_json[k] = str(v)
        else:
            pass
    save_json(args_json, json_file, verbose=verbose)


def load_args_by_json_file(json_file, exist_args=None, verbose=True):
    args_json = load_json(json_file)
    if exist_args is None:
        args = argparse.Namespace()
    else:
        args = exist_args
    for k, v in args_json.items():
        setattr(args, k, v)
    if verbose:
        print(f'Load {json_file} for {"exist" if exist_args else "new"} args success!')
    return args


def print_vars(obj, sort=False, maxlen=200):
    """
    print_vars(args)
    """
    if hasattr(obj, '__dict__'):  # normal object, args, ...
        dct = vars(obj)
    else:
        dct = obj
    assert isinstance(dct, dict)
    it = sorted(dct.items()) if sort else dct.items()
    for k, v in it:
        v = str(v)
        print(f'{k}: {v[:maxlen] + "..." if len(v) > maxlen else v}')


def setup_seed(seed: int, np, torch):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.cuda.cudnn_enabled = False
    # torch.backends.cudnn.enabled = False


def stats(array):
    mean = np.mean(array)
    std = np.std(array)
    out = f'{mean:.2f}±{std:.2f}'
    print(out)
    return out


def print_setup(np, torch):
    np.set_printoptions(linewidth=4000, suppress=True)  # 多长换行 不打印科学计数法
    torch.set_printoptions(linewidth=4000, sci_mode=False)  # 多长换行 不打印科学计数法
    # np.set_printoptions(precision=2)  # decimal 小数点位数
    # np.set_printoptions(threshold=sys.maxsize)  # 打印是否截断
    # np.set_printoptions(linewidth=4000)  # 多长换行
    # np.set_printoptions(suppress=True)  # 不打印科学计数法


def sort_by_idx(lst, ids=None, prev_num=None):
    if ids is None:
        return lst
    sorted_lst = []
    for idx in ids:
        sorted_lst.append(lst[idx])
    if prev_num is not None:
        sorted_lst.extend(lst[prev_num:])
    return sorted_lst


class Remote_Open:
    """
    ro = Remote_Open(host_ip, port, username, password)
    f = ro.open(json_file_path)  # need close()
    json = ro.load_json(json_file_path)
    jsonl = ro.load_jsonl(jsonl_file_path)
    nparray = ro.load_np(np_file_path)
    tenosr = ro.load_pt(pt_file_path)
    """

    def __init__(self, host_ip, port, username, password, timeout=5):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host_ip, port, username, password, timeout=timeout)
        self.client = client
        self.sftp_client = self.client.open_sftp()
        logging.info(self.sftp_client)
        self.default_dir = None

    def set_default_dir(self, dir_path):
        self.default_dir = dir_path

    def clean_default_dir(self):
        self.default_dir = None

    def get_path(self, filepath):
        if self.default_dir is not None:
            return os.path.join(self.default_dir, filepath)
        else:
            return filepath

    def open(self, filepath, mode='r'):
        """
        return file-like object
        encoding is automatic process.
        file-like object having function: seek() read() readline() if have encoding...
        """
        remote_file = self.sftp_client.open(self.get_path(filepath), mode)
        return remote_file

    def load_json(self, json_file):
        with self.open(json_file, 'r') as f:
            ret = json.load(f)
        return ret

    def load_jsonl(self, jsonl_file):
        with self.open(jsonl_file, 'r') as f:
            ret = [json.loads(line.strip()) for line in f]
        return ret

    def load_np(self, np_file):
        if np_file.endswith('npy'):
            return self.load_npy(np_file)
        elif np_file.endswith('npz'):
            return self.load_npz(np_file)
        else:
            print('should specify np_file type (.npy or .npz)')

    def load_npz(self, npz_file):
        with self.open(npz_file, 'r') as f:
            with np.load(f) as npzfile:
                ret = dict(npzfile)
        # use list(ret.values()) to resume to List
        return ret

    def load_npy(self, npy_file):
        with self.open(npy_file, 'r') as f:
            ret = np.load(f)
        return ret  # ndarray

    # def load_pt(self, pt_file):
    #     with self.open(pt_file, 'r') as f:
    #         ret = torch.load(f)
    #     return ret  # ndarray


class Meaner:
    def __init__(self):
        self.mean = 0
        self.exist_len = 0

    def add(self, value):
        if self.exist_len == 0:
            self.mean = value
            self.exist_len = 1
        else:
            self.mean = (self.mean * self.exist_len + value) / (self.exist_len + 1)
            self.exist_len += 1

    def reset(self):
        self.__init__()

    def __str__(self):
        return str(self.mean)

    @property
    def v(self):
        return self.mean


class F1_Meaner:
    """
    累进式计算F1均值
    需要num_gold, num_pred, tp
    在用span level ner时 计算参考代码为:
    # calc f1
    num_gold = torch.sum(span_ner_tgt_lst_onehot)  # [*,ent]
    num_pred = torch.sum(span_ner_pred_lst_onehot)  # [*,ent]
    tp = torch.sum(span_ner_tgt_lst_onehot * span_ner_pred_lst_onehot)
    f1 = torch.tensor(1.) if num_gold == num_pred == 0 else 2 * tp / (num_gold + num_pred + 1e-12)
    return f1.item(), (num_gold.item(), num_pred.item(), tp.item())
    """

    def __init__(self):
        self.f1 = 0
        self.num_gold = 0
        self.num_pred = 0
        self.tp = 0
        self.prec = 0
        self.rec = 0

    def add(self, num_gold, num_pred, tp):
        self.num_gold += num_gold
        self.num_pred += num_pred
        self.tp += tp
        self.f1 = 1. if self.num_gold == self.num_pred == 0 else 2 * self.tp / (self.num_gold + self.num_pred + 1e-12)
        self.prec = 1. if self.num_pred == 0 else self.tp / self.num_pred
        self.rec = 1. if self.num_gold == 0 else self.tp / self.num_gold

    def reset(self):
        self.__init__()

    def __str__(self):
        return str(self.f1)

    @property
    def v(self):
        return self.f1

    def f1(self):
        return self.f1

    def prec(self):
        return self.prec

    def rec(self):
        return self.rec


class Time_Meaner:
    def __init__(self):
        self.mean = 0
        self.exist_len = 0
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def end(self):
        if self.start_time is None:
            print('please call start() first!')
            return
        value = time.time() - self.start_time
        if self.exist_len == 0:
            self.mean = value
            self.exist_len = 1
        else:
            self.mean = (self.mean * self.exist_len + value) / (self.exist_len + 1)
            self.exist_len += 1

    def reset(self):
        self.__init__()

    def __str__(self):
        return str(self.mean)

    @property
    def v(self):
        return self.mean


class Time_Meaners:
    def __init__(self, num, start_with=0):
        self.meaners = {}
        for idx in range(start_with, start_with + num):
            self.meaners[idx] = Time_Meaner()

    def reset(self, idx=None):
        if idx is None:
            for i in self.meaners:
                self.meaners[i].reset()
        else:
            self.meaners[idx].reset()

    def __getitem__(self, k):
        return self.meaners.__getitem__(k)


class CUDA_Recorder:
    def __init__(self, torch):
        self.torch = torch
        self.surpress = False
        self.target_value = 0
        try:
            self.previous_value = self.torch.cuda.memory_allocated()
        except:
            pass

    def record(self, print_info=None):
        if self.surpress:
            return
        curr_value = self.torch.cuda.memory_allocated()
        self.target_value = curr_value - self.previous_value
        if print_info is not None:
            print(f'{print_info}: (CUDA){self.target_value}')
        self.previous_value = curr_value

    def reset(self):
        self.previous_value = self.torch.cuda.memory_allocated()


class Time_Recorder:
    def __init__(self):
        self.previous_value = time.time()
        self.surpress = False
        self.target_value = 0

    def record(self, print_info=None):
        if self.surpress:
            return
        curr_value = time.time()
        self.target_value = curr_value - self.previous_value
        if print_info is not None:
            print(f'{print_info}: (Time){self.target_value}')
        self.previous_value = curr_value

    def reset(self):
        self.previous_value = time.time()


class NerExample:
    def __init__(self, char_lst, ent_dct, ent_span_dct=None, tag_lst=None, token_deli='', dedup=True):
        """
        ent_dct: {1:[[0,2], [4,7]], 2:[[9,12]], ...}
        ent_span_dct: {(0,2):1, (4,7):1, ...}
        """
        self.char_lst = char_lst  # must
        if isinstance(ent_dct, defaultdict): ent_dct = dict(ent_dct)
        self.ent_dct = ent_dct  # must
        if dedup:
            self.duplicate_ent_dct()  # 去除重复
        self.ent_span_dct = ent_span_dct  # option
        self.tag_lst = tag_lst  # option
        self.token_deli = token_deli  # 中文是'' 英文是' '
        self.text = self.token_deli.join(self.char_lst)

    def update_text(self):
        if hasattr(self, 'text'):
            self.text = self.token_deli.join(self.char_lst)

    def initial_new(self, dedup=True):
        """ return a copy but with other unnecessary attributes """
        new_exm = NerExample(char_lst=self.char_lst, ent_dct=self.ent_dct, token_deli=self.token_deli, dedup=dedup)
        new_exm.update(anchor='ent_dct', dedup=dedup)
        return new_exm

    def set_token_deli(self, token_deli):
        self.token_deli = token_deli

    def get_text(self, start, end):
        return self.token_deli.join(self.char_lst[start:end])

    def truncate(self, max_size=512, direction='bothway'):
        if len(self.char_lst) <= max_size:
            return
        diff = len(self.char_lst) - max_size

        ent_lst = self.get_ent_lst()

        # 先修改实体起始坐标 左边截断就要
        if direction == 'head':
            self.char_lst = self.char_lst[diff:]
            for ent in ent_lst:
                ent[1] = ent[1] - diff
                ent[2] = ent[2] - diff

        elif direction == 'tail':
            self.char_lst = self.char_lst[:-diff]

        elif direction == 'bothway':
            left = diff // 2
            right = diff - left
            self.char_lst = self.char_lst[left:-right]

            for ent in ent_lst:
                ent[1] = ent[1] - left
                ent[2] = ent[2] - left

        else:
            raise NotImplementedError

        # 再根据新的长度修正实体起始坐标
        length = len(self.char_lst)
        ent_lst_ = []
        for ent_type, start, end, *other in ent_lst:
            end = end - 1  # end 也变为闭区间好判断一点
            if start >= 0 and end < length:  # 正常实体
                ent_lst_.append([ent_type, start, end + 1] + other)
                continue

            if start < 0:  # 左边界越过最左
                if end < 0:  # 右边界也越过最左
                    continue  # 不要该实体
                else:
                    start = 0  # 实体不要被截断左边部分

            if end >= length:  # 右边界越过最右
                if start >= length:  # 左边界也越过最右
                    continue  # 不要该实体
                else:
                    end = length - 1  # 实体不要被截断右边部分

            ent_lst_.append([ent_type, start, end + 1] + other)

        ent_dct_ = defaultdict(list)
        for ent_type, start, end, *other in ent_lst_:
            ent_dct_[ent_type].append([start, end] + other)
        self.ent_dct = dict(ent_dct_)
        self.update_text()
        self.update(anchor='ent_dct')

    def truncate_by_bert_tok_char_lst(self, max_size=512, direction='bothway'):
        assert direction == 'tail'
        # 防止bert tokenize 之后的tok数量变多了超过长度
        # 直接裁剪tok, 最后再反映回去原始word
        if hasattr(self, 'bert_tok_char_lst') and len(self.bert_tok_char_lst) > max_size:
            tok_start_idx = self.ori_2_tok
            tok_end_idx = self.ori_2_tok[1:] + [len(self.bert_tok_char_lst)]
            num_tok_per_word = [e - s for s, e in zip(tok_start_idx, tok_end_idx)]
            num_tok_need_truncate = len(self.bert_tok_char_lst) - max_size
            num_word_to_truncate = 0
            num_accu_tok = 0
            for num_tok in reversed(num_tok_per_word):
                num_accu_tok += num_tok
                num_word_to_truncate += 1
                self.ori_2_tok.pop(-1)
                self.bert_tok_char_lst = self.bert_tok_char_lst[:-num_tok]
                if num_accu_tok >= num_tok_need_truncate:
                    break

            self.truncate(max_size=len(self.char_lst) - num_word_to_truncate, direction=direction)

        if hasattr(self, 'ori_indexes'):
            self.ori_indexes = self.ori_indexes[: len(self.bert_tok_char_lst)]
        if hasattr(self, 'sub_tokens'):
            self.sub_tokens = self.sub_tokens[: len(self.bert_tok_char_lst)]

    @staticmethod
    def _segment_exm(exm, max_size=512, prefix_context_len=0):
        """ split too-long exm into two
            prefix_context_len: the rear one should contain how long prefix context
        """
        if hasattr(exm, 'sub_tokens'):
            assert hasattr(exm, 'ori_indexes')
            """ char_lst:   0   1   2   3   4   5
                sub_tokens: 0   12  3   45  6   78
                ori_indexes:0   11  2   33  4   55
            """
            if len(exm.sub_tokens) > max_size:
                seg_offset = max_size - 1  # the last offset meet the requirement
                while exm.ori_indexes[seg_offset] == exm.ori_indexes[seg_offset + 1]:  # in case segment a complete word
                    seg_offset -= 1
                ori_seg_offset = exm.ori_indexes[seg_offset]

                prev_num_sub_toknes = seg_offset + 1
                prev_num_chars = ori_seg_offset + 1

                prev_exm = copy.deepcopy(exm)
                prev_exm.truncate(max_size=prev_num_chars, direction='tail')
                prev_exm.sub_tokens = prev_exm.sub_tokens[:prev_num_sub_toknes]
                prev_exm.ori_indexes = prev_exm.ori_indexes[:prev_num_sub_toknes]

                rear_exm = copy.deepcopy(exm)
                rear_num_chars = len(exm.char_lst) - prev_num_chars + prefix_context_len
                rear_exm.truncate(max_size=rear_num_chars, direction='head')
                # rear_start_offset = prev_num_chars - prefix_context_len - 1
                rear_start_offset = ori_seg_offset - prefix_context_len
                for i in range(seg_offset, -1, -1):
                    if exm.ori_indexes[i] == rear_start_offset:
                        break
                rear_num_remove_sub_toknes = i + 1
                rear_exm.sub_tokens = rear_exm.sub_tokens[rear_num_remove_sub_toknes:]
                rear_exm.ori_indexes = rear_exm.ori_indexes[rear_num_remove_sub_toknes:]
                rear_exm.ori_indexes = [i - (prev_num_chars - prefix_context_len) for i in rear_exm.ori_indexes]

                return [prev_exm, rear_exm]
            else:
                return [exm, None]
        else:
            if len(exm.char_lst) > max_size:
                prev_num_chars = max_size
                prev_exm = copy.deepcopy(exm)
                prev_exm.truncate(max_size=prev_num_chars, direction='tail')
                rear_exm = copy.deepcopy(exm)
                rear_num_chars = len(exm.char_lst) - prev_num_chars + prefix_context_len
                rear_exm.truncate(max_size=rear_num_chars, direction='head')
                return [prev_exm, rear_exm]
            else:
                return [exm, None]

    @staticmethod
    def segment_exm(exm, max_size=512, prefix_context_len=0):
        """split too-long exm into two or more valid sub exms"""
        segmented_exm_lst = []

        while True:  # 切分成 符合长度的头 和 剩余的尾
            prev_exm, rear_exm = NerExample._segment_exm(exm, max_size=max_size, prefix_context_len=prefix_context_len)
            segmented_exm_lst.append(prev_exm)
            if rear_exm is None:
                break
            else:
                exm = rear_exm

        return segmented_exm_lst

    def remove_ent_by_type(self, ent_type_lst, input_keep=False):
        # input_keep True 仅保留ent_type_lst中的
        # input_keep False 去除在ent_type_lst中的
        ent_types = list(self.ent_dct.keys())  # 目前已有的
        if input_keep:  # 输入的列表是要保存的
            for ent_type in ent_types:
                if ent_type not in ent_type_lst:
                    self.ent_dct.pop(ent_type)
        else:  # 输入的列表是要去除的
            for ent_type in ent_types:
                if ent_type in ent_type_lst:
                    self.ent_dct.pop(ent_type)

        self.update(anchor='ent_dct')

    def filter_ent_by_startswith(self, prefix, mode='keep'):
        assert mode in ['keep', 'remove']  # 保留/去除 指定前缀的实体
        ent_types = list(self.ent_dct.keys())  # 目前已有的
        for ent_type in ent_types:
            if mode == 'keep' and not ent_type.startswith(prefix):  # 保留指定前缀的实体
                self.ent_dct.pop(ent_type)
            if mode == 'remove' and ent_type.startswith(prefix):  # 去除指定前缀的实体
                self.ent_dct.pop(ent_type)
        self.update(anchor='ent_dct')

    def get_filter_ent_dct_by_startswith(self, prefix, mode='keep'):
        assert mode in ['keep', 'remove']  # 保留/去除 指定前缀的实体
        ent_types = list(self.ent_dct.keys())  # 目前已有的
        ent_dct = copy.deepcopy(self.ent_dct)
        for ent_type in ent_types:
            if mode == 'keep' and not ent_type.startswith(prefix):  # 保留指定前缀的实体
                ent_dct.pop(ent_type)
            if mode == 'remove' and ent_type.startswith(prefix):  # 去除指定前缀的实体
                ent_dct.pop(ent_type)
        return ent_dct

    def add_ent(self, ent_type, start, end):
        if ent_type in self.ent_dct:
            if [start, end] not in self.ent_dct[ent_type]:
                self.ent_dct[ent_type].append([start, end])
        else:
            self.ent_dct[ent_type] = [[start, end]]

    def add_ent_dct(self, new_ent_dct):
        for new_ent_type in new_ent_dct:
            if new_ent_type not in self.ent_dct:
                self.ent_dct[new_ent_type] = new_ent_dct[new_ent_type]
            else:
                for start, end in new_ent_dct[new_ent_type]:
                    if [start, end] not in self.ent_dct[new_ent_type]:
                        self.ent_dct[new_ent_type].append([start, end])

    def __str__(self):
        return self.to_json_str(val_at_end=True)

    # def __repr__(self):
    #     return str(self.__dict__)

    def __repr__(self):
        return self.__str__()

    def to_json_str_old(self, for_human_read=False, external_attrs=None):
        json_dct = {'text': self.text,
                    'ent_dct': self.ent_dct,
                    'char_lst': self.char_lst,
                    }
        if hasattr(self, 'pred_ent_dct'):
            json_dct['pred_ent_dct'] = self.pred_ent_dct
        if hasattr(self, 'cls_tgt'):
            json_dct['cls_tgt'] = self.cls_tgt

        if for_human_read:  # 把ent_value表示出来
            json_dct.pop('char_lst')
            json_dct['ent_dct'] = self.get_detail_ent_dct()
            if hasattr(self, 'pred_ent_dct'):
                pred_ent_dct = copy.deepcopy(self.pred_ent_dct)
                # get detail
                for pos_lst in pred_ent_dct.values():
                    for pos in pos_lst:
                        start, end, *_ = pos
                        pos.append(self.get_text(start, end))
                json_dct['pred_ent_dct'] = pred_ent_dct

        if external_attrs is not None:
            for attr in external_attrs:
                if hasattr(self, attr):
                    json_dct[attr] = getattr(self, attr)

        return json.dumps(json_dct, ensure_ascii=False)

    def to_json_str(self, val_at_end=True, only_pred_str=False, flat_pred_ent=False, external_attrs=None, for_human_read=True):
        """ val_at_end = True: 省略end直接将ent输出
            only_pred_str = True  # remove pred_score and other
        """
        json_dct = {'text': self.text, 'ent_dct': copy.deepcopy(self.ent_dct)}
        for ent, v_lst in json_dct['ent_dct'].items():
            for v in v_lst:
                if val_at_end:
                    ent_str = self.get_text(v[0], v[1])
                v[1] = ent_str  # 省略end直接将ent输出
        if hasattr(self, 'pred_ent_dct'):
            if flat_pred_ent:
                pred_ent_dct = self.get_flat_pred_ent_dct()
            else:
                pred_ent_dct = copy.deepcopy(self.pred_ent_dct)
            json_dct['pred_ent_dct'] = pred_ent_dct
            for ent, v_lst in json_dct['pred_ent_dct'].items():
                for i, v in enumerate(v_lst):
                    s, e, *_ = v
                    if val_at_end:
                        ent_str = self.get_text(s, e)
                        v[1] = ent_str
                    if isinstance(v[2], float):  # round 3 if pred_score
                        v[2] = round(v[2], 3)
                    if only_pred_str:  # remove pred_score and other
                        v_lst[i] = v[:2]
        if hasattr(self, 'cls_tgt'):
            json_dct['cls_tgt'] = self.cls_tgt
        if external_attrs is not None:
            for attr in external_attrs:
                if hasattr(self, attr):
                    json_dct[attr] = getattr(self, attr)

        return json.dumps(json_dct, ensure_ascii=False)

    def get_detail_ent_dct(self):
        # 把ent_value表示出来
        ent_dct = copy.deepcopy(self.ent_dct)
        for pos_lst in ent_dct.values():
            for pos in pos_lst:
                start, end, *_ = pos  # 防止除了start end信息还有其他
                pos.append(self.get_text(start, end))
        return ent_dct

    def __eq__(self, other):
        if self.char_lst == other.char_lst and self.ent_dct == other.ent_dct:
            return True
        else:
            return False

    def __hash__(self):
        ent_lst = self.get_ent_lst()
        # ent_lst_set = set([(ent_type, start, end) for ent_type, start, end in ent_lst])
        return hash(''.join(self.char_lst)) + hash(str(sorted(ent_lst)))

    def is_neg(self):
        if not self.ent_dct:
            return True
        else:
            return False

    def has_ent_type(self, ent_type):
        return ent_type in self.ent_dct

    def has_ent_types(self, ent_types: List, constrained='all'):
        if constrained == 'all':  # need contain all ent in passing ent_types
            for ent_type in ent_types:
                if not self.has_ent_type(ent_type):
                    return False
            return True
        elif constrained == 'any':  # just contain one of ent in passing ent_types is OK
            for ent_type in ent_types:
                if self.has_ent_type(ent_type):
                    return True
            return False
        else:
            raise NotImplementedError

    def has_ent_type_startswith(self, ent_type_prefix):
        for ent_type in self.ent_dct:
            if ent_type.startswith(ent_type_prefix):
                return True
        return False

    def check_valid(self):
        if self.tag_lst is not None:
            if not len(self.char_lst) == len(self.tag_lst): return False
        return True

    @classmethod
    def from_tag_lst(cls, char_lst, tag_lst, token_deli='', schema='IOB'):
        assert len(char_lst) == len(tag_lst)
        ent_dct, ent_span_dct = NerExample.extract_entity_by_tags(tag_lst, schema=schema)
        return cls(char_lst, ent_dct, ent_span_dct=ent_span_dct, tag_lst=tag_lst, token_deli=token_deli)

    @classmethod
    def from_ent_lst(cls, char_lst, ent_lst):
        # ent_lst [[ent_type, start, end, *_],..] start end还可能是str
        ent_dct = NerExample.ent_lst_to_ent_dct(ent_lst)
        return cls(char_lst, ent_dct)

    @staticmethod
    def ent_lst_to_ent_dct(ent_lst):
        ent_dct = defaultdict(list)
        for ent_type, start, end, *other in ent_lst:
            ent_dct[ent_type].append([int(start), int(end)] + other)
        return dict(ent_dct)

    def get_ent_lst(self, for_human=False):
        ent_lst = []  # [[ent_type, start, end],..]
        for ent_type, pos_lst in self.ent_dct.items():
            for pos in pos_lst:  # pos: [start, end, *_]
                start, end = pos[0], pos[1]
                new_pos = [ent_type] + pos
                if for_human:
                    new_pos += [self.get_text(start, end)]
                ent_lst.append(new_pos)
        return ent_lst

    def get_pred_ent_lst(self, for_human=False):
        ent_lst = []  # [[ent_type, start, end],..]
        if hasattr(self, 'pred_ent_dct'):
            for ent_type, pos_lst in self.pred_ent_dct.items():
                for start, end, prob in pos_lst:
                    if for_human:
                        ent_lst.append([ent_type, start, end, prob, self.get_text(start, end)])
                    else:
                        ent_lst.append([ent_type, start, end, prob])
        return ent_lst

    def get_tag_lst(self, schema='BIO', combine_stragety='prev'):
        return NerExample.to_tag_lst(self.char_lst, self.ent_dct, schema=schema, combine_stragety=combine_stragety)

    @staticmethod
    def assign_ent_to_tag_lst(tag_lst, ent_type, start, end, schema='BIO'):
        assert schema == 'BIO'  # TODO
        assert start >= 0 and end <= len(tag_lst) and end > start, f'start:{start} end:{end} tag_lst:{tag_lst}'
        tag_lst[start:end] = [f'B-{ent_type}'] + [f'I-{ent_type}'] * (end - start - 1)

    @staticmethod
    def to_tag_lst(char_lst, ent_dct, schema='BIO', combine_stragety='prev'):
        """
        目前合并策略仅支持:
        prev: 谁先出现保留谁的 同样位置B优先保留长的
        max: 谁最长保留谁的
        """
        tag_lst = ['O'] * len(char_lst)
        assert schema == 'BIO' and combine_stragety == 'prev'

        ent_lst = []  # [[ent_type, start, end],..]
        for ent_type, pos_lst in ent_dct.items():
            for start, end, *_ in pos_lst:
                ent_lst.append([ent_type, start, end])
        ent_lst.sort(key=lambda e: e[2], reverse=True)  # 先把end按照大到小排好 相同B保留长的
        ent_lst.sort(key=lambda e: e[1])  # 按start来排序

        prev_end = 0  # 开区间
        for ent_type, start, end in ent_lst:
            if start >= prev_end:  # 引文end是闭区间 所以可以等于
                # 标注该entity
                NerExample.assign_ent_to_tag_lst(tag_lst, ent_type, start, end, schema=schema)
                # tag_lst[start:end] = [f'B-{ent_type}'] + [f'I-{ent_type}'] * (end - start - 1)
                # tag_lst[start:end] = ['O'] * len(end - start)
                prev_end = end  # 记录此entity结尾坐标
        return tag_lst

    def to_tag_lst_by_pred(self):
        """
        根据pred_ent_dct中的概率来决定嵌套实体如何展开成平展实体
        优先分数(概率)大的。
        分数一样则优先长的
        """
        if not hasattr(self, 'pred_ent_dct'):
            print('to_tag_lst_by_pred method should have attribute pred_ent_dct')
            return None
        ent_lst = []
        for ent_type, v_lst in self.pred_ent_dct.items():
            for v in v_lst:
                start = v[0]
                end = v[1]
                if len(v) == 2:  # no prob score
                    prob = end - start  # len as score
                else:
                    prob = v[2]
                ent_lst.append([ent_type, start, end, prob])
        ent_lst.sort(key=lambda e: e[3], reverse=True)

        tag_lst = ['O'] * len(self.char_lst)
        for ent_type, start, end, prob in ent_lst:
            if set(tag_lst[start:end]) == {'O'}:  # 都没有分配到实体
                NerExample.assign_ent_to_tag_lst(tag_lst, ent_type, start, end)

        return tag_lst

    def get_flat_pred_ent_dct(self):
        """
        根据pred_ent_dct中的概率来决定嵌套实体如何展开成平展实体，并覆盖回原pred_ent_dct
        优先概率大的。
        """
        prob_dct = {}  # 为了保留pred分数
        for ent_type, pos_lst in self.pred_ent_dct.items():
            for start, end, prob in pos_lst:
                prob_dct[(ent_type, start, end)] = prob

        flat_pred_ent_dct, _ = NerExample.extract_entity_by_tags(self.to_tag_lst_by_pred())
        for ent_type, pos_lst in flat_pred_ent_dct.items():
            for pos in pos_lst:
                pos.append(prob_dct[(ent_type, pos[0], pos[1])])  # 把pred分数映射回去
        return flat_pred_ent_dct
        # self.pred_ent_dct = flat_pred_ent_dct

    @staticmethod
    def Flat_ent_dct_by_prob(ent_dct, char_lst_len, set_prob=None):
        ent_dct = copy.deepcopy(ent_dct)
        if set_prob is not None:
            for ent, v_lst in ent_dct.items():
                for v in v_lst:
                    v.append(set_prob)
        prob_dct = {}  # 为了保留pred分数
        ent_lst = []
        for ent, v_lst in ent_dct.items():
            for start, end, prob in v_lst:
                prob_dct[(ent, start, end)] = prob
                ent_lst.append([ent, start, end, prob, end - start])

        ent_lst.sort(key=lambda e: e[4], reverse=True)  # 如果分数相同优先长的
        ent_lst.sort(key=lambda e: e[3], reverse=True)  # 先判断分数

        tag_lst = ['O'] * char_lst_len
        for ent, start, end, prob, l in ent_lst:
            if set(tag_lst[start:end]) == {'O'}:  # 都没有分配到实体
                NerExample.assign_ent_to_tag_lst(tag_lst, ent, start, end)

        flat_ent_dct, _ = NerExample.extract_entity_by_tags(tag_lst)

        for ent, v_lst in flat_ent_dct.items():  # 把pred分数映射回去
            for v in v_lst:
                v.append(prob_dct[(ent, v[0], v[1])])

        return flat_ent_dct

    def flat_ent_dct(self):
        """
        效果与to_tag_lst一样
        目前合并策略仅支持:
        prev: 谁先出现保留谁的 同样位置B优先保留长的
        max: 谁最长保留谁的
        """
        ent_dct, _ = NerExample.extract_entity_by_tags(NerExample.to_tag_lst(self.char_lst, self.ent_dct))
        self.ent_dct = ent_dct

    def duplicate_ent_dct(self):
        """去重ent deduplication"""
        ent_dct = {}
        for ent_type, pos_lst in self.ent_dct.items():
            dupli_pos_lst = []
            pos_set = set()
            for pos in pos_lst:
                if (pos[0], pos[1]) not in pos_set:
                    dupli_pos_lst.append(pos)
                    pos_set.add((pos[0], pos[1]))
            ent_dct[ent_type] = dupli_pos_lst
        self.ent_dct = ent_dct

    def remove_invalid_ent_dct(self):
        """去掉start和end不满足end>start的ent"""
        for ent_type, pos_lst in self.ent_dct.items():
            self.ent_dct[ent_type] = [pos for pos in pos_lst if pos[1] > pos[0]]  # end > start

    @staticmethod
    def ent_dct_to_ent_span_dct(ent_dct):
        ent_span_dct = {}
        for ent_type, pos_lst in ent_dct.items():
            for start, end, *_ in pos_lst:
                ent_span_dct[(start, end)] = ent_type
        return ent_span_dct

    @staticmethod
    def ent_span_dct_to_ent_dct(ent_span_dct):
        ent_dct = defaultdict(list)
        for (start, end), ent_type in ent_span_dct.items():
            ent_dct[ent_type].append([start, end])
        return dict(ent_dct)

    def update(self, anchor='ent_dct', dedup=True):
        """
        anchor: 以哪个作为标准重新生成标签实体字段 ent_dct|ent_span_dct|tag_lst
        """
        if anchor == 'ent_dct':
            if dedup:
                self.duplicate_ent_dct()  # 去重
            self.tag_lst = NerExample.to_tag_lst(self.char_lst, self.ent_dct)
            self.ent_span_dct = NerExample.ent_dct_to_ent_span_dct(self.ent_dct)
        elif anchor == 'tag_lst':
            self.ent_dct, self.ent_span_dct = NerExample.extract_entity_by_tags(self.tag_lst)
        # elif anchor == 'ent_span_dct':
        #     self.ent_dct = NerExample.ent_span_dct_to_ent_dct(self.ent_span_dct)
        #     self.tag_lst = NerExample.to_tag_lst(self.char_lst, self.ent_dct)
        else:
            raise NotImplementedError

    def ent_type_convert(self, ent2ent_map: Union[dict, Callable], default='None'):
        """ ent_type 必须是字符类型 """
        ent_dct = {}
        if isinstance(ent2ent_map, dict):
            ent2ent_map = {str(k): str(v) for k, v in ent2ent_map.items()}
            for old_ent, v in self.ent_dct.items():
                if old_ent in ent2ent_map:  #
                    ent_dct[ent2ent_map[old_ent]] = v
                else:
                    ent_dct[old_ent] = v

        elif hasattr(ent2ent_map, '__call__'):
            for old_ent, v in self.ent_dct.items():
                ent_dct[ent2ent_map(old_ent)] = v
        else:
            raise NotImplementedError
        # print(self.ent_dct)
        # print(ent_dct)
        self.ent_dct = ent_dct
        self.update(anchor='ent_dct')

    @staticmethod
    def extract_entity_by_tags(tag_lst, schema='IOB'):
        """ 根据tags获得有意义的entity并返回对应start-end索引 end开区间
        @return: ent_dct: {1:[[0,2], [4,7]], 2:[[9,12]], ...}  键是ent对应的id  [start,end)形式 不包括end
        @return: ent_span_dct: {(0,2):1, (4,7):1, ...}
        """
        ent_dct, ent_span_dct = defaultdict(list), {}
        curr_ent_id, curr_ids = '', []

        def add_to_dct(curr_ent_id, curr_ids):
            curr_span = [curr_ids[0], curr_ids[-1] + 1]  # end + 1
            ent_dct[curr_ent_id].append(curr_span)
            ent_span_dct[tuple(curr_span)] = curr_ent_id

        if schema == 'IOB':
            for idx, tag in enumerate(tag_lst):
                if tag.startswith('B'):
                    # 要把之前识别到的ent放进去
                    if curr_ent_id or curr_ids:
                        add_to_dct(curr_ent_id, curr_ids)
                        curr_ent_id, curr_ids = '', []
                    curr_ent_id = tag[2:]
                    curr_ids.append(idx)

                elif tag.startswith('I'):
                    ent_id = tag[2:]
                    if ent_id == curr_ent_id:
                        curr_ids.append(idx)

                else:
                    if curr_ent_id or curr_ids:
                        add_to_dct(curr_ent_id, curr_ids)
                        curr_ent_id, curr_ids = '', []

            if curr_ent_id or curr_ids:  # 记住结尾要进行处理
                add_to_dct(curr_ent_id, curr_ids)

        elif schema == 'IO':
            for idx, tag in enumerate(tag_lst):
                if tag != 'O':
                    ent_id = tag
                    if not curr_ent_id:  # 新实体
                        curr_ent_id = ent_id
                        curr_ids.append(idx)
                    else:
                        if ent_id != curr_ent_id:  # 新实体 前面也是实体
                            add_to_dct(curr_ent_id, curr_ids)
                            curr_ent_id, curr_ids = '', []
                            curr_ent_id = ent_id
                            curr_ids.append(idx)
                        else:  # 原来的实体
                            curr_ent_id = ent_id
                            curr_ids.append(idx)
                else:  # 'O'
                    if curr_ent_id:
                        add_to_dct(curr_ent_id, curr_ids)
                        curr_ent_id, curr_ids = '', []

            if curr_ent_id:  # 记住结尾要进行处理
                add_to_dct(curr_ent_id, curr_ids)

        else:
            raise NotImplementedError
        return ent_dct, ent_span_dct

    @staticmethod
    def save_to_col_format_file(ner_exm_lst: List, output_file):
        make_sure_dir_exist(output_file)
        with open(output_file, 'w', encoding='U8') as f:
            for ner_exm in ner_exm_lst:
                for char, tag in zip(ner_exm.char_lst, ner_exm.tag_lst):
                    f.write(f'{char}\t{tag}\n')
                f.write('\n')  # 样本分割符

    @staticmethod
    def get_from_col_format_file(input_file, deli='\t'):
        """ 读取行 {char}\t{tag}\n 格式数据  \n换行符分割样本
            tag可能有多个 真实和预测 {char}\t{tag}\t{tag}\n 格式数据  \n换行符分割样本
            转为char_lst, tag_lst
            deli为 char 与 tag 的分隔符
        """
        example_lst = []

        with open(input_file, 'r', encoding='U8') as f:
            lines = f.readlines()
        items = [l.strip().split(deli) for l in lines]

        curr_char_lst = []
        curr_tag_lst = []
        for item in items:
            if len(item) == 1:  # 分隔标志 ['']
                if curr_char_lst and curr_tag_lst:
                    example_lst.append(NerExample.from_tag_lst(curr_char_lst, curr_tag_lst))
                    curr_char_lst, curr_tag_lst = [], []
                continue
            curr_char_lst.append(item[0])
            curr_tag_lst.append(item[1])
        if curr_char_lst and curr_tag_lst:
            example_lst.append(NerExample.from_tag_lst(curr_char_lst, curr_tag_lst))
        return example_lst

    @staticmethod
    def save_to_jsonl_old(ner_exm_lst: List, output_file, for_human_read=False, external_attrs=None, overwrite=True):
        make_sure_dir_exist(output_file)
        if os.path.exists(output_file) and not overwrite:
            return
        with open(output_file, 'w', encoding='U8') as f:
            for exm in ner_exm_lst:
                f.write(exm.to_json_str(for_human_read=for_human_read, external_attrs=external_attrs) + '\n')

    @staticmethod
    def save_to_jsonl(ner_exm_lst: List, output_file, val_at_end=True, only_pred_str=False, flat_pred_ent=False, external_attrs=None, overwrite=True,
                      for_human_read=True):
        """ val_at_end = True: 省略end直接将ent输出
            only_pred_str = True  # remove pred_score and other
        """
        make_sure_dir_exist(output_file)
        if os.path.exists(output_file) and not overwrite:
            return
        with open(output_file, 'w', encoding='U8') as f:
            for exm in ner_exm_lst:
                f.write(exm.to_json_str(val_at_end=val_at_end, only_pred_str=only_pred_str, flat_pred_ent=flat_pred_ent, external_attrs=external_attrs) + '\n')

    @staticmethod
    def load_from_jsonl(jsonl_file, token_deli='', external_attrs=None, dedup=True, eid=True):
        """ dedup: deduplication ents"""
        with open(jsonl_file, 'r', encoding='U8') as f:
            obj_lst = [json.loads(line.strip()) for line in f]
        exm_lst = []
        for obj in obj_lst:
            if 'char_lst' in obj:
                char_lst = obj['char_lst']
            else:
                if 'text' in obj:  # 只有text
                    if token_deli == '':
                        char_lst = list(obj['text'])
                    else:
                        char_lst = obj['text'].split(token_deli)
                else:
                    raise Exception('there should exist either of char_lst or text field')

            ent_dct = obj['ent_dct']
            for k, pos_lst in ent_dct.items():
                for pos in pos_lst:
                    start = pos[0]
                    if isinstance(pos[1], str):  # 精简模式 start, mention
                        mention = pos[1]
                        end = start
                        while len(token_deli.join(char_lst[start: end])) < len(mention):
                            # while token_deli.join(char_lst[start: end]) != mention:  # 当mention有误时会死循环
                            end += 1
                        # end = start + len(mention)  # 仅对于字符级样本有用
                        pos[1] = end
                    if isinstance(pos[-1], str) and pos[-1] == token_deli.join(char_lst[pos[0]:pos[1]]):  # 如果最后是4h的text 则去掉
                        pos.pop(-1)

            exm = NerExample(char_lst=char_lst, ent_dct=ent_dct, token_deli=token_deli, dedup=dedup)
            exm.remove_invalid_ent_dct()
            exm.update(anchor='ent_dct', dedup=dedup)

            if 'pred_ent_dct' in obj:
                pred_ent_dct = obj['pred_ent_dct']
                for k, pos_lst in pred_ent_dct.items():
                    for pos in pos_lst:
                        if isinstance(pos[1], str):  # 精简模式 start, mention
                            mention = pos[1]
                            end = start
                            while len(token_deli.join(char_lst[start: end])) < len(mention):
                                end += 1
                            # end = start + len(mention)  # 仅对于字符级样本有用
                            pos[1] = end
                        if isinstance(pos[-1], str) and pos[-1] == token_deli.join(char_lst[pos[0]:pos[1]]):  # 如果最后是4h的text 则去掉
                            pos.pop(-1)
                exm.pred_ent_dct = pred_ent_dct

            if external_attrs is not None:
                [exm.__setattr__(attr, obj[attr]) for attr in external_attrs if attr in obj]
            if eid:
                exm.eid = len(exm_lst)
            exm_lst.append(exm)

        return exm_lst

    @staticmethod
    def load_from_jsonl_4h(jsonl_file):
        return NerExample.load_from_jsonl(jsonl_file)

    @staticmethod
    def combine_by_text(exm_lst):
        """根据text来合并样本
           主要是合并ent_dct
        """
        # 去重和合并标签
        text2exms_dct = defaultdict(list)
        for exm in exm_lst:
            text2exms_dct[exm.text].append(exm)

        for text, exm_lst in text2exms_dct.items():
            stand_exm = exm_lst[0]
            for exm in exm_lst[1:]:
                stand_exm.add_ent_dct(exm.ent_dct)

            text2exms_dct[text] = stand_exm

        return list(text2exms_dct.values())

    def get_span_level_ner_tgt_lst(self, neg_symbol: str = 'O') -> List:
        # TODO 每个span只能属于一个ent
        """
        只有上三角
        tags:  O, O, B1, I1, O
        span_index_lst:  [(0,0)-(0,1)-(0,2)-(0,3)-(0,4)-(1,1)-(1,2)-(1,3)-(1,4)-(2,2)-(2,3)-(2,4)-(3,3)-(3,4)-(4,4)]
        span_index_lst_len:  n * (n+1) / 2 = 15  (n=5)
        option: ent_len_lst 可选，用以存储实体长度，方便统计平均实体长度
        @return: span_ner_tgt_lst: [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,]
        """
        if self.ent_span_dct is None:
            self.update(anchor='ent_dct')
        span_ner_tgt_lst = []
        # length = len(self.ori_char_lst) if hasattr(self, 'ori_char_lst') else len(self.char_lst)  # TODO
        length = len(self.char_lst)
        for i in range(length):
            for j in range(i, length):
                span_ner_tgt_lst.append(self.ent_span_dct.get((i, j + 1), neg_symbol))  # 要记得加1，因为end是闭区间
        # if ret == 'int':
        #     span_ner_tgt_lst = [int(e) if isinstance(e, str) and e.isdigit() else e for e in span_ner_tgt_lst]
        return span_ner_tgt_lst

    def get_link_lst(self) -> List:
        link_lst = [0.] * (len(self.char_lst) - 1)
        for ent, v_lst in self.ent_dct.items():
            for s, e, *_ in v_lst:
                for i in range(s, e - 1):
                    link_lst[i] = 1.
        return link_lst

    @staticmethod
    def from_span_level_ner_tgt_lst(span_ner_tgt_lst: List, length: int, id2ent: dict, negative_set=None):
        """
        上三角
        span_ner_tgt_lst: 2维 [len*(len+1)//2, num_label]
        """
        assert len(span_ner_tgt_lst) == length * (length + 1) // 2
        if negative_set is None: negative_set = {0, 1}
        ent_lst = []

        pred_ent_ids = np.argmax(span_ner_tgt_lst, -1).tolist()  # [len*(len+1)//2]
        pred_ent_probs = np.max(span_ner_tgt_lst, -1).tolist()  # [len*(len+1)//2]
        span_index_lst = [(i, j) for i in range(length) for j in range(i, length)]  # (0,0)-(0,1)-(0,2)-(0,3)-(0,4)...

        for pred_ent_id, pred_ent_prob, (start, end) in zip(pred_ent_ids, pred_ent_probs, span_index_lst):
            if pred_ent_id in negative_set:
                continue
            ent_lst.append([pred_ent_id, start, end + 1, pred_ent_prob])

        pred_ent_dct = defaultdict(list)
        for pred_ent_id, start, end, pred_ent_prob in ent_lst:
            pred_ent_dct[id2ent[pred_ent_id]].append([start, end, pred_ent_prob])

        return dict(pred_ent_dct)

    @staticmethod
    def from_span_level_ner_tgt_lst3(span_ner_tgt_lst: List, length: int, id2ent: dict, negative_set=None):
        """
        获取span实体分数topk的
        """
        assert len(span_ner_tgt_lst) == length * (length + 1) // 2
        if negative_set is None: negative_set = {0, 1}
        ent_lst = []
        top_k = 3
        top_k_pred_ent_ids = np.argsort(-span_ner_tgt_lst, -1)[:, :top_k]  # [len*(len+1)//2, top_k]
        top_k_pred_ent_probs = np.take_along_axis(span_ner_tgt_lst, top_k_pred_ent_ids, axis=-1)  # [len*(len+1)//2, top_k]
        span_index_lst = [(i, j) for i in range(length) for j in range(i, length)]  # (0,0)-(0,1)-(0,2)-(0,3)-(0,4)...

        for top_k_pred_ent_id, top_k_pred_ent_prob, (start, end) in zip(top_k_pred_ent_ids.tolist(), top_k_pred_ent_probs.tolist(), span_index_lst):
            for pred_ent_id, pred_ent_prob in zip(top_k_pred_ent_id, top_k_pred_ent_prob):
                if pred_ent_id not in negative_set and end + 1 - start > 1:  # 长度大于1
                    if pred_ent_prob > 0.1:
                        ent_lst.append([pred_ent_id, start, end + 1, pred_ent_prob])
                    # ent_lst.append([pred_ent_id, start, end + 1, pred_ent_prob])

        pred_ent_dct = defaultdict(list)
        for pred_ent_id, start, end, pred_ent_prob in ent_lst:
            pred_ent_dct[id2ent[pred_ent_id]].append([start, end, pred_ent_prob])

        return dict(pred_ent_dct)

    @staticmethod
    def from_span_level_ner_tgt_lst2(span_ner_tgt_lst: List, length: int, id2ent: dict, negative_set=None):
        """
        获取span实体分数大于某个阈值的
        """
        assert len(span_ner_tgt_lst) == length * (length + 1) // 2
        if negative_set is None: negative_set = {0, 1}  # PAD O

        ent_lst = []
        # pred_ent_ids = np.argmax(span_ner_tgt_lst, -1).tolist()  # [len*(len+1)//2]
        # pred_ent_probs = np.max(span_ner_tgt_lst, -1).tolist()  # [len*(len+1)//2]

        thres = 0.5
        span_ner_tgt_lst = np.array(span_ner_tgt_lst)
        span_index_lst = [(i, j) for i in range(length) for j in range(i, length)]  # (0,0)-(0,1)-(0,2)-(0,3)-(0,4)...
        indices = list(zip(*np.where(span_ner_tgt_lst > thres)))  # [(0,2),(0,3),(1,2)...]
        for span_id, pred_ent_id in indices:
            if pred_ent_id not in negative_set:
                start, end = span_index_lst[span_id]
                ent_lst.append([pred_ent_id, start, end + 1, span_ner_tgt_lst[span_id, pred_ent_id]])

        pred_ent_dct = defaultdict(list)
        for pred_ent_id, start, end, pred_ent_prob in ent_lst:
            pred_ent_dct[id2ent[pred_ent_id]].append([start, end, pred_ent_prob])

        return dict(pred_ent_dct)

    @staticmethod
    def from_span_level_ner_tgt_lst_sigmoid(span_ner_tgt_lst, length: int, id2ent: dict, threshold=0.5):
        """
        上三角
        span_ner_tgt_lst: 2维 [len*(len+1)//2, num_label]
        """
        assert len(span_ner_tgt_lst) == length * (length + 1) // 2
        ent_lst = []

        # pred_ent_probs = span_ner_tgt_lst.tolist()  # [len*(len+1)//2, num_ent]  >0就是存在实体 <0不存在实体  sigmoid后则是>0.5 < 0.5
        # [len*(len+1)//2, num_ent]  >0就是存在实体 <0不存在实体  sigmoid后则是>0.5 < 0.5
        span_index_lst = [(i, j) for i in range(length) for j in range(i, length)]  # (0,0)-(0,1)-(0,2)-(0,3)-(0,4)...

        # for pred_ent_prob, (start, end) in zip(pred_ent_probs, span_index_lst):
        #     for ent_id, ent_pred_ent_prob in enumerate(pred_ent_prob):
        #         if ent_pred_ent_prob >= 0.5:
        #             ent_lst.append([ent_id, start, end + 1, ent_pred_ent_prob])

        pred_ent_dct = defaultdict(list)
        pred_ent_probs = np.array(span_ner_tgt_lst)  # [num_span, num_ent]
        for span_idx, ent_id in zip(*np.where(pred_ent_probs >= threshold)):
            start, end = span_index_lst[span_idx]
            pred_ent_prob = pred_ent_probs[span_idx, ent_id]
            pred_ent_dct[id2ent[ent_id]].append([int(start), int(end) + 1, float(pred_ent_prob)])

        # pred_ent_dct = defaultdict(list)
        # for pred_ent_id, start, end, pred_ent_prob in ent_lst:
        #     pred_ent_dct[id2ent[pred_ent_id]].append([start, end, pred_ent_prob])

        return dict(pred_ent_dct)

    def get_conj_info(self, conj_scores: List, decimal=None):
        # conj_res: tok1 conj_score1 tok2 conj_score2...
        # char_lst = self.ori_char_lst if hasattr(self, 'ori_char_lst') else self.char_lst  # TODO
        if decimal is not None:
            conj_scores = [round(float(e), decimal) for e in conj_scores]
        valid_score_len = len(self.char_lst) - 1
        conj_scores = conj_scores[:valid_score_len]
        assert len(conj_scores) == valid_score_len
        conj_res = []
        for i in range(valid_score_len):
            conj_res.append(self.char_lst[i])
            conj_res.append(conj_scores[i])
        conj_res.append(self.char_lst[-1])
        return conj_scores, conj_res

    @staticmethod
    def get_span_conj_score(conj_scores, start, end, mode='min'):
        # start 开区间 end 闭区间
        s = start
        e = end - 1
        conj_scores = conj_scores[s:e]
        if len(conj_scores) > 0:
            if mode == 'mean':
                span_conj_score = np.mean(conj_scores)
            elif mode == 'sum':
                span_conj_score = np.sum(conj_scores)
            elif mode == 'max':
                span_conj_score = np.max(conj_scores)
            elif mode == 'min':
                span_conj_score = np.min(conj_scores)
            else:
                raise NotImplementedError
            span_conj_score = float(span_conj_score)
        else:  # 只有1个token
            span_conj_score = 0.
        return span_conj_score

    def get_flat_pred_ent_dct_by_conj_scores(self, conj_scores):
        # char_lst = self.ori_char_lst if hasattr(self, 'ori_char_lst') else self.char_lst  # TODO
        pred_ent_lst = []
        for ent_type, pos_lst in self.pred_ent_dct.items():
            for pos in pos_lst:
                start, end, prob = pos[:3]
                span_conj_score = NerExample.get_span_conj_score(conj_scores, start, end)
                pos.append(span_conj_score)
                finnal_score = prob + span_conj_score
                pos.append(finnal_score)
                pred_ent_lst.append([ent_type, start, end, prob, span_conj_score, finnal_score])
        pred_ent_lst.sort(key=lambda e: e[5], reverse=True)
        assigned_flags = [False for i in range(len(self.char_lst))]
        new_pred_ent_lst = []
        for ent_type, start, end, prob, span_conj_score, finnal_score in pred_ent_lst:
            if True in assigned_flags[start:end]:
                continue
            new_pred_ent_lst.append([ent_type, start, end, prob, span_conj_score, finnal_score])
            assigned_flags[start:end] = [True] * (end - start)
        new_pred_ent_dct = {}
        for ent_type, start, end, prob, span_conj_score, finnal_score in new_pred_ent_lst:
            new_pred_ent_dct.setdefault(ent_type, []).append([start, end, prob, span_conj_score, finnal_score])
        return new_pred_ent_dct

    def is_ent_overlap(self):
        ent_lst = self.get_ent_lst(for_human=True)
        ent_lst.sort(key=lambda e: e[2], reverse=True)  # end从大到小
        ent_lst.sort(key=lambda e: e[1], reverse=False)  # start从小到大
        self.ent_lst = ent_lst
        prev_end = 0
        for ent_type, start, end, ment in ent_lst:
            if start < prev_end:
                return True
            else:
                prev_end = end
        return False

    @staticmethod
    def get_ent_type_set(exm_lst, out_file=None):
        ent_type_set = set()
        for exm in exm_lst:
            for k in exm.ent_dct:
                ent_type_set.add(k)
        ent_type_set = list(ent_type_set)
        if out_file is not None:
            list2file(ent_type_set, out_file)
        return ent_type_set

    @staticmethod
    def eval(exm_lst, macro=False, verbose=True, use_flat_pred_ent_dct=False, anal_exm_out_file=None, anal_exm_out_file_external_attrs=None, use_table=True):
        """
        # 原始的通过conll计算
        # total_true_tags = [utils.NerExample.to_tag_lst(exm.char_lst, exm.ent_dct) for exm in exm_lst]
        # total_pred_tags = [exm.to_tag_lst_by_pred() for exm in exm_lst]
        # p, r, f = evaluate(sum(total_true_tags, []), sum(total_pred_tags, []), verbose=True)
        现在替换成这个了use_flat_pred_ent_dct=True
        """
        detail_info_str = ''
        detail_stat = defaultdict(lambda: {
            'tp': 0.,
            'fp': 0.,
            'fn': 0.,
        })

        anal_exm = defaultdict(lambda: {'fp': [], 'fn': []})
        tp = 0
        fp = 0
        fn = 0
        for exm in exm_lst:
            if use_flat_pred_ent_dct:  # 使用平展(后处理)过的实体来评估，则要对调一下属性
                exm.ori_pred_ent_dct = exm.pred_ent_dct
                exm.pred_ent_dct = exm.get_flat_pred_ent_dct()

            gold_ent_lst = set()
            for ent_type, pos_lst in exm.ent_dct.items():
                for start, end, *other in pos_lst:
                    gold_ent_lst.add((ent_type, start, end))

            pred_ent_lst = set()
            for ent_type, pos_lst in exm.pred_ent_dct.items():
                for start, end, prob, *other in pos_lst:
                    pred_ent_lst.add((ent_type, start, end))

            # 总体
            union = gold_ent_lst | pred_ent_lst
            intersect = gold_ent_lst & pred_ent_lst
            tp += len(intersect)
            fp += len(pred_ent_lst) - len(intersect)
            fn += len(gold_ent_lst) - len(intersect)

            # 按ent_type来
            ent_type_lst = set([e[0] for e in union])
            for ent_type in ent_type_lst:
                gold_ent_lst_ = set([e for e in gold_ent_lst if e[0] == ent_type])
                pred_ent_lst_ = set([e for e in pred_ent_lst if e[0] == ent_type])
                intersect_ = gold_ent_lst_ & pred_ent_lst_
                detail_stat[ent_type]['tp'] += len(intersect_)
                detail_stat[ent_type]['fp'] += len(pred_ent_lst_) - len(intersect_)
                detail_stat[ent_type]['fn'] += len(gold_ent_lst_) - len(intersect_)
                if len(pred_ent_lst_) - len(intersect_) > 0:
                    anal_exm[ent_type]['fp'].append(exm)
                if len(gold_ent_lst_) - len(intersect_) > 0:
                    anal_exm[ent_type]['fn'].append(exm)

        # num_preds = tp + fp
        # num_golds = tp + fn
        # prec = (tp + 1e-10) / (tp + fp + 1e-10)
        # rec = (tp + 1e-10) / (tp + fn + 1e-10)
        # f1 = (2.0 * prec * rec + 1e-10) / (prec + rec + 1e-10)
        # # f1_ = (2.0 * tp + 1e-10) / (num_preds + num_golds + 1e-10)  # 另一种等价算法
        # for v in detail_stat.values():
        #     v['num_preds'] = v['tp'] + v['fp']
        #     v['num_golds'] = v['tp'] + v['fn']
        #     v['prec'] = (v['tp'] + 1e-10) / (v['num_preds'] + 1e-10)
        #     v['rec'] = (v['tp'] + 1e-10) / (v['num_golds'] + 1e-10)
        #     v['f1'] = (2.0 * v['prec'] * v['rec'] + 1e-10) / (v['prec'] + v['rec'] + 1e-10)
        # print(f'{" " * 19}precision: {prec:7.2%}; recall: {rec:7.2%}; FB1: {f1:7.2%};  num_preds: {num_preds:4.0f}; num_golds: {num_golds:4.0f}; num_correct: {tp:4.0f};')
        # if verbose:  # 按各实体类型分别输出指标
        #     for ent_type in sorted(detail_stat):
        #         print(f'{ent_type:>17}: precision: {detail_stat[ent_type]["prec"]:7.2%}; recall: {detail_stat[ent_type]["rec"]:7.2%}; FB1: {detail_stat[ent_type]["f1"]:7.2%};'
        #               f'  num_preds: {detail_stat[ent_type]["num_preds"]:4.0f}; num_golds: {detail_stat[ent_type]["num_golds"]:4.0f};')

        num_preds = tp + fp
        num_golds = tp + fn
        prec = (tp) / (tp + fp + 1e-10)
        rec = (tp) / (tp + fn + 1e-10)
        f1 = (2.0 * prec * rec) / (prec + rec + 1e-10)
        # f1_ = (2.0 * tp + 1e-10) / (num_preds + num_golds + 1e-10)  # 另一种等价算法
        for v in detail_stat.values():
            v['num_preds'] = v['tp'] + v['fp']
            v['num_golds'] = v['tp'] + v['fn']
            # v['prec'] = 1. if v['num_preds'] == 0. else v['tp'] / v['num_preds']  # 如果num_preds=0即tp=fp=0
            # v['rec'] = 1. if v['num_golds'] == 0. else v['tp'] / v['num_golds']  # 如果num_golds=0即tp=fn=0
            # v['f1'] = 1. if v['num_golds'] == v['num_preds'] == 0. else 2. * v['tp'] / (v['num_golds'] + v['num_preds'] + 1e-12)
            v['prec'] = (v['tp']) / (v['num_preds'] + 1e-10)
            v['rec'] = (v['tp']) / (v['num_golds'] + 1e-10)
            v['f1'] = (2.0 * v['prec'] * v['rec']) / (v['prec'] + v['rec'] + 1e-10)

        if macro:  # show macro f1
            macro_f1 = mean([v['f1'] for v in detail_stat.values()])
            macro_prec = mean([v['prec'] for v in detail_stat.values()])
            macro_rec = mean([v['rec'] for v in detail_stat.values()])
        else:
            macro_f1 = None
            macro_prec = None
            macro_rec = None

        # Conlleval format
        detail_info_str += f'{" " * 19}precision: {prec:7.2%}; recall: {rec:7.2%}; FB1: {f1:7.2%};  num_preds: {num_preds:4.0f}; num_golds: {num_golds:4.0f}; num_correct: {tp:4.0f};\n'
        for ent_type in sorted(detail_stat):
            # if verbose:  # 按各实体类型分别输出指标
            #     print(f'{ent_type:>17}: precision: {detail_stat[ent_type]["prec"]:7.2%}; recall: {detail_stat[ent_type]["rec"]:7.2%}; FB1: {detail_stat[ent_type]["f1"]:7.2%};'
            #           f'  num_preds: {detail_stat[ent_type]["num_preds"]:4.0f}; num_golds: {detail_stat[ent_type]["num_golds"]:4.0f};')
            detail_info_str += f'{ent_type:>17}: precision: {detail_stat[ent_type]["prec"]:7.2%}; recall: {detail_stat[ent_type]["rec"]:7.2%}; FB1: {detail_stat[ent_type]["f1"]:7.2%};'
            detail_info_str += f'  num_preds: {detail_stat[ent_type]["num_preds"]:4.0f}; num_golds: {detail_stat[ent_type]["num_golds"]:4.0f};\n'

        if use_table:
            detail_info_table = prettytable.PrettyTable()
            detail_info_table.field_names = ['Entity Name', "Prec", "Rec", "F1", 'Num_golds', 'Num_preds']
            for ent_type, v in sorted(detail_stat.items()):
                detail_info_table.add_row(f'{ent_type}|{v["prec"]:.2%}|{v["rec"]:.2%}|{v["f1"]:.2%}|{v["num_golds"]:.0f}|{v["num_preds"]:.0f}'.split('|'))
            detail_info_table.add_row(f'Micro|{prec:.2%}|{rec:.2%}|{f1:.2%}|{num_golds:.0f}|{num_preds:.0f}'.split('|'))
            if macro:
                detail_info_table.add_row(f'Macro|{macro_prec:.2%}|{macro_rec:.2%}|{macro_f1:.2%}|{num_golds:.0f}|{num_preds:.0f}'.split('|'))
            detail_info_table.set_style(13)  # MSWORD_FRIENDLY = 11 PLAIN_COLUMNS = 12 MARKDOWN = 13
            detail_info_str = detail_info_table.get_string()

            # Rich Table
            # detail_info_table = table.Table('Entity Name', "Prec", "Rec", table.Column(header="F1", style='green'), 'Num_golds', 'Num_preds', title="Details", )
            # for ent_type, v in sorted(detail_stat.items()):
            #     detail_info_table.add_row(*f'{ent_type}|{v["prec"]:.2%}|{v["rec"]:.2%}|{v["f1"]:.2%}|{v["num_golds"]:.0f}|{v["num_preds"]:.0f}'.split('|'))
            # detail_info_table.add_row(*f'Micro|{prec:.2%}|{rec:.2%}|{f1:.2%}|{num_golds:.0f}|{num_preds:.0f}'.split('|'))
            # if macro:
            #     detail_info_table.add_row(*f'Macro|{macro_prec:.2%}|{macro_rec:.2%}|{macro_f1:.2%}|{num_golds:.0f}|{num_preds:.0f}'.split('|'))

        if verbose:
            print(detail_info_str)
            # print(detail_info_table)  # Rich Table

        if anal_exm_out_file is not None:
            output = []
            for ent_type in anal_exm:
                output.append(f'***ent_type:{ent_type}:')
                output.append(json.dumps(detail_stat[ent_type], ensure_ascii=False))
                output.append('\n***missing Recall(fn):')
                for exm in anal_exm[ent_type]['fn']:
                    output.append(exm.to_json_str(for_human_read=True, external_attrs=anal_exm_out_file_external_attrs))
                    gold_ent_lst = set((start, end) for start, end, *_ in exm.ent_dct.get(ent_type, []))
                    pred_ent_lst = set((start, end) for start, end, *_ in exm.pred_ent_dct.get(ent_type, []))
                    ent_lst = '\n'.join([f'({start}-{end}):{exm.get_text(start, end)}' for start, end in (gold_ent_lst - pred_ent_lst)])
                    pred_ent_lst = '\n'.join([f'({start}-{end}):{exm.get_text(start, end)}' for start, end in pred_ent_lst])
                    output.append(f'{ent_lst}\npred_ent:{pred_ent_lst}')
                    # output.append(f'{ent_type}:  {ent_lst}')
                output.append('\n***wrong Prediction(fp):')
                for exm in anal_exm[ent_type]['fp']:
                    output.append(exm.to_json_str(for_human_read=True, external_attrs=anal_exm_out_file_external_attrs))
                    gold_ent_lst = set((start, end) for start, end, *_ in exm.ent_dct.get(ent_type, []))
                    pred_ent_lst = set((start, end) for start, end, *_ in exm.pred_ent_dct.get(ent_type, []))
                    ent_lst = '\n'.join([f'({start}-{end}):{exm.get_text(start, end)}' for start, end in (pred_ent_lst - gold_ent_lst)])
                    gold_ent_lst = '\n'.join([f'({start}-{end}):{exm.get_text(start, end)}' for start, end in gold_ent_lst])
                    output.append(f'{ent_lst}\ngold_ent:{gold_ent_lst}')
                output.append('')
            list2file(output, anal_exm_out_file)

        if use_flat_pred_ent_dct:
            for exm in exm_lst:  # 要转回原来的
                exm.pred_ent_dct = exm.ori_pred_ent_dct

        if macro:
            return prec, rec, f1, macro_prec, macro_rec, macro_f1, detail_info_str, dict(detail_stat),
        else:
            return prec, rec, f1, detail_info_str, dict(detail_stat),

    def gen_ent_set(self):
        self.gold_set = set()
        for ent, v_lst in self.ent_dct.items():
            for s, e, *_ in v_lst:
                self.gold_set.add((ent, s, e))
        if hasattr(self, 'pred_ent_dct'):
            self.pred_set = set()
            for ent, v_lst in self.pred_ent_dct.items():
                for s, e, *_ in v_lst:
                    self.pred_set.add((ent, s, e))

    @staticmethod
    def compare_results(exm_lst1, exm_lst2, other_exm_lst=None):
        assert len(exm_lst1) == len(exm_lst2)
        for exm in exm_lst1:
            exm.gen_ent_set()
        for exm in exm_lst2:
            exm.gen_ent_set()
        ret = []
        for eid, (exm1, exm2) in enumerate(zip(exm_lst1, exm_lst2)):
            if not exm1.gold_set: continue
            if exm1.pred_set != exm2.pred_set:
                if exm1.pred_is_right():
                    print(f'===={eid}====')
                    print(exm1)
                    if other_exm_lst is not None:
                        print(other_exm_lst[eid])
                    print(exm2)
                    print('\n')
                    ret.append([exm1, exm2])
        # print(*ret, sep='\n')

    def pred_is_right(self):
        assert hasattr(self, 'pred_ent_dct')
        if not hasattr(self, 'gold_set') or not hasattr(self, 'pred_set'):
            self.gen_ent_set()
        return self.gold_set == self.pred_set

    @staticmethod
    def split_exm_by_deli(exm, deli='。', drop_cross_ent=True):
        ret_exm_lst = []
        ent_lst = exm.get_ent_lst()
        # print(ent_lst)
        prev_end = 0
        for idx, char in enumerate(exm.char_lst):
            if char == deli:
                curr_start = prev_end
                curr_end = idx + 1
                curr_char_lst = exm.char_lst[curr_start: curr_end]
                curr_ent_lst = [e for e in ent_lst if e[1] >= curr_start and e[2] <= curr_end]
                for e in curr_ent_lst:
                    e[1] -= curr_start
                    e[2] -= curr_start
                # curr_ent_lst = [[ent_type, start - curr_start, end - curr_start] for ent_type, start, end, *_ in curr_ent_lst]
                curr_exm = NerExample.from_ent_lst(curr_char_lst, curr_ent_lst)
                ret_exm_lst.append(curr_exm)

                prev_end = curr_end

        # 还有最后的
        curr_start = prev_end
        curr_char_lst = exm.char_lst[curr_start:]
        if curr_char_lst:
            curr_end = len(exm.char_lst)
            curr_ent_lst = [e for e in ent_lst if e[1] >= curr_start and e[2] <= curr_end]
            for e in curr_ent_lst:
                e[1] -= curr_start
                e[2] -= curr_start
            # curr_ent_lst = [[ent_type, start - curr_start, end - curr_start] for ent_type, start, end, *_ in curr_ent_lst]
            curr_exm = NerExample.from_ent_lst(curr_char_lst, curr_ent_lst)
            ret_exm_lst.append(curr_exm)
        # print(*ret_exm_lst, sep='\n')
        return ret_exm_lst

    @staticmethod
    def combine_exm(exm_lst, include_pred_ent=True):
        """按顺序合成多个exm为一个exm text1+text2 = comb_text"""
        curr_length = 0
        combine_char_lst = []
        combine_ent_dct = defaultdict(list)

        if include_pred_ent:
            # combine_pred_ent_dct = defaultdict(list)
            combine_pred_ent_dct = {}

        for exm in exm_lst:
            combine_char_lst.extend(exm.char_lst)
            for ent_type, pos_lst in exm.ent_dct.items():
                new_pos_lst = copy.deepcopy(pos_lst)
                for pos in new_pos_lst:
                    pos[0] += curr_length
                    pos[1] += curr_length
                # new_pos_lst = [[start + curr_length, end + curr_length] for start, end in pos_lst]
                combine_ent_dct[ent_type].extend(new_pos_lst)

            if include_pred_ent and hasattr(exm, 'pred_ent_dct'):
                for ent_type, pos_lst in exm.pred_ent_dct.items():
                    new_pos_lst = copy.deepcopy(pos_lst)
                    for pos in new_pos_lst:
                        pos[0] += curr_length
                        pos[1] += curr_length
                    # new_pos_lst = [[start + curr_length, end + curr_length, prob] for start, end, prob in pos_lst]
                    # combine_pred_ent_dct[ent_type].extend(new_pos_lst)
                    combine_pred_ent_dct[ent_type] = new_pos_lst
            curr_length += len(exm.char_lst)

        combine_exm = NerExample(char_lst=combine_char_lst, ent_dct=combine_ent_dct)
        combine_exm.update(anchor='ent_dct')

        # if include_pred_ent:
        #     combine_exm.pred_ent_dct = dict(combine_pred_ent_dct)

        return combine_exm

    @staticmethod
    def split_exm_by_deli_multi_sent(exm, deli='。'):
        """前后各一句"""
        split_exm_lst = NerExample.split_exm_by_deli(exm, deli=deli)
        ret_exm_lst = []
        length = len(split_exm_lst)
        for idx in range(length):
            if idx == 0:
                comb_exm = NerExample.combine_exm([split_exm_lst[idx], split_exm_lst[idx + 1]], include_pred_ent=False)
            elif idx == length - 1:
                comb_exm = NerExample.combine_exm([split_exm_lst[idx - 1], split_exm_lst[idx]], include_pred_ent=False)
            else:
                comb_exm = NerExample.combine_exm([split_exm_lst[idx - 1], split_exm_lst[idx], split_exm_lst[idx + 1]], include_pred_ent=False)
            ret_exm_lst.append(comb_exm)
        return ret_exm_lst

    @staticmethod
    def stats(exm_lst, ent_anal_out_file=None):
        ent_dct = defaultdict(list)
        char_lens = []
        ent_lens = []
        for exm in exm_lst:
            char_lens.append(len(exm.char_lst))
            ent_lst = exm.get_ent_lst()
            for ent_type, start, end in ent_lst:
                ent_lens.append(end - start)
                ent_dct[ent_type].append(exm.get_text(start, end))
        print('========stats exm_lst===========')
        total_nums = len(exm_lst)
        num_neg = sum(1 for exm in exm_lst if exm.is_neg())
        num_pos = total_nums - num_neg
        print(f'sample_num: {total_nums}, num_neg:{num_neg}, num_pos:{num_pos}, neg_rate: {num_neg / total_nums}')
        print(f'tokens_num: {sum(char_lens)}')
        print(f'ent_num: {len(ent_lens)}')
        ent_type_set = NerExample.get_ent_type_set(exm_lst)
        print(f'ent_type_num: {len(ent_type_set)} -> {ent_type_set}')
        print(f'stats char_len: {stats_lst(char_lens)}')
        print(f'stats ent_len: {stats_lst(ent_lens)}', )
        print()

        if ent_anal_out_file is not None:
            out = []
            ent_freq_dct = {}
            ent_num_lst = [[ent_type, len(lst)] for ent_type, lst in ent_dct.items()]
            ent_num_lst.sort(key=lambda e: e[1], reverse=True)

            for ent_type, lst in ent_dct.items():
                ent_freq_dct[ent_type] = Counter(lst).most_common()

            for ent_type, num in ent_num_lst:
                out.append(f'{ent_type}\tnum:{num}')
                for ment, count in ent_freq_dct[ent_type]:
                    out.append(f'{count}\t{ment}')
                out.append('')
            list2file(out, ent_anal_out_file)

    @staticmethod
    def get_from_cluener_format_file(cluener_format_file):
        obj_lst = load_jsonl(cluener_format_file)
        exm_lst = []
        for obj in obj_lst:
            text = obj['text']
            if 'label' not in obj:
                ent_dct = {}
            else:
                ent_dct = defaultdict(list)
                for ent_type, v in obj['label'].items():
                    for ment, pos_lst in obj['label'][ent_type].items():
                        start, end = pos_lst[0]
                        ent_dct[ent_type].append([start, end + 1])

            exm_lst.append(NerExample(char_lst=list(text), ent_dct=dict(ent_dct)))
        return exm_lst

    @staticmethod
    def get_from_conll_format_file(conllr_format_file, deli='\t', digit2zero=False, schema='IOB'):
        # ontonote5
        """ 读取行 {char}\t{tag}\n 格式数据  \n换行符分割样本
            tag可能有多个 真实和预测 {char}\t{tag}\t{tag}\n 格式数据  \n换行符分割样本
            转为char_lst, tag_lst
            deli为 char 与 tag 的分隔符
        """
        example_lst = []

        with open(conllr_format_file, 'r', encoding='U8') as f:
            lines = f.readlines()
        items = [l.strip().split(deli) for l in lines]

        curr_char_lst = []
        curr_tag_lst = []
        for item in items:
            if len(item) == 1:  # 分隔标志 ['']
                if curr_char_lst and curr_tag_lst:
                    example_lst.append(NerExample.from_tag_lst(curr_char_lst, curr_tag_lst, token_deli=' ', schema=schema))
                    curr_char_lst, curr_tag_lst = [], []
                continue
            word, tag = item[0], item[1]
            if digit2zero:
                word = re.sub('\d', '0', word)  # replace digit with 0.item[1]
            curr_char_lst.append(word)
            curr_tag_lst.append(tag)
        if curr_char_lst and curr_tag_lst:
            example_lst.append(NerExample.from_tag_lst(curr_char_lst, curr_tag_lst, token_deli=' '))
        return example_lst

    @staticmethod
    def get_from_conll_format_file0807(conllr_format_file, deli='\t', digit2zero=False):
        # ontonote5
        """ 读取行 {char}\t{tag}\n 格式数据  \n换行符分割样本
            tag可能有多个 真实和预测 {char}\t{tag}\t{tag}\n 格式数据  \n换行符分割样本
            转为char_lst, tag_lst
            deli为 char 与 tag 的分隔符
        """
        example_lst = []
        en_ontonotes_trans = {'-LRB-': '(',
                              '-RRB-': ')',
                              '-LSB-': '[',
                              '-RSB-': ']',
                              '-LCB-': '{',
                              '-RCB-': '}'
                              }

        with open(conllr_format_file, 'r', encoding='U8') as f:
            lines = f.readlines()
        items = [l.strip().split(deli) for l in lines]

        curr_char_lst = []
        curr_tag_lst = []
        for item in items:
            if len(item) == 1:  # 分隔标志 ['']
                if curr_char_lst and curr_tag_lst:
                    example_lst.append(NerExample.from_tag_lst(curr_char_lst, curr_tag_lst, token_deli=' '))
                    curr_char_lst, curr_tag_lst = [], []
                continue
            word, tag = item[1], item[10]
            # clean word
            if word == '/.': word = '.'
            if word in en_ontonotes_trans:
                word = en_ontonotes_trans[word]
            if digit2zero:
                word = re.sub('\d', '0', word)  # replace digit with 0.item[1]

            curr_char_lst.append(word)
            curr_tag_lst.append(tag)
        if curr_char_lst and curr_tag_lst:
            example_lst.append(NerExample.from_tag_lst(curr_char_lst, curr_tag_lst, token_deli=' '))
        return example_lst

    def update_to_bert_tokenize(self, bert_tokenizer, is_split_into_words=True):
        """
        from transformers import AutoTokenizer # bert_tokenizer 需要是AutoTokenizer才能得到word_ids
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        is_split_into_words: True 代表通过List[str] 来调用分词器 表示已经分词 不需要再切分 只要用bpe等方法表示每个str就行了 适合NER
        最后结果: 增加了两个字段属性
        """
        char_lst = ''.join(self.char_lst) if not is_split_into_words else self.char_lst
        tokenized_inputs = bert_tokenizer(char_lst, is_split_into_words=is_split_into_words, add_special_tokens=False)  # 去掉[CLS]和[SEP]

        tokenized_ids_lst = tokenized_inputs['input_ids']

        tokenized_char_lst = bert_tokenizer.convert_ids_to_tokens(tokenized_ids_lst)
        # word_ids 分字词后的字符在原来文本中的位置
        tokenized_word_ids = tokenized_inputs.word_ids()  # [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        ori_char_lst_2_tok_char_lst = [-1] * len(self.char_lst)  # ori_char_lst[i] -> tok_char_lst[ori_char_lst_2_tok_char_lst[i]]
        for idx, word_id in enumerate(tokenized_word_ids):
            if ori_char_lst_2_tok_char_lst[word_id] == -1:
                ori_char_lst_2_tok_char_lst[word_id] = idx
        # Add two attr
        self.bert_tok_char_lst = tokenized_char_lst
        self.ori_2_tok = ori_char_lst_2_tok_char_lst

    def process_EN_by_tokenizer(self, bert_tokenizer):
        splited_words = self.char_lst  # is_split_into_words=True  e.g. [Austria,1996,-,12,-,07,Brian]
        nested_sub_tokens = self._tokenized2nested(splited_words, bert_tokenizer)  # [[Austria],[1996,-,12,-,07],[Brian]]
        sub_tokens = [sub_tok for i, tok in enumerate(nested_sub_tokens) for sub_tok in tok]  # flat [Austria,1996,-,12,-,07,Brian]
        ori_indexes = [i for i, tok in enumerate(nested_sub_tokens) for sub_tok in tok]  # flat [0,1,1,1,1,1,2]
        sub_tokens_ids = bert_tokenizer.convert_tokens_to_ids(sub_tokens)  # can be used as input ids.
        # Add two attr
        self.sub_tokens = sub_tokens
        self.ori_indexes = ori_indexes

    def process_ZHENG_by_tokenizer(self, bert_tokenizer):
        """ Actually tokenizing with is_split_into_words=False
            Use Tokenizer to auto split "raw_text"
            split by char for ZH, split by space for EN
            Text:' 备受“瞩目”的《迷失》CHINA IN SURPRISE DEFEAT,玩家的魔兽' -> words:[备|受|“|瞩|目|”|的|《|迷|失|》|CHINA|IN|SURPRISE|DEFEAT|,|玩|家|的|魔|兽]  # will strip ' '
            -> subwords:[备|受|“|瞩|目|”|的|《|迷|失|》|CH|##IN|##A|IN|SU|##R|##PR|##ISE|DE|##F|##EA|##T|,|玩|家|的|魔|兽]
            ret: add self.char_lst,sub_tokensm,ori_indexes and update ent_pos
        """
        raw_text = ''.join(self.char_lst)  # ' 备受“瞩目”的《迷失》CHINA IN SURPRISE DEFEAT,玩家的魔兽'
        o = bert_tokenizer.encode_plus(raw_text, return_offsets_mapping=True, add_special_tokens=False)
        sub_tokens = bert_tokenizer.convert_ids_to_tokens(o['input_ids'])  # = bert_tokenizer.tokenize(raw_text)
        self.sub_tokens = sub_tokens  # ['备|受|[UNK]|瞩|目|[UNK]|的|《|迷|失|》|CH|##IN|##A|IN|SU|##R|##PR|##ISE|DE|##F|##EA|##T|,|玩|家|的|魔|兽'] # 开头的空格会去掉。
        offset_mapping = o['offset_mapping']  # [(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,14),(14,16),(16,17),
        # (18,20),(21,23),(23,24),(24,26),(26,29),(30,32),(32,33),(33,35),(35,36),(36,37),(37,38),(38,39),(39,40),(40,41),(41,42)]  # sub_tokens在原文raw_text中的位置
        # print(offset_mapping)

        # get nested_sub_tokens and nested_offset_mapping
        # nested_offset_mapping [[(1,2)],[(2,3)],[(3,4)],[(4,5)],[(5,6)],[(6,7)],[(7,8)],[(8,9)],[(9,10)],[(10,11)],[(11,12)],[(12,14),(14,16),(16,17)],[(18,20)],[(21,23),
        # (23,24),(24,26),(26,29)],[(30,32),(32,33),(33,35),(35,36)],[(36,37)],[(37,38)],[(38,39)],[(39,40)],[(40,41)],[(41,42)]]
        # nested_sub_tokens: [[备],[受],[[UNK]],[瞩],[目],[[UNK]],[的],[《],[迷],[失],[》],[CH,##IN,##A],[IN],[SU,##R,##PR,##ISE],[DE,##F,##EA,##T],[,],[玩],[家],[的],[魔],[兽]]
        nested_offset_mapping = [[offset_mapping[0]]]
        nested_sub_tokens = [[sub_tokens[0]]]
        for e, of in zip(sub_tokens[1:], offset_mapping[1:]):
            if len(e) > 2 and e[:2] == '##':
                nested_sub_tokens[-1].append(e)
                nested_offset_mapping[-1].append(of)
            else:
                nested_sub_tokens.append([e])
                nested_offset_mapping.append([of])
        self.ori_indexes = [i for i, tok in enumerate(nested_sub_tokens) for sub_tok in tok]  # [0,1,2,3,4,5,6,7,8,9,10,11,11,11,12,13,13,13,13,14,14,14,14,15,16,17,18,19,20]
        # print(nested_sub_tokens)
        # print(nested_offset_mapping)

        # get char_lst ['备|受|[UNK]|瞩|目|[UNK]|的|《|迷|失|》|CHINA|IN|SURPRISE|DEFEAT|,|玩|家|的|魔|兽']
        char_lst = []
        char_to_sub_tokens_indexes = []  # # map char_lst indices to sub_tokens, not used yet
        ptn = 0
        for e in nested_sub_tokens:
            s = e[0] + ''.join(e_[2:] for e_ in e[1:])
            char_to_sub_tokens_indexes.append(list(range(ptn, ptn + len(e))))
            ptn += len(e)
            char_lst.append(s)
        self.char_lst = char_lst  # ['备|受|[UNK]|瞩|目|[UNK]|的|《|迷|失|》|CHINA|IN|SURPRISE|DEFEAT|,|玩|家|的|魔|兽']
        # print(char_lst)

        char_to_raw_lst_indexes = {i: None for i in range(len(raw_text))}  # used to map original indices of ent in raw_text to char_lst
        for char_lst_idx, lst in enumerate(nested_offset_mapping):
            for raw_start, raw_end in lst:
                char_to_raw_lst_indexes[raw_start] = char_lst_idx
                for i in range(raw_start, raw_end):
                    char_to_raw_lst_indexes[i] = char_lst_idx
        for i in reversed(sorted(char_to_raw_lst_indexes)):
            if char_to_raw_lst_indexes[i] == None:
                char_to_raw_lst_indexes[i] = char_to_raw_lst_indexes[i + 1]
        # print(char_to_raw_lst_indexes)  # {0:0,1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,11:10,12:11,13:11,14:11,15:11,16:11,17:12,18:12,19:12,20:13,21:13,
        # 22:13,23:13,24:13,25:13,26:13,27:13,28:13,29:14,30:14,31:14,32:14,33:14,34:14,35:14,36:15,37:16,38:17,39:18,40:19,41:20}

        for ent_type, v_lst in self.ent_dct.items():
            for idx in range(len(v_lst)):
                start = v_lst[idx][0]
                end = v_lst[idx][1]
                new_start = char_to_raw_lst_indexes[start]
                if end == len(raw_text):
                    new_end = char_to_raw_lst_indexes[end - 1] + 1
                else:
                    new_end = char_to_raw_lst_indexes[end]
                v_lst[idx][0] = new_start
                v_lst[idx][1] = new_end
        self.update_text()

    @staticmethod
    def _tokenized2nested(tokenized_raw_text: List[str], tokenizer, max_len: int = 5):
        nested_sub_tokens = []
        for word in tokenized_raw_text:
            sub_tokens = tokenizer.tokenize(word)
            if len(sub_tokens) == 0:
                # The tokenizer returns an empty list if the input is a space-like string
                sub_tokens = [tokenizer.unk_token]
            elif len(sub_tokens) > max_len:
                # The tokenizer may return a very long list if the input is a url
                sub_tokens = sub_tokens[:max_len]
            nested_sub_tokens.append(sub_tokens)

        return nested_sub_tokens  # [[Austria],[1996,-,12,-,07],[Brian]]


def mean(lst):
    return sum(lst) / len(lst)


def metric_aggregater(detail_stat, ents_schema):
    # ents_schema = {}
    # new_detail_stat = copy.deepcopy(detail_stat)
    new_detail_stat = {}
    for agg_ent, ents in ents_schema.items():
        agg_ent_dct = dict(tp=0, fp=0, fn=0, num_preds=0, num_golds=0, mif1=0., miprec=0., mirec=0.,
                           maf1=0., maprec=0., marec=0.)
        for ent in ents:
            if ent in detail_stat:
                for field in ['tp', 'fp', 'fn', 'num_preds', 'num_golds']:
                    agg_ent_dct[field] += detail_stat[ent][field]
        if agg_ent_dct['num_golds'] == 0:
            continue
        tp = agg_ent_dct['tp']
        num_golds = agg_ent_dct['num_golds']
        num_preds = agg_ent_dct['num_preds']
        agg_ent_dct['mif1'] = 1. if num_golds == num_preds == 0 else 2 * tp / (num_golds + num_preds + 1e-12)
        agg_ent_dct['miprec'] = tp / (num_preds + 1e-12)
        agg_ent_dct['mirec'] = tp / (num_golds + 1e-12)

        agg_ent_dct['maf1'] = mean([detail_stat[ent]['f1'] for ent in ents if ent in detail_stat])
        agg_ent_dct['maprec'] = mean([detail_stat[ent]['prec'] for ent in ents if ent in detail_stat])
        agg_ent_dct['marec'] = mean([detail_stat[ent]['rec'] for ent in ents if ent in detail_stat])
        new_detail_stat[agg_ent] = agg_ent_dct
    return new_detail_stat


def header_format(content: str, sep='=', width=100):
    side_width = max(width - len(content) - 2, 10)
    left_width = side_width // 2
    right_width = side_width - left_width
    return f"{sep * left_width} {content} {sep * right_width}"


def auto_device(logger, torch, numpy, subprocess, min_memory: int = 2048):
    """
    https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow
    """
    logger.info("Automatically allocating device...")

    if not torch.cuda.is_available():
        logger.info("Cuda device is unavailable, device `cpu` returned")
        return torch.device('cpu')

    try:
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        free_memories = subprocess.check_output(COMMAND.split()).decode().strip().split('\n')[1:]  # ['15339 MiB', '27572 MiB', '22769 MiB', '13094 MiB']
        free_memories = [int(x.split()[0]) for x in free_memories]
        assert len(free_memories) == torch.cuda.device_count()
    except:
        logger.warning("Cuda device information inquiry failed, device `cpu` returned")
        return torch.device('cpu')
    else:
        selected_id = numpy.argmax(free_memories)
        selected_mem = free_memories[selected_id]
        if selected_mem < min_memory:
            logger.warning(f"Cuda device `cuda:{selected_id}` with maximum free memory {selected_mem} MiB "
                           f"fails to meet the requirement {min_memory} MiB, device `cpu` returned")
            return torch.device('cpu')
        else:
            logger.info(f"Cuda device `cuda:{selected_id}` with free memory {selected_mem} MiB "
                        f"successfully allocated, device `cuda:{selected_id}` returned")
            return torch.device('cuda', selected_id)


def get_refined_score_mask(exm):
    num_chars = len(exm.char_lst)
    link_flags = [0] * (num_chars - 1)
    for ent_type, pos_lst in exm.ent_dct.items():
        for start, end, *other in pos_lst:
            for i in range(start, end - 1):
                link_flags[i] = 1

    mask = np.zeros([num_chars, num_chars]).astype('float')
    # for s in range(num_chars):
    #     for e in range(s, num_chars):
    #         if
    # print(num_chars)
    # print(link_flags)
    for i, link_flag in enumerate(link_flags):
        # link_start = i
        # link_end = i + 1
        if link_flag == 0:
            mask[:i + 1, i + 1:] = 1.
            # print(i)

    for ent_type, pos_lst in exm.ent_dct.items():
        for start, end, *other in pos_lst:
            mask[start, end - 1] = 1.

    for i in range(num_chars):
        mask[i, i] = 1.
    return mask


if __name__ == '__main__':
    exm_lst = NerExample.load_from_jsonl('data/onto/test.jsonl', token_deli=' ', external_attrs=['bert_tok_char_lst', 'ori_2_tok'])
    for exm in exm_lst:
        input()
        print(exm)
        print(get_refined_score_mask(exm))
