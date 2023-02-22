import numpy as np
import sys
from train_clner import onto_sorted_ids_dct, fewnerd_sorted_ids_dct
from ner_loader import onto_entity_task_lst, fewnerd_entity_task_lst
import datautils as utils

np.set_printoptions(precision=2)  # decimal 小数点位数
np.set_printoptions(threshold=sys.maxsize)  # 打印是否截断
np.set_printoptions(linewidth=4000)  # 多长换行
np.set_printoptions(suppress=True)  # 不打印科学计数法


def sort_by_idx(lst, ids=None):
    if ids is None:
        return lst
    sorted_lst = []
    for idx in ids:
        sorted_lst.append(lst[idx])
    return sorted_lst


def mean_and_std(lst):
    mean = np.mean(lst)
    std = np.std(lst)
    return f'{mean:.2f}±{std:.2f}'


global model_ckpt
model_ckpt = ''


class CL_Metric:
    def __init__(self, tasks):

        self.tasks = tasks  # [[ent1,ent2],[ent3]]
        self.num_tasks = len(tasks)
        self.all_ents = sum(tasks, [])
        self.num_all_ents = len(self.all_ents)
        self.num_ents_per_task = [len(t) for t in tasks]
        self.taskid2ents = {taskid: ents for taskid, ents in enumerate(tasks)}
        self.ent2taskid = {}
        for taskid, ents in enumerate(tasks):
            for ent in ents:
                self.ent2taskid[ent] = taskid

    def aggregate_metrics(self, metrics_lst):
        # metrics_lst: [dict, dict]
        ret_metrics = dict(tp=0, fp=0, fn=0, num_preds=0, num_golds=0, f1=0., prec=0., rec=0.)
        for metrics in metrics_lst:
            for field in ['tp', 'fp', 'fn', 'num_preds', 'num_golds']:
                ret_metrics[field] += metrics[field]
        tp = ret_metrics['tp']
        num_golds = ret_metrics['num_golds']
        num_preds = ret_metrics['num_preds']
        ret_metrics['f1'] = 1. if num_golds == num_preds == 0 else 2 * tp / (num_golds + num_preds + 1e-12)
        ret_metrics['prec'] = tp / (num_preds + 1e-12)
        ret_metrics['rec'] = tp / (num_golds + 1e-12)
        return ret_metrics

    def calc(self, test_metrics, epo=None):
        # test_metrics structure:
        # learned_taskid: test_epo: {per_ent_detailed_metrics}
        # e.g.
        """
        "test_metric": {
            "0": {
              "9": {
                "location-GPE": {
                  "tp": 17296.0,
                  "fp": 11672.0,
                  "fn": 3113.0,
                  "num_preds": 28968.0,
                  "num_golds": 20409.0,
                  "prec": 0.5970726318696472,
                  "rec": 0.8474692537605918,
                  "f1": 0.700569090823768
                },
                "location-park": {...
        """
        actual_num_task = sorted(map(int, test_metrics))[-1] + 1  # 模型训练到一半任务的情况
        for i in range(actual_num_task):  # all_tasks的情况，只有最后一个任务的数据
            if str(i) not in test_metrics:
                test_metrics[str(i)] = test_metrics[str(actual_num_task - 1)]
        self.__init__(self.tasks[:actual_num_task])
        res = {learned_taskid: {} for learned_taskid in range(self.num_tasks)}

        if epo is None:  # 取唯一一个或最后一个
            epo = str(sorted(map(int, utils.get_first_value_of_dict(test_metrics)))[-1])

        for learned_taskid in sorted(res):
            per_ent_detailed_metrics = test_metrics[str(learned_taskid)][epo]
            so_far_tasks = []
            for testing_taskid in range(learned_taskid + 1):
                # process per testing taskid metrics
                per_task_metrics = {}
                for ent in self.taskid2ents[testing_taskid]:
                    if ent in per_ent_detailed_metrics:  # 一般来说肯定有 除非测试集非完整 缺少某个实体
                        per_task_metrics[ent] = per_ent_detailed_metrics[ent]
                per_task_metrics['TASK_TOTAL'] = self.aggregate_metrics([per_ent_detailed_metrics[ent] for ent in self.taskid2ents[testing_taskid]
                                                                         if ent in per_ent_detailed_metrics])
                res[learned_taskid][testing_taskid] = per_task_metrics
                so_far_tasks.append(per_task_metrics['TASK_TOTAL'])
            res[learned_taskid]['SOFAR_TASKS_TOTAL'] = self.aggregate_metrics(so_far_tasks)

        # for tid in range(sorted(res)[-1]):
        #     if tid not in res:
        #         res[tid] = res[sorted(res)[-1]]
        self.res = res

    def print(self, info=None, detail=True):
        if info is not None:
            print(info)
        # matrix: [learned_tid, test_tid]
        # so_far_overall [learned_tid]
        sofar_task_f1_lst = [self.res[learned_taskid]['SOFAR_TASKS_TOTAL']['f1'] for learned_taskid in range(self.num_tasks)
                             if 'SOFAR_TASKS_TOTAL' in self.res[learned_taskid]]
        sofar_task_f1_lst = np.array(sofar_task_f1_lst)  # [learned_taskid]
        num_learned_task = len(sofar_task_f1_lst)

        task_matrix = np.zeros([num_learned_task, self.num_tasks])  # [learned_tid, test_tid]
        detailed_ent_matrix = np.zeros([num_learned_task, self.num_all_ents])  # [learned_tid, ents]

        for learned_taskid in range(num_learned_task):
            for testing_taskid in range(learned_taskid + 1):
                task_matrix[learned_taskid][testing_taskid] = self.res[learned_taskid][testing_taskid]['TASK_TOTAL']['f1']

            for entid, ent in enumerate(self.all_ents):
                curr_ent_tid = self.ent2taskid[ent]
                if curr_ent_tid <= learned_taskid:
                    if ent in self.res[learned_taskid][curr_ent_tid]:
                        detailed_ent_matrix[learned_taskid][entid] = self.res[learned_taskid][curr_ent_tid][ent]['f1']

        self.print_metrix(task_matrix, sofar_task_f1_lst)
        if detail:
            self.print_metrix(detailed_ent_matrix, sofar_task_f1_lst)

    def print_metrix(self, matrix, sofar_task_f1_lst):
        matrix = matrix * 100
        sofar_task_f1_lst = sofar_task_f1_lst * 100
        ori_matrix = matrix
        # calc bt (forget)

        # 计算marco_f1 任务级别
        macro_f1 = np.sum(matrix, 1)  # [learned_tid]
        macro_f1 = macro_f1 / np.arange(1, len(macro_f1) + 1)
        macro_f1 = np.vstack([macro_f1[:, None], np.array([0.])])

        max_score = np.max(matrix, 0)  # [test_tid]
        bt_score = matrix[-1] - max_score
        matrix = np.vstack([matrix, bt_score])

        mean_bt_score = np.mean(bt_score)

        print(model_ckpt)
        print('mean of sofar_task_f1_lst', np.mean(sofar_task_f1_lst))

        sofar_task_f1_lst = np.hstack([sofar_task_f1_lst, mean_bt_score])[:, None]
        matrix = np.hstack([matrix, sofar_task_f1_lst])
        matrix = np.hstack([matrix, macro_f1])

        print(matrix)
        print(repr(matrix))
        print(repr(ori_matrix))



describe = """Table Format
                entity1_metric entity2_metric ... MicroF1 MacroF1
Learn to Task1
Learn to Task2
...
BI (backward interference / forgetting)
"""
print(describe)

# remotely open the overview_metric.json in the server and parse, print the CL metric.
ro232 = utils.Remote_Open('[ipaddress]', '[port]', '[user]', '[password]')
ro232.set_default_dir('/home/zyn/SpanKL/model_ckpt/')

model_ckpt = 'fewnerd-0-2022-07-20_11-56-28-7430-spankl_split_perm0'
model_ckpt = 'onto-0-2022-07-21_10-02-57-1824-spankl_split_perm0'
metrics_files = model_ckpt + '/overview_metric.json'
metrics_json = ro232.load_json(metrics_files)
test_metric = metrics_json['test_metric']
filter_test_metric = metrics_json['filter_test_metric'] if 'filter_test_metric' in metrics_json else None

# perm = 'perm0'
perm = model_ckpt.split('_')[-1]


def print_cl_metric(model_ckpt, test_metric, perm, filter_test_metric=None):
    if model_ckpt.startswith('onto'):
        print('======onto======')
        print('task_best_dev_epo:', metrics_json['task_best_dev_epo']) if 'task_best_dev_epo' in metrics_json else None
        perm_ids = onto_sorted_ids_dct[perm]
        onto = sort_by_idx(onto_entity_task_lst, list(map(int, perm_ids)))

        cl_metrics = CL_Metric(onto)
        cl_metrics.calc(test_metric)
        cl_metrics.print('\n***Test All***', detail=False)
        if filter_test_metric is not None:
            cl_metrics.calc(filter_test_metric)
            cl_metrics.print('\n***Test Filter***', detail=False)

    if model_ckpt.startswith('fewnerd'):
        print('======fewnerd======')
        print('task_best_dev_epo:', metrics_json['task_best_dev_epo']) if 'task_best_dev_epo' in metrics_json else None
        perm_ids = fewnerd_sorted_ids_dct[perm]
        fewnerd = sort_by_idx(fewnerd_entity_task_lst, list(map(int, perm_ids)))
        cl_metrics = CL_Metric(fewnerd)
        cl_metrics.calc(test_metric)
        cl_metrics.print('\n***Test All***', detail=True)
        cl_metrics.calc(filter_test_metric)
        cl_metrics.print('\n***Test Filter***', detail=True)

print_cl_metric(model_ckpt, test_metric, perm, filter_test_metric)