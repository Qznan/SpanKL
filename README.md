<h1 align="center">SpanKL</h1>
<p align="center">
  <a href="https://github.com/Qznan/SpanKL">
    <img src="https://img.shields.io/github/stars/Qznan/SpanKL.svg?colorA=orange&colorB=orange&logo=github" alt="GitHub stars">
  </a>
  <a href="https://github.com/Qznan/SpanKL/issues">
        <img src="https://img.shields.io/github/issues/Qznan/SpanKL.svg"
             alt="GitHub issues">
  </a>
  <a href="https://github.com/Qznan/SpanKL/">
        <img src="https://img.shields.io/github/last-commit/Qznan/SpanKL.svg">
  </a>
   <a href="https://github.com/Qznan/SpanKL/blob/main/LICENSE">
        <img src="https://img.shields.io/github/license/Qznan/SpanKL.svg">
  </a>
  
</p>

# A Neural Span-Based Continual Named Entity Recognition Model

Source code for AAAI 2023 paper: A Neural Span-Based Continual Named Entity Recognition Model [[arxiv](https://arxiv.org/pdf/2302.12200.pdf)]  
[[Appendix for AAAI paper](Paper_Appendix.pdf)]

## 1. Prerequisites

```
# Environment
- python >= 3.6

# Dependencies
- torch >= 1.8
- transformers == 4.3.0
- rich
- prettytable
- paramiko  # [Option] if need to remotely open file.
```


## 2. Dataset
- [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)
- [Few-NERD](https://ningding97.github.io/fewnerd)


We provide the pre-processed datasets in `data/` DIR, which keep the same sample-task allocation for split setup experimented in the paper for reproduction.

[Update 20230525🔥] We provide the zip file of `data/` as `data.zip` for convenient, since the original `data/` used LFS and need Github quota 😑. Please unzip it before running the code.😊
## 3. Training

```shell
python train_clner.py \
      --gpu 0 \
      --m spankl \
      --corpus onto \
      --setup split \
      --perm perm0
      
# gpu: which gpu to use. (-2: auto allocate gpu -1: use cpu)
# m: types of model (spankl|add|ext)
# corpus: (onto|fewnerd)
# setup: (split|filter)
# perm: task learning order, default perm0 (perm0|perm1|...)
```
**Note** that this yields results regarding **one of** the Task Permutation (e.g., --perm0), while Tab.1 and Tab.2 in the paper are the results averaged over all the Task Permutations.  

**Update 0305! #1** For convenience, we log the final metric used in the paper at each incremental step (Task) during training, as belows:
```angular2html
...
INFO     Test All Final MacroF1 over Task-Level:{metric}
...
INFO     Test Filter Final MacroF1 over Task-Level:{metric}
...
```
and also provide an overivew of the performance learned so far:
```angular2html
# Logging example when the running finishs (All tasks have been learned on OntoNotes).
======onto======

onto-0-2022-07-21_10-02-57-1824-spankl_split_perm0

***Test All***
[[87.9   0.    0.    0.    0.    0.   87.9  87.9 ]
 [87.18 93.33  0.    0.    0.    0.   90.43 90.25]
 [88.13 93.41 95.47  0.    0.    0.   92.61 92.34]
 [87.4  93.26 95.16 85.02  0.    0.   90.71 90.21]
 [87.13 92.65 95.05 84.71 81.37  0.   89.38 88.18]
 [87.88 93.02 95.08 85.93 83.04 93.01 90.3  89.66]
 [-0.26 -0.39 -0.39  0.    0.    0.   -0.17  0.  ]]
```
The table is organized as (Table Format):
```angular2html
                Task1_metric Task2_metric ... MicroF1 MacroF1
Learn to Task1
Learn to Task2
...
BI (backward interference / forgetting)
```
The **last columns** are the metrics of each step used in the paper.

##### More ways to print the above CL metrics:
After and during training, an `overview_metric.json` file will be generated in `model_ckpt/[MODEL_INFO]/` recording the required details, and you can:
- refer to `__main__` in `print_cl_metric.py` to use `print_cl_metric()` func to **remotely** print results.
- or refer to the end of `train_clner.py` to use a wrapped `simply_print_cl_metric()` func to **locally** print results.

For any questions please notice the comments in the code or contact me.  
Welcome to star or raise issues and PR! :)
## 4. License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## 5. Citation

If you use this work or code, please kindly cite this paper:

```
@inproceedings{zhang2023spankl,
  title={A Neural Span-Based Continual Named Entity Recognition Model},
  author={Zhang, Yunan and Chen, Qingcai},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```


