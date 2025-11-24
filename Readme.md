# MHQG-llama3-reranker

## Introduction
多跳问题生成（Multi-hop Question Generation, QG）指的是需要整合来自多个段落的多个分散证据片段，并对其进行推理以生成与答案相关、事实一致的问题。它在教育系统（Heilman和Smith，2010；Lindberg等人，2013；Yao等人，2018）和智能虚拟助手系统（Shum等人，2018；Pan等人，2019）中具有重要作用，还可以与问答（QA）模型结合作为双重任务，以增强QA系统的推理能力（Tang等人，2017）。本研究提出了一种创新的多跳问题生成方法，通过结合LLaMA3模型的PISSA（Pretrained Iterative Self-Supervised Augmentation）微调、思维链（Chain-of-Thought, CoT）引导生成策略以及重排序（Reranker）技术，以提升多跳推理任务中的性能。我们对LLaMA3模型进行了PISSA微调，以增强模型在多跳问题生成中的泛化能力和精度。其次，采用思维链引导生成策略，通过逐步推理生成中间步骤，提高了推理过程的透明度和逻辑一致性。最后，引入重排序技术，先让模型推理出最关键的两个依据，再通过文本余弦相似度计算选择最相近的两个选项，确保了关键依据选择的准确性和可靠性。我们使用HotpotQA（Yang等人，2018）数据集进行实验。我们基于验证集对各大模型进行了对比实验。实验结果表明，我们的模型相较于传统模型有着较大的改良和提升。最终，我们在验证集上测试的结果为：BLEU-4为21.86，BertScore为48.64，综合成绩为35.26。

## Overview

![image](https://github.com/kagomeSh4ron/MHQG-llama3-reranker/assets/138695155/fae2a828-8c58-4c85-b4ef-2136af6a2030)


### Directory
```bash
Code/
│
├── pissa/                 # 微调代码文件夹
├── finetune_data.json     # 提高生成多跳问题能力的微调数据集
├── nice_prompt.txt        # 桥接型提示词样例
├── nice_prompt2.txt       # 比较型提示词样例
├── peft_run.py            # 模型推理
├── requirements.txt       # 环境配置
├── reranker.py            # 词袋检索器的实现
├── results_dev.json       # 最终生成的多跳问题
├── search_data.json       # 提高模型检索能力的微调数据集

```


# Getting Started

## Installation

### Requirment
```bash
pip install -r requirements.txt
```

### PiSSA
![image](https://github.com/kagomeSh4ron/MHQG-llama3-reranker/assets/138695155/b63bcede-8830-4d3e-b361-7e4a826c85e8)

> [!IMPORTANT]
> Installation is mandatory.

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```
启动UI可视化界面
```bash
python webui.py
```
## Inference
微调模型的推理代码
```bash
python peft_run.py
```

## Citation

```bibtex
@inproceedings{zheng2024llamafactory,
  title={LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models},
  author={Yaowei Zheng and Richong Zhang and Junhao Zhang and Yanhan Ye and Zheyan Luo and Zhangchi Feng and Yongqiang Ma},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)},
  address={Bangkok, Thailand},
  publisher={Association for Computational Linguistics},
  year={2024},
  url={http://arxiv.org/abs/2403.13372}
}
```
