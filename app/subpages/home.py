#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/5 20:20
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   home.py
# @Desc     :   

from streamlit import title, expander, caption, empty

empty_message = empty()
empty_message.info("Please check the details at the different pages of core functions.")

title("RNN Translater 1.0")
with expander("**INTRODUCTION (CN)**", expanded=False):
    caption("+ 基于 Streamlit 的交互式机器翻译评估系统，支持实时推理与结果可视化。")
    caption("+ 支持 GRU seq2seq 模型，集成贪心搜索与集束搜索两种解码策略。")
    caption("+ 集成 SQLite 数据库存储平行语料，支持自动化的训练/验证/测试集划分。")
    caption("+ 实现基于 spaCy 的双语分词，支持可配置的严格模式与词形还原处理。")
    caption("+ 提供基于词典的 token-to-index 转换，支持 OOV 词汇的 UNK 处理机制。")
    caption("+ 支持交互式模型选择、解码策略切换与随机数据抽样功能。")
    caption("+ 可视化展示原始中文句子、参考英文翻译与模型生成译文。")
    caption("+ 使用 NLTK 计算 BLEU 分数，集成平滑函数确保评估稳定性。")
    caption("+ 将 BLEU 分数映射到多领域质量基准（新闻、技术文档、口语对话）。")
    caption("+ 完整工作流支持：模型初始化 → 数据选择 → 推理 → 评估 → 可视化展示。")
    caption("+ 支持一键重新预测与数据重新抽样，保障实验可重复性。")
    caption("+ 专为机器翻译模型的教学演示与科研评估设计。")
    caption("+ 采用模块化设计，易于扩展至其他 seq2seq 架构与语言对。")
with expander("**INTRODUCTION (EN)**", expanded=True):
    caption("+ Streamlit-based interactive machine translation evaluation system with real-time inference.")
    caption("+ Supports GRU seq2seq models with both greedy and beam search decoding strategies.")
    caption("+ Integrates SQLite database for parallel corpus storage and automatic train/val/test splitting.")
    caption("+ Implements spaCy-based bilingual tokenisation with customisable strictness and lemmatisation.")
    caption("+ Provides dictionary-based token-to-index conversion with UNK handling for out-of-vocabulary words.")
    caption("+ Features interactive model selection (GRU), decoding strategy toggling, and random data sampling.")
    caption("+ Displays original Chinese sentences, reference English translations, and model-generated hypotheses.")
    caption("+ Computes BLEU scores using NLTK with smoothing functions for robust evaluation.")
    caption("+ Maps BLEU scores to multi-domain quality benchmarks (news, documentation, conversation).")
    caption("+ Supports full workflow: model initialisation → data selection → inference → evaluation → visualisation.")
    caption("+ Offers one-click re-prediction and data re-sampling for reproducible experimentation.")
    caption("+ Designed as an educational and research tool for neural machine translation model assessment.")
    caption("+ Built with modular components for easy extension to other seq2seq architectures and languages.")
