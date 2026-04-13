# GPT-Poet: 基于深度 Transformer 与核采样机制的古诗生成模型

本项目基于 PyTorch 深度学习框架，从零构建并实现了一个类 GPT 架构的自回归文本生成系统。项目利用大规模古诗语料（全唐诗）进行训练，完成了对古典诗词格律与语义的深度建模，深入实践了大语言模型的底层运作机制。

## 🌟 核心特性 (Core Features)

* **自回归架构与因果掩码**：采用标准 Decoder-only 的 Transformer 架构，独立设计并实现下三角因果掩码 (Causal Mask)，严格阻断前向信息泄露，实现从左至右逐字生成逻辑。
* **序列特征提取与优化**：引入正余弦位置编码 (Positional Encoding) 赋予并行数据时序特征。通过对交叉熵损失 (Loss) 与困惑度 (PPL) 的实时监控调优，确保深层模型稳定收敛。
* **高级解码策略**：针对文本生成任务常见的“复读机”缺陷，自主实现了带温度控制 (Temperature) 的 Top-P 核采样 (Nucleus Sampling) 与重复惩罚机制，在保障逻辑合理性的同时显著提升了生成文本的多样性与意境创造力。
* **自注意力可视化**：提取多头自注意力 (Multi-head Attention) 矩阵，通过热力图直观展示模型在生成过程中的内部注意力分布与词元关联。

## 📁 目录结构 (Repository Structure)

```text
.
├── chinese-poetry-master/         # 大规模古诗原始数据集（全唐诗）
├── poetry_output/                 # 模型训练输出与可视化结果目录
│   ├── poetry_gpt_best.pth        # 训练得到的最佳模型权重 (97.5 MB)
│   ├── report_attention_heatmap.png # 多头自注意力机制热力图可视化
│   ├── report_loss_curve_redrawn.png # 训练过程 Loss 曲线图
│   ├── report_ppl_curve_redrawn.png  # 训练过程 PPL (困惑度) 曲线图
│   ├── training_data_report.csv   # 训练指标数据记录
│   ├── training_history.json      # 训练历史日志
│   └── word2id.json               # 词表到索引的映射字典
├── 基于深度 Transformer 与核采样机制的古诗生成模型.ipynb             # 核心代码：模型定义与训练脚本
├── 基于深度 Transformer 与核采样机制的古诗生成模型——古诗生成与热力图分析.ipynb # 核心代码：推理生成与自注意力热力图绘制
├── 基于深度 Transformer 与核采样机制的古诗生成模型可视化.ipynb        # 核心代码：Loss与PPL曲线图表绘制
└── simhei.ttf                     # 中文字体文件，用于 Matplotlib 图表中的中文渲染
