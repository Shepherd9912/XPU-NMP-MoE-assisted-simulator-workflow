# DyCAR-MoE: Dynamic Cost-Aware Routing & Scheduling for MoE on xPU-NMP Systems

This repository contains the cycle-accurate architectural simulator and scheduling framework for the paper **"Cost-Aware Routing for Efficient MoE Inference on 3D Near-Memory-Processing Systems"**. 

该项目提供了一套从底层物理算力、3D HBM 显存容量到上层微架构调度的全链路仿真环境，专门用于评估大语言模型 (LLM) 和混合专家模型 (MoE) 在异构“逻辑计算层 (xPU) + 3D近存处理层 (NMP)”架构上的推理性能与能效 (ED²P)。

## 📂 目录结构与核心模块 (Directory Structure)

项目主要由模型配置目录和四大核心 Python 物理模拟引擎组成：

```text
.
├── configs/                    # 各种主流 MoE 模型的硬件评估配置文件
│   ├── deepseek_v2_lite.json
│   ├── olmoe_1b_7b.json
│   ├── phi_mini.json
│   └── qwen_1_5.json
├── hardware_config.py          # 物理基座：硬件常量、能耗模型与 MACs 动态推导
├── capacity_profiler.py        # 空间管家：3D HBM 容量高水位探测与 OOM 防御
├── latency_energy.py           # 时间与能量引擎：关键路径流水线重叠与 Joule 结算
└── dynamic_scheduler.py        # 调度大脑与主入口：动态 α 切分与单向递减记忆锁

```
## ⚙️ 模型配置与参数提取 (configs/)

为了保证框架的泛化能力，我们将模型的架构超参数与物理引擎完全解耦。在 `configs/` 目录下，存储了主流 MoE 模型的结构参数（JSON 格式）。

框架会在运行时通过 `ModelProfile` 动态解析这些参数，并自动计算出：
- 算子级别的乘加操作数 (MACs)，如 `MAC_ROUTER`, `MAC_EXPERT`, `MAC_SUM`。
- 动态 KV Cache 和静态 Hot Expert 的物理内存占用边界 (Bytes)。

**当前支持评估的模型：**
- Phi-mini-MoE (Microsoft)
- Qwen-1.5-MoE (Alibaba)
- OLMoE-1B-7B (AllenAI)
- DeepSeek-V2-Lite (DeepSeek - MLA Compressed KV)

## 🧠 核心架构创新实现 (Core Innovations)

我们的代码严格实现了论文中的以下微架构创新：

### Capacity Boundary Detection (`capacity_profiler.py`)
严格遵守 4GB 3D HBM 的物理上限。计算公式囊括了：静态热专家权重 + Attention 投影矩阵权重 + 动态 KV Cache。在面临长文本/大并发导致 KV Cache 暴涨时，主动生成警戒水位线限制。

### Pipeline Overlap & Cycle Stealing (`latency_energy.py`)
极其诚实地模拟了总线带宽 (LPDDR5/D2D Bus) 与 NPU 算力的博弈。支持评估纯串行 (Serial) 和流水线双发重叠 (Pipeline Overlap) 的时间掩盖效应，精准捕获总线带宽窃取惩罚 (Exposed Penalty)。

### Data-Driven Empirical Roofline Intersection (`dynamic_scheduler.py`)
调度器内置“微型试算引擎”，在每个 Token 生成前，于合法容量域内扫描 Attention 硬件切分比例 (α∈[0.0, 1.0])，寻找 ED²P 最优点。

### State-Aware Monotonicity Lock (`dynamic_scheduler.py`)
为消除硬件调度过程中的剧烈抖动（即 α 频繁升降引发的海量 KV Cache 在片间总线上来回倒腾的开销），实现了：
- **单向递减锁定 (Monotonicity Lock)**: α 只能下调不能回升。
- **迟滞大阶梯下折 (Hysteresis Step-down)**: 面临容量极限时，强制阶梯式让出显存缓冲区，从根本上削减额外通信开销。


## 🚀 快速启动 (Quick Start)

主评估程序基于 Python 的 `multiprocessing` 编写，支持利用多核 CPU 针对上述多个模型、多种 Batch Size 和不同的调度策略展开全空间笛卡尔积式的矩阵评估。

```bash
# 启动包含 Baseline, DyCAR 等 7 种策略的全系统仿真验证
python dynamic_scheduler.py
```

## 📊 输出文件 (Outputs)

程序执行完毕后，将在根目录下生成两种级别的日志文件：

- **宏观性能矩阵：`Ultimate_Flat_Metrics_MultiModel_pim.csv`**
  包含所有模型在不同序列长度锚点 (1K, 2K, 4K, 8K...) 下的系统总延迟 (Cycles)、总能耗 (Joules) 以及各硬件组件（Router, Attention, Cold/Hot Expert, Bus In/Out）的细粒度账本。可直接用于绘制消融实验柱状图与折线图。

- **微观调度轨迹：`Alpha_Eviction_Trace_<Model>_B<Batch>.csv`**
  专门用于追踪 `Hetero_Dynamic_Optimal` 策略下，调度大脑对每一个 Token 作出的动态决策（记录了 Seq_Len, 动态调整的 Alpha, 以及引发的 Evicted_KB）。可用于验证 Monotonicity Lock 的平滑降级效果。

## 📜 依赖环境

- `Python 3.8+`
- `numpy` (用于性能查表与线性插值计算)
- `tqdm` (用于多进程执行进度展示)


