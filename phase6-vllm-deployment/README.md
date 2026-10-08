# Phase 6: vLLM 真实推理部署与通信瓶颈发现

> **任务来源**：NVIDIA 侧一作，2026-10-08
> **目的**：在 BW1000 双机上部署 vLLM 真实推理，观察 TP=16 下通信-计算交互暴露的实际问题
> **需要**：SCNet 模型训练模块，双机 × 8 卡 BW1000

---

## 一、为什么要部署 vLLM

Phase 4 测了到达异质性，Phase 5 测了端到端分解，但都是**合成实验**（手工构造的 AG + GEMM）。
在真实 LLM 推理中，通信-计算的交互模式可能完全不同：

- 多层叠加（32+ 层 MLP+Attn 的连续通信）
- KV cache 管理引入额外内存操作
- Batching/scheduling 影响 pipeline 行为
- 实际模型 shape 下的通信-计算比值

**目标：发现合成实验中看不到的问题，这些真实问题才是创新点的最佳来源。**

## 二、部署方案

### 模型选择

**Qwen2.5-72B-Instruct**（纯文本，不用 VL 版——ViT 会混入无关的通信模式）

- 参数量：72B，fp16 约 144 GB
- 16 卡 × 64 GB = 1 TB，每卡 ~9 GB 权重，充裕
- TP=16 是生产环境典型配置

### 安装

```bash
# 使用海光 DTK 适配版（不是社区原版——原版用 CUDA 内核，BW1000 不兼容）
pip install -r requirements.txt \
  -i https://pypi.sourcefind.cn/release/dtk2604/ \
  --no-deps

# 如果 requirements 方式有问题，备选：
pip install vllm \
  -i https://pypi.sourcefind.cn/release/dtk2604/ \
  --no-deps

# 查看源上有什么版本：
pip index versions vllm -i https://pypi.sourcefind.cn/release/dtk2604/
```

**前提**：DTK 版 PyTorch 已装好（确认 `python3 -c "import torch; print(torch.__version__)"` 能跑）

### 启动（双机 16 卡）

```bash
# 使用 vLLM 的多机启动方式（Ray 或 torchrun，取决于海光适配版支持哪种）
# 具体启动命令需要根据海光版 vLLM 的文档调整

# 环境变量
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,COLLECTIVE
# RDMA 环境变量（参照 phase5 的 run_phase5_scnet.sh 里的 NET=rdma 模式）

vllm serve Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 16 \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --log-request-stats
```

### 模型下载

72B 模型约 144 GB，需要提前下载到 SCNet 可访问的存储：
- HuggingFace：`Qwen/Qwen2.5-72B-Instruct`
- 或 ModelScope（国内更快）：`qwen/Qwen2.5-72B-Instruct`

```bash
# 用 modelscope 下载（国内网络）
pip install modelscope
modelscope download --model qwen/Qwen2.5-72B-Instruct --local_dir ./Qwen2.5-72B-Instruct
```

## 三、部署后要观察什么（四个关键问题）

| # | 观察什么 | 怎么看 | 对应已有发现 |
|---|---|---|---|
| 1 | **TP 推理中 AG/RS 实际占多少时间？** | NCCL debug 日志 + vLLM timeline | Phase 5 Q1 测的 37%（合成），真实场景是多少？ |
| 2 | **多层叠加时通信行为？** | 32 层连续推理的 timeline | Phase 5 Q4 的"零惩罚"在真实 workload 里成立吗？ |
| 3 | **有没有 GPU 空转（等数据）？** | rocm-smi / 性能计数器 | 合成实验看不到的 pipeline 气泡 |
| 4 | **实际到达模式？** | 逐层计时 + NCCL 日志 | Phase 4 的洗牌在真实场景是否更严重？ |

## 四、可能遇到的问题

| 问题 | 预期解决方案 |
|---|---|
| 海光版 vLLM 版本较旧 | 只要 TP 推理能跑就行，不需要最新特性 |
| RCCL 与 vLLM 的集成问题 | 参照 phase5 的 RDMA 配置（shca 插件） |
| 72B 模型下载慢 | 用 ModelScope，或先试 32B 验证流程 |
| 双机启动方式（Ray/torchrun） | 根据海光版文档调整，可能需要手动设置 |

## 五、给海光侧的说明

- 这批实验是**探索性的**——不预设结论，跑起来看有什么问题
- 如果 vLLM 部署本身遇到困难（安装/启动/兼容性），把报错发到交流窗，标【急】
- 部署成功后，先用简单 prompt 跑几轮推理，把基本 timeline 和 NCCL 日志发过来
- 不需要做复杂的 benchmark——先跑起来，看看有什么

## 六、与前面 Phase 的关系

```
Phase 4（到达异质性）→ 证明"到达时间有差异"
Phase 5（端到端分解）→ 证明"朴素重叠失败 + 并发安全"
Phase 6（vLLM 部署）→ 看真实场景下这些问题是否存在、是否更严重、有没有新问题
```

Phase 6 的目的是**发现真实问题**——如果 vLLM 推理中暴露了合成实验没看到的新瓶颈，那些就是论文创新点的最佳来源。
