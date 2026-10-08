# 【NVIDIA 侧 → 海光侧】Phase 6 新任务：BW1000 双机部署 vLLM 真实推理

> 发信人：一作（NVIDIA 侧）
> 时间：2026-10-08
> 优先级：高（发现真实问题的核心实验）
> 需要资源：SCNet 模型训练模块，双机 × 8 卡 BW1000

---

## 背景（1 分钟）

Phase 4 证明了到达异质性存在（TCP 128× / RDMA 1.3×），Phase 5 证明了朴素细粒度重叠灾难性失败（-207%）。但这些都是合成实验。现在需要在**真实 LLM 推理框架**中观察通信-计算交互，发现合成实验暴露不了的问题。

## 要做什么

**在 BW1000 双机上部署 vLLM（海光 DTK 适配版），跑 Qwen2.5-72B-Instruct TP=16 推理。**

详细说明在 `phase6-vllm-deployment/README.md`。

### 安装

```bash
# 用海光 DTK 源（不是社区原版——原版 CUDA 内核不兼容 BW1000）
pip install -r requirements.txt \
  -i https://pypi.sourcefind.cn/release/dtk2604/ \
  --no-deps
```

### 模型

**Qwen2.5-72B-Instruct**（纯文本版，不用 VL）——72B 是 TP=16 的生产典型大小。
如果 72B 下载/加载有困难，可以先用 **Qwen2.5-32B** 验证流程。

### 启动后观察什么

1. TP 推理中 AG/RS 通信占多少时间（NCCL debug 日志）
2. 多层叠加时通信是否有竞争/气泡（timeline）
3. GPU 利用率有没有空转间隙（等数据时 SM 空闲）
4. 逐层通信到达时间模式（与 Phase 4 的洗牌对比）

## 注意事项

- vLLM 只是**观察工具**——不需要 benchmark，跑起来看行为就行
- 遇到安装/启动问题把报错发到交流窗，标【急】
- 模型从 ModelScope 下载比 HuggingFace 快（国内网络）
- RDMA 环境变量参照 phase5 的 `run_phase5_scnet.sh` NET=rdma 模式

## Phase 5 结果确认

Phase 5 的四问数据质量很好，感谢。三个可选补充我们评估后回复：
1. TCP 对照批——**值得跑**（一行数据补齐传输维度，建议做）
2. M=16384 解锁 Q3 细档——暂缓（粒度结论已清晰）
3. Q2 固定 2 批变体——暂缓（Phase 6 vLLM 优先级更高）

—— 一作（NVIDIA 侧）
