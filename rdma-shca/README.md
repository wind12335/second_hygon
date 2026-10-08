# rdma-shca — BW1000 分区 shca RDMA 组件（SCNet 平台管理员提供，2026-10-08）

背景：训练模块镜像的用户态只带 Mellanox OFED provider，枚举不了节点的 shca 400G NDR 网卡
（内核驱动与 /dev/infiniband 挂载均正常，`ibv_devinfo` 却是 0 HCAs），RCCL 只能回退 TCP socket。
管理员提供了下列组件后，RCCL 走外置网络插件 `NET/IBext_v8`（GPUDirect RDMA）全通。

## 文件清单

| 文件 | 作用 | 是否入 git |
|---|---|---|
| `shca-tools_2.500.4.B074-Ubuntu22.04_amd64.deb`（58MB） | shca 用户态：libshca-rdmav34.so（ibverbs provider）、配套 libibverbs/librdmacm rdmav34、perftest 工具 | **否**（共享存储常驻；sha256 见下） |
| `topo_lib/lib/librccl-net-shca.so.0.0.0`（+.so/.so.0 符号链接） | RCCL 外置网络插件，RCCL 认 shca 的正确路径（`NCCL_NET_PLUGIN=shca` → `NET/IBext_v8`） | 是 |
| `topo_lib/lib/librccl-net-shca.a` / `.la` | 同插件静态库（未用，随包保留） | 是 |
| `topo_lib/built-in-508-topo-input-tj-default.xml` | 平台 GPU/网卡拓扑（`NCCL_TOPO_FILE`） | 是 |
| `mlxtoshca_B074.sh` | 官方安装脚本（apt 卸 Mellanox 栈 + dpkg -i 装本包）——**需 apt 网络，pod 内不可用**，仅留档 | 是 |
| `topo_lib.tar.gz` | topo_lib 原始打包（已解压，gitignore `*.tar.gz`） | 否 |

`shca-tools` deb：sha256 `ab64d8b249f11d3dad935e6ee2815345748c1c7806a7f85330802d4fea5e0fdf`，
来源：SCNet 内部文件服务器（管理员提供下载链接，见交流窗 2026-10-08）。

## 用法

**无需手动操作**：`../phase4-arrival-heterogeneity/run_phase4_scnet.sh`（v7+，默认 `NET=rdma`）
每次启动自动完成安装与环境变量；`NET=check` 只输出可转发管理员的分层诊断报告。

手动等价步骤（任一 RCCL 任务通用，pod 文件系统临时，**每个作业一次**，约 10 秒）：

```bash
# 1) 装 shca 用户态（离线：dpkg -x 解包拷库；官方 mlxtoshca 脚本需网络，pod 内不可用）
dpkg -x rdma-shca/shca-tools_2.500.4.B074-Ubuntu22.04_amd64.deb /tmp/shca
cp -a /tmp/shca/usr/lib/x86_64-linux-gnu/. /usr/lib/x86_64-linux-gnu/
#    并删掉镜像里 SONAME 更高的 Mellanox 版本库、把 libibverbs.so.1 等指向 rdmav34 版，ldconfig
#    （细节见 run_phase4_scnet.sh 第 3a 段）

# 2) 管理员配方环境变量
export NCCL_IB_DISABLE=0
export NCCL_NET_PLUGIN=shca
export NCCL_IB_HCA=shca_0:1,shca_1:1,shca_2:1,shca_3:1
export NCCL_TOPO_FILE=$PWD/rdma-shca/topo_lib/built-in-508-topo-input-tj-default.xml
export LD_LIBRARY_PATH=$PWD/rdma-shca/topo_lib/lib:$LD_LIBRARY_PATH
```

验证：`ibv_devinfo -l` 应列出 shca_0..3；任务日志通道应为 `NET/IBext_v8`，出现 `NET/Socket` 即回退。
前提：作业必须由**模型训练模块**创建（notebook 不挂 /dev/infiniband）。
