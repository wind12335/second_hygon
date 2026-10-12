#!/usr/bin/env python3
"""vLLM 负载发生器:并发打满 N 秒,输出吞吐/延迟统计(Phase 6 profiler 采集窗口用).

用法: loadgen.py <host> <port> <model_path> <conc> <dur> <maxtok> [mode] [ptokens] [seed]
  mode = natural(默认,自然语言 prompt,真实 prefill/decode 混合)
       | fixed   (token-id 定长 prompt,ptokens 个随机 token → prefill GEMM 的 M≈ptokens,
                  与微基准 M 系列直接可比;逐请求随机,避开 prefix cache 命中)
  seed = fixed 模式随机种子(默认 20261009;多相位扫描时每相位换种子,
         防止不同相位生成相同 prompt 命中 prefix cache)
"""
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

host, port, model = sys.argv[1], sys.argv[2], sys.argv[3]
conc, dur, maxtok = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
mode = sys.argv[7] if len(sys.argv) > 7 else "natural"
ptok = int(sys.argv[8]) if len(sys.argv) > 8 else 0
seed = int(sys.argv[9]) if len(sys.argv) > 9 else 20261009

try:
    has_chat = bool(json.load(open(model + "/tokenizer_config.json")).get("chat_template"))
except Exception:
    has_chat = False
base = f"http://{host}:{port}/v1"

if mode == "fixed":
    VOCAB = json.load(open(model + "/config.json")).get("vocab_size", 128256)
    rng = random.Random(seed)
    tag = f"[loadgen:fixed ptok={ptok}]"
else:
    PROMPT = ("请就\"高性能计算中通信与计算的重叠\"写一段论述文字,不少于200字。"
              "Discuss the overlap of communication and computation in HPC in detail.")
    tag = "[loadgen:natural]"

def one(_):
    t0 = time.time()
    try:
        if mode == "fixed":
            body = {"model": model, "max_tokens": maxtok, "ignore_eos": True,
                    "prompt": [rng.randrange(0, VOCAB) for _ in range(ptok)]}
            target = base + "/completions"
        elif has_chat:
            body = {"model": model, "max_tokens": maxtok,
                    "messages": [{"role": "user", "content": PROMPT}]}
            target = base + "/chat/completions"
        else:
            body = {"model": model, "max_tokens": maxtok, "prompt": PROMPT}
            target = base + "/completions"
        r = urllib.request.urlopen(
            urllib.request.Request(target, json.dumps(body).encode(),
                                   {"Content-Type": "application/json"}),
            timeout=600)
        r.read()
        return time.time() - t0, True
    except Exception as e:
        print(f"{tag} request failed: {e}", flush=True)
        return time.time() - t0, False

lat, ok = [], 0
t_end = time.time() + dur
with ThreadPoolExecutor(conc) as ex:
    while time.time() < t_end:
        for t, good in ex.map(one, range(conc)):
            lat.append(t)
            ok += good
lat.sort()
if lat:
    print(f"{tag} conc={conc} dur={dur}s maxtok={maxtok} -> "
          f"{len(lat)} reqs, ok {ok}, lat ms p50={lat[len(lat)//2]*1e3:.0f} "
          f"p90={lat[int(len(lat)*0.9)]*1e3:.0f} max={lat[-1]*1e3:.0f}, "
          f"throughput={ok/max(dur,1):.1f} req/s", flush=True)
else:
    print(f"{tag} no completed requests!", flush=True)
