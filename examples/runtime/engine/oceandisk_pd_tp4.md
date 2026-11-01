# OceanDisk notify 运行指南（132／134）

**132、134 统一在 `/home/j00977581/sglang-pd` 内切换分支。** 当前运行分支为 `dev_oceandisk_linker_pd_notify`，个人远端为 `mine = https://github.com/jxrjxrjxr/sglang.git`。后续新分支继续使用同一目录，不新建 clone 或 worktree。

notify 的路径是 P 分组 PUT 完成后发送 READY，D 按通知 GET，不在 D 端轮询 EXIST；尾部和 metadata 继续走 Mooncake。P 使用 interleave CP，D 不开 CP。单机 P GPU0–3、D GPU4–7，各总 TP4；router 端口 31345。

服务和测试分两步运行：先 `sh run_oceandisk_pd_tp4.sh start`，看到三个端口就绪后，再进入 `Round` 目录执行 `sh run_rounds.sh`。多轮脚本前台读取回车，P、D、router 在后台继续运行。

## 1. 启动服务的容器终端设置环境

复用已准备好的模型、数据集、GIDS 库与本机配置，在 Bash 中粘贴。132／134 共用同一段：

```bash
export W=/home/j00977581
export H=$(hostname)
case "$H" in
  fbs-h20-01) export SGLANG_HOST_IP=141.111.65.132 ;;
  fbs-h20-02) export SGLANG_HOST_IP=141.111.65.134 ;;
esac
export REPO=$W/sglang-pd
export PD_BRANCH=dev_oceandisk_linker_pd_notify
export TRANSFER_BACKEND=notify
export LOGS=$W/pd_logs/$H
export MODEL_PATH=/home/models/GLM-5.2-W4AFP8/
export DATASET_PATH=/home/z00591349/dataset/multifieldqa_zh.jsonl
export OPEXTENSION=$W/gids/lib/libop_extension.so
export LIBGIDS=$W/gids/lib/libgids_kv_host.so
export GIDS_DIR=$W/gids/cfg_$H/
export MOONCAKE_IB_DEVICES=mlx5_3,mlx5_5,mlx5_7,mlx5_9,mlx5_11,mlx5_13,mlx5_15
export SGLANG_DISAGGREGATION_LINKER_NOTIFY_LAYERS=50
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
mkdir -p "$LOGS"
```

模型或数据集放在其他位置时，调整对应变量。GIDS 的 8 份配置仍在 `$W/gids/cfg_$H/`；132 的 `gids_config_4` 应已修正为 GID index 3，134 的本地 `.200` 应已修正为 `.201`。

## 2. 切换到 notify 并拉取

首次运行 notify 或以后切换分支时执行下段；换其他分支只需先设置其 `PD_BRANCH`，并采用该分支对应的后端配置。切换前保存当前工作，Git 会在本地改动与目标分支冲突时停止。

```bash
(
  set -e
  PD_URL=https://github.com/jxrjxrjxr/sglang.git
  if git -C "$REPO" remote get-url mine >/dev/null 2>&1; then
    git -C "$REPO" remote set-url mine "$PD_URL"
  else
    git -C "$REPO" remote add mine "$PD_URL"
  fi
  git -C "$REPO" config --replace-all remote.mine.fetch '+refs/heads/*:refs/remotes/mine/*'
  git -C "$REPO" config http.proxy http://141.5.133.251:7777
  git -C "$REPO" config http.sslVerify false
  git -C "$REPO" fetch mine "refs/heads/$PD_BRANCH:refs/remotes/mine/$PD_BRANCH"
  if git -C "$REPO" show-ref --verify --quiet "refs/heads/$PD_BRANCH"; then
    git -C "$REPO" switch "$PD_BRANCH"
  else
    git -C "$REPO" switch --track -c "$PD_BRANCH" "mine/$PD_BRANCH"
  fi
  git -C "$REPO" branch --set-upstream-to="mine/$PD_BRANCH" "$PD_BRANCH"
  git -C "$REPO" pull --ff-only
  git -C "$REPO" status --short --branch
  git -C "$REPO" log --oneline -1
)
```

本次输出的分支应为 `dev_oceandisk_linker_pd_notify`。远端名 `origin` 保留原用途，运行分支统一跟踪 `mine`。

若 Git 提示目标分支已被旧 worktree 占用，先用 `git -C "$REPO" worktree list` 找到旧目录，在该旧目录执行 `git switch --detach`，再回到 `sglang-pd` 重跑上段。旧目录和文件保留；以后运行统一从 `sglang-pd` 发起。

已处于目标分支时，后续更新只需：

```bash
git -C "$REPO" pull --ff-only &&
git -C "$REPO" log --oneline -1
```

两台机器做同一轮对照时，提交号应相同。

## 3. 准备多轮脚本与语料

仓库内 `examples/runtime/engine/Round/run_rounds.py` 原样复用 `D:\CodeSpace\LLMTest\Round` 的版本，保留轮间 `input("等待清洗hbm")`。同目录的 `run_rounds.sh` 只有一条 Python 命令，参数默认 10 条 128K、10 轮、每轮增量 256、输出 16 token，地址改为 router 的 `31345` 端口。

语料继续使用机器上已有的 `/home/j00977581/Round/doupocangqiong.txt`。首次使用仓库内 Round 目录时创建链接；已有该文件或链接时跳过：

```bash
ln -s /home/j00977581/Round/doupocangqiong.txt /home/j00977581/sglang-pd/examples/runtime/engine/Round/doupocangqiong.txt
```

以后只需在 `sglang-pd` 内拉取分支，语料链接保留。

## 4. 第一步：启动 P、D、router

8 张卡空闲后，在已完成第 1 节环境设置的容器终端执行：

```bash
cd /home/j00977581/sglang-pd
LOG_DIR="/home/j00977581/pd_logs/$(hostname)/notify_rounds" sh examples/runtime/engine/run_oceandisk_pd_tp4.sh start
```

脚本依次用 `nohup setsid` 启动 P、D、router。等待 P、D 的 `/health` 和 router 的 `/readiness` 成功后，打印：

```text
Ready: router=http://127.0.0.1:31345 prefill=http://127.0.0.1:31002 decode=http://127.0.0.1:31001
logs=... pids=.../server.pids
```

随后脚本退出并让出终端，三个服务继续运行。启动过程中失败或 Ctrl+C 会清理已启动的服务。启动成功后，服务日志为该 `LOG_DIR` 下的 `prefill.log`、`decode.log`、`router.log`，`server.pids` 依次记录 P、D、router 的进程组 ID。再次启动前按第 7 节停止上一组服务，换运行名可保留原日志。

`start` 默认 context length 为 163840，传输及 bootstrap 超时为 1800 秒；P 开 interleave CP，D 不开 CP。notify 层组默认 50，尾部默认保留一个 chunk 走 Mooncake RDMA。

## 5. 第二步：前台运行多轮测试

看到 `Ready` 后，在同一终端执行：

```bash
cd /home/j00977581/sglang-pd/examples/runtime/engine/Round
sh run_rounds.sh
```

参数直接修改 `run_rounds.sh`。它只运行同目录的 `python3 run_rounds.py`，没有环境设置、后台启动或管道包装。

每轮出现 `等待清洗hbm` 后，在另一个终端按第 6 节执行 flush，成功后回到测试终端按回车继续。原脚本最后一轮也会等待回车；按回车后输出汇总并保存 `rounds.json`，对话记录在同目录 `conversations/`。下一次运行会覆盖这些固定路径，保留结果时先移走或修改参数中的输出路径。

测试结束或 Ctrl+C 只结束多轮客户端，后台服务继续运行。需要再次测试时，直接重跑 `sh run_rounds.sh`。

## 6. 另一个终端：curl 对话与轮间清缓存

以下命令在**同一个 pd 容器**内执行，不依赖第 1 节的环境变量。`--noproxy '*'` 使本机请求绕过黄区代理。

确认端口：

```bash
curl --noproxy '*' -i http://127.0.0.1:31345/readiness
curl --noproxy '*' -sS http://127.0.0.1:31002/model_info
curl --noproxy '*' -sS http://127.0.0.1:31001/model_info
```

通过 router 测一个简单对话，走完整 P/D 路径：

```bash
curl --noproxy '*' -sS http://127.0.0.1:31345/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/home/models/GLM-5.2-W4AFP8/","messages":[{"role":"user","content":"你好，请用一句话介绍自己。"}],"max_tokens":32,"temperature":0,"stream":false}'
```

**只清 D，验证下一轮 D 从 OceanDisk 读回**：

```bash
curl --noproxy '*' -i --max-time 60 -X POST 'http://127.0.0.1:31001/flush_cache?timeout=30'
```

**同时清 P、D，测试两侧缓存清空后的下一轮**：

```bash
curl --noproxy '*' -i --max-time 60 -X POST 'http://127.0.0.1:31002/flush_cache?timeout=30'
curl --noproxy '*' -i --max-time 60 -X POST 'http://127.0.0.1:31001/flush_cache?timeout=30'
```

在轮间暂停、请求已完成时调用。每个接口都返回 HTTP 200 和 `Cache flushed.` 后，再按回车；HTTP 400 表示本次未清理成功，查看响应和对应服务日志。`timeout=30` 允许等待服务空闲。

这里清理的是服务端 KV/radix 缓存和分配状态，保留模型权重及 OceanDisk 中已写的数据。预分配的 KV 池仍在，`nvidia-smi` 的显存占用不需要降到零。清 D 与清 P+D 是不同实验条件，同一组对照保持一致。

## 7. 日志与停止服务

```bash
tail -f "/home/j00977581/pd_logs/$(hostname)/notify_rounds/prefill.log"
tail -f "/home/j00977581/pd_logs/$(hostname)/notify_rounds/decode.log"
```

结束本组实验后，在同一个容器中停止记录的进程组，包括服务子进程：

```bash
while read -r pid; do
  kill -TERM -- "-$pid"
done < "/home/j00977581/pd_logs/$(hostname)/notify_rounds/server.pids"
```

需要查看 PUT/READY/GET 时，启动服务的命令前加 `SGLANG_OCEANDISK_TRACE=1`。默认关闭，避免影响计时。

```bash
grep -E 'PUT_DONE|READY_SEND' "/home/j00977581/pd_logs/$(hostname)/notify_rounds/prefill.log"
grep -E 'READY_RECV|LOAD_STARTED|LOAD_DONE' "/home/j00977581/pd_logs/$(hostname)/notify_rounds/decode.log"
```

P 应先完成 PUT 再发 READY；D 收到 READY 后装载对应页／层组。D 的 load tag 为 `rid@start_page:layer_start`。

## 8. check、bench 与 Mooncake 对照

check、bench 仍保留原来的自动启动、测试、收尾流程；运行前先停止上面的常驻服务，避免占用相同端口。

```bash
cd /home/j00977581/sglang-pd
LOG_DIR="/home/j00977581/pd_logs/$(hostname)/notify_check" sh examples/runtime/engine/run_oceandisk_pd_tp4.sh check
LOG_DIR="/home/j00977581/pd_logs/$(hostname)/notify_bench_tail0" SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS=0 sh examples/runtime/engine/run_oceandisk_pd_tp4.sh bench
```

check 默认 trace 开启、tail=0；上面的 bench 显式指定 tail=0。原生 Mooncake 对照使用同一分支，启动命令前加 `TRANSFER_BACKEND=mooncake` 并更换 `LOG_DIR`，多轮客户端命令相同。EXIST 轮询方案需在同一 `sglang-pd` 目录切到 `dev_oceandisk_linker_pd_chunkload`，按该分支 guide 运行；notify 分支的 `chunkload` 值只是 notify 的兼容别名。
