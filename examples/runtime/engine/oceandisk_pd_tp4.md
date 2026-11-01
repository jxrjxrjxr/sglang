# OceanDisk notify 运行指南（132／134）

**132、134 统一在 `/home/j00977581/sglang-pd` 内切换分支。** 当前运行分支为 `dev_oceandisk_linker_pd_notify`，个人远端为 `mine = https://github.com/jxrjxrjxr/sglang.git`。后续新分支继续使用同一目录，不新建 clone 或 worktree。

notify 的路径是 P 分组 PUT 完成后发送 READY，D 按通知 GET，不在 D 端轮询 EXIST；尾部和 metadata 继续走 Mooncake。P 使用 interleave CP，D 不开 CP。单机 P GPU0–3、D GPU4–7，各总 TP4；router 端口 31345。

## 1. 每个容器终端设置环境

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
pdrun() {  # pdrun <运行名> <check|bench|rounds> [变量=值 ...]
  local name=$1 mode=$2; shift 2
  (
    set -o pipefail
    cd "$REPO" || exit
    env "$@" LOG_DIR="$LOGS/$name" sh examples/runtime/engine/run_oceandisk_pd_tp4.sh "$mode" 2>&1 | tee "$LOGS/$name.tee"
  )
}
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

## 3. 运行 check、bench、多轮对话

拉取成功且 8 张卡空闲后运行。每次换一个运行名，以保留前一次日志。

```bash
pdrun notify_check check
```

```bash
pdrun notify_bench_tail0 bench SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS=0
```

```bash
pdrun notify_rounds rounds ROUNDS_DIR="$W/Round"
```

启动首行应包含 `backend=notify`。脚本自动给 P、D 添加 `--disaggregation-linker-notify`，层组默认 50 层，由 `SGLANG_DISAGGREGATION_LINKER_NOTIFY_LAYERS` 控制。`check` 默认 tail=0；上面的 bench 显式指定 tail=0。rounds 默认保留一个 chunk 的 RDMA 尾部。

rounds 复用本机 `$W/Round/run_rounds.py` 和 `doupocangqiong.txt`，默认 10 条 128K、3 轮。先跑单条 128K 定位时：

```bash
pdrun notify_rounds_one rounds ROUNDS_DIR="$W/Round" ROUNDS_ARGS='--rounds 3 --batch-size 1 --initial-tokens 128000 --increment-tokens 256 --output-len 16 --record-response --request-usage'
```

原生 Mooncake 对照使用同一分支，命令末尾加 `TRANSFER_BACKEND=mooncake`，并使用新的运行名。EXIST 轮询方案的对照需在同一 `sglang-pd` 目录切换到 `dev_oceandisk_linker_pd_chunkload`，设置 `TRANSFER_BACKEND=chunkload`；notify 分支里的 `chunkload` 值只是 notify 的兼容别名。

## 4. 日志与确认

服务日志位于 `$LOGS/<运行名>/`，客户端输出在 `$LOGS/<运行名>.tee`；`pdrun` 保留脚本退出码。脚本结束、失败或 Ctrl+C 时由启动脚本清理 P、D、router 和客户端。

```bash
tail -f "$LOGS/notify_rounds/rounds.log"
```

check 默认打开 trace，bench 默认关闭。check 后查看：

```bash
L=$LOGS/notify_check
grep -E 'PUT_DONE|READY_SEND' "$L/prefill.log"
grep -E 'READY_RECV|LOAD_STARTED|LOAD_DONE' "$L/decode.log"
grep -nE 'ERR! recv cid|Traceback|Generation aborted|Generation returned no output tokens' "$L"/*.log "$L.tee"
```

P 应先完成 PUT 再发 READY；D 收到 READY 后装载对应页／层组。D 的 load tag 为 `rid@start_page:layer_start`。check 报告路径检查通过后，再核对答案与错误日志；客户端遇到流内 abort 或无 token 响应会报错退出。

本地 CPU 回归已覆盖通知顺序、前缀补写、层组 GET 与 DMA 回收；CUDA／RDMA／GIDS 和吞吐由本次黄区实测确认。原生库仍需满足“P save event 完成后，D 可读到相同 key 的已写数据”的契约。
