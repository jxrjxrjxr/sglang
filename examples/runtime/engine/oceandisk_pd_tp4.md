# OceanDisk PD 单机运行指南（132／134）

**本文用于 chunkload 对照。当前 notify 运行请使用 [notify 指南](https://github.com/jxrjxrjxr/sglang/blob/dev_oceandisk_linker_pd_notify/examples/runtime/engine/oceandisk_pd_tp4.md)。** 132／134 所有分支统一在 `/home/j00977581/sglang-pd` 切换。

适用于已有模型、数据集、GIDS 库／配置及容器的 132、134。代码目录为 `/home/j00977581/sglang-pd`。单机 P 用 GPU0–3、D 用 GPU4–7，各总 TP4；P 开 interleave CP，D 不开 CP。

当前版本包含 CP 启动与 P/D 缓存 key 对齐修复。2026-10-31 的 bench 仍出现 `1842 / Batch_Retrive`（批量 GET 非全成功）；本次更新只发布运行说明，该故障修复仍待实施。

## 1. 每个容器终端设置环境

在 Bash 中粘贴以下内容。两台机器共用同一段，GIDS 配置按主机名选择。

```bash
export H=$(hostname)
case "$H" in
  fbs-h20-01) export HOST_IP=141.111.65.132 ;;
  fbs-h20-02) export HOST_IP=141.111.65.134 ;;
esac
export W=/home/j00977581
export H=$(hostname)
export SGLANG_HOST_IP=$HOST_IP
export REPO=$W/sglang-pd
export PD_BRANCH=dev_oceandisk_linker_pd_chunkload
export TRANSFER_BACKEND=chunkload
export LOGS=$W/pd_logs/$H
export MODEL_PATH=/home/models/GLM-5.2-W4AFP8/
export DATASET_PATH=/home/z00591349/dataset/multifieldqa_zh.jsonl
export OPEXTENSION=$W/gids/lib/libop_extension.so
export LIBGIDS=$W/gids/lib/libgids_kv_host.so
export GIDS_DIR=$W/gids/cfg_$H/
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
mkdir -p $LOGS $W/gids/lib $GIDS_DIR
pdrun() {  # pdrun <运行名> <check|bench|rounds> [变量=值 ...]
  local name=$1 mode=$2; shift 2
  (
    set -o pipefail
    cd "$REPO" || exit
    env "$@" LOG_DIR="$LOGS/$name" sh examples/runtime/engine/run_oceandisk_pd_tp4.sh "$mode" 2>&1 | tee "$LOGS/$name.tee"
  )
}
```

GIDS 库位于 `$W/gids/lib`，8 份配置位于 `$W/gids/cfg_$H/`。132 的 `gids_config_4` 应已将 GID index 修正为 3；134 的配置应已修正本地 `.200` 为 `.201`。模型或数据集放在别处时，修改相应环境变量。

## 2. 在固定目录切换分支

发布仓库为 `https://github.com/jxrjxrjxr/sglang.git`，远端统一为 `mine`。本文对照使用 `dev_oceandisk_linker_pd_chunkload`、`TRANSFER_BACKEND=chunkload`。已有 `sglang-pd` 目录中执行：

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

如果目标分支被旧 worktree 占用，在旧目录执行 `git switch --detach` 后重试；以后都在 `sglang-pd` 切换分支，不创建新目录。

此后每次更新只执行：

```bash
git -C "$REPO" pull --ff-only &&
git -C "$REPO" log --oneline -1
```

两台机器做同一轮对照时，提交号应相同。Git 代理保存在仓库配置中，服务环境不导出代理。

## 3. 运行测试

拉取成功后，在 8 张卡空闲时按需运行；每次换一个运行名，保留旧日志。

```bash
pdrun c_check check
```

```bash
pdrun c_bench_tail0 bench SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS=0
```

```bash
pdrun c_rounds rounds ROUNDS_DIR="$W/Round"
```

rounds 复用本机 `$W/Round/run_rounds.py` 和 `doupocangqiong.txt`，默认 10 条 128K、3 轮。原生 PD 对照在命令末尾加 `TRANSFER_BACKEND=mooncake`；其他多轮规模用 `ROUNDS_ARGS` 覆盖。

服务日志在 `$LOGS/<运行名>/`，客户端输出在 `$LOGS/<运行名>.tee`。rounds 中途查看：

```bash
tail -f "$LOGS/c_rounds/rounds.log"
```

`pdrun` 保留启动脚本退出码；脚本结束、失败或 Ctrl+C 时由启动脚本清理 P、D、router 和客户端进程。check 的通过表示路径检查通过，仍需核对回答及日志里的 `ERR! recv cid`。N/A 或 GET PARTIAL 的 bench 结果应保留用于排查。

P 侧传输日志、超时监控和 fail-fast 默认关闭。已有开关及含义见 [启动脚本](run_oceandisk_pd_tp4.sh) 顶部。
