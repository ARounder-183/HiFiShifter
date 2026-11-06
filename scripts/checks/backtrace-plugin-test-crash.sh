#!/usr/bin/env bash
# 测试进程被信号打死时抓一份栈。
#
#     bash scripts/checks/backtrace-plugin-test-crash.sh [过滤器...]
#
# 不传过滤器时重放默认的两组：渲染路径（画一条音高线 → 声码器）与 VST3 外壳。
# 它们在 CI 上各崩过一次，而 cargo 只留下一句 "signal: 11, SIGSEGV"。
#
# 【为什么单独成文件】"测试判据只能写在 scripts/checks/" 是本仓的硬约定
# （scripts/checks/check-product-consistency.ps1 会拦工作流里的 --skip /
# --test-threads / --offline）。这一条是诊断而不是判据，但它的参数同样不该散落在
# YAML 里 —— 否则下一个人会以为那是门禁的一部分。
#
# 永远以 0 退出：它是诊断，不该把一个已经在红的 job 变成另一种红。
set -uo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
readonly TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/backend/target}"

if ! command -v gdb >/dev/null 2>&1; then
    echo "没有 gdb：跳过栈抓取（macOS runner 不预装）"
    exit 0
fi

BIN="$(ls -t "$TARGET_DIR"/debug/deps/hifishifter_plugin-* 2>/dev/null | grep -v '\.d$' | head -1 || true)"
if [ -z "$BIN" ]; then
    echo "没有测试二进制：跳过（先跑一遍 cargo test --no-run）"
    exit 0
fi
echo "测试二进制：$BIN"

# 【为什么默认按整组重放】崩溃需要"同一个进程里跑过前面上百条用例"这个前提 ——
# 单独跑那条可疑用例是过得去的（实测 71 秒通过）。只重放单条用例等于什么都测不到。
if [ $# -eq 0 ]; then
    set -- 'editor::' 'vst3::'
fi

for filter in "$@"; do
    echo "::group::gdb $filter"
    # 崩了就停在信号上、bt 打的是崩溃现场；没崩就是一次正常的栈，同样有用。
    # 回放必须和正式测试用同一套 EP 判据：脚本可能从本地 shell 或单独的步骤里
    # 被调起，那时 job 级 env 不在场，Dawn 会在这台只有 Lavapipe 的机器上把堆写坏，
    # 于是「抓栈」这一步自己制造一次 SIGSEGV，看起来像原测试又崩了。
    HIFISHIFTER_ORT_EP=cpu \
      gdb -batch -ex run -ex 'thread apply all bt 12' --args "$BIN" "$filter" --test-threads=1 || true
    echo "::endgroup::"
done
exit 0
