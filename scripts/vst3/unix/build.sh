#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# HiFiShifter VST3 —— macOS / Linux 构建入口。
#
# 一条命令从零到可安装的 .vst3 bundle + 校验和：
#
#     scripts/vst3/unix/build.sh
#
# 阶段（--stage=…，默认 all）：
#     sdk      按 scripts/env/plugin-sdks.json 的 pin 取 ARA_SDK / vst3sdk 并校验
#     frontend 构建前端产物（vite）：bundle 里必须有 Contents/Resources/frontend
#     build    cargo build -p hifishifter-plugin（默认 release 档：strip，与 Windows 出货档一致）
#     bundle   按 VST3 3.7+ 的目录规范组装 HiFiShifter.vst3（含依赖、模型、Info.plist）
#     verify   布局 / 符号 / 依赖自洽 / 黑盒加载（调用 verify.sh）
#     package  tar.gz + .sha256 + 许可证
#
# 设计取舍见 docs/plans/2026-10-24-vst3-unix-build-plan.md。三条硬规则：
#   1. 不依赖 PowerShell、不依赖 Windows 的任何脚本；
#   2. 产物里的每一条动态依赖都必须能在 bundle 内解析（$ORIGIN / @loader_path）；
#   3. 组装的最后一步是"真的 dlopen 一次"——没有加载验证的产物不算产物。
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
readonly BACKEND_DIR="$REPO_ROOT/backend"
readonly SDK_PINS="$REPO_ROOT/scripts/env/plugin-sdks.json"
readonly API_NAME="HiFiShifter"
readonly FRONTEND_DIST="$REPO_ROOT/frontend/dist"
# 前端构建用的 Node：钉死一个版本，装不下就装到仓库自己的 .build-tmp/tools 下。
#
# 【为什么钉 22.12.0 而不是"最新 LTS"】它正好是 vite 8 / rolldown 那条
# `engines: ^20.19.0 || >=22.12.0` 的下界 —— 比它新的任何 22/24 都行，但"最新"
# 会随上游漂移，而构建脚本不该每天换一个工具链。`--node-download-version` 可覆盖。
readonly NODE_PIN="22.12.0"
readonly NODE_TOOLS="$REPO_ROOT/.build-tmp/tools"

STAGE="all"
PROFILE="release"
OUT_DIR=""
SKIP_VERIFY=0
SKIP_FRONTEND=0
ALLOW_NODE_DOWNLOAD=1

log()  { printf "[vst3] %s\n" "$*"; }
step() { printf "\n[vst3] ── %s\n" "$*"; }
warn() { printf "[vst3] WARNING: %s\n" "$*" >&2; }
die()  { printf "[vst3] ERROR: %s\n" "$*" >&2; exit 1; }

usage() {
    cat <<'USAGE'
用法：scripts/vst3/unix/build.sh [选项]

  --stage=all|sdk|frontend|build|bundle|verify|package
                                                只跑到某个阶段（默认 all）
  --profile=release|dist|dev                    cargo profile（默认 release）
  --out=DIR                                     产物目录（默认 .build-tmp/vst3-unix）
  --skip-frontend                               跳过前端构建（本地只改 Rust 时用；要求 dist 已存在）
  --no-node-download                            缺少可用的 Node 时不去下载（离线机器用）
  --no-verify                                   跳过 verify（只在该平台跑不了探针时用）
  -h, --help                                    显示本说明
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --stage=*)   STAGE="${1#*=}"; shift ;;
        --stage)     STAGE="${2:?--stage needs a value}"; shift 2 ;;
        --profile=*) PROFILE="${1#*=}"; shift ;;
        --out=*)     OUT_DIR="${1#*=}"; shift ;;
        --skip-frontend) SKIP_FRONTEND=1; shift ;;
        --no-node-download) ALLOW_NODE_DOWNLOAD=0; shift ;;
        --no-verify) SKIP_VERIFY=1; shift ;;
        -h|--help)   usage; exit 0 ;;
        *)           die "unknown argument: $1（--help 看用法）" ;;
    esac
done

case "$STAGE" in
    all|sdk|frontend|build|bundle|verify|package) ;;
    *) die "--stage 只能是 all|sdk|frontend|build|bundle|verify|package，收到：$STAGE" ;;
esac

# ── 平台 ─────────────────────────────────────────────────────────────────────
#
# 这里只有一张表：平台 → (bundle 内模块目录, 模块文件名, 依赖目录, 产物架构名)。
# 所有平台差异都收敛在这几行，后面的逻辑一条 if 都不该再有。
case "$(uname -s)" in
    Linux)  HOST_OS="linux" ;;
    Darwin) HOST_OS="macos" ;;
    *)      die "只支持 Linux 与 macOS，当前是 $(uname -s)" ;;
esac
case "$(uname -m)" in
    x86_64|amd64) HOST_ARCH="x86_64" ;;
    arm64|aarch64) HOST_ARCH="arm64" ;;
    *) die "不支持的 CPU 架构：$(uname -m)" ;;
esac

if [ "$HOST_OS" = "linux" ]; then
    [ "$HOST_ARCH" = "arm64" ] && PLATFORM_ARCH="aarch64" || PLATFORM_ARCH="x86_64"
    MODULE_SUBDIR="Contents/${PLATFORM_ARCH}-linux"
    MODULE_FILE="$API_NAME.so"
    RUNTIME_LIB_SUBDIR="$MODULE_SUBDIR"
    CARGO_LIB="libhifishifter_plugin.so"
    BUNDLE_TAG="linux-$PLATFORM_ARCH"
    # 模块 NEEDED 里那两个非系统库（见 plan §3 的 ldd 实测）。
    RUNTIME_DEPS="libSoundTouchDLL.so libwebgpu_dawn.so"
else
    PLATFORM_ARCH="$HOST_ARCH"
    MODULE_SUBDIR="Contents/MacOS"
    MODULE_FILE="$API_NAME"
    RUNTIME_LIB_SUBDIR="Contents/Frameworks"
    CARGO_LIB="libhifishifter_plugin.dylib"
    BUNDLE_TAG="macos-$PLATFORM_ARCH"
    RUNTIME_DEPS="libSoundTouchDLL.dylib libwebgpu_dawn.dylib"
fi

readonly HOST_OS HOST_ARCH PLATFORM_ARCH MODULE_SUBDIR MODULE_FILE RUNTIME_LIB_SUBDIR
readonly CARGO_LIB BUNDLE_TAG RUNTIME_DEPS

if [ -z "$OUT_DIR" ]; then
    OUT_DIR="$REPO_ROOT/.build-tmp/vst3-unix"
fi
readonly OUT_DIR
readonly TARGET_DIR="${CARGO_TARGET_DIR:-$BACKEND_DIR/target}"
# cargo 的 profile 名与输出目录名不是一回事：dev → debug（其余同名）。
PROFILE_DIR="$PROFILE"
[ "$PROFILE" = "dev" ] && PROFILE_DIR="debug"
readonly PROFILE_DIR
readonly BUNDLE_DIR="$OUT_DIR/$API_NAME.vst3"

PRODUCT_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$REPO_ROOT/backend/src-tauri/tauri.conf.json")"
readonly PRODUCT_VERSION
readonly ARTIFACT_BASE="${API_NAME}_v${PRODUCT_VERSION}_${BUNDLE_TAG}-vst3"

# ── 小工具 ───────────────────────────────────────────────────────────────────
sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d" " -f1
    else
        shasum -a 256 "$1" | cut -d" " -f1
    fi
}

# 文件内容的 sha256。用 python3 而不是 `sha256sum`/`shasum`：前者只有 Linux 有、
# 后者只有 macOS 自带，而 `python3` 是 build.sh 已经在用的唯一跨平台解析器
# （见下面的 `pin()`）。
hash_file() {
    python3 - "$1" <<'PY'
import hashlib, sys
with open(sys.argv[1], 'rb') as handle:
    print(hashlib.sha256(handle.read()).hexdigest())
PY
}


# ── 前端工具链：找到（或装一个）能跑 vite 的 Node ───────────────────────────
#
# 【为什么构建脚本要管这件事】本仓的 vite 8 用 rolldown，它的原生绑定声明了
# `engines: ^20.19.0 || >=22.12.0`。而 Ubuntu 24.04 的 `nodejs` 包是 18.19 ——
# 一台"只装了 apt 那些包"的开发机**必然**不满足，而它的报错（`styleText` 缺失、
# `@rolldown/binding-*` 找不到）看起来都像是前端代码的问题。
# 让用户自己去装 Node 是把这个坑推给下一个人；这里直接把一个钉死版本的 Node
# 装进仓库自己的 `.build-tmp/tools`（已被 gitignore），`./build.sh` 就能开箱即用。

# 判定式照抄 vite/rolldown 原生绑定的 `engines`，不自己发明区间。
node_version_ok() {
    local binary="${1:-}"
    [ -n "$binary" ] && [ -x "$binary" ] || return 1
    [ "$("$binary" -p 'const [a, b] = process.versions.node.split(".").map(Number);
        (a === 20 && b >= 19) || a > 22 || (a === 22 && b >= 12) ? "yes" : "no"' 2>/dev/null)" = "yes" ]
}

# Node 官方压缩包的平台标签。
node_platform() {
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)  printf 'linux-x64' ;;
        Linux-aarch64) printf 'linux-arm64' ;;
        Linux-arm64)   printf 'linux-arm64' ;;
        Darwin-x86_64) printf 'darwin-x64' ;;
        Darwin-arm64)  printf 'darwin-arm64' ;;
        *)             printf '' ;;
    esac
}

# 已装好的候选：仓库内缓存 → `~/.nvm` → `~/.local` → PATH 上那个。
node_candidates() {
    printf '%s\n' "$NODE_TOOLS"/node-v"$NODE_PIN"-*/bin/node
    printf '%s\n' "$HOME"/.nvm/versions/node/*/bin/node
    printf '%s\n' "$HOME"/.local/node-v*/bin/node
    printf '%s\n' "$(command -v node 2>/dev/null || true)"
}

ensure_node() {
    local candidate
    while IFS= read -r candidate; do
        [ -n "$candidate" ] || continue
        if node_version_ok "$candidate"; then
            export PATH="$(dirname "$candidate"):$PATH"
            log "使用 Node $("$candidate" -v)（$candidate）"
            return 0
        fi
    done <<< "$(node_candidates)"

    local platform
    platform="$(node_platform)"
    if [ -z "$platform" ]; then
        die "本平台（$(uname -s)-$(uname -m)）没有自动安装 Node 的路径；请自行安装 Node ^20.19.0 或 >=22.12.0 后重跑"
    fi
    local name="node-v$NODE_PIN-$platform"
    local dir="$NODE_TOOLS/$name"
    if [ "$ALLOW_NODE_DOWNLOAD" != "1" ]; then
        die "缺少可用的 Node（需要 ^20.19.0 或 >=22.12.0），且按 --no-node-download 不下载。
  手动装一个即可：
      mkdir -p ~/.local && curl -fsSL https://nodejs.org/dist/v$NODE_PIN/$name.tar.xz | tar -xJ -C ~/.local
      export PATH=~/.local/$name/bin:\$PATH
  然后再跑本脚本（它会自动认 ~/.local 下的那份）。"
    fi
    command -v curl >/dev/null 2>&1 || die "缺少 curl，无法下载 Node（或用 --no-node-download 自行安装）"

    local base="https://nodejs.org/dist/v$NODE_PIN"
    local tmp="$NODE_TOOLS/.download"
    mkdir -p "$NODE_TOOLS" "$tmp"
    log "当前 Node 太旧（$(node -v 2>/dev/null || echo none)），正在装 $name → $NODE_TOOLS"
    curl -fsSL -o "$tmp/$name.tar.xz" "$base/$name.tar.xz" \
        || die "下载 Node 失败：$base/$name.tar.xz（离线机器请用 --no-node-download 并自行安装）"

    # 【为什么校验和不硬编码在脚本里】四个平台各一份，钉死只会随上游漂移而失效；
    # 官方 `SHASUMS256.txt` 是同一批发布的清单，拿它比对才是"钉版本"的正确姿势。
    if curl -fsSL -o "$tmp/SHASUMS256.txt" "$base/SHASUMS256.txt"; then
        local expected actual
        expected="$(awk -v f="$name.tar.xz" '$2 == f { print $1 }' "$tmp/SHASUMS256.txt")"
        actual="$(hash_file "$tmp/$name.tar.xz")"
        [ -n "$expected" ] && [ "$expected" = "$actual" ] \
            || die "Node 压缩包校验失败（期望 ${expected:-无}，实际 $actual）"
        log "Node 压缩包 sha256 校验通过"
    else
        warn "拿不到 SHASUMS256.txt，跳过完整性校验"
    fi

    rm -rf "$dir"
    tar -xJf "$tmp/$name.tar.xz" -C "$NODE_TOOLS" || die "解压 Node 失败"
    rm -rf "$tmp"
    node_version_ok "$dir/bin/node" || die "解压出来的 Node 仍不可用：$dir/bin/node"
    export PATH="$dir/bin:$PATH"
    log "使用 Node $("$dir/bin/node" -v)（$dir）"
}


# ── Rust 工具链：钉死的那个没装时，找一个 rustc 版本**完全相同**的已装工具链 ──
#
# 【为什么必须管这件事】`rust-toolchain.toml` 钉死了 `channel = "1.97.1"`。在只装过
# `rustup default stable` 的机器上，rustup 看到仓库里这个文件就会去**下载**那个具体
# 版本 —— 离线/受限网络下它会长时间重试，表现是构建"卡在编译阶段一动不动"（实测
# 卡了 20 分钟以上，日志里只有一句 `downloading component 'cargo'`）。
#
# 而 `stable` 常常就是同一个 rustc（本机实测 `stable` = rustc 1.97.1）。此时正确
# 做法不是降级、也不是让用户去装网络，而是：**版本相同就复用**，把 `RUSTUP_TOOLCHAIN`
# 指过去。版本不同则明确失败 —— 悄悄换工具链比卡住更糟。
ensure_toolchain() {
    if [ -n "${RUSTUP_TOOLCHAIN:-}" ]; then
        log "工具链由 RUSTUP_TOOLCHAIN=$RUSTUP_TOOLCHAIN 指定"
        return 0
    fi
    command -v rustup >/dev/null 2>&1 || return 0
    local pinned
    pinned="$(sed -n 's/^channel[[:space:]]*=[[:space:]]*"\(.*\)"/\1/p' "$REPO_ROOT/rust-toolchain.toml" 2>/dev/null | head -1)"
    [ -n "$pinned" ] || return 0
    if rustup toolchain list 2>/dev/null | grep -qE "^${pinned}([-[:space:]]|$)"; then
        return 0
    fi
    local line name version
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        name="${line%%[[:space:]]*}"
        version="$(RUSTUP_TOOLCHAIN="$name" rustc -V 2>/dev/null | awk '{ print $2 }')"
        if [ "$version" = "$pinned" ]; then
            export RUSTUP_TOOLCHAIN="$name"
            log "钉死的工具链 $pinned 未安装；复用已装的 $name（rustc $version，版本相同）"
            return 0
        fi
    done <<< "$(rustup toolchain list 2>/dev/null)"
    die "本机没有 rustup 工具链 $pinned（rust-toolchain.toml 钉死），也没有 rustc 版本相同的替代。
  有网络时：rustup toolchain install $pinned
  或临时指定：export RUSTUP_TOOLCHAIN=<已装工具链名>
  已装：$(rustup toolchain list 2>/dev/null | tr '\n' ' ')"
}

# 读 plugin-sdks.json 的某个字段（python3 是唯一跨平台且一定在 CI 镜像里的解析器）。
pin() {
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])' "$SDK_PINS" "$1" "$2"
}

for tool in git python3 tar; do
    command -v "$tool" >/dev/null 2>&1 || die "缺少必需工具：$tool"
done

log "平台 $BUNDLE_TAG｜版本 $PRODUCT_VERSION｜profile $PROFILE｜输出 $OUT_DIR"
# ── 阶段 1：SDK ──────────────────────────────────────────────────────────────
#
# ara2-bridge 的 build.rs 会对两个 SDK 做 triple 校验（仓库 + commit + tree hash），
# 且**工作区一脏就拒绝编译**。所以这里不"尽力而为"：commit 与 tree 对不上直接失败，
# 而不是让 cargo 在 20 分钟后以一个看不懂的 panic 结束。
sync_sdk() {
    local name="$1" subdir="$2"
    local repo commit tree dir
    repo="$(pin "$name" repository)"
    commit="$(pin "$name" commit)"
    tree="$(pin "$name" tree)"
    dir="$REPO_ROOT/third_party/sdk/$subdir"

    if [ ! -d "$dir/.git" ]; then
        log "初始化 $name 检出：$dir"
        mkdir -p "$(dirname "$dir")"
        git init -q "$dir"
        git -C "$dir" remote add origin "$repo" 2>/dev/null || git -C "$dir" remote set-url origin "$repo"
    fi

    if ! git -C "$dir" cat-file -e "${commit}^{commit}" 2>/dev/null; then
        log "取 $name@${commit:0:12}"
        # 先试浅取指定 commit（GitHub 支持）；失败再退回完整 fetch，别在小网管下假装成功。
        git -C "$dir" fetch --depth 1 --filter=blob:none origin "$commit" >/dev/null 2>&1 \
            || git -C "$dir" fetch origin "$commit" >/dev/null 2>&1 \
            || die "$name：无法从 $repo 取得 commit $commit"
    fi

    git -C "$dir" checkout -q --detach --force "$commit"
    local actual_tree
    actual_tree="$(git -C "$dir" rev-parse "HEAD^{tree}")"
    [ "$actual_tree" = "$tree" ] || die "$name 的 tree 与 pin 不符：期望 $tree，实际 $actual_tree"
    log "$name 校验通过（$(git -C "$dir" rev-parse --short HEAD)）"
}

stage_sdk() {
    step "准备 ARA / VST3 SDK"
    command -v git >/dev/null 2>&1 || die "缺少 git"
    sync_sdk ara ARA_SDK
    # ARA_SDK 的 ARA_API 是子模块，ara2-bridge 同样会对它做 triple 校验。
    git -C "$REPO_ROOT/third_party/sdk/ARA_SDK" submodule update --init --recursive >/dev/null 2>&1 \
        || die "ARA_API 子模块拉取失败"
    sync_sdk vst3 vst3sdk

    export ARA_SDK_DIR="$REPO_ROOT/third_party/sdk/ARA_SDK"
    export ARA_VST3_SDK_DIR="$REPO_ROOT/third_party/sdk/vst3sdk"
    log "ARA_SDK_DIR=$ARA_SDK_DIR"
    log "ARA_VST3_SDK_DIR=$ARA_VST3_SDK_DIR"
}

# ── 阶段 2：前端产物 ─────────────────────────────────────────────────────────

# 前端是**同一份**产物：bundle 里的 `Contents/Resources/frontend` 与 Windows 完全同构，
# 所以 `assets()` 的实现也只有一份（见 `editor/resources.rs`）。
#
# 【为什么构建在这里而不是在 CI 里】"我本地跑过 build.sh，产物里却没有前端"是最典型的
# 打包漏项；把 npm 放进同一条流水线，这个坑就不存在。`--skip-frontend` 只给反复调
# Rust 的循环用，而且它要求 `dist` 已经存在（缺了照样失败，不会静默产出空包）。
stage_frontend() {
    if [ "$SKIP_FRONTEND" = "1" ]; then
        warn "按 --skip-frontend 跳过前端构建"
        [ -f "$FRONTEND_DIST/plugin.html" ] \
            || die "跳过前端构建但 $FRONTEND_DIST/plugin.html 不存在"
        # 【为什么还要比时间】`--skip-frontend` 的用途是"只改 Rust 时省一次 npm"。
        # 但改了前端却忘了重建，产出的就是一个**看起来正常、内容是旧的**包 ——
        # 用户报的每一个界面 bug 都会被"我本地是对的"糊过去。这里只警告不失败：
        # 有时确实要用一份手工构建的 dist。
        if [ -n "$(find "$REPO_ROOT/frontend/src" -newer "$FRONTEND_DIST/plugin.html" \
                -name '*.ts' -o -name '*.tsx' -newer "$FRONTEND_DIST/plugin.html" 2>/dev/null | head -1)" ]; then
            warn "frontend/dist 比 frontend/src 旧：这个包里是**旧前端**（去掉 --skip-frontend 重建）"
        fi
        return
    fi
    step "构建前端（vite，含 plugin.html 入口）"
    # 前端工具链：PATH 上的 Node 太旧就找一个能用的（必要时装一个钉死版本）。
    # 见文件上方的 ensure_node —— 这一句就是"为什么报错不再指向前端代码"的全部答案。
    ensure_node
    command -v npm >/dev/null 2>&1 || die "缺少 npm（Node 装好了却没有 npm？检查压缩包是否完整）"
    # 【为什么用 `package-lock.json` 的哈希当判据，而不是"`node_modules` 存在吗"】
    # "存在但残缺"是最常见的形态：一次被中断的 `npm ci`、一次从别处拷来的目录、
    # 或者上游刚加了依赖。那三种情况下"存在"都是 true，而 `npm run build` 会以
    # 几十条 `Cannot find module 'vitest'` 失败 —— 报错指向的是症状，不是原因。
    # 哈希标记把"依赖树是否对应当前锁"变成一条可判定的判据。
    # 标记里**同时**带上 Node 大版本：原生依赖（`@rolldown/binding-linux-x64-gnu` 之类
    # 的可选包）是按平台+ABI 装的。换一个 Node 版本重装之前，那份目录看起来"完整"，
    # 实际缺的正是那个 .node 二进制 —— 报错会是 `Cannot find module
    # '@rolldown/binding-linux-x64-gnu'`，离原因有十万八千里。
    local stamp="$REPO_ROOT/frontend/node_modules/.hfs-lock-hash"
    local want have=""
    want="$(hash_file "$REPO_ROOT/frontend/package-lock.json")-node$(node -v)"
    [ -f "$stamp" ] && have="$(cat "$stamp")"
    if [ "$have" != "$want" ]; then
        log "安装前端依赖（npm ci，锁文件已变或依赖树不完整）"
        # `--include=dev`：`npm run build` 要 `tsc`（devDependency）；CI 上若设了
        # `NODE_ENV=production`，默认会跳过 devDependencies，然后构建失败在一个
        # 与前端无关的地方。
        ( cd "$REPO_ROOT/frontend" && npm ci --include=dev --no-audit --no-fund ) \
            || die "npm ci 失败（检查网络与 frontend/package-lock.json）"
        printf '%s' "$want" >"$stamp"
    fi
    if ! ( cd "$REPO_ROOT/frontend" && npm run build ); then
        die "前端构建失败：若报 Cannot find module，删掉 frontend/node_modules 重跑本脚本"
    fi
    [ -f "$FRONTEND_DIST/plugin.html" ] || die "前端构建完成但 $FRONTEND_DIST/plugin.html 不存在"
    log "前端产物就绪：$FRONTEND_DIST"
}

# ── 阶段 3：编译 ─────────────────────────────────────────────────────────────
stage_build() {
    step "编译 hifishifter-plugin（profile=$PROFILE）"
    command -v cargo >/dev/null 2>&1 || die "缺少 cargo"
    ensure_toolchain
    # 把实际用的 rustc 记进日志：产物与 CI 的差异里，"用的哪把编译器"排第一位。
    log "$(rustc -V 2>/dev/null || echo 'rustc 版本未知')"
    mkdir -p "$OUT_DIR"
    local cargo_log="$OUT_DIR/cargo-build.log"
    # --locked：构建必须证明"提交的锁"能编译，而不是让 CI 现解一棵依赖树。
    # 输出重定向到日志：本 crate 在 Unix 上有近两百条"Windows 专属代码在别处没人用"
    # 的 dead_code 告警，直接刷屏会把真正重要的错误推出视野。失败时回放末尾。
    if ! cargo build --manifest-path "$BACKEND_DIR/Cargo.toml" -p hifishifter-plugin \
            --profile "$PROFILE" --locked >"$cargo_log" 2>&1; then
        tail -60 "$cargo_log" >&2
        die "编译失败，完整日志：$cargo_log"
    fi
    local warnings
    warnings="$(grep -c "^warning" "$cargo_log" || true)"
    log "编译完成（$warnings 条 warning；日志 $cargo_log）"
    local lib="$TARGET_DIR/$PROFILE_DIR/$CARGO_LIB"
    [ -f "$lib" ] || die "编译成功但找不到产物：$lib"
    log "产物 $lib（$(du -h "$lib" | cut -f1)）"
}

# ── 阶段 3：组装 bundle ──────────────────────────────────────────────────────

# macOS 的 .vst3 是 CFBundle：没有 Info.plist 宿主根本不会看这个目录。
write_info_plist() {
    # CFBundleVersion 只接受"点分的整数"：0.1.0-beta.15 → 0.1.0；
    # 完整版本号另存自定义键，报障时人要看到的是后者。
    local numeric
    numeric="$(printf "%s" "$PRODUCT_VERSION" | sed -E "s/^v//; s/-.*$//")"
    case "$numeric" in
        [0-9]*.[0-9]*.[0-9]*) ;;
        *) numeric="0.0.0" ;;
    esac
    local plist="$BUNDLE_DIR/Contents/Info.plist"
    {
        printf '<?xml version="1.0" encoding="UTF-8"?>\n'
        printf '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        printf '<plist version="1.0">\n<dict>\n'
        printf '  <key>CFBundleDevelopmentRegion</key><string>en</string>\n'
        printf '  <key>CFBundleExecutable</key><string>%s</string>\n' "$API_NAME"
        printf '  <key>CFBundleIdentifier</key><string>com.hifishifter.vst3</string>\n'
        printf '  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>\n'
        printf '  <key>CFBundleName</key><string>%s</string>\n' "$API_NAME"
        printf '  <key>CFBundleDisplayName</key><string>%s</string>\n' "$API_NAME"
        printf '  <key>CFBundlePackageType</key><string>BNDL</string>\n'
        printf '  <key>CFBundleSignature</key><string>!hif</string>\n'
        printf '  <key>CFBundleShortVersionString</key><string>%s</string>\n' "$numeric"
        printf '  <key>CFBundleVersion</key><string>%s</string>\n' "$numeric"
        printf '  <key>NSHighResolutionCapable</key><true/>\n'
        printf '  <key>HiFiShifterVersion</key><string>%s</string>\n' "$PRODUCT_VERSION"
        printf '</dict>\n</plist>\n'
    } > "$plist"
    log "Info.plist 已生成（CFBundleVersion=$numeric）"
}

sign_bundle() {
    command -v codesign >/dev/null 2>&1 || { warn "没有 codesign，跳过签名"; return 0; }
    # arm64 上未签名的 Mach-O 根本加载不了；ad-hoc 签名是"能跑"的下限。
    # 真正的发布签名 / 公证需要开发者证书，属发布决策，构建脚本不假装拥有它。
    local file
    for file in "$BUNDLE_DIR/$RUNTIME_LIB_SUBDIR"/*.dylib; do
        [ -e "$file" ] || continue
        codesign --force --sign - "$file" >/dev/null 2>&1 || die "依赖签名失败：$file"
    done
    codesign --force --sign - "$BUNDLE_DIR/$MODULE_SUBDIR/$MODULE_FILE" >/dev/null 2>&1 \
        || die "模块签名失败"
    codesign --force --sign - "$BUNDLE_DIR" >/dev/null 2>&1 || die "bundle 签名失败"
    log "已完成 ad-hoc 签名"
}

stage_bundle() {
    step "组装 $API_NAME.vst3（$BUNDLE_TAG）"
    local lib="$TARGET_DIR/$PROFILE_DIR/$CARGO_LIB"
    [ -f "$lib" ] || die "缺少编译产物：$lib（先跑 --stage=build）"

    rm -rf "$BUNDLE_DIR"
    mkdir -p "$BUNDLE_DIR/$MODULE_SUBDIR" "$BUNDLE_DIR/$RUNTIME_LIB_SUBDIR" \
             "$BUNDLE_DIR/Contents/Resources/models"

    cp -f "$lib" "$BUNDLE_DIR/$MODULE_SUBDIR/$MODULE_FILE"

    local dep src
    for dep in $RUNTIME_DEPS; do
        src="$TARGET_DIR/$PROFILE_DIR/$dep"
        [ -e "$src" ] || die "缺少运行时依赖：$src（先跑 --stage=build）"
        # -L：ORT 的 dawn 在 target 下是**符号链接**，直接 cp 会在压缩包里留一个悬空链接。
        cp -Lf "$src" "$BUNDLE_DIR/$RUNTIME_LIB_SUBDIR/$dep"
        log "依赖 $dep（$(du -hL "$BUNDLE_DIR/$RUNTIME_LIB_SUBDIR/$dep" | cut -f1)）"
    done

    mkdir -p "$BUNDLE_DIR/Contents/Resources/frontend"
    [ -f "$FRONTEND_DIST/plugin.html" ] || die "缺少前端产物：$FRONTEND_DIST/plugin.html（先跑 --stage=frontend）"
    cp -f "$FRONTEND_DIST/plugin.html" "$BUNDLE_DIR/Contents/Resources/frontend/"
    if [ -d "$FRONTEND_DIST/assets" ]; then
        cp -R "$FRONTEND_DIST/assets" "$BUNDLE_DIR/Contents/Resources/frontend/"
    fi
    log "前端资源（$(du -sh "$BUNDLE_DIR/Contents/Resources/frontend" | cut -f1)）"

    cp -R "$REPO_ROOT/backend/src-tauri/resources/models/." "$BUNDLE_DIR/Contents/Resources/models/"
    [ -f "$BUNDLE_DIR/Contents/Resources/models/models.json" ] \
        || die "模型清单缺失：先跑 scripts/build/write-models-manifest.ps1 生成 models.json"

    if [ "$HOST_OS" = "macos" ]; then
        write_info_plist
        sign_bundle
    fi

    log "bundle 就绪：$BUNDLE_DIR"
}
# ── 阶段 5：自检 ─────────────────────────────────────────────────────────────
stage_verify() {
    step "验证 bundle"
    [ -d "$BUNDLE_DIR" ] || die "还没有 bundle：$BUNDLE_DIR（先跑 --stage=bundle）"
    bash "$SCRIPT_DIR/verify.sh" "$BUNDLE_DIR"
}

# ── 阶段 6：打包 ─────────────────────────────────────────────────────────────
stage_package() {
    step "打包 $ARTIFACT_BASE"
    local stage="$OUT_DIR/stage"
    rm -rf "$stage"
    mkdir -p "$stage"
    cp -R "$BUNDLE_DIR" "$stage/"

    # 许可证随产物走：ARA SDK 是 Apache-2.0，VST3 SDK 自 3.8.0 起是 MIT，
    # SoundTouch 是 LGPL-2.1（动态链接 + 本仓的修改说明）。缺一份就是许可瑕疵。
    [ -f "$REPO_ROOT/LICENSE" ] && cp "$REPO_ROOT/LICENSE" "$stage/"
    [ -f "$REPO_ROOT/licenses/COPYING.LGPL-2.1" ] && cp "$REPO_ROOT/licenses/COPYING.LGPL-2.1" "$stage/"
    [ -f "$REPO_ROOT/licenses/SOUNDTOUCH-CHANGES.md" ] && cp "$REPO_ROOT/licenses/SOUNDTOUCH-CHANGES.md" "$stage/"

    local tarball="$OUT_DIR/$ARTIFACT_BASE.tar.gz"
    rm -f "$tarball"
    # 保留可执行位与 .vst3 的目录结构；在 stage 里打包，解压出来就是一整棵
    # 可以直接拖进宿主 VST3 目录的树。许可证有就带上，缺了不致命。
    local archive_members=("$API_NAME.vst3")
    local extra
    for extra in LICENSE COPYING.LGPL-2.1 SOUNDTOUCH-CHANGES.md; do
        [ -f "$stage/$extra" ] && archive_members+=("$extra")
    done
    ( cd "$stage" && tar -czf "$tarball" "${archive_members[@]}" )
    local digest
    digest="$(sha256_of "$tarball")"
    printf "%s  %s\n" "$digest" "$(basename "$tarball")" > "$tarball.sha256"
    log "产物 $(du -h "$tarball" | cut -f1)：$tarball"
    log "sha256 $digest"
}

# ── 入口 ─────────────────────────────────────────────────────────────────────
maybe_verify() {
    if [ "$SKIP_VERIFY" = "1" ]; then
        warn "按 --no-verify 跳过自检"
    else
        stage_verify
    fi
}

case "$STAGE" in
    sdk)      stage_sdk ;;
    frontend) stage_frontend ;;
    build)    stage_sdk; stage_frontend; stage_build ;;
    bundle)   stage_sdk; stage_frontend; stage_build; stage_bundle ;;
    verify)   stage_verify ;;
    package)  stage_sdk; stage_frontend; stage_build; stage_bundle; maybe_verify; stage_package ;;
    all)      stage_sdk; stage_frontend; stage_build; stage_bundle; maybe_verify; stage_package ;;
esac

log "完成"