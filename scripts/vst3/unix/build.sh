#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# HiFiShifter VST3 —— macOS / Linux 构建入口。
#
#     scripts/vst3/unix/build.sh                  # 一条命令：SDK → 前端 → 编译 → bundle → 自检 → tar.gz
#     scripts/vst3/unix/build.sh --skip-frontend  # 只改 Rust 时跳过 npm
#     scripts/vst3/unix/build.sh --install        # 顺便装进 ~/.vst3
#
# 产物在 dist/：HiFiShifter.vst3、<artifact>.tar.gz 与 .sha256。
#
# 三条硬规则：
#   1. 不依赖 PowerShell，也不依赖 Windows 的任何脚本；
#   2. 产物里的每条动态依赖都能在 bundle 内解析（$ORIGIN / @loader_path）；
#   3. 组装的最后一步是真的 dlopen 一次 —— 没有加载验证的产物不算产物。
#
# 平台规则（运行时依赖、rpath、前端、许可证、签名）在 scripts/plugin/unix/lib.sh，
# 与 CLAP 共用同一份实现；本脚本只描述 VST3 自己的布局。
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
readonly BACKEND_DIR="$REPO_ROOT/backend"
readonly SDK_PINS="$REPO_ROOT/scripts/env/plugin-sdks.json"
readonly API_NAME="HiFiShifter"
readonly FRONTEND_DIST="$REPO_ROOT/frontend/dist"
readonly MODELS_SRC="$BACKEND_DIR/src-tauri/resources/models"

# shellcheck source=../../plugin/unix/lib.sh
. "$REPO_ROOT/scripts/plugin/unix/lib.sh"

STAGE="all"
PROFILE="release"
OUT_DIR="$REPO_ROOT/dist"
SKIP_VERIFY=0
SKIP_FRONTEND=0
WANT_INSTALL=0

log()  { printf '[vst3] %s\n' "$*"; }
step() { printf '\n[vst3] ── %s\n' "$*"; }
die()  { printf '[vst3] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'USAGE'
用法：scripts/vst3/unix/build.sh [选项]

  --stage=all|sdk|frontend|build|bundle|verify|install|package
                                     只跑到某个阶段（默认 all）
  --profile=release|dist|dev         cargo profile（默认 release）
  --out=DIR                          产物目录（默认 dist）
  --skip-frontend                    跳过前端构建（要求 dist 已存在）
  --install                          组装后装进 ~/.vst3（macOS 是 ~/Library/…/VST3）
  --no-verify                        跳过自检（只在该平台跑不了探针时用）
  -h, --help                         显示本说明
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --stage=*)       STAGE="${1#*=}"; shift ;;
        --stage)         STAGE="$2"; shift 2 ;;
        --profile=*)     PROFILE="${1#*=}"; shift ;;
        --out=*)         OUT_DIR="${1#*=}"; shift ;;
        --skip-frontend) SKIP_FRONTEND=1; shift ;;
        --install)       WANT_INSTALL=1; shift ;;
        --no-verify)     SKIP_VERIFY=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               die "未知参数：$1（--help 看用法）" ;;
    esac
done

case "$STAGE" in
    all|sdk|frontend|build|bundle|verify|install|package) ;;
    *) die "--stage 只能是 all|sdk|frontend|build|bundle|verify|install|package，收到：$STAGE" ;;
esac

# ── 平台表 ───────────────────────────────────────────────────────────────────
#
# 平台差异全部收敛在这里，后面的逻辑一条平台 if 都不该再有。
hfs_detect_platform
if [ "$HFS_OS" = "linux" ]; then
    MODULE_SUBDIR="Contents/${HFS_PLATFORM_ARCH}-linux"
    MODULE_FILE="$API_NAME.so"
    RUNTIME_LIB_SUBDIR="$MODULE_SUBDIR"
    CARGO_LIB="libhifishifter_plugin.so"
else
    MODULE_SUBDIR="Contents/MacOS"
    MODULE_FILE="$API_NAME"
    RUNTIME_LIB_SUBDIR="Contents/Frameworks"
    CARGO_LIB="libhifishifter_plugin.dylib"
fi
readonly MODULE_SUBDIR MODULE_FILE RUNTIME_LIB_SUBDIR CARGO_LIB
readonly BUNDLE_TAG="$HFS_TAG"
readonly OUT_DIR
readonly TARGET_DIR="${CARGO_TARGET_DIR:-$BACKEND_DIR/target}"
PROFILE_DIR="$PROFILE"
if [ "$PROFILE" = "dev" ]; then PROFILE_DIR="debug"; fi
readonly PROFILE_DIR
readonly BUNDLE_DIR="$OUT_DIR/$API_NAME.vst3"
readonly CARGO_LOG="$TARGET_DIR/cargo-build.log"

PRODUCT_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$BACKEND_DIR/src-tauri/tauri.conf.json")"
readonly PRODUCT_VERSION
readonly ARTIFACT_BASE="${API_NAME}_v${PRODUCT_VERSION}_${BUNDLE_TAG}-vst3"

# 读 plugin-sdks.json 的某个字段（python3 是唯一一定在 CI 镜像里的跨平台解析器）。
pin() {
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])' "$SDK_PINS" "$1" "$2"
}

for tool in git python3 tar; do
    command -v "$tool" >/dev/null 2>&1 || die "缺少必需工具：$tool"
done

log "平台 $BUNDLE_TAG｜版本 $PRODUCT_VERSION｜profile $PROFILE｜输出 $OUT_DIR"

# ── 阶段 1：SDK ──────────────────────────────────────────────────────────────
#
# ara2-bridge 的 build.rs 会对两个 SDK 做 triple 校验（仓库 + commit + tree），工作区
# 一脏就拒绝编译。所以这里不"尽力而为"：tree 对不上直接失败，而不是让 cargo 在 20 分钟
# 后以一个看不懂的 panic 结束。
sync_sdk() {
    local name="$1" subdir="$2"
    local repo commit tree dir actual_tree
    repo="$(pin "$name" repository)"
    commit="$(pin "$name" commit)"
    tree="$(pin "$name" tree)"
    dir="$REPO_ROOT/third_party/sdk/$subdir"

    if [ ! -d "$dir/.git" ]; then
        log "初始化 $name 检出：$dir"
        mkdir -p "$(dirname "$dir")"
        git init -q "$dir"
        git -C "$dir" remote add origin "$repo"
    fi
    if ! git -C "$dir" cat-file -e "${commit}^{commit}" 2>/dev/null; then
        log "取 $name${commit:0:12}"
        git -C "$dir" fetch --depth 1 origin "$commit" >/dev/null 2>&1 \
            || git -C "$dir" fetch origin "$commit" >/dev/null 2>&1 \
            || die "$name：无法从 $repo 取得 commit $commit"
    fi
    git -C "$dir" checkout -q --detach --force "$commit"
    actual_tree="$(git -C "$dir" rev-parse "HEAD^{tree}")"
    [ "$actual_tree" = "$tree" ] || die "$name 的 tree 与 pin 不符：期望 $tree，实际 $actual_tree"
    log "$name 校验通过（$(git -C "$dir" rev-parse --short HEAD)）"
}

stage_sdk() {
    step "准备 ARA / VST3 SDK"
    sync_sdk ara ARA_SDK
    # ARA_API 是 ARA_SDK 的子模块，ara2-bridge 同样会对它做 triple 校验。
    git -C "$REPO_ROOT/third_party/sdk/ARA_SDK" submodule update --init --recursive >/dev/null 2>&1 \
        || die "ARA_API 子模块拉取失败"
    sync_sdk vst3 vst3sdk
    export ARA_SDK_DIR="$REPO_ROOT/third_party/sdk/ARA_SDK"
    export ARA_VST3_SDK_DIR="$REPO_ROOT/third_party/sdk/vst3sdk"
    log "ARA_SDK_DIR=$ARA_SDK_DIR"
    log "ARA_VST3_SDK_DIR=$ARA_VST3_SDK_DIR"
}

# ── 阶段 2：前端产物 ─────────────────────────────────────────────────────────
#
# 前端是同一份产物：bundle 里的 Contents/Resources/frontend 与 Windows 完全同构。
# 把 npm 放进同一条流水线，"我本地跑过构建，产物里却没有前端"这个坑就不存在。
stage_frontend() {
    if [ "$SKIP_FRONTEND" = "1" ]; then
        [ -f "$FRONTEND_DIST/plugin.html" ] || die "--skip-frontend 但 $FRONTEND_DIST/plugin.html 不存在"
        log "按 --skip-frontend 复用已有前端产物"
        return
    fi
    step "构建前端（vite，含 plugin.html 入口）"
    # vite 8 用 rolldown，原生绑定声明了 engines: ^20.19.0 || >=22.12.0。
    # 版本不够时的报错（styleText 缺失、@rolldown/binding-* 找不到）看起来都像前端代码的
    # 问题，所以在入口就把真实原因说出来。
    command -v node >/dev/null 2>&1 || die "缺少 node：需要 ^20.19.0 或 >=22.12.0"
    [ "$(node -p 'const [a,b]=process.versions.node.split(".").map(Number); (a===20&&b>=19)||a>22||(a===22&&b>=12)?"yes":"no"')" = "yes" ] \
        || die "node $(node -v) 太旧：vite 8 需要 ^20.19.0 或 >=22.12.0"
    if [ ! -d "$REPO_ROOT/frontend/node_modules" ]; then
        # --include=dev：npm run build 要 tsc（devDependency），NODE_ENV=production 会跳过它。
        ( cd "$REPO_ROOT/frontend" && npm ci --include=dev --no-audit --no-fund ) \
            || die "npm ci 失败（检查网络与 frontend/package-lock.json）"
    fi
    ( cd "$REPO_ROOT/frontend" && npm run build ) || die "前端构建失败"
    [ -f "$FRONTEND_DIST/plugin.html" ] || die "前端构建完成但 $FRONTEND_DIST/plugin.html 不存在"
    log "前端产物就绪：$FRONTEND_DIST"
}

# ── 阶段 3：编译 ─────────────────────────────────────────────────────────────
#
# rust-toolchain.toml 钉死了 channel。在只装过 rustup default stable 的机器上，rustup
# 看到仓库里这个文件会去**下载**那个具体版本 —— 离线时表现为"编译阶段一动不动"。
# stable 常常就是同一个 rustc，版本相同就复用；不同则明确失败（悄悄换工具链比卡住更糟）。
ensure_toolchain() {
    if [ -n "${RUSTUP_TOOLCHAIN:-}" ]; then
        log "工具链由 RUSTUP_TOOLCHAIN=$RUSTUP_TOOLCHAIN 指定"
        return 0
    fi
    command -v rustup >/dev/null 2>&1 || return 0
    local pinned
    pinned="$(sed -n 's/^channel[[:space:]]*=[[:space:]]*"\(.*\)"/\1/p' "$REPO_ROOT/rust-toolchain.toml" 2>/dev/null | head -1)"
    [ -n "$pinned" ] || return 0
    rustup toolchain list 2>/dev/null | grep -qE "^${pinned}([-[:space:]]|$)" && return 0
    if [ "$(RUSTUP_TOOLCHAIN=stable rustc -V 2>/dev/null | awk '{ print $2 }')" = "$pinned" ]; then
        export RUSTUP_TOOLCHAIN=stable
        log "钉死的 $pinned 未安装；复用 stable（rustc 版本相同）"
        return 0
    fi
    die "找不到 rustup 工具链 $pinned（rust-toolchain.toml 钉死）。
  有网络时：rustup toolchain install $pinned
  或临时指定：export RUSTUP_TOOLCHAIN=<已装工具链名>"
}

stage_build() {
    step "编译 hifishifter-plugin（profile=$PROFILE）"
    command -v cargo >/dev/null 2>&1 || die "缺少 cargo"
    ensure_toolchain
    log "$(rustc -V 2>/dev/null || echo 'rustc 版本未知')"
    mkdir -p "$OUT_DIR" "$(dirname "$CARGO_LOG")"
    # --locked：构建必须证明"提交的锁"能编译，而不是现解一棵依赖树。
    # 输出进日志：本 crate 在 Unix 上有近两百条 Windows 专属代码的 dead_code 告警，
    # 直接刷屏会把真正的错误推出视野；失败时回放末尾。
    if ! cargo build --manifest-path "$BACKEND_DIR/Cargo.toml" -p hifishifter-plugin \
            --profile "$PROFILE" --locked >"$CARGO_LOG" 2>&1; then
        tail -60 "$CARGO_LOG" >&2
        die "编译失败，完整日志：$CARGO_LOG"
    fi
    log "编译完成（$(grep -c "^warning" "$CARGO_LOG" || true) 条 warning；日志 $CARGO_LOG）"
    local lib="$TARGET_DIR/$PROFILE_DIR/$CARGO_LIB"
    [ -f "$lib" ] || die "编译成功但找不到产物：$lib"
    log "产物 $lib（$(du -h "$lib" | cut -f1)）"
}

# ── 阶段 4：组装 bundle ──────────────────────────────────────────────────────

# Linux 的编辑器跑在独立进程里：只要在宿主进程里创建 GTK 窗口，那个进程的绘制就整个
# 废掉（实测 xwd 从 3.1 MB 变 0 字节）。插件按"与 .so 同目录"去找它。
build_gui() {
    case "$MODULE_SUBDIR" in *-linux) ;; *) return 0 ;; esac
    log "构建渲染进程 hfs-gui"
    # 显式 fetch --locked 而不是 --offline：后者依赖"插件构建恰好下载过相同版本的 crate"，
    # 锁一旦漂移就以一条离线错误红掉，离真实原因很远。
    ( cd "$REPO_ROOT/backend/hfs-gui" && cargo fetch --locked && \
      cargo build --release --locked --target-dir "$TARGET_DIR/hfs-gui" ) || die "hfs-gui 构建失败"
    cp -f "$TARGET_DIR/hfs-gui/release/hfs-gui" "$BUNDLE_DIR/$MODULE_SUBDIR/hfs-gui"
    chmod +x "$BUNDLE_DIR/$MODULE_SUBDIR/hfs-gui"
    log "渲染进程就绪（$(du -h "$BUNDLE_DIR/$MODULE_SUBDIR/hfs-gui" | cut -f1)）"
}

# 模型权重不在仓库里（见 scripts/setup/fetch-models.py）：缺了就取，已有则是一次
# 目录扫描。CI 用 actions/cache 以清单哈希为键缓存结果，因此这里通常什么都不做。
stage_models_fetch() {
    step "模型权重"
    python3 "$REPO_ROOT/scripts/setup/fetch-models.py" || die "模型权重获取失败"
}

stage_bundle() {
    step "组装 $API_NAME.vst3（$BUNDLE_TAG）"
    local lib="$TARGET_DIR/$PROFILE_DIR/$CARGO_LIB"
    [ -f "$lib" ] || die "缺少编译产物：$lib（先跑 --stage=build）"

    rm -rf "$BUNDLE_DIR"
    mkdir -p "$BUNDLE_DIR/$MODULE_SUBDIR" "$BUNDLE_DIR/$RUNTIME_LIB_SUBDIR" \
             "$BUNDLE_DIR/Contents/Resources/models" "$BUNDLE_DIR/Contents/Resources/frontend"
    cp -f "$lib" "$BUNDLE_DIR/$MODULE_SUBDIR/$MODULE_FILE"

    hfs_stage_runtime_deps "$TARGET_DIR/$PROFILE_DIR" "$BUNDLE_DIR/$RUNTIME_LIB_SUBDIR"

    build_gui

    hfs_stage_frontend "$BUNDLE_DIR/Contents/Resources" "$FRONTEND_DIST"
    hfs_stage_models "$BUNDLE_DIR/Contents/Resources" "$MODELS_SRC"

    if [ "$HFS_OS" = "macos" ]; then
        hfs_write_info_plist "$BUNDLE_DIR/Contents/Info.plist" "$API_NAME" "com.hifishifter.vst3" "$PRODUCT_VERSION" "!hif"
        hfs_sign_bundle "$BUNDLE_DIR" "$BUNDLE_DIR/$MODULE_SUBDIR/$MODULE_FILE"
    fi
    hfs_assert_rpath "$BUNDLE_DIR/$MODULE_SUBDIR/$MODULE_FILE"
    hfs_check_no_dangling "$BUNDLE_DIR"
    log "bundle 就绪：$BUNDLE_DIR"
}

# ── 阶段 5：自检 ─────────────────────────────────────────────────────────────
stage_verify() {
    step "验证 bundle"
    [ -d "$BUNDLE_DIR" ] || die "还没有 bundle：$BUNDLE_DIR（先跑 --stage=bundle）"
    bash "$SCRIPT_DIR/verify.sh" "$BUNDLE_DIR"
}

# ── 阶段 6：装进用户的 VST3 目录（开发循环用）─────────────────────────────
#
# 默认不执行：往用户的插件目录写东西不该是"构建"的隐含副作用。
stage_install() {
    step "安装到用户的 VST3 目录"
    [ -d "$BUNDLE_DIR" ] || die "还没有 bundle：$BUNDLE_DIR（先跑 --stage=bundle）"
    local target
    if [ "$HFS_OS" = "macos" ]; then
        target="$HOME/Library/Audio/Plug-Ins/VST3"
    else
        target="$HOME/.vst3"
    fi
    mkdir -p "$target"
    rm -rf "${target:?}/$API_NAME.vst3"
    cp -R "$BUNDLE_DIR" "$target/$API_NAME.vst3"
    log "已安装：$target/$API_NAME.vst3（$(du -sh "$target/$API_NAME.vst3" | cut -f1)）"
}

# ── 阶段 7：打包 ─────────────────────────────────────────────────────────────
stage_package() {
    step "打包 $ARTIFACT_BASE"
    local stage="$OUT_DIR/stage"
    rm -rf "$stage"
    mkdir -p "$stage"
    cp -R "$BUNDLE_DIR" "$stage/"
    # 许可证随产物走：根 LICENSE（MIT + 第三方清单）+ licenses/ 目录（全文与声明）。
    # SoundTouch 是 LGPL-2.1（动态链接、上游未修改），ARA SDK 是 Apache-2.0，
    # 声码器权重是 CC BY-NC-SA 4.0 —— 缺任何一份都是许可瑕疵。
    hfs_stage_licenses "$stage" "$REPO_ROOT"
    hfs_tarball "$OUT_DIR/$ARTIFACT_BASE.tar.gz" "$stage" "$API_NAME.vst3" LICENSE licenses
    rm -rf "$stage"
}

# ── 入口 ─────────────────────────────────────────────────────────────────────
case "$STAGE" in
    sdk)      stage_sdk ;;
    frontend) stage_frontend ;;
    build)    stage_sdk; stage_frontend; stage_build ;;
    bundle)   stage_sdk; stage_frontend; stage_build; stage_models_fetch; stage_bundle ;;
    verify)   stage_verify ;;
    install)  stage_sdk; stage_frontend; stage_build; stage_models_fetch; stage_bundle; stage_install ;;
    package)  stage_sdk; stage_frontend; stage_build; stage_models_fetch; stage_bundle; stage_verify; stage_package ;;
    all)
        stage_sdk; stage_frontend; stage_build; stage_models_fetch; stage_bundle
        if [ "$SKIP_VERIFY" = "1" ]; then log "按 --no-verify 跳过自检"; else stage_verify; fi
        stage_package
        ;;
esac

# --install 与 --stage 正交：all/package 之后也能顺便装。
if [ "$WANT_INSTALL" = "1" ] && [ "$STAGE" != "install" ]; then
    stage_install
fi

log "完成"
