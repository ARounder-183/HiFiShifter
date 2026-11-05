#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# HiFiShifter CLAP —— macOS / Linux 构建入口。
#
#     scripts/clap/unix/build.sh                  # 编译 → 组装 → 自检 → tar.gz
#     scripts/clap/unix/build.sh --reuse-frontend # 只改 Rust 时跳过 npm
#     scripts/clap/unix/build.sh --stage=bundle   # 只重装 bundle（不打包）
#
# 布局（宿主按这个位置找文件，不是我们自选的）：
#   Linux   dist/clap/HiFiShifter.clap              单文件模块
#           dist/clap/HiFiShifter-resources/        <stem>-resources，与模块同级
#   macOS   dist/clap/HiFiShifter.clap/Contents/…   bundle（Info.plist + MacOS + Frameworks + Resources）
#
# 三条硬规则与 VST3 那条流水线相同：不依赖 PowerShell；产物里的每条动态依赖都能在
# 包内解析（$ORIGIN / @loader_path）；没有真的 dlopen 过一次的产物不算产物。
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
readonly BACKEND_DIR="$REPO_ROOT/backend"
readonly API_NAME="HiFiShifter"
readonly FRONTEND_DIST="$REPO_ROOT/frontend/dist"
readonly MODELS_SRC="$BACKEND_DIR/src-tauri/resources/models"

# shellcheck source=../../plugin/unix/lib.sh
. "$REPO_ROOT/scripts/plugin/unix/lib.sh"

STAGE="all"
PROFILE="release"
OUT_DIR="$REPO_ROOT/dist"
REUSE_FRONTEND=0
SKIP_VERIFY=0

log()  { printf '[clap] %s\n' "$*"; }
step() { printf '\n[clap] ── %s\n' "$*"; }
die()  { printf '[clap] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'USAGE'
用法：scripts/clap/unix/build.sh [选项]

  --stage=all|build|bundle|verify|package
                                     只跑到某个阶段（默认 all）
  --profile=release|dist|dev         cargo profile（默认 release）
  --out=DIR                          产物目录（默认 dist）
  --reuse-frontend                   复用已有 frontend/dist（要求它已存在）
  --no-verify                        跳过自检
  -h, --help                         显示本说明
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --stage=*)        STAGE="${1#*=}"; shift ;;
        --stage)          STAGE="$2"; shift 2 ;;
        --profile=*)      PROFILE="${1#*=}"; shift ;;
        --out=*)          OUT_DIR="${1#*=}"; shift ;;
        --out)            OUT_DIR="$2"; shift 2 ;;
        --reuse-frontend) REUSE_FRONTEND=1; shift ;;
        --no-verify)      SKIP_VERIFY=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *)                die "未知参数：$1（--help 看用法）" ;;
    esac
done

case "$STAGE" in
    all|build|bundle|verify|package) ;;
    *) die "--stage 只能是 all|build|bundle|verify|package，收到：$STAGE" ;;
esac

hfs_detect_platform
TARGET_DIR="${CARGO_TARGET_DIR:-$BACKEND_DIR/target}"
PROFILE_DIR="$PROFILE"
[ "$PROFILE" = "dev" ] && PROFILE_DIR="debug"
PRODUCT_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$BACKEND_DIR/src-tauri/tauri.conf.json")"
readonly TARGET_DIR PROFILE_DIR PRODUCT_VERSION

if [ "$HFS_OS" = "linux" ]; then
    CARGO_LIB="libhifishifter_clap.so"
    BUNDLE_DIR="$OUT_DIR/clap"
    MODULE="$BUNDLE_DIR/$API_NAME.clap"
    RESOURCES="$BUNDLE_DIR/$API_NAME-resources"
    BUNDLE_ID="com.hifishifter.clap"
else
    CARGO_LIB="libhifishifter_clap.dylib"
    BUNDLE_DIR="$OUT_DIR/clap/$API_NAME.clap"
    MODULE="$BUNDLE_DIR/Contents/MacOS/$API_NAME"
    RESOURCES="$BUNDLE_DIR/Contents/Resources"
    BUNDLE_ID="com.hifishifter.clap"
fi
ARTIFACT="$OUT_DIR/${API_NAME}_v${PRODUCT_VERSION}_${HFS_TAG}-clap.tar.gz"

log "平台 ${HFS_TAG}｜版本 ${PRODUCT_VERSION}｜profile ${PROFILE}｜输出 $OUT_DIR"

# ── 阶段 1：编译 ─────────────────────────────────────────────────────────────
stage_build() {
    step "编译 hifishifter-clap（profile=${PROFILE}）"
    command -v cargo >/dev/null 2>&1 || die "缺少 cargo"
    mkdir -p "$OUT_DIR"
    local cargo_log="$TARGET_DIR/cargo-clap.log"
    mkdir -p "$(dirname "$cargo_log")"
    # 退出码单独取、颜色强制关掉、失败时按 error 行定位：原因见 lib.sh 的
    # hfs_report_build_failure 与 vst3/unix/build.sh 的同名注释。
    local status=0
    CARGO_TERM_COLOR=never cargo build --manifest-path "$BACKEND_DIR/Cargo.toml" \
        -p hifishifter-clap --profile "$PROFILE" --locked >"$cargo_log" 2>&1 || status=$?
    if [ "$status" -ne 0 ]; then
        hfs_report_build_failure "$cargo_log" "$status"
        die "编译失败（exit ${status}），完整日志：$cargo_log"
    fi
    local lib="$TARGET_DIR/$PROFILE_DIR/$CARGO_LIB"
    [ -f "$lib" ] || die "编译成功但找不到产物：$lib"
    log "产物 ${lib}（$(du -h "$lib" | cut -f1)）"
}

# ── 阶段 2：前端产物 ─────────────────────────────────────────────────────────
stage_frontend() {
    if [ "$REUSE_FRONTEND" = "1" ]; then
        [ -s "$FRONTEND_DIST/plugin.html" ] || die "--reuse-frontend 但 $FRONTEND_DIST/plugin.html 不存在"
        log "按 --reuse-frontend 复用已有前端产物"
        return
    fi
    step "构建前端（vite，含 plugin.html 入口）"
    command -v node >/dev/null 2>&1 || die "缺少 node：需要 ^20.19.0 或 >=22.12.0"
    if [ ! -d "$REPO_ROOT/frontend/node_modules" ]; then
        ( cd "$REPO_ROOT/frontend" && npm ci --include=dev --no-audit --no-fund ) \
            || die "npm ci 失败（检查网络与 frontend/package-lock.json）"
    fi
    ( cd "$REPO_ROOT/frontend" && npm run build ) || die "前端构建失败"
    [ -s "$FRONTEND_DIST/plugin.html" ] || die "前端构建完成但 $FRONTEND_DIST/plugin.html 不存在"
    log "前端产物就绪：$FRONTEND_DIST"
}

# ── 阶段 3：组装 ─────────────────────────────────────────────────────────────
#
# 【为什么必须带运行时依赖】hifishifter-clap/build.rs 声明了 -Wl,-rpath,$ORIGIN
# （macOS 是 @loader_path/../Frameworks）。那是"允许在模块旁边找库"，不是"库会在那里"。
# 之前的 CLAP 脚本只拷了模块，于是 ldd 报 not found、macOS 的 dlopen 直接
# Library not loaded: @rpath/libSoundTouchDLL.dylib —— 承诺在代码里，兑现没人做。
stage_bundle() {
    step "组装 CLAP（${HFS_TAG}）"
    local lib="$TARGET_DIR/$PROFILE_DIR/$CARGO_LIB"
    [ -f "$lib" ] || die "缺少编译产物：${lib}（先跑 --stage=build）"

    rm -rf "$OUT_DIR/clap"
    if [ "$HFS_OS" = "linux" ]; then
        mkdir -p "$BUNDLE_DIR" "$RESOURCES/frontend" "$RESOURCES/models"
        cp -f "$lib" "$MODULE"
        hfs_stage_runtime_deps "$TARGET_DIR/$PROFILE_DIR" "$BUNDLE_DIR"
    else
        mkdir -p "$BUNDLE_DIR/Contents/MacOS" "$BUNDLE_DIR/Contents/Frameworks" \
                 "$RESOURCES/frontend" "$RESOURCES/models"
        cp -f "$lib" "$MODULE"
        hfs_stage_runtime_deps "$TARGET_DIR/$PROFILE_DIR" "$BUNDLE_DIR/Contents/Frameworks"
    fi

    hfs_stage_frontend "$RESOURCES" "$FRONTEND_DIST"
    hfs_stage_models "$RESOURCES" "$MODELS_SRC"

    if [ "$HFS_OS" = "macos" ]; then
        hfs_write_info_plist "$BUNDLE_DIR/Contents/Info.plist" "$API_NAME" "$BUNDLE_ID" "$PRODUCT_VERSION" "!hfc"
        hfs_sign_bundle "$BUNDLE_DIR" "$MODULE"
    fi
    hfs_assert_rpath "$MODULE"
    hfs_check_no_dangling "$OUT_DIR/clap"
    log "bundle 就绪：$BUNDLE_DIR"
}

# ── 阶段 4：自检 ─────────────────────────────────────────────────────────────
stage_verify() {
    step "验证 CLAP 产物"
    bash "$SCRIPT_DIR/../verify.sh" "$MODULE"
}

# ── 阶段 5：打包 ─────────────────────────────────────────────────────────────
stage_package() {
    step "打包 $(basename "$ARTIFACT")"
    local stage="$OUT_DIR/stage-clap"
    rm -rf "$stage"
    mkdir -p "$stage"
    if [ "$HFS_OS" = "linux" ]; then
        cp -f "$MODULE" "$stage/"
        cp -R "$RESOURCES" "$stage/"
        hfs_stage_licenses "$stage" "$REPO_ROOT"
        hfs_tarball "$ARTIFACT" "$stage" "$API_NAME.clap" "$API_NAME-resources" LICENSE licenses
    else
        cp -R "$BUNDLE_DIR" "$stage/"
        hfs_stage_licenses "$stage" "$REPO_ROOT"
        hfs_tarball "$ARTIFACT" "$stage" "$API_NAME.clap" LICENSE licenses
    fi
    rm -rf "$stage"
}

case "$STAGE" in
    build)   stage_build ;;
    bundle)  stage_build; stage_frontend; stage_bundle ;;
    verify)  stage_verify ;;
    package) stage_build; stage_frontend; stage_bundle; stage_verify; stage_package ;;
    all)
        stage_build; stage_frontend; stage_bundle
        if [ "$SKIP_VERIFY" = "1" ]; then log "按 --no-verify 跳过自检"; else stage_verify; fi
        stage_package
        ;;
esac

log "完成"
