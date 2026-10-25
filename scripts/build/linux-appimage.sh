#!/usr/bin/env bash
set -euo pipefail

# HiFiShifter AppImage 打包：**唯一**一处决定"包里带什么、运行时要宿主给什么"。
#
# ── 为什么要有这个文件 ─────────────────────────────────────────────────
# AppImage 的默认行为是"把可执行文件的动态依赖尽量复制进来"。对 Tauri 应用而言，
# 这条规则会把宿主桌面栈里**必须由宿主提供**的东西也一起卷进来：WebKitGTK、
# GStreamer、以及一串只有 WebKit 才用的 ICU / libsoup / enchant。于是同一个子
# 系统被劈成两半——GTK 来自包内（Ubuntu 24.04 的副本），libEGL / libwayland-client
# / Mesa 来自宿主。缝上两边的版本一旦对不上，WebKit 就在启动时直接 abort：
#
#     Could not create default EGL display: EGL_BAD_PARAMETER. Aborting...
#
# 这不是假设：v0.1.0-beta.13 的反馈（Arch/CachyOS + AMD RX560，需要用
# LD_PRELOAD 加载 libwayland-client.so 才能启动）就是这条缝的症状。当时的
# "修复"（2026-07-30 的两个提交）用 find -type f 收集库文件，而 Ubuntu 上这些
# SONAME 全是符号链接，于是**一个文件都没打进包**，问题原样保留。
#
# ── 现在的策略：桌面栈整块交给宿主 ─────────────────────────────────────
# 包内只保留"产品自己的东西"和 GTK3（Tauri 的 linuxdeploy-plugin-gtk 的既有
# 行为，全生态的 AppImage 都这么发，ABI 稳定）：
#
#   * 随包：HiFiShifter 可执行文件、libSoundTouchDLL.so、libwebgpu_dawn.so、models
#   * 宿主提供：WebKitGTK、GStreamer、ICU、libsoup、libEGL/GLES、libgbm/libdrm、
#     libwayland-*、libxkbcommon
#
# 这样 EGL / Wayland / Mesa 只有宿主那一份，WebKit 与它们同源，缝不存在。
#
# 代价：宿主必须有 webkit2gtk-4.1（Arch/CachyOS、Ubuntu 24.04+、Fedora 39+、
# Debian 13+、openSUSE Tumbleweed 都提供；Ubuntu 22.04 只有 4.0，不再支持）。
#
# ── 被否决的另一条路 ──────────────────────────────────────────────────
# "把 WebKit 连同 GL / DRM / GStreamer 插件 / DRI 驱动整套打进包"也能消除缝，
# 但要同时背下：包体翻倍、NVIDIA 用户的 libEGL 根本没法随包分发（驱动专有）、
# Mesa 与内核版本强耦合、以及 glvnd vendor JSON 的重定向。那只是换一种方式继续
# 制造同一类故障，所以不采用。
#
# ── 用法 ──────────────────────────────────────────────────────────────
#   linux-appimage.sh package <release-dir>      # 剪裁 → 校验 → 重打包 → 冒烟
#   linux-appimage.sh prune   <AppDir>           # 只剪裁
#   linux-appimage.sh verify  <AppDir>           # 只校验（不修改任何东西）
#   linux-appimage.sh repack  <AppDir> [out]     # 只重打包
#   linux-appimage.sh smoke   <AppImage>         # 只冒烟
#
# 环境变量：
#   HIFISHIFTER_SMOKE_REQUIRED=1    没有 xvfb-run 时视为失败（CI 用）
#   HIFISHIFTER_SMOKE_TIMEOUT=45    冒烟超时秒数
#   ARCH=x86_64|aarch64             appimagetool 的目标架构（默认 uname -m）

SELF_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SELF_DIR/../.." && pwd)

# 必须随包提供、缺一不可的文件（相对 AppDir）。
REQUIRED_FILES=(
    "usr/bin/HiFiShifter"
    "usr/lib/HiFiShifter/libSoundTouchDLL.so"
    "usr/lib/HiFiShifter/libwebgpu_dawn.so"
    "usr/lib/HiFiShifter/models/models.json"
)

# 必须由宿主提供、包内一份都不许留的库（glob，作用于 usr/lib 与 usr/lib64 顶层）。
#
# 判据只有一条：**它的消费者是谁**。WebKit / GStreamer / GL / Wayland 的消费者
# 是宿主，所以这些库必须整组来自宿主；留下半组就是 2026-07 那个故障的复现条件。
HOST_OWNED_GLOBS=(
    'libwebkit2gtk-*.so*'
    'libjavascriptcoregtk-*.so*'
    'libgst*.so*'
    'libicu*.so*'
    'libharfbuzz-icu.so*'
    'libenchant-2.so*'
    'libsoup-3.0.so*'
    'libwayland-*.so*'
    'libxkbcommon*.so*'
    'libgbm.so*'
    'libdrm*.so*'
    'libEGL*.so*'
    'libGLESv2.so*'
    'libglapi.so*'
)

# 可执行文件的直接 NEEDED 里，必须"落在包外"的条目 —— 这就是"单一图形栈"的
# 可执行断言。libwayland-client 单独列出，因为 2026-07 的用户偏方正是 preload 它。
MUST_RESOLVE_HOST=(
    libwebkit2gtk-4.1.so.0
    libjavascriptcoregtk-4.1.so.0
    libsoup-3.0.so.0
    libwayland-client.so.0
)

# 直接 NEEDED 里必须"落在包内"的条目（自有库 + Tauri 打包的 GTK）。
MUST_RESOLVE_BUNDLE=(
    libgtk-3.so.0
    libgdk-3.so.0
    libSoundTouchDLL.so
    libwebgpu_dawn.so
)

usage() {
    cat <<'EOF'
用法：
  linux-appimage.sh package <release-dir>   剪裁 → 校验 → 重打包 → 冒烟（CI 与本地都用这一条）
  linux-appimage.sh prune   <AppDir>        删除必须由宿主提供的库
  linux-appimage.sh verify  <AppDir>        断言包内该有什么、不该有什么、动态解析落在哪里
  linux-appimage.sh repack  <AppDir> [out]  用 appimagetool 重新生成 AppImage
  linux-appimage.sh smoke   <AppImage>      在 xvfb 下启动，要求出现启动日志且没有 abort
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

note() {
    printf '\n=== %s ===\n' "$*"
}

require_dir() {
    [ -d "$1" ] || die "不是目录：$1"
}

normalize_arch() {
    case "${ARCH:-$(uname -m)}" in
        x86_64 | amd64) printf 'x86_64\n' ;;
        aarch64 | arm64) printf 'aarch64\n' ;;
        *) die "不支持的架构：${ARCH:-$(uname -m)}" ;;
    esac
}

# 找一个可用的 appimagetool：先看 PATH，再看缓存，最后下载。
ensure_appimagetool() {
    local arch="$1"
    if command -v appimagetool >/dev/null 2>&1; then
        command -v appimagetool
        return 0
    fi
    local cache_dir="$HOME/.cache/hifishifter"
    local cached="$cache_dir/appimagetool-$arch.AppImage"
    if [ ! -x "$cached" ]; then
        mkdir -p "$cache_dir"
        printf 'downloading appimagetool (%s)...\n' "$arch" >&2
        curl -fsSL -o "$cached" \
            "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$arch.AppImage" \
            || die "appimagetool 下载失败；请把它放到 PATH 上"
        chmod +x "$cached"
    fi
    printf '%s\n' "$cached"
}

# ── prune ─────────────────────────────────────────────────────────────

cmd_prune() {
    local appdir="$1"
    require_dir "$appdir"

    local before after removed=0
    before=$(du -sm "$appdir" | cut -f1)

    local dir glob path
    for dir in "$appdir/usr/lib" "$appdir/usr/lib64"; do
        [ -d "$dir" ] || continue
        for glob in "${HOST_OWNED_GLOBS[@]}"; do
            while IFS= read -r -d '' path; do
                rm -f -- "$path"
                removed=$((removed + 1))
            done < <(find "$dir" -maxdepth 1 -name "$glob" -print0 2>/dev/null)
        done
    done

    # WebKit 的辅助进程（WebKitWebProcess / WebKitNetworkProcess / injected bundle）
    # 与 GStreamer 插件目录：前者只有 WebKit 用，后者与宿主核心版本强绑定。
    while IFS= read -r -d '' path; do
        rm -rf -- "$path"
        removed=$((removed + 1))
    done < <(find "$appdir/usr/lib" -maxdepth 3 -type d \
        \( -name 'webkit2gtk-4.1' -o -name 'webkitgtk-4.1' -o -name 'gstreamer-1.0' \) -print0 2>/dev/null)

    after=$(du -sm "$appdir" | cut -f1)
    printf 'pruned %d entries: %s MiB -> %s MiB\n' "$removed" "$before" "$after"

    if [ "$removed" -eq 0 ]; then
        # 剪裁掉 0 项通常意味着 AppDir 已经剪过，不是错误，但值得说一句。
        printf 'note: 没有匹配到任何宿主库（可能已经剪裁过）\n'
    fi
}

# ── verify ────────────────────────────────────────────────────────────

# 从 ldd 输出里取某个 soname 的解析路径（取不到则为空）。
resolve_of() {
    local ldd_output="$1" soname="$2"
    printf '%s\n' "$ldd_output" | awk -v n="$soname" '$1 == n && $2 == "=>" { print $3; exit }'
}

cmd_verify() {
    local appdir="$1"
    require_dir "$appdir"

    local failures=0
    local f path glob name resolved

    note "随包文件"
    for f in "${REQUIRED_FILES[@]}"; do
        if [ -e "$appdir/$f" ]; then
            printf 'ok    %s\n' "$f"
        else
            printf 'FAIL  %s 不存在\n' "$f" >&2
            failures=$((failures + 1))
        fi
    done

    note "宿主库不得入包"
    local bundled=0
    for glob in "${HOST_OWNED_GLOBS[@]}"; do
        while IFS= read -r -d '' path; do
            printf 'FAIL  %s 必须由宿主提供，却被打了进来\n' "${path#"$appdir"/}" >&2
            failures=$((failures + 1))
            bundled=$((bundled + 1))
        done < <(find "$appdir/usr/lib" "$appdir/usr/lib64" -maxdepth 1 -name "$glob" -print0 2>/dev/null)
    done
    [ "$bundled" -eq 0 ] && printf 'ok    包内没有任何 GL / Wayland / WebKit / GStreamer 库\n'

    note "动态解析"
    local ld_path="$appdir/usr/lib:$appdir/usr/lib/HiFiShifter"
    # GTK 的模块（gdk-pixbuf / immodules）在带 triplet 的子目录里，一并加入搜索路径。
    for path in "$appdir"/usr/lib/*-linux-gnu; do
        [ -d "$path" ] && ld_path="$ld_path:$path"
    done

    local ldd_output
    if ! ldd_output=$(LD_LIBRARY_PATH="$ld_path" ldd "$appdir/usr/bin/HiFiShifter" 2>&1); then
        printf '%s\n' "$ldd_output" >&2
        die "ldd 无法解析 $appdir/usr/bin/HiFiShifter"
    fi

    if printf '%s\n' "$ldd_output" | grep -q 'not found'; then
        printf '%s\n' "$ldd_output" | grep 'not found' >&2
        die "有依赖既不在包里、也不在宿主上"
    fi

    for name in "${MUST_RESOLVE_HOST[@]}"; do
        resolved=$(resolve_of "$ldd_output" "$name")
        if [ -z "$resolved" ]; then
            printf 'FAIL  %s 不在 NEEDED 里 —— 断言已失效，请更新脚本\n' "$name" >&2
            failures=$((failures + 1))
        elif [[ "$resolved" == "$appdir"/* ]]; then
            printf 'FAIL  %s 仍然解析到包内：%s\n' "$name" "$resolved" >&2
            failures=$((failures + 1))
        else
            printf 'ok    %s -> %s（宿主）\n' "$name" "$resolved"
        fi
    done

    for name in "${MUST_RESOLVE_BUNDLE[@]}"; do
        resolved=$(resolve_of "$ldd_output" "$name")
        if [ -z "$resolved" ]; then
            printf 'FAIL  %s 不在 NEEDED 里 —— 断言已失效，请更新脚本\n' "$name" >&2
            failures=$((failures + 1))
        elif [[ "$resolved" != "$appdir"/* ]]; then
            printf 'FAIL  %s 应当随包提供，却解析到 %s\n' "$name" "$resolved" >&2
            failures=$((failures + 1))
        else
            printf 'ok    %s -> 包内\n' "$name"
        fi
    done

    if [ "$failures" -ne 0 ]; then
        die "$failures 项校验未通过"
    fi
    printf '\n全部校验通过\n'
}

# ── repack ────────────────────────────────────────────────────────────

cmd_repack() {
    local appdir="$1" output="${2:-}"
    require_dir "$appdir"

    if [ -z "$output" ]; then
        # 沿用目录里已有的 AppImage 名字；没有就按产品名 + 版本 + 架构拼一个。
        output=$(find "$(dirname "$appdir")" -maxdepth 1 -name '*.AppImage' -print -quit 2>/dev/null || true)
        if [ -z "$output" ]; then
            local version product
            version=$(jq -r '.version' "$REPO_ROOT/backend/src-tauri/tauri.conf.json")
            product=$(jq -r '.productName' "$REPO_ROOT/backend/src-tauri/tauri.conf.json")
            output="$(dirname "$appdir")/${product// /_}_${version}_$(normalize_arch).AppImage"
        fi
    fi

    local arch tool
    arch=$(normalize_arch)
    tool=$(ensure_appimagetool "$arch")

    rm -f -- "$output"
    printf 'repacking %s -> %s (%s)\n' "$appdir" "$output" "$arch"
    APPIMAGE_EXTRACT_AND_RUN=1 ARCH="$arch" "$tool" "$appdir" "$output" >/dev/null
    [ -f "$output" ] || die "appimagetool 没有产出 $output"
    printf 'repacked: %s (%s)\n' "$output" "$(du -h "$output" | cut -f1)"
}

# ── smoke ─────────────────────────────────────────────────────────────

cmd_smoke() {
    local appimage="$1"
    [ -f "$appimage" ] || die "找不到 AppImage：$appimage"

    local runner=()
    if command -v xvfb-run >/dev/null 2>&1; then
        runner=(xvfb-run -a)
    elif [ "${HIFISHIFTER_SMOKE_REQUIRED:-0}" = "1" ]; then
        die "需要 xvfb-run（CI 里请安装 xvfb）"
    else
        printf 'skip: 没有 xvfb-run，跳过冒烟（本地可选装 xvfb）\n' >&2
        return 0
    fi

    local log rc=0 workdir
    log=$(mktemp)
    # 给这次运行一个独立的 TMPDIR：APPIMAGE_EXTRACT_AND_RUN 会按固定名字解包，
    # 复用同一个目录时每次都要重新核对全部文件（几万行 "file size matches"），
    # 既拖慢冒烟也把真正的启动日志埋掉。
    workdir=$(mktemp -d)
    set +e
    TMPDIR="$workdir" APPIMAGE_EXTRACT_AND_RUN=1 timeout "${HIFISHIFTER_SMOKE_TIMEOUT:-45}" \
        "${runner[@]}" "$appimage" >"$log" 2>&1
    rc=$?
    set -e
    rm -rf "$workdir"

    # 124 = 到点了还活着（正常：这是 GUI 应用）；0 / 143 = 自己退出或收到 TERM。
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && [ "$rc" -ne 143 ]; then
        printf '%s\n' '--- 冒烟输出 ---' >&2
        cat "$log" >&2
        rm -f "$log"
        die "AppImage 启动失败（退出码 $rc）"
    fi

    local failed=0
    if ! grep -q 'starting; log file' "$log"; then
        printf '%s\n' 'FAIL  没有出现启动日志' >&2
        failed=1
    fi
    if grep -qiE 'could not create default egl display|Aborting\.\.\.|error while loading shared libraries' "$log"; then
        printf '%s\n' 'FAIL  启动过程中出现致命错误：' >&2
        grep -iE 'could not create default egl display|Aborting\.\.\.|error while loading shared libraries' "$log" >&2
        failed=1
    fi
    if [ "$failed" -ne 0 ]; then
        printf '%s\n' '--- 冒烟输出 ---' >&2
        cat "$log" >&2
        rm -f "$log"
        die "冒烟未通过"
    fi

    printf 'smoke ok（%ss 内保持运行，启动日志正常）\n' "${HIFISHIFTER_SMOKE_TIMEOUT:-45}"
    rm -f "$log"
}

# ── package ───────────────────────────────────────────────────────────

find_appdir() {
    local release_dir="$1"
    local candidate="$release_dir/bundle/appimage/HiFiShifter.AppDir"
    [ -d "$candidate" ] || return 1
    printf '%s\n' "$candidate"
}

cmd_package() {
    local release_dir="$1"
    require_dir "$release_dir"

    local appdir
    appdir=$(find_appdir "$release_dir") || die "找不到 AppDir：$release_dir/bundle/appimage/HiFiShifter.AppDir"

    note "prune"
    cmd_prune "$appdir"

    note "verify"
    cmd_verify "$appdir"

    note "repack"
    cmd_repack "$appdir"

    note "smoke"
    cmd_smoke "$(find "$(dirname "$appdir")" -maxdepth 1 -name '*.AppImage' -print -quit)"
}

cmd="${1:-help}"
if [ "$#" -gt 0 ]; then shift; fi
case "$cmd" in
    package) cmd_package "${1:-}" ;;
    prune) cmd_prune "${1:-}" ;;
    verify) cmd_verify "${1:-}" ;;
    repack) cmd_repack "${1:-}" "${2:-}" ;;
    smoke) cmd_smoke "${1:-}" ;;
    help | -h | --help) usage ;;
    *)
        usage
        exit 2
        ;;
esac
