#!/usr/bin/env bash
# HiFiShifter Linux 安装脚本。
#
#   ./install.sh              当前用户：~/.local（AppImage + ~/.vst3 + ~/.clap + XDG 模型库）
#   sudo ./install.sh --system 系统级：/opt/hifishifter + /usr/lib/vst3 + /usr/lib/clap
#                              + /usr/local/share/HiFiShifter 模型库
#
# 三合一包里同时存在 HiFiShifter.AppImage、VST3、CLAP 与 models/；纯插件包
# 只有后两者，脚本会按实际存在的条目自动跳过。只碰自己的名字，不改别的插件。
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
system=0

for arg in "$@"; do
    case "$arg" in
        --system) system=1 ;;
        -h|--help) sed -n '2,9p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) printf '未知参数：%s\n' "$arg" >&2; exit 2 ;;
    esac
done

if [ "$system" = "1" ]; then
    if [ "$(id -u)" != "0" ]; then
        printf -- '--system 需要 root：sudo ./install.sh --system\n' >&2
        exit 1
    fi
    app_dir="/opt/hifishifter"
    bin_dir="/usr/local/bin"
    data_root="/usr/local/share/HiFiShifter"
    vst3_dir="/usr/lib/vst3"
    clap_dir="/usr/lib/clap"
else
    app_dir="$HOME/.local/lib/hifishifter"
    bin_dir="$HOME/.local/bin"
    data_root="${XDG_DATA_HOME:-$HOME/.local/share}/HiFiShifter"
    vst3_dir="$HOME/.vst3"
    clap_dir="$HOME/.clap"
fi

# ── 独立 App ────────────────────────────────────────────────────────────────
if [ -f "$here/HiFiShifter.AppImage" ]; then
    mkdir -p "$app_dir" "$bin_dir"
    cp -f "$here/HiFiShifter.AppImage" "$app_dir/HiFiShifter.AppImage"
    chmod 755 "$app_dir/HiFiShifter.AppImage"
    # 用一层薄 wrapper 而不是直接 symlink：AppImage 的运行器在部分版本里会
    # 把 argv[0] 当成镜像路径，经 symlink 启动时解析到错误位置。
    printf '#!/bin/sh\nexec "%s" "$@"\n' "$app_dir/HiFiShifter.AppImage" >"$bin_dir/hifishifter"
    chmod 755 "$bin_dir/hifishifter"
    printf 'AppImage → %s/HiFiShifter.AppImage（启动命令：%s/hifishifter）\n' "$app_dir" "$bin_dir"
fi

# ── 共享模型库 ──────────────────────────────────────────────────────────────
if [ -d "$here/models" ]; then
    # 这个目录由安装器独占；先整体替换，避免升级后旧版本权重永久占着磁盘。
    # 运行中的插件只读模型，升级前请先退出宿主。
    rm -rf "$data_root/models"
    mkdir -p "$data_root/models"
    for version_dir in "$here/models"/*; do
        [ -d "$version_dir" ] || continue
        version="$(basename "$version_dir")"
        cp -R "$version_dir" "$data_root/models/$version"
        printf '模型库 → %s/models/%s\n' "$data_root" "$version"
    done

    # 模型库的许可与根 LICENSE 一起放到共享目录，卸载插件时也能找到。
    [ -f "$here/LICENSE" ] && cp -f "$here/LICENSE" "$data_root/LICENSE"
    if [ -d "$here/licenses" ]; then
        rm -rf "$data_root/licenses"
        cp -R "$here/licenses" "$data_root/licenses"
    fi
fi

# ── VST3 ────────────────────────────────────────────────────────────────────
if [ -d "$here/HiFiShifter.vst3" ]; then
    mkdir -p "$vst3_dir"
    rm -rf "$vst3_dir/HiFiShifter.vst3"
    cp -R "$here/HiFiShifter.vst3" "$vst3_dir/"
    printf 'VST3 → %s/HiFiShifter.vst3\n' "$vst3_dir"
fi

# ── CLAP ────────────────────────────────────────────────────────────────────
if [ -f "$here/HiFiShifter.clap" ]; then
    mkdir -p "$clap_dir"
    rm -f "$clap_dir/HiFiShifter.clap"
    rm -rf "$clap_dir/HiFiShifter-resources"
    cp "$here/HiFiShifter.clap" "$clap_dir/"
    cp -R "$here/HiFiShifter-resources" "$clap_dir/"
    printf 'CLAP → %s/HiFiShifter.clap\n' "$clap_dir"
fi

printf '\n完成。重启宿主（或在宿主里重新扫描插件）。\n'
