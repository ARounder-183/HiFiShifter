#!/usr/bin/env bash
# HiFiShifter 插件安装脚本（随 Linux 插件包分发）。
#
#   ./install.sh              装进当前用户：~/.vst3、~/.clap
#   sudo ./install.sh --system 装进系统目录：/usr/lib/vst3、/usr/lib/clap
#
# 只碰自己那两个名字：删掉 $vst3_dir/HiFiShifter.vst3、$clap_dir/HiFiShifter.clap
# 与 $clap_dir/HiFiShifter-resources 就是完整卸载，不会影响别的插件。
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
        printf '--system 需要 root：sudo ./install.sh --system\n' >&2
        exit 1
    fi
    vst3_dir="/usr/lib/vst3"
    clap_dir="/usr/lib/clap"
else
    vst3_dir="$HOME/.vst3"
    clap_dir="$HOME/.clap"
fi

if [ -d "$here/HiFiShifter.vst3" ]; then
    mkdir -p "$vst3_dir"
    rm -rf "$vst3_dir/HiFiShifter.vst3"
    cp -R "$here/HiFiShifter.vst3" "$vst3_dir/"
    printf 'VST3 → %s/HiFiShifter.vst3\n' "$vst3_dir"
fi

if [ -f "$here/HiFiShifter.clap" ]; then
    mkdir -p "$clap_dir"
    rm -f "$clap_dir/HiFiShifter.clap"
    rm -rf "$clap_dir/HiFiShifter-resources"
    cp "$here/HiFiShifter.clap" "$clap_dir/"
    cp -R "$here/HiFiShifter-resources" "$clap_dir/"
    printf 'CLAP → %s/HiFiShifter.clap\n' "$clap_dir"
fi

printf '\n完成。重启宿主（或在宿主里重新扫描插件）。\n'
