#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# HiFiShifter VST3 bundle 自检（macOS / Linux）。
#
#     scripts/vst3/unix/verify.sh <path/to/HiFiShifter.vst3>
#
# 六条判据，任何一条不过就退出非零。设计原则：**只信外部证据** ——
# 符号表、动态段、以及真的 dlopen 一次，而不是"构建脚本说它没问题"。
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly BUNDLE="${1:?用法：verify.sh <path/to/HiFiShifter.vst3>}"

FAILED=0
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
bad()  { printf "  \033[31m✗\033[0m %s\n" "$*"; FAILED=1; }
note() { printf "    %s\n" "$*"; }

[ -d "$BUNDLE" ] || { printf "bundle 不存在：%s\n" "$BUNDLE" >&2; exit 2; }

case "$(uname -s)" in
    Linux)  HOST_OS="linux" ;;
    Darwin) HOST_OS="macos" ;;
    *)      printf "不支持的平台\n" >&2; exit 2 ;;
esac
case "$(uname -m)" in
    x86_64|amd64) HOST_ARCH="x86_64" ;;
    arm64|aarch64) HOST_ARCH="arm64" ;;
esac

if [ "$HOST_OS" = "linux" ]; then
    [ "$HOST_ARCH" = "arm64" ] && PLATFORM_ARCH="aarch64" || PLATFORM_ARCH="x86_64"
    MODULE="$BUNDLE/Contents/${PLATFORM_ARCH}-linux/HiFiShifter.so"
    LIB_DIR="$(dirname "$MODULE")"
    REQUIRED_SYMBOLS="GetPluginFactory ModuleEntry ModuleExit"
else
    MODULE="$BUNDLE/Contents/MacOS/HiFiShifter"
    LIB_DIR="$BUNDLE/Contents/Frameworks"
    REQUIRED_SYMBOLS="GetPluginFactory bundleEntry bundleExit"
fi

printf "\n[vst3:verify] %s\n" "$BUNDLE"

# ── 1. 布局 ──────────────────────────────────────────────────────────────────
printf "\n1) 布局\n"
if [ -f "$MODULE" ]; then
    ok "模块在规范路径：${MODULE#"$BUNDLE"/}"
else
    bad "找不到模块：$MODULE"
    printf "\n布局不对，后续检查没有意义。\n" >&2
    exit 1
fi
[ -f "$BUNDLE/Contents/Resources/models/models.json" ] \
    && ok "模型清单存在" || bad "缺少 Contents/Resources/models/models.json"

# ── 2. 导出符号 ──────────────────────────────────────────────────────────────
printf "\n2) 导出符号\n"
if [ "$HOST_OS" = "linux" ]; then
    SYMBOLS="$(nm -D --defined-only "$MODULE" 2>/dev/null || true)"
else
    SYMBOLS="$(nm -g "$MODULE" 2>/dev/null || true)"
fi
for symbol in $REQUIRED_SYMBOLS; do
    if printf "%s" "$SYMBOLS" | grep -Eq "(^|[[:space:]])_?${symbol}$"; then
        ok "$symbol"
    else
        bad "缺少导出符号 $symbol（宿主会显示 0 个类，且不报错）"
    fi
done

# ── 3. 依赖自洽 ──────────────────────────────────────────────────────────────
#
# 判据：每一条动态依赖要么在 bundle 内，要么是**系统库**（/lib、/usr/lib 前缀）。
# 其余一律算泄漏 —— 构建机上的绝对路径在用户机上必然不存在。
printf "\n3) 动态依赖\n"
ESCAPED=()
if [ "$HOST_OS" = "linux" ]; then
    LINKAGE="$(ldd "$MODULE" 2>&1 || true)"
    if printf "%s" "$LINKAGE" | grep -q "not found"; then
        bad "有依赖解析不到："
        printf "%s\n" "$LINKAGE" | grep "not found" | sed "s/^/    /"
    else
        ok "ldd 全部解析成功"
    fi
    while IFS= read -r linkage_line; do
        dep="$(printf "%s" "$linkage_line" | sed -n "s/.*=> \(\/[^ ]*\).*/\1/p")"
        [ -z "$dep" ] && continue
        case "$dep" in
            "$BUNDLE"/*) continue ;;
            /lib/*|/lib64/*|/usr/lib/*|/usr/lib64/*) continue ;;
            *) ESCAPED+=("$dep") ;;
        esac
    done <<< "$LINKAGE"
else
    LINKAGE="$(otool -L "$MODULE" 2>&1 || true)"
    while IFS= read -r linkage_line; do
        dep="$(printf "%s" "$linkage_line" | sed -E "s/^[[:space:]]*//; s/ \(compatibility.*$//")"
        [ -z "$dep" ] && continue
        case "$dep" in
            /usr/lib/*|/System/*) continue ;;
            @rpath/*|@loader_path/*)
                # @rpath 的条目必须能在 Frameworks 里找到实体，否则照样加载失败。
                base="$(basename "$dep")"
                [ -f "$LIB_DIR/$base" ] || bad "依赖 $base 声明为 $dep 但 $LIB_DIR 里没有它"
                ;;
            *) ESCAPED+=("$dep") ;;
        esac
    done <<< "$(printf "%s\n" "$LINKAGE" | tail -n +2)"
fi
if [ ${#ESCAPED[@]} -eq 0 ]; then
    ok "没有指向 bundle 外部的非系统依赖"
else
    bad "以下依赖指向 bundle 外部（用户机上大概率不存在）："
    printf "    %s\n" "${ESCAPED[@]}"
fi

# ── 4. 黑盒加载 ──────────────────────────────────────────────────────────────
printf "\n4) 黑盒加载（dlopen → 入口 → 工厂）\n"
PROBE_DIR="$(mktemp -d)"
trap "rm -rf \"$PROBE_DIR\"" EXIT
EXTRA_LIBS=""
[ "$HOST_OS" = "linux" ] && EXTRA_LIBS="-ldl"
if cc -O1 -Wall -o "$PROBE_DIR/probe" "$SCRIPT_DIR/probe.c" $EXTRA_LIBS 2>"$PROBE_DIR/cc.log"; then
    if "$PROBE_DIR/probe" "$MODULE" >"$PROBE_DIR/out.log" 2>&1; then
        ok "模块可加载，工厂可用"
        sed "s/^/    /" "$PROBE_DIR/out.log"
    else
        bad "黑盒加载失败："
        sed "s/^/    /" "$PROBE_DIR/out.log"
    fi
else
    bad "探针编译失败（需要 C 编译器）："
    sed "s/^/    /" "$PROBE_DIR/cc.log"
fi

# ── 5. macOS bundle 元数据 ───────────────────────────────────────────────────
if [ "$HOST_OS" = "macos" ]; then
    printf "\n5) macOS bundle 元数据\n"
    PLIST="$BUNDLE/Contents/Info.plist"
    if [ -f "$PLIST" ]; then
        ok "Info.plist 存在"
        EXEC_NAME="$(/usr/libexec/PlistBuddy -c "Print :CFBundleExecutable" "$PLIST" 2>/dev/null || true)"
        if [ -n "$EXEC_NAME" ] && [ -f "$BUNDLE/Contents/MacOS/$EXEC_NAME" ]; then
            ok "CFBundleExecutable=$EXEC_NAME 指向真实文件"
        else
            bad "CFBundleExecutable 缺失或指向不存在的文件（宿主不会加载这个 bundle）"
        fi
    else
        bad "缺少 Contents/Info.plist"
    fi
    if codesign --verify --strict "$BUNDLE" >/dev/null 2>&1; then
        ok "签名有效（ad-hoc）"
    else
        bad "签名校验失败（arm64 上无法加载）"
    fi
else
    printf "\n5) macOS bundle 元数据 —— 本平台不适用\n"
fi

# ── 6. 打包卫生 ──────────────────────────────────────────────────────────────
printf "\n6) 打包卫生\n"
# 【为什么查悬空链接】ORT 的 libwebgpu_dawn 在 target 下是**指向 ~/.cache 的符号链接**；
# cp 时不加 -L 就会把链接本身打进产物，在用户机上指向一个不存在的路径。
DANGLING="$(find "$BUNDLE" -type l ! -exec test -e {} \; -print 2>/dev/null || true)"
if [ -z "$DANGLING" ]; then
    ok "没有悬空符号链接"
else
    bad "存在悬空符号链接："
    printf "%s\n" "$DANGLING" | sed "s/^/    /"
fi
MODULE_KIND="$(file -b "$MODULE" 2>/dev/null || echo unknown)"
if [ "$HOST_OS" = "linux" ]; then
    case "$MODULE_KIND" in
        *ELF*shared\ object*) ok "模块是 ELF 共享库" ;;
        *) bad "模块不是 ELF 共享库：$MODULE_KIND" ;;
    esac
else
    case "$MODULE_KIND" in
        *Mach-O*) ok "模块是 Mach-O" ;;
        *) bad "模块不是 Mach-O：$MODULE_KIND" ;;
    esac
fi

if [ "$FAILED" = "0" ]; then
    printf "\n[vst3:verify] 全部通过\n\n"
    exit 0
fi
printf "\n[vst3:verify] 有检查未通过\n\n" >&2
exit 1