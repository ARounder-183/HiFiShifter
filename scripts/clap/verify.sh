#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# HiFiShifter CLAP 产物自检（macOS / Linux）。
#
#     scripts/clap/verify.sh <path/to/HiFiShifter.clap>
#
# macOS 上这个路径是 bundle 目录，Linux 上是单文件模块。
#
# 判据与 VST3 那条流水线同源：**只信外部证据** —— 符号表、动态段、真的 dlopen 一次
# 并跑完 process 与 state 往返，而不是"构建脚本说它没问题"。
#
#   1 布局  2 导出符号  3 依赖自洽  4 黑盒加载  5 macOS bundle 元数据
#   6 打包卫生  7 前端产物  8 模型清单
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly TARGET="${1:?用法：verify.sh <path/to/HiFiShifter.clap>}"

FAILED=0
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
bad()  { printf "  \033[31m✗\033[0m %s\n" "$*"; FAILED=1; }
note() { printf "    %s\n" "$*"; }

[ -e "$TARGET" ] || { printf "产物不存在：%s\n" "$TARGET" >&2; exit 2; }

case "$(uname -s)" in
    Linux)  HOST_OS="linux" ;;
    Darwin) HOST_OS="macos" ;;
    *)      printf "不支持的平台\n" >&2; exit 2 ;;
esac

if [ "$HOST_OS" = "linux" ]; then
    # Linux 的 CLAP 是单文件：同级还有一个 <stem>-resources 目录。
    if [ -d "$TARGET" ]; then MODULE="$TARGET/HiFiShifter.clap"; else MODULE="$TARGET"; fi
    BUNDLE_DIR="$(cd -- "$(dirname -- "$MODULE")" && pwd)"
    RESOURCES="$BUNDLE_DIR/HiFiShifter-resources"
else
    # macOS 的 CLAP 是 bundle：Contents/MacOS + Contents/Frameworks + Contents/Resources。
    if [ -d "$TARGET" ]; then BUNDLE_DIR="$(cd -- "$TARGET" && pwd)"; else BUNDLE_DIR="$(cd -- "$(dirname -- "$TARGET")/../.." && pwd)"; fi
    MODULE="$BUNDLE_DIR/Contents/MacOS/HiFiShifter"
    RESOURCES="$BUNDLE_DIR/Contents/Resources"
fi

printf "\n[clap:verify] %s\n" "$MODULE"

# ── 1. 布局 ──────────────────────────────────────────────────────────────────
printf "\n1) 布局\n"
if [ -f "$MODULE" ]; then
    ok "模块存在：$MODULE"
else
    bad "找不到模块：$MODULE"
    printf "\n布局不对，后续检查没有意义。\n" >&2
    exit 1
fi
if [ "$HOST_OS" = "linux" ]; then
    # 单文件 CLAP 没有 Contents/，资源按 <stem>-resources 与模块同级（见 plugin-core 的
    # ResourceLayout::Sibling）。少了这个目录，插件的 frontend_dir() 会在第一次打开
    # 编辑器时返回错误 —— 宿主侧表现为"窗口空白"，日志里一行都没有。
    if [ -d "$RESOURCES" ]; then ok "资源目录与模块同级：$(basename "$RESOURCES")"
    else bad "缺少同级资源目录：$RESOURCES"; fi
else
    if [ -d "$RESOURCES" ]; then ok "bundle 内有 Contents/Resources"
    else bad "缺少 Contents/Resources"; fi
fi
[ -f "$RESOURCES/models/models.json" ] && ok "模型清单存在" || bad "缺少 models/models.json"

# ── 2. 导出符号 ──────────────────────────────────────────────────────────────
printf "\n2) 导出符号\n"
if [ "$HOST_OS" = "linux" ]; then
    SYMBOLS="$(nm -D --defined-only "$MODULE" 2>/dev/null || true)"
else
    SYMBOLS="$(nm -gU "$MODULE" 2>/dev/null || true)"
fi
if printf "%s" "$SYMBOLS" | grep -Eq "(^|[[:space:]])_?clap_entry$"; then
    ok "clap_entry"
else
    bad "缺少导出符号 clap_entry（宿主会认为这不是一个 CLAP 插件）"
fi

# ── 3. 依赖自洽 ──────────────────────────────────────────────────────────────
#
# 判据：每一条动态依赖要么在包内，要么是**系统库**（/lib、/usr/lib 前缀）。
# 其余一律算泄漏 —— 编译环境的绝对路径在目标系统上必然不存在。
printf "\n3) 动态依赖\n"
ESCAPED=()
if [ "$HOST_OS" = "linux" ]; then
    # 【为什么要清掉 LD_LIBRARY_PATH】本检查回答的是"用户机上能否解析"：用户没有
    # 构建树的搜索路径，依赖只能靠模块自带的 $ORIGIN rpath 与同目录的随包副本。
    # CI 任务级的 LD_LIBRARY_PATH（linuxdeploy / 打包步需要它，指向
    # backend/target/dist）优先级高于 DT_RUNPATH，会把 libwebgpu_dawn.so 劫持到
    # 构建树 —— 判据因此被编译环境污染，模块旁边明明有也会被误报成"包外部"。
    LINKAGE="$(env -u LD_LIBRARY_PATH ldd "$MODULE" 2>&1 || true)"
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
            "$BUNDLE_DIR"/*) continue ;;
            /lib/*|/lib64/*|/usr/lib/*|/usr/lib64/*) continue ;;
            *) ESCAPED+=("$dep") ;;
        esac
    done <<< "$LINKAGE"
    # 依赖躺在模块旁边，就必须有 \$ORIGIN 的搜索路径；否则用户机上只能靠
    # LD_LIBRARY_PATH —— 那是宿主进程的环境，我们无权要求。
    RUNPATH="$(readelf -d "$MODULE" 2>/dev/null | sed -n 's/.*(\(RPATH\|RUNPATH\)).*\[\(.*\)\]/\2/p' || true)"
    case ":$RUNPATH:" in
        *':$ORIGIN'*) ok "rpath 含 \$ORIGIN（${RUNPATH}）" ;;
        *) bad "模块没有 \$ORIGIN 的 rpath（当前：${RUNPATH:-无}）" ;;
    esac
else
    LINKAGE="$(otool -L "$MODULE" 2>&1 || true)"
    # 【为什么要跳过自己】otool -L 的第一条"依赖"其实是模块自己的 install name
    # （LC_ID_DYLIB），链接时写的是编译环境的绝对路径 —— 它不是依赖，却会在这条
    # "有没有指向包外部的东西"的判据里自己举报自己。用 otool -D 精确取出它。
    SELF_ID="$(otool -D "$MODULE" 2>/dev/null | tail -n +2 | head -1 || true)"
    while IFS= read -r linkage_line; do
        dep="$(printf "%s" "$linkage_line" | sed -E "s/^[[:space:]]*//; s/ \(compatibility.*$//")"
        [ -z "$dep" ] && continue
        case "$dep" in
            # 【为什么要跳过自己】otool -L 的第一条"依赖"其实是模块自己的 install name
            # （LC_ID_DYLIB），链接时写的是编译环境的绝对路径。它不是依赖，但在
            # "有没有指向包外部的东西"这条判据里会自己举报自己。
            "$MODULE") continue ;;
            "$SELF_ID") continue ;;
            /usr/lib/*|/System/*) continue ;;
            @rpath/*|@loader_path/*)
                base="$(basename "$dep")"
                [ -f "$BUNDLE_DIR/Contents/Frameworks/$base" ] \
                    || bad "依赖 $base 声明为 $dep 但 Contents/Frameworks 里没有它"
                ;;
            *) ESCAPED+=("$dep") ;;
        esac
    done <<< "$(printf "%s\n" "$LINKAGE" | tail -n +2)"
    if otool -l "$MODULE" 2>/dev/null | grep -A2 'LC_RPATH' | grep -q '@loader_path'; then
        ok "rpath 含 @loader_path"
    else
        bad "模块没有 @loader_path 的 rpath"
    fi
fi
if [ ${#ESCAPED[@]} -eq 0 ]; then
    ok "没有指向包外部的非系统依赖"
else
    bad "以下依赖指向包外部（用户机上大概率不存在）："
    printf "    %s\n" "${ESCAPED[@]}"
fi

# ── 4. 黑盒加载 ──────────────────────────────────────────────────────────────
#
# 探针真的走一遍 dlopen → init → 工厂 → 实例 → activate → process（逐样本比对直通）
# → state 存取往返，并核对 ARA 工厂表与 descriptor 的配对。任何一条不过都说明
# "能编译"与"能用"之间还有距离。
printf "\n4) 黑盒加载（dlopen → entry → 工厂 → 实例 → process → state）\n"
PROBE_DIR="$(mktemp -d)"
trap 'rm -rf "$PROBE_DIR"' EXIT
EXTRA_LIBS=""
[ "$HOST_OS" = "linux" ] && EXTRA_LIBS="-ldl"
if cc -O1 -Wall -o "$PROBE_DIR/probe" "$SCRIPT_DIR/probe.c" $EXTRA_LIBS 2>"$PROBE_DIR/cc.log"; then
    # 与上面同理：黑盒加载是打包形态的事实核查，不带编译树的搜索路径。
    if env -u LD_LIBRARY_PATH "$PROBE_DIR/probe" "$MODULE" >"$PROBE_DIR/out.log" 2>&1; then
        ok "模块可加载，工厂与实例可用"
        sed "s/^/    /" "$PROBE_DIR/out.log"
        grep -q "plugin_id=com.hifishifter.clap" "$PROBE_DIR/out.log" || bad "描述符 id 不是 com.hifishifter.clap"
        grep -q "process=ok" "$PROBE_DIR/out.log" || bad "process 没有跑通（直通比对失败）"
        # ── ARA（阶段 2）────────────────────────────────────────────────────
        # 判据从"必须缺席"反过来：宿主扫描时就要看到工厂表，并用 get_plugin_id
        # 把它与 descriptor 配对；两者差一个字符，插件永远不会被 ARA 认领，而
        # 宿主日志里一个字都不会说。
        grep -q "ara_factory=present plugin_id=com.hifishifter.clap" "$PROBE_DIR/out.log" \
            || bad "entry 没有发布与 descriptor 配对的 ARA 工厂表"
        grep -q "ara_plugin_extension=present" "$PROBE_DIR/out.log" \
            || bad "实例没有 ARA 插件扩展：宿主无法绑定文档控制器"
        # REAPER 的 2022 私有草案 ID 必须被**拒绝**：它的宿主实现在需要真正渲染时
        # 会崩在自身代码里（reaper.exe+0x5a677）。这条断言防止它被"顺手兼容"回来。
        grep -q "ara_reaper_legacy=absent" "$PROBE_DIR/out.log" \
            || bad "实例应答了 REAPER 的私有草案 ARA 扩展 ID（会让 REAPER 崩溃）"
        grep -q "feature=ara:supported" "$PROBE_DIR/out.log" \
            || bad "描述符没有声明 ara:supported（宿主不会把本插件当作 ARA 候选）"
        if grep -q "feature=ara:required" "$PROBE_DIR/out.log"; then
            bad "描述符声明了 ara:required（不支持 ARA 的宿主会直接拒绝建立实例）"
        fi
        # 【为什么逐项断言扩展】探针打印 "absent" 后返回 0 是老版判据的漏洞：
        # 少了 clap.state 的产物照样绿，宿主保存工程时编辑全丢。现在把"必须存在"
        # 与"必须缺席"都写成硬判据，避免能力声明与自检互相放水。
        grep -q "round-trip([0-9]* bytes)" "$PROBE_DIR/out.log" || bad "state 往返没有跑通"
        grep -q "render_extension=present" "$PROBE_DIR/out.log" || bad "clap.render 扩展缺失"
        grep -q "audio_ports_extension=present" "$PROBE_DIR/out.log" || bad "clap.audio-ports 扩展缺失"
        grep -q "timer_support_extension=present" "$PROBE_DIR/out.log" || bad "clap.timer-support 扩展缺失"
        grep -q "gui_extension=present" "$PROBE_DIR/out.log" || bad "clap.gui 扩展缺失：宿主无法创建编辑器窗口"
        grep -q "latency_extension=present latency=0" "$PROBE_DIR/out.log" || bad "clap.latency 必须如实返回 0"
        grep -q "tail_extension=present tail=0" "$PROBE_DIR/out.log" || bad "clap.tail 必须如实返回 0"
        grep -Eq "host_log_bytes=[1-9][0-9]*" "$PROBE_DIR/out.log" || bad "宿主没有收到任何 clap.log 日志"
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
    PLIST="$BUNDLE_DIR/Contents/Info.plist"
    if [ -f "$PLIST" ]; then
        ok "Info.plist 存在"
        EXEC_NAME="$(/usr/libexec/PlistBuddy -c "Print :CFBundleExecutable" "$PLIST" 2>/dev/null || true)"
        if [ -n "$EXEC_NAME" ] && [ -f "$BUNDLE_DIR/Contents/MacOS/$EXEC_NAME" ]; then
            ok "CFBundleExecutable=$EXEC_NAME 指向真实文件"
        else
            bad "CFBundleExecutable 缺失或指向不存在的文件（宿主不会加载这个 bundle）"
        fi
    else
        bad "缺少 Contents/Info.plist"
    fi
    if codesign --verify --strict "$BUNDLE_DIR" >/dev/null 2>&1; then
        ok "签名有效"
    else
        bad "签名校验失败（arm64 上无法加载）"
    fi
else
    printf "\n5) macOS bundle 元数据 —— 本平台不适用\n"
fi

# ── 6. 打包卫生 ──────────────────────────────────────────────────────────────
printf "\n6) 打包卫生\n"
DANGLING="$(find "$BUNDLE_DIR" -type l ! -exec test -e {} \; -print 2>/dev/null || true)"
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

# ── 7. 前端产物 ──────────────────────────────────────────────────────────────
#
# 缺一个 asset 的包是最常见的漏项，而它的表现是"插件窗口一片空白" —— 宿主不报错，
# 日志里也没有一行。三条判据：有没有 html、html 引用的 js/css 在不在、有没有远程引用。
printf "\n7) 前端产物\n"
FRONTEND="$RESOURCES/frontend"
if [ -s "$FRONTEND/plugin.html" ]; then
    ok "plugin.html 存在（$(du -h "$FRONTEND/plugin.html" | cut -f1)）"
else
    bad "缺少或为空的 frontend/plugin.html"
fi
MISSING_ASSETS=""
if [ -s "$FRONTEND/plugin.html" ]; then
    while IFS= read -r asset; do
        [ -z "$asset" ] && continue
        case "$asset" in
            /*) rel="${asset#/}" ;;
            *)  rel="$asset" ;;
        esac
        [ -f "$FRONTEND/$rel" ] || MISSING_ASSETS="$MISSING_ASSETS $rel"
    done <<< "$(grep -oE '(src|href)="[^"]+[.](js|css)"' "$FRONTEND/plugin.html" \
        | sed -E 's/^(src|href)="//; s/"$//' || true)"
fi
if [ -z "$MISSING_ASSETS" ]; then
    ok "html 引用的 js/css 都在包内"
else
    bad "html 引用了不存在的资源：$MISSING_ASSETS"
fi
REMOTE=""
if [ -d "$FRONTEND" ]; then
    HTML_REMOTE="$(grep -rInE "(src|href)[[:space:]]*=[[:space:]]*[\"']https?://" --include=*.html "$FRONTEND" 2>/dev/null || true)"
    CSS_REMOTE="$(grep -rInE "url\([[:space:]]*[\"']?https?://" --include=*.css "$FRONTEND" 2>/dev/null || true)"
    REMOTE="$(printf "%s\n%s\n" "$HTML_REMOTE" "$CSS_REMOTE" | sed "/^$/d")"
fi
if [ -z "$REMOTE" ]; then
    ok "html/css 里没有远程引用"
else
    note "下列位置引用了远程资源："
    printf "%s\n" "$REMOTE" | sed "s/^/    /"
    bad "前端产物里有远程引用（离线打开会失败）"
fi

# ── 8. 模型自包含 ────────────────────────────────────────────────────────────
#
# Unix 没有安装器拥有的共享模型库（见 model_store 的模块文档：非 Windows 上
# shared_store_dir 返回 None），插件直接用 bundle 里的副本。所以判据不是"没有权重"，
# 而是"清单里的每个文件都在" —— 少一个的表现是推理静默降级，宿主与日志都不会报错。
printf "\n8) 模型自包含\n"
if [ -f "$RESOURCES/models/models.json" ]; then
    ok "models.json 存在"
    if command -v python3 >/dev/null 2>&1; then
        if python3 - "$RESOURCES/models" <<'PY'
import json, os, sys
root = sys.argv[1]
with open(os.path.join(root, "models.json"), encoding="utf-8") as fh:
    manifest = json.load(fh)
missing = [f["path"] for f in manifest.get("files", []) if not os.path.isfile(os.path.join(root, f["path"]))]
if missing:
    print("missing: " + ", ".join(missing))
    sys.exit(1)
print("files=%d version=%s" % (len(manifest.get("files", [])), manifest.get("version")))
PY
        then
            ok "清单里的文件都在包里"
        else
            bad "清单引用的模型文件缺失（推理会静默降级）"
        fi
    else
        note "SKIP：没有 python3，无法核对清单"
    fi
else
    bad "缺少 models/models.json"
fi

if [ "$FAILED" = "0" ]; then
    printf "\n[clap:verify] 全部通过\n\n"
    exit 0
fi
printf "\n[clap:verify] 有检查未通过\n\n" >&2
exit 1
