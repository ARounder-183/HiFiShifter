#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Unix 插件打包的共享规则（VST3 / CLAP 共用，macOS + Linux）。
#
# 【为什么要有这一层】两个格式的**布局**不同：VST3 在 Linux 上也是
# Contents/<arch>-linux/ 的 bundle，CLAP 在 Linux 上是单文件、在 macOS 上才是
# bundle。但**平台规则完全相同**：运行时依赖必须躺在模块旁边、rpath 必须是
# $ORIGIN / @loader_path、前端产物只有一份、许可证必须随包、产物里不许有悬空符号
# 链接。这些规则原先只写在 vst3/unix/build.sh 里，CLAP 那边一条都没写 —— 于是
# hifishifter-clap/build.rs 声明了 $ORIGIN，打包时却没人把库放进去，clap.yml 在
# ldd 与 dyld 上同时红掉。
#
# 这里只放"与格式无关"的实现。格式脚本负责自己的布局、入口符号与黑盒探针。
#
# 约定：本文件不设置 set -e，也不读写调用方的变量；所有状态放在 HFS_* 里。
# ─────────────────────────────────────────────────────────────────────────────

hfs_die() { printf '[plugin] ERROR: %s\n' "$*" >&2; exit 1; }
hfs_log() { printf '[plugin] %s\n' "$*"; }

# 平台三元组。macOS 的架构名沿用 uname（arm64），Linux 用 dpkg 的叫法（aarch64）——
# 这与既有产物命名 HiFiShifter_v<版本>_<os>-<arch>-* 一致，改名会让已发布的文件名断档。
hfs_detect_platform() {
    case "$(uname -s)" in
        Linux)  HFS_OS="linux" ;;
        Darwin) HFS_OS="macos" ;;
        *)      hfs_die "只支持 Linux 与 macOS，当前是 $(uname -s)" ;;
    esac
    case "$(uname -m)" in
        x86_64|amd64)  HFS_ARCH="x86_64" ;;
        arm64|aarch64) HFS_ARCH="arm64" ;;
        *)             hfs_die "不支持的 CPU 架构：$(uname -m)" ;;
    esac
    if [ "$HFS_OS" = "linux" ] && [ "$HFS_ARCH" = "arm64" ]; then
        HFS_PLATFORM_ARCH="aarch64"
    else
        HFS_PLATFORM_ARCH="$HFS_ARCH"
    fi
    HFS_TAG="$HFS_OS-$HFS_PLATFORM_ARCH"
    export HFS_OS HFS_ARCH HFS_PLATFORM_ARCH HFS_TAG
}

hfs_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

# CFBundleVersion 只接受点分的整数；完整版本号（含 -beta）另存自定义键。
hfs_numeric_version() {
    local version
    version="$(printf '%s' "${1#v}" | sed -E 's/-.*$//')"
    case "$version" in
        [0-9]*.[0-9]*.[0-9]*) printf '%s\n' "$version" ;;
        *) printf '0.0.0\n' ;;
    esac
}

# 产物里必须与模块同级的运行时库。ort 的 copy-dylibs 会把它们放在 target 目录下；
# Linux ARM64 的 ort 刻意不带 webgpu feature（没有预编译的 dawn），产物里既不会
# NEEDED 也不会存在 libwebgpu_dawn.so —— 列进来会让构建在该平台上永远失败。
hfs_runtime_deps() {
    if [ "$HFS_OS" = "linux" ]; then
        # 【为什么判 HFS_ARCH 而不是 HFS_PLATFORM_ARCH】后者在 Linux/arm64 上被改名成
        # "aarch64"（dpkg 的叫法，产物名沿用它），与 "arm64" 永远不相等 —— 这条分支
        # 因此一次都没命中过，linux-aarch64 会去要一个该平台根本没有的 dawn。
        if [ "$HFS_ARCH" = "arm64" ]; then
            printf '%s\n' "libSoundTouchDLL.so"
        else
            printf '%s\n' "libSoundTouchDLL.so" "libwebgpu_dawn.so"
        fi
    elif [ "$HFS_ARCH" = "x86_64" ]; then
        # macOS Intel 走 ort 的 alternative-backend（纯 Rust 的 tract，见 kernel 的
        # Cargo.toml）：既不链接 libonnxruntime，也不链接 dawn。把 dawn 列进来会让
        # 这条腿在"缺少运行时依赖"上失败，而产物其实没有任何问题。
        printf '%s\n' "libSoundTouchDLL.dylib"
    else
        printf '%s\n' "libSoundTouchDLL.dylib" "libwebgpu_dawn.dylib"
    fi
}

# <target-dir> <dest-dir>
hfs_stage_runtime_deps() {
    local target="$1" dest="$2" dep src
    mkdir -p "$dest"
    while IFS= read -r dep; do
        [ -n "$dep" ] || continue
        src="$target/$dep"
        [ -e "$src" ] || hfs_die "缺少运行时依赖 ${src}（先编译：ort 的 copy-dylibs 会把它放在 target 目录下）"
        # -L：ORT 的 dawn 在 target 下是指向 ~/.cache 的符号链接，直接 cp 会在压缩包里
        # 留下一个悬空链接（用户机上指向不存在的路径）。
        cp -Lf "$src" "$dest/$dep"
        hfs_log "依赖 ${dep}（$(du -h "$dest/$dep" | cut -f1)）"
    done <<< "$(hfs_runtime_deps)"
}

# <resources-dir> [<frontend-dist>]
hfs_stage_frontend() {
    local res="$1" src="${2:-${HFS_FRONTEND_DIST:-}}"
    [ -n "$src" ] || hfs_die "没有指定前端产物目录"
    [ -s "$src/plugin.html" ] || hfs_die "缺少前端产物：$src/plugin.html"
    [ -d "$src/assets" ] || hfs_die "缺少 $src/assets：插件入口没有脚本可加载"
    mkdir -p "$res/frontend"
    cp -f "$src/plugin.html" "$res/frontend/"
    rm -rf "$res/frontend/assets"
    cp -R "$src/assets" "$res/frontend/"
    hfs_log "前端资源（$(du -sh "$res/frontend" | cut -f1)）"
}

# 模型必须进包。Windows 有安装器拥有的共享模型库，Unix 没有 —— 见
# hifishifter-kernel/src/model_store.rs 的模块文档：非 Windows 上 shared_store_dir 返回
# None，插件直接用 bundle 里的副本。所以 Unix 的包是自包含的（约 150 MB），
# 代价是两个格式各带一份；换来的是"解压就能用、不依赖任何外部目录"。
# <resources-dir> <models-source>
hfs_stage_models() {
    local res="$1" src="$2"
    [ -d "$src" ] || hfs_die "缺少模型资源目录：$src"
    mkdir -p "$res/models"
    cp -R "$src/." "$res/models/"
    [ -f "$res/models/models.json" ] \
        || hfs_die "模型清单缺失：先跑 python3 scripts/setup/fetch-models.py --write-manifest"
}

# <dest-dir> <repo-root>
hfs_stage_licenses() {
    local dest="$1" root="$2" extra
    for extra in LICENSE licenses; do
        [ -e "$root/$extra" ] || hfs_die "缺少许可证：$extra"
        cp -R "$root/$extra" "$dest/"
    done
}

# <plist-path> <executable> <bundle-id> <version> <signature>
hfs_write_info_plist() {
    local path="$1" exec_name="$2" bundle_id="$3" version="$4" signature="$5"
    local numeric
    numeric="$(hfs_numeric_version "$version")"
    {
        printf '<?xml version="1.0" encoding="UTF-8"?>\n'
        printf '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        printf '<plist version="1.0">\n<dict>\n'
        printf '  <key>CFBundleDevelopmentRegion</key><string>en</string>\n'
        printf '  <key>CFBundleExecutable</key><string>%s</string>\n' "$exec_name"
        printf '  <key>CFBundleIdentifier</key><string>%s</string>\n' "$bundle_id"
        printf '  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>\n'
        printf '  <key>CFBundleName</key><string>HiFiShifter</string>\n'
        printf '  <key>CFBundleDisplayName</key><string>HiFiShifter</string>\n'
        printf '  <key>CFBundlePackageType</key><string>BNDL</string>\n'
        printf '  <key>CFBundleSignature</key><string>%s</string>\n' "$signature"
        printf '  <key>CFBundleShortVersionString</key><string>%s</string>\n' "$numeric"
        printf '  <key>CFBundleVersion</key><string>%s</string>\n' "$numeric"
        printf '  <key>NSHighResolutionCapable</key><true/>\n'
        printf '  <key>HiFiShifterVersion</key><string>%s</string>\n' "$version"
        printf '</dict>\n</plist>\n'
    } >"$path"
    hfs_log "Info.plist 已生成（CFBundleVersion=${numeric}）"
}

# 签名身份：默认 ad-hoc（arm64 上未签名的 Mach-O 根本加载不了）。发布签名与公证由
# HIFISHIFTER_CODESIGN_IDENTITY 打开，构建脚本不假装拥有证书。
# <bundle-dir> <module-file>
hfs_sign_bundle() {
    local bundle="$1" module="$2" identity file
    identity="${HIFISHIFTER_CODESIGN_IDENTITY:--}"
    command -v codesign >/dev/null 2>&1 || hfs_die "缺少 codesign"
    local -a args=(--force --sign "$identity")
    if [ "$identity" != "-" ]; then
        # 公证要求 hardened runtime + 安全时间戳；ad-hoc 签名带这两个参数没有意义。
        args+=(--options runtime --timestamp)
    fi
    # 顺序是硬约束：先依赖、再模块、最后 bundle —— bundle 的签名覆盖整棵目录树，
    # 反过来会让前两步的签名立刻失效。
    while IFS= read -r file; do
        [ -n "$file" ] || continue
        codesign "${args[@]}" "$file" >/dev/null 2>&1 || hfs_die "依赖签名失败：$file"
    done < <(find "$bundle" -type f \( -name '*.dylib' -o -name '*.so' \) -print)
    codesign "${args[@]}" "$module" >/dev/null 2>&1 || hfs_die "模块签名失败：$module"
    codesign "${args[@]}" "$bundle" >/dev/null 2>&1 || hfs_die "bundle 签名失败：$bundle"
    if [ "$identity" = "-" ]; then
        hfs_log "已完成 ad-hoc 签名"
    else
        hfs_log "已用 $identity 签名"
    fi
}

# 依赖放在模块旁边就必须有 $ORIGIN / @loader_path 的搜索路径：用户机上没有
# LD_LIBRARY_PATH，那是宿主进程的环境，我们无权要求。
# <module-file>
hfs_assert_rpath() {
    local module="$1" paths
    if [ "$HFS_OS" = "linux" ]; then
        command -v readelf >/dev/null 2>&1 || { hfs_log "SKIP rpath：没有 readelf"; return 0; }
        paths="$(readelf -d "$module" 2>/dev/null | sed -n 's/.*(\(RPATH\|RUNPATH\)).*\[\(.*\)\]/\2/p' | tr '\n' ':')"
        case ":$paths" in
            *':$ORIGIN'*) hfs_log "rpath 含 \$ORIGIN（${paths}）"; return 0 ;;
        esac
        hfs_die "模块没有 \$ORIGIN 的 rpath（当前：${paths:-无}）：用户机上依赖会解析失败"
    else
        command -v otool >/dev/null 2>&1 || { hfs_log "SKIP rpath：没有 otool"; return 0; }
        otool -l "$module" 2>/dev/null | grep -A2 'LC_RPATH' | grep -q '@loader_path' \
            || hfs_die "模块没有 @loader_path 的 rpath：bundle 内的依赖会解析失败"
        hfs_log "rpath 含 @loader_path"
    fi
}

# <dir>
hfs_check_no_dangling() {
    local dir="$1" dangling
    dangling="$(find "$dir" -type l ! -exec test -e {} \; -print 2>/dev/null || true)"
    [ -z "$dangling" ] || hfs_die "产物里有悬空符号链接：$dangling"
}

# cargo 构建失败时的现场回放。**不退出**：调用方保留自己的 [vst3] / [clap] 前缀。
# <log-file> <exit-status>
#
# 【为什么要先找 error 行】build script 的 panic 写在日志**中段**：cargo 会让并行的作业
# 跑完才退出，所以它后面还跟着几百行原生依赖（World 的 C++）的编译告警。只回放末尾，
# 得到的现场就是"一堆 warning + 退出码 101"，看上去像没有任何原因 —— vst3-unix.yml
# 为此被误判成 OOM 过一轮，直到把完整日志翻出来才看见真正的 panic。
hfs_report_build_failure() {
    local log="$1" status="$2" first_line window
    if [ ! -f "$log" ]; then
        printf '[plugin] 没有构建日志：%s\n' "$log" >&2
        return 0
    fi
    if [ "$status" -eq 137 ] || [ "$status" -eq 143 ]; then
        printf '[plugin] 进程被信号杀死（exit %s）：runner 内存或磁盘不足。\n' "$status" >&2
    fi
    # CARGO_TERM_COLOR=never 是这条判据成立的前提（否则行首是转义序列）。
    first_line="$(grep -n -m1 -E '^error(\[|:)' "$log" 2>/dev/null | cut -d: -f1 || true)"
    if [ -n "$first_line" ]; then
        window=$((first_line + 40))
        printf '[plugin] 日志里第一处 error（第 %s 行起）：\n' "$first_line" >&2
        sed -n "${first_line},${window}p" "$log" >&2
        printf '[plugin] 日志末尾 20 行：\n' >&2
        tail -20 "$log" >&2
    else
        printf '[plugin] 日志里没有 error 行，末尾 120 行：\n' >&2
        tail -120 "$log" >&2
    fi
}

# <output.tar.gz> <workdir> <entry>...
hfs_tarball() {
    local out="$1" work="$2"
    shift 2
    [ $# -gt 0 ] || hfs_die "打包没有指定条目"
    rm -f "$out"
    ( cd "$work" && tar -czf "$out" "$@" ) || hfs_die "打包失败：$out"
    local digest
    digest="$(hfs_sha256 "$out")"
    printf '%s  %s\n' "$digest" "$(basename "$out")" >"$out.sha256"
    hfs_log "产物 $(du -h "$out" | cut -f1)：$out"
    hfs_log "sha256 $digest"
}
