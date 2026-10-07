//! 共享的 git 构建信息注入。
//!
//! 【为什么是源码级 include 而不是 `use`】build script 在它所属的 crate **之前**
//! 编译，因此不能依赖 workspace 里的任何 crate（那会要求内核已经构建完成，
//! 而内核的构建脚本又可能要等 app 的产物）。两个 build.rs 各自用
//! `#[path = "…"] mod git_info;` 引入本文件，共享的是源码而不是依赖。
//!
//! 使用方式：
//! ```ignore
//! #[path = "../build-support/git_info.rs"]
//! mod git_info;
//!
//! fn main() {
//!     git_info::emit(&["src", "resources"]);
//! }
//! ```
//!
//! `watched` 是**影响该产物内容**的包根相对路径：只有它们变化才应该翻转脏标志。
//! README / docs 之类不影响产物的改动不该标脏 —— 那会让「这份二进制对应哪次提交」
//! 的答案变成"总是脏的"，等于没有信号。

use std::path::Path;

/// git 构建信息的纯函数（判定脏标志、归一化远程 URL）。
///
/// 这份模块同时被 app 的单元测试引用，因此它的实现住在 `src-tauri/src/` 下，
/// 这里按相对路径引入 —— 两份实现分叉会让「构建脚本判定的脏」与「测试断言的脏」
/// 悄悄不一致。
#[path = "../src-tauri/src/build_git.rs"]
mod build_git;

/// 运行 git 命令并返回 trim 后的 stdout；git 不可用或命令失败时返回 None。
fn git_output(args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// 声明 rerun-if-changed（路径存在时才声明，避免不存在路径的指令干扰缓存指纹）。
fn declare_rerun_if_exists(path: &Path) {
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

/// 把当前 commit / 脏工作区标志 / GitHub 仓库链接烘进二进制，供关于对话框、
/// 日志会话头与诊断包展示，便于把用户日志精确追溯到某一份构建。
///
/// 缓存语义：build.rs 的工作目录是包根，而 `.git` 在仓库根目录 —— 因此
/// rerun-if-changed 必须使用 `git rev-parse --absolute-git-dir` 解析出的
/// **绝对路径**（相对路径 `.git/...` 永远不存在，指令形同虚设，commit 后不会
/// 重跑本脚本，哈希就冻结在旧值上）。声明的信号：
/// - `<gitdir>/logs/HEAD`（reflog：每次 commit/checkout 都会追加，最可靠的
///   "有新提交"信号）；
/// - `<gitdir>/HEAD`、当前分支 ref 文件与 packed-refs（ref 可能松散或打包存储；
///   worktree 下分支 ref 在 common dir，故两处都声明）；
/// - `watched` 给出的脏标志监视集。
///
/// 注意：一旦打印任何 rerun-if-changed，cargo 的"包内任意文件变化即重跑
/// build script"默认行为即被替换 —— 调用方的 build.rs 其余部分需各自显式
/// 声明其依赖路径。
pub fn emit(watched: &[&str]) {
    // 非 git 构建（如 GitHub 源码 zip）：注入空值，运行时回退为纯版本号。
    // 不打印任何 git 相关的 rerun 指令，保持既有指令集不变。
    let Some(full) = git_output(&["rev-parse", "HEAD"]) else {
        println!("cargo:rustc-env=HIFISHIFTER_GIT_COMMIT=");
        println!("cargo:rustc-env=HIFISHIFTER_GIT_COMMIT_SHORT=");
        println!("cargo:rustc-env=HIFISHIFTER_GIT_DIRTY=false");
        println!("cargo:rustc-env=HIFISHIFTER_GIT_REPO_URL=");
        for path in watched {
            declare_rerun_if_exists(Path::new(path));
        }
        return;
    };

    // 真实 git 目录（绝对路径；worktree 下为 .git/worktrees/<name>）。
    let git_dir = git_output(&["rev-parse", "--absolute-git-dir"]);
    // packed-refs 存放在 common dir（主仓库与 gitdir 相同；worktree 下为主 .git）。
    // `--path-format=absolute` 需要 git ≥ 2.31，失败则跳过该指令。
    let common_dir = git_output(&["rev-parse", "--path-format=absolute", "--git-common-dir"]);

    if let Some(dir) = &git_dir {
        let dir_path = Path::new(dir);
        for relative in ["HEAD", "logs/HEAD", "packed-refs"] {
            declare_rerun_if_exists(&dir_path.join(relative));
        }
    }
    if let Some(common) = &common_dir {
        declare_rerun_if_exists(&Path::new(common).join("packed-refs"));
    }
    // 当前分支的 ref 文件：提交时 mtime 变化（松散 ref 在 common dir，
    // per-worktree ref 在 gitdir，两处都声明）。
    if let Some(reference) = git_output(&["rev-parse", "--symbolic-full-name", "HEAD"]) {
        for dir in [&git_dir, &common_dir].into_iter().flatten() {
            declare_rerun_if_exists(&Path::new(dir).join(&reference));
        }
    }
    for path in watched {
        declare_rerun_if_exists(Path::new(path));
    }

    let short = git_output(&["rev-parse", "--short=9", "HEAD"])
        .unwrap_or_else(|| full.chars().take(9).collect());
    let dirty = git_output(&["status", "--porcelain"])
        .map(|status| build_git::is_dirty(&status))
        .unwrap_or(false);
    let repo_url = git_output(&["config", "--get", "remote.origin.url"])
        .and_then(|raw| build_git::normalize_github_remote_url(&raw))
        .unwrap_or_default();

    println!("cargo:rustc-env=HIFISHIFTER_GIT_COMMIT={full}");
    println!("cargo:rustc-env=HIFISHIFTER_GIT_COMMIT_SHORT={short}");
    println!("cargo:rustc-env=HIFISHIFTER_GIT_DIRTY={dirty}");
    println!("cargo:rustc-env=HIFISHIFTER_GIT_REPO_URL={repo_url}");
}
