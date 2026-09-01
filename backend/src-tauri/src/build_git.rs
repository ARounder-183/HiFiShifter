//! git 构建信息的纯函数。由 build.rs（`#[path]` 引入）与单元测试共享，
//! 只依赖 std，保证在两个编译单元中行为一致。

/// 判定工作区是否脏（`git status --porcelain` 有任何输出即视为脏）。
pub fn is_dirty(status_porcelain: &str) -> bool {
    !status_porcelain.trim().is_empty()
}

/// 把 git 远程 URL 归一化为 GitHub 仓库主页链接；上游不是 GitHub 时返回 None。
///
/// 支持：
/// - `https://github.com/owner/repo`（含 `http://` 与结尾 `.git`）
/// - `git@github.com:owner/repo`（SCP 语法）
/// - `ssh://git@github.com/owner/repo`
pub fn normalize_github_remote_url(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let without_suffix = raw.strip_suffix(".git").unwrap_or(raw);
    let path = if let Some(rest) = without_suffix.strip_prefix("https://github.com/") {
        rest
    } else if let Some(rest) = without_suffix.strip_prefix("http://github.com/") {
        rest
    } else if let Some(rest) = without_suffix.strip_prefix("git@github.com:") {
        rest
    } else if let Some(rest) = without_suffix.strip_prefix("ssh://git@github.com/") {
        rest
    } else {
        return None;
    };
    let path = path.trim_end_matches('/');
    if path.is_empty() || !path.contains('/') {
        return None;
    }
    Some(format!("https://github.com/{path}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_https_remotes() {
        assert_eq!(
            normalize_github_remote_url("https://github.com/ARounder-183/HiFiShifter.git"),
            Some("https://github.com/ARounder-183/HiFiShifter".to_string())
        );
        assert_eq!(
            normalize_github_remote_url("https://github.com/ARounder-183/HiFiShifter"),
            Some("https://github.com/ARounder-183/HiFiShifter".to_string())
        );
    }

    #[test]
    fn normalizes_http_and_scp_remotes() {
        assert_eq!(
            normalize_github_remote_url("http://github.com/owner/repo.git"),
            Some("https://github.com/owner/repo".to_string())
        );
        assert_eq!(
            normalize_github_remote_url("git@github.com:owner/repo.git"),
            Some("https://github.com/owner/repo".to_string())
        );
        assert_eq!(
            normalize_github_remote_url("ssh://git@github.com/owner/repo"),
            Some("https://github.com/owner/repo".to_string())
        );
    }

    #[test]
    fn rejects_non_github_and_invalid_remotes() {
        assert_eq!(normalize_github_remote_url("git@gitlab.com:owner/repo.git"), None);
        assert_eq!(normalize_github_remote_url("https://gitlab.com/owner/repo.git"), None);
        assert_eq!(normalize_github_remote_url("https://github.com/only-owner"), None);
        assert_eq!(normalize_github_remote_url(""), None);
        assert_eq!(normalize_github_remote_url("   "), None);
    }

    #[test]
    fn dirty_flag_reflects_porcelain_output() {
        assert!(!is_dirty(""));
        assert!(!is_dirty("  \n "));
        assert!(is_dirty(" M src/main.rs"));
        assert!(is_dirty("?? notes.txt\n"));
    }
}
