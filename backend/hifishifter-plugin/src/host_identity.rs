//! 宿主识别：从「宿主自称什么」升级为「证据链」。
//!
//! # 它解决的是什么
//!
//! 此前只问 IHostApplication::getName 一次，然后直接拿它当宿主名。实测的四种形态：
//!
//! | 宿主 | getName 返回 | 结果 |
//! |---|---|---|
//! | REAPER | reaper | 正确 |
//! | VEGAS Pro | 扫描期 audio_plugin_scan_server_vst3_x64，运行期 sfvstwrap | 认不出是 VEGAS |
//! | FL Studio | fruity-wrapper | 同上 |
//! | Cubase | 加载期 unknown，使用期 cubase | 只读一次 ⇒ 永远停在 unknown |
//!
//! 包装器进程（sfvstwrap / fruity-wrapper）是宿主**故意**的隔离层，所以"问宿主"这条路
//! 天生问不出答案。要认出它，只能看**进程树**：包装器的父进程就是 DAW 本体。
//!
//! # 判据
//!
//! 收集全部证据（不短路），按可信度取第一个能识别出宿主的：
//!
//! 1. 环境变量覆盖 HIFISHIFTER_HOST_NAME（未收录宿主的逃生门）；
//! 2. ARA 宿主名（只有绑定 ARA 时才有）；
//! 3. 当前进程映像名（sfvstwrap.exe / fruity-wrapper.exe）；
//! 4. 祖先进程链（近 → 远，最多 4 层）；
//! 5. VST3 getName。
//!
//! 包装器命中且祖先也命中时取**祖先**那条，并在显示名里注明经由哪个包装器 ——
//! 那才是用户认得的名字。
//!
//! # 为什么证据要留着
//!
//! 没收录的宿主必须能靠证据被认出来。只报一个 unknown，用户报障时我们仍然只能猜；
//! 把 image 与祖先链写进日志，下一次就能直接加一行别名。

/// 一条证据来自哪里（写进日志，便于下次扩展别名表）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceSource {
    /// 环境变量覆盖。
    EnvOverride,
    /// ARA 宿主应用名。
    AraName,
    /// 当前进程映像名。
    ProcessImage,
    /// 祖先进程链里的某一层。
    Ancestor,
    /// VST3 IHostApplication::getName。
    Vst3Name,
}

impl EvidenceSource {
    /// 稳定的编码（只用于日志与诊断包，**只增不改**）。
    pub fn as_str(self) -> &'static str {
        match self {
            EvidenceSource::EnvOverride => "env-override",
            EvidenceSource::AraName => "ara-name",
            EvidenceSource::Vst3Name => "vst3-name",
            EvidenceSource::ProcessImage => "process-image",
            EvidenceSource::Ancestor => "process-ancestor",
        }
    }
}

/// 一条证据。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Evidence {
    pub source: EvidenceSource,
    pub name: String,
}

/// 识别结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostIdentity {
    /// 显示名（日志文件名与界面用它）。识别不出时为 "unknown"。
    pub display: String,
    /// 命中别名表的那条证据来源。
    pub source: Option<EvidenceSource>,
    /// 命中的原文（别名表里的键或宿主原话）。
    pub matched: Option<String>,
    /// 若经由包装器认出宿主，这里是包装器的名字。
    pub via: Option<String>,
    /// 全部证据（按收集顺序），供日志与诊断包。
    pub evidence: Vec<Evidence>,
}

impl HostIdentity {
    /// 未识别时的占位。
    pub fn unknown() -> Self {
        Self {
            display: "unknown".to_string(),
            source: None,
            matched: None,
            via: None,
            evidence: Vec::new(),
        }
    }

    /// 日志首行：一行说清"是谁、凭什么认出来的"。
    pub fn describe(&self) -> String {
        let mut parts = vec![format!("display=\"{}\"", self.display)];
        match self.source {
            Some(source) => parts.push(format!("source={}", source.as_str())),
            None => parts.push("source=none".to_string()),
        }
        if let Some(matched) = &self.matched {
            parts.push(format!("matched=\"{matched}\""));
        }
        if let Some(via) = &self.via {
            parts.push(format!("via=\"{via}\""));
        }
        let evidence: Vec<String> = self
            .evidence
            .iter()
            .map(|item| format!("{}:{}", item.source.as_str(), item.name))
            .collect();
        parts.push(format!("evidence=[{}]", evidence.join(", ")));
        format!("host identity: {}", parts.join(" "))
    }
}

/// 别名表：**子串**匹配（小写、去扩展名），新增宿主 = 加一行。
///
/// 用子串而不是全等：vegas180.exe / VEGAS Pro 21 / FL64.exe 这些变体必须命中同一条。
const HOST_ALIASES: &[(&str, &str)] = &[
    ("reaper", "REAPER"),
    ("vegas", "VEGAS Pro"),
    ("sfvstwrap", "VEGAS Pro"),
    ("audio_plugin_scan_server", "VEGAS Pro (VST3 scan server)"),
    ("fruity-wrapper", "FL Studio"),
    ("fl64", "FL Studio"),
    ("cubase", "Cubase"),
    ("nuendo", "Nuendo"),
    ("studio one", "Studio One"),
    ("studioone", "Studio One"),
    ("bitwig", "Bitwig Studio"),
    ("ableton", "Ableton Live"),
    ("live", "Ableton Live"),
    ("resolve", "DaVinci Resolve"),
    ("audition", "Adobe Audition"),
    ("wavelab", "WaveLab"),
    ("samplitude", "Samplitude"),
    ("sequoia", "Sequoia"),
    ("pyramix", "Pyramix"),
    ("renoise", "Renoise"),
    ("reason", "Reason"),
    ("maschine", "Maschine"),
    ("ardour", "Ardour"),
    ("mixbus", "Mixbus"),
    ("tracktion", "Tracktion"),
    ("cakewalk", "Cakewalk"),
    ("harrison", "Harrison Mixbus"),
    ("sadie", "SADiE"),
];

/// 包装器进程：它们自己就是别名表里的键（于是能认出来），但**显示名要取祖先**。
///
/// 只放"确实是隔离层"的：sfvstwrap 是 VEGAS 的 VST 宿主壳，fruity-wrapper 是 FL 的。
const WRAPPER_MARKERS: &[&str] = &["sfvstwrap", "fruity-wrapper"];

/// 规范化：小写、去路径、去 .exe。
fn normalize(name: &str) -> String {
    let base = name
        .rsplit(|c| c == '/' || c == '\\')
        .next()
        .unwrap_or(name);
    let lower = base.to_ascii_lowercase();
    lower
        .strip_suffix(".exe")
        .map(|stripped| stripped.to_string())
        .unwrap_or(lower)
}

/// 别名表命中：返回显示名。
fn lookup(name: &str) -> Option<&'static str> {
    let key = normalize(name);
    if key.is_empty() || key == "unknown" {
        return None;
    }
    HOST_ALIASES
        .iter()
        .find(|(needle, _)| key.contains(needle))
        .map(|(_, display)| *display)
}

/// 是否包装器进程。
fn is_wrapper(name: &str) -> bool {
    let key = normalize(name);
    WRAPPER_MARKERS.iter().any(|marker| key.contains(marker))
}

/// 从证据链里定出宿主。
///
/// 顺序见模块头。包装器命中时**继续往后找祖先**，因为包装器只说明"我在谁的壳里"，
/// 而用户要的是壳里那个 DAW 的名字。
pub fn resolve_identity(evidence: &[Evidence]) -> HostIdentity {
    let identity = HostIdentity {
        display: "unknown".to_string(),
        source: None,
        matched: None,
        via: None,
        evidence: evidence.to_vec(),
    };

    // 1. 环境变量覆盖优先，且不参与别名表（用户写的名字就是答案）。
    if let Some(item) = evidence
        .iter()
        .find(|item| item.source == EvidenceSource::EnvOverride)
    {
        let name = item.name.trim();
        if !name.is_empty() {
            return HostIdentity {
                display: name.to_string(),
                source: Some(EvidenceSource::EnvOverride),
                matched: Some(name.to_string()),
                via: None,
                ..identity
            };
        }
    }

    // 2. ARA 名最可信：它来自宿主自己的文档控制器，不经过包装器。
    if let Some(item) = evidence
        .iter()
        .find(|item| item.source == EvidenceSource::AraName)
    {
        if let Some(display) = lookup(&item.name) {
            return HostIdentity {
                display: display.to_string(),
                source: Some(EvidenceSource::AraName),
                matched: Some(item.name.clone()),
                via: None,
                ..identity
            };
        }
    }

    // 3/4. 进程映像与祖先链。先找第一个**非包装器**的命中。
    for item in evidence.iter().filter(|item| {
        item.source == EvidenceSource::ProcessImage || item.source == EvidenceSource::Ancestor
    }) {
        if is_wrapper(&item.name) {
            continue;
        }
        if let Some(display) = lookup(&item.name) {
            return HostIdentity {
                display: display.to_string(),
                source: Some(item.source),
                matched: Some(item.name.clone()),
                via: None,
                ..identity
            };
        }
    }

    // 4'. 只剩包装器命中：报包装器认出的宿主，并注明经由谁。
    if let Some(wrapper) = evidence
        .iter()
        .find(|item| {
            item.source == EvidenceSource::ProcessImage || item.source == EvidenceSource::Ancestor
        })
        .filter(|item| is_wrapper(&item.name))
    {
        if let Some(display) = lookup(&wrapper.name) {
            return HostIdentity {
                display: format!("{display} (via {})", normalize(&wrapper.name)),
                source: Some(wrapper.source),
                matched: Some(wrapper.name.clone()),
                via: Some(normalize(&wrapper.name)),
                ..identity
            };
        }
    }

    // 5. 最后才用 VST3 的自我声明 —— 它可能就是个包装器名。
    if let Some(item) = evidence
        .iter()
        .find(|item| item.source == EvidenceSource::Vst3Name)
    {
        if let Some(display) = lookup(&item.name) {
            return HostIdentity {
                display: display.to_string(),
                source: Some(EvidenceSource::Vst3Name),
                matched: Some(item.name.clone()),
                via: None,
                ..identity
            };
        }
    }

    identity
}

/// 收集本进程的证据（除 ARA / VST3 名以外的部分）。
///
/// 【为什么只做一次并缓存】进程树在会话期间不会变（DAW 不会换父进程），而枚举一次要
/// 走 Toolhelp 快照。缓存之后重解析（Cubase 那种"加载期 unknown"）只剩一次字符串比较。
pub fn process_evidence() -> Vec<Evidence> {
    let mut evidence = Vec::new();
    if let Ok(name) = std::env::var("HIFISHIFTER_HOST_NAME") {
        evidence.push(Evidence {
            source: EvidenceSource::EnvOverride,
            name,
        });
    }
    #[cfg(windows)]
    {
        if let Some(image) = win::current_image_name() {
            evidence.push(Evidence {
                source: EvidenceSource::ProcessImage,
                name: image,
            });
        }
        for ancestor in win::ancestor_names(4) {
            evidence.push(Evidence {
                source: EvidenceSource::Ancestor,
                name: ancestor,
            });
        }
    }
    evidence
}

#[cfg(windows)]
mod win {
    use windows::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W,
        TH32CS_SNAPPROCESS,
    };
    use windows::Win32::System::Threading::GetCurrentProcessId;

    /// pid → (父 pid, 映像名)。
    fn process_table() -> Vec<(u32, u32, String)> {
        let mut table = Vec::new();
        // SAFETY: 快照句柄在函数末尾关闭；PROCESSENTRY32W 先给出 dwSize 再传给 API。
        unsafe {
            let Ok(snapshot) = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) else {
                return table;
            };
            let mut entry = PROCESSENTRY32W::default();
            entry.dwSize = std::mem::size_of::<PROCESSENTRY32W>() as u32;
            if Process32FirstW(snapshot, &mut entry).is_ok() {
                loop {
                    let name = String::from_utf16_lossy(
                        &entry.szExeFile[..entry
                            .szExeFile
                            .iter()
                            .position(|unit| *unit == 0)
                            .unwrap_or(entry.szExeFile.len())],
                    );
                    table.push((entry.th32ProcessID, entry.th32ParentProcessID, name));
                    if Process32NextW(snapshot, &mut entry).is_err() {
                        break;
                    }
                }
            }
            let _ = windows::Win32::Foundation::CloseHandle(snapshot);
        }
        table
    }

    /// 本进程的映像名。
    pub(super) fn current_image_name() -> Option<String> {
        let table = process_table();
        let pid = unsafe { GetCurrentProcessId() };
        table
            .into_iter()
            .find(|(process, _, _)| *process == pid)
            .map(|(_, _, name)| name)
    }

    /// 祖先进程名（近 → 远，最多 max_depth 层，去掉空名与 System Idle）。
    pub(super) fn ancestor_names(max_depth: usize) -> Vec<String> {
        let table = process_table();
        let mut names = Vec::new();
        let mut pid = unsafe { GetCurrentProcessId() };
        for _ in 0..max_depth {
            let Some(parent) = table
                .iter()
                .find(|(process, _, _)| *process == pid)
                .map(|(_, parent, _)| *parent)
            else {
                break;
            };
            if parent == 0 || parent == pid {
                break;
            }
            match table.iter().find(|(process, _, _)| *process == parent) {
                Some((_, _, name)) if !name.is_empty() => names.push(name.clone()),
                _ => break,
            }
            pid = parent;
        }
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(source: EvidenceSource, name: &str) -> Evidence {
        Evidence {
            source,
            name: name.to_string(),
        }
    }

    /// REAPER 自报其名：直接命中。
    #[test]
    fn reaper_is_recognised_from_the_vst3_name() {
        let identity = resolve_identity(&[evidence(EvidenceSource::Vst3Name, "reaper")]);
        assert_eq!(identity.display, "REAPER");
        assert_eq!(identity.source, Some(EvidenceSource::Vst3Name));
    }

    /// VEGAS 的 VST 宿主壳：包装器名不该被当成宿主，祖先才是。
    #[test]
    fn vegas_is_recognised_from_the_wrapper_and_its_ancestor_chain() {
        let identity = resolve_identity(&[
            evidence(EvidenceSource::ProcessImage, "sfvstwrap.exe"),
            evidence(EvidenceSource::Vst3Name, "sfvstwrap"),
            evidence(EvidenceSource::Ancestor, "vegas180.exe"),
            evidence(EvidenceSource::Ancestor, "explorer.exe"),
        ]);
        assert_eq!(identity.display, "VEGAS Pro");
        assert_eq!(identity.source, Some(EvidenceSource::Ancestor));
        assert_eq!(identity.via, None);
    }

    /// 只有包装器、没有祖先时仍然报出宿主，并注明经由谁。
    #[test]
    fn a_wrapper_alone_still_names_the_daw_and_says_how() {
        let identity = resolve_identity(&[
            evidence(EvidenceSource::ProcessImage, "fruity-wrapper.exe"),
            evidence(EvidenceSource::Vst3Name, "fruity-wrapper"),
        ]);
        assert_eq!(identity.display, "FL Studio (via fruity-wrapper)");
        assert_eq!(identity.via.as_deref(), Some("fruity-wrapper"));
    }

    /// 扫描服务器不是 DAW 本体：能认出它是谁的扫描器，但必须标出来。
    #[test]
    fn vegas_scan_server_is_reported_as_a_scan_server() {
        let identity = resolve_identity(&[evidence(
            EvidenceSource::Vst3Name,
            "audio_plugin_scan_server_vst3_x64",
        )]);
        assert_eq!(identity.display, "VEGAS Pro (VST3 scan server)");
    }

    /// Cubase 加载期只给 unknown：此时**不能**编一个名字出来。
    #[test]
    fn an_unknown_vst3_name_is_not_evidence() {
        let identity = resolve_identity(&[evidence(EvidenceSource::Vst3Name, "unknown")]);
        assert_eq!(identity.display, "unknown");
        assert_eq!(identity.source, None);
    }

    /// 未收录的宿主保留全部证据 —— 下次加别名就靠它。
    #[test]
    fn an_unlisted_host_keeps_every_piece_of_evidence() {
        let identity = resolve_identity(&[
            evidence(EvidenceSource::ProcessImage, "sfvstwrap.exe"),
            evidence(EvidenceSource::Ancestor, "some-unknown-daw.exe"),
        ]);
        assert_eq!(identity.evidence.len(), 2);
        assert!(identity
            .describe()
            .contains("process-ancestor:some-unknown-daw.exe"));
    }

    /// 环境变量覆盖压过一切（未收录宿主的逃生门）。
    #[test]
    fn the_env_override_wins_over_everything() {
        let identity = resolve_identity(&[
            evidence(EvidenceSource::EnvOverride, "My DAW"),
            evidence(EvidenceSource::Vst3Name, "reaper"),
            evidence(EvidenceSource::Ancestor, "vegas180.exe"),
        ]);
        assert_eq!(identity.display, "My DAW");
        assert_eq!(identity.source, Some(EvidenceSource::EnvOverride));
    }

    /// ARA 名比进程证据更可信：它不经过包装器。
    #[test]
    fn the_ara_name_outranks_process_evidence() {
        let identity = resolve_identity(&[
            evidence(EvidenceSource::AraName, "Nuendo"),
            evidence(EvidenceSource::Ancestor, "vegas180.exe"),
        ]);
        assert_eq!(identity.display, "Nuendo");
        assert_eq!(identity.source, Some(EvidenceSource::AraName));
    }

    /// 大小写与扩展名不影响命中。
    #[test]
    fn names_are_normalised_before_matching() {
        assert_eq!(lookup("REAPER.exe"), Some("REAPER"));
        assert_eq!(lookup("C:/Program Files/REAPER/reaper.exe"), Some("REAPER"));
        assert_eq!(lookup("Cubase15.exe"), Some("Cubase"));
    }
}
