//! 宿主识别：从「宿主自称什么」升级为「证据链」。
//!
//! # 它解决的是什么
//!
//! 此前只问 IHostApplication::getName 一次，然后直接拿它当宿主名。实测的四种形态：
//!
//! | 宿主 | getName 返回 | 结果 |
//! |---|---|---|
//! | REAPER | reaper | 正确 |
//! | VEGAS Creative Software 家族 | 扫描期 audio_plugin_scan_server_vst3_x64，运行期 sfvstwrap | 认不出是哪个 DAW |
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
//! 包装器命中且祖先也命中时取**祖先**那条 —— 那才是用户认得的名字；包装器本身只留在
//! `evidence` 里（`via` 只在**没有**产品级证据时才有值，见下）。
//!
//! # 包装器只证明"哪个家族"，不证明"哪个产品"
//!
//! `sfvstwrap` 是 VEGAS Creative Software **全家**（VEGAS Pro、ACID Pro、Sound Forge、
//! Movie Studio …）共用的 VST 宿主壳，`audio_plugin_scan_server_vst3_x64` 同样是家族
//! 共用的组件名 —— 两个名字里都不含产品名。因此它们能证明的只有"属于这个家族"。
//! 产品名只能由**祖先链里那个真正的 DAW 进程**给出（`vegas180.exe` / `acidpro.exe` /
//! `soundforge.exe` …）；祖先认不出来时如实报家族，绝不猜成 VEGAS Pro。
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

/// 进程级“宿主属于 VEGAS Creative Software 家族”的读数。
///
/// 【为什么是全局原子】音频线程要无锁判断是否允许冻结预览；宿主身份是进程级事实，
/// 与具体实例无关。只在识别成功时写一次。
static HOST_IS_VEGAS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 是否已经成功解析过一次宿主身份（诊断；`false` = 只有 unknown）。
static HOST_IDENTITY_RESOLVED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
/// “首次识别结果”是否已经打过日志（避免每次 setupProcessing 都刷一行）。
static HOST_IDENTITY_REPORTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// 记录一次识别结果（只由 `resolve_and_record_host_identity` 调用）。
pub fn note_identity(identity: &HostIdentity) {
    let display = identity.display.to_ascii_lowercase();
    let is_vegas = display.contains("vegas")
        || display.contains("acid")
        || display.contains("sound forge")
        || identity.evidence.iter().any(|item| {
            let name = item.name.to_ascii_lowercase();
            name.contains("sfvstwrap")
                || name.contains("audio_plugin_scan_server")
                || name.contains("vegas")
        });
    if identity.display != "unknown" {
        HOST_IDENTITY_RESOLVED.store(true, std::sync::atomic::Ordering::Release);
    }
    if is_vegas {
        HOST_IS_VEGAS.store(true, std::sync::atomic::Ordering::Release);
    }
    // 【为什么用 INFO】宿主身份是会话边界，按语义归「中」档；而“宿主到底被认成谁”正是预览
    // 判定唯一的先验。现场日志文件名是 plugin-<format>-unknown.log，这一行能直接回答它。
    if !HOST_IDENTITY_REPORTED.swap(true, std::sync::atomic::Ordering::AcqRel) {
        log::warn!(
            "[host] identity resolved at create/setup: display=\"{}\" vegas_family={} evidence={}",
            identity.display,
            is_vegas,
            identity.evidence.len()
        );
    }
}

/// 是否已经解析出非 unknown 的宿主身份。
#[allow(dead_code)]
pub fn identity_resolved() -> bool {
    HOST_IDENTITY_RESOLVED.load(std::sync::atomic::Ordering::Acquire)
}

/// 当前进程的宿主是否属于 VEGAS Creative Software 家族。
pub fn is_vegas_family() -> bool {
    HOST_IS_VEGAS.load(std::sync::atomic::Ordering::Acquire)
}

/// VST3 组件是否对本宿主**隐藏** ARA 入口点（`IPlugInEntryPoint` / `IPlugInEntryPoint2`）。
///
/// 这是本模块第一个按宿主**分支**的决定，与模块头的「绝不按宿主名分支」并不矛盾：
/// 那条纪律针对的是**编辑器能力**（没见过的宿主必须完整可用）；而这里 gate 的是
/// **加载兼容性** —— 一旦被扫描器标成 ARA，这个家族的桥会拒绝加载模块本身，
/// 插件在宿主里从「ARA 缺失」劣化成「完全不可用」。与 CLAP 侧拒绝 REAPER 的
/// 私有 ARA 草案 ID（`docs/host-extension-boundary.md` §3.1）同类：宿主显然
/// 处理不了我们按规范给出的东西，规范让位于「能加载」。
///
/// # 证据链（ACID Pro 11.0.2.21 x64，2026-11 现场排查，错误 0x8004E204）
///
/// 1. 扫描器按 ARA SDK 规范探测组件上的 `IPlugInEntryPoint`（12814E54-…）与
///    `IPlugInEntryPoint2`（CD9A5913-…），命中后把插件标记为 ARA（家族插件缓存
///    `acidpro_bridgeaware_plugincache_x64.xml` 记 `isAra="1"`）；
/// 2. 把插件添加到轨道时，家族共用的 VST3 桥（`sfvstwrap.dll`，宿主进程内；错误码
///    `0x8004E204` 的立即数只出现在 sfvstwrap/sfvstserver 一对二进制里）走 **ARA
///    专用路径**：先打开名为 `plugin_finder_shared_memory` 的 boost::interprocess
///    共享内存做校验 —— 该映射在 ACID Pro 11 里**不存在**（添加尝试前后与尝试期间
///    共 1.8 秒 ~4878 万次 `OpenFileMappingW` 竞速捕获 0 次命中）；
/// 3. 打开失败 ⇒ 立即返回 `0x8004E204`（弹窗文案「消息缺失」= 该错误码在消息表里
///    没有条目的兜底格式），全程**不加载插件模块**：引擎日志零会话、shim 的
///    OutputDebugString 零输出、也没有 sfvstserver 进程被拉起；
/// 4. 与插件内容无关：把缓存记录的版本字段 0→1、二进制不动，错误原样复现；
///    把缓存记录改成 `isAra="0"` 后，同一插件正常加载，捕获工作流完整可用；
/// 5. ACID Pro 11 自带的 ARA 插件（Melodyne、acidFX MorphPad）由主程序里的**定制
///    集成类**承载（RTTI 可见 `MelodynePluginView` / `FXPadPluginView`），不经这条
///    通用路径；官方口径里 ACID 的 ARA2 即 Melodyne 集成。
///
/// 结论：宿主侧基础设施缺失，插件侧无可补救。入口点只由**组件**的 queryInterface
/// 暴露，在这里对家族隐藏即可让扫描器记 `isAra="0"`，插件以普通插入形态加载。
/// 只圈定 VEGAS 家族（`is_vegas_family`）：REAPER / Cubase / Studio One 等真 ARA2
/// 宿主不受影响；Samplitude / Sequoia 不落进这个判定（它们不用 sfvstwrap 壳）。
/// Sound Forge 同壳同桥，按家族一并隐藏 —— 文档表里它的 ARA2「是」从未验收过。
pub fn hide_ara_entry_points() -> bool {
    let no_ara = std::env::var("HIFISHIFTER_NO_ARA").ok();
    let force_ara = std::env::var("HIFISHIFTER_FORCE_ARA").ok();
    ara_entry_points_hidden(is_vegas_family(), no_ara.as_deref(), force_ara.as_deref())
}

/// [`hide_ara_entry_points`] 的纯判定（单元测试用）。
///
/// `no_ara` / `force_ara` 是 `HIFISHIFTER_NO_ARA` / `HIFISHIFTER_FORCE_ARA` 的读数
/// （`None` = 未设置）。优先级：`NO_ARA=1` 是全局杀开关，压过一切；`FORCE_ARA=1`
/// 只覆盖家族闸门（给未来修好了桥的 MAGIX 版本与现场排查留门）；两者都为 1 时
/// **隐藏**获胜 —— 显式关闭压过显式开启。
fn ara_entry_points_hidden(
    vegas_family: bool,
    no_ara: Option<&str>,
    force_ara: Option<&str>,
) -> bool {
    if no_ara.map(str::trim) == Some("1") {
        return true;
    }
    if force_ara.map(str::trim) == Some("1") {
        return false;
    }
    vegas_family
}

/// 宿主应用名的**宿主自报读数**（VST3：`IHostApplication::getName` 的最近一次结果）。
///
/// 【为什么单独存一份】报障时最需要的一条信息是"你在哪个宿主里"——而这份信息此前只有
/// 一个**声明过却从未调用**的 vtable 槽位。存下来供证据链与诊断包读取。
///
/// 【为什么在共享层而不是 VST3 外壳】ARA 绑定路径（`render/extension/binding.rs`）在
/// 绑定窗口里也读它；而"谁写入它"只有 VST3 外壳（`IHostApplication` 握手）—— CLAP
/// 没有对应接口、不写。读数是进程级事实，写入方是格式细节。
///
/// 【为什么绝不用于分支】按宿主名分支会让"没见过的宿主"退化成不可用；一切差异都应由
/// 能力探测（`BootstrapFlags` / `NativeHost::from_context` 的 Option bundle）决定。
static HOST_APPLICATION_NAME: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// 登记宿主自报的应用名（VST3 外壳在 `IHostApplication` 握手成功后调用）。
pub fn note_host_application_name(name: &str) {
    *HOST_APPLICATION_NAME
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = Some(name.to_owned());
}

/// 读回宿主自报的应用名（未识别为 `None`）。只读，供证据链与诊断输出。
pub fn host_application_name() -> Option<String> {
    HOST_APPLICATION_NAME
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// 收集全部证据、定出宿主、把日志切到 `plugin-<format>-<host>.log`。
///
/// 【为什么住在这里而不是某个外壳】进程证据（映像名、祖先链、环境变量）与格式无关，
/// 而证据的**追加**是格式的事实：VST3 外壳能在 `IHostApplication::getName` 返回后追加
/// [`EvidenceSource::Vst3Name`]，CLAP 没有对应接口就只给进程证据。两个外壳调用这**同一个**
/// 解析与登记流程，`plugin-<format>-<host>.log` 的宿主段才有同一个判据。
///
/// 【为什么可以重复调用】Cubase 在加载期只给 unknown，真正使用时的名字更晚才到。
/// 第一次拿不到就留 unknown，后续调用（首次 process）再解析一次；一旦得到非 unknown
/// 的结果就不再重算 —— 否则日志文件名会来回跳（`note_identity` 只在首次报告，
/// `note_host_application` 只在拿到名字时切换）。
pub fn resolve_and_record(evidence: &[Evidence]) -> HostIdentity {
    let identity = resolve_identity(evidence);
    note_identity(&identity);
    crate::log_line(&identity.describe());
    if identity.display != "unknown" {
        // 日志按宿主分文件：拿到名字的这一刻把日志切过去
        //（此前攒下的行会一起补写进去，见 `diagnostics::note_host_application`）。
        crate::diagnostics::note_host_application(&identity.display);
    }
    identity
}

/// 只用进程证据解析宿主身份（CLAP 与未来 AU 的入口：没有格式专属证据可追加）。
///
/// 【为什么公开】外壳的 `lib.rs::resolve_host_identity` 退化为对本函数的转发；编排
/// （收集 → 解析 → 登记 → 切日志）只有这一份。
pub fn resolve_and_record_host_identity() -> HostIdentity {
    resolve_and_record(&process_evidence())
}

/// 别名表：**子串**匹配（小写、去扩展名），新增宿主 = 加一行。
///
/// 用子串而不是全等：vegas180.exe / VEGAS Pro 21 / FL64.exe 这些变体必须命中同一条。
const HOST_ALIASES: &[(&str, &str)] = &[
    ("reaper", "REAPER"),
    // VEGAS Creative Software 家族：产品名只能由各自的可执行文件名给出；家族共用的
    // 壳与扫描器**不在**这张表里，见 `FAMILY_MARKERS`。
    ("vegas", "VEGAS Pro"),
    ("acidpro", "ACID Pro"),
    ("acid", "ACID Pro"),
    ("sound forge", "Sound Forge"),
    ("sound_forge", "Sound Forge"),
    ("soundforge", "Sound Forge"),
    ("movie studio", "Movie Studio"),
    ("movie_studio", "Movie Studio"),
    ("moviestudio", "Movie Studio"),
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

/// 包装器进程：宿主**故意**的隔离层，显示名要取祖先。
///
/// 只放"确实是隔离层"的：`sfvstwrap` 是 VEGAS Creative Software 家族的 VST 宿主壳，
/// `fruity-wrapper` 是 FL Studio 的。
const WRAPPER_MARKERS: &[&str] = &["sfvstwrap", "fruity-wrapper"];

/// 家族级标记：**同一个名字被家族里多个产品共用**，因此只能证明"属于哪个家族"。
///
/// 【为什么必须单列，而不是放进别名表】别名表的值是**产品名**。此前 `sfvstwrap` 直接
/// 映射成 "VEGAS Pro"，于是 ACID Pro / Sound Forge / Movie Studio 的用户在日志里看到的
/// 是 VEGAS Pro —— 一个凭证据得不出的结论。这些名字里没有产品信息，产品只能由祖先链
/// 里那个真正的 DAW 进程给出；给不出来就如实报家族。
///
/// 【为什么 `fruity-wrapper` 不在这里】那个壳就是 FL Studio 自己的，没有兄弟产品与它
/// 共用，所以它留在别名表里按产品报。
const FAMILY_MARKERS: &[(&str, &str)] = &[
    ("sfvstwrap", "VEGAS Creative Software"),
    ("audio_plugin_scan_server", "VEGAS Creative Software"),
];

/// 规范化：小写、去路径、去 .exe。
fn normalize(name: &str) -> String {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
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

/// 家族级命中：返回家族显示名（**不含产品**，见 [`FAMILY_MARKERS`]）。
fn family_of(name: &str) -> Option<&'static str> {
    let key = normalize(name);
    if key.is_empty() || key == "unknown" {
        return None;
    }
    FAMILY_MARKERS
        .iter()
        .find(|(needle, _)| key.contains(needle))
        .map(|(_, family)| *family)
}

/// 从证据链里定出宿主。
///
/// 顺序见模块头。包装器命中时**继续往后找祖先**，因为包装器只说明"我在谁的壳里"，
/// 而用户要的是壳里那个 DAW 的名字。
pub fn resolve_identity(evidence: &[Evidence]) -> HostIdentity {
    let identity = HostIdentity {
        evidence: evidence.to_vec(),
        ..HostIdentity::unknown()
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

    // 4a. 只有家族级证据（家族共用的壳 / 扫描器）：报到家族，并说清经由谁。
    //     产品名这里给不出来，也**不猜** —— 见 `FAMILY_MARKERS`。
    let family_hit = evidence
        .iter()
        .filter(|item| {
            item.source == EvidenceSource::ProcessImage
                || item.source == EvidenceSource::Ancestor
                || item.source == EvidenceSource::Vst3Name
        })
        .find_map(|item| family_of(&item.name).map(|family| (item, family)));
    if let Some((item, family)) = family_hit {
        let via = evidence
            .iter()
            .filter(|other| {
                other.source == EvidenceSource::ProcessImage
                    || other.source == EvidenceSource::Ancestor
            })
            .find(|other| is_wrapper(&other.name))
            .map(|other| normalize(&other.name));
        return HostIdentity {
            display: match &via {
                Some(wrapper) => format!("{family} (via {wrapper})"),
                None => family.to_string(),
            },
            source: Some(item.source),
            matched: Some(item.name.clone()),
            via,
            ..identity
        };
    }

    // 4b. 只剩**产品名包装器**命中（`fruity-wrapper`）：报它，并注明经由谁。
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

// ── VEGAS 家族包裹层的 DPI 提升（DllMain 阶段） ──────────────────────────────
//
// 【现场事实】ACID Pro 11（125% 缩放）里，承载插件编辑器的包裹窗口是在
// DPI-unaware 线程上创建的（附着点实测 parent-dpi=96 parent-unaware=true，而
// 宿主主界面是感知的 —— 截图里 ACID 自己的 UI 清晰、插件面板整体模糊）。
// unaware 窗口内的所有内容先按 96-DPI 渲染、再被 DWM 整体拉伸 1.25×：这就是
// "插件面板模糊"；感知宿主按物理像素给包裹窗留位、包裹窗把数字虚拟化，两边
// 各说一套长度，这就是"溢出 ⇒ 滚动条 + 右/下被裁"。
//
// 【为什么在 DllMain 提升线程】包裹窗口与插件模块同进程（`acidpro.exe` 内的
// sfvstwrap），窗口由**加载我们之前的那个线程**创建则来不及 —— 但若包裹层先
// LoadLibrary 再开窗（典型顺序：装载插件、问 getSize、再建窗），那么在
// DLL_PROCESS_ATTACH 把**加载线程**的上下文切成 PMv2，随后创建的包裹窗口与
// 子窗口就都在感知上下文里：与感知宿主同尺度，无拉伸（清晰）、无虚拟化错位
// （无溢出）。对已经感知的线程（VEGAS 的 system-aware 包裹层，实测工作正常）
// 不做任何事 —— 能工作的东西不动。
//
// 【为什么只圈 VEGAS 家族】与 `hide_ara_entry_points` 同一条纪律：这是**加载
// 兼容性**闸门，不是编辑器能力分支。让任意宿主的加载线程永久改上下文是越权
// 行为；家族包裹层（sfvstwrap）被现场证据钉死了"unaware 包裹 + 感知宿主"这
// 一种形状，才值得做。
//
// 【逃生门】`HIFISHIFTER_NO_DPI_PROMOTE=1` 停用提升，回到包裹层原生行为。

/// 提升结果（`wrapper_thread_dpi_promote_outcome` 的读数；附着点打进日志）。
#[cfg(windows)]
static WRAPPER_PROMOTE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(windows)]
const PROMOTE_NOT_ATTEMPTED: u8 = 0;
#[cfg(windows)]
const PROMOTE_APPLIED: u8 = 1;
#[cfg(windows)]
const PROMOTE_THREAD_ALREADY_AWARE: u8 = 2;
#[cfg(windows)]
const PROMOTE_NOT_VEGAS_FAMILY: u8 = 3;
#[cfg(windows)]
const PROMOTE_ESCAPE_HATCH: u8 = 4;
#[cfg(windows)]
const PROMOTE_FAILED: u8 = 5;

/// DllMain 阶段调用：把 DPI-unaware 的加载线程提升为 PMv2（仅 VEGAS 家族映像）。
///
/// 只做三类调用：进程映像名查询、本线程上下文查询、上下文设置 —— 都是 kernel32
/// 的无同步标志位操作，loader lock 下安全（不碰 COM / 线程同步 / 其它 DLL）。
/// 结果只落原子量，日志推迟到附着点（`log_host_dpi_facts`）再打：DllMain 里做
/// 文件 I/O 是 loader-lock 事故。
#[cfg(windows)]
pub fn promote_wrapper_thread_dpi() {
    use windows::Win32::System::LibraryLoader::GetModuleFileNameW;
    use windows::Win32::UI::HiDpi::{
        GetThreadDpiAwarenessContext, SetThreadDpiAwarenessContext,
        DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, DPI_AWARENESS_CONTEXT_UNAWARE,
    };

    // 一次进程只试一次：多次 LoadLibrary（扫描 + 实例化）不重复动手。
    let previous = WRAPPER_PROMOTE.swap(PROMOTE_NOT_ATTEMPTED, std::sync::atomic::Ordering::AcqRel);
    if previous != PROMOTE_NOT_ATTEMPTED {
        WRAPPER_PROMOTE.store(previous, std::sync::atomic::Ordering::Release);
        return;
    }
    if std::env::var_os("HIFISHIFTER_NO_DPI_PROMOTE").is_some_and(|value| value == "1") {
        WRAPPER_PROMOTE.store(PROMOTE_ESCAPE_HATCH, std::sync::atomic::Ordering::Release);
        return;
    }
    // 进程映像名（GetModuleFileNameW(NULL)）比 Toolhelp 快照轻得多 —— 这段跑在
    // loader lock 里。
    let mut buffer = [0_u16; 512];
    let length = unsafe { GetModuleFileNameW(None, &mut buffer) } as usize;
    let image = String::from_utf16_lossy(&buffer[..length.min(buffer.len())]).to_ascii_lowercase();
    let in_vegas_family = ["acidpro", "vegas", "soundforge", "sound forge", "sfvst"]
        .iter()
        .any(|marker| image.contains(marker));
    if !in_vegas_family {
        WRAPPER_PROMOTE.store(
            PROMOTE_NOT_VEGAS_FAMILY,
            std::sync::atomic::Ordering::Release,
        );
        return;
    }
    // SAFETY: 只读查询，伪句柄比较。
    let thread_context = unsafe { GetThreadDpiAwarenessContext() };
    let thread_unaware = unsafe {
        windows::Win32::UI::HiDpi::AreDpiAwarenessContextsEqual(
            thread_context,
            DPI_AWARENESS_CONTEXT_UNAWARE,
        )
    }
    .as_bool();
    if !thread_unaware {
        // 感知线程（VEGAS 的 system-aware 包裹层 / REAPER 等 PMv2 宿主）不动。
        WRAPPER_PROMOTE.store(
            PROMOTE_THREAD_ALREADY_AWARE,
            std::sync::atomic::Ordering::Release,
        );
        return;
    }
    // SAFETY: 参数是常量伪句柄；失败返回 null 伪句柄。**刻意不还原**：提升的意义
    // 就是让本线程此后创建的包裹窗口与子窗口都落在感知上下文里。
    let switched =
        unsafe { SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) };
    let outcome = if switched.0.is_null() {
        PROMOTE_FAILED
    } else {
        PROMOTE_APPLIED
    };
    WRAPPER_PROMOTE.store(outcome, std::sync::atomic::Ordering::Release);
}

/// 提升结果的日志读数（附着点的 `editor DPI facts` 行）。
pub fn wrapper_thread_dpi_promote_outcome() -> &'static str {
    #[cfg(windows)]
    {
        match WRAPPER_PROMOTE.load(std::sync::atomic::Ordering::Acquire) {
            PROMOTE_APPLIED => "applied",
            PROMOTE_THREAD_ALREADY_AWARE => "thread-already-aware",
            PROMOTE_NOT_VEGAS_FAMILY => "not-vegas-family",
            PROMOTE_ESCAPE_HATCH => "escape-hatch",
            PROMOTE_FAILED => "failed",
            _ => "not-attempted",
        }
    }
    #[cfg(not(windows))]
    {
        "not-applicable"
    }
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

    /// 扫描服务器是家族共用组件，名字里没有产品信息：只能报到家族。
    #[test]
    fn the_scan_server_only_proves_the_family() {
        let identity = resolve_identity(&[evidence(
            EvidenceSource::Vst3Name,
            "audio_plugin_scan_server_vst3_x64",
        )]);
        assert_eq!(identity.display, "VEGAS Creative Software");
        assert_eq!(identity.via, None);
    }

    /// `sfvstwrap` 是 VEGAS Creative Software **全家**共用的壳：单独出现时只能证明家族，
    /// **推不出** VEGAS Pro（ACID Pro / Sound Forge / Movie Studio 用的是同一个壳）。
    #[test]
    fn the_vegas_wrapper_alone_only_proves_the_family() {
        let identity = resolve_identity(&[
            evidence(EvidenceSource::ProcessImage, "sfvstwrap.exe"),
            evidence(EvidenceSource::Vst3Name, "sfvstwrap"),
        ]);
        assert_eq!(identity.display, "VEGAS Creative Software (via sfvstwrap)");
        assert_eq!(identity.via.as_deref(), Some("sfvstwrap"));
        assert!(
            !identity.display.contains("VEGAS Pro"),
            "包装器名里没有产品信息，不能猜成 VEGAS Pro"
        );
    }

    /// 同一个壳后面的兄弟产品：祖先链认出来就报它，而不是 VEGAS Pro。
    #[test]
    fn a_sibling_product_in_the_family_is_named_instead_of_vegas_pro() {
        for (exe, expected) in [
            ("vegas180.exe", "VEGAS Pro"),
            ("acidpro.exe", "ACID Pro"),
            ("soundforge.exe", "Sound Forge"),
            ("movie_studio_17.exe", "Movie Studio"),
        ] {
            let identity = resolve_identity(&[
                evidence(EvidenceSource::ProcessImage, "sfvstwrap.exe"),
                evidence(EvidenceSource::Ancestor, exe),
            ]);
            assert_eq!(identity.display, expected, "{exe}");
            assert_eq!(identity.source, Some(EvidenceSource::Ancestor), "{exe}");
            assert_eq!(identity.via, None, "{exe}");
        }
    }

    /// 祖先认不出来时如实报家族 —— 这正是此前会错报 VEGAS Pro 的那个形状。
    #[test]
    fn an_unrecognised_ancestor_behind_the_wrapper_does_not_become_vegas_pro() {
        let identity = resolve_identity(&[
            evidence(EvidenceSource::ProcessImage, "sfvstwrap.exe"),
            evidence(EvidenceSource::Ancestor, "some-unknown-daw.exe"),
        ]);
        assert_eq!(identity.display, "VEGAS Creative Software (via sfvstwrap)");
        assert!(identity
            .describe()
            .contains("process-ancestor:some-unknown-daw.exe"));
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
        // 家族共用的壳与扫描器**不是**产品名，别名表必须查不到它们。
        assert_eq!(lookup("sfvstwrap.exe"), None);
        assert_eq!(lookup("audio_plugin_scan_server_vst3_x64"), None);
        assert_eq!(family_of("sfvstwrap.exe"), Some("VEGAS Creative Software"));
        assert_eq!(
            family_of("audio_plugin_scan_server_vst3_x64"),
            Some("VEGAS Creative Software")
        );
        assert_eq!(family_of("fruity-wrapper"), None);
    }

    /// VEGAS 家族宿主隐藏 ARA 入口点（家族桥拒绝 ARA 标记的模块，ACID Pro 11 实测）。
    #[test]
    fn the_vegas_family_hides_ara_entry_points() {
        assert!(ara_entry_points_hidden(true, None, None));
    }

    /// 其它宿主保持 ARA 暴露（REAPER / Cubase / Studio One 不受影响）。
    #[test]
    fn other_hosts_still_expose_ara_entry_points() {
        assert!(!ara_entry_points_hidden(false, None, None));
    }

    /// `HIFISHIFTER_NO_ARA=1` 是全局杀开关：压过家族闸门，也压过 FORCE_ARA。
    #[test]
    fn no_ara_wins_over_everything() {
        assert!(ara_entry_points_hidden(false, Some("1"), None));
        assert!(ara_entry_points_hidden(true, Some("1"), None));
        assert!(ara_entry_points_hidden(true, Some("1"), Some("1")));
        // 值要 trim 后等于 "1"：首尾空白不影响判定，其它值一律视为未设置。
        assert!(ara_entry_points_hidden(true, Some(" 1 "), None));
        assert!(ara_entry_points_hidden(false, Some(" 1 "), Some("0")));
        assert!(!ara_entry_points_hidden(true, Some("on"), Some("1")));
    }

    /// `HIFISHIFTER_FORCE_ARA=1` 只覆盖家族闸门，给修好的宿主版本与现场排查留门。
    #[test]
    fn force_ara_only_overrides_the_family_gate() {
        assert!(!ara_entry_points_hidden(true, None, Some("1")));
        assert!(!ara_entry_points_hidden(false, None, Some("1")));
        assert!(!ara_entry_points_hidden(true, Some("0"), Some("1")));
    }
}
