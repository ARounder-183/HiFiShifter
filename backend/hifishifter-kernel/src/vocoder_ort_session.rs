//! 共用ORT会话策略；整段HNSEP的动态CPU工作区不保留巨型arena，不改变模型/DSP或分块语义。
//! Shared ORT session builder with consistent optimization policy.
//!
//! All three ONNX models (NSF-HiFiGAN, FCPE, HNSEP) should use the same
//! optimization stack. GPU acceleration is provided by:
//!   - CoreML EP (Metal / Neural Engine) — macOS ARM64, primary GPU path
//!   - WebGPU EP (Dawn backend)          — Linux x86_64 (Vulkan), macOS (Metal)
//!   - DirectML EP (DX12)                — Windows only
//!
//! On platforms without GPU prebuilt binaries, sessions gracefully fall
//! back to CPU. This module centralizes EP registration so individual
//! vocoder modules don't drift.
//!
//! # Shape policy
//!
//! Sessions keep the models' dynamic axes (`batch`, `time`) **unpinned**.
//! An earlier revision pinned the vocoder's `time` axis to a constant so the
//! CoreML EP could compile the graph, but that made CoreML ~1.7x *slower*
//! than CPU (measured: 10.2 s vs 6.2 s for 47.6 s of audio) and forced every
//! chunk — however short — to be padded up to that constant.  On ONNX Runtime
//! 1.28 the CoreML EP compiles the dynamic graph without help and runs it
//! ~7.7x faster than CPU with bit-comparable output.  See
//! `docs/hifigan-gpu-acceleration.md` for the full measurement table.

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use serde::Serialize;
use std::cell::Cell;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// 声码器烟测探针的 mel 帧数：**等于生产推理长度**（`CHUNK_MAX_FRAMES_DEFAULT`
/// = 512 帧 ≈ 5.9 s @ hop 512 / 44.1 kHz）。
///
/// # 为什么必须等于生产长度，而不是"越大越保险"
///
/// 探针只需证明"这个 EP 能在真实形状上跑通一次推理"，所以最合适的长度就是
/// 生产里真实使用的长度 —— 它**必然合法**（渲染每块就是这么大），且代价最小。
///
/// 这里曾经是 4096：那是**旧**分块方案（`RENDER_PIPELINE_VERSION` v6 之前，
/// 每块 47.6 s）的遗留值，分块降到 512 后没有跟着改。后果不是"多花一点时间"，
/// 而是把一台**能用的设备判成挂起**：烟测要求 GPU 一次合成 47.6 s 音频，比生产
/// 中任何一次推理都大 8 倍。实测（`pc_nsf_hifigan.onnx`，Intel UHD Graphics，
/// device_id=1）：4096 帧探针 10 s 内不返回 → 被判为"首次推理挂起" →
/// `disable_directml` 把**整个进程**的 DirectML 关掉，于是同一台机器上的
/// RTX 3050 从未被尝试，基准测试的 DirectML 分支静默退化成 CPU。
///
/// 该值与生产分块长度的耦合由 `nsf_hifigan_onnx` 的
/// `vocoder_smoke_probe_matches_the_render_chunk` 测试钉住。
pub(crate) const VOCODER_SMOKE_FRAMES: usize = 512;

/// 音高检测（FCPE）烟测探针的 mel 帧数。
///
/// 与声码器不同，FCPE 没有可对齐的"生产分块长度"（它按整段分析），而 4096
/// 帧在这台机器的集显上 3.4 s 返回、从未被误判 —— 没有实测证据支持收紧，
/// 改了只会引入未经验证的风险，故保持原值。
const PITCH_SMOKE_FRAMES: usize = 4096;

/// 声码器的探针必须比音高检测的更便宜（前者对齐生产分块，后者是历史值）。
/// 编译期断言：两个常量若被改成相同量级，这里先失败，不必等运行时才发现。
const _: () = assert!(VOCODER_SMOKE_FRAMES < PITCH_SMOKE_FRAMES);

/// 烟测探针的帧轴长度（按角色）。
///
/// 【为什么不能对所有动态维都用同一个值】HNSEP 的 mask-only 模型输入是
/// `[batch, 2, 1025, n_frames]`，若把 4096 套到**每一个**动态维，探针张量
/// 会变成 `[1, 2, 1025, 4096]` —— **32 MB**，而波形域旧模型只要 16 KB
/// （差 2000 倍）。烟测只需证明"EP 能跑通一次推理"，不需要真实批量。
///
/// 取值须满足各模型对帧轴的整除约束：HNSEP 要求 `n_frames % 32 == 0`。
fn smoke_probe_frames(role: OrtSessionRole) -> usize {
    match role {
        // 32 帧 = 一个完整 segment，是模型要求的最小合法长度。
        OrtSessionRole::Separator => 32,
        OrtSessionRole::Vocoder => VOCODER_SMOKE_FRAMES,
        OrtSessionRole::PitchDetector => PITCH_SMOKE_FRAMES,
    }
}

/// 烟测（首次推理预热）超时。
///
/// 健康的首次推理只需 0.15~1.7s（含 DML 着色器编译）；超时意味着 EP 在
/// 本进程中已不可用。此值必须**远小于**用户可忍受的冻结时长：设备切换
/// （如 CPU→GPU）后新建的 DirectML 会话偶发首次推理挂起（观察到会话创建
/// 成功后烟测永不返回），而烟测在构建路径内 —— 超时值直接决定"渲染进度
/// 卡在 0%"的时长。旧值 60s（且 strict 与回退两次尝试累计 120s）会让用户
/// 以为应用永久卡死（实测用户在 20s 后强杀应用）。
const SMOKE_TEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// 各 EP 的烟测超时：CoreML 的 MLProgram 首次推理包含图编译，健康路径也
/// 可能长达数十秒，必须保持宽松；DirectML/WebGPU 的健康首次推理在秒级，
/// 挂起必须快速检出。
fn smoke_test_timeout(ep_name: &str) -> std::time::Duration {
    if ep_name == "CoreML" {
        std::time::Duration::from_secs(60)
    } else {
        SMOKE_TEST_TIMEOUT
    }
}

/// 各 EP 的会话创建（commit）超时。
///
/// ORT 的 `commit_from_file` 没有超时 API，而实测设备切换后重建 DirectML
/// 会话时 commit 本身也会挂起（不只是烟测阶段）。超时后放弃本次创建并让
/// 调用方回退，避免一个挂起的 commit 永久持有全局单飞锁。
/// CoreML 的首次模型编译（无语料缓存时）可能数十秒，保持宽松。
fn creation_timeout(ep_name: &str) -> std::time::Duration {
    if ep_name == "CoreML" {
        std::time::Duration::from_secs(120)
    } else {
        SMOKE_TEST_TIMEOUT
    }
}

/// Set once a CoreML smoke test times out or fails hard.  The CoreML EP is
/// then skipped for the rest of the process (WebGPU/CPU take over) so a
/// hung CoreML inference can never block the benchmark or rendering again.
static COREML_DISABLED: OnceLock<std::sync::atomic::AtomicBool> = OnceLock::new();

/// Set once a WebGPU inference (smoke test or real run) times out.  The
/// WebGPU EP is then skipped for the rest of the process (CPU takes over),
/// mirroring the CoreML latch.  Without it, a timed-out Dawn/Vulkan hang
/// would be retried with a *fresh* WebGPU session after every timeout —
/// each retry costing the full 120 s again (see the vocoder's
/// `disable_hung_ep_and_rebuild`).
static WEBGPU_DISABLED: OnceLock<std::sync::atomic::AtomicBool> = OnceLock::new();

/// Reader is only referenced by the WebGPU-capable session-build branches
/// (Linux x86_64 / macOS ARM64); gate it like `coreml_disabled` so Windows
/// builds don't see a dead function.
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
fn webgpu_disabled() -> bool {
    WEBGPU_DISABLED
        .get_or_init(|| std::sync::atomic::AtomicBool::new(false))
        .load(std::sync::atomic::Ordering::Relaxed)
}

pub fn disable_webgpu(reason: &str) {
    log::warn!("ort_session: disabling WebGPU EP for this process: {reason}");
    WEBGPU_DISABLED
        .get_or_init(|| std::sync::atomic::AtomicBool::new(false))
        .store(true, std::sync::atomic::Ordering::Relaxed);
}

/// DirectML 的"坏设备"记账。
///
/// 【为什么按**设备**记账，而不是按进程】DirectML 的 `device_id` 是 DXGI
/// 适配器序号：一台机器上通常既有独显（0）又有集显（1）。旧实现用一个
/// `AtomicBool` 把"某台设备首次推理超时"升级成"本进程再也不用 DirectML" ——
/// 集显上的一次超时会把独显一起废掉，而**基准测试正是用来分辨这两者的工具**。
/// 实测：集显超时后，基准测试的 DirectML 分支静默退化成 CPU，用户看到的是
/// "DirectML 可用 / 失败"，日志里连一句原因都没有。
///
/// 记账规则：
///   - `devices`：已知在该设备上挂起/超时的适配器序号；
///   - `all`：无法确定设备（DXGI 枚举失败、只能走 DML2 自动选择）时的整体禁用。
#[cfg(target_os = "windows")]
#[derive(Debug, Default)]
struct DmlDisableState {
    all: bool,
    devices: std::collections::BTreeSet<i32>,
}

#[cfg(target_os = "windows")]
impl DmlDisableState {
    /// 该设备是否已被判为不可用。`None` = 不知道设备，只有整体禁用才算数。
    fn is_disabled(&self, device: Option<i32>) -> bool {
        self.all || device.is_some_and(|d| self.devices.contains(&d))
    }

    /// 记下一次失败。返回"此前是否可用"（只在状态真正翻转时打日志）。
    fn disable(&mut self, device: Option<i32>) -> bool {
        match device {
            Some(d) => self.devices.insert(d),
            None => !std::mem::replace(&mut self.all, true),
        }
    }

    /// 撤销禁用（基准测试实测跑通该设备时调用）。返回"此前是否被禁用"。
    fn enable(&mut self, device: Option<i32>) -> bool {
        match device {
            Some(d) => self.devices.remove(&d),
            None => std::mem::replace(&mut self.all, false),
        }
    }
}

#[cfg(target_os = "windows")]
fn dml_disable_state() -> &'static Mutex<DmlDisableState> {
    static SLOT: OnceLock<Mutex<DmlDisableState>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(DmlDisableState::default()))
}

#[cfg(target_os = "windows")]
fn directml_disabled(device: Option<i32>) -> bool {
    dml_disable_state()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .is_disabled(device)
}

/// 把某台设备（`None` = 无法确定设备，即全部）标记为不可用。
///
/// 触发场景：设备切换后重建 DirectML 会话时首次推理挂起。不这样做的话，
/// strict 与回退两次尝试各要等一次超时，且其他模型（FCPE/HNSEP）的预热
/// 也排队在全局构建锁后，整条 ORT 链冻结。禁用后：一次超时（≤10s）即收敛，
/// 渲染随即在 CPU 上继续推进。
#[cfg(target_os = "windows")]
pub(crate) fn disable_directml(device: Option<i32>, reason: &str) {
    let newly = dml_disable_state()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .disable(device);
    if newly {
        let device_desc = describe_dml_device(device);
        log::error!(
            "ort_session: disabling DirectML for device {device_desc}: {reason}. \
             Rendering continues on CPU for that device; run the benchmark to re-test it."
        );
    }
}

/// 撤销某台设备的禁用（基准测试实测跑通时调用）。
///
/// 【为什么必须能撤销】旧实现只能"重启应用重试" —— 用户明明刚在基准测试里看到
/// 这台设备 300 ms 跑完一轮，渲染却仍被闩锁挡在 CPU 上，直到下次启动。
pub(crate) fn enable_directml(device: Option<i32>) {
    #[cfg(target_os = "windows")]
    {
        let was = dml_disable_state()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .enable(device);
        if was {
            let device_desc = describe_dml_device(device);
            log::warn!(
                "ort_session: DirectML re-enabled for device {device_desc} (a diagnostic run completed on it)"
            );
        }
    }
    #[cfg(not(target_os = "windows"))]
    let _ = device;
}

#[cfg(target_os = "windows")]
fn describe_dml_device(device: Option<i32>) -> String {
    match device {
        Some(id) => format!("#{id}"),
        None => "(auto)".to_string(),
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn coreml_disabled() -> bool {
    COREML_DISABLED
        .get_or_init(|| std::sync::atomic::AtomicBool::new(false))
        .load(std::sync::atomic::Ordering::Relaxed)
}

pub fn disable_coreml(reason: &str) {
    log::warn!("ort_session: disabling CoreML EP for this process: {reason}");
    COREML_DISABLED
        .get_or_init(|| std::sync::atomic::AtomicBool::new(false))
        .store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Build a CoreML execution provider with the options that make the
/// NSF-HiFiGAN model compile reliably on Apple Silicon.
///
/// - `MLProgram` format: supports more operators and is required for many
///   models that the legacy NeuralNetwork format rejects.
/// - `CPUAndGPU`: prefer the Metal GPU for real-time vocoding, with CPU
///   fallback for unsupported ops.
/// - Dynamic input shapes: the models keep their dynamic `batch`/`time`
///   axes.  Requiring static shapes here was measured to make the vocoder
///   ~1.7x *slower* than the CPU EP (see the module docs).
/// - A persistent model cache avoids recompiling the CoreML model on every
///   session creation (can take tens of seconds for this 56 MB model).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn build_coreml_ep() -> ort::ep::CoreML {
    use ort::ep::coreml::{ComputeUnits, ModelFormat};

    let mut ep = ort::ep::CoreML::default()
        // Use the GPU instead of the Neural Engine: HiFi-GAN's ConvTranspose
        // upsampling layers (stride/kernel 16/8/4/2) are known to hang the
        // ANE compiler (see Apple Developer Forums: ConvTranspose2d with
        // stride(16,1) kernel(16,1) breaks the ANE).  CPUAndGPU keeps the
        // model on the Metal GPU where these layers run correctly.
        .with_compute_units(ComputeUnits::CPUAndGPU)
        .with_model_format(ModelFormat::MLProgram)
        // Keep FP32 accumulation: HiFi-GAN audio quality is sensitive to
        // low-precision GPU accumulation.
        .with_low_precision_accumulation_on_gpu(false)
        // Every role keeps the model's dynamic `batch`/`time` axes.
        // Requiring static shapes forces CoreML to compile one fully-static
        // MLProgram, which benchmarked ~1.7x SLOWER than the CPU EP.
        .with_static_input_shapes(false);

    // Cache compiled CoreML programs under ~/Library/Caches/HiFiShifter so
    // repeated session creation (e.g. after an EP switch) does not recompile
    // the whole model.  If the cache directory cannot be created, continue
    // without caching -- it is purely an optimization.
    if let Some(home) = std::env::var_os("HOME") {
        let cache = std::path::PathBuf::from(home)
            .join("Library")
            .join("Caches")
            .join("HiFiShifter")
            // Versioned separately from the old ORT 1.24 cache: compiled
            // CoreML artifacts from the previous runtime can reuse stale
            // partitions and must not be loaded by the new build.
            .join("coreml-ort1.28");
        if std::fs::create_dir_all(&cache).is_ok() {
            // with_model_cache_dir takes `impl ToString`, so convert the
            // PathBuf explicitly (PathBuf itself does not implement
            // ToString).
            ep = ep.with_model_cache_dir(cache.to_string_lossy().into_owned());
        }
    }

    ep
}

/// Runtime override for EP choice. Set by `set_runtime_ep_override()`.
/// Takes precedence over the `HIFISHIFTER_ORT_EP` env var.
static RUNTIME_EP_OVERRIDE: OnceLock<Mutex<Option<String>>> = OnceLock::new();

/// Runtime override for DirectML device ID. Set by `set_runtime_dml_device_id()`.
/// Takes precedence over the `HIFISHIFTER_DML_DEVICE_ID` env var.
static RUNTIME_DML_DEVICE_ID: OnceLock<Mutex<Option<i32>>> = OnceLock::new();

/// EP 设置代数：`set_runtime_ep_override` / `set_runtime_dml_device_id` 每次
/// 实际变更取值时递增。
///
/// 会话构建是秒级操作 —— 构建期间用户再次切换设备时，构建结果就是旧
/// EP / 旧设备的陈旧会话。构建方在构建前后各读一次该值即可检测并丢弃
/// 陈旧结果（见各模块 `get_or_init_shared_session`）。
static EP_SETTINGS_GENERATION: AtomicU64 = AtomicU64::new(0);

/// 当前 EP 设置代数。
pub fn ep_settings_generation() -> u64 {
    EP_SETTINGS_GENERATION.load(Ordering::Acquire)
}

/// Set the runtime EP override. Pass `None` to clear the override.
pub fn set_runtime_ep_override(ep: Option<String>) {
    let mut guard = RUNTIME_EP_OVERRIDE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if guard.as_deref() != ep.as_deref() {
        EP_SETTINGS_GENERATION.fetch_add(1, Ordering::AcqRel);
        *guard = ep;
    }
}

/// Set the runtime DirectML device ID override. Pass `None` to clear.
pub fn set_runtime_dml_device_id(device_id: Option<i32>) {
    let mut guard = RUNTIME_DML_DEVICE_ID
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if *guard != device_id {
        EP_SETTINGS_GENERATION.fetch_add(1, Ordering::AcqRel);
        *guard = device_id;
    }
}

/// 全局会话构建**单飞**语义（三个 ONNX 模块 Vocoder / FCPE / HNSEP 共用，
/// 同一时刻全局只允许一次会话构建：创建 + 烟测整体），实现见
/// [`acquire_session_build_lock`] 的**可抢占租约**。
///
/// 为什么必须全局单飞：DirectML 的 D3D12 设备创建与首次推理都不能与其他
/// DML 操作并发 —— 实测两种致命并发都会在驱动层挂起（渲染进度永久卡 0%）：
///   1. 两个线程并发创建 DML 会话；
///   2. 一个会话正在跑首次推理（烟测）时，另一个线程创建新会话。
/// 启动时三个模型的构建/烟测是顺序完成的，从未挂起 —— 单飞即复现该安全
/// 模式。挂起由烟测超时兜底（SMOKE_TEST_TIMEOUT，DirectML 10s）：
/// 超时后 `disable_directml` 使后续构建立即走 CPU，等待本锁的线程最多等
/// 一个超时周期即可继续（不会永久卡死）。
///
/// ★ 纪律：持有本锁期间**绝不允许**再去锁各模块的会话容器（容器锁的持有
/// 时间必须保持在微秒级）。各模块流程：容器锁快速检查（随即释放）→ 本锁
/// 内构建 + 烟测 → 容器锁写回。
///
/// 会话构建租约：持有者票据（0 = 空闲）。票据高位为取得时刻（Unix 毫秒）。
///
/// 为什么不是 `Mutex`：实测存在构建线程**永久卡死**的情形（ORT/DirectML 在
/// 驱动层挂起，任何超时都无法打断它）。`Mutex` 一旦被这样的线程持有，全应用
/// 的会话构建与渲染线程都会永久阻塞 —— 表现为渲染进度永远卡在 0%。租约式
/// 锁允许后续请求在持有者远超租约后**抢占**并记录日志，使应用总能恢复。
static BUILD_LEASE: OnceLock<AtomicU64> = OnceLock::new();
static LEASE_SEQ: AtomicU64 = AtomicU64::new(1);

/// 持有者超过该时长仍未释放（且无心跳推进）即视为卡死，可被抢占。
///
/// 【为什么必须是"最坏构建 + 最慢单轮"的上界】心跳只在每轮推理（含预热轮）之间
/// 调用，相邻两次心跳之间隔着：严格构建 10s + 烟测 10s + 回退构建 10s + 烟测 10s
/// + 最慢单轮推理 120s。阈值低于这个间隔，就会把"慢而活着"的持有者判死 ——
/// 误抢的后果正是这把锁要防的驱动层挂起。240s 留出一倍余量。
const LEASE_STEAL_AFTER: std::time::Duration = std::time::Duration::from_secs(240);

fn lease() -> &'static AtomicU64 {
    BUILD_LEASE.get_or_init(|| AtomicU64::new(0))
}

fn now_ms() -> u64 {
    // 测试钩子：租约用例用它把"持有时长"拨到任意值，免去真实等待一个
    // 抢占阈值。u64::MAX 表示"未接管"。只有 `LEASE_SERIAL` 串行的用例会写它，
    // 生产路径永远走下面的单调时钟。
    #[cfg(test)]
    {
        let faked = TEST_CLOCK_MS.load(Ordering::Relaxed);
        if faked != u64::MAX {
            return faked;
        }
    }
    process_start().elapsed().as_millis() as u64
}

/// `now_ms()` 的零点。票据只在本进程内有意义、从不持久化，因此用单调时钟
/// 而不是 Unix 时间：系统时钟前跳（NTP 校时）会把"持有时长"虚增过抢占阈值，
/// 让健康的持有者被误抢 —— 而误抢的后果正是这把锁要防的驱动层挂起。
fn process_start() -> &'static std::time::Instant {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START.get_or_init(std::time::Instant::now)
}

#[cfg(test)]
static TEST_CLOCK_MS: AtomicU64 = AtomicU64::new(u64::MAX);

fn make_ticket() -> u64 {
    ((now_ms() & 0x0000_FFFF_FFFF_FFFF) << 16)
        | (LEASE_SEQ.fetch_add(1, Ordering::Relaxed) & 0xFFFF)
}

fn ticket_ms(ticket: u64) -> u64 {
    ticket >> 16
}

thread_local! {
    /// 本线程持有的**票据**（0 = 未持有）。可重入判据。
    ///
    /// 【为什么记票据而不是记布尔】租约可被抢占。只记"我持过"的话，被抢后本线程
    /// 仍会被记为持锁者，嵌套获取会拿到一个不设防的空守卫直接去建会话 —— 重新
    /// 打开两种致命并发。记票据才能发现"我的租约已经不在我手里"，如实按普通
    /// 路径重新竞争。
    ///
    /// 【为什么必须可重入】诊断运行现在**整段**持有租约（见 [`begin_diagnostic_run`]），
    /// 而它内部要调构建路径。不可重入的话，嵌套的 `acquire_session_build_lock` 会看到
    /// "租约被占用"（占用者其实是自己），等满超时后失败 —— 一次纯粹的自锁。
    static LEASE_HELD: Cell<u64> = const { Cell::new(0) };
}

/// 会话构建租约守卫：`Drop` 时释放。
///
/// 票据的**高位时间戳**是可变的（见 [`Self::heartbeat`]），所以它是 `Cell` 而不是裸
/// `u64`：心跳要推进时间戳，而 `Drop` 必须用**当前**票据去释放，否则释放不掉。
pub struct SessionBuildLease {
    ticket: Cell<u64>,
    /// 重入得到的守卫：外层已经持有租约，`Drop` 什么都不做。
    nested: bool,
}

impl SessionBuildLease {
    /// 把"最后一次有进展"的时刻推到此刻。
    ///
    /// 【为什么必须有它】抢占阈值 [`LEASE_STEAL_AFTER`] 想回答的是"持有者多久没有进展"，
    /// 而它读的是**取得时刻** —— 于是一个**合法但慢**的持有者（基准测试实测 31–134 秒）
    /// 会在阈值处被判成卡死并被抢占。抢占的后果恰恰是这把锁存在的理由：两个线程并发
    /// 建 DML 会话、或建会话时另一个会话正在推理，都会在驱动层挂起。有心跳之后，
    /// "跑得久"与"卡死了"才真正可分。
    pub fn heartbeat(&self) {
        if self.nested {
            return;
        }
        loop {
            let stored = lease().load(Ordering::Acquire);
            // 租约已经不属于自己（被抢占 / 已释放）：不要把它"复活"成自己的。
            if stored != self.ticket.get() {
                return;
            }
            let refreshed = (now_ms() & 0x0000_FFFF_FFFF_FFFF) << 16 | (stored & 0xFFFF);
            if lease()
                .compare_exchange(stored, refreshed, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                self.ticket.set(refreshed);
                // TLS 里的可重入判据跟着刷新：它记的是票据本身，心跳改了高位
                // 时间戳之后若不同步，下一次重入会误判"我的租约不在了"。
                LEASE_HELD.with(|held| {
                    if held.get() == stored {
                        held.set(refreshed);
                    }
                });
                return;
            }
        }
    }
}

impl Drop for SessionBuildLease {
    fn drop(&mut self) {
        if self.nested {
            return;
        }
        let _ = lease().compare_exchange(self.ticket.get(), 0, Ordering::AcqRel, Ordering::Acquire);
        // 只有 TLS 里记的确实是**本守卫**的票据才清零：守卫可能跨线程 Drop，
        // 或本线程已重新持有了一张新票据 —— 两种情况下清零都会弄丢真记录。
        LEASE_HELD.with(|held| {
            if held.get() == self.ticket.get() {
                held.set(0);
            }
        });
    }
}

/// 会话构建失败的两类原因：**租约忙**（瞬态：等一会儿就能成）与其余失败。
///
/// 【为什么是类型而不是字符串】预热（`prewarm_verdict`）与 nsf 的线程本地缓存都要
/// 区分"**现在**建不了"与"建不起来"：前者不记账、允许重试，后者要让渲染链整条
/// 跳过。曾经的实现按错误文本的**子串**分类（匹配 "session build lock busy"），
/// 被 `check-ffi-guards.py` 的结构化错误门禁拦下 —— 本仓的协议语义必须以结构化
/// 形式旅行（见 `scripts/checks/baseline-error-contains.txt` 的头注），字符串只用于显示。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionBuildError {
    /// 全局构建租约被占（诊断运行整段持有它），等了 `waited` 没等到。
    Busy {
        /// 等待了多久才放弃。
        waited: std::time::Duration,
    },
    /// 其余失败（模型缺失、构建失败、配置非法……）。文本面向日志与用户。
    Failed(String),
}

impl std::fmt::Display for SessionBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Busy { waited } => write!(
                f,
                "session build lock busy for {waited:?} (another build appears stuck)"
            ),
            Self::Failed(error) => f.write_str(error),
        }
    }
}

impl From<String> for SessionBuildError {
    fn from(error: String) -> Self {
        Self::Failed(error)
    }
}

impl From<&str> for SessionBuildError {
    fn from(error: &str) -> Self {
        Self::Failed(error.to_string())
    }
}

impl From<SessionBuildError> for String {
    fn from(error: SessionBuildError) -> String {
        error.to_string()
    }
}

/// 有界地获取全局会话构建租约（供各模块 `get_or_init_shared_session` 使用）。
///
/// - 等待每 5s 打一次警告（日志可直接指认"卡在会话构建锁上"）；
/// - 超过 `timeout` 返回 [`SessionBuildError::Busy`]，调用方的会话加载失败 → 该
///   Clip 失败 → **渲染 pass 继续推进**（而不是整条链停摆），下次请求会重试；
/// - 持有者超过 `LEASE_STEAL_AFTER` 未释放 → **抢占**（ERROR 日志），
///   保证应用不会因一个卡死的构建而永久失去渲染能力。
pub fn acquire_session_build_lock(
    timeout: std::time::Duration,
) -> Result<SessionBuildLease, SessionBuildError> {
    // 重入：本线程已持有 ⇒ 直接给一个"不释放"的空守卫，绝不等自己。
    // 但 TLS 里记的是**票据**：租约若已被他人抢走（对不上自己的票据），
    // 这里必须如实清掉记录、按普通路径竞争 —— 布尔判据在这里会交出
    // 一个不设防的空守卫。
    let held = LEASE_HELD.with(Cell::get);
    if held != 0 {
        if lease().load(Ordering::Acquire) == held {
            return Ok(SessionBuildLease {
                ticket: Cell::new(0),
                nested: true,
            });
        }
        LEASE_HELD.with(|held| held.set(0));
    }
    let started = std::time::Instant::now();
    let mut next_log = std::time::Duration::from_secs(5);
    loop {
        let ticket = make_ticket();
        match lease().compare_exchange(0, ticket, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                if started.elapsed() > std::time::Duration::from_secs(1) {
                    log::warn!(
                        "ort_session: acquired the global session build lock after {:?} (a build was in progress)",
                        started.elapsed()
                    );
                }
                LEASE_HELD.with(|held| held.set(ticket));
                return Ok(SessionBuildLease {
                    ticket: Cell::new(ticket),
                    nested: false,
                });
            }
            Err(current) => {
                let held_ms = now_ms().saturating_sub(ticket_ms(current));
                if held_ms > LEASE_STEAL_AFTER.as_millis() as u64 {
                    log::error!(
                        "ort_session: session build lock has been held for {} ms (> {:?}) — the holder is stuck; \
                         stealing the lock so session builds can proceed",
                        held_ms,
                        LEASE_STEAL_AFTER
                    );
                    // 抢占成功 ⇒ 本线程就是新持有者：必须把守卫交出去。若这里
                    // continue 而不是 return，下一轮看到的是自己刚写入的票据，
                    // CAS(0, …) 永远不再成立，等待者只会等满超时失败 —— 而租约
                    // 停在一个无人持有的票据上（原持有者释放不掉、抢占者没拿到
                    // 守卫），之后所有会话构建都会失败到进程重启。
                    if lease()
                        .compare_exchange(current, ticket, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        LEASE_HELD.with(|held| held.set(ticket));
                        return Ok(SessionBuildLease {
                            ticket: Cell::new(ticket),
                            nested: false,
                        });
                    }
                    // CAS 失败（恰好被释放/被他人抢走）⇒ 落到下面的超时检查与
                    // 50ms 睡眠，下一轮按最新状态重试。
                }
            }
        }
        if started.elapsed() >= timeout {
            log::error!(
                "ort_session: could not acquire the global session build lock within {timeout:?}; \
                 another session build appears to be stuck. Failing this request so rendering can continue."
            );
            return Err(SessionBuildError::Busy { waited: timeout });
        }
        if started.elapsed() >= next_log {
            log::warn!(
                "ort_session: waiting for the global session build lock ({:?} elapsed) — another build is in progress",
                started.elapsed()
            );
            next_log += std::time::Duration::from_secs(5);
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

/// 一次诊断运行独占会话构建的**整个窗口**（建会话 + 全部计时推理）。
///
/// 【为什么必须整段持有，而不是只在建会话时持有】本模块已经把两种致命并发写清楚了
/// （见 [`BUILD_LEASE`] 的说明）：两个线程并发建 DML 会话、或一个会话正在跑首次推理
/// 时另一个线程建会话，**都会在驱动层挂起**。基准测试两条都占 —— 它自己建会话时不取
/// 租约，跑推理时也不取。此前它跑在"命令线程"上，恰好被单线程**顺带**护住；一旦把它
/// 挪到独立线程（插件的慢车道），这个窗口就从"几毫秒偶发"变成"31 秒必然"。
/// 所以"把长任务挪走"的前提，是先把窗口关上。
pub struct DiagnosticRun {
    lease: SessionBuildLease,
}

impl DiagnosticRun {
    /// 见 [`SessionBuildLease::heartbeat`]：长任务每跑一两轮推理调一次。
    pub fn heartbeat(&self) {
        self.lease.heartbeat();
    }
}

/// 开始一次诊断运行：整段持有会话构建租约。
///
/// 【为什么等不到就失败，而不是排队】诊断是用户主动触发、且自带"运行中"提示的动作。
/// 若另一个诊断已经在跑，让第二个静默等两分钟，比直接告诉用户"已经有一条在跑"更糟。
pub fn begin_diagnostic_run(label: &'static str) -> Result<DiagnosticRun, String> {
    let lease = acquire_session_build_lock(std::time::Duration::from_secs(20))
        .map_err(|error| format!("another diagnostic is already running: {error}"))?;
    log::info!("ort_session: diagnostic '{label}' holds the session build lease");
    Ok(DiagnosticRun { lease })
}

/// 预热失败里哪些**不构成**"模型不可用"的结论。
///
/// 【为什么必须有】预热把结果缓存进各模块的 `PREWARM`，而 `is_available() == false`
/// 会让渲染链**整条跳过**直到重启。于是"用户恰好在预热跑完前点了基准测试"就会变成
/// 一次永久停机：预热在租约上等到 20 秒超时 → 缓存 Err → 模型被判不可用。
/// 这两类失败回答的是"**现在**建不了"，不是"建不起来"，所以不记账，并允许下一次重试。
pub fn prewarm_verdict(outcome: Result<(), SessionBuildError>) -> Option<Result<(), String>> {
    // 【为什么只认"租约被占"】诊断运行现在整段持有构建租约（见 `DiagnosticRun`），
    // 于是预热有可能在它上面等到超时。那回答的是"**现在**建不了"，不是"建不起来"。
    // 判据是类型匹配，不再是字符串 —— 分类在编译期成立，措辞改动不再影响判定。
    let transient = matches!(&outcome, Err(SessionBuildError::Busy { .. }));
    if transient {
        log::info!(
            "[prewarm] session build deferred (a diagnostic is running); it will be built on demand"
        );
        None
    } else {
        Some(outcome.map_err(|error| error.to_string()))
    }
}

/// Resolve the DirectML device ID to use, in priority order:
/// 1. Runtime override (set via UI/settings) — **only when the user pinned a GPU**
/// 2. `HIFISHIFTER_DML_DEVICE_ID` env var
/// 3. Auto-detect via DXGI: pick the GPU with most VRAM
///
/// # 设备序号是**本进程**的，不是身份
///
/// 第 1 步的运行时覆盖由 [`dml_device_index_for_key`] 从**稳定的 PCI 标识**
/// （`vendor:device`）解析而来，见 `dml_adapters::DmlAdapterInfo::stable_key`。
/// 这里拿到的是**本进程**当前的 DXGI 序号 —— 实测同一分钟内 App 进程与 VEGAS
/// 插件进程的序号是互换的，所以持久化序号会让"同一份设置"在两个进程里指向
/// 两块不同的卡。
///
/// Always returns an explicit device_id. This uses `with_device_id(n)`
/// which calls `SessionOptionsAppendExecutionProvider_DML` (old API).
/// The newer `SessionOptionsAppendExecutionProvider_DML2` (used by
/// filter/preference options when device_id is None) has been observed
/// to create DML devices with significantly worse performance on
/// Ada Lovelace (RTX 4060) GPUs despite passing HighPerformance hints.
/// The old API with explicit device_id performs consistently across
/// all GPU architectures tested (Ampere, Ada, Pascal).
// DirectML 是 Windows-only EP：调用方只有下方 Windows 版会话构建，
// 非 Windows 目标不编译（否则 CI 上是死代码）。
#[cfg(target_os = "windows")]
fn resolve_dml_device_id() -> Option<i32> {
    // 1. Runtime override (set via UI/settings) — already resolved to this
    //    process's DXGI index by `ui_settings_apply` (see `dml_device_index_for_key`).
    if let Some(id) = RUNTIME_DML_DEVICE_ID
        .get_or_init(|| Mutex::new(None))
        .lock()
        .ok()
        .and_then(|g| *g)
    {
        log::warn!("ort_session: DirectML device_id={id} (from the stored adapter key)");
        return Some(id);
    }
    // 2. Env var — explicit override, honoured for every choice.
    if let Ok(val) = std::env::var("HIFISHIFTER_DML_DEVICE_ID") {
        if let Ok(id) = val.trim().parse::<i32>() {
            log::warn!("ort_session: HIFISHIFTER_DML_DEVICE_ID={id}");
            return Some(id);
        }
    }
    // 3. Auto-detect: pick the GPU with most VRAM.
    let adapters = crate::dml_adapters::enumerate_dml_adapters().adapters;
    if let Some(best) = adapters.first() {
        let device_id = best.device_id as i32;
        log::warn!(
            "ort_session: auto-detected DML device_id={device_id} name='{}' vram={}MB",
            best.name,
            best.dedicated_video_memory_mb
        );
        return Some(device_id);
    }
    // 4. No DXGI adapters — must rely on DML2 as last resort
    None
}

/// 当前设置下 DirectML 会落在哪台设备上（`None` = 非 Windows，或没有 DXGI 适配器）。
///
/// 【为什么需要它】诊断/基准测试要报告"实际用的是哪块卡"。旧实现把
/// `gpu_device_id` 硬写成 0，于是日志里永远显示 device 0（独显），而实际会话
/// 建在 device 1（集显）上 —— 两个数字互相矛盾，排查时只会把人带偏。
pub fn effective_dml_device_id() -> Option<i32> {
    #[cfg(target_os = "windows")]
    {
        resolve_dml_device_id()
    }
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

/// 把持久化的**稳定设备标识**（`vendor:device`）解析成**本进程**的 DXGI 序号。
///
/// 【为什么必须在这一层解析】设置里存的是 PCI 标识（见
/// `crate::dml_adapters::DmlAdapterInfo::stable_key`），而 DirectML 的
/// `with_device_id` 只认本进程的适配器序号。两者之间的换算**只能**在枚举之后做，
/// 且必须每个进程各做一次 —— 实测同一分钟内独立 App 进程与 VEGAS 插件进程的
/// 序号是互换的：`1` 在 App 里是 RTX 3050，在插件里是 Intel UHD。
///
/// 找不到（换了显卡 / 拔了 eGPU）时返回 `None` 并记一条日志，调用方回落到
/// 自动探测（显存最大的那块）。
pub fn dml_device_index_for_key(key: &str) -> Option<i32> {
    #[cfg(target_os = "windows")]
    {
        let adapters = crate::dml_adapters::enumerate_dml_adapters().adapters;
        match adapters.iter().find(|a| a.stable_key() == key) {
            Some(adapter) => {
                log::warn!(
                    "ort_session: stored DirectML adapter {key} ('{}') is device_id={} in this process",
                    adapter.name,
                    adapter.device_id
                );
                Some(adapter.device_id as i32)
            }
            None => {
                log::warn!(
                    "ort_session: stored DirectML adapter {key} is not present on this system — falling back to auto-detect"
                );
                None
            }
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = key;
        None
    }
}

/// Default EP for each role when nothing is explicitly configured.
///
/// Only HNSEP (Separator) deviates, and the reason is now **input-length
/// dependent** rather than "GPU does not help here":
///
/// HNSEP moved to a mask-only (spectrum-domain) model, so the graph is just the
/// mask network and the STFT/ISTFT run in Rust.  That made CoreML worthwhile on
/// short clips -- measured 1.29x at 2 s, 1.90x at 10 s, ~2.0-2.4x up to 45 s.
///
/// But CoreML then **degrades sharply past ~45-50 s**: at 50 s it takes 4953 ms
/// against 3732 ms on CPU (0.75x), and at 60 s 6749 ms vs 4377 ms (0.65x).
/// Reproduced across repeated runs, so it is not noise.  Since a single default
/// cannot be right for both regimes, the default stays CPU and users who want
/// the short-clip win can opt in per model with
/// `HIFISHIFTER_HNSEP_ORT_EP=coreml`.
///
/// TODO: pick the EP by input length (CPU above ~40 s), or find the root cause of
/// the CoreML long-input regression, and then flip this default.
///
/// Measurements and methodology: `docs/hifigan-gpu-acceleration.md` §4.1.
fn default_ep_for_role(role: OrtSessionRole) -> &'static str {
    match role {
        OrtSessionRole::Separator => "cpu",
        OrtSessionRole::Vocoder | OrtSessionRole::PitchDetector => "auto",
    }
}

/// Resolve the EP choice for one model, in priority order:
/// 1. Per-model env var (e.g. `HIFISHIFTER_HNSEP_ORT_EP=coreml`) — an
///    explicit, per-model request always wins, including for HNSEP.
/// 2. The process-wide runtime override (set only by the UI settings; the
///    benchmark passes its EP as an argument instead — see
///    [`build_ort_session_with_ep`]).
/// 3. The global `HIFISHIFTER_ORT_EP` env var.
/// 4. The role's default (see [`default_ep_for_role`]).
fn ep_choice_for_role(role: OrtSessionRole) -> String {
    // 1. Per-model env var — highest priority so a user can opt any single
    //    model (including HNSEP) onto the GPU even though its default is CPU.
    let role_env = match role {
        OrtSessionRole::Vocoder => "HIFISHIFTER_HIFIGAN_ORT_EP",
        OrtSessionRole::PitchDetector => "HIFISHIFTER_FCPE_ORT_EP",
        OrtSessionRole::Separator => "HIFISHIFTER_HNSEP_ORT_EP",
    };
    if let Ok(val) = std::env::var(role_env) {
        let v = val.trim().to_ascii_lowercase();
        if !v.is_empty() {
            return v;
        }
    }

    // 2. Runtime override (set via UI or benchmark)
    if let Some(ov) = RUNTIME_EP_OVERRIDE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .ok()
        .and_then(|g| g.clone())
    {
        let ov = ov.to_ascii_lowercase();
        // HNSEP stays on its default unless the request names a concrete EP,
        // because its per-clip GPU gain does not repay the compile cost.
        if !matches!(role, OrtSessionRole::Separator)
            || !matches!(ov.as_str(), "auto" | "gpu" | "directml")
        {
            return ov;
        }
    }

    // 3. Global env var, else 4. the role default.
    global_env_ep_choice().unwrap_or_else(|| default_ep_for_role(role).to_string())
}

/// The global `HIFISHIFTER_ORT_EP` env var, normalised, when it is set to a
/// non-empty value.  `None` means "no global preference configured".
fn global_env_ep_choice() -> Option<String> {
    std::env::var("HIFISHIFTER_ORT_EP")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .filter(|v| !v.is_empty())
}

/// The global env var, defaulting to `"auto"`.  Only used for diagnostics.
fn env_ep_choice() -> String {
    global_env_ep_choice().unwrap_or_else(|| "auto".to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtSessionRole {
    /// NSF-HiFiGAN vocoder — full GPU budget, aggressive optimizations.
    Vocoder,
    /// FCPE pitch analysis — smaller model, share GPU with vocoder.
    PitchDetector,
    /// HNSEP harmonic+noise separation — medium model, share GPU with vocoder.
    Separator,
}

// ── EP registration helpers ──────────────────────────────────────────────

/// Try to register WebGPU EP on a session builder.
///
/// Uses the ort crate's high-level `ep::WebGPU` type. The WebGPU EP
/// leverages Dawn (Google's WebGPU implementation) with platform-native
/// backends: Vulkan on Linux, Metal on macOS.
///
/// **Not compiled on Windows.** The Dawn/D3D12 backend in the `+wgpu`
/// ORT static binary can cause native crashes during D3D12 device
/// initialization on some GPU/driver combinations. Windows uses
/// DirectML instead, which is the mature, stable GPU path.
///
/// On WSL2, Vulkan hardware acceleration is not directly available.
/// Mesa's Lavapipe software renderer may work but with poor performance.
/// The EP will gracefully fall back to CPU if Dawn cannot find a
/// usable Vulkan device.
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
fn try_register_webgpu_ep(
    builder: ort::session::builder::SessionBuilder,
    _role: OrtSessionRole,
) -> Result<(ort::session::builder::SessionBuilder, &'static str), String> {
    // Detect WSL2 for diagnostic purposes
    let is_wsl = is_wsl2();

    // On Linux, Dawn uses the Vulkan backend. Configure it explicitly
    // and disable features that may cause issues on software renderers.
    //
    // On Windows, Dawn auto-selects D3D12. We don't try to force a
    // backend because the option string format differs between ORT
    // versions and an incorrect value can crash Dawn internally.
    //
    // IMPORTANT: WebGPU EP registration calls into Dawn native code.
    // On some GPU/driver combinations this can crash at the C level.
    // We wrap the registration in catch_unwind, but note that C-level
    // SIGSEGV cannot be caught — the best defense is to not auto-probe
    // WebGPU on Windows (which we already avoid).
    let wgpu = {
        let result = std::panic::catch_unwind(|| {
            if cfg!(target_os = "linux") {
                ort::ep::WebGPU::default()
                    .with_dawn_backend_type(ort::ep::webgpu::DawnBackendType::Vulkan)
                    .with_validation_mode(ort::ep::webgpu::ValidationMode::Disabled)
                    .with_enable_graph_capture(false)
                    .build()
            } else {
                ort::ep::WebGPU::default().build()
            }
        });
        match result {
            Ok(ep) => ep,
            Err(panic) => {
                let msg = format!(
                    "WebGPU EP build panicked: {}",
                    panic.downcast_ref::<&str>().copied().unwrap_or("unknown")
                );
                log::warn!("ort_session: {msg}");
                return Err(msg);
            }
        }
    };

    let wsl_note = if is_wsl {
        " (WSL2: Vulkan may not be available; ensure mesa-vulkan-drivers are installed)"
    } else {
        ""
    };

    // Wrap the actual EP registration in catch_unwind as well.
    // Dawn init happens inside with_execution_providers() → register().
    let register_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        builder.with_execution_providers([wgpu.clone()])
    }));

    match register_result {
        Ok(Ok(b)) => {
            log::info!("ort_session: WebGPU EP registered successfully{wsl_note}");
            Ok((b, "webgpu"))
        }
        Ok(Err(e)) => {
            let msg = format!("WebGPU EP registration failed: {e}{wsl_note}");
            log::error!("ort_session: {msg}");
            Err(msg)
        }
        Err(panic) => {
            let msg = format!(
                "WebGPU EP registration panicked (likely Dawn/D3D12 init crash): {}{wsl_note}",
                panic.downcast_ref::<&str>().copied().unwrap_or("unknown")
            );
            log::warn!("ort_session: {msg}");
            log_vulkan_diagnostics();
            Err(msg)
        }
    }
}

/// Stub: WebGPU EP not compiled on this platform (Windows, Linux ARM64, macOS x86_64).
/// On these platforms, DirectML (Windows) or CPU fallback is used instead.
#[cfg(not(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
)))]
#[allow(dead_code)] // call sites live in linux-x86_64 / macos-ARM64 branches only
fn try_register_webgpu_ep(
    builder: ort::session::builder::SessionBuilder,
    _role: OrtSessionRole,
) -> Result<(ort::session::builder::SessionBuilder, &'static str), String> {
    let _ = (builder, _role);
    Err("WebGPU EP is not compiled on this platform. Use DirectML (Windows) or CPU.".to_string())
}

/// Try to register the CoreML EP on a session builder (macOS ARM64).
///
/// CoreML is the primary GPU/Neural Engine path on Apple Silicon: it uses
/// Apple's Core ML framework (CPU + Neural Engine + GPU depending on the
/// selected compute units) instead of Dawn/WebGPU.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn try_register_coreml_ep(
    builder: ort::session::builder::SessionBuilder,
) -> Result<(ort::session::builder::SessionBuilder, &'static str), String> {
    let build_result = std::panic::catch_unwind(|| build_coreml_ep().build());
    let ep = match build_result {
        Ok(ep) => ep,
        Err(panic) => {
            let msg = format!(
                "CoreML EP build panicked: {}",
                panic.downcast_ref::<&str>().copied().unwrap_or("unknown")
            );
            log::error!("ort_session: {msg}");
            return Err(msg);
        }
    };

    let register_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        builder.with_execution_providers([ep.clone()])
    }));

    match register_result {
        Ok(Ok(b)) => {
            log::info!("ort_session: CoreML EP registered successfully");
            Ok((b, "coreml"))
        }
        Ok(Err(e)) => {
            let msg = format!("CoreML EP registration failed: {e}");
            log::error!("ort_session: {msg}");
            Err(msg)
        }
        Err(panic) => {
            let msg = format!(
                "CoreML EP registration panicked: {}",
                panic.downcast_ref::<&str>().copied().unwrap_or("unknown")
            );
            log::warn!("ort_session: {msg}");
            Err(msg)
        }
    }
}

/// Check if we're running under WSL2 (Windows Subsystem for Linux).
/// WSL2 does not expose native hardware Vulkan to Linux guests — the
/// Windows GPU driver provides D3D12/DirectX passthrough via /dev/dxg.
/// Mesa's Lavapipe software renderer may be available but offers poor
/// performance and limited SPIR-V feature support.
#[allow(dead_code)] // call sites live in linux-x86_64-only branches
fn is_wsl2() -> bool {
    if cfg!(target_os = "linux") {
        if let Ok(version) = std::fs::read_to_string("/proc/version") {
            return version.to_lowercase().contains("microsoft")
                || version.to_lowercase().contains("wsl");
        }
    }
    false
}

/// Emit diagnostic info about the Vulkan environment. Helps debug
/// WebGPU/Dawn initialization failures, especially in WSL2 where
/// Vulkan ICD availability is limited.
#[allow(dead_code)] // call sites live in linux-x86_64-only branches
fn log_vulkan_diagnostics() {
    // Check for Vulkan ICDs
    for icd_dir in ["/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"] {
        if let Ok(entries) = std::fs::read_dir(icd_dir) {
            log::warn!(
                "ort_session: vulkan ICD dir {icd_dir}: {} entries",
                entries.count()
            );
        }
    }
    // Check if /dev/dxg is present (WSL2 D3D12 passthrough)
    if std::path::Path::new("/dev/dxg").exists() {
        log::warn!("ort_session: WSL2 D3D12 passthrough (/dev/dxg) is available — WebGPU/Dawn cannot use this on Linux (Vulkan backend only)");
    }
    if is_wsl2() {
        log::warn!("ort_session: WSL2 detected — Vulkan hardware acceleration is not available from the Windows GPU driver.");
        log::warn!("ort_session: Only Mesa Lavapipe (software Vulkan) may be available, which is slow and may lack required Dawn features.");
        log::warn!("ort_session: For GPU acceleration on WSL2, use the Windows build of HiFiShifter with DirectML instead.");
    }
}

/// Try to register DirectML EP on a session builder.
///
/// DirectML uses DirectX 12 to accelerate ONNX models on any GPU
/// (NVIDIA, AMD, Intel Arc). It is Windows-only and requires no
/// additional SDK or runtime DLLs beyond the ORT provider DLL.
///
/// Registers BOTH DirectML AND CPU EP explicitly. When only DirectML
/// is registered, ORT implicitly adds CPU as a fallback but the graph
/// partitioner may not make optimal partitioning decisions. Explicit
/// registration of both EPs lets the partitioner plan the full EP
/// assignment upfront, reducing partition boundaries.
#[cfg(target_os = "windows")]
fn try_register_directml_ep(
    builder: ort::session::builder::SessionBuilder,
    role: OrtSessionRole,
    device_id: Option<i32>,
) -> Result<(ort::session::builder::SessionBuilder, &'static str), String> {
    // 设备序号由调用方解析并传进来：禁用闩锁按设备记账，二者必须是同一个值。
    let dml = if let Some(id) = device_id {
        log::warn!("ort_session[{role:?}]: DirectML device_id={id} (old API)");
        ort::ep::DirectML::default().with_device_id(id).build()
    } else {
        // Last resort: no DXGI adapters found, fall back to DML2
        log::warn!("ort_session[{role:?}]: DirectML auto-select (DML2 fallback)");
        ort::ep::DirectML::default()
            .with_performance_preference(ort::ep::directml::PerformancePreference::HighPerformance)
            .with_device_filter(ort::ep::directml::DeviceFilter::Gpu)
            .build()
    };
    builder
        .with_execution_providers([dml])
        .map(|b| (b, "directml"))
        .map_err(|e| format!("enable DirectML EP failed: {e}"))
}

/// Build an ORT session with the full optimization policy.
///
/// All three models (NSF-HiFiGAN, FCPE, HNSEP) should call this instead of
/// building sessions ad-hoc with inconsistent settings.
///
/// EP selection priority (for choice="auto"):
///   Windows:  1. DirectML (DX12, proven stable)  2. CPU fallback
///   Linux:    1. WebGPU (Dawn/Vulkan)            2. CPU fallback
///   macOS ARM64: 1. CoreML (Neural Engine/GPU)   2. WebGPU (Dawn/Metal)
///               3. CPU fallback
///
/// WebGPU on Windows is only used when explicitly selected
/// (choice="webgpu"), because Dawn/D3D12 probing can crash on some
/// GPU/driver combinations.  On Linux, Dawn/Vulkan is the primary
/// GPU path since there is no DirectML alternative.
///
/// Returns `(Session, selected_ep_name)`.
pub fn build_ort_session(
    onnx_path: &Path,
    role: OrtSessionRole,
) -> Result<(Session, String), String> {
    build_ort_session_with_ep(onnx_path, role, None)
}

/// 同 [`build_ort_session`]，但**显式指定** EP，不改任何进程级设置。
///
/// 【为什么必须有它】基准测试要分别建 CPU / 各 GPU 候选 / DirectML 三种会话。此前的
/// 做法是在每段开头改**进程级**的 `RUNTIME_EP_OVERRIDE`（旧 `EpOverrideGuard`），而那个
/// setter 会推进 `EP_SETTINGS_GENERATION` —— 于是渲染侧正在构建的会话被判为陈旧并
/// 重试，播放中可能整条重建。一次基准测试要改 6 次，撞上的概率并不小。
///
/// 把 EP 当参数传进来就没有这个问题：诊断只影响它自己建的那几个会话。
pub fn build_ort_session_with_ep(
    onnx_path: &Path,
    role: OrtSessionRole,
    ep_override: Option<&str>,
) -> Result<(Session, String), String> {
    build_session_inner(onnx_path, role, ep_override, false)
        .map(|(session, ep, _device)| (session, ep))
}

/// 诊断/基准测试用的一次性会话，附带它**实际**落在哪台 DirectML 设备上。
pub struct DiagnosticSession {
    pub session: Session,
    /// 实际生效的 EP（`"directml"` / `"cpu"` / …）。
    pub ep: String,
    /// 建在 DirectML 的哪台设备上（非 DirectML 会话为 `None`）。
    pub dml_device_id: Option<i32>,
}

/// 建一个**诊断用**会话：显式指定 EP，并**无视进程级 DirectML 禁用闩锁**。
///
/// 【为什么必须能无视闩锁】[`build_ort_session_with_ep`] 尊重"这台设备首次推理
/// 超时过"的记账 —— 渲染路径需要它，否则每次播放都要重付一次超时。但基准测试
/// 的**全部意义**就是回答"这块卡到底行不行"；如果它也被闩锁挡住，用户看到的
/// 就只有"DirectML 可用 / 失败"，而原因藏在另一个进程状态里。
///
/// 实测（VST 插件内）：加载时集显首次推理超时 → 闩锁置位 → 基准测试的 DirectML
/// 分支静默拿到一个 CPU 会话（日志 `DirectML session created in 85ms (ep=cpu)`），
/// 随后 `ep == "directml"` 为假，整段被跳过且**不留下任何错误信息**。
///
/// 诊断跑通后调用方应 [`enable_directml`] 撤销闩锁 —— 用户不必重启应用。
pub fn build_diagnostic_session(
    onnx_path: &Path,
    role: OrtSessionRole,
    ep: &str,
) -> Result<DiagnosticSession, String> {
    let (session, ep, dml_device_id) = build_session_inner(onnx_path, role, Some(ep), true)?;
    Ok(DiagnosticSession {
        session,
        ep,
        dml_device_id,
    })
}

/// 会话构建的实际实现。
///
/// `bypass_disable_latch` 只由 [`build_diagnostic_session`] 置真：渲染路径必须
/// 尊重"这台设备挂起过"的记账，诊断路径必须能重新验证它。
fn build_session_inner(
    onnx_path: &Path,
    role: OrtSessionRole,
    ep_override: Option<&str>,
    bypass_disable_latch: bool,
) -> Result<(Session, String, Option<i32>), String> {
    let choice = match ep_override {
        Some(ep) => ep.trim().to_ascii_lowercase(),
        None => ep_choice_for_role(role),
    };

    // Only an explicit "cpu" choice short-circuits here.  HNSEP resolves to
    // "cpu" by default via `ep_choice_for_role`, but a per-model override
    // (HIFISHIFTER_HNSEP_ORT_EP=coreml) must be able to reach the GPU path.
    if choice == "cpu" {
        return build_cpu_session(onnx_path, role, &choice).map(|(s, ep)| (s, ep, None));
    }

    // ── Windows: DirectML first (proven stable, no crash risk) ─────────
    #[cfg(target_os = "windows")]
    if choice == "auto" || choice == "directml" || choice == "gpu" {
        // 设备只解析一次：闩锁按设备记账，构建与记账必须是同一个值。
        let device = resolve_dml_device_id();
        if !bypass_disable_latch && directml_disabled(device) {
            let device_desc = describe_dml_device(device);
            log::warn!(
                "ort_session[{role:?}]: DirectML is disabled for device {device_desc} in this process (an earlier first inference timed out) — using CPU. Run the benchmark to re-test that device."
            );
            return build_cpu_session(onnx_path, role, &choice).map(|(s, ep)| (s, ep, None));
        }
        // DirectML with strict mode first, then with fallback
        match build_dml_session_inner(onnx_path, role, &choice, true, device) {
            Ok((session, ep)) => return Ok((session, ep, device)),
            Err(e) => log::error!(
                "ort_session[{role:?}]: strict DirectML failed (will retry with CPU fallback): {e}"
            ),
        }
        // 烟测超时会把该设备标记为不可用（见 disable_directml）——
        // 此时第二次尝试无需再建会话，直接走 CPU，避免第二次超时冻结。
        if !bypass_disable_latch && directml_disabled(device) {
            let device_desc = describe_dml_device(device);
            log::warn!(
                "ort_session[{role:?}]: DirectML disabled for device {device_desc} after strict-attempt timeout — using CPU"
            );
            return build_cpu_session(onnx_path, role, &choice).map(|(s, ep)| (s, ep, None));
        }
        match build_dml_session_inner(onnx_path, role, &choice, false, device) {
            Ok((session, ep)) => return Ok((session, ep, device)),
            Err(e) => log::error!("ort_session[{role:?}]: DirectML with fallback failed: {e}"),
        }
    }

    // ── Windows: WebGPU not compiled (uses DirectML instead) ──────────
    #[cfg(target_os = "windows")]
    if choice == "webgpu" {
        log::warn!("ort_session[{role:?}]: WebGPU is not available on Windows. Use DirectML ('auto' or 'directml' or 'gpu') for GPU acceleration, or 'cpu' for CPU-only.");
        // Fall through to CPU below.
    }

    // ── Linux x86_64: WebGPU first, then CPU ───────────────────────────
    // NOTE: Linux ARM64 does NOT have the webgpu feature (no prebuilt
    // ORT binary for aarch64+wgpu), so this block is excluded there.
    // NOTE: WSL2 is skipped because Dawn/Vulkan init hangs at shutdown
    // even when WebGPU sessions are never used for inference.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    if (choice == "auto" || choice == "webgpu" || choice == "gpu") && !is_wsl2() {
        if webgpu_disabled() {
            log::warn!(
                "ort_session[{role:?}]: WebGPU is disabled in this process (an earlier inference timed out) — using CPU. Run the benchmark to re-test."
            );
        } else {
            match Session::builder() {
                Ok(builder) => match try_register_webgpu_ep(builder, role) {
                    Ok((b, ep)) => {
                        let session = build_gpu_session_finalize(b, onnx_path, role, "WebGPU")?;
                        return Ok((session, ep.to_string(), None));
                    }
                    Err(e) => {
                        log::warn!("ort_session[{role:?}]: WebGPU unavailable — {e}");
                        log_vulkan_diagnostics();
                    }
                },
                Err(e) => {
                    log::error!(
                        "ort_session[{role:?}]: failed to create session builder for WebGPU — {e}"
                    );
                    log_vulkan_diagnostics();
                }
            }
        }
    }

    // ── Fallback: CPU ──────────────────────────────────────────────────
    // macOS ARM64: CoreML (Neural Engine/GPU) first, WebGPU fallback, then CPU.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if choice == "auto"
        || choice == "coreml"
        || choice == "webgpu"
        || choice == "gpu"
        || choice == "directml"
    {
        // CoreML is the primary GPU path on Apple Silicon. An explicit
        // "webgpu" selection skips CoreML and goes straight to Dawn/Metal.
        //
        // The session deliberately keeps the model's dynamic `batch`/`time`
        // axes: ONNX Runtime 1.28's CoreML EP compiles the dynamic graph
        // without help, and pinning `time` to a constant benchmarked ~1.7x
        // SLOWER than the CPU EP (see the module docs).
        if choice != "webgpu" && !coreml_disabled() {
            match Session::builder() {
                Ok(builder) => {
                    match try_register_coreml_ep(builder) {
                        Ok((b, ep)) => {
                            match build_gpu_session_finalize(b, onnx_path, role, "CoreML") {
                            Ok(session) => return Ok((session, ep.to_string(), None)),
                            Err(e) => log::error!(
                                "ort_session[{role:?}]: CoreML session creation failed (will try WebGPU): {e}"
                            ),
                        }
                        }
                        Err(e) => log::error!("ort_session[{role:?}]: CoreML unavailable: {e}"),
                    }
                }
                Err(e) => log::error!(
                    "ort_session[{role:?}]: failed to create session builder for CoreML: {e}"
                ),
            }
        }
        if !webgpu_disabled() {
            match Session::builder() {
                Ok(builder) => match try_register_webgpu_ep(builder, role) {
                    Ok((b, ep)) => match build_gpu_session_finalize(b, onnx_path, role, "WebGPU") {
                        Ok(session) => return Ok((session, ep.to_string(), None)),
                        Err(e) => log::error!(
                            "ort_session[{role:?}]: WebGPU session creation failed (will try CPU): {e}"
                        ),
                    },
                    Err(e) => log::error!("ort_session[{role:?}]: WebGPU unavailable: {e}"),
                },
                Err(e) => {
                    log::error!(
                        "ort_session[{role:?}]: failed to create session builder for WebGPU: {e}"
                    )
                }
            }
        } else {
            log::warn!(
                "ort_session[{role:?}]: WebGPU is disabled in this process (an earlier inference timed out) — using CPU. Run the benchmark to re-test."
            );
        }
    }

    build_cpu_session(onnx_path, role, &choice).map(|(session, ep)| (session, ep, None))
}

/// Run a minimal inference through a newly-created GPU session to
/// verify that the EP can actually execute compute shaders.  On some
/// platforms (WSL2 with Lavapipe, misconfigured drivers, headless
/// systems) the EP registers successfully but runtime inference fails
/// — this catches that early so we can fall back to CPU.
fn smoke_test_gpu_session(
    mut session: Session,
    role: OrtSessionRole,
    ep_name: &str,
    // DirectML 设备序号：超时时要按设备记账（`None` = 非 DirectML 会话）。
    dml_device: Option<i32>,
) -> Result<Session, String> {
    use ort::value::{Tensor, ValueType};

    // 分步日志：烟测若卡住（历史上曾永久持有全局构建锁 → 渲染永久卡 0%），
    // 这些标记能直接指出卡在哪一步。
    let smoke_started = std::time::Instant::now();
    log::warn!("ort_session[{role:?}]: smoke stage=begin ep={ep_name}");

    // Collect f32 tensor input metadata first so the `session` borrow ends
    // before the session is moved into the helper thread below.
    let mut plans: Vec<(String, Vec<usize>)> = Vec::new();
    for input in session.inputs() {
        let (tensor_ty, shape) = match input.dtype() {
            ValueType::Tensor { ty, shape, .. } => (ty, shape),
            _ => continue,
        };
        if *tensor_ty != ort::value::TensorElementType::Float32 {
            continue;
        }
        if shape.contains(&0) {
            continue; // scalar or zero-dim - skip
        }
        // Replace dynamic dimensions (-1) with realistic test values:
        //   dim 0 -> 1 (batch),  other dims -> the role's probe length.
        // Tiny values (e.g. 4) make ORT's buffer-reuse optimizer collide
        // with the model's fixed intermediate shapes ("{1,4,1} !=
        // {1,2048,1}"), so use a length the model accepts;
        // see `smoke_probe_frames` for why it is role-dependent.
        let fallback_dim = smoke_probe_frames(role) as i64;
        let test_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if d > 0 {
                    d as usize
                } else if i == 0 {
                    1
                } else {
                    fallback_dim as usize
                }
            })
            .collect();
        plans.push((input.name().to_string(), test_shape));
    }
    log::warn!(
        "ort_session[{role:?}]: smoke stage=inputs_planned count={} elapsed_ms={}",
        plans.len(),
        smoke_started.elapsed().as_millis()
    );
    let mut input_pairs: Vec<(String, ort::value::Value)> = Vec::with_capacity(plans.len());
    for (name, test_shape) in plans {
        let total: usize = test_shape.iter().product::<usize>().max(1);
        // Use a non-zero f0: this model's f0 pre-processing computes a
        // differential that becomes the Pad "pads" input, and an all-zero f0
        // produces an empty/zero-sized tensor that crashes ORT's buffer
        // reuse ("{1,0,112}", "{1,4096,1} vs {1,4096,4096}").  440 Hz is a
        // realistic mid-range pitch and keeps every intermediate shape valid.
        let fill = if name == "f0" { 440.0f32 } else { 0.0f32 };
        let data: Vec<f32> = vec![fill; total];
        let tensor = Tensor::from_array((test_shape, data.into_boxed_slice()))
            .map_err(|e| format!("smoke test: tensor '{name}' creation failed: {e}"))?;
        input_pairs.push((name, tensor.into()));
    }

    if input_pairs.is_empty() {
        log::warn!("ort_session[{role:?}]: GPU smoke test skipped (no f32 tensor inputs)");
        return Ok(session);
    }

    // CoreML can take a long time (or hang forever) on its first inference:
    // MLProgram compilation is deferred to the first run, and a dynamic
    // batch is a known hang source.  Run the smoke test on a helper thread
    // with a generous timeout so a stuck CoreML session can never freeze the
    // benchmark.  If it times out we return an error and the caller falls
    // back to WebGPU/CPU; the orphaned thread (if truly hung) is harmless.
    //
    // DirectML 同理：设备切换后新建的 DML 会话偶发首次推理挂起。超时后
    // 除本次回退外，还会把 DirectML 标记为本进程不可用（见 disable_directml），
    // 使后续构建（strict 回退尝试、其他模型、后续切换）立即走 CPU。
    log::warn!(
        "ort_session[{role:?}]: smoke stage=tensors_ready elapsed_ms={} — spawning inference thread",
        smoke_started.elapsed().as_millis()
    );
    let (tx, rx) = std::sync::mpsc::channel();
    let timeout = smoke_test_timeout(ep_name);
    // ★ 绝不 join 此线程（历史致命缺陷，已定位到确切位置）：
    // 线程把结果与会话经 channel 送出后即可结束，调用方无需 join 也能拿到
    // 全部数据。而实测 ORT/DirectML 推理线程的**退出清理**（TLS/线程本地
    // 分配器、D3D12 每线程状态）会偶发挂起 —— 一旦 join，构建线程就永久
    // 阻塞在退出清理上，进而永久持有全局会话构建锁：其他模块的构建、渲染
    // pass 的全部 Clip、乃至关机清理全部卡死（日志表现为渲染进度永久 0%，
    // 且烟测超时/禁用日志都不出现，因为线程根本没走到超时分支）。
    // 线程句柄在此丢弃（分离），其退出由 OS 处理；即便退出清理挂起，泄漏的
    // 也只是一个已无用的线程（会话已移交调用方）。
    let _run_thread = std::thread::spawn(move || {
        // Run inside a block so the returned SessionOutputs (which borrow
        // the session) are dropped before we move the session back.
        let result = {
            let outputs = session.run(input_pairs);
            outputs.map(|_| ())
        };
        let _ = tx.send((result, session));
    });

    match rx.recv_timeout(timeout) {
        Ok((result, session)) => {
            log::warn!(
                "ort_session[{role:?}]: smoke stage=inference_returned elapsed_ms={}",
                smoke_started.elapsed().as_millis()
            );
            match result {
                Ok(_) => {
                    // 成功用例：info 而非 error —— EP 功能正常是健康状态，
                    // error 级别会误导用户/CI 把正常运行当作失败。
                    log::info!("ort_session[{role:?}]: {ep_name} smoke test passed - EP is functional");
                    Ok(session)
                }
                Err(e) => Err(format!(
                    "{ep_name} inference is not functional on this system.                      The EP registered but compute shader execution failed: {e}.                      Falling back to CPU."
                )),
            }
        }
        Err(_) => {
            log::error!(
                "ort_session[{role:?}]: smoke stage=timeout elapsed_ms={} (inference did not return within {timeout:?})",
                smoke_started.elapsed().as_millis()
            );
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            disable_coreml(&format!(
                "{ep_name} smoke test exceeded {timeout:?} (first inference hung)"
            ));
            #[cfg(target_os = "windows")]
            if ep_name == "DirectML" {
                // 只禁用**这台设备**：集显超时不该把独显一起废掉。
                disable_directml(
                    dml_device,
                    &format!("DirectML smoke test exceeded {timeout:?} (first inference hung)"),
                );
            }
            Err(format!(
                "{ep_name} smoke test timed out after {timeout:?}.                  The EP registered but the first inference did not complete.                  Falling back to WebGPU/CPU."
            ))
        }
    }
}

/// Finalize a GPU session (CoreML / WebGPU) with appropriate optimization
/// settings, then smoke-test it so broken GPU backends fall back to CPU early.
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
fn build_gpu_session_finalize(
    mut builder: ort::session::builder::SessionBuilder,
    onnx_path: &Path,
    role: OrtSessionRole,
    ep_name: &str,
) -> Result<Session, String> {
    let model_name = onnx_path
        .file_name()
        .map(|n| n.to_string_lossy())
        .unwrap_or_default();
    log::warn!(
        "ort_session[{role:?}]: model={model_name} ep={ep_name} (global_env={})",
        env_ep_choice(),
    );

    let coreml_session = ep_name == "CoreML";
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("set graph optimization level failed: {e}"))?
        // CoreML MLProgram sessions have their own execution queues.  ORT's
        // CPU memory-pattern/parallel-execution optimizers have been observed
        // to leave CoreML sessions stuck on repeated runs, so disable both for
        // CoreML and let Apple's runtime manage its buffers.
        .with_memory_pattern(!coreml_session)
        .map_err(|e| format!("set memory pattern failed: {e}"))?
        .with_parallel_execution(!coreml_session)
        .map_err(|e| format!("set parallel execution failed: {e}"))?;

    // CoreML keeps ORT's default (`0`): the reference benchmark of the
    // dynamic-shape CoreML session was taken with this setting and it is the
    // configuration the ~59x rtf was measured on.
    let threads = if coreml_session {
        0
    } else {
        cpu_intra_threads(role)
    };
    builder = builder
        .with_intra_threads(threads)
        .map_err(|e| format!("set intra op threads failed: {e}"))?;

    let t_create = std::time::Instant::now();
    let mut session = builder.commit_from_file(onnx_path).map_err(|e| {
        let msg = format!("load onnx into {ep_name} ort session failed: {e}");
        log::warn!("ort_session[{role:?}]: {msg}");
        msg
    })?;
    let create_ms = t_create.elapsed().as_millis();

    log::warn!(
        "ort_session[{role:?}]: created session ep={ep_name} intra_threads={threads} commit_ms={create_ms}",
    );

    // Log session I/O metadata
    for input in session.inputs() {
        log::warn!(
            "ort_session[{:?}]:   input name='{}' dtype={:?}",
            role,
            input.name(),
            input.dtype()
        );
    }
    for output in session.outputs() {
        log::warn!(
            "ort_session[{:?}]:   output name='{}' dtype={:?}",
            role,
            output.name(),
            output.dtype()
        );
    }

    // ── Smoke test: verify WebGPU can actually run inference ──────────
    // On some platforms (WSL2 with Lavapipe, headless systems, etc.)
    // the WebGPU EP registers successfully (Dawn finds a Vulkan device)
    // but compute shader execution fails at runtime.  Running a tiny
    // inference now catches this early and lets us fall back to CPU
    // instead of silently returning a broken session.
    match smoke_test_gpu_session(session, role, ep_name, None) {
        Ok(s) => session = s,
        Err(e) => {
            log::error!("ort_session[{role:?}]: {ep_name} smoke test failed, discarding session and falling back to CPU: {e}");
            return Err(e);
        }
    }

    Ok(session)
}

/// Build a DirectML session with optional strict mode (no CPU fallback).
#[cfg(target_os = "windows")]
fn build_dml_session_inner(
    onnx_path: &Path,
    role: OrtSessionRole,
    choice: &str,
    strict: bool,
    device: Option<i32>,
) -> Result<(Session, String), String> {
    // ★ 创建（含 builder 配置）在辅助线程内完成并施加超时。
    // ORT 的 `commit_from_file` 没有超时 API，且 `SessionBuilder` 不是 Send
    // （内部持有裸指针，无法在主线程配置后移交）—— 因此配置必须与创建同在
    // 辅助线程内。实测设备切换后重建 DML 会话时 commit 会挂起；没有超时兜底
    // 就会永久持有全局单飞锁，渲染进度永久卡在 0%。
    let timeout = creation_timeout("DirectML");
    let path = onnx_path.to_path_buf();
    let choice_owned = choice.to_string();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("ort-dml-commit".to_string())
        .spawn(move || {
            let _ = tx.send(create_dml_session(
                &path,
                role,
                &choice_owned,
                strict,
                device,
            ));
        })
        .map_err(|e| format!("spawn DirectML session creation thread failed: {e}"))?;

    let (mut session, selected, threads, create_ms) = match rx.recv_timeout(timeout) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return Err(e),
        Err(_) => {
            // 创建超时（strict 与回退尝试一视同仁）：strict 在本模型上本就
            // 预期"快速失败"（图节点不支持 DML），超时意味着驱动层挂起而非
            // 能力问题 —— 直接禁用 DirectML，使回退立即以 CPU 完成，把最坏
            // 情况锁定为一次性 ~10s。
            disable_directml(
                device,
                &format!("DirectML session creation exceeded {timeout:?} (commit hung)"),
            );
            return Err(format!(
                "DirectML session creation timed out after {timeout:?} (ORT commit hung); falling back"
            ));
        }
    };

    // ── Detailed diagnostic logging ────────────────────────────────────
    log::warn!(
        "ort_session[{role:?}]: created session ep={selected} strict={strict} intra_threads={threads} commit_ms={create_ms}",
    );
    // Log session I/O metadata (names, shapes, types)
    for input in session.inputs() {
        log::warn!(
            "ort_session[{:?}]:   input name='{}' dtype={:?}",
            role,
            input.name(),
            input.dtype()
        );
    }
    for output in session.outputs() {
        log::warn!(
            "ort_session[{:?}]:   output name='{}' dtype={:?}",
            role,
            output.name(),
            output.dtype()
        );
    }

    match smoke_test_gpu_session(session, role, "DirectML", device) {
        Ok(s) => session = s,
        Err(e) => {
            log::error!("ort_session[{role:?}]: DirectML smoke test failed, discarding session and falling back to CPU: {e}");
            return Err(e);
        }
    }

    Ok((session, selected.to_string()))
}

/// 在（辅助）线程内配置并创建 DirectML 会话。
///
/// `SessionBuilder` 不是 Send，无法跨线程移交，因此配置必须与创建在同一
/// 线程内完成；本函数由 [`build_dml_session_inner`] 的辅助线程调用，使其
/// 可以施加创建超时。
#[cfg(target_os = "windows")]
fn create_dml_session(
    onnx_path: &Path,
    role: OrtSessionRole,
    choice: &str,
    strict: bool,
    device: Option<i32>,
) -> Result<(Session, &'static str, usize, u128), String> {
    let builder =
        Session::builder().map_err(|e| format!("create ort session builder failed: {e}"))?;

    let (builder, selected) = try_register_directml_ep(builder, role, device)?;

    let device_desc = describe_dml_device(device);
    log::warn!(
        "ort_session[{role:?}]: model={} ep={selected} device={device_desc} strict={strict} (choice={choice}, global_env={})",
        onnx_path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(),
        env_ep_choice(),
    );

    // ── DirectML-specific config ────────────────────────────────────────
    let builder = builder
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(|e| format!("set graph optimization level failed: {e}"))?
        .with_memory_pattern(false)
        .map_err(|e| format!("set memory pattern failed: {e}"))?
        .with_device_allocated_initializers()
        .map_err(|e| format!("enable device allocated initializers failed: {e}"))?
        .with_flush_to_zero()
        .map_err(|e| format!("enable flush-to-zero failed: {e}"))?
        .with_prepacking(false)
        .map_err(|e| format!("disable prepacking failed: {e}"))?
        // Override dynamic dimensions to fixed values. DirectML performs
        // best when shapes are known at session creation time because it
        // can pre-compile shaders and optimize GPU memory layouts. Dynamic
        // dimensions force DML to use less-optimized generic kernels.
        //
        // 【只对声码器生效】这两个覆盖依赖模型**导出的维度名**：
        //   - `with_dimension_override("batch", 1)` 按维度**名**匹配
        //   - `with_dimension_override_by_denotation("time", 4096)` 按 denotation 匹配
        // NSF-HiFiGAN / FCPE 用 `batch` / `time`，而 HNSEP 的 mask-only 模型用
        // `batch_size` / `n_frames`（已实测确认）—— 对它两者都是 **no-op**：
        // 不报错，但也固定不住任何维度，DML 仍走通用 kernel。
        //
        // 这不会造成失败，但也不能留着装作"已优化"：更危险的是**万一将来**
        // HNSEP 的维度被改名成 `time`，4096 会被当成谱帧数套到一个
        // `[batch, 2, 1025, n_frames]` 的 4-D 输入上 —— 那会把隐式
        // `time` 维（若存在）固定成 4096，与 HNSEP 要求的 `n_frames % 32 == 0`
        // 及真实帧数都无关，属于静默的语义错配。
        // 因此显式按角色门控：只有声码器/音高检测固定 `time`。
        .with_dimension_override("batch", 1)
        .map_err(|e| format!("override batch dim failed: {e}"))?;

    // `time` 维的固定单独按角色施加（`builder` 在此处仍是不可变绑定，
    // 需要一次可变重绑定才能追加这个条件性的构建器步骤）。
    let mut builder = builder;
    if matches!(
        role,
        OrtSessionRole::Vocoder | OrtSessionRole::PitchDetector
    ) {
        builder = builder
            .with_dimension_override_by_denotation("time", 4096)
            .map_err(|e| format!("override time dim failed: {e}"))?;
    }

    if strict {
        // Disable CPU fallback: if ANY op can't run on DirectML, session
        // creation FAILS. If it succeeds, the ENTIRE graph runs on GPU
        // with ZERO partition boundaries → no GPU↔CPU copies → maximum
        // throughput. This is the key to unlocking Pascal (GTX 10xx)
        // performance where ORT's partitioner otherwise sends too many
        // ops to CPU.
        builder = builder
            .with_disable_cpu_fallback()
            .map_err(|e| format!("disable cpu fallback failed: {e}"))?;
    } else {
        builder = builder
            .with_parallel_execution(true)
            .map_err(|e| format!("enable parallel execution failed: {e}"))?
            .with_inter_threads(2)
            .map_err(|e| format!("set inter threads failed: {e}"))?;
    }

    // ── Thread config ───────────────────────────────────────────────────
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(2);
    let threads = if strict {
        // Strict mode: ALL ops on GPU. CPU threads are irrelevant —
        // keep 2 to avoid overhead.
        2
    } else {
        // Fallback mode: CPU handles unsupported ops.
        // Max threads for fastest CPU fallback throughput.
        cores.max(4)
    };

    builder = builder
        .with_intra_threads(threads)
        .map_err(|e| format!("set intra op threads failed: {e}"))?;

    let t_create = std::time::Instant::now();
    let session = builder
        .commit_from_file(onnx_path)
        .map_err(|e| format!("load onnx into ort session failed: {e}"))?;
    let create_ms = t_create.elapsed().as_millis();

    Ok((session, selected, threads, create_ms))
}

/// 诊断期间的 CPU 线程预算上限（0 = 不限制）。
///
/// 【为什么需要】基准测试与预热会跑真实推理。用渲染级预算（`cores/2`，多核机器上
/// 是 8~16 条）会把实时音频线程挤到没有时间片，用户听到断续与爆音。
///
/// 【为什么是进程级静态而不是参数】线程预算在四个构建器的最深处取值
/// （`cpu_intra_threads`），把它一路穿下去要改四层签名；而它**只在一个很窄的时间窗**
/// 内有值（`DiagnosticBudget` 的存活期）。诊断期间不会有新的**渲染**会话被构建 ——
/// 渲染会话在预热 / 首次使用时就已经建好并复用，因此这个窗口是安全的。
static DIAGNOSTIC_THREAD_CAP: AtomicUsize = AtomicUsize::new(0);

/// 并发预算作用域的深度：`enter` 加一、`Drop` 减一。
static DIAGNOSTIC_BUDGET_DEPTH: AtomicUsize = AtomicUsize::new(0);

/// 诊断期间的线程预算租约：进入时把上限压到 `threads`，**最后一个离开者**清零。
///
/// 【为什么不是"离开时恢复旧值"】启动预热是三个模型各自 spawn 的线程并发进入
/// （fcpe / hnsep / nsf），租约在拿到会话构建租约**之前**就开始 —— 两个作用域
/// 必然重叠。旧实现记 `previous` 并在 `Drop` 无条件恢复：A.enter(prev=0)、
/// B.enter(prev=2)、A 先 Drop 恢复 0、B 后 Drop 恢复 **2** —— 进程级上限从此
/// 恒为 2，此后任何会话重建（设备切换、EP 变更、首次回退）得到的 CPU intra
/// 线程被钳到 2，多核机上渲染推理慢 3~4 倍，直到重启（第 5 轮审查 P2-1）。
/// 计数 + "最后一个离开者关灯"让重叠在结构上安全。
pub struct DiagnosticBudget {
    _private: (),
}

impl DiagnosticBudget {
    /// 进入诊断预算。建议 `threads = 2`。重叠作用域时后进者的值生效
    /// （当前全部调用点都是 2，无实际分歧）。
    pub fn enter(threads: usize) -> Self {
        DIAGNOSTIC_BUDGET_DEPTH.fetch_add(1, Ordering::AcqRel);
        DIAGNOSTIC_THREAD_CAP.store(threads.max(1), Ordering::Release);
        Self { _private: () }
    }
}

impl Drop for DiagnosticBudget {
    fn drop(&mut self) {
        if DIAGNOSTIC_BUDGET_DEPTH.fetch_sub(1, Ordering::AcqRel) == 1 {
            DIAGNOSTIC_THREAD_CAP.store(0, Ordering::Release);
        }
    }
}

/// 把诊断上限套到基础预算上（纯函数，便于单测）。
fn apply_diagnostic_cap(base: usize, cap: usize) -> usize {
    if cap == 0 {
        base
    } else {
        base.min(cap).max(1)
    }
}

/// Intra-op thread budget for sessions whose ops run on the CPU (the whole
/// graph for a CPU session, or the unsupported ops of a GPU session).
///
/// Apple Silicon uses every core: measured on an 8-core M-series with 1024 mel
/// frames, `intra=cores/2` took 1586 ms vs `intra=cores` at 1202 ms (−24%).
/// Apple's performance and efficiency cores are homogeneous enough that using
/// all of them is a straight win.
///
/// Windows and Linux keep `cores/2`: Intel hybrid CPUs mix P-cores with much
/// slower E-cores and many workstations are multi-socket/NUMA, so pinning
/// intra-op work to every logical core can oversubscribe the slow cores and
/// regress. The conservative half-core budget stays there until it is measured
/// on those topologies.
fn cpu_intra_threads(role: OrtSessionRole) -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(2);
    let base = match role {
        OrtSessionRole::Separator => cores,
        OrtSessionRole::Vocoder | OrtSessionRole::PitchDetector => {
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                cores
            } else {
                (cores / 2).max(2)
            }
        }
    };
    apply_diagnostic_cap(base, DIAGNOSTIC_THREAD_CAP.load(Ordering::Acquire))
}

/// 整段HNSEP尺寸随素材长度变化，CPU arena/pattern会留住整段巨型中间块。
/// 其它模型保持原复用策略；Intel macOS的ort-tract不改原生ORT专用配置。
fn cpu_memory_reuse(role: OrtSessionRole) -> bool {
    !matches!(role, OrtSessionRole::Separator)
        || cfg!(all(target_os = "macos", target_arch = "x86_64"))
}

/// Build a pure CPU session (used for "cpu" choice or fallback).
fn build_cpu_session(
    onnx_path: &Path,
    role: OrtSessionRole,
    choice: &str,
) -> Result<(Session, String), String> {
    build_cpu_session_with_memory_reuse(onnx_path, role, choice, cpu_memory_reuse(role))
}

/// 固定同一图优化/线程/模型，分离分配策略以便短输入oracle对照；生产入口按角色选择。
fn build_cpu_session_with_memory_reuse(
    onnx_path: &Path,
    role: OrtSessionRole,
    choice: &str,
    memory_reuse: bool,
) -> Result<(Session, String), String> {
    let mut builder =
        Session::builder().map_err(|e| format!("create ort session builder failed: {e}"))?;

    log::warn!(
        "ort_session[{role:?}]: model={} ep=cpu (choice={choice}, global_env={})",
        onnx_path
            .file_name()
            .map(|n| n.to_string_lossy())
            .unwrap_or_default(),
        env_ep_choice(),
    );

    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("set graph optimization level failed: {e}"))?
        .with_memory_pattern(memory_reuse)
        .map_err(|e| format!("set memory pattern failed: {e}"))?;

    if !memory_reuse {
        // 核对ort-rc.13 ep/cpu.rs：显式注册CPU(false)才会调用DisableCpuMemArena。
        // 不注册EP的默认CPU仍开arena；只关memory_pattern不能释放该大缓存。
        builder = builder
            .with_execution_providers([ort::ep::CPU::default().with_arena_allocator(false).build()])
            .map_err(|e| format!("disable separator CPU arena failed: {e}"))?;
    }
    if matches!(role, OrtSessionRole::Separator) {
        // 仅显式诊断启用；常规App/插件不生成ORT算子日志，路径由隔离探针提供。
        if let Some(prefix) = std::env::var_os("HIFISHIFTER_HNSEP_ORT_PROFILE_PREFIX") {
            builder = builder
                .with_profiling(std::path::PathBuf::from(prefix))
                .map_err(|error| format!("enable separator operator profile failed: {error}"))?;
        }
    }

    let threads = cpu_intra_threads(role);
    builder = builder
        .with_intra_threads(threads)
        .map_err(|e| format!("set intra op threads failed: {e}"))?;

    let t_create = std::time::Instant::now();
    let session = builder
        .commit_from_file(onnx_path)
        .map_err(|e| format!("load onnx into ort session failed: {e}"))?;
    let create_ms = t_create.elapsed().as_millis();

    log::warn!(
        "ort_session[{role:?}]: created session ep=cpu intra_threads={threads} commit_ms={create_ms}",
    );

    Ok((session, "cpu".to_string()))
}

// ─── GPU Diagnostic & Provider Enumeration ────────────────────────────────

/// Diagnostic info about GPU setup for user-facing reporting.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuDiagnostic {
    /// List of all ONNX Runtime execution provider names available in the DLL.
    pub available_providers: Vec<String>,
    /// The EP that was actually selected (e.g. "webgpu", "directml", "cpu").
    pub selected_ep: String,
    /// GPU device ID that was requested (from env or default 0).
    pub gpu_device_id: i32,
    /// The ONNX Runtime build info string.
    pub ort_build_info: String,
}

/// Enumerate available ONNX Runtime execution providers.
///
/// Checks each provider by attempting to query its availability through ORT.
///
/// NOTE: WebGPU probing is performed on Linux x86_64 and macOS ARM64,
/// where Dawn uses the Vulkan/Metal backend which is safe to probe.
/// On Windows, Dawn uses D3D12 and probing can trigger native crashes on
/// some GPU/driver combos; WebGPU on Windows is only used when the user
/// explicitly selects it.
pub fn diagnose_available_providers() -> Vec<String> {
    let mut providers = vec!["CPUExecutionProvider".to_string()];

    // WebGPU — compiled on Linux x86_64 / macOS ARM64 only.
    // Excluded: Windows (Dawn/D3D12 crash risk), Linux ARM64 (no prebuilt binary),
    //           WSL2 (Vulkan not available; Dawn init hangs at shutdown).
    // 【为什么要分三种说法】此前 WSL 上这里只是**不列** WebGPU，日志于是只剩
    // "CPUExecutionProvider" —— 用户无法区分「没编进去」「编了但被跳过」「编了但探测失败」，
    // 而它们的下一步完全不同（换构建 / 换机器 / 查驱动）。
    //
    // 只在日志里分档，**不改** `available_providers` 的内容：那是给 UI 与诊断包读的
    // 名单，塞进带括号的说明会让每个消费方都要解析一遍。
    #[cfg(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64")
    ))]
    {
        if cfg!(target_os = "linux") && is_wsl2() {
            log::warn!(
                "[ort] webgpu: skipped(wsl2) — dawn needs a real vulkan device, \
                 WSL2 only exposes the d3d12 passthrough"
            );
        } else if probe_webgpu_ep_available() {
            log::warn!("[ort] webgpu: available");
            providers.push("WebGpuExecutionProvider".to_string());
        } else {
            log::warn!("[ort] webgpu: probe-failed(dawn could not register this ep)");
        }
    }

    // CoreML -- macOS ARM64 only (Apple Neural Engine / GPU).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if probe_coreml_ep_available() {
        providers.push("CoreMLExecutionProvider".to_string());
    }

    // DirectML — Windows only
    if probe_directml_ep_available() {
        providers.push("DmlExecutionProvider".to_string());
    }

    providers
}

/// Quick check: try registering WebGPU EP on a temporary session builder.
/// Returns true if WebGPU EP is available in the loaded ORT binary.
///
/// Wrapped in catch_unwind because Dawn native code (Vulkan init)
/// can crash on some platforms. Only compiled on Linux/macOS ARM
/// where Dawn/Vulkan and Dawn/Metal are stable backends.
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
fn probe_webgpu_ep_available() -> bool {
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match Session::builder() {
            Ok(builder) => {
                let wgpu = if cfg!(target_os = "linux") {
                    ort::ep::WebGPU::default()
                        .with_dawn_backend_type(ort::ep::webgpu::DawnBackendType::Vulkan)
                        .build()
                } else {
                    ort::ep::WebGPU::default().build()
                };
                match builder.with_execution_providers([wgpu]) {
                    Ok(_) => {
                        log::warn!("ort_session: probe_webgpu_ep — AVAILABLE");
                        true
                    }
                    Err(e) => {
                        log::warn!("ort_session: probe_webgpu_ep — NOT available: {e}");
                        false
                    }
                }
            }
            Err(e) => {
                log::error!("ort_session: probe_webgpu_ep — session builder failed: {e}");
                false
            }
        }));

    match result {
        Ok(available) => available,
        Err(panic) => {
            let msg = panic.downcast_ref::<&str>().copied().unwrap_or("unknown");
            log::error!("ort_session: probe_webgpu_ep — PANICKED: {msg}");
            log_vulkan_diagnostics();
            false
        }
    }
}

/// Quick check: try registering the CoreML EP on a temporary session builder.
/// Returns true if CoreML EP is available in the loaded ORT binary.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn probe_coreml_ep_available() -> bool {
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match Session::builder() {
            Ok(builder) => {
                let ep = build_coreml_ep().build();
                match builder.with_execution_providers([ep]) {
                    Ok(_) => {
                        log::warn!("ort_session: probe_coreml_ep - AVAILABLE");
                        true
                    }
                    Err(e) => {
                        log::warn!("ort_session: probe_coreml_ep - NOT available: {e}");
                        false
                    }
                }
            }
            Err(e) => {
                log::error!("ort_session: probe_coreml_ep - session builder failed: {e}");
                false
            }
        }));

    match result {
        Ok(available) => available,
        Err(panic) => {
            let msg = panic.downcast_ref::<&str>().copied().unwrap_or("unknown");
            log::error!("ort_session: probe_coreml_ep - PANICKED: {msg}");
            false
        }
    }
}

/// Quick check: try registering DirectML EP on a temporary session builder.
/// Returns true if DirectML EP is available in the loaded ORT binary.
#[cfg(target_os = "windows")]
fn probe_directml_ep_available() -> bool {
    match Session::builder() {
        Ok(builder) => {
            let ep = ort::ep::DirectML::default().build();
            builder.with_execution_providers([ep]).is_ok()
        }
        Err(_) => false,
    }
}

/// Stub: DirectML EP not compiled in.
#[cfg(not(target_os = "windows"))]
const fn probe_directml_ep_available() -> bool {
    false
}

/// Full GPU diagnostic: providers, device info.
///
/// Does NOT include a smoke test (that requires a model, handled by nsf_hifigan_onnx).
///
/// 【两个字段都必须是"实际生效"的取值，不能是默认值】
/// `selected_ep` 曾经取 `env_ep_choice()`（只读环境变量，UI 设置完全不参与），
/// `gpu_device_id` 曾经硬写成 `0`。于是插件日志里写着 `device 0`（独显），
/// 而会话实际建在 `device 1`（集显）上 —— 排查"GPU 为什么没跑起来"时，
/// 这两个数字互相矛盾，只会把人带偏。
pub fn diagnose_gpu() -> GpuDiagnostic {
    let available_providers = diagnose_available_providers();
    let selected_ep = ep_choice_for_role(OrtSessionRole::Vocoder);
    let gpu_device_id = effective_dml_device_id().unwrap_or(0);
    let ort_build_info = std::panic::catch_unwind(|| ort::info().to_string())
        .unwrap_or_else(|_| "ort::info() unavailable".to_string());

    GpuDiagnostic {
        available_providers,
        selected_ep,
        gpu_device_id,
        ort_build_info,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 预算作用域重叠（启动时三个预热线程并发进入）时，先结束的不得解除上限，
    /// 最后一个离开者必须清零 —— 旧实现会把上限永久泄漏成 2（第 5 轮审查 P2-1）。
    #[test]
    fn overlapping_diagnostic_budgets_clear_the_cap_only_when_the_last_one_ends() {
        DIAGNOSTIC_THREAD_CAP.store(0, Ordering::Release);
        DIAGNOSTIC_BUDGET_DEPTH.store(0, Ordering::Release);
        let first = DiagnosticBudget::enter(2);
        let second = DiagnosticBudget::enter(2);
        drop(first);
        assert_eq!(
            DIAGNOSTIC_THREAD_CAP.load(Ordering::Acquire),
            2,
            "另一个作用域还活着，线程上限不得解除"
        );
        drop(second);
        assert_eq!(
            DIAGNOSTIC_THREAD_CAP.load(Ordering::Acquire),
            0,
            "最后一个离开者必须把上限清零"
        );
    }

    /// 全局租约是进程级的：涉及它的用例必须串行，否则一个用例持锁会让另一个用例
    /// 读到"别人在构建"。
    static LEASE_SERIAL: Mutex<()> = Mutex::new(());

    fn serial() -> std::sync::MutexGuard<'static, ()> {
        LEASE_SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// 每个租约用例的收尾：无论成败（panic 也走 Drop），把进程级租约与测试时钟
    /// 复位。进程级状态 + cargo 并行测试意味着一个用例留下的孤儿票据会把
    /// 后续**所有**租约用例污染成"busy" —— 失败必须失败得干净。
    struct LeaseReset;
    impl Drop for LeaseReset {
        fn drop(&mut self) {
            lease().store(0, Ordering::SeqCst);
            TEST_CLOCK_MS.store(u64::MAX, Ordering::Relaxed);
        }
    }

    /// **不变量：租约挡得住别的线程、挡不住自己、释放后能再取。**
    ///
    /// 【为什么这三条必须一起钉】诊断运行现在整段持有它（`begin_diagnostic_run`）：
    /// * 挡不住别的线程 ⇒ 两个 DML 会话并发创建 ⇒ 驱动层挂起；
    /// * 挡不住自己 ⇒ 诊断内部的构建路径等满 20 秒后失败 ⇒ 基准测试永远跑不完；
    /// * 释放不掉 ⇒ 之后所有渲染都拿不到会话。
    #[test]
    fn the_build_lease_is_held_exclusively_and_is_reentrant_on_its_own_thread() {
        let _serial = serial();
        let _reset = LeaseReset;
        let outer = acquire_session_build_lock(std::time::Duration::from_secs(5))
            .expect("空闲时租约必须可得");

        // 同线程重入：立刻成功（1 ms 预算），而不是等满再失败。
        let nested = acquire_session_build_lock(std::time::Duration::from_millis(1))
            .expect("同线程重入必须立刻成功，否则诊断会自锁");
        drop(nested);
        // 内层释放不得放掉外层。
        assert!(
            lease().load(Ordering::Acquire) != 0,
            "内层守卫 Drop 不得释放外层租约"
        );

        // 另一个线程：有界失败，而不是永久阻塞。
        let blocked = std::thread::spawn(|| {
            acquire_session_build_lock(std::time::Duration::from_millis(50)).is_err()
        })
        .join()
        .expect("join");
        assert!(blocked, "持有期间其它线程必须拿不到租约");

        drop(outer);
        let again = acquire_session_build_lock(std::time::Duration::from_secs(5));
        assert!(again.is_ok(), "释放后必须可以再取");
    }

    /// **不变量：心跳推进持有时间戳，但不改变票据身份。**
    ///
    /// 【为什么这是回归测试】抢占阈值读的是票据里的时间戳。若心跳改写了票据本身
    /// （而不只是高位时间戳），`Drop` 的 `compare_exchange` 就对不上 —— 租约永远
    /// 释放不掉，之后所有会话构建都会在 20 秒后失败。若没有心跳，一次合法但慢的
    /// 基准测试（实测 31–134 秒）会在 90 秒处被判卡死并遭抢占，而抢占的后果正是
    /// 那把锁要防的驱动层挂起。
    #[test]
    fn a_heartbeat_refreshes_the_hold_timestamp_and_keeps_the_ticket() {
        let _serial = serial();
        let _reset = LeaseReset;
        let run_lease = acquire_session_build_lock(std::time::Duration::from_secs(5))
            .expect("空闲时租约必须可得");
        let before = run_lease.ticket.get();
        std::thread::sleep(std::time::Duration::from_millis(20));
        run_lease.heartbeat();
        let after = run_lease.ticket.get();
        assert!(
            ticket_ms(after) >= ticket_ms(before) + 20,
            "心跳必须把持有时间戳推到此刻（before={} after={}）",
            ticket_ms(before),
            ticket_ms(after)
        );
        assert_eq!(
            after & 0xFFFF,
            before & 0xFFFF,
            "心跳不得改变票据身份（低位序号），否则 Drop 释放不掉"
        );
        drop(run_lease);
        assert!(
            acquire_session_build_lock(std::time::Duration::from_secs(5)).is_ok(),
            "心跳之后 Drop 仍必须释放租约"
        );
    }

    /// **不变量：抢占成功必须把守卫交给抢占者，而不是把租约孤立。**
    ///
    /// 【为什么这是回归测试】曾经的实现里，抢占 CAS 成功后走的是 `continue`：下一轮
    /// 看到的是**自己上一轮写入的票据**，`CAS(0, …)` 永远不再成立，等待者只能等满
    /// 超时失败 —— 而租约从此停在一个无人持有的票据上：原持有者的心跳与 Drop 都
    /// 对不上号，抢占者没拿到守卫。之后每一次会话构建都会等满超时失败，直到进程
    /// 重启 —— 恰好是这把锁要防的"永久失去渲染能力"。
    #[test]
    fn a_successful_steal_returns_a_usable_guard() {
        let _serial = serial();
        let _reset = LeaseReset;
        // 伪造一个远超阈值前取得、再无人推进的票据（模拟持有者卡死）。
        // 【为什么拨快测试时钟】单调时钟的零点是进程启动，刚起进程时 now_ms 很小，
        // "now - 100s" 会饱和成 0 —— 票据反而显得年轻。测试时钟把时刻抬到
        // 1000s，伪造与真实取锁共用同一时钟，读数一致。
        TEST_CLOCK_MS.store(1_000_000, Ordering::Relaxed);
        let age = LEASE_STEAL_AFTER.as_millis() as u64 + 30_000;
        let stale = ((now_ms().saturating_sub(age) & 0x0000_FFFF_FFFF_FFFF) << 16) | 0x0B0B;
        lease().store(stale, Ordering::SeqCst);

        let stolen = acquire_session_build_lock(std::time::Duration::from_millis(300))
            .expect("卡死的持有者必须可被抢占，且抢占者必须拿到守卫");
        assert_ne!(
            lease().load(Ordering::Acquire),
            0,
            "抢占成功后租约必须由抢占者持有"
        );

        drop(stolen);
        assert_eq!(
            lease().load(Ordering::Acquire),
            0,
            "抢占得到的守卫必须能释放租约"
        );
        assert!(
            acquire_session_build_lock(std::time::Duration::from_millis(50)).is_ok(),
            "抢占与释放之后，下一个请求必须能正常拿到租约"
        );
    }

    /// **不变量：本线程的租约被抢占后，再获取必须是真实持有，不得走嵌套旁路。**
    ///
    /// 【为什么这是回归测试】可重入判据若只记"我持过锁"（布尔）而不记票据，被抢后
    /// 本线程仍会被记为持锁者，嵌套获取会拿到一个**不设防的空守卫**直接去建会话
    /// —— 重新打开两种致命并发（并发建 DML 会话、建会话撞上首次推理）。这里让
    /// "小偷"拿完就走（租约归零），隔离出纯粹的"TLS 记录 vs 实际租约"失配场景。
    #[test]
    fn a_thread_whose_lease_was_stolen_re_acquires_normally() {
        let _serial = serial();
        let _reset = LeaseReset;
        let victim = acquire_session_build_lock(std::time::Duration::from_secs(5))
            .expect("空闲时租约必须可得");
        // 模拟"被抢走又释放"：租约现在是空的，victim 的 TLS 记录还指着自己的票据。
        lease().store(0, Ordering::SeqCst);

        let again = acquire_session_build_lock(std::time::Duration::from_secs(5))
            .expect("租约空闲时本线程必须能重新真实持有");
        assert!(!again.nested, "TLS 记录失配时不得返回空守卫（嵌套旁路）");

        // 而且这次持有是真的：另一线程必须拿不到。
        let blocked = std::thread::spawn(|| {
            acquire_session_build_lock(std::time::Duration::from_millis(50)).is_err()
        })
        .join()
        .expect("join");
        assert!(blocked, "重新持有期间其它线程必须拿不到租约");

        drop(again);
        drop(victim); // 它的释放 CAS 对不上号，必须无害
        assert_eq!(lease().load(Ordering::Acquire), 0, "最终租约必须归零");
    }

    /// **不变量：持续心跳的合法长跑绝不遭抢占；静默超过阈值才可被抢。**
    ///
    /// 【为什么两条钉在一起】心跳机制的全部意义就是把"跑得久"与"卡死了"分开：
    /// 单轮推理的超时（120s）一度大于抢占阈值（90s），且预热轮之前没有心跳 ——
    /// 健康但慢的基准测试会被误判卡死，抢占者与被抢占者同时在驱动层操作 DML。
    /// 用测试时钟拨快"持有时长"，免去真实等待。
    #[test]
    fn a_heartbeating_holder_is_never_stolen_and_a_silent_one_is() {
        let _serial = serial();
        let _reset = LeaseReset;
        // 阶段一：持有者持续心跳，时钟推进远超阈值 —— 等待者必须等满自己的超时失败。
        TEST_CLOCK_MS.store(1_000, Ordering::Relaxed);
        let holder = acquire_session_build_lock(std::time::Duration::from_secs(5))
            .expect("空闲时租约必须可得");
        let waiter = std::thread::spawn(|| {
            acquire_session_build_lock(std::time::Duration::from_secs(2)).is_err()
        });
        let start = std::time::Instant::now();
        while start.elapsed() < std::time::Duration::from_secs(2) {
            // 每 50ms 真实时间把虚拟时钟推 30s：远超阈值，但心跳总在前头。
            TEST_CLOCK_MS.fetch_add(30_000, Ordering::Relaxed);
            holder.heartbeat();
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        assert!(
            waiter.join().expect("join"),
            "心跳中的持有者不得被抢占（等待者应超时失败）"
        );

        // 阶段二：停止心跳，把时钟推过阈值 —— 等待者必须抢占成功并拿到守卫。
        let holder_ticket = holder.ticket.get();
        TEST_CLOCK_MS.fetch_add(4 * LEASE_STEAL_AFTER.as_millis() as u64, Ordering::Relaxed);
        let stealer = std::thread::spawn(|| {
            acquire_session_build_lock(std::time::Duration::from_secs(2))
                .expect("静默超阈值的持有者必须可被抢占")
        })
        .join()
        .expect("join");
        assert_ne!(
            stealer.ticket.get(),
            holder_ticket,
            "抢占者必须拿到自己的新票据"
        );
        drop(stealer);
        drop(holder); // 它的释放 CAS 对不上号（租约已易主又归零），必须无害
        assert_eq!(lease().load(Ordering::Acquire), 0, "收尾时租约必须归零");
    }

    /// **不变量：租约被占不算"模型不可用"。**
    ///
    /// 【为什么这是回归测试】预热把结果缓存进 `PREWARM`，而 `is_available() == false`
    /// 会让渲染链整条跳过直到重启。于是"用户恰好在预热跑完前点了基准测试"会变成
    /// 一次永久停机 —— 除非这两类失败不被记账。
    #[test]
    fn a_deferred_prewarm_is_not_verdict_about_the_model() {
        assert!(
            prewarm_verdict(Err(SessionBuildError::Busy {
                waited: std::time::Duration::from_secs(20)
            }))
            .is_none(),
            "租约被占不是结论"
        );
        assert!(
            prewarm_verdict(Err(SessionBuildError::Failed(
                "model file missing".to_string()
            )))
            .is_some(),
            "真失败仍必须记账"
        );
        assert!(prewarm_verdict(Ok(())).is_some());
    }

    /// 诊断预算只能**收紧**线程数，绝不能把它放大到超过渲染级预算。
    #[test]
    fn the_diagnostic_cap_only_ever_shrinks_the_thread_budget() {
        assert_eq!(super::apply_diagnostic_cap(16, 0), 16, "0 = 不限制");
        assert_eq!(super::apply_diagnostic_cap(16, 2), 2);
        assert_eq!(
            super::apply_diagnostic_cap(2, 8),
            2,
            "上限比基础值大时不得放大"
        );
        assert_eq!(super::apply_diagnostic_cap(1, 2), 1);
        assert_eq!(super::apply_diagnostic_cap(0, 2), 1, "下限恒为 1");
    }
    use super::{smoke_probe_frames, OrtSessionRole, PITCH_SMOKE_FRAMES, VOCODER_SMOKE_FRAMES};
    /// 动态整段分离不复用巨型工作区；其它角色和Intel macOS原策略保持。
    #[test]
    fn cpu_allocator_reuse_is_role_specific() {
        assert!(super::cpu_memory_reuse(OrtSessionRole::Vocoder));
        assert!(super::cpu_memory_reuse(OrtSessionRole::PitchDetector));
        assert_eq!(
            super::cpu_memory_reuse(OrtSessionRole::Separator),
            cfg!(all(target_os = "macos", target_arch = "x86_64"))
        );
    }
    /// 新旧CPU分配策略运行同一真实HNSEP mask，输出逐bit一致；不以更小内存改变DSP语义。
    #[test]
    #[ignore = "真实HNSEP CPU分配策略短输入oracle：显式运行"]
    fn separator_cpu_allocator_policy_preserves_real_mask_bits() {
        let model = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../src-tauri/resources/models/hnsep/hnsep.onnx");
        let samples = (0..2 * 1025 * 32)
            .map(|index| (index % 257) as f32 / 256. - 0.5)
            .collect::<Vec<_>>();
        let run = |reuse| {
            let (mut session, _) = super::build_cpu_session_with_memory_reuse(
                &model,
                OrtSessionRole::Separator,
                "cpu-oracle",
                reuse,
            )
            .unwrap();
            let input = ort::value::Tensor::from_array((
                [1usize, 2, 1025, 32],
                samples.clone().into_boxed_slice(),
            ))
            .unwrap();
            let output = session
                .run(ort::inputs![input])
                .unwrap()
                .into_iter()
                .next()
                .unwrap()
                .1;
            output.try_extract_tensor::<f32>().unwrap().1.to_vec()
        };
        let old = run(true);
        let new = run(false);
        assert_eq!(old.len(), new.len());
        assert!(old
            .iter()
            .zip(&new)
            .all(|(old, new)| old.to_bits() == new.to_bits()));
    }
    /// 10秒整段输入记录真实ORT算子形状/输出大小，用于定位长源峰值，不修改模型或切块。
    #[test]
    #[ignore = "真实HNSEP算子资源profile：显式运行并提供隔离日志前缀"]
    fn profile_full_separator_cpu_operators() {
        assert!(std::env::var_os("HIFISHIFTER_HNSEP_ORT_PROFILE_PREFIX").is_some());
        let model = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../src-tauri/resources/models/hnsep/hnsep.onnx");
        let (mut session, _) = super::build_cpu_session_with_memory_reuse(
            &model,
            OrtSessionRole::Separator,
            "cpu-profile",
            false,
        )
        .unwrap();
        let samples = (0..2 * 1025 * 864)
            .map(|index| (index % 257) as f32 / 256. - 0.5)
            .collect::<Vec<_>>();
        let input =
            ort::value::Tensor::from_array(([1usize, 2, 1025, 864], samples.into_boxed_slice()))
                .unwrap();
        let output = session.run(ort::inputs![input]).unwrap();
        let first = output.into_iter().next().unwrap().1;
        assert!(first
            .try_extract_tensor::<f32>()
            .unwrap()
            .1
            .iter()
            .all(|sample| sample.is_finite()));
        let path = session.end_profiling().unwrap();
        println!("SEPARATOR_OPERATOR_PROFILE {path}");
    }

    /// 烟测探针的帧数必须按角色区分。
    ///
    /// 【为什么必须钉住】HNSEP 的 mask-only 模型输入是 `[batch, 2, 1025, n_frames]`；
    /// 若把声码器的帧数套到**每个**动态维，探针张量是 `[1, 2, 1025, 4096]`
    /// = **32 MB**（旧波形域模型只要 16 KB）。烟测只为证明"EP 能跑通一次推理"，
    /// 不该分配几十 MB。
    ///
    /// 声码器的探针还必须等于**生产推理长度**（512 帧）—— 见
    /// [`VOCODER_SMOKE_FRAMES`] 的说明：探针比生产大 8 倍，会把"慢"误判成"挂起"。
    /// 该值与 `nsf_hifigan_onnx::CHUNK_MAX_FRAMES_DEFAULT` 的耦合由那边的
    /// `vocoder_smoke_probe_matches_the_render_chunk` 测试钉住。
    #[test]
    fn smoke_probe_frames_are_role_specific() {
        assert_eq!(
            smoke_probe_frames(OrtSessionRole::Separator),
            32,
            "HNSEP probe must stay small"
        );
        assert_eq!(
            smoke_probe_frames(OrtSessionRole::Vocoder),
            VOCODER_SMOKE_FRAMES
        );
        assert_eq!(
            smoke_probe_frames(OrtSessionRole::PitchDetector),
            PITCH_SMOKE_FRAMES
        );
    }

    /// HNSEP 探针长度必须满足模型对帧轴的整除约束（`n_frames % 32 == 0`）。
    ///
    /// 【为什么重要】该约束来自 `hnsep_dsp::SEGMENT_FRAMES`（网络要求帧数是 32 的
    /// 倍数，见该常量说明）。探针若给出非法帧数，DML/CoreML 可能仅在烟测阶段失败，
    /// 表现为"GPU 不可用"而实际是探针长度选错 —— 极难定位。
    #[test]
    fn hnsep_probe_frames_satisfy_model_divisibility() {
        let f = smoke_probe_frames(OrtSessionRole::Separator);
        assert_eq!(f % 32, 0, "HNSEP probe frames {f} must be a multiple of 32");
        assert!(f >= 32, "must fit at least one segment, got {f}");
    }

    /// 探针张量规模必须保持在**千字节级**，不得因新增模型而膨胀到 MB 级。
    ///
    /// 用 HNSEP 的真实形状 `[1, 2, 1025, frames]` 计算并断言上界；
    /// 这是防止"未来某个模型又套上 4096"的回归闸门。
    #[test]
    fn hnsep_probe_tensor_stays_small() {
        let bins = 1025usize; // n_fft/2 + 1
        let f = smoke_probe_frames(OrtSessionRole::Separator);
        // 形状 [1, 2, 1025, frames]（batch, channels, bins, frames）
        let shape = [1usize, 2, bins, f];
        let bytes = shape.iter().product::<usize>() * std::mem::size_of::<f32>();
        assert!(
            bytes <= 512 * 1024,
            "HNSEP smoke probe would allocate {} KB; keep it small",
            bytes / 1024
        );
        // 与旧值对比：若误用 4096 会是 32 MB
        let wrong_shape = [1usize, 2, bins, PITCH_SMOKE_FRAMES];
        let wrong = wrong_shape.iter().product::<usize>() * std::mem::size_of::<f32>();
        assert!(
            wrong > 16 * 1024 * 1024,
            "sanity: the naive 4096 would indeed be huge ({})",
            wrong
        );
    }

    /// DirectML 的禁用记账必须**按设备**，不能按进程。
    ///
    /// 【为什么必须钉住】旧实现是一个 `AtomicBool`：集显（device 1）首次推理
    /// 超时就把整个进程的 DirectML 关掉 —— 独显（device 0）从此不会被尝试，
    /// 而基准测试的 DirectML 分支会静默退化成 CPU（用户只看到"DirectML 可用 /
    /// 失败"）。实测日志：`disable_directml` 之后紧跟
    /// `DirectML disabled for this process (earlier smoke test hung) — using CPU`。
    #[cfg(target_os = "windows")]
    #[test]
    fn the_directml_disable_latch_is_per_device() {
        let mut state = super::DmlDisableState::default();
        assert!(!state.is_disabled(Some(0)));
        assert!(!state.is_disabled(Some(1)));

        // 只禁用集显：独显必须保持可用。
        assert!(state.disable(Some(1)), "first disable flips the state");
        assert!(state.is_disabled(Some(1)));
        assert!(!state.is_disabled(Some(0)), "the dGPU must survive");
        assert!(!state.disable(Some(1)), "second disable is a no-op");

        // 诊断跑通后可以撤销，用户不必重启应用。
        assert!(state.enable(Some(1)));
        assert!(!state.is_disabled(Some(1)));
        assert!(!state.enable(Some(1)), "re-enabling twice is a no-op");

        // 不知道设备时（DXGI 枚举失败）只能整体禁用，且此时任何设备都算禁用。
        assert!(state.disable(None));
        assert!(state.is_disabled(None));
        assert!(state.is_disabled(Some(0)));
        assert!(state.is_disabled(Some(1)));
    }
}
