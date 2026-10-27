//! 实时回调发布的宿主播放时钟；GUI读近似最新块，process只执行原子写入。
//!
//! 【本模块回答三个问题】宿主在哪儿（`position`）、在不在播（`playing`）、
//! **这个读数是谁给的**（`provenance`）。
//!
//! 第三个问题是这一轮新增的，也是"光标冻住"这类报障反复复发的真正原因：没有它，
//! 「读数为 0」「采样停摆」「宿主拒绝采样」在日志里完全同形，每一次都只能从现象反推。
//! 现在每一个读数都能被指认到具体的来源与年龄。
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU32, AtomicU64, Ordering};

/// 宿主读数在**墙钟**上的生存期。
///
/// 【为什么需要上界】宿主采样与音频块是两个来源：前者 30 Hz、来自宿主主线程，
/// 后者每块、来自音频线程。谁新用谁 —— 但"新"必须能被判出来，而 `host_valid`
/// 是一个**只置真、从不复位**的闩锁：一次成功的采样之后，它会把音频线程的实时
/// 读数**永久**压住。现场症状正是"光标停在原地、偏移越拉越大"，以及偶尔一帧
/// 跳回真实位置（"随机闪"）。
///
/// 宿主节拍约 30 Hz（33 ms）。150 ms 是 4~5 个周期：足以容忍调度抖动，
/// 又远短于用户能察觉的漂移。
const HOST_READING_STALE_MS: u64 = 150;

/// 播放中单次采样允许的最大前进量（秒）。
///
/// 【为什么要设上界】现场症状是"光标大部分时间正常，偶尔跳到正常位置之后约 1.4 秒、
/// 再跳回来"。1.4 秒远大于一个宿主节拍（33 ms）或一个音频块（10 ms），不可能是
/// 正常推进 —— 它是一个**不可能值**。此前我们照单全收，于是界面忠实地把它画出来。
/// 0.5 秒留出 15 倍余量：真实的快速前进（高倍速播放）也不会在一拍里跨过它。
const MAX_PLAUSIBLE_ADVANCE_SEC: f64 = 0.5;

/// 宿主读数在**音频块计数**上的生存期（约 8 块 ≈ 85 ms @ 48 kHz / 512 帧）。
///
/// 【为什么还要一条块计数的判据】音频线程不能读墙钟（实时纪律），所以它的进度
/// 只能用自己的块序号表达。宿主采样时记下当时的块号；块号走得太远就说明宿主
/// 采样已经落后于音频，此时以音频为准。
const HOST_STALE_BLOCKS: u64 = 8;

/// 一次定位请求最多等宿主多久。
///
/// ARA 明确写着播放请求"typically scheduled and executed with some delay"
/// （`ARAInterface.h:2157-2161`），而实测 250 ms 内多半回不来。1 s 足够覆盖宿主的一次
/// 模型回调，又短到"宿主拒绝定位"时用户不会以为插件卡死。
const SEEK_TTL_MS: u64 = 1000;

/// 宿主报回的位置与定位目标相差不到这个值，就认为定位**已经落地**。
///
/// 0.25 s 的依据：宿主的定位精度是采样级，真正落地时差值应当接近 0；而"跳转前"的
/// 旧位置与之的差通常是大段距离。0.25 s 既容忍播放中采样的一拍抖动，又不会把
/// "宿主还在原地"误判成落地。
const SEEK_SETTLE_EPS_SEC: f64 = 0.25;

/// prefetch 块贡献的**播放位**的生存期。
///
/// 【为什么需要上界】prefetch 块的 `kPlaying` 与 realtime 块同源（都来自宿主传输态），
/// 但**块会停**：宿主停播后不再有 prefetch 块，而最后一块的播放位仍是真。没有上界，
/// 光标会在停播之后继续按 1x 前漂 —— 那正是我们要消灭的另一类"说谎"。
const PREFETCH_PLAYING_STALE_MS: u64 = 300;

/// 进程启动以来的毫秒数。只用于新鲜度判定，不参与任何音视频时序。
fn now_ms() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    // +1：0 是"从未"的哨兵（见 `seek_at_ms`）。进程启动后的第一毫秒里
    // `as_millis()` 正好是 0，直接返回它会把一次刚发出的定位请求当成"没有请求"。
    EPOCH
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis() as u64
        + 1
}

/// 一个读数**是谁给的**。
///
/// 【为什么要把它做成类型】"位置是 0" 有四种完全不同的含义：宿主真的在 0 秒、
/// 宿主从没被采样过、宿主的采样过期了、以及我们自己刚请求了一次定位。四者在界面上
/// 都画成同一条竖线，只有来源能把它们分开。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Provenance {
    /// 宿主主线程采样（REAPER 的 `GetPlayPositionEx` / `GetCursorPositionEx`）。
    HostSample,
    /// 某个实例的 **realtime** 处理块（Capture / 普通 VST3 插入）。
    RealtimeBlock,
    /// 我们自己请求的定位还没被宿主确认（见 `SeekIntent`）。
    SeekIntent,
    /// 以上都没有：位置是上一次的残值。
    Stale,
}

impl Provenance {
    /// 载荷/日志用的稳定短名。
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Provenance::HostSample => "host",
            Provenance::RealtimeBlock => "block",
            Provenance::SeekIntent => "seek",
            Provenance::Stale => "stale",
        }
    }
}

/// 一次完整的传输读数：位置 + 播放位 + 来源 + 年龄。
///
/// 【为什么把它们绑在一起返回】分开取会在两次读取之间被别的线程推进，于是
/// "位置是 31.2、来源是宿主、年龄 400 ms"这种自相矛盾的组合会被画到界面上。
/// 一个结构体一次读出来，读者拿到的永远是一份自洽的快照。
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct TransportView {
    pub position_sec: f64,
    pub playing: bool,
    pub provenance: Provenance,
    /// 距离产生这个读数过去了多久；`Stale` 与 `SeekIntent` 下无意义。
    pub age_ms: u32,
    /// 有一次定位在飞（见 `SeekIntent`）。
    pub seek_pending: bool,
}

impl TransportView {
    /// 载荷用的 JSON；字段名与 `get_playback_state` 的既有字段一一对应。
    pub(crate) fn to_json(self) -> serde_json::Value {
        serde_json::json!({
            "position_sec": self.position_sec,
            "is_playing": self.playing,
            "position_source": self.provenance.as_str(),
            "position_age_ms": self.age_ms,
            "seek_pending": self.seek_pending,
        })
    }
}

#[derive(Default)]
pub(crate) struct TransportClock {
    position: AtomicU64,
    playing: AtomicBool,
    tempo: AtomicU64,
    /// 拍号分子/分母。与 tempo 同源（VST3 进程上下文），同为**宿主权威**的只读读数。
    time_sig_numerator: AtomicI32,
    time_sig_denominator: AtomicI32,
    host_pose: AtomicU64,
    host_valid: AtomicBool,
    /// 最近一次宿主采样的墙钟毫秒（见 `HOST_READING_STALE_MS`）。
    host_at_ms: AtomicU64,
    /// 音频线程推进的块序号；只用它的**差值**判新鲜度，绝不当作时间。
    audio_serial: AtomicU64,
    /// 宿主采样那一刻的块序号。
    host_audio_serial: AtomicU64,
    /// **最近一次已知位置**（宿主采样或音频块都会写）。`Stale` 读它，而不是读 0。
    ///
    /// 【为什么不能读 `position`】那个量只在 realtime 块里被写；一场只在宿主侧发生的
    /// 播放（编辑器实例不处理音频）会让它永远停在 0 —— 界面于是把"上一次定位"抹掉。
    last_known: AtomicU64,
    /// 最近一次**读到**块推进的墙钟毫秒（读者侧打戳，音频线程不碰时钟）。
    block_seen_at_ms: AtomicU64,
    block_seen_serial: AtomicU64,
    /// prefetch 块贡献的播放位，以及它的新鲜度（用"块序号 + 读者打戳"表达）。
    ///
    /// 【为什么位置与播放位要分开】**播放位**在 prefetch 块里是忠实的（与 realtime
    /// 同源），而**位置**不是（那是渲染进度，按设计领先播放头）。所以 prefetch 只
    /// 能投这一票。
    prefetch_playing: AtomicBool,
    prefetch_serial: AtomicU64,
    prefetch_seen_serial: AtomicU64,
    prefetch_seen_at_ms: AtomicU64,
    /// 定位意图：目标 / 请求时刻 / 请求时的播放位 / 代数。
    seek_target: AtomicU64,
    seek_at_ms: AtomicU64,
    seek_playing: AtomicBool,
    /// 有多少次定位请求**没能**在 TTL 内被宿主确认（诊断）。
    seek_unsettled: AtomicU64,
    /// 宿主采样失败次数（诊断）。
    ///
    /// 【为什么必须计数】授权被撤销、项目指针失效、宿主线程不泵 —— 三者都会让
    /// `sample()` 返回 `Err`，而调用点此前把它整个吞掉。没有这个计数，
    /// "光标不动"与"我们从没读过"在日志里一模一样。
    sample_failures: AtomicU64,
    /// 当前定位代数（诊断/测试）。u32 足够：它是计数器，不是时间戳。
    seek_epoch: AtomicU32,
    realtime: AtomicU64,
    prefetch: AtomicU64,
    offline: AtomicU64,
    switches: AtomicU64,
    backwards: AtomicU64,
    writer: AtomicU64,
    role: AtomicI32,
    mode: AtomicI32,
    sample: AtomicI64,
    rate: AtomicU64,
    observed_playing: AtomicBool,
    system_time: AtomicI64,
    system_valid: AtomicBool,
    continuous: AtomicI64,
    continuous_valid: AtomicBool,
    stops: AtomicU64,
    playback_stops: AtomicU64,
    editor_stops: AtomicU64,
    stop_writer: AtomicU64,
    stop_role: AtomicI32,
    /// 是否**收到过**任何宿主 process 上下文（不论播没播）。
    received: AtomicBool,
}

impl TransportClock {
    /// UI线程发布所属工程实际听到位置；bool占f64最低位，误差小于1ULP且元组原子一致。
    pub fn publish_host(&self, position: f64, playing: bool) {
        if !position.is_finite() {
            return;
        }
        // 【只诊断，不修改】把不可能的前进量记下来：它是"光标偶尔跳 1.4 秒又跳回"
        // 的唯一直接证据，而我们目前连它来自宿主还是来自我们自己都还不知道。
        // 不在这里纠偏：纠偏会掩盖真正的原因，还会把合法的快速定位一起改掉。
        if self.host_valid.load(Ordering::Acquire) {
            let previous = self.host_pose.load(Ordering::Acquire);
            let previous_playing = previous & 1 != 0;
            let previous_position = f64::from_bits(previous & !1);
            let advance = position - previous_position;
            if playing
                && previous_playing
                && advance.is_finite()
                && advance > MAX_PLAUSIBLE_ADVANCE_SEC
            {
                hifishifter_kernel::log_limiter::emit_limited_key(
                    log::Level::Warn,
                    "transport.implausible-advance",
                    format_args!(
                        "[transport] implausible host advance: {previous_position:.3}s -> {position:.3}s \
                         (+{advance:.3}s in one sample); the editor will draw it as given"
                    ),
                );
            }
        }
        // 【为什么先结算定位】宿主报回的位置就是"落地"的判据：它一旦落进目标附近，
        // 我们持有的乐观值就该退场，此后一切都是宿主读数。
        self.settle_seek(position);
        // 先读块序号再发布位置：并发的音频块只会让宿主读数显得**更旧**，
        // 绝不会让它显得更新 —— 判据因此永远偏向音频线程（安全的方向）。
        let audio_serial = self.audio_serial.load(Ordering::Acquire);
        self.host_pose.store(
            (position.to_bits() & !1) | u64::from(playing),
            Ordering::Release,
        );
        self.host_audio_serial
            .store(audio_serial, Ordering::Release);
        self.host_at_ms.store(now_ms(), Ordering::Release);
        self.host_valid.store(true, Ordering::Release);
        self.last_known.store(position.to_bits(), Ordering::Release);
    }

    /// 记一次宿主采样失败（原因已由调用方分类）。
    ///
    /// 【为什么日志在限流器里】采样是 30–50 Hz 的路径，宿主一旦持续拒绝（授权撤销、
    /// 项目指针失效），没有限流就是刷屏；有限流则恰好一行说明"从什么时候开始、
    /// 累计多少次"。
    pub fn note_sample_failure(&self, reason: &str) {
        let count = self.sample_failures.fetch_add(1, Ordering::Relaxed) + 1;
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            "transport.sample-failed",
            format_args!("[transport] host sample failed ({count} total): {reason}"),
        );
    }

    /// 只采集原子诊断，不修改播放权威/不打印日志；并发快照是近似观测，不作精确同步证明。
    pub fn observe(
        &self,
        context: &crate::audio_abi::ProcessContext,
        writer: u64,
        role: i32,
        mode: i32,
    ) {
        match mode {
            0 => {
                self.realtime.fetch_add(1, Ordering::Relaxed);
            }
            1 => {
                self.prefetch.fetch_add(1, Ordering::Relaxed);
            }
            2 => {
                self.offline.fetch_add(1, Ordering::Relaxed);
            }
            _ => return,
        }
        if !context.sample_rate.is_finite() || context.sample_rate <= 0. {
            return;
        }
        let old_writer = self.writer.swap(writer, Ordering::Relaxed);
        if old_writer != 0 && old_writer != writer {
            self.switches.fetch_add(1, Ordering::Relaxed);
        }
        let old_sample = self
            .sample
            .swap(context.project_time_samples, Ordering::Relaxed);
        let old_rate = self
            .rate
            .swap(context.sample_rate.to_bits(), Ordering::Relaxed);
        let playing = context.state & (1 << 1) != 0;
        let old_playing = self.observed_playing.swap(playing, Ordering::Relaxed);
        if playing
            && old_playing
            && old_rate == context.sample_rate.to_bits()
            && context.project_time_samples < old_sample
        {
            self.backwards.fetch_add(1, Ordering::Relaxed);
        }
        self.role.store(role, Ordering::Relaxed);
        self.mode.store(mode, Ordering::Relaxed);
        self.system_valid
            .store(context.state & (1 << 8) != 0, Ordering::Relaxed);
        self.system_time
            .store(context.system_time, Ordering::Relaxed);
        self.continuous_valid
            .store(context.state & (1 << 17) != 0, Ordering::Relaxed);
        self.continuous
            .store(context.continuous_time_samples, Ordering::Relaxed);
    }

    /// 记下 prefetch 块带来的**播放位**。位置**绝不**由它写入。
    ///
    /// 【为什么这条不是"放宽纪律"】prefetch 块的位置是渲染进度（领先播放头约 1~1.5 秒），
    /// 但它的 `kPlaying` 位与 realtime 块同源。把两者拆开之后，"宿主在播"这个布尔事实
    /// 在 ARA 形态下终于有了来源 —— 而这正是编辑器实例唯一缺的那一票。
    pub fn observe_prefetch_playing(&self, playing: bool) {
        self.prefetch_playing.store(playing, Ordering::Release);
        self.prefetch_serial.fetch_add(1, Ordering::Release);
    }

    /// 停处理的实例来源也需记录；调用者可能是瞬时clip renderer，并不等于宿主整体停止。
    pub fn observe_stop(&self, writer: u64, role: i32) {
        self.stops.fetch_add(1, Ordering::Relaxed);
        match role {
            1 => {
                self.playback_stops.fetch_add(1, Ordering::Relaxed);
            }
            2 => {
                self.editor_stops.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        self.stop_writer.store(writer, Ordering::Relaxed);
        self.stop_role.store(role, Ordering::Relaxed);
    }

    /// 请求宿主把播放头移到 `target_sec`：在宿主确认之前，读数就停在那里。
    ///
    /// 【为什么把乐观值放进权威】前端已经有三个"我以为"（在途请求表、传输纪元、
    /// 本地写入时刻），它们各自都对，合起来却互相矛盾。定位是否落地是**宿主**的事实，
    /// 只有这一层能观测到；把它放在这里，前端就只需要照抄读数。
    pub fn request_seek(&self, target_sec: f64) {
        if !target_sec.is_finite() {
            return;
        }
        // 请求时刻的播放位必须快照下来：停止态的定位不该让光标自己走起来。
        self.seek_playing.store(self.current_playing(), Ordering::Release);
        self.seek_target
            .store(target_sec.to_bits(), Ordering::Release);
        self.seek_at_ms.store(now_ms(), Ordering::Release);
        self.seek_epoch.fetch_add(1, Ordering::AcqRel);
    }

    /// 宿主报回一个位置：如果它落在定位目标附近，这次定位就算落地了。
    fn settle_seek(&self, observed_sec: f64) {
        let at = self.seek_at_ms.load(Ordering::Acquire);
        if at == 0 || !observed_sec.is_finite() {
            return;
        }
        let target = f64::from_bits(self.seek_target.load(Ordering::Acquire));
        if (observed_sec - target).abs() <= SEEK_SETTLE_EPS_SEC {
            let _ = self.seek_at_ms.compare_exchange(at, 0, Ordering::AcqRel, Ordering::Acquire);
        }
    }

    /// 当前时刻的播放位（不含定位意图）。
    fn current_playing(&self) -> bool {
        let now = now_ms();
        let host_fresh = self.host_is_fresh_at(now);
        let host_playing = host_fresh && (self.host_pose.load(Ordering::Acquire) & 1 != 0);
        host_playing || self.playing.load(Ordering::Acquire) || self.prefetch_playing_at(now)
    }

    /// 仅非实时actor查询，JSON分配绝不进入process/setProcessing。
    pub fn diagnostics(&self) -> serde_json::Value {
        let view = self.read_view();
        serde_json::json!({"realtime_callbacks":self.realtime.load(Ordering::Relaxed),
            "host_position_authority":self.host_valid.load(Ordering::Acquire),
            // 【为什么要报"当前用的是谁"】"光标不跟手"的两种成因（宿主采样停摆 /
            // 音频块停摆）在这两个计数上完全分得开，不必让用户复述现象。
            "host_reading_fresh":self.host_is_fresh_at(now_ms()),
            "audio_serial":self.audio_serial.load(Ordering::Acquire),
            "position_source":view.provenance.as_str(),
            "position_age_ms":view.age_ms,
            "seek_pending":view.seek_pending,
            "seek_epoch":self.seek_epoch.load(Ordering::Relaxed),
            "seek_unsettled":self.seek_unsettled.load(Ordering::Relaxed),
            "sample_failures":self.sample_failures.load(Ordering::Relaxed),
            "prefetch_playing":self.prefetch_playing.load(Ordering::Acquire),
            "prefetch_callbacks":self.prefetch.load(Ordering::Relaxed),"offline_callbacks":self.offline.load(Ordering::Relaxed),
            "writer_switches":self.switches.load(Ordering::Relaxed),"backwards_while_playing":self.backwards.load(Ordering::Relaxed),
            "last_writer":self.writer.load(Ordering::Relaxed),"last_role":self.role.load(Ordering::Relaxed),
            "last_mode":self.mode.load(Ordering::Relaxed),"last_project_samples":self.sample.load(Ordering::Relaxed),
            "last_sample_rate":f64::from_bits(self.rate.load(Ordering::Relaxed)),"last_playing":self.observed_playing.load(Ordering::Relaxed),
            "system_time":self.system_time.load(Ordering::Relaxed),"system_time_valid":self.system_valid.load(Ordering::Relaxed),
            "time_sig_numerator":self.time_sig_numerator.load(Ordering::Relaxed),"time_sig_denominator":self.time_sig_denominator.load(Ordering::Relaxed),
            "continuous_samples":self.continuous.load(Ordering::Relaxed),"continuous_valid":self.continuous_valid.load(Ordering::Relaxed),
            "processing_stops":self.stops.load(Ordering::Relaxed),"playback_stops":self.playback_stops.load(Ordering::Relaxed),
            "editor_stops":self.editor_stops.load(Ordering::Relaxed),"last_stop_writer":self.stop_writer.load(Ordering::Relaxed),
            "last_stop_role":self.stop_role.load(Ordering::Relaxed)})
    }

    /// VST3锁定header中kPlaying=1<<1；project_time_samples不需要音乐时间valid位。
    pub fn update(&self, context: &crate::audio_abi::ProcessContext) {
        if context.sample_rate.is_finite() && context.sample_rate > 0. {
            let seconds = context.project_time_samples as f64 / context.sample_rate;
            // 同 publish_host：音频线程报回的位置同样可以结算一次定位。
            self.settle_seek(seconds);
            self.received.store(true, Ordering::Release);
            // 块序号只递增；它是"音频走到了第几块"的唯一表达（见 HOST_STALE_BLOCKS）。
            self.audio_serial.fetch_add(1, Ordering::Release);
            self.position.store(seconds.to_bits(), Ordering::Release);
            self.last_known.store(seconds.to_bits(), Ordering::Release);
            self.playing
                .store(context.state & (1 << 1) != 0, Ordering::Release);
            // 锁定ivstprocesscontext.h的kTempoValid=1<<10；不可用字段不能覆盖最后有效BPM。
            if context.state & (1 << 10) != 0 && context.tempo.is_finite() && context.tempo > 0. {
                self.tempo.store(context.tempo.to_bits(), Ordering::Release);
            }
            // 锁定ivstprocesscontext.h的kTimeSigValid=1<<13。与 tempo 同样的规矩：
            // 有效位不成立时**不覆盖**最后有效拍号 —— 宿主在某些块里不发它，
            // 那不是"拍号变成 0/0"，只是这一块没说。
            if context.state & (1 << 13) != 0
                && context.time_sig_numerator > 0
                && context.time_sig_denominator > 0
            {
                self.time_sig_numerator
                    .store(context.time_sig_numerator, Ordering::Release);
                self.time_sig_denominator
                    .store(context.time_sig_denominator, Ordering::Release);
            }
        }
    }

    pub fn stopped(&self) {
        self.playing.store(false, Ordering::Release);
    }

    /// 没有有效宿主tempo时返回None，由映射初始值兜底。
    pub fn tempo(&self) -> Option<f64> {
        let value = f64::from_bits(self.tempo.load(Ordering::Acquire));
        (value > 0. && value.is_finite()).then_some(value)
    }

    /// 没有有效宿主拍号时返回None。
    ///
    /// 【为什么返回 Option 而不是默认 4/4】"宿主没告诉我们"与"宿主说是 4/4"是两件
    /// 事：前者该保持上一次的读数（或落回用户设置），后者该如实显示 4/4。
    pub fn time_signature(&self) -> Option<(i32, i32)> {
        let numerator = self.time_sig_numerator.load(Ordering::Acquire);
        let denominator = self.time_sig_denominator.load(Ordering::Acquire);
        (numerator > 0 && denominator > 0).then_some((numerator, denominator))
    }

    /// 是否**收到过**任何宿主 process 上下文。
    ///
    /// 【为什么不能用 read() 的返回值代替】read() 在"宿主在 0 秒处没播放"与"宿主
    /// 从未送过上下文"两种情况下都给 (0.0, false)。对展示来说两者一样（都显示 0），
    /// 但对**让路判定**完全不同：前者可以用来判断"宿主动没动"，后者不能 —— 把"还没
    /// 收到读数"当成"宿主没动"，会让探针每次都误判，于是每次都白白多等一个探针窗口。
    pub fn has_reading(&self) -> bool {
        self.host_valid.load(Ordering::Acquire) || self.received.load(Ordering::Acquire)
    }

    /// 宿主读数是否仍然**新鲜**（两次判据同时成立，见两个常量）。
    ///
    /// 【为什么不是"曾经读到过"】host_valid 只置真、从不复位。用它当权威，
    /// 一次成功的采样就会把音频线程的实时读数永久压住 —— 光标停住、偏移越拉越大，
    /// 而宿主偶尔来一次回调时又跳回真实位置（"随机闪"）。
    fn host_is_fresh_at(&self, now: u64) -> bool {
        if !self.host_valid.load(Ordering::Acquire) {
            return false;
        }
        let at = self.host_at_ms.load(Ordering::Acquire);
        if now.saturating_sub(at) > HOST_READING_STALE_MS {
            return false;
        }
        let blocks = self
            .audio_serial
            .load(Ordering::Acquire)
            .saturating_sub(self.host_audio_serial.load(Ordering::Acquire));
        blocks <= HOST_STALE_BLOCKS
    }

    /// prefetch 块带来的播放位此刻还算数吗。
    fn prefetch_playing_at(&self, now: u64) -> bool {
        if !self.prefetch_playing.load(Ordering::Acquire) {
            return false;
        }
        let serial = self.prefetch_serial.load(Ordering::Acquire);
        let seen = self.prefetch_seen_serial.load(Ordering::Acquire);
        if serial != seen {
            // 第一眼看到新的 prefetch 块：打戳并立刻采信（不给自己加一拍延迟）。
            self.prefetch_seen_serial.store(serial, Ordering::Release);
            self.prefetch_seen_at_ms.store(now, Ordering::Release);
            return true;
        }
        let at = self.prefetch_seen_at_ms.load(Ordering::Acquire);
        at != 0 && now.saturating_sub(at) <= PREFETCH_PLAYING_STALE_MS
    }

    /// 块读数的年龄：读者侧给"块序号推进"打戳（音频线程不读墙钟）。
    fn block_age_ms(&self, now: u64) -> u32 {
        let serial = self.audio_serial.load(Ordering::Acquire);
        let seen = self.block_seen_serial.load(Ordering::Acquire);
        if serial != seen {
            self.block_seen_serial.store(serial, Ordering::Release);
            self.block_seen_at_ms.store(now, Ordering::Release);
            return 0;
        }
        let at = self.block_seen_at_ms.load(Ordering::Acquire);
        if at == 0 {
            return u32::MAX;
        }
        now.saturating_sub(at).min(u32::MAX as u64) as u32
    }

    /// 定位意图的读数；到期即结算（并记一次"宿主没落地"）。
    fn pending_seek_view(&self, now: u64) -> Option<TransportView> {
        let at = self.seek_at_ms.load(Ordering::Acquire);
        if at == 0 {
            return None;
        }
        if now.saturating_sub(at) < SEEK_TTL_MS {
            return Some(TransportView {
                position_sec: f64::from_bits(self.seek_target.load(Ordering::Acquire)),
                playing: self.seek_playing.load(Ordering::Acquire),
                provenance: Provenance::SeekIntent,
                age_ms: now.saturating_sub(at).min(u32::MAX as u64) as u32,
                seek_pending: true,
            });
        }
        if self
            .seek_at_ms
            .compare_exchange(at, 0, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.seek_unsettled.fetch_add(1, Ordering::Relaxed);
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "transport.seek-unsettled",
                format_args!(
                    "[transport] the host never reported the requested position within {SEEK_TTL_MS} ms; the cursor returns to the last observed position"
                ),
            );
        }
        None
    }

    /// 一次完整的读数（测试用：显式给定时钟，避免依赖真实时间）。
    pub(crate) fn read_view_at(&self, now: u64) -> TransportView {
        if let Some(view) = self.pending_seek_view(now) {
            return view;
        }
        let host_fresh = self.host_is_fresh_at(now);
        let block_playing = self.playing.load(Ordering::Acquire);
        let prefetch_playing = self.prefetch_playing_at(now);
        if host_fresh {
            let pose = self.host_pose.load(Ordering::Acquire);
            // 【播放位取**并集**，位置取宿主】两者回答的是不同的问题：位置只有宿主
            // 知道准数，而"在不在播"只要**任一**来源说是真就是真。
            //
            // 【为什么不能取"新鲜者"的播放位】编辑器实例不处理音频，它的音频块播放位
            // 恒为 false（或停在最后一次块值）；而宿主采样在它身上又常常刚过期。
            // 于是"宿主说在播、光标却冻住"—— 用户看到的是"跳到位之后卡一会儿"。
            // 反过来错一次（宿主已经停了、我们多前进几毫秒）代价小得多：下一拍采样
            // 就把位置纠回来，而"该走的时候不走"是看得见的卡顿。
            return TransportView {
                position_sec: f64::from_bits(pose & !1),
                playing: pose & 1 != 0 || block_playing || prefetch_playing,
                provenance: Provenance::HostSample,
                age_ms: now
                    .saturating_sub(self.host_at_ms.load(Ordering::Acquire))
                    .min(u32::MAX as u64) as u32,
                seek_pending: false,
            };
        }
        let serial = self.audio_serial.load(Ordering::Acquire);
        if serial > 0 {
            return TransportView {
                position_sec: f64::from_bits(self.position.load(Ordering::Acquire)),
                playing: block_playing || prefetch_playing,
                provenance: Provenance::RealtimeBlock,
                age_ms: self.block_age_ms(now),
                seek_pending: false,
            };
        }
        TransportView {
            position_sec: f64::from_bits(self.last_known.load(Ordering::Acquire)),
            playing: block_playing || prefetch_playing,
            provenance: Provenance::Stale,
            age_ms: u32::MAX,
            seek_pending: false,
        }
    }

    /// 一次完整的读数：位置 + 播放位 + 来源 + 年龄。
    ///
    /// 【为什么不保留两个返回值】那正是"位置来自宿主、播放位来自音频线程"这类
    /// 自相矛盾的组合得以出现的形状。一个结构体，一个时刻，一份快照。
    pub fn read_view(&self) -> TransportView {
        self.read_view_at(now_ms())
    }

    /// 只要位置与播放位的旧接口。新代码请用 read_view。
    pub fn read(&self) -> (f64, bool) {
        let view = self.read_view();
        (view.position_sec, view.playing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(samples: i64, playing: bool) -> crate::audio_abi::ProcessContext {
        crate::audio_abi::ProcessContext {
            state: if playing { 1 << 1 } else { 0 },
            sample_rate: 44100.,
            project_time_samples: samples,
            ..Default::default()
        }
    }

    /// 真实工程时钟不受预取提前位置或其它clip停止覆盖，后退seek/loop仍能发布。
    #[test]
    fn actual_host_position_survives_prefetch_noise_and_accepts_backward_seeks() {
        let clock = TransportClock::default();
        clock.publish_host(1.25, true);
        clock.update(&context(441000, true));
        clock.stopped();
        assert_eq!(clock.read(), (1.25, true));
        clock.publish_host(0.5, true);
        assert_eq!(clock.read(), (0.5, true));
        clock.publish_host(0.25, false);
        assert_eq!(clock.read(), (0.25, false));
        assert_eq!(clock.diagnostics()["host_position_authority"], true);
    }

    /// 宿主采样一旦**停摆**，音频线程的实时读数必须重新赢回来。
    ///
    /// 【为什么这条最要紧】它就是"光标停在原地、偏移越拉越大"的回归：旧实现在
    /// host_valid 为真时无条件返回宿主值，于是停掉的采样会把播放头钉死在最后
    /// 一次成功的位置上，直到下一次宿主回调把它"闪"回去。
    #[test]
    fn a_stalled_host_sample_yields_to_the_audio_thread() {
        let clock = TransportClock::default();
        clock.publish_host(1.0, true);
        for block in 1..=12i64 {
            clock.update(&context(44100 * block, true));
        }
        // 12 块 > HOST_STALE_BLOCKS ⇒ 宿主读数作废，取音频线程的块位置。
        assert_eq!(clock.read(), (12.0, true));
        assert_eq!(clock.diagnostics()["host_reading_fresh"], false);
        assert_eq!(clock.read_view().provenance, Provenance::RealtimeBlock);
    }

    /// 但一个**新鲜**的宿主采样必须立刻重新成为权威 —— 暂停时的光标位置只有宿主知道。
    #[test]
    fn a_fresh_host_sample_wins_back_immediately() {
        let clock = TransportClock::default();
        clock.publish_host(1.0, true);
        for block in 1..=12i64 {
            clock.update(&context(44100 * block, true));
        }
        assert_eq!(clock.read(), (12.0, true));
        clock.publish_host(12.5, true);
        assert_eq!(clock.read(), (12.5, true));
        assert_eq!(clock.diagnostics()["host_reading_fresh"], true);
    }

    /// 音频不跑时（暂停、插件未被处理）宿主读数不会因为"没有块"而过期。
    #[test]
    fn a_host_sample_without_any_audio_blocks_stays_authoritative() {
        let clock = TransportClock::default();
        clock.publish_host(3.25, false);
        assert_eq!(clock.read(), (3.25, false));
        assert!(clock.diagnostics()["host_reading_fresh"]
            .as_bool()
            .unwrap_or(false));
    }

    /// 诊断真实区分三模式/多writer/回跳/瞬时停止，但本身不能改变GUI播放时间。
    #[test]
    fn diagnostics_count_sources_without_changing_the_transport() {
        let clock = TransportClock::default();
        let mut ctx = context(44100, true);
        ctx.state = (1 << 1) | (1 << 8) | (1 << 17);
        ctx.system_time = 123456;
        ctx.continuous_time_samples = 44100;
        clock.observe(&ctx, 1, 2, 0);
        ctx.project_time_samples = 88200;
        clock.observe(&ctx, 2, 1, 1);
        ctx.project_time_samples = 44100;
        clock.observe(&ctx, 1, 2, 0);
        clock.observe(&ctx, 3, 1, 2);
        clock.observe_stop(2, 1);
        let info = clock.diagnostics();
        assert_eq!(info["realtime_callbacks"], 2);
        assert_eq!(info["prefetch_callbacks"], 1);
        assert_eq!(info["offline_callbacks"], 1);
        assert_eq!(info["writer_switches"], 3);
        assert_eq!(info["backwards_while_playing"], 1);
        assert_eq!(info["playback_stops"], 1);
        assert_eq!(info["system_time_valid"], true);
        assert_eq!(info["continuous_valid"], true);
        assert_eq!(clock.read(), (0.0, false));
        assert_eq!(info["position_source"], "stale");
    }

    #[test]
    fn only_valid_host_tempo_updates_bpm() {
        let clock = TransportClock::default();
        let mut ctx = context(0, false);
        ctx.tempo = 150.;
        clock.update(&ctx);
        assert_eq!(clock.tempo(), None);
        ctx.state = 1 << 10;
        clock.update(&ctx);
        assert_eq!(clock.tempo(), Some(150.));
        for tempo in [f64::NAN, 0., -1.] {
            ctx.tempo = tempo;
            clock.update(&ctx);
            assert_eq!(clock.tempo(), Some(150.));
        }
    }

    /// 拍号与 tempo 同一套规矩：有效位成立才写入，且不可用值不覆盖最后有效读数。
    ///
    /// 【为什么"不覆盖"是重点】宿主并非每个处理块都发拍号。把缺位当成"拍号 0/0"
    /// 会让界面在播放中随机闪回默认值 —— 用户会以为工程被改了。
    #[test]
    fn only_valid_host_time_signature_updates_the_meter() {
        let clock = TransportClock::default();
        let mut ctx = context(0, false);
        ctx.time_sig_numerator = 7;
        ctx.time_sig_denominator = 8;
        // 无有效位：不采纳。
        clock.update(&ctx);
        assert_eq!(clock.time_signature(), None);
        // 有效位成立：采纳。
        ctx.state = 1 << 13;
        clock.update(&ctx);
        assert_eq!(clock.time_signature(), Some((7, 8)));
        // 后续块没有有效位：保持上一次读数，而不是退回默认。
        ctx.state = 0;
        clock.update(&ctx);
        assert_eq!(clock.time_signature(), Some((7, 8)));
        // 有效但非正：拒绝（0/0 与负数都不是合法拍号）。
        ctx.state = 1 << 13;
        for (numerator, denominator) in [(0, 8), (7, 0), (-4, 4)] {
            ctx.time_sig_numerator = numerator;
            ctx.time_sig_denominator = denominator;
            clock.update(&ctx);
            assert_eq!(
                clock.time_signature(),
                Some((7, 8)),
                "{numerator}/{denominator}"
            );
        }
    }

    /// **不变量：定位请求在被宿主确认之前，读数就停在目标上。**
    ///
    /// 【为什么这条最要紧】它就是"跳过去又被拽回来"的回归：旧实现里前端只能自己
    /// 压住采样，而"压多久"没有任何事实依据 —— 命令回执不等于宿主落地。
    #[test]
    fn a_seek_holds_the_target_until_the_host_reports_it() {
        let clock = TransportClock::default();
        clock.publish_host(12.3, true);
        clock.request_seek(20.0);
        let view = clock.read_view();
        assert_eq!(view.position_sec, 20.0);
        assert_eq!(view.provenance, Provenance::SeekIntent);
        assert!(view.seek_pending);
        assert!(view.playing, "播放中定位不该把播放位弄丢");
        // 宿主还在旧位置：忽略它。
        clock.publish_host(12.4, true);
        assert_eq!(clock.read_view().position_sec, 20.0);
        // 宿主到位：窗口关闭，读数回到宿主。
        clock.publish_host(20.05, true);
        let view = clock.read_view();
        assert_eq!(view.provenance, Provenance::HostSample);
        // f64 的最低一位被"播放位"占着（见 `publish_host` 的打包），因此读回来是
        // 20.049999999999997 —— 1 ULP 级的量化，不是错误。
        assert!((view.position_sec - 20.05).abs() < 1e-9);
        assert_eq!(view.position_sec, f64::from_bits(20.05f64.to_bits() & !1));
        assert!(!view.seek_pending);
    }

    /// **不变量：停止态的定位不会伪造"在播放"。**
    #[test]
    fn a_seek_while_stopped_does_not_fake_playback() {
        let clock = TransportClock::default();
        clock.publish_host(5.0, false);
        clock.request_seek(8.0);
        let view = clock.read_view();
        assert_eq!(view.position_sec, 8.0);
        assert!(!view.playing);
        assert_eq!(view.provenance, Provenance::SeekIntent);
    }

    /// 宿主从不落地时，窗口必须自己关闭并留下读数。
    #[test]
    fn a_pending_seek_expires_and_says_so() {
        let clock = TransportClock::default();
        clock.publish_host(3.0, false);
        clock.request_seek(9.0);
        let view = clock.read_view_at(now_ms() + SEEK_TTL_MS + 50);
        assert!(!view.seek_pending);
        assert_eq!(view.position_sec, 3.0, "到期后回到最后观测位置");
        assert_eq!(view.provenance, Provenance::Stale);
        assert_eq!(clock.diagnostics()["seek_unsettled"], 1);
    }

    /// **不变量：prefetch 块能证明"在播"，但绝不能当位置。**
    #[test]
    fn prefetch_blocks_raise_the_playing_bit_but_never_the_position() {
        let clock = TransportClock::default();
        let t0 = now_ms();
        clock.observe_prefetch_playing(true);
        let view = clock.read_view_at(t0);
        assert!(view.playing, "prefetch 的 kPlaying 是可信的宿主事实");
        assert_eq!(view.position_sec, 0.0, "渲染进度不得充当播放位置");
        assert_eq!(view.provenance, Provenance::Stale);
        // 块停了 ⇒ 播放位也必须停（否则光标会在停播后继续前漂）。
        assert!(!clock
            .read_view_at(t0 + PREFETCH_PLAYING_STALE_MS + 1)
            .playing);
    }

    /// 每个读数都要能说出自己是谁给的。
    #[test]
    fn the_transport_view_names_its_source() {
        let clock = TransportClock::default();
        assert_eq!(clock.read_view().provenance, Provenance::Stale);
        clock.publish_host(1.0, true);
        assert_eq!(clock.read_view().provenance, Provenance::HostSample);
        clock.request_seek(2.0);
        assert_eq!(clock.read_view().provenance, Provenance::SeekIntent);
    }

    /// 采样失败必须留下计数，而不是被 if let Ok 吞掉。
    #[test]
    fn a_failed_host_sample_is_counted() {
        let clock = TransportClock::default();
        clock.note_sample_failure("host query authorization revoked");
        clock.note_sample_failure("host query authorization revoked");
        assert_eq!(clock.diagnostics()["sample_failures"], 2);
    }
}
