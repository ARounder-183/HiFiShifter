//! 实时回调发布的宿主播放时钟；GUI读近似最新块，process只执行原子写入。
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU64, Ordering};

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

/// 宿主读数在**音频块计数**上的生存期（约 8 块 ≈ 85 ms @ 48 kHz / 512 帧）。
///
/// 【为什么还要一条块计数的判据】音频线程不能读墙钟（实时纪律），所以它的进度
/// 只能用自己的块序号表达。宿主采样时记下当时的块号；块号走得太远就说明宿主
/// 采样已经落后于音频，此时以音频为准。
const HOST_STALE_BLOCKS: u64 = 8;

/// 进程启动以来的毫秒数。只用于新鲜度判定，不参与任何音视频时序。
fn now_ms() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis() as u64
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
    /// 仅非实时actor查询，JSON分配绝不进入process/setProcessing。
    pub fn diagnostics(&self) -> serde_json::Value {
        serde_json::json!({"realtime_callbacks":self.realtime.load(Ordering::Relaxed),
            "host_position_authority":self.host_valid.load(Ordering::Acquire),
            // 【为什么要报"当前用的是谁"】"光标不跟手"的两种成因（宿主采样停摆 /
            // 音频块停摆）在这两个计数上完全分得开，不必让用户复述现象。
            "host_reading_fresh":self.host_is_fresh(),
            "audio_serial":self.audio_serial.load(Ordering::Acquire),
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
            self.received.store(true, Ordering::Release);
            // 块序号只递增；它是"音频走到了第几块"的唯一表达（见 `HOST_STALE_BLOCKS`）。
            self.audio_serial.fetch_add(1, Ordering::Release);
            self.position.store(
                (context.project_time_samples as f64 / context.sample_rate).to_bits(),
                Ordering::Release,
            );
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
    /// 【为什么不能用 `read()` 的返回值代替】`read()` 在"宿主在 0 秒处没播放"与"宿主
    /// 从未送过上下文"两种情况下都给 `(0.0, false)`。对展示来说两者一样（都显示 0），
    /// 但对**让路判定**完全不同：前者可以用来判断"宿主动没动"，后者不能 —— 把"还没
    /// 收到读数"当成"宿主没动"，会让探针每次都误判，于是每次都白白多等一个探针窗口。
    pub fn has_reading(&self) -> bool {
        self.host_valid.load(Ordering::Acquire) || self.received.load(Ordering::Acquire)
    }
    /// 宿主读数是否仍然**新鲜**（两次判据同时成立，见两个常量）。
    ///
    /// 【为什么不是"曾经读到过"】`host_valid` 只置真、从不复位。用它当权威，
    /// 一次成功的采样就会把音频线程的实时读数永久压住 —— 光标停住、偏移越拉越大，
    /// 而宿主偶尔来一次回调时又跳回真实位置（"随机闪"）。
    fn host_is_fresh(&self) -> bool {
        if !self.host_valid.load(Ordering::Acquire) {
            return false;
        }
        let age_ms = now_ms().saturating_sub(self.host_at_ms.load(Ordering::Acquire));
        if age_ms > HOST_READING_STALE_MS {
            return false;
        }
        let blocks = self
            .audio_serial
            .load(Ordering::Acquire)
            .saturating_sub(self.host_audio_serial.load(Ordering::Acquire));
        blocks <= HOST_STALE_BLOCKS
    }

    /// 多个 renderer 可能并行发布同一文档块；展示容忍一块误差，不参与 DSP 寻址。
    ///
    /// 【为什么是"新鲜者胜"而不是"宿主优先"】两个来源回答的是不同的问题：宿主采样
    /// 给的是**暂停时**的光标位置（那里只有宿主知道），音频线程给的是**播放中**的
    /// 连续位置。两者天然有 1~2 帧相位差，而任一方都可能停摆 —— 用布尔闩锁选边，
    /// 就等于让"最后一次成功"永远赢。这里按新鲜度仲裁：谁在最近一轮里还在推进，
    /// 就用谁。
    pub fn read(&self) -> (f64, bool) {
        if self.host_is_fresh() {
            let pose = self.host_pose.load(Ordering::Acquire);
            return (f64::from_bits(pose & !1), pose & 1 != 0);
        }
        (
            f64::from_bits(self.position.load(Ordering::Acquire)),
            self.playing.load(Ordering::Acquire),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// 真实工程时钟不受预取提前位置或其它clip停止覆盖，后退seek/loop仍能发布。
    #[test]
    fn actual_host_position_survives_prefetch_noise_and_accepts_backward_seeks() {
        let clock = TransportClock::default();
        clock.publish_host(1.25, true);
        clock.update(&crate::audio_abi::ProcessContext {
            state: 1 << 1,
            sample_rate: 44100.,
            project_time_samples: 441000,
            ..Default::default()
        });
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
    /// `host_valid` 为真时无条件返回宿主值，于是停掉的采样会把播放头钉死在最后
    /// 一次成功的位置上，直到下一次宿主回调把它"闪"回去。
    #[test]
    fn a_stalled_host_sample_yields_to_the_audio_thread() {
        let clock = TransportClock::default();
        clock.publish_host(1.0, true);
        for block in 1..=12u64 {
            clock.update(&crate::audio_abi::ProcessContext {
                state: 1 << 1,
                sample_rate: 44100.,
                project_time_samples: 44100 * block as i64,
                ..Default::default()
            });
        }
        // 12 块 > HOST_STALE_BLOCKS ⇒ 宿主读数作废，取音频线程的块位置。
        assert_eq!(clock.read(), (12.0, true));
        assert_eq!(clock.diagnostics()["host_reading_fresh"], false);
    }

    /// 但一个**新鲜**的宿主采样必须立刻重新成为权威 —— 暂停时的光标位置只有宿主知道。
    #[test]
    fn a_fresh_host_sample_wins_back_immediately() {
        let clock = TransportClock::default();
        clock.publish_host(1.0, true);
        for block in 1..=12u64 {
            clock.update(&crate::audio_abi::ProcessContext {
                state: 1 << 1,
                sample_rate: 44100.,
                project_time_samples: 44100 * block as i64,
                ..Default::default()
            });
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
        assert!(clock.diagnostics()["host_reading_fresh"].as_bool().unwrap_or(false));
    }

    /// 诊断真实区分三模式/多writer/回跳/瞬时停止，但本身不能改变GUI播放时间。
    #[test]
    fn diagnostics_count_sources_without_changing_the_transport() {
        let clock = TransportClock::default();
        let mut context = crate::audio_abi::ProcessContext {
            state: (1 << 1) | (1 << 8) | (1 << 17),
            sample_rate: 44100.,
            project_time_samples: 44100,
            system_time: 123456,
            continuous_time_samples: 44100,
            ..Default::default()
        };
        clock.observe(&context, 1, 2, 0);
        context.project_time_samples = 88200;
        clock.observe(&context, 2, 1, 1);
        context.project_time_samples = 44100;
        clock.observe(&context, 1, 2, 0);
        clock.observe(&context, 3, 1, 2);
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
    }
    #[test]
    fn only_valid_host_tempo_updates_bpm() {
        let clock = TransportClock::default();
        let mut context = crate::audio_abi::ProcessContext {
            sample_rate: 44100.,
            tempo: 150.,
            ..Default::default()
        };
        clock.update(&context);
        assert_eq!(clock.tempo(), None);
        context.state = 1 << 10;
        clock.update(&context);
        assert_eq!(clock.tempo(), Some(150.));
        for tempo in [f64::NAN, 0., -1.] {
            context.tempo = tempo;
            clock.update(&context);
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
        let mut context = crate::audio_abi::ProcessContext {
            sample_rate: 44100.,
            time_sig_numerator: 7,
            time_sig_denominator: 8,
            ..Default::default()
        };
        // 无有效位：不采纳。
        clock.update(&context);
        assert_eq!(clock.time_signature(), None);
        // 有效位成立：采纳。
        context.state = 1 << 13;
        clock.update(&context);
        assert_eq!(clock.time_signature(), Some((7, 8)));
        // 后续块没有有效位：保持上一次读数，而不是退回默认。
        context.state = 0;
        clock.update(&context);
        assert_eq!(clock.time_signature(), Some((7, 8)));
        // 有效但非正：拒绝（0/0 与负数都不是合法拍号）。
        context.state = 1 << 13;
        for (numerator, denominator) in [(0, 8), (7, 0), (-4, 4)] {
            context.time_sig_numerator = numerator;
            context.time_sig_denominator = denominator;
            clock.update(&context);
            assert_eq!(
                clock.time_signature(),
                Some((7, 8)),
                "{numerator}/{denominator}"
            );
        }
    }
}
