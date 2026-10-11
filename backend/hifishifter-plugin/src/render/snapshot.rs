//! 不可变播放快照；实时读者仅进入原子读区，退役缓冲由非实时发布线程安全回收。

use super::source::SourcePcm;
use crate::ara::AraPlaybackRegion;
use crate::audio_abi::AudioBusBuffers;
use std::collections::HashMap;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// `reclaim` 等待 `registering` 归零的自旋上限（见 [`SnapshotPublisher::reclaim`]）。
/// 该窗口只有几条指令；仅在非实时线程使用，不阻塞音频线程。
const RECLAIM_SPIN_LIMIT: u32 = 4096;

#[derive(Debug)]
pub(crate) struct PlaybackSnapshot {
    pub sample_rate: u32,
    pub origin_sample: i64,
    pub left: Vec<f32>,
    /// worker逐bit校验相同后可省去右平面；空右平面表示读取左平面，不是右侧静音。
    pub right: Vec<f32>,
    pub _reservation: Option<super::budget::Reservation>,
}

impl PlaybackSnapshot {
    /// 只在非实时准备/发布线程执行。真正立体声与有符号零不同的平面绝不合并。
    pub fn compact_mono(&mut self) -> Result<(), SnapshotError> {
        if self.right.is_empty() {
            return Ok(());
        }
        if self.left.len() != self.right.len() {
            return Err(SnapshotError::InvalidGeometry);
        }
        if !self
            .left
            .iter()
            .zip(&self.right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
        {
            return Ok(());
        }
        let bytes = self
            .right
            .len()
            .checked_mul(4)
            .ok_or(SnapshotError::BudgetExceeded)?;
        let refund = match &mut self._reservation {
            Some(reservation) => Some(
                reservation
                    .split_off(bytes)
                    .ok_or(SnapshotError::BudgetExceeded)?,
            ),
            None => None,
        };
        // 先释放实际缓冲再退额度，另一worker不能借已经退还但尚未释放的额度越过峰值。
        drop(std::mem::take(&mut self.right));
        drop(refund);
        Ok(())
    }
    /// 实际平面占用；mono归并和退役快照均按存储字节而不是固定stereo倍率计费。
    fn storage_bytes(&self) -> Option<usize> {
        self.left
            .len()
            .checked_add(self.right.len())?
            .checked_mul(4)
    }
}

/// 整批结果在下一renderer开始分配前归并，不能等全部大快照堆齐后才节省预算。
pub(crate) fn compact_prepared(snapshots: &mut [PlaybackSnapshot]) -> Result<(), String> {
    for snapshot in snapshots {
        snapshot
            .compact_mono()
            .map_err(|error| format!("invalid prepared snapshot: {error:?}"))?;
    }
    Ok(())
}

#[derive(Debug, PartialEq)]
pub(crate) enum SnapshotError {
    InvalidGeometry,
    MissingSource,
    UnsupportedTransform,
    BudgetExceeded,
}

/// 共享几何/授权PCM校验；允许拉伸只用于真正kernel渲染入口，plain混音仍拒绝变速。
pub(crate) fn validate_regions(
    regions: &[AraPlaybackRegion],
    sources: &HashMap<String, Arc<SourcePcm>>,
    sample_rate: u32,
    allow_stretch: bool,
) -> Result<(i64, usize), SnapshotError> {
    validate_regions_with_fades(regions, sources, sample_rate, allow_stretch, false)
}
/// 原kernel已实现委托的包络；plain路径仍必须拒绝content fade，不暗中重复宿主处理。
pub(crate) fn validate_regions_with_fades(
    regions: &[AraPlaybackRegion],
    sources: &HashMap<String, Arc<SourcePcm>>,
    sample_rate: u32,
    allow_stretch: bool,
    allow_fades: bool,
) -> Result<(i64, usize), SnapshotError> {
    if sample_rate == 0 {
        return Err(SnapshotError::InvalidGeometry);
    }
    let mut origin = i64::MAX;
    let mut end = i64::MIN;
    for region in regions {
        let geometry = [
            region.start_in_modification_time,
            region.duration_in_modification_time,
            region.start_in_playback_time,
            region.duration_in_playback_time,
        ];
        if geometry.iter().any(|value| !value.is_finite())
            || region.start_in_modification_time < 0.0
            || region.duration_in_modification_time <= 0.0
            || region.duration_in_playback_time <= 0.0
        {
            return Err(SnapshotError::InvalidGeometry);
        }
        if (!allow_stretch
            && (region.duration_in_modification_time - region.duration_in_playback_time).abs()
                > 1e-9)
            || !allow_fades
                && (region.has_content_based_fade_at_head || region.has_content_based_fade_at_tail)
        {
            return Err(SnapshotError::UnsupportedTransform);
        }
        let source = sources
            .get(&region.audio_source_persistent_id)
            .ok_or(SnapshotError::MissingSource)?;
        if source.sample_rate == 0
            || !(1..=2).contains(&source.planes.len())
            || source.planes[0].is_empty()
            || source
                .planes
                .iter()
                .any(|plane| plane.len() != source.planes[0].len())
        {
            return Err(SnapshotError::InvalidGeometry);
        }
        // REAPER的秒域item尾可按源采样网格向上取整一帧（现场ka/n恰好如此）。
        // 按同一源采样网格round后最多允许一帧零尾。倍率跨f32/f64会留下很小的分数帧，
        // 不能仅用f64 epsilon拒绝合法的网格边界；不改宿主几何，不绕回源头。
        let available = source.planes[0].len() as f64;
        let source_start = region.start_in_modification_time * source.sample_rate as f64;
        let source_end = (region.start_in_modification_time + region.duration_in_modification_time)
            * source.sample_rate as f64;
        if source_start >= available || source_end.round() > available + 1. {
            return Err(SnapshotError::InvalidGeometry);
        }
        let start = region.start_in_playback_time * sample_rate as f64;
        let stop =
            (region.start_in_playback_time + region.duration_in_playback_time) * sample_rate as f64;
        if start.abs() >= i64::MAX as f64 || stop.abs() >= i64::MAX as f64 {
            return Err(SnapshotError::InvalidGeometry);
        }
        origin = origin.min(start.round() as i64);
        end = end.max(stop.round() as i64);
    }
    if regions.is_empty() {
        return Ok((0, 0));
    }
    let frames = end
        .checked_sub(origin)
        .and_then(|span| usize::try_from(span).ok())
        .ok_or(SnapshotError::BudgetExceeded)?;
    if frames
        .checked_mul(8)
        .is_none_or(|bytes| bytes > super::budget::global_budget().limit())
    {
        return Err(SnapshotError::BudgetExceeded);
    }
    Ok((origin, frames))
}

/// 普通/裁切的纯PCM混音不能用重采样冒充保调拉伸；后者由原kernel管线承担。
pub(crate) fn mix_plain_regions(
    regions: &[AraPlaybackRegion],
    sources: &HashMap<String, Arc<SourcePcm>>,
    sample_rate: u32,
) -> Result<PlaybackSnapshot, SnapshotError> {
    let (origin, frames) = validate_regions(regions, sources, sample_rate, false)?;
    if regions.is_empty() {
        return Ok(PlaybackSnapshot {
            sample_rate,
            origin_sample: 0,
            left: vec![],
            right: vec![],
            _reservation: None,
        });
    }
    let reservation = super::budget::global_budget()
        .reserve(frames * 8)
        .ok_or(SnapshotError::BudgetExceeded)?;
    let mut snapshot = PlaybackSnapshot {
        sample_rate,
        origin_sample: origin,
        left: vec![0.0; frames],
        right: vec![0.0; frames],
        _reservation: Some(reservation),
    };
    for region in regions {
        let source = &sources[&region.audio_source_persistent_id];
        let begin = (region.start_in_playback_time * sample_rate as f64).round() as i64;
        let finish = ((region.start_in_playback_time + region.duration_in_playback_time)
            * sample_rate as f64)
            .round() as i64;
        for sample in begin..finish {
            let position = (region.start_in_modification_time + sample as f64 / sample_rate as f64
                - region.start_in_playback_time)
                * source.sample_rate as f64;
            let index = position.max(0.0).floor() as usize;
            let fraction = (position - index as f64) as f32;
            let output = (sample - origin) as usize;
            for (channel, target) in [&mut snapshot.left, &mut snapshot.right]
                .into_iter()
                .enumerate()
            {
                let plane = &source.planes[channel.min(source.planes.len() - 1)];
                let first = index.min(plane.len() - 1);
                let second = (first + 1).min(plane.len() - 1);
                target[output] += plane[first] + (plane[second] - plane[first]) * fraction;
            }
        }
    }
    Ok(snapshot)
}

/// 已发布快照 + **它自己的**读者计数。
///
/// 【为什么读者计数必须逐快照，而不是一个全局计数】退役快照只有在"读过**它**的读者"
/// 全部退出后才能释放。全局计数做不到这一点：播放时音频线程几乎总在读**当前**快照，
/// 全局计数因此恒非零，退役快照永远收不回来 ⇒ 预算被撑满 ⇒ 后续 `publish` 直接
/// `BudgetExceeded`，表现为"播放中改不动任何东西"。逐快照计数把"读当前"与"读退役"
/// 分开：退役快照的计数一归零就能回收，不必等整个音频线程静默。
struct RetainedSnapshot {
    readers: AtomicUsize,
    snapshot: PlaybackSnapshot,
}

pub(crate) struct SnapshotPublisher {
    retained: Mutex<(Vec<Box<RetainedSnapshot>>, usize)>,
    current: AtomicPtr<RetainedSnapshot>,
    /// 正在"载入当前指针并登记到该快照"的读者数。
    ///
    /// 【为什么需要它】`enter` 是"载入指针 → 递增该快照的读者数"两步，中间有一个极短
    /// 窗口。回收若只看逐快照计数，就可能在这个窗口里判定"无人读"而释放一个刚被载入
    /// 指针的快照。`registering` 把窗口封住：`reclaim` 只在它为 0 时才判定可释放。
    /// 它只覆盖**几条指令**（不是整个读取区），所以等待它归零是真正短暂的。
    registering: AtomicUsize,
    limit: usize,
    pub misses: AtomicU64,
}
impl Default for SnapshotPublisher {
    fn default() -> Self {
        Self::new(512 * 1024 * 1024)
    }
}

impl SnapshotPublisher {
    /// IPC/诊断线程有界读当前实际发布的PCM；不复制音频、不调用宿主、不在实时回调执行。
    pub(crate) fn diagnose(&self, ranges: &[(String, f64, f64)]) -> serde_json::Value {
        let read = self.enter();
        // SAFETY: 与 copy_block 相同读区契约；本读区登记的快照在返回前不会被释放。
        let Some(snapshot) = read.snapshot() else {
            return serde_json::json!({"ready":false,"misses":self.misses.load(Ordering::Relaxed)});
        };
        let clips=ranges.iter().take(8).map(|(id,start,duration)| {
            let lo=((*start*snapshot.sample_rate as f64).round() as i64).checked_sub(snapshot.origin_sample);
            let frames=(*duration*snapshot.sample_rate as f64).round().max(0.).min(snapshot.sample_rate as f64*2.) as usize;
            let mut energy=0.;let mut peak=0_f32;let mut nonzero=0;let mut covered=0;
            for frame in 0..frames {
                let Some(index)=lo.and_then(|lo|lo.checked_add(frame as i64)).and_then(|n|usize::try_from(n).ok()) else {continue;};
                let Some(left)=snapshot.left.get(index) else {continue;};
                let right=snapshot.right.get(index).unwrap_or(left);covered+=1;
                peak=peak.max(left.abs()).max(right.abs());energy+=(*left as f64).powi(2)+(*right as f64).powi(2);
                if *left!=0.||*right!=0. {nonzero+=1;}
            }
            serde_json::json!({"clip_id":id,"covered_frames":covered,"inspected_frames":frames,"nonzero_frames":nonzero,
                "peak":peak,"rms":(energy/(covered.max(1)*2) as f64).sqrt()})
        }).collect::<Vec<_>>();
        serde_json::json!({"ready":true,"sample_rate":snapshot.sample_rate,"origin_sample":snapshot.origin_sample,
            "frames":snapshot.left.len(),"misses":self.misses.load(Ordering::Relaxed),"clips":clips})
    }
    /// 只读原子指针，不解引用/持锁，用于离线setup在版本事务内核对两输出率。
    pub fn is_ready(&self) -> bool {
        !self.current.load(Ordering::SeqCst).is_null()
    }
    /// 发布前在非实时线程预检，两个采样率必须都可容纳才替换编辑状态。
    pub fn has_capacity(&self, snapshot: &PlaybackSnapshot) -> bool {
        let mut retained = self.retained.lock().unwrap();
        self.reclaim(&mut retained);
        snapshot
            .storage_bytes()
            .and_then(|bytes| retained.1.checked_add(bytes))
            .is_some_and(|n| n <= self.limit)
    }
    /// 预算包含所有退役快照，避免原子替换时在音频线程析构大缓冲。
    pub fn new(limit: usize) -> Self {
        Self {
            retained: Mutex::new((Vec::new(), 0)),
            current: AtomicPtr::new(std::ptr::null_mut()),
            registering: AtomicUsize::new(0),
            limit,
            misses: AtomicU64::new(0),
        }
    }
    /// 非实时发布。
    pub fn publish(&self, mut snapshot: PlaybackSnapshot) -> Result<(), SnapshotError> {
        if (!snapshot.right.is_empty() && snapshot.left.len() != snapshot.right.len())
            || snapshot.sample_rate == 0
        {
            return Err(SnapshotError::InvalidGeometry);
        }
        snapshot.compact_mono()?;
        let bytes = snapshot
            .storage_bytes()
            .ok_or(SnapshotError::BudgetExceeded)?;
        let mut retained = self.retained.lock().unwrap();
        self.reclaim(&mut retained);
        let total = retained
            .1
            .checked_add(bytes)
            .filter(|total| *total <= self.limit)
            .ok_or(SnapshotError::BudgetExceeded)?;
        if snapshot._reservation.is_none() {
            snapshot._reservation = Some(
                super::budget::global_budget()
                    .reserve(bytes)
                    .ok_or(SnapshotError::BudgetExceeded)?,
            );
        }
        let mut entry = Box::new(RetainedSnapshot {
            readers: AtomicUsize::new(0),
            snapshot,
        });
        let pointer = &mut *entry as *mut RetainedSnapshot;
        retained.0.push(entry);
        retained.1 = total;
        self.current.store(pointer, Ordering::SeqCst);
        self.reclaim(&mut retained);
        Ok(())
    }
    /// 撤销只切换原子指针，不在音频线程释放存储。
    pub fn clear(&self) {
        self.current.store(std::ptr::null_mut(), Ordering::SeqCst);
    }
    /// 必须持发布锁且仅在非实时线程调用。
    ///
    /// 【回收判据】退役快照只在**读过它的**读者全部退出（逐快照计数归零）后释放，
    /// 而不再等整个音频线程静默 —— 播放时音频线程几乎总在读**当前**快照，用全局计数
    /// 会让退役缓冲永不回收、预算被撑满，表现为"播放中改不动任何东西"。等 `registering`
    /// 归零只是封住 `enter` 的极短窗口（几条指令），不是等一整个读取区。
    ///
    /// 【为什么仍是有界等待】`registering` 只在 `enter` 的两条指令之间非零，正常情况
    /// 一两次自旋即可；万一读者被抢占在窗口内，超限就直接返回、把回收留给下一次调用
    /// （不阻塞、不分配，也不释放任何仍有读者的快照）。
    fn reclaim(&self, retained: &mut (Vec<Box<RetainedSnapshot>>, usize)) {
        let current = self.current.load(Ordering::SeqCst);
        let mut spins = 0u32;
        while self.registering.load(Ordering::SeqCst) != 0 {
            if spins >= RECLAIM_SPIN_LIMIT {
                return;
            }
            spins += 1;
            std::hint::spin_loop();
            if spins.is_multiple_of(64) {
                std::thread::yield_now();
            }
        }
        retained.0.retain(|entry| {
            std::ptr::eq(&**entry, current) || entry.readers.load(Ordering::SeqCst) != 0
        });
        retained.1 = retained
            .0
            .iter()
            // `storage_bytes()` 只在字节数溢出 `usize` 时返回 `None`（实际不可能发生）。
            // 这里**不能** `unwrap` —— 这条路径在每次 `publish`/`has_capacity` 上，
            // 一次 panic 会毒化发布器的锁、把整个音频链路卡死。按"未知即视为已满"
            // 处理：宁可拒绝继续发布，也不要 OOM 或 panic。
            .map(|entry| entry.snapshot.storage_bytes().unwrap_or(usize::MAX))
            .fold(0usize, usize::saturating_add);
    }
    /// 分配下一批音频前回收已退役缓冲，避免全局预算已满时无法进入publish回收。
    pub fn collect_retired(&self) {
        let mut retained = self.retained.lock().unwrap();
        self.reclaim(&mut retained);
    }
    /// 实时guard只增减计数，不分配、不释放、不等待mutex。
    ///
    /// 【顺序】先取当前指针、再递增该快照的读者数，两步之间用 `registering` 标记；
    /// `reclaim` 只在这个标记为 0 时才判定可释放，因此不存在"载入指针后、登记前被回收"。
    fn enter(&self) -> ReadSection {
        self.registering.fetch_add(1, Ordering::SeqCst);
        let pointer = self.current.load(Ordering::SeqCst);
        if !pointer.is_null() {
            // SAFETY: 指针来自 `retained` 中存活的 Box；`registering` 非零期间不会被回收。
            unsafe { (*pointer).readers.fetch_add(1, Ordering::SeqCst) };
        }
        self.registering.fetch_sub(1, Ordering::SeqCst);
        ReadSection { snapshot: pointer }
    }
    /// 按项目绝对 sample 时间读取，seek/loop 不使用累计 playhead。
    ///
    /// 【支持的输出形状】1 或 2 声道。内部快照恒为立体声：立体声输出逐轨取，单声道
    /// 输出取左右**平均**（丢掉右声道是有损的，平均才是正确的下混）。此前只认 2 声道
    /// ⇒ 单声道轨道上 `process` 失败 ⇒ 宿主**静默旁路**插件 —— 而人声轨单声道极常见。
    ///
    /// 【为什么按比例重采样】内部快照只有 44.1/48k 两种率。宿主用别的率（96k/192k
    /// 工程）时，此前 `sample_rate` 不等即判"没有快照"⇒ 整条插件**静默输出静音**。
    /// 这里用线性插值把快照重采样到宿主率：这是**降级路径**（线性插值在高采样率下
    /// 有可闻的高频镜像），但"能听见、略有染色"远好于"什么都听不见"。真正的多率
    /// 渲染管线是独立一期工作（见 `vst3.rs` 的 `setupProcessing` 注释）。
    ///
    /// # Safety
    /// output 为已校验 f32 SDK 缓冲（1 或 2 声道），非空 plane 至少含 frames 个样本。
    pub unsafe fn copy_block(
        &self,
        sample: i64,
        rate: u32,
        output: &mut AudioBusBuffers,
        frames: usize,
    ) -> bool {
        if output.num_channels == 0 || output.num_channels > 2 || output.channel_buffers.is_null() {
            return false;
        }
        let read_guard = self.enter();
        // SAFETY: 本读区登记的快照在 `read_guard` 存活期间不会被回收（逐快照读者计数非零）。
        let snapshot = read_guard.snapshot();
        if snapshot.is_none() {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        let channels = output.num_channels.clamp(0, 2) as usize;
        // 源率/目标率之比：把宿主帧号换算成快照帧号。
        let ratio = snapshot
            .map(|snapshot| snapshot.sample_rate as f64 / rate.max(1) as f64)
            .unwrap_or(1.0);
        let resampling = (ratio - 1.0).abs() > 1e-9;
        // 一个输出声道在给定宿主帧上应取的采样值（含可选线性插值）。
        let read = |channel: usize, offset: usize| -> f32 {
            snapshot
                .and_then(|snapshot| {
                    let local = sample
                        .checked_add(offset as i64)?
                        .checked_sub(snapshot.origin_sample)?;
                    let at = |position: f64| -> Option<f32> {
                        if position < 0.0 {
                            return None;
                        }
                        let index = usize::try_from(position as u64).ok()?;
                        if channel == 0 || snapshot.right.is_empty() {
                            snapshot.left.get(index).copied()
                        } else {
                            snapshot.right.get(index).copied()
                        }
                    };
                    if !resampling {
                        return at(local as f64);
                    }
                    let position = local as f64 * ratio;
                    let base = position.floor();
                    let frac = (position - base) as f32;
                    let a = at(base)?;
                    // 末尾一帧没有后继：保持最后一个采样（不外推、不补零）。
                    let b = at(base + 1.0).unwrap_or(a);
                    Some(a + (b - a) * frac)
                })
                .unwrap_or(0.0)
        };
        let mut silence = (1_u64 << channels) - 1;
        for channel in 0..channels {
            // SAFETY: 调用方已验证 SDK 数组有 `channels` 个指针，允许 inactive null plane。
            let plane = unsafe { *output.channel_buffers.add(channel) };
            if plane.is_null() {
                continue;
            }
            for offset in 0..frames {
                let value = if channels == 1 {
                    // 立体声快照 → 单声道总线：平均，而不是丢掉右声道。
                    (read(0, offset) + read(1, offset)) * 0.5
                } else {
                    read(channel, offset)
                };
                // SAFETY: 宿主提供至少 frames 样本的可写平面。
                unsafe { plane.add(offset).write(value) };
                if value != 0.0 {
                    silence &= !(1 << channel);
                }
            }
        }
        output.silence_flags = silence;
        snapshot.is_some()
    }
}
/// 一个实时读区的 guard：持有它期间，它登记的那份快照不会因回收而释放。
struct ReadSection {
    snapshot: *const RetainedSnapshot,
}
impl ReadSection {
    /// 本读区登记的不可变快照；`None` = 进入读区时没有已发布快照。
    fn snapshot(&self) -> Option<&PlaybackSnapshot> {
        if self.snapshot.is_null() {
            return None;
        }
        // SAFETY: 本 guard 存活期间 `snapshot` 指向的 Box 因读者计数非零不会被回收。
        Some(unsafe { &(*self.snapshot).snapshot })
    }
}
impl Drop for ReadSection {
    fn drop(&mut self) {
        if !self.snapshot.is_null() {
            // SAFETY: `enter` 登记过该快照；guard 只析构一次。
            unsafe { (*self.snapshot).readers.fetch_sub(1, Ordering::SeqCst) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// 一帧源尾允许有界补零；超过一帧或全窗口已在EOF之外仍必须拒绝。
    #[test]
    fn source_eof_rounding_does_not_allow_real_out_of_bounds_windows() {
        let mut r = region();
        r.start_in_modification_time = 0.;
        r.duration_in_modification_time = 5. / 44100.;
        r.start_in_playback_time = 0.;
        r.duration_in_playback_time = 5. / 44100.;
        let sources = HashMap::from([(
            r.audio_source_persistent_id.clone(),
            Arc::new(SourcePcm {
                sample_rate: 44100,
                planes: vec![vec![0.1; 4]],
                version: 0,
                _reservation: None,
            }),
        )]);
        assert!(validate_regions(&[r.clone()], &sources, 44100, false).is_ok());
        r.duration_in_modification_time = 5.5 / 44100.;
        r.duration_in_playback_time = r.duration_in_modification_time;
        assert_eq!(
            validate_regions(&[r.clone()], &sources, 44100, false),
            Err(SnapshotError::InvalidGeometry)
        );
        r.start_in_modification_time = 4. / 44100.;
        r.duration_in_modification_time = 0.5 / 44100.;
        r.duration_in_playback_time = 0.5 / 44100.;
        assert_eq!(
            validate_regions(&[r], &sources, 44100, false),
            Err(SnapshotError::InvalidGeometry)
        );
    }

    /// 归并只减少相同平面存储，stereo双侧仍逐bit读取；有符号零和真正不同声道不合并。
    #[test]
    fn mono_compaction_refunds_duplicate_pcm_and_preserves_stereo_samples() {
        let budget = super::super::budget::global_budget();
        let baseline = budget.used();
        let mut snapshot = PlaybackSnapshot {
            sample_rate: 4,
            origin_sample: 0,
            left: vec![0.125, -0.25],
            right: vec![0.125, -0.25],
            _reservation: Some(budget.reserve(16).unwrap()),
        };
        snapshot.compact_mono().unwrap();
        assert!(snapshot.right.is_empty());
        assert_eq!(budget.used(), baseline + 8);
        snapshot.compact_mono().unwrap();
        assert_eq!(budget.used(), baseline + 8);
        let publisher = SnapshotPublisher::new(16);
        publisher.publish(snapshot).unwrap();
        let mut left = [9_f32; 3];
        let mut right = [9_f32; 3];
        let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut bus = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: planes.as_mut_ptr(),
        };
        // SAFETY: 两平面均提供3帧写入范围；尾帧位于快照之外，必须补零。
        assert!(unsafe { publisher.copy_block(0, 4, &mut bus, 3) });
        assert_eq!(left, [0.125, -0.25, 0.]);
        assert_eq!(right, left);
        drop(publisher);
        assert_eq!(budget.used(), baseline);
        let mut stereo = PlaybackSnapshot {
            sample_rate: 4,
            origin_sample: 0,
            left: vec![0., 1.],
            right: vec![-0., 1.],
            _reservation: None,
        };
        stereo.compact_mono().unwrap();
        assert_eq!(stereo.right[0].to_bits(), (-0_f32).to_bits());
        stereo.right[0] = 0.5;
        stereo.compact_mono().unwrap();
        assert_eq!(stereo.right, [0.5, 1.]);
    }

    fn region() -> AraPlaybackRegion {
        crate::ara::ara_document_from_json(include_str!(
            "../../tests/fixtures/ara-model.reaper.json"
        ))
        .unwrap()
        .playback_regions
        .remove(0)
    }

    /// 纯采样率转换与真正的时间拉伸区别开；后者必须明确拒绝。
    #[test]
    fn stereo_resampling_preserves_planes_and_rejects_time_stretch() {
        let mut r = region();
        r.start_in_modification_time = 0.0;
        r.start_in_playback_time = 0.0;
        r.duration_in_modification_time = 1.0;
        r.duration_in_playback_time = 1.0;
        let sources = [(
            r.audio_source_persistent_id.clone(),
            Arc::new(SourcePcm {
                sample_rate: 2,
                planes: vec![vec![0.0, 1.0], vec![1.0, 0.0]],
                version: 0,
                _reservation: None,
            }),
        )]
        .into_iter()
        .collect();
        let snapshot = mix_plain_regions(&[r.clone()], &sources, 4).unwrap();
        assert_eq!(snapshot.left, [0.0, 0.5, 1.0, 1.0]);
        assert_eq!(snapshot.right, [1.0, 0.5, 0.0, 0.0]);
        r.duration_in_playback_time = 0.5;
        assert_eq!(
            mix_plain_regions(&[r], &sources, 4).unwrap_err(),
            SnapshotError::UnsupportedTransform
        );
    }

    /// 拼接三十秒小块并跳回首块，必须等于一次读取的字面输入，不靠 cursor 累计。
    #[test]
    fn thirty_seconds_of_blocks_and_seek_match_the_source_snapshot() {
        let frames = 44100 * 30;
        let publisher = SnapshotPublisher::new(16 * 1024 * 1024);
        let source = (0..frames)
            .map(|index| (index % 1000) as f32 / 2000.0)
            .collect::<Vec<_>>();
        publisher
            .publish(PlaybackSnapshot {
                sample_rate: 44100,
                origin_sample: 0,
                left: source.clone(),
                right: source.clone(),
                _reservation: None,
            })
            .unwrap();
        let mut left = [0.0_f32; 512];
        let mut right = [0.0_f32; 512];
        let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut bus = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: planes.as_mut_ptr(),
        };
        for start in (0..frames).step_by(512) {
            let count = (frames - start).min(512);
            // SAFETY: 稳定 stereo 缓冲最多读取512帧。
            assert!(unsafe { publisher.copy_block(start as i64, 44100, &mut bus, count) });
            assert_eq!(&left[..count], &source[start..start + count]);
            assert_eq!(&right[..count], &source[start..start + count]);
        }
        // SAFETY: seek 后同一缓冲仍合法。
        assert!(unsafe { publisher.copy_block(0, 44100, &mut bus, 512) });
        assert_eq!(left, source[..512]);
    }
    /// 手算源第 1 帧起两帧放到工程第 2 帧，验证裁切、位置和单声道复制。
    #[test]
    fn plain_clip_placement_matches_a_hand_computed_oracle() {
        let mut r = region();
        r.start_in_modification_time = 0.25;
        r.duration_in_modification_time = 0.5;
        r.start_in_playback_time = 0.5;
        r.duration_in_playback_time = 0.5;
        let sources = [(
            r.audio_source_persistent_id.clone(),
            Arc::new(SourcePcm {
                sample_rate: 4,
                planes: vec![vec![0.1, 0.2, 0.3, 0.4]],
                version: 0,
                _reservation: None,
            }),
        )]
        .into_iter()
        .collect();
        let snapshot = mix_plain_regions(&[r], &sources, 4).unwrap();
        assert_eq!(snapshot.origin_sample, 2);
        assert_eq!(snapshot.left, [0.2, 0.3]);
        assert_eq!(snapshot.right, [0.2, 0.3]);
    }

    /// 随机 seek、负时间、尾部补零与撤销必须精确；尾哨兵不得被覆盖。
    #[test]
    fn callback_seeks_zero_pads_and_revokes_without_releasing_storage() {
        let publisher = SnapshotPublisher::new(1024);
        publisher
            .publish(PlaybackSnapshot {
                sample_rate: 4,
                origin_sample: 2,
                left: vec![0.2, 0.3],
                right: vec![0.4, 0.5],
                _reservation: None,
            })
            .unwrap();
        let mut left = [9.0_f32; 4];
        let mut right = [9.0_f32; 4];
        let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut bus = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: planes.as_mut_ptr(),
        };
        // SAFETY: 每个 output plane 至少 3 帧，最后一帧作为哨兵。
        assert!(unsafe { publisher.copy_block(3, 4, &mut bus, 3) });
        assert_eq!(left, [0.3, 0.0, 0.0, 9.0]);
        assert_eq!(right, [0.5, 0.0, 0.0, 9.0]);
        // SAFETY: 缓冲仍然存活。
        assert!(unsafe { publisher.copy_block(-1, 4, &mut bus, 3) });
        assert_eq!(&left[..3], &[0.0, 0.0, 0.0]);
        publisher.clear();
        // SAFETY: 读取空快照应当零输出，不能继续消费旧音频。
        assert!(!unsafe { publisher.copy_block(2, 4, &mut bus, 3) });
        assert_eq!(&left[..3], &[0.0, 0.0, 0.0]);
        assert_eq!(publisher.retained.lock().unwrap().0.len(), 1);
    }

    /// 单声道总线与"宿主率 ≠ 内部率"都必须能出声。
    ///
    /// 【为什么这是回归测试】此前 `copy_block` 要求 `output.num_channels == 2` **且**
    /// `snapshot.sample_rate == rate`，两条不满足就返回 false ⇒ 输出全零。宿主于是看到
    /// 一个"成功加载、但永远静音"的插件：单声道人声轨与 96k 工程都命中这条。
    #[test]
    fn mono_outputs_and_foreign_rates_still_produce_audio() {
        let publisher = SnapshotPublisher::new(1024);
        publisher
            .publish(PlaybackSnapshot {
                sample_rate: 4,
                origin_sample: 0,
                left: vec![0.2, 0.4, 0.6, 0.8],
                right: vec![0.4, 0.8, 1.2, 1.6],
                _reservation: None,
            })
            .unwrap();

        // 单声道输出：左右**平均**（而不是丢掉右声道）。
        let mut mono = [9.0_f32; 3];
        let mut mono_planes = [mono.as_mut_ptr()];
        let mut mono_bus = AudioBusBuffers {
            num_channels: 1,
            silence_flags: 0,
            channel_buffers: mono_planes.as_mut_ptr(),
        };
        // SAFETY: 单个 plane 至少 3 帧。
        assert!(unsafe { publisher.copy_block(0, 4, &mut mono_bus, 3) });
        let close = |actual: [f32; 3], expected: [f32; 3]| {
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() < 1e-6)
        };
        assert!(close(mono, [0.3, 0.6, 0.9]), "{mono:?}");

        // 宿主 8 帧/秒、快照 4 帧/秒 ⇒ 比例 0.5，线性插值取中间值。
        let mut left = [9.0_f32; 3];
        let mut right = [9.0_f32; 3];
        let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut bus = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: planes.as_mut_ptr(),
        };
        // SAFETY: 两个 plane 各至少 3 帧。
        assert!(unsafe { publisher.copy_block(0, 8, &mut bus, 3) });
        // 位置 0.0 / 0.5 / 1.0 → 0.2 / 0.3 / 0.4。
        assert!(close(left, [0.2, 0.3, 0.4]), "{left:?}");
        assert!(close(right, [0.4, 0.6, 0.8]), "{right:?}");
    }

    /// 活跃读者期间退役快照必须保留；失败保留旧音频，不能将失败伪报成已应用。
    #[test]
    fn snapshot_budget_counts_retired_buffers() {
        let publisher = SnapshotPublisher::new(16);
        let snapshot = || PlaybackSnapshot {
            sample_rate: 4,
            origin_sample: 0,
            left: vec![1.0, 2.0],
            right: vec![3.0, 4.0],
            _reservation: None,
        };
        publisher.publish(snapshot()).unwrap();
        let read = publisher.enter();
        assert_eq!(
            publisher.publish(snapshot()),
            Err(SnapshotError::BudgetExceeded)
        );
        assert!(!publisher.current.load(Ordering::SeqCst).is_null());
        drop(read);
    }
    /// 连续自动编辑没有读者时只保留当前快照，不能因为累计历史永远耗尽预算。
    #[test]
    fn repeated_automatic_publication_reclaims_retired_buffers_off_audio_thread() {
        let publisher = SnapshotPublisher::new(32);
        for number in 0..1000 {
            publisher
                .publish(PlaybackSnapshot {
                    sample_rate: 4,
                    origin_sample: 0,
                    left: vec![number as f32; 2],
                    right: vec![number as f32; 2],
                    _reservation: None,
                })
                .unwrap();
            assert_eq!(publisher.retained.lock().unwrap().0.len(), 1);
            assert_eq!(publisher.retained.lock().unwrap().1, 8);
        }
    }
    /// **读"当前"快照不能阻止回收已退役的那一份。**
    ///
    /// 【为什么这是回归测试】此前用**全局**读者计数：播放时音频线程几乎总在读当前快照，
    /// 计数恒非零 ⇒ 退役缓冲永不回收 ⇒ 预算撑满后 `publish` 直接 `BudgetExceeded`，
    /// 用户表现为"播放中改不动任何东西"。逐快照计数把"读当前"与"读退役"分开，退役的那
    /// 一份只要没人读就能立刻回收。这里把预算压到两份，读者横跨两次发布：旧实现第二次
    /// 发布就必然预算耗尽。
    #[test]
    fn a_reader_of_the_current_snapshot_does_not_block_reclaiming_the_retired_one() {
        let publisher = SnapshotPublisher::new(32);
        let snapshot = || PlaybackSnapshot {
            sample_rate: 4,
            origin_sample: 0,
            // 左右不同 ⇒ 不触发 compact_mono，每份占 16 字节，预算 32 = 两份。
            left: vec![1.; 2],
            right: vec![2.; 2],
            _reservation: None,
        };
        publisher.publish(snapshot()).unwrap();
        // r1 只读 A；A 退役后因它而被保留。
        let r1 = publisher.enter();
        publisher.publish(snapshot()).unwrap();
        drop(r1);
        // r2 读**当前**的 B。它不该阻止"已无人读的 A"被回收 —— 旧实现用全局计数，
        // 这里会因 r2 而放弃回收，预算（=两份）随即被撑满。
        let r2 = publisher.enter();
        publisher.publish(snapshot()).unwrap();
        assert_eq!(
            publisher.retained.lock().unwrap().0.len(),
            2,
            "只应留下正被读的 B 与当前 C"
        );
        drop(r2);
        publisher.collect_retired();
        assert_eq!(publisher.retained.lock().unwrap().0.len(), 1);
    }

    /// 一个读者横跨发布时旧快照保持，退出后下一次非实时预检安全回收。
    #[test]
    fn active_reader_delays_reclamation_until_next_non_realtime_boundary() {
        let publisher = SnapshotPublisher::new(64);
        let snapshot = || PlaybackSnapshot {
            sample_rate: 4,
            origin_sample: 0,
            left: vec![1.; 2],
            right: vec![1.; 2],
            _reservation: None,
        };
        publisher.publish(snapshot()).unwrap();
        let read = publisher.enter();
        publisher.publish(snapshot()).unwrap();
        assert_eq!(publisher.retained.lock().unwrap().0.len(), 2);
        drop(read);
        assert!(publisher.has_capacity(&snapshot()));
        assert_eq!(publisher.retained.lock().unwrap().0.len(), 1);
    }
}
