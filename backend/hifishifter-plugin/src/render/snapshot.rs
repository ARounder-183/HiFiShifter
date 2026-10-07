//! 不可变播放快照；实时读者仅进入原子读区，退役缓冲由非实时发布线程安全回收。

use super::source::SourcePcm;
use crate::ara::AraPlaybackRegion;
use crate::audio_abi::AudioBusBuffers;
use std::collections::HashMap;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

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

pub(crate) struct SnapshotPublisher {
    retained: Mutex<(Vec<Box<PlaybackSnapshot>>, usize)>,
    current: AtomicPtr<PlaybackSnapshot>,
    readers: AtomicUsize,
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
        let _read = self.enter();
        let pointer = self.current.load(Ordering::SeqCst);
        if pointer.is_null() {
            return serde_json::json!({"ready":false,"misses":self.misses.load(Ordering::Relaxed)});
        }
        // SAFETY: 与copy_block相同读区契约，退休缓冲在本方法返回前不会释放。
        let snapshot = unsafe { &*pointer };
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
            readers: AtomicUsize::new(0),
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
        let mut snapshot = Box::new(snapshot);
        let pointer = &mut *snapshot as *mut PlaybackSnapshot;
        retained.0.push(snapshot);
        retained.1 = total;
        self.current.store(pointer, Ordering::SeqCst);
        self.reclaim(&mut retained);
        Ok(())
    }
    /// 撤销只切换原子指针，不在音频线程释放存储。
    pub fn clear(&self) {
        self.current.store(std::ptr::null_mut(), Ordering::SeqCst);
    }
    /// 必须持发布锁且仅在非实时线程调用。SeqCst保证旧指针读者先登记；新的读者只能
    /// 取得当前指针，故readers=0时可释放所有非current对象，不在音频线程析构大缓冲。
    fn reclaim(&self, retained: &mut (Vec<Box<PlaybackSnapshot>>, usize)) {
        let current = self.current.load(Ordering::SeqCst);
        if self.readers.load(Ordering::SeqCst) != 0 {
            return;
        }
        retained
            .0
            .retain(|snapshot| std::ptr::eq(&**snapshot, current));
        retained.1 = retained
            .0
            .iter()
            .map(|snapshot| snapshot.storage_bytes().unwrap())
            .sum();
    }
    /// 分配下一批音频前回收已退役缓冲，避免全局预算已满时无法进入publish回收。
    pub fn collect_retired(&self) {
        let mut retained = self.retained.lock().unwrap();
        self.reclaim(&mut retained);
    }
    /// 实时guard只增减计数，不分配、不释放、不等待mutex。
    fn enter(&self) -> ReadSection<'_> {
        self.readers.fetch_add(1, Ordering::SeqCst);
        ReadSection(&self.readers)
    }
    /// 按项目绝对 sample 时间读取，seek/loop 不使用累计 playhead。
    /// # Safety
    /// output 为已校验 stereo f32 SDK 缓冲，非空 plane 至少含 frames 个样本。
    pub unsafe fn copy_block(
        &self,
        sample: i64,
        rate: u32,
        output: &mut AudioBusBuffers,
        frames: usize,
    ) -> bool {
        if output.num_channels != 2 || output.channel_buffers.is_null() {
            return false;
        }
        let _read = self.enter();
        let pointer = self.current.load(Ordering::SeqCst);
        // SAFETY: 进入SeqCst读区先于加载指针；非实时回收只有readers=0才释放退役Box。
        let snapshot = if pointer.is_null() {
            None
        } else {
            Some(unsafe { &*pointer }).filter(|snapshot| snapshot.sample_rate == rate)
        };
        if snapshot.is_none() {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        let mut silence = 3_u64;
        for channel in 0..2 {
            // SAFETY: 调用方已验证 SDK stereo 数组，允许 inactive null plane。
            let plane = unsafe { *output.channel_buffers.add(channel) };
            if plane.is_null() {
                continue;
            }
            for offset in 0..frames {
                let value = snapshot
                    .and_then(|snapshot| {
                        let local = sample
                            .checked_add(offset as i64)?
                            .checked_sub(snapshot.origin_sample)?;
                        let index = usize::try_from(local).ok()?;
                        if channel == 0 || snapshot.right.is_empty() {
                            snapshot.left.get(index)
                        } else {
                            snapshot.right.get(index)
                        }
                        .copied()
                    })
                    .unwrap_or(0.0);
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
struct ReadSection<'a>(&'a AtomicUsize);
impl Drop for ReadSection<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
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
            "../../../../probe/ara/captures/ara-model.reaper.json"
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
