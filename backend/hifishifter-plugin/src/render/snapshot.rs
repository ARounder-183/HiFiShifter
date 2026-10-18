//! 不可变播放快照；实时读者仅进入原子读区，退役缓冲由非实时发布线程安全回收。

use super::source::SourcePcm;
use crate::ara::AraPlaybackRegion;
use crate::audio_abi::AudioBusBuffers;
use crate::sync::MutexExt;
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
        // 【为什么源窗口越界不再是错误】源窗口超出源本身是**宿主可以合法给出**的：素材
        // 采样率与工程采样率不同时，宿主按工程速率折算修改时长（实测 Cubase 15：48kHz
        // 素材进 44.1kHz 工程，`durationMod=34.610703`，而 34.610703 × 44100 恰好等于源
        // 帧数；乘源速率 48000 则凭空多出 8.2%）。此前这里回 `InvalidGeometry`，代价是
        // **整条时间线一条片段都发布不出来**，用户只看到
        // "尚未应用：unsupported host region: InvalidGeometry"。
        //
        // ARA 对"读到源尾之后"的定义就是静音（宿主自己的 reader 也这么返回），所以越界
        // 只需读成静音，不必赔上整条时间线。映射层已在源头把窗口夹到源长度并记下现场
        // 数字（`ara/mapping.rs`），这里不再复算一遍 —— 越界的兜底在 `mix_plain_regions`
        // 的取点处（越界记静音），内核路径则由 `formant_cache` 夹到 `total_sec`。
        // 分配量取自播放域，与源窗口是否越界无关。
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
///
/// 【`project_rate` 是干什么的】`startInModificationTime` 的**单位由宿主决定**：素材率与
/// 工程率不同时，宿主按**工程**速率折算它（实测 Cubase 15：48 kHz 素材进 44.1 kHz 工程，
/// `durationMod` 是工程速率秒）。所以从"工程时间"定位到"源样本"必须除以一个速率比值，
/// 而不是直接用 `source.sample_rate` —— 后者隐含假设了"修改时长是源速率秒"。
///
/// 传 `None` 表示拿不到工程速率（宿主没通报、也无法从 ARA 本体取得），此时按宿主的原始
/// 单位直读，与改动前完全一致。这条路径只在**没有内核渲染**时使用（`RenderInput.timeline`
/// 为 `None`），而有内核渲染的那条已经由 `ara::mapping` 折算过。
pub(crate) fn mix_plain_regions(
    regions: &[AraPlaybackRegion],
    sources: &HashMap<String, Arc<SourcePcm>>,
    sample_rate: u32,
    project_rate: Option<f64>,
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
        // 【为什么是 `project / source` 而不是 `source`】宿主给的修改时长是**工程**速率秒，
        // 要先折回源自己的秒，再乘源速率换算成样本下标。工程速率缺席时退回旧行为。
        let rate = match project_rate {
            Some(rate) if rate.is_finite() && rate > 0.0 && source.sample_rate > 0 => {
                (source.sample_rate as f64 / rate).max(0.0)
            }
            _ => source.sample_rate as f64,
        };
        let begin = (region.start_in_playback_time * sample_rate as f64).round() as i64;
        let finish = ((region.start_in_playback_time + region.duration_in_playback_time)
            * sample_rate as f64)
            .round() as i64;
        for sample in begin..finish {
            let position = (region.start_in_modification_time + sample as f64 / sample_rate as f64
                - region.start_in_playback_time)
                * rate;
            let index = position.max(0.0).floor() as usize;
            let fraction = (position - index as f64) as f32;
            let output = (sample - origin) as usize;
            for (channel, target) in [&mut snapshot.left, &mut snapshot.right]
                .into_iter()
                .enumerate()
            {
                let plane = &source.planes[channel.min(source.planes.len() - 1)];
                // 【为什么越界要记静音，而不是重复最后一个样本】宿主可以合法给出超出源尾的
                // 修改时长（素材与工程采样率不同时按工程速率折算，见 `validate_regions`）。
                // 旧写法把 `first`/`second` 都夹到 `len()-1`，于是越界段会**一直重复最后一个
                // 样本** —— 一段直流，听感上是咔哒或嗡鸣。ARA 对"读到源尾之后"的定义是静音。
                if index >= plane.len() {
                    continue;
                }
                let first = index;
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
    /// 快照内容代次：每次 `publish` / `clear` 前进。音频线程据此判断"输出源变了"。
    generation: AtomicU64,
}
impl Default for SnapshotPublisher {
    /// 【为什么从全局预算派生而不是写死 512 MiB】每个 owner 有 **2 个** publisher（两个
    /// 输出率），folder 轨多实例下"每个 512 MiB"会让人误以为天花板是 N × 1 GiB。事实上
    /// 唯一权威是 [`super::budget::global_budget`]：`publish` 每次都要向它 `reserve`，
    /// 进程级总和恒被它夹住。这里的 per-publisher 限额只是本地理智上限，取全局上限即
    /// "一个 publisher 不可能比全局预算还大"，不再制造第二套可叠加的额度。
    fn default() -> Self {
        Self::new(super::budget::global_budget().limit())
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
        let mut retained = self.retained.lock_or_recover();
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
            generation: AtomicU64::new(0),
        }
    }

    /// 当前快照代次；音频线程无锁读取。
    pub(crate) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
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
        let mut retained = self.retained.lock_or_recover();
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
        self.generation.fetch_add(1, Ordering::AcqRel);
        self.reclaim(&mut retained);
        Ok(())
    }
    /// 撤销只切换原子指针，不在音频线程释放存储。
    ///
    /// 【什么时候才该调用】只有"这个实例从此不再渲染"（文档关闭、编辑器停止）。
    /// 宿主编辑造成的失效用 [`Self::invalidate_generation`]，否则清空 `current`
    /// 会让 bound 实例的每一个音频块都写出纯零 —— 用户听到的就是间歇性完全无声。
    pub fn clear(&self) {
        self.current.store(std::ptr::null_mut(), Ordering::SeqCst);
        self.generation.fetch_add(1, Ordering::AcqRel);
    }

    /// 只推进代次，**不**动 `current`。
    ///
    /// 【为什么需要它】revision 门（`processed_ready` / `raw_tape_ready`、以及 ARA 侧
    /// `PreparedVersion`）通过比较代次判断"这份快照是不是当前内容"。宿主编辑时需要
    /// 让这个比较失效以触发重排，但**不能**同时把可听的快照撤掉：撤掉之后
    /// `copy_block_at` 逐样本写零，而 bound 实例固定 `layered == false`，没有兜底。
    pub(crate) fn invalidate_generation(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
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
        // 【为什么未知时**保留旧总量**而不是写 `usize::MAX`】`storage_bytes()` 只在字节数
        // 溢出 `usize` 时返回 `None`（实际不可能发生）。这里不能 `unwrap` —— 这条路径在
        // 每次 `publish`/`has_capacity` 上，一次 panic 会毒化发布器的锁、把整个音频链路
        // 卡死。但也不能写 `usize::MAX`：`publish` 用 `retained.1 + bytes <= limit` 判容量，
        // 一旦写成 MAX 就**永久**拒绝任何后续发布，而且没有任何复位路径 —— 一次理论上的
        // 溢出会变成"播放中再也改不动任何东西"。保留上一次的有效总量，语义是"这次算不出
        // 变化"，容量判据退回上一次的读数。
        let total = retained.0.iter().try_fold(0usize, |total, entry| {
            entry
                .snapshot
                .storage_bytes()
                .and_then(|bytes| total.checked_add(bytes))
        });
        if let Some(total) = total {
            retained.1 = total;
        } else {
            log::warn!(
                "[ara] snapshot byte accounting is unknown (size overflow); keeping the previous                  total so publishing is not wedged permanently"
            );
        }
    }
    /// 分配下一批音频前回收已退役缓冲，避免全局预算已满时无法进入publish回收。
    pub fn collect_retired(&self) {
        let mut retained = self.retained.lock_or_recover();
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
        // SAFETY: 与 `copy_block_at` 相同；输出偏移为 0。
        unsafe { self.copy_block_at(sample, rate, output, 0, frames) }
    }

    /// 把快照的一段写到输出缓冲的 `out_offset` 处。
    ///
    /// 【为什么需要偏移】宿主可以不按循环边界切块；一个块的后半段要写到输出缓冲的
    /// 对应偏移上，而不是从 0 开始覆盖前半段。
    ///
    /// # Safety
    /// 非空输出 plane 至少含 `out_offset + frames` 个可写样本。
    pub unsafe fn copy_block_at(
        &self,
        sample: i64,
        rate: u32,
        output: &mut AudioBusBuffers,
        out_offset: usize,
        frames: usize,
    ) -> bool {
        // 【为什么接受任意声道数】宿主可能给 5.1 / 7.1 总线（见 `audio_abi::validate_bus`）。
        // 我们只处理**前两个**声道，其余保持调用方（`clear_outputs`）已经写好的静音，
        // 并把它们记进静音掩码 —— 拒绝整条总线会让宿主把插件踢出插槽。
        if output.num_channels == 0 || output.channel_buffers.is_null() {
            return false;
        }
        let read_guard = self.enter();
        // SAFETY: 本读区登记的快照在 `read_guard` 存活期间不会被回收（逐快照读者计数非零）。
        let snapshot = read_guard.snapshot();
        if snapshot.is_none() {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        // 实际写入的声道数：单声道（平均 L/R）或立体声；更宽的总线只写前两个。
        let declared = output.num_channels as usize;
        let channels = declared.min(2);
        // 静音掩码：先假定**所有**声明声道都静音（未写入的第 3..N 个由 `clear_outputs`
        // 清零，确实静音），再按实际写入的非零值逐位清除。
        // 【为什么不能写成 `1 << declared.min(64)`】`declared >= 64` 时那是 `1u64 << 64`，
        // 在 debug 下直接 panic（移位溢出）、release 下是未定义结果。宿主给超过 64 声道的
        // 总线极罕见，但这条路径在**音频线程**上 —— 一次 panic 会带走整个 DAW 会话。
        // 全静音掩码用 `u64::MAX` 表达，语义上也正好是"所有声道都静音"。
        let mut silence: u64 = if declared >= 64 {
            u64::MAX
        } else {
            (1_u64 << declared) - 1
        };
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
                // SAFETY: 宿主提供至少 out_offset + frames 个可写样本。
                unsafe { plane.add(out_offset + offset).write(value) };
                if value != 0.0 {
                    silence &= !(1 << channel);
                }
            }
        }
        output.silence_flags = silence;
        snapshot.is_some()
    }

    /// 与 [`Self::copy_block`] 相同，但快照覆盖范围**之外**的帧取输入总线。
    ///
    /// 【为什么必须区分"范围内静音"与"范围外"】`copy_block` 把范围外也写成 0，调用方
    /// 无法在事后补救 —— 写完之后两种零一模一样。Transfer 要的是"录过的那一段被替换成
    /// 处理结果、其余保持宿主原声"，这个区别只能在写之前知道。
    ///
    /// 【为什么按帧判断而不是整段替换】快照与宿主的块并不对齐：一段录制可以在任何一帧
    /// 开始和结束，宿主块的首帧也一样。
    ///
    /// 【没有快照时怎么办】整块取输入 —— 于是"刚录完、合成还没好"这一段是宿主原声，
    /// 而不是静音。这是本函数存在的另一个理由：它让"没有输出"不可能发生。
    ///
    /// # Safety
    /// 与 [`Self::copy_block`] 相同；`planes` 中每个非空切片至少含 `frames` 帧。
    pub unsafe fn copy_block_layered(
        &self,
        sample: i64,
        rate: u32,
        output: &mut AudioBusBuffers,
        frames: usize,
        planes: &[Option<&[f32]>; 2],
    ) -> bool {
        // SAFETY: 与 `copy_block_layered_at` 相同；输入/输出偏移均为 0。
        unsafe { self.copy_block_layered_at(sample, rate, output, 0, frames, planes, 0) }
    }

    /// [`Self::copy_block_layered`] 的带偏移版本。
    ///
    /// `out_offset` = 写到输出缓冲的哪里；`input_offset` = 从输入平面读的起点。
    /// 循环切分的第 2 段两者都等于第一段的帧数。
    ///
    /// # Safety
    /// 与 [`Self::copy_block_layered`] 相同，且非空输入平面至少含
    /// `input_offset + frames` 帧、非空输出平面至少含 `out_offset + frames` 个样本。
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn copy_block_layered_at(
        &self,
        sample: i64,
        rate: u32,
        output: &mut AudioBusBuffers,
        out_offset: usize,
        frames: usize,
        planes: &[Option<&[f32]>; 2],
        input_offset: usize,
    ) -> bool {
        if output.num_channels == 0 || output.channel_buffers.is_null() {
            return false;
        }
        let read_guard = self.enter();
        // SAFETY: 本读区登记的快照在 `read_guard` 存活期间不会被回收（逐快照读者计数非零）。
        let snapshot = read_guard.snapshot();
        if snapshot.is_none() {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        let declared = output.num_channels as usize;
        let channels = declared.min(2);
        let mut silence: u64 = if declared >= 64 {
            u64::MAX
        } else {
            (1_u64 << declared) - 1
        };
        let ratio = snapshot
            .map(|snapshot| snapshot.sample_rate as f64 / rate.max(1) as f64)
            .unwrap_or(1.0);
        let resampling = (ratio - 1.0).abs() > 1e-9;
        // 输入是单声道时两个输出声道取同一个平面（与 `pass_through` 的展开一致）。
        let input_channels = planes.iter().filter(|plane| plane.is_some()).count();
        let input_at = |channel: usize, offset: usize| -> Option<f32> {
            let index = if input_channels <= 1 { 0 } else { channel };
            planes
                .get(index)
                .copied()
                .flatten()
                .and_then(|plane| plane.get(input_offset + offset))
                .copied()
        };
        let read = |channel: usize, offset: usize| -> f32 {
            let Some(snapshot) = snapshot else {
                return input_at(channel, offset).unwrap_or(0.0);
            };
            let local = sample
                .checked_add(offset as i64)
                .and_then(|sample| sample.checked_sub(snapshot.origin_sample));
            let Some(local) = local else {
                return input_at(channel, offset).unwrap_or(0.0);
            };
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
            // 快照的覆盖范围是 `[0, left.len())` 帧（`origin_sample` 已扣掉）。
            let inside = |position: f64| position >= 0.0 && position < snapshot.left.len() as f64;
            if !resampling {
                let position = local as f64;
                return if inside(position) {
                    at(position).unwrap_or(0.0)
                } else {
                    input_at(channel, offset).unwrap_or(0.0)
                };
            }
            let position = local as f64 * ratio;
            if !inside(position) {
                return input_at(channel, offset).unwrap_or(0.0);
            }
            let base = position.floor();
            let frac = (position - base) as f32;
            let a = at(base).unwrap_or(0.0);
            // 末尾一帧没有后继：保持最后一个采样（不外推、不补零）。
            let b = at(base + 1.0).unwrap_or(a);
            a + (b - a) * frac
        };
        for channel in 0..channels {
            // SAFETY: 调用方已验证 SDK 数组有 `channels` 个指针，允许 inactive null plane。
            let plane = unsafe { *output.channel_buffers.add(channel) };
            if plane.is_null() {
                continue;
            }
            for offset in 0..frames {
                let value = if channels == 1 {
                    (read(0, offset) + read(1, offset)) * 0.5
                } else {
                    read(channel, offset)
                };
                // SAFETY: 宿主提供至少 out_offset + frames 个可写样本。
                unsafe { plane.add(out_offset + offset).write(value) };
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
    /// 源窗口越界**不再**是致命错误，且越界段渲染成静音。
    ///
    /// 【为什么这条契约变了】宿主按**工程**速率折算修改时长时必然越界：实测 Cubase 15
    /// 把 48kHz / 1526332 帧素材放进 44.1kHz 工程，给出 `durationMod=34.610703`，而
    /// 34.610703 × 44100 = 1526332（源帧数）、× 48000 = 1661314（越界 8.2%）。旧行为回
    /// `InvalidGeometry`，代价是**整条时间线一条片段都发布不出来** —— 用户只看到
    /// "尚未应用：unsupported host region: InvalidGeometry"。一个越界窗口绝不该赔上整条
    /// 时间线：ARA 对"读到源尾之后"的定义就是静音。
    #[test]
    fn a_source_window_past_the_end_is_tolerated_and_renders_silence() {
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
        // 曾经被拒的第一种：窗口尾越过源尾一帧以上。
        r.duration_in_modification_time = 5.5 / 44100.;
        r.duration_in_playback_time = r.duration_in_modification_time;
        assert!(
            validate_regions(&[r.clone()], &sources, 44100, false).is_ok(),
            "a window running past the source end must not fail the whole timeline"
        );
        // 曾经被拒的第二种：整个窗口都在源尾之后。
        r.start_in_modification_time = 4. / 44100.;
        r.duration_in_modification_time = 0.5 / 44100.;
        r.duration_in_playback_time = 0.5 / 44100.;
        assert!(
            validate_regions(&[r.clone()], &sources, 44100, false).is_ok(),
            "a window entirely past the source end must not fail the whole timeline"
        );

        // 越界段必须是**静音**，不是重复最后一个样本：后者是一段直流，听感上是咔哒/嗡鸣。
        let snapshot =
            mix_plain_regions(&[r], &sources, 44100, None).expect("out-of-range window mixes");
        let nonzero = snapshot
            .left
            .iter()
            .filter(|sample| sample.abs() > f32::EPSILON)
            .count();
        assert_eq!(
            nonzero, 0,
            "a window past the source end must render silence, not a held sample"
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
        let snapshot = mix_plain_regions(&[r.clone()], &sources, 4, None).unwrap();
        assert_eq!(snapshot.left, [0.0, 0.5, 1.0, 1.0]);
        assert_eq!(snapshot.right, [1.0, 0.5, 0.0, 0.0]);
        r.duration_in_playback_time = 0.5;
        assert_eq!(
            mix_plain_regions(&[r], &sources, 4, None).unwrap_err(),
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
        let snapshot = mix_plain_regions(&[r], &sources, 4, None).unwrap();
        assert_eq!(snapshot.origin_sample, 2);
        assert_eq!(snapshot.left, [0.2, 0.3]);
        assert_eq!(snapshot.right, [0.2, 0.3]);
    }

    /// **工程速率已知时，直通路径必须按 `source / project` 折算源下标。**
    ///
    /// 【为什么这是回归测试】`mix_plain_regions` 原本用 `source.sample_rate` 直接乘，
    /// 隐含假设"修改时长是源速率秒"。但宿主按**工程**速率折算修改时长（实测 Cubase 15：
    /// 48 kHz 素材进 44.1 kHz 工程），那条假设让整条素材被提前 8.2% 读完、尾部留白。
    /// 有内核渲染的路径已由 `ara::mapping` 折算，直通路径必须同口径，否则两条路径给出
    /// 不同结果。
    ///
    /// 夹具：源 8 Hz / 8 帧（1 s 内容），工程 4 Hz，宿主给的修改时长 1.0 s **工程**秒。
    /// 折算后源下标 = t × (8/4) = 2t，第 4 个输出帧（下标 8）起越过源尾。
    #[test]
    fn the_plain_path_converts_the_project_rate_before_indexing_the_source() {
        let mut r = region();
        r.start_in_modification_time = 0.0;
        r.duration_in_modification_time = 1.0;
        r.start_in_playback_time = 0.0;
        r.duration_in_playback_time = 1.0;
        let sources = [(
            r.audio_source_persistent_id.clone(),
            Arc::new(SourcePcm {
                sample_rate: 8,
                planes: vec![vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
                version: 0,
                _reservation: None,
            }),
        )]
        .into_iter()
        .collect();
        // 输出率 = 工程率 4 Hz：1.0 s ⇒ 4 帧；折算后源位置 t/2，线性插值取 0.1 / 0.15 / 0.2 / 0.25。
        let snapshot = mix_plain_regions(&[r.clone()], &sources, 4, Some(4.0)).unwrap();
        assert_eq!(snapshot.left.len(), 4);
        assert_eq!(snapshot.left, [0.1, 0.15, 0.2, 0.25]);

        // 同一份输入但**不告诉**工程速率：退回"修改时长即源速率秒"，源按 8 Hz 直读，
        // 4 帧输出的源位置 t×2 ⇒ 下标 0 / 2 / 4 / 6，即 0.1 / 0.3 / 0.5 / 0.7。
        let blind = mix_plain_regions(&[r], &sources, 4, None).unwrap();
        assert_eq!(blind.left.len(), 4);
        assert_eq!(blind.left, [0.1, 0.3, 0.5, 0.7]);
        assert_ne!(blind.left, snapshot.left, "两条路径必须给出不同结果");
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
        assert_eq!(publisher.retained.lock_or_recover().0.len(), 1);
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

    /// **回归：比立体声更宽的总线必须被接受，且只写前两个声道。**
    ///
    /// 【为什么这是回归测试】此前 `copy_block` 对 `num_channels > 2` 直接返回 `false`，
    /// 而 `setBusArrangements` 也拒绝非 mono/stereo 布局 —— 多数宿主据此**把插件踢出
    /// 插槽**：5.1 / 7.1 轨道上用户看到的是"插件不见了"，没有任何提示。
    /// 现在的取舍是"部分工作、留在插槽里"：处理前两个声道，其余保持静音并如实回报。
    #[test]
    fn a_wider_than_stereo_bus_is_accepted_and_the_extra_channels_stay_silent() {
        let publisher = SnapshotPublisher::new(1024);
        publisher
            .publish(PlaybackSnapshot {
                sample_rate: 4,
                origin_sample: 0,
                left: vec![0.2, 0.4, 0.6],
                right: vec![0.4, 0.8, 1.2],
                _reservation: None,
            })
            .unwrap();

        // 6 声道（5.1）：调用方已按 `clear_outputs` 语义清零，这里只验证写前两个。
        let mut planes: Vec<[f32; 3]> = vec![[0.0; 3]; 6];
        let mut pointers: Vec<*mut f32> = planes.iter_mut().map(|p| p.as_mut_ptr()).collect();
        let mut bus = AudioBusBuffers {
            num_channels: 6,
            silence_flags: 0,
            channel_buffers: pointers.as_mut_ptr(),
        };
        // SAFETY: 6 个 plane 各至少 3 帧。
        assert!(
            unsafe { publisher.copy_block(0, 4, &mut bus, 3) },
            "宽总线不能被拒绝（拒绝即被宿主踢出插槽）"
        );
        assert!((planes[0][0] - 0.2).abs() < 1e-6, "{:?}", planes[0]);
        assert!((planes[1][0] - 0.4).abs() < 1e-6, "{:?}", planes[1]);
        // 第 3..6 个声道未被写入，且必须被标成静音（宿主据此跳过它们）。
        for plane in &planes[2..] {
            assert!(plane.iter().all(|sample| *sample == 0.0), "{plane:?}");
        }
        for channel in 2..6 {
            assert!(
                bus.silence_flags & (1 << channel) != 0,
                "未处理的声道必须标静音，flags={:#b}",
                bus.silence_flags
            );
        }
        // 前两个声道有内容 ⇒ 不能标静音。
        assert_eq!(bus.silence_flags & 0b11, 0, "{:#b}", bus.silence_flags);
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
            assert_eq!(publisher.retained.lock_or_recover().0.len(), 1);
            assert_eq!(publisher.retained.lock_or_recover().1, 8);
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
            publisher.retained.lock_or_recover().0.len(),
            2,
            "只应留下正被读的 B 与当前 C"
        );
        drop(r2);
        publisher.collect_retired();
        assert_eq!(publisher.retained.lock_or_recover().0.len(), 1);
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
        assert_eq!(publisher.retained.lock_or_recover().0.len(), 2);
        drop(read);
        assert!(publisher.has_capacity(&snapshot()));
        assert_eq!(publisher.retained.lock_or_recover().0.len(), 1);
    }

    /// **并发压测：读者进入读区与发布回收同时进行时，读者读到的必须仍是有效快照。**
    ///
    /// 【为什么需要这条】`enter` 是"载入指针 → 递增该快照读者数"两步，`reclaim` 的自旋
    /// 条件是 `registering == 0`。安全性依赖一个内存序事实：`enter` 里
    /// `readers.fetch_add` 与 `registering.fetch_sub` 同为 SeqCst，且**程序序上前者在先**；
    /// 因此任何观察到 `registering == 0` 的线程，按 SeqCst 的全序必然也观察到那次
    /// `readers.fetch_add`。也就是说"读者刚载入指针、还没来得及登记就被回收"这个窗口
    /// 是被封住的 —— 但这层保证只存在于 SeqCst 的全序里，一旦有人把某个序放松就会失效，
    /// 而既有测试全是单线程的，**测不出**来。
    ///
    /// 这里让一个发布者与多个读者真并发跑，读者对读到的样本做校验和累加。若不变式被
    /// 破坏（读到已释放的内存），在 debug 下通常会 panic、在校验和上会表现为野值 ——
    /// 两种情况都会让这条测试失败。
    #[test]
    fn concurrent_enter_and_publish_never_hand_a_reader_a_reclaimed_snapshot() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        let publisher = Arc::new(SnapshotPublisher::new(64 * 1024));
        let stop = Arc::new(AtomicBool::new(false));
        let mut readers = Vec::new();
        for _ in 0..4 {
            let publisher = Arc::clone(&publisher);
            let stop = Arc::clone(&stop);
            readers.push(std::thread::spawn(move || {
                let mut checksum = 0.0_f64;
                let mut observed = 0_usize;
                while !stop.load(Ordering::Relaxed) {
                    let read = publisher.enter();
                    if let Some(snapshot) = read.snapshot() {
                        // 读区内校验：内容必须是发布者写入的那个常量图案。
                        // 已释放的内存几乎不可能仍处处等于它。
                        assert_eq!(snapshot.sample_rate, 4);
                        assert!(snapshot.left.iter().all(|sample| *sample == 0.25));
                        assert!(snapshot.right.iter().all(|sample| *sample == -0.5));
                        checksum += snapshot.left.len() as f64;
                        observed += 1;
                    }
                }
                (checksum, observed)
            }));
        }
        // 发布者持续替换快照，让 `reclaim` 反复尝试回收退役缓冲。
        for _ in 0..2000 {
            publisher
                .publish(PlaybackSnapshot {
                    sample_rate: 4,
                    origin_sample: 0,
                    left: vec![0.25; 2],
                    right: vec![-0.5; 2],
                    _reservation: None,
                })
                .expect("publish stays within the concurrent budget");
        }
        stop.store(true, Ordering::Relaxed);
        let mut total = 0_usize;
        for reader in readers {
            let (_, observed) = reader.join().expect("reader thread must not panic");
            total += observed;
        }
        assert!(total > 0, "读者至少应成功进入过一次读区");
    }

    /// **区间外直通输入**：录过的那一段被替换成快照，其余保持宿主原声。
    ///
    /// 【为什么这是回归测试】`copy_block` 把快照范围外也写成 0 —— 于是"只录了 10 秒"
    /// 表现成"整条轨静音"，用户以为自己录坏了。这一条同时钉住两件事：范围内取快照、
    /// 范围外取输入总线（而不是零）。
    #[test]
    fn layered_copy_substitutes_only_the_snapshot_span() {
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
        let mut left = [0.0_f32; 4];
        let mut right = [0.0_f32; 4];
        let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut bus = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: planes.as_mut_ptr(),
        };
        // 输入总线是常量，便于识别"这一帧取的是输入"。
        let input_left = [7.0_f32; 4];
        let input_right = [8.0_f32; 4];
        let input = [Some(&input_left[..]), Some(&input_right[..])];
        // 块首 sample=0、4 帧：快照覆盖 [2,4) ⇒ 前两帧取输入，后两帧取快照。
        assert!(unsafe { publisher.copy_block_layered(0, 4, &mut bus, 4, &input) });
        assert_eq!(left, [7.0, 7.0, 0.2, 0.3]);
        assert_eq!(right, [8.0, 8.0, 0.4, 0.5]);
    }

    /// **带输出偏移的拷贝只覆盖请求的子区间。**
    #[test]
    fn copy_block_at_writes_only_the_requested_output_range() {
        let publisher = SnapshotPublisher::new(1024);
        publisher
            .publish(PlaybackSnapshot {
                sample_rate: 4,
                origin_sample: 0,
                left: vec![1.0, 2.0, 3.0, 4.0],
                right: vec![5.0, 6.0, 7.0, 8.0],
                _reservation: None,
            })
            .unwrap();
        let mut left = [9.0_f32; 6];
        let mut right = [9.0_f32; 6];
        let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut bus = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: planes.as_mut_ptr(),
        };
        // SAFETY: 输出平面提供 6 帧；写入 [2, 6)。
        assert!(unsafe { publisher.copy_block_at(0, 4, &mut bus, 2, 4) });
        assert_eq!(left, [9.0, 9.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(right, [9.0, 9.0, 5.0, 6.0, 7.0, 8.0]);
    }

    /// **分层拷贝的输入偏移与输出偏移各自独立。**
    #[test]
    fn layered_copy_uses_independent_input_and_output_offsets() {
        let publisher = SnapshotPublisher::new(16);
        publisher
            .publish(PlaybackSnapshot {
                sample_rate: 4,
                origin_sample: 2,
                left: vec![0.2, 0.3],
                right: vec![0.4, 0.5],
                _reservation: None,
            })
            .unwrap();
        let mut left = [9.0_f32; 5];
        let mut right = [9.0_f32; 5];
        let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut bus = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: planes.as_mut_ptr(),
        };
        let input_left = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let input_right = [10.0_f32, 11.0, 12.0, 13.0, 14.0, 15.0];
        let input = [Some(&input_left[..]), Some(&input_right[..])];
        // SAFETY: 输入/输出都提供 offset + frames 的长度。
        assert!(unsafe { publisher.copy_block_layered_at(2, 4, &mut bus, 1, 4, &input, 2) });
        // [0] 保留原值；[1..3) 取快照；[3..5) 取输入下标 4、5。
        assert_eq!(left, [9.0, 0.2, 0.3, 4.0, 5.0]);
        assert_eq!(right, [9.0, 0.4, 0.5, 14.0, 15.0]);
    }

    /// **没有快照时整块取输入**：于是"刚录完、合成还没好"这一段是宿主原声，不是静音。
    ///
    /// 【为什么这是这条通路存在的第二个理由】它是"任何中间状态下都有声音"的最后一层
    /// 兜底 —— 用户按下停止之后立刻听到东西，而不是等一次神经合成跑完。
    #[test]
    fn layered_copy_falls_back_to_the_input_without_a_snapshot() {
        let publisher = SnapshotPublisher::new(1024);
        let mut left = [0.0_f32; 2];
        let mut right = [0.0_f32; 2];
        let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut bus = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: planes.as_mut_ptr(),
        };
        let input_left = [0.5_f32; 2];
        // 单声道输入（右平面缺失）⇒ 两个输出声道都取左。
        let input = [Some(&input_left[..]), None];
        assert!(!unsafe { publisher.copy_block_layered(0, 4, &mut bus, 2, &input) });
        assert_eq!(left, [0.5, 0.5]);
        assert_eq!(right, [0.5, 0.5]);
        assert_eq!(bus.silence_flags, 0, "有非零输出就不该报静音");
    }
}
