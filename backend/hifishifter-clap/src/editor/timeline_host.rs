//! CLAP 外壳的**编辑器时间线**：捕获投影 + 用户对参数曲线的编辑。
//!
//! 【为什么需要一个常驻的时间线对象】此前 `get_timeline_state` 每次调用都现场投影一份
//! 新的 `TimelineState`。只读时没问题，但参数编辑器是**有状态**的：用户画的音高线存在
//! `TrackParamsState` 里，下一次投影会把它连同对象一起丢掉 —— 表现是"画一笔就没了"、
//! "关掉再开线全不见"。所以时间线必须在两次命令之间活下来。
//!
//! 【与捕获存储的关系】捕获存储是**素材**（段、take、PCM）；这里是它的**视图**（轨道、
//! 片段、参数曲线）。素材每变一次（新一轮录制、清空、删 take），`CaptureStore::revision`
//! 就前进一次，本对象据此重建视图 —— 但只在真的变了的时候重建，否则用户的编辑会被
//! 一次无害的轮询冲掉。
//!
//! 【音高原线从哪来】VST3/App 用 FCPE（ONNX）分析；CLAP 外壳刻意不链接推理内核，
//! 因此这里用**静态链接的 WORLD**（`world_vocoder::estimate_f0_hz`）在后台线程上算一条
//! 基线。它比 FCPE 粗，但它是真实的、可编辑的，且不把 40 MB 的推理栈拖进插件。
//! 分析在**后台线程**上跑：一段几分钟的素材要几秒到几十秒，卡在 WebView2 的消息
//! 处理线程上就是"界面假死"。

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use hifishifter_capture::store::CaptureStore;
use hifishifter_capture::types::CapturePcm;
use hifishifter_kernel::editor::ParamHost;
use hifishifter_kernel::state::{HistoryOp, TimelineState};
use serde_json::{json, Value};

/// 页面事件出口：`(事件名, 载荷)`。
///
/// 【为什么是一张闭包而不是直接拿 HWND】音高基线在**后台线程**上算完，而窗口
/// 句柄属于宿主 UI 线程（且不是 `Send`）。外壳在这里只看见"发一条事件"这件事，
/// 具体怎么送到页面由 `editor::windows` 决定（Windows 上是 `PostMessage` 回 UI
/// 线程再 `PostWebMessageAsJson`）。于是本文件在任何平台上都能编译，也能在
/// 单测里换成一个记录器。
pub type Notifier = Arc<dyn Fn(&str, Value) + Send + Sync>;

/// 捕获时间线上唯一的轨道 id（与 `hifishifter_capture::timeline` 同一个常量）。
pub const CAPTURE_TRACK_ID: &str = hifishifter_capture::timeline::CAPTURE_TRACK_ID;

/// 一次音高基线分析的状态。
#[derive(Default)]
struct PitchState {
    /// 已经产出基线的捕获修订号（`u64::MAX` = 还没有）。
    ready_revision: u64,
    /// **当前这一轮**分析对应的捕获修订号（`u64::MAX` = 没有在跑）。
    ///
    /// 【为什么不能只看 `running`】分析期间素材可能又变了（再录一遍、清空）：
    /// `refresh` 会把 `running` 清零并允许新一轮启动。旧线程随后回来时若只读
    /// `running`，就会把**新一轮**的"正在跑"当成自己的、并写进一份属于旧素材的
    /// 曲线 —— 用户看到的是"刚录的那条音高线变成了上一段的形状"。
    job_revision: u64,
    /// 有线程正在算。
    running: bool,
}

/// 编辑器时间线 + 它的派生数据。
pub struct TimelineHost {
    timeline: Mutex<TimelineState>,
    /// `timeline` 对应的捕获修订号。
    revision: AtomicU64,
    pitch: Mutex<PitchState>,
    /// 页面事件出口（窗口没开时为 `None`）。
    notifier: Mutex<Option<Notifier>>,
    /// 渲染引擎：音高基线落地后要**重渲染**（渲染读的正是这条 `pitch_orig`）。
    render: Mutex<Option<Arc<crate::render::RenderEngine>>>,
}

impl Default for TimelineHost {
    fn default() -> Self {
        Self {
            timeline: Mutex::new(TimelineState::default()),
            // 【为什么初值不是 0】空捕获存储的修订号也是 0；用 MAX 保证第一次访问一定
            // 会建一次视图（默认的 `TimelineState` 带着一条 "Main" 轨，不该被看到）。
            revision: AtomicU64::new(u64::MAX),
            pitch: Mutex::new(PitchState {
                ready_revision: u64::MAX,
                job_revision: u64::MAX,
                running: false,
            }),
            notifier: Mutex::new(None),
            render: Mutex::new(None),
        }
    }
}

impl TimelineHost {
    fn lock(&self) -> std::sync::MutexGuard<'_, TimelineState> {
        self.timeline.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// 安装 / 清除页面事件出口（窗口创建时装、销毁时清）。
    pub fn set_notifier(&self, notifier: Option<Notifier>) {
        *self.notifier.lock().unwrap_or_else(|e| e.into_inner()) = notifier;
    }

    /// 发一条页面事件。没有窗口时静默丢弃（后台线程不该因此失败）。
    fn emit(&self, event: &str, payload: Value) {
        // 【为什么克隆再放锁】出口可能在回调里被换掉（窗口关闭）；持着锁调用
        // 页面代码会让"关窗"等它，而它可能正等着 UI 线程。
        let notifier = self
            .notifier
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if let Some(notifier) = notifier {
            notifier(event, payload);
        }
    }

    /// 接上渲染引擎（音高基线落地后按需重渲染）。
    pub fn attach_render(&self, render: Arc<crate::render::RenderEngine>) {
        *self.render.lock().unwrap_or_else(|e| e.into_inner()) = Some(render);
    }

    /// 请求一次重渲染（音高基线从"没有"变成"有"，渲染输入真的变了）。
    ///
    /// 【为什么基线落地要重渲染】没有 `pitch_orig` 时渲染侧的兜底曲线是空的，
    /// `apply_host_pitch_edit` 会**跳过**那个 clip（"等分析补齐后由重渲染补上"）。
    /// 此前没有任何东西补这一刀：用户先画线、基线后到，就永远听不到变化。
    fn request_render(&self, sources: HashMap<String, Arc<CapturePcm>>, sample_rate: u32) {
        let render = self
            .render
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let Some(render) = render else { return };
        render.request(self.snapshot(), sources, sample_rate);
    }

    /// 素材变了就重建视图；没变就什么都不做（保住用户的编辑）。
    pub fn refresh(&self, store: &CaptureStore) {
        let revision = store.revision();
        if self.revision.load(Ordering::Acquire) == revision {
            return;
        }
        let (fresh, _sources) = store.session_timeline();
        let mut timeline = self.lock();
        // 【为什么再判一次】两个线程可能同时发现"版本变了"；先到的那个已经建好了，
        // 后到的不能再覆盖一次 —— 那会把刚发生的编辑冲掉。
        if self.revision.load(Ordering::Acquire) == revision {
            return;
        }
        *timeline = fresh;
        // 【为什么把算法钉成 WORLD】渲染要用声码器，而本外壳不链接 ONNX 推理栈
        // （模块 1.5 MB，VST3 那套引擎 38 MB）。WORLD 是静态链接的，因此它是这里
        // **唯一**能真正产出处理结果的算法。不钉的话，轨道默认的 nsf_hifigan_onnx
        // 在渲染侧会被判成"后端不可用"，结果是"渲染成功但一个样本都没变"。
        for track in &mut timeline.tracks {
            track.pitch_analysis_algo = hifishifter_kernel::state::PitchAnalysisAlgo::WorldDll;
        }
        self.revision.store(revision, Ordering::Release);
        let mut pitch = self.pitch.lock().unwrap_or_else(|e| e.into_inner());
        pitch.ready_revision = u64::MAX;
        // 【为什么连 job_revision 一起清】在跑的那一轮属于**旧**素材；留着它，
        // 旧线程回来时就会把新一轮的 `running` 当成自己的（见 `PitchState`）。
        pitch.job_revision = u64::MAX;
        pitch.running = false;
    }

    /// 时间线的一份只读快照（渲染作业用）。
    pub fn snapshot(&self) -> TimelineState {
        self.lock().clone()
    }

    /// 时间线载荷（编辑器渲染的那一份）。
    pub fn payload(&self, playhead_sec: f64) -> Value {
        let timeline = self.lock();
        let mut payload =
            serde_json::to_value(timeline.to_payload()).unwrap_or_else(|_| serde_json::json!({}));
        payload["playhead_sec"] = serde_json::json!(playhead_sec.max(0.0));
        payload["undo_depth"] = serde_json::json!(0);
        payload["redo_depth"] = serde_json::json!(0);
        payload
    }

    /// 轨道是否存在（参数命令的前置校验）。
    pub fn track_exists(&self, track_id: &str) -> bool {
        self.lock().tracks.iter().any(|track| track.id == track_id)
    }

    /// 当前音高基线是否已经就绪（`false` = 还在后台算）。
    pub fn pitch_pending(&self) -> bool {
        self.pitch.lock().unwrap_or_else(|e| e.into_inner()).running
    }

    /// 确保音高基线存在；需要时**在后台线程**上算。
    ///
    /// 返回 `true` 表示"现在就有一条可用的基线"。
    pub fn ensure_pitch_baseline(self: &Arc<Self>, store: &CaptureStore) -> bool {
        self.refresh(store);
        let revision = self.revision.load(Ordering::Acquire);
        {
            let mut state = self.pitch.lock().unwrap_or_else(|e| e.into_inner());
            if state.ready_revision == revision {
                return true;
            }
            if state.running {
                return false;
            }
            state.running = true;
            state.job_revision = revision;
        }
        // 【为什么要在这里物化源表】线程里不能再碰捕获存储（它归属于插件实例，
        // 生命周期不保证跨线程）；`material_sources` 返回的是 Arc，在线程里读是安全的。
        let sources = store.material_sources();
        let (root, frame_period_ms, target, clips) = {
            let timeline = self.lock();
            let Some(root) = timeline
                .tracks
                .first()
                .map(|track| track.id.clone())
                .or_else(|| Some(CAPTURE_TRACK_ID.to_owned()))
            else {
                return false;
            };
            let frame_period_ms = timeline.frame_period_ms();
            let target = timeline.target_param_frames(frame_period_ms);
            let mut clips = Vec::new();
            for clip in &timeline.clips {
                if clip.muted {
                    continue;
                }
                let Some(source) = clip.source_path.clone() else {
                    continue;
                };
                clips.push(ClipWindow {
                    source,
                    start_sec: clip.start_sec.max(0.0),
                    length_sec: clip.length_sec.max(0.0),
                    source_start_sec: clip.source_start_sec.max(0.0),
                    source_end_sec: clip.source_end_sec.max(0.0),
                    playback_rate: if clip.playback_rate.is_finite() && clip.playback_rate > 0.0 {
                        f64::from(clip.playback_rate)
                    } else {
                        1.0
                    },
                });
            }
            (root, frame_period_ms, target, clips)
        };
        if clips.is_empty() {
            let mut state = self.pitch.lock().unwrap_or_else(|e| e.into_inner());
            state.running = false;
            state.job_revision = u64::MAX;
            state.ready_revision = revision;
            return true;
        }
        let rate = store.project_rate().round().max(0.0) as u32;
        // 线程要拿走 `root`；事件载荷在下面还要用一次。
        let notify_root = root.clone();
        // 【为什么先发 started 再起线程】反过来的话，线程可能在 UI 线程发出 started
        // 之前就发完 updated —— 页面先收到"完成"、再收到"开始"，状态栏会永远停在
        // "分析中"。先发再起，顺序由这一条语句固定。
        self.emit(
            "pitch_orig_analysis_started",
            json!({ "rootTrackId": notify_root }),
        );
        let host = Arc::clone(self);
        let spawned = std::thread::Builder::new()
            .name("hfs-clap-pitch".into())
            .spawn(move || {
                // 【为什么接住 panic】这条线程一旦带着 panic 结束，`running` 就永远
                // 停在 true：界面一直"分析中"，基线永远不来，而且不会重试。把它收敛
                // 成"这条曲线是空的"，至少让状态回到"没在跑"，下一次请求还能再来一遍。
                let curve = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    compute_pitch_curve(&clips, &sources, frame_period_ms, target)
                }))
                .unwrap_or_else(|_| {
                    log::error!("[clap][pitch] baseline analysis panicked");
                    Vec::new()
                });
                host.finish_pitch(&root, curve, revision, sources, rate);
            });
        if spawned.is_err() {
            let mut state = self.pitch.lock().unwrap_or_else(|e| e.into_inner());
            state.running = false;
            state.job_revision = u64::MAX;
            drop(state);
            // 线程没起来 ⇒ 不会再有 updated。把界面从"分析中"放回静止，否则那条
            // 提示会一直亮着（这也是页面唯一能收到的"这一轮结束了"）。
            self.emit("pitch_orig_updated", json!({ "rootTrackId": notify_root }));
            return false;
        }
        false
    }

    /// 后台线程：把算好的基线写进时间线，并通知页面。
    ///
    /// 【为什么要通知】参数编辑器只在**事件**里得知"原线分析完了"（VST3 那条路
    /// 由内核发 `pitch_orig_updated`）。CLAP 外壳此前什么也不发，于是取到的那一次
    /// 还是"分析中、曲线为空"，之后**再也没有第二次取数** —— 用户看到的是一片空白，
    /// 直到他自己画一笔（那次写入顺带触发了一次重取）。
    fn finish_pitch(
        &self,
        root: &str,
        curve: Vec<f32>,
        revision: u64,
        sources: HashMap<String, Arc<CapturePcm>>,
        sample_rate: u32,
    ) {
        // 【为什么先取帧数再拿第二把锁】`refresh` 的加锁顺序是"时间线 → pitch"。
        // 这里若持着 pitch 再去拿时间线，两个线程就会各持一把互相等（死锁）。
        // 锁序必须全局一致：**时间线在前，pitch 在后**。
        let mut frames = 0_usize;
        let mut needs_render = false;
        {
            let mut timeline = self.lock();
            // 【为什么在这里判修订号】分析在后台跑，期间素材可能又变了。`refresh`
            // 已经把视图换成新的、并允许新一轮分析启动；这份结果属于**旧素材**，
            // 写进去就是把刚录的那条音高线换成上一段的形状。
            if self.revision.load(Ordering::Acquire) != revision {
                log::info!("[clap][pitch] baseline for revision {revision} is stale; dropped");
                return;
            }
            timeline.ensure_params_for_root(root);
            let key = hifishifter_kernel::pitch_analysis::build_root_pitch_key(&timeline, root);
            let layout =
                hifishifter_kernel::pitch_analysis::build_root_pitch_layout_key(&timeline, root);
            if let Some(entry) = timeline.params_by_root_track.get_mut(root) {
                entry.pitch_orig = curve;
                // 【为什么只在没有用户编辑时覆盖 edit】用户可能已经画过线；分析回来
                // 时把它整条冲掉，等于"画完等一会儿就没了"。
                if !entry.pitch_edit_user_modified
                    || entry.pitch_edit.len() != entry.pitch_orig.len()
                {
                    entry.pitch_edit = entry.pitch_orig.clone();
                    entry.pitch_edit_user_modified = false;
                }
                // 【为什么要写这两个 key】内核的 `maybe_schedule_pitch_orig` 靠它们判断
                // "原线已是最新"。不写的话，它会认为该组装，于是用**空缓存**重新组装，
                // 把我们刚算好的曲线覆盖成零。
                entry.pitch_orig_key = Some(key);
                entry.pitch_layout_key = Some(layout);
                frames = entry.pitch_orig.len();
                // 有用户画的线（或算法自带调整）时，这条基线**就是**渲染输入：
                // 它从空变成有，之前被跳过的那次合成现在能跑了。
                needs_render = entry.pitch_edit_user_modified || entry.has_pitch_adjustment_active;
            }
        }
        {
            let mut state = self.pitch.lock().unwrap_or_else(|e| e.into_inner());
            // 【为什么再判一次】这一轮可能已经被 `refresh` 作废、或被更晚的一轮取代；
            // 那种情况下 `running` 属于**别人**，清掉就等于让新任务永远"没在跑"。
            if state.job_revision != revision {
                return;
            }
            state.running = false;
            state.ready_revision = revision;
            state.job_revision = u64::MAX;
        }
        log::info!(
            "[clap][pitch] baseline ready for '{root}' at revision {revision} ({frames} frame(s))"
        );
        // 通知页面：原线到了。前端据此重取曲线（`pitch_orig_updated` 与 VST3 同名）。
        self.emit("pitch_orig_updated", json!({ "rootTrackId": root }));
        if needs_render {
            self.request_render(sources, sample_rate);
        }
    }
}

/// 一个片段在时间线上的窗口（后台线程只读这份拷贝）。
struct ClipWindow {
    source: String,
    start_sec: f64,
    length_sec: f64,
    source_start_sec: f64,
    source_end_sec: f64,
    playback_rate: f64,
}

/// 把 Hz 换成 MIDI 音高（0 = 没有音高）。
fn hz_to_midi(hz: f32) -> f32 {
    if !(hz.is_finite() && hz > 0.0) {
        return 0.0;
    }
    69.0 + 12.0 * (f64::from(hz) / 440.0).log2() as f32
}

/// 逐片段算音高并铺到工程帧网格上。
fn compute_pitch_curve(
    clips: &[ClipWindow],
    sources: &HashMap<String, Arc<CapturePcm>>,
    frame_period_ms: f64,
    target: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; target];
    if !frame_period_ms.is_finite() || frame_period_ms <= 0.0 {
        return out;
    }
    for clip in clips {
        let Some(pcm) = sources.get(&clip.source) else {
            continue;
        };
        let channels = pcm.planes.len();
        let frames = pcm.planes.first().map(Vec::len).unwrap_or(0);
        if frames == 0 || channels == 0 {
            continue;
        }
        let rate = f64::from(pcm.sample_rate.max(1));
        let start = (clip.source_start_sec * rate).round().max(0.0) as usize;
        let declared_end = if clip.source_end_sec > 0.0 {
            (clip.source_end_sec * rate).round().max(0.0) as usize
        } else {
            frames
        };
        let end = declared_end.min(frames).max(start);
        if end <= start {
            continue;
        }
        // 单声道混合：WORLD 只吃单声道，取各声道平均（与渲染侧的下混同口径）。
        let mono: Vec<f32> = (start..end)
            .map(|index| {
                let mut sum = 0.0_f32;
                for plane in &pcm.planes {
                    sum += plane[index];
                }
                sum / channels as f32
            })
            .collect();
        let f0 = hifishifter_kernel::world_vocoder::estimate_f0_hz(
            &mono,
            pcm.sample_rate,
            frame_period_ms,
        );
        if f0.is_empty() {
            continue;
        }
        let clip_start_frame = (clip.start_sec * 1000.0 / frame_period_ms).round() as usize;
        let clip_frames = (clip.length_sec * 1000.0 / frame_period_ms).round() as usize;
        for index in 0..clip_frames {
            let frame = clip_start_frame + index;
            if frame >= target {
                break;
            }
            // 工程帧 → 片段内秒 → 源内秒（播放速率 > 1 时更快消耗源）。
            let elapsed = index as f64 * frame_period_ms / 1000.0 * clip.playback_rate;
            let source_frame = (elapsed * 1000.0 / frame_period_ms).round() as usize;
            let Some(hz) = f0.get(source_frame) else {
                continue;
            };
            let midi = hz_to_midi(*hz);
            if midi > 0.0 {
                out[frame] = midi;
            }
        }
    }
    out
}

impl ParamHost for TimelineHost {
    fn timeline(&self) -> &Mutex<TimelineState> {
        &self.timeline
    }

    fn checkpoint_timeline(&self, _timeline: &TimelineState, _operation: HistoryOp) {
        // 【为什么是空实现】CLAP 外壳没有撤销栈（宿主也没有给它一个）。写曲线本身
        // 仍然生效；只是"撤销这一步"不成立，界面上也不会出现一条假的历史记录。
    }

    fn mark_dirty(&self) {
        // 没有工程文件可标脏；保留这个方法是为了让共享命令能跑通。
    }

    fn publish_timeline(&self, _timeline: TimelineState) {
        // 【为什么什么都不做】发布是"把时间线交给渲染/设备"。CLAP 外壳本阶段是直通，
        // 没有渲染可发布；曲线留在时间线里，下一次读取就是它自己。
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 波形：`hz_to_midi` 的锚点（A4 = 69，C4 ≈ 60）。
    #[test]
    fn hz_maps_to_midi_notes() {
        assert!((hz_to_midi(440.0) - 69.0).abs() < 1e-4);
        assert!((hz_to_midi(261.63) - 60.0).abs() < 0.02);
        assert_eq!(hz_to_midi(0.0), 0.0, "没有音高必须是 0（前端据此判定未画）");
    }

    /// 空捕获存储：第一次访问必须建出**捕获轨道**（而不是默认的 Main 轨）。
    #[test]
    fn a_fresh_store_builds_the_capture_track() {
        let store = CaptureStore::default();
        let host = TimelineHost::default();
        host.refresh(&store);
        let timeline = host.lock();
        assert_eq!(timeline.tracks.len(), 1);
        assert_eq!(timeline.tracks[0].id, CAPTURE_TRACK_ID);
        assert!(timeline.clips.is_empty());
    }

    /// 没有素材时音高基线立刻"就绪"（不该留一个永远转的等待）。
    #[test]
    fn an_empty_timeline_has_no_pending_pitch_job() {
        let store = CaptureStore::default();
        let host = Arc::new(TimelineHost::default());
        assert!(host.ensure_pitch_baseline(&store));
        assert!(!host.pitch_pending());
    }
}
