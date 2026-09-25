/**
 * mockBackend — 纯浏览器开发模式下的后端替身（仅 dev，仅显式开启）。
 *
 * 【主要内容】
 * 在 `window.pywebview.api` 上安装一套假后端实现：启动路径（运行时信息 / UI 设置 /
 * 时间轴状态 / 播放状态）返回构造好的假工程数据，其余未显式实现的方法由 Proxy
 * 兜底返回 `{ ok: true }`，避免任何一次 invoke 抛错中断启动。
 * `select_clip` / `select_track` 与 `get_timeline_state` 同形：三者都返回全量时间轴
 * 快照，并共享一份模块级的"记住的选中"（选中 clip / 当前轨道）——这是复现
 * 「点空白切轨时选中被后端快照复活」类缺陷的前提。
 *
 * 【作用】
 * 本项目前端依赖 pywebview / Tauri 后端（`services/invoke.ts` 的 pywebview 分支），
 * 直接 `npm run dev` 在浏览器里打开会因 `Python API not available` 而无法进入界面，
 * 使得「改渲染 / 调样式」这类纯前端工作无法在浏览器里快速迭代与截图验证。
 * 本模块提供最小可用的后端替身，让时间轴面板能在浏览器里完整渲染。
 * 已显式实现的方法必须与真实后端的**载荷形状**保持一致：返回 `{ ok: true }` 这类
 * 缺字段的替身会让依赖具体字段的前端分支永不触发，从而把真实缺陷掩盖成"已修好"。
 *
 * 【开启方式】
 * URL 追加 `?mock=1`（`main.tsx` 在 dev 下按该参数动态 import 本模块）。
 *
 * 【与其他模块的关系】
 * - 被 `main.tsx` 动态加载（生产构建完全不打包）。
 * - 依赖契约：`services/invoke.ts` 的 pywebview 分支——`window.pywebview.api[method]`
 *   按**位置参数**调用，返回 Promise。
 *
 * 【边界】
 * 只保证「界面能起来 + 时间轴有内容可看」；不模拟真实音频处理、不模拟文件对话框、
 * 不模拟导出。未实现的方法返回 `{ ok: true }`，调用方若依赖具体字段会走各自的兜底分支。
 */

/** 假工程总时长（秒）。 */
const MOCK_PROJECT_SEC = 120;

/** 假工程 BPM。 */
const MOCK_BPM = 120;

/** 合成参数曲线的中心值（pitch 为 MIDI）。 */
const MOCK_PITCH_CENTER = 72;

/** 轨道配色（与生产默认调色板同量级，便于视觉对比）。 */
const TRACK_COLORS = ["#4b8fd1", "#5fb27a", "#d19a4b", "#c25f8f", "#7a6fd1", "#4bb0b0"];

// ── 假波形数据 ────────────────────────────────────────────────────────
// 波形走 `get_waveform_mipmap_binary` 返回的 Base64 二进制（协议见
// `utils/waveformBinaryCodec`：20B header "WFPK" + min f32[] + max f32[]）。
// 这里按同一协议合成随机但有「音频感」的包络，使浏览器里也能验证波形层的
// 渲染、级别切换与滚动复用。

/** 假波形源采样率（与 clip 的 source_sample_rate 一致）。 */
const MOCK_WAVEFORM_SAMPLE_RATE = 44100;

/** 假波形源时长（秒）：覆盖最长 clip。 */
const MOCK_WAVEFORM_DURATION_SEC = 30;

/**
 * 三个 mipmap 级别的除数因子。
 *
 * 与 `waveformMipmapStore` 的级别阈值对齐（`SPP_THRESHOLDS = [512, 1024]`，
 * `spp = sampleRate / pxPerSec`）：默认缩放下 spp ≈ 300–450 → 命中 L0，
 * 因此 L0 取 512 既贴合真实数据量级，又让点数为「时长 × 采样率 / 512」——
 * 30 秒仅 2584 点（约 20KB），浏览器里生成与传输都无压力。
 */
const MOCK_DIVISION_FACTORS = [512, 1024, 2048] as const;

/**
 * 合成一段「像音频」的 min/max 包络。
 *
 * 结构：慢速乐句包络（段落强弱）× 中频音符起伏 × 高频细节毛刺 + 微小直流偏移。
 * 用线性同余伪随机（确定性，不引入依赖）：同一路径每次生成一致，不同路径形状不同。
 *
 * @param peakCount 峰值点数量。
 * @param seed 随机种子。
 * @returns min / max 数组（值域 [-1, 1]）。
 */
function buildMockPeaks(
    peakCount: number,
    seed: number,
): {
    min: Float32Array;
    max: Float32Array;
} {
    const min = new Float32Array(peakCount);
    const max = new Float32Array(peakCount);
    let state = (seed * 2654435761) >>> 0;
    const random = () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 0xffffffff;
    };
    for (let index = 0; index < peakCount; index += 1) {
        const t = peakCount <= 1 ? 0 : index / (peakCount - 1);
        const phrase = 0.35 + 0.5 * Math.abs(Math.sin(t * Math.PI * 4 + seed));
        const note = 0.75 + 0.25 * Math.sin(t * Math.PI * 60 + seed * 2);
        const detail = 0.85 + 0.15 * random();
        const amplitude = Math.min(1, phrase * note * detail);
        const center = (random() - 0.5) * 0.06;
        max[index] = Math.min(1, Math.max(-1, center + amplitude));
        min[index] = Math.min(1, Math.max(-1, center - amplitude * (0.85 + random() * 0.3)));
    }
    return { min, max };
}

/**
 * 把 peaks 编码为后端协议一致的 Base64 字符串。
 *
 * 协议：`[magic "WFPK" 4B][sample_rate u32][division_factor u32][peak_count u32]
 * [level u32][min f32 × n][max f32 × n]`，全部小端。
 *
 * @param args 级别、除数因子与 peaks。
 * @returns Base64 编码的二进制。
 */
function encodeWaveformMipmap(args: {
    level: number;
    divisionFactor: number;
    min: Float32Array;
    max: Float32Array;
}): string {
    const peakCount = args.min.length;
    const buffer = new ArrayBuffer(20 + peakCount * 8);
    const view = new DataView(buffer);
    view.setUint8(0, 0x57); // W
    view.setUint8(1, 0x46); // F
    view.setUint8(2, 0x50); // P
    view.setUint8(3, 0x4b); // K
    view.setUint32(4, MOCK_WAVEFORM_SAMPLE_RATE, true);
    view.setUint32(8, args.divisionFactor, true);
    view.setUint32(12, peakCount, true);
    view.setUint32(16, args.level, true);
    new Float32Array(buffer, 20, peakCount).set(args.min);
    new Float32Array(buffer, 20 + peakCount * 4, peakCount).set(args.max);

    // Base64：分块拼接（一次展开过多参数会爆栈）。
    const bytes = new Uint8Array(buffer);
    let binary = "";
    const CHUNK = 8192;
    for (let offset = 0; offset < bytes.length; offset += CHUNK) {
        binary += String.fromCharCode(...bytes.subarray(offset, offset + CHUNK));
    }
    return btoa(binary);
}

/** 已生成的波形（按源路径缓存：同一文件重复请求不重复合成）。 */
const mockWaveformCache = new Map<string, [string, string, string]>();

/**
 * 取得某个源路径的三级波形（懒生成 + 缓存）。
 *
 * @param sourcePath 源文件路径（clip 的 source_path）。
 * @returns `[L0, L1, L2]` 的 Base64 数组。
 */
function getMockWaveformLevels(sourcePath: string): [string, string, string] {
    const cached = mockWaveformCache.get(sourcePath);
    if (cached !== undefined) return cached;
    let seed = 0;
    for (let index = 0; index < sourcePath.length; index += 1) {
        seed = (seed * 31 + sourcePath.charCodeAt(index)) >>> 0;
    }
    const levels = MOCK_DIVISION_FACTORS.map((divisionFactor, level) => {
        const peakCount = Math.max(
            1,
            Math.floor((MOCK_WAVEFORM_DURATION_SEC * MOCK_WAVEFORM_SAMPLE_RATE) / divisionFactor),
        );
        const { min, max } = buildMockPeaks(peakCount, seed + level * 7);
        return encodeWaveformMipmap({ level, divisionFactor, min, max });
    }) as [string, string, string];
    mockWaveformCache.set(sourcePath, levels);
    return levels;
}

/**
 * 假后端**记住的** clip 选中（`None` 用 `null` 表示）。
 *
 * 存在的理由：真实后端把选中态记在自己的 TimelineState 里，`to_payload()` **总是**
 * 带上 `selected_clip_id`，且 `select_track` 刻意**不**清它。若 mock 不维护该字段，
 * 前端的 `selectTrackRemote.fulfilled` 会因为载荷里没有 `selected_clip_id` 而永远
 * 不进"恢复后端记住的选中"分支，从而**掩盖**"点空白切轨时选中被复活"的缺陷。
 */
let mockSelectedClipId: string | null = null;

/** 假后端**记住的**轨道焦点；由 `select_track` 改写。 */
let mockSelectedTrackId: string | null = null;

/**
 * 假后端**记住的** Take 声道模式（takeId → channel_mode）。
 *
 * mock 的 take 集合是每次重建的静态数据；若不记住 `set_clip_take_channel_mode`
 * 的写入，乐观更新会被**下一次权威快照打回原形**——与真实后端（快照携带刚
 * 写入的字段）行为不一致，会让"改声道模式"在浏览器里表现为时灵时不灵。
 */
const mockTakeChannelModes = new Map<string, number>();

/**
 * 假 UI 设置（内存里的"配置文件"），默认值与 sessionSlice 的出厂初始值同口径。
 *
 * 此前兜底返回 `{ ok: true }`，所有读设置的面板（MIDI 导入对话框、参数编辑器、
 * 记事本、停靠布局）在浏览器里永远拿到"无设置"，各自的归一化/回填分支无法
 * 被验证。`get_ui_settings` 返回整份对象，`save_ui_settings` 合并部分字段
 * ——与真实后端的读-改-写语义一致，保存后的重读也能观察到写入。
 */
const mockUiSettings: Record<string, unknown> = {
    autoCrossfade: true,
    showAllTakes: true,
    syncEditsAcrossTakes: true,
    loopNewClips: true,
    splitTransitionEnabled: true,
    splitTransitionMode: "overlap",
    splitTransitionDurationUnit: "seconds",
    splitTransitionDurationSec: 0.01,
    splitTransitionOverlapCrossfade: "auto",
    snapEnabled: true,
    tempoMapVisible: true,
    primaryTimeUnit: "seconds",
    secondaryTimeUnit: "none",
    pitchSnap: false,
    pitchSnapUnit: "semitone",
    pitchSnapScale: "C",
    playheadZoom: false,
    autoScroll: false,
    showClipboardPreview: true,
    showParamValuePopup: true,
    lockParamLines: true,
    paramEditorSeekPlayhead: true,
    paramEditorSyncTimeline: true,
    autoReloadModifiedMedia: true,
    // MIDI 导入对话框的出厂默认（与 App.tsx 的本地初始状态一致）。
    midiImportPosition: "selection",
    midiFillGaps: false,
    midiMultiTrackMerge: true,
    midiImportBpmAsProject: false,
    midiNoteBpmMode: "midi",
    midiSpecifiedBpm: 120,
    midiCloseLeadingGap: true,
    midiImportTargetMenu: "pitchRef",
    midiImportTargetDragDrop: "pitchRef",
    midiImportAsTempoMap: false,
    midiImportTempoMapTempo: true,
    midiImportTempoMapTimeSignature: true,
    midiImportTempoMapKeySignature: false,
};

/** 内存附件（字节以 base64 形式留在模块变量里，仅本次 dev 会话有效）。 */
interface MockNotebookAsset {
    kind: "image" | "clip_payload";
    ext: string;
    mime: string;
    dataBase64: string;
    meta: unknown;
}

/**
 * 假记事本附件库。
 *
 * 真实后端把字节随工程文件内嵌落盘；mock 没有磁盘，但"插入图片 → 立刻读回"
 * 是记事本的核心路径，返回空库会让该路径在浏览器里无法验证。
 */
const mockNotebookAssets = new Map<string, MockNotebookAsset>();

/** base64 字节数的近似换算（供 byteLen 展示用，mock 不追求精确）。 */
function base64ByteLen(dataBase64: string): number {
    return Math.max(0, Math.floor((dataBase64.length * 3) / 4));
}

/**
 * 构造假的时间轴状态。
 *
 * 数据刻意覆盖渲染器的各条视觉分支：淡入淡出（含不同形状 / 方向）、增益与速率
 * 标签、静音徽标、编组徽标、首尾相接的分隔缝、超长名称截断、多 take 展开。
 *
 * 特殊说明：每次调用**重新构造** tracks / clips，但 `selected_track_id` /
 * `selected_clip_id` 取模块级的"记住值"（首次调用回落到首个轨道 / 无选中），
 * 使 `select_clip` / `select_track` 与 `get_timeline_state` 返回同一个可变会话态
 * ——与真实后端 `to_payload()` 的行为一致。
 *
 * @returns 与后端 `TimelineState` 结构一致的假状态。
 */
function buildMockTimeline(): Record<string, unknown> {
    const tracks = Array.from({ length: 6 }, (_, index) => ({
        id: `track-${index + 1}`,
        name: `Track ${index + 1}`,
        parent_id: null,
        depth: 0,
        child_track_ids: [],
        muted: index === 2,
        solo: false,
        volume: 1,
        compose_enabled: true,
        pitch_analysis_algo: "rmvpe",
        color: TRACK_COLORS[index % TRACK_COLORS.length],
    }));

    const clips: Record<string, unknown>[] = [];
    for (let trackIndex = 0; trackIndex < tracks.length; trackIndex += 1) {
        const trackId = tracks[trackIndex].id;
        const clipCount = 3 + (trackIndex % 3);
        /**
         * 轨道 0 刻意做成**重叠 + 交叉淡化**，用于验证内核的重叠区交互
         * （前一个 clip 的右缘 / 淡出控件在重叠区内是否可达）。
         * 其余轨道保持 0.5s 间隔，覆盖「不重叠」这条常规路径。
         */
        const overlaps = trackIndex === 0;
        const gapSec = overlaps ? -0.8 : 0.5;
        let cursor = 2 + trackIndex * 1.5;
        for (let index = 0; index < clipCount; index += 1) {
            const lengthSec = 4 + ((index * 3 + trackIndex) % 9);
            const isFirst = index === 0;
            const isLast = index === clipCount - 1;
            clips.push({
                id: `${trackId}-clip-${index + 1}`,
                track_id: trackId,
                name:
                    index === 1 && trackIndex === 0
                        ? "一个非常长的音频片段名称用于验证截断行为"
                        : `Take ${index + 1}`,
                start_sec: cursor,
                length_sec: lengthSec,
                color: TRACK_COLORS[trackIndex % TRACK_COLORS.length],
                source_path: `/mock/audio-${trackIndex + 1}.wav`,
                source_start_sec: 0,
                source_end_sec: lengthSec,
                duration_sec: lengthSec + 8,
                duration_frames: Math.round((lengthSec + 8) * 44100),
                source_sample_rate: 44100,
                gain: index % 3 === 0 ? 0 : index % 3 === 1 ? -3.5 : 4.2,
                muted: index === 2 && trackIndex === 1,
                playback_rate: index === 1 ? 1.5 : 1,
                clip_playback_rate: index === 1 ? 1.5 : 1,
                reversed: false,
                loop_enabled: false,
                fade_in_sec: isFirst ? 0.6 : 0,
                fade_out_sec: isLast ? 0.8 : 0,
                fade_in_shape: trackIndex % 7,
                fade_out_shape: (trackIndex + 3) % 7,
                fade_in_dir: trackIndex % 3 === 0 ? -0.5 : 0,
                fade_out_dir: trackIndex % 3 === 1 ? 0.4 : 0,
                // 重叠轨道用自动交叉淡化覆盖重叠长度（与真实工程一致：重叠处
                // 通常就是交叉淡化），从而在重叠区内产生淡入 / 淡出控件。
                auto_fade_in_sec: overlaps && !isFirst ? 0.8 : 0,
                auto_fade_out_sec: overlaps && !isLast ? 0.8 : 0,
                snap_offset_sec: index === 1 ? 0.25 : 0,
                // 编组覆盖两条轨道（轨道 4 与轨道 5 的首个 clip 同属 group-a）：
                // 只有单成员编组时无法验证「同组联动」——拖动一个成员必须带动
                // 另一个，是内核参与集合展开的关键回归点。
                group_id:
                    (trackIndex === 3 && index === 0) ||
                    (trackIndex === 4 && index === 0) ||
                    (trackIndex === 3 && index === 1)
                        ? "group-a"
                        : undefined,
                // `midi_note_count` **只在 MIDI clip 上出现**（前端以
                // `midiNoteCount != null` 判定，0 也算 MIDI）。音频 clip 必须省略该
                // 字段：写成 0 会让它们被判成 MIDI clip，从而走 pitch 分支的
                // header 布局（隐藏增益旋钮与共振峰徽标），浏览器里就验证不到
                // 音频 clip 的完整控件。
                //
                // **必须同时给 `midi_note_data`**：`midi_note_count` 只决定"这是不是
                // MIDI clip"（header 走 pitch 布局），折线内容完全来自音符数据。
                // 此前只写 count 时该 clip 在时间线上渲染为空白 —— 与真实工程里
                // 「音高参考块内部为空」的缺陷**视觉上无法区分**，因此这个回归
                // 在浏览器里一直复现不出来（排查时曾因此误判为渲染层没画）。
                //
                // `pitch_range` 取 `0..127`（**绝对音高域**）：与后端
                // `convert_clips_to_pitch_reference` 给音高参考块写入的值一致。
                // 音频 clip 的 `-24..24` 是半音偏移语义，套到音高折线上会把
                // 所有音符顶到上边距（折线被压成一条直线）。
                ...(trackIndex === 2 && index === clipCount - 1
                    ? {
                          midi_note_count: 8,
                          pitch_range: { min: 0, max: 127 },
                          midi_note_data: [
                              { start_sec: 0, end_sec: 0.5, note: 60, velocity: 100, channel: 0 },
                              { start_sec: 0.5, end_sec: 1.0, note: 64, velocity: 100, channel: 0 },
                              { start_sec: 1.0, end_sec: 1.6, note: 67, velocity: 100, channel: 0 },
                              { start_sec: 1.6, end_sec: 2.2, note: 72, velocity: 100, channel: 0 },
                              { start_sec: 2.2, end_sec: 2.8, note: 69, velocity: 100, channel: 0 },
                              { start_sec: 2.8, end_sec: 3.4, note: 65, velocity: 100, channel: 0 },
                              { start_sec: 3.4, end_sec: 4.0, note: 62, velocity: 100, channel: 0 },
                              { start_sec: 4.0, end_sec: 4.6, note: 60, velocity: 100, channel: 0 },
                          ],
                      }
                    : {}),
                // 多 Take：轨道 0 的首个 clip 平铺两条 Take（其余保持单 Take）——
                // 内核的 lane 分界线绘制与「点击 inactive lane 切换活跃 Take」
                // 需要多 Take 数据才能验证。
                takes:
                    trackIndex === 0 && index === 0
                        ? [
                              {
                                  id: `${trackId}-take-1`,
                                  name: "Take 1",
                                  source_path: `/mock/audio-${trackIndex + 1}.wav`,
                                  channel_mode:
                                      mockTakeChannelModes.get(`${trackId}-take-1`) ?? 0,
                              },
                              {
                                  id: `${trackId}-take-2`,
                                  name: "Take 2",
                                  source_path: `/mock/audio-${trackIndex + 1}-alt.wav`,
                                  channel_mode:
                                      mockTakeChannelModes.get(`${trackId}-take-2`) ?? 0,
                              },
                          ]
                        : [],
                active_take_id: trackIndex === 0 && index === 0 ? `${trackId}-take-1` : undefined,
            });
            cursor += lengthSec + gapSec;
        }
    }

    return {
        ok: true,
        tracks,
        clips,
        selected_track_id: mockSelectedTrackId ?? tracks[0].id,
        selected_clip_id: mockSelectedClipId,
        bpm: MOCK_BPM,
        playhead_sec: 12.5,
        project_sec: MOCK_PROJECT_SEC,
        project: {
            name: "Mock Project",
            path: "/mock/project.hsp",
            dirty: false,
            recent: [],
            beats_per_bar: 4,
            time_signature_denominator: 4,
            grid_size: "1/4",
        },
        missing_files: [],
        skipped_files: [],
        disabled_group_ids: [],
        tempo_map: null,
    };
}

/** 显式实现的 mock 方法表；未命中的方法由 Proxy 兜底。 */
function buildHandlers(): Record<string, (...args: unknown[]) => unknown> {
    /**
     * 静态 clips（供 `analyze_clip_silence` 这类只读几何查询使用）。
     *
     * 选中态（`mockSelectedClipId` / `mockSelectedTrackId`）**不**缓存进来：
     * 它们随 `select_clip` / `select_track` 变化，快照必须每次重建才能反映最新值。
     */
    const timeline = buildMockTimeline();
    return {
        ping: () => ({ ok: true }),
        get_runtime_info: () => ({
            ok: true,
            device: "mock",
            model_loaded: false,
            audio_loaded: false,
            has_synthesized: false,
            is_playing: false,
            playback_target: null,
            gpuBackend: "CPU",
        }),
        /**
         * 取时间轴全量快照。
         *
         * 与 `select_clip` / `select_track` 返回**同一形状**：真实后端这三个命令都
         * 返回 `to_payload()`，其中**总是**带 `selected_clip_id` /
         * `selected_track_id`。前端的 `selectTrackRemote.fulfilled` 正是在
         * `selected_clip_id !== undefined` 时才会用后端记住的选中覆盖前端状态，
         * 因此 mock 若省略该字段就会**掩盖**"点空白切轨时选中被复活"的缺陷。
         */
        get_timeline_state: () => buildMockTimeline(),
        /**
         * 静音检测（干跑）：返回**固定几何**的假静音区。
         *
         * 真实后端做的是能量分析（浏览器里无法复现），而内核的「静音预览红色覆盖层」
         * 只消费区域几何——给固定区域即可在 mock 下验证绘制与钳制路径。
         * 区域取每个 clip 的中段 30% 与后段 15%（全部落在 clip 本体之内）。
         */
        analyze_clip_silence: (...args: unknown[]) => {
            const clipIds = Array.isArray(args[0]) ? (args[0] as unknown[]).map(String) : [];
            const clips = (timeline.clips ?? []) as Array<Record<string, unknown>>;
            const reports = clipIds.flatMap((clipId) => {
                const clip = clips.find((item) => item.id === clipId);
                if (clip === undefined) return [];
                const startSec = Number(clip.start_sec) || 0;
                const lengthSec = Number(clip.length_sec) || 0;
                const regions = [
                    {
                        startSec: startSec + lengthSec * 0.25,
                        endSec: startSec + lengthSec * 0.55,
                    },
                    {
                        startSec: startSec + lengthSec * 0.8,
                        endSec: startSec + lengthSec * 0.95,
                    },
                    // 第三段**故意越界**（后端区域越过 clip 末端）：用于验证渲染端
                    // 把区间钳制在 clip 本体内——不钳制会把红色画到相邻 clip 上。
                    {
                        startSec: startSec + lengthSec * 1.05,
                        endSec: startSec + lengthSec * 1.2,
                    },
                ];
                return [
                    {
                        clipId,
                        ok: true,
                        fullySilent: false,
                        totalSilentSec: regions.reduce(
                            (sum, region) => sum + (region.endSec - region.startSec),
                            0,
                        ),
                        regions,
                    },
                ];
            });
            return { ok: true, reports };
        },
        get_playback_state: () => ({
            ok: true,
            is_playing: false,
            playhead_sec: 12.5,
            playback_anchor_sec: 12.5,
            bpm: MOCK_BPM,
            project_sec: MOCK_PROJECT_SEC,
            duration_sec: MOCK_PROJECT_SEC,
        }),
        get_ui_settings: () => ({ ...mockUiSettings }),
        get_auto_backup_settings: () => ({ ok: true, enabled: false, intervalMinutes: 10 }),
        get_recording_settings: () => ({ ok: true }),
        get_recording_state: () => ({ ok: true, isRecording: false }),
        get_recording_devices: () => ({ ok: true, devices: [] }),
        get_recording_apps: () => ({ ok: true, apps: [] }),
        get_export_audio_defaults: () => ({ ok: true }),
        get_gpu_devices: () => ({ ok: true, devices: [] }),
        get_onnx_status: () => ({ ok: true, available: false }),
        get_dml_adapters: () => ({ ok: true, adapters: [] }),
        // ── 波形（假数据）────────────────────────────────────────────
        get_waveform_mipmap_binary: (...args: unknown[]) => {
            const sourcePath = String(args[0] ?? "");
            const level = Number(args[1] ?? 0);
            const levels = getMockWaveformLevels(sourcePath);
            return levels[level] ?? "";
        },
        batch_get_waveform_mipmap: (...args: unknown[]) => {
            const paths = Array.isArray(args[0]) ? (args[0] as unknown[]) : [];
            const out: Record<string, [string, string, string]> = {};
            for (const path of paths) {
                const key = String(path);
                out[key] = getMockWaveformLevels(key);
            }
            return out;
        },
        preload_waveform_mipmap: () => ({ ok: true }),
        /**
         * 处理器参数清单（dev-only）。
         *
         * 【为什么必须有它】`getVisibleSecondaryParamIds` 依赖后端返回的处理器参数
         * 才能列出副参数曲线。此前 mock 未实现本方法，参数编辑器里**只有主参数一条
         * 曲线**——而"未选中的其它参数线被染色"这类缺陷的前提就是**有多条曲线**。
         *
         * 【为什么必须含 volume / pan / dyn】混音级参数是所有算法共通的
         * （见后端 `renderer/common_params.rs` 的 `COMMON_MIX_PARAMS`），前端据此
         * 渲染工具栏最右侧的「音量 / 动态」药丸。mock 若只给算法专有参数，音量 /
         * 动态面板在浏览器里**根本无法进入**，而「可听结果波形」（响度映射）正是
         * 只在这两个面板下才挂载的路径 —— 与它相关的性能与渲染缺陷都无法验证。
         * 这里按后端同一份值域（volume 0..2、pan −1..1、dyn 0..2）列出。
         */
        get_processor_params: () => [
            {
                id: "tension",
                display_name: "Tension",
                group: "Voice",
                kind: {
                    type: "automation_curve",
                    unit: "",
                    default_value: 0.5,
                    min_value: 0,
                    max_value: 1,
                },
            },
            {
                id: "breathiness",
                display_name: "Breathiness",
                group: "Voice",
                kind: {
                    type: "automation_curve",
                    unit: "",
                    default_value: 0.5,
                    min_value: 0,
                    max_value: 1,
                },
            },
            {
                id: "volume",
                display_name: "Volume",
                group: "Mix",
                kind: {
                    type: "automation_curve",
                    unit: "×",
                    default_value: 1.0,
                    min_value: 0.0,
                    max_value: 2.0,
                },
            },
            {
                id: "pan",
                display_name: "Pan",
                group: "Mix",
                kind: {
                    type: "automation_curve",
                    unit: "",
                    default_value: 0.0,
                    min_value: -1.0,
                    max_value: 1.0,
                },
            },
            {
                id: "dyn",
                display_name: "Dynamics",
                group: "Mix",
                kind: {
                    type: "automation_curve",
                    unit: "×",
                    default_value: 1.0,
                    min_value: 0.0,
                    max_value: 2.0,
                },
            },
        ],
        /**
         * 参数曲线（dev-only 合成数据）。
         *
         * 【为什么必须有它】此前 mock 未实现本方法，Proxy 兜底返回 `{ ok: true }`
         * ——而 `usePianoRollData` 需要 `orig` / `edit` 才能画出任何曲线。结果是
         * **参数编辑器里根本没有曲线可看**，任何与曲线渲染相关的缺陷（例如"选区
         * 存在时未选中的线被染色"）在浏览器里都无法复现，只能靠读代码推断。
         * 这里合成一条有真实形态的曲线（长趋势 + 颤音 + 跳变），使曲线层与生产
         * 环境一样被实际绘制。
         *
         * 【混音级参数走各自的量纲】volume / pan / dyn 的值域与语义都和音高不同
         * （见后端 `renderer/common_params.rs`）：音量是 0..2 的乘性增益、动态是
         * 0..2 的**目标电平**（其 `orig` 是原声基线 —— 正是响度映射用来算增益的
         * 分母）。若沿用音高曲线，音量面板会画出 −6..+10 的越界值，且动态基线为 0
         * 会走"真静音"分支，波形的可听结果映射形同未验证。
         */
        get_param_frames: (...args: unknown[]) => {
            const trackId = String(args[0] ?? "");
            const param = String(args[1] ?? "pitch");
            const startFrame = Math.max(0, Math.floor(Number(args[2] ?? 0)));
            const frameCount = Math.max(1, Math.floor(Number(args[3] ?? 1)));
            const stride = Math.max(1, Math.floor(Number(args[4] ?? 1)));
            const fpMs = 5;
            const orig: number[] = new Array(frameCount);
            const edit: number[] = new Array(frameCount);
            for (let i = 0; i < frameCount; i += 1) {
                const frame = startFrame + i * stride;
                const t = (frame * fpMs) / 1000;
                if (param === "volume") {
                    // 0..2 的乘性音量曲线（含缓慢起伏与一小段静音）。
                    const v = 1 + 0.45 * Math.sin(t * 0.35) + 0.15 * Math.sin(t * 2.1);
                    orig[i] = v;
                    edit[i] = v;
                    continue;
                }
                if (param === "dyn") {
                    // 原声基线（0.2..0.9 缓慢起伏）+ 目标电平（基线 × 0.4..1.6）。
                    const base = 0.5 + 0.35 * Math.sin(t * 0.21) + 0.1 * Math.sin(t * 1.3);
                    const baseline = Math.max(0.05, base);
                    orig[i] = baseline;
                    edit[i] = baseline * (1 + 0.6 * Math.sin(t * 0.17));
                    continue;
                }
                // 长趋势 + 颤音 + 周期性跳变：让曲线有真实的拐角与折返
                const base =
                    MOCK_PITCH_CENTER +
                    Math.sin(t * 0.55) * 6 +
                    Math.sin(t * 7.9) * 0.9 +
                    (Math.floor(t / 6) % 2 === 0 ? 0 : 4);
                orig[i] = base;
                edit[i] = base + Math.sin(t * 1.7) * 1.2;
            }
            return {
                ok: true,
                root_track_id: trackId,
                param,
                frame_period_ms: fpMs,
                start_frame: startFrame,
                orig,
                edit,
                reference_kind: "source_curve",
                analysis_pending: false,
                analysis_progress: 1,
                pitch_edit_user_modified: false,
                pitch_edit_backend_available: true,
            };
        },
        set_transport: () => ({ ok: true }),
        set_project_length: () => ({ ok: true }),
        /**
         * 选中 clip：记住选中并返回全量快照。
         *
         * 与真实后端同形（`select_clip` → `tl.select_clip(...)` → `to_payload()`）：
         * - 传 `null` 表示清空选中（前端"点空白"用的是**本地** reducer，不会走到
         *   这里，所以后端会一直记着旧值——这正是需要保真的语义）；
         * - 选中一条存在的 clip 时，真实后端还会把 `selected_track_id` 跟到该
         *   clip 所在轨道（`state.rs::select_clip`）；前端以此决定参数编辑器的
         *   编辑目标，不保真会让"点 clip 切轨"这条路径在 mock 下失真。
         */
        select_clip: (...args: unknown[]) => {
            const clipId = args[0] == null ? null : String(args[0]);
            mockSelectedClipId = clipId;
            if (clipId !== null) {
                const clips = buildMockTimeline().clips as Array<Record<string, unknown>>;
                const trackId = clips.find((clip) => clip.id === clipId)?.track_id;
                if (trackId != null) mockSelectedTrackId = String(trackId);
            }
            return buildMockTimeline();
        },
        /**
         * 切换当前轨道：记住轨道焦点并返回全量快照，**刻意不修改
         * `mockSelectedClipId`**——真实后端的 `select_track` 也只改
         * `selected_track_id`，快照里的 `selected_clip_id` 仍是上次选中的 clip。
         * 前端 `selectTrackRemote.fulfilled` 会据此"恢复"该 clip，因此这条路径
         * 必须能真正观察到该字段，否则缺陷不可复现。
         */
        select_track: (...args: unknown[]) => {
            const trackId = args[0] == null ? null : String(args[0]);
            if (trackId !== null) mockSelectedTrackId = trackId;
            return buildMockTimeline();
        },
        // 与真实后端同语义：部分字段合并进内存"配置文件"。pywebview 通道的
        // 调用形状是 invoke("save_ui_settings", { settings })，第一参即包裹对象。
        save_ui_settings: (...args: unknown[]) => {
            const wrapper = (args[0] ?? {}) as { settings?: Record<string, unknown> };
            Object.assign(mockUiSettings, wrapper.settings ?? {});
            return { ok: true };
        },
        // ── 渲染缓存 ────────────────────────────────────────────────
        // mock 没有磁盘缓存：返回全零但字段齐全的统计（enabled=true 保持面板
        // 可见），清理/打开目录按"无事发生"成功返回。
        get_render_cache_stats: () => ({
            ok: true,
            enabled: true,
            dir: "/mock/render-cache",
            writable: true,
            totalBytes: 0,
            entries: 0,
            byKind: [],
            sessionHits: 0,
            sessionMisses: 0,
            sessionStored: 0,
            sessionWriteErrors: 0,
            maxSizeBytes: 4096 * 1024 * 1024,
            maxAgeDays: 90,
        }),
        clear_render_cache: () => ({ ok: true, removedFiles: 0, removedBytes: 0 }),
        open_render_cache_dir: () => ({ ok: true, path: "/mock/render-cache" }),
        // ── Take 声道模式 / 假立体声扫描 ────────────────────────────
        // 与真实后端同形：返回**携带刚写入字段**的全量时间轴快照（写入由
        // mockTakeChannelModes 记住，buildMockTimeline 写进 take 载荷）。
        set_clip_take_channel_mode: (...args: unknown[]) => {
            const takeId = String(args[1] ?? "");
            if (takeId) mockTakeChannelModes.set(takeId, Number(args[2] ?? 0));
            return buildMockTimeline();
        },
        scan_and_convert_fake_stereo: () => {
            // mock 数据里没有可折叠素材：scanned 报全部 clip，converted 恒 0
            //（真实转换在浏览器里无法复现，字段齐全即可让状态栏分支走到）。
            const clips = (buildMockTimeline().clips ?? []) as Array<Record<string, unknown>>;
            return { ok: true, scanned: clips.length, converted: 0 };
        },
        // ── 工程笔记 / 参数手势 ─────────────────────────────────────
        set_project_notes: (...args: unknown[]) => ({
            ok: true,
            project: { notes_markdown: String(args[0] ?? "") },
        }),
        record_param_selection_step: () => ({ ok: true }),
        // 真实转换在浏览器里无法复现：字段齐全的"零转换"结果让调用方走成功分支。
        convert_mix_param: () => ({ ok: true, convertedFrames: 0, skippedFrames: 0 }),
        // ── 记事本（内存附件库 / 剪贴板暂存）────────────────────────
        notebook_put_asset: (...args: unknown[]) => {
            const assetId = String(args[0] ?? "");
            const dataBase64 = String(args[4] ?? "");
            if (!assetId || !dataBase64) {
                return { ok: false, error: "mock: assetId / dataBase64 required" };
            }
            mockNotebookAssets.set(assetId, {
                kind: args[1] === "clip_payload" ? "clip_payload" : "image",
                ext: String(args[2] ?? ""),
                mime: args[3] == null ? "" : String(args[3]),
                dataBase64,
                meta: args[5] ?? null,
            });
            return { ok: true, assetId, byteLen: base64ByteLen(dataBase64) };
        },
        notebook_read_asset: (...args: unknown[]) => {
            const asset = mockNotebookAssets.get(String(args[0] ?? ""));
            if (!asset) return { ok: true, missing: true };
            return { ok: true, mime: asset.mime || undefined, base64: asset.dataBase64 };
        },
        notebook_list_assets: () => ({
            ok: true,
            assets: [...mockNotebookAssets.entries()].map(([id, asset]) => ({
                id,
                kind: asset.kind,
                ext: asset.ext,
                mime: asset.mime,
                byteLen: base64ByteLen(asset.dataBase64),
                createdAtMs: Date.now(),
                meta: asset.meta,
                hasData: true,
            })),
        }),
        notebook_remove_asset: (...args: unknown[]) => ({
            ok: true,
            removed: mockNotebookAssets.delete(String(args[0] ?? "")),
        }),
        notebook_prune_assets: () => {
            const removed = mockNotebookAssets.size;
            mockNotebookAssets.clear();
            return { ok: true, removed };
        },
        // mock 里没有本地文件系统：读文件按失败返回（调用方有各自的错误兜底）。
        notebook_read_file_base64: () => ({ ok: false, error: "mock: fs read not supported" }),
        // 剪贴板载荷是 MessagePack 字节流，mock 无法解析出摘要；按"暂存不可读"
        // 返回（写入成功、读取空），界面显示为空暂存而非报错。
        notebook_write_clipboard_payload: () => ({ ok: true }),
        notebook_read_clipboard_payload: () => ({ ok: true, available: false }),
        notebook_read_clipboard_image: () => ({ ok: true, available: false }),
        // 涉及系统对话框的导出 / 另存：按"用户取消"返回，调用方当作无害 no-op。
        notebook_export_document: () => ({ ok: true, canceled: true }),
        notebook_save_asset_as: () => ({ ok: true, canceled: true }),
        begin_undo_group: () => ({ ok: true }),
        end_undo_group: () => ({ ok: true }),
    };
}

/**
 * 安装 mock 后端。
 *
 * 流程：构造方法表 → 用 Proxy 包一层（未实现的方法返回 `{ ok: true }`）→ 挂到
 * `window.pywebview.api`。`services/invoke.ts` 会优先命中已存在的 `window.pywebview.api`，
 * 因此不会进入 1.5s 的等待分支。
 *
 * 特殊说明：`{ ok: true }` 而不是 `null`——前端多数调用点会读 `result.ok` 或直接
 * 解构字段，返回对象比返回 null 更不容易触发空指针。
 *
 * @returns 无返回值。
 */
/**
 * 记录写操作的入参（截断深拷贝）到 `window.__mockArgs`。
 *
 * 用途：调几何相关手势（trim / 拖拽）时需要核对**实际提交的数值**——截图只能看
 * 趋势，方向与数值必须读参数。只记 `set_ / move_ / update_ / save_` 前缀的方法，
 * 避免把只读查询的参数也堆进来。
 *
 * @param method 被调用的后端方法名。
 * @param args 位置参数。
 */
function recordWriteArgs(method: string, args: unknown[]): void {
    if (!/^(set_|move_|update_|save_)/.test(method) || args.length === 0) return;
    const holder = window as unknown as {
        __mockArgs?: Record<string, unknown>;
        __mockArgsAll?: Record<string, unknown[]>;
    };
    const record = holder.__mockArgs ?? {};
    try {
        record[method] = JSON.parse(JSON.stringify(args[0]));
    } catch {
        record[method] = String(args[0]);
    }
    holder.__mockArgs = record;
    // 完整位置参数（`__mockArgs` 只记第一个参数，多参数命令——如
    // `set_clip_active_take(clipId, takeId, checkpoint)`——无法只靠它核对）。
    const all = holder.__mockArgsAll ?? {};
    try {
        all[method] = JSON.parse(JSON.stringify(args)) as unknown[];
    } catch {
        all[method] = args.map((arg) => String(arg));
    }
    holder.__mockArgsAll = all;
}

export function installMockBackend(): void {
    const handlers = buildHandlers();
    /** 调用轨迹（调试用：`window.__mockCalls` 可读出被调用的方法顺序）。 */
    const calls: string[] = [];
    (window as unknown as { __mockCalls?: string[] }).__mockCalls = calls;
    const api = new Proxy(handlers, {
        get(target, property) {
            if (typeof property !== "string") return undefined;
            // 关键：绝不能把 then / catch / finally 暴露成函数。invoke 层会
            // `await` 本对象，Promise 解析协议检测到 thenable 就会调用 `then`
            // 等待其回调——而假实现不会回调 resolve，导致 await 永久挂起
            // （表现为所有 invoke 卡死、界面停在"正在刷新运行时…"）。
            if (property === "then" || property === "catch" || property === "finally") {
                return undefined;
            }
            const existing = target[property];
            if (existing !== undefined) {
                return (...args: unknown[]) => {
                    calls.push(property);
                    recordWriteArgs(property, args);
                    return existing(...args);
                };
            }
            // 未实现的方法：返回一个恒为 `{ ok: true }` 的假实现。
            return (...args: unknown[]) => {
                calls.push(`${property}(fallback)`);
                recordWriteArgs(property, args);
                return { ok: true };
            };
        },
    });

    (window as unknown as { pywebview?: { api: unknown } }).pywebview = { api };
}
