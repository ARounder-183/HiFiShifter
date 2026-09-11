/**
 * mockBackend — 纯浏览器开发模式下的后端替身（仅 dev，仅显式开启）。
 *
 * 【主要内容】
 * 在 `window.pywebview.api` 上安装一套假后端实现：启动路径（运行时信息 / UI 设置 /
 * 时间轴状态 / 播放状态）返回构造好的假工程数据，其余未显式实现的方法由 Proxy
 * 兜底返回 `{ ok: true }`，避免任何一次 invoke 抛错中断启动。
 *
 * 【作用】
 * 本项目前端依赖 pywebview / Tauri 后端（`services/invoke.ts` 的 pywebview 分支），
 * 直接 `npm run dev` 在浏览器里打开会因 `Python API not available` 而无法进入界面，
 * 使得「改渲染 / 调样式」这类纯前端工作无法在浏览器里快速迭代与截图验证。
 * 本模块提供最小可用的后端替身，让时间轴面板能在浏览器里完整渲染。
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

/** 轨道配色（与生产默认调色板同量级，便于视觉对比）。 */
const TRACK_COLORS = ["#4b8fd1", "#5fb27a", "#d19a4b", "#c25f8f", "#7a6fd1", "#4bb0b0"];

/**
 * 构造假的时间轴状态。
 *
 * 数据刻意覆盖渲染器的各条视觉分支：淡入淡出（含不同形状 / 方向）、增益与速率
 * 标签、静音徽标、编组徽标、首尾相接的分隔缝、超长名称截断、多 take 展开。
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
                auto_fade_in_sec: 0,
                auto_fade_out_sec: 0,
                snap_offset_sec: index === 1 ? 0.25 : 0,
                group_id: trackIndex === 3 && index === 1 ? "group-a" : undefined,
                midi_note_count: 0,
                takes: [],
                active_take_id: undefined,
            });
            cursor += lengthSec + 0.5;
        }
    }

    return {
        ok: true,
        tracks,
        clips,
        selected_track_id: tracks[0].id,
        selected_clip_id: null,
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
        get_timeline_state: () => timeline,
        get_playback_state: () => ({
            ok: true,
            is_playing: false,
            playhead_sec: 12.5,
            playback_anchor_sec: 12.5,
            bpm: MOCK_BPM,
            project_sec: MOCK_PROJECT_SEC,
            duration_sec: MOCK_PROJECT_SEC,
        }),
        get_ui_settings: () => ({ ok: true }),
        get_auto_backup_settings: () => ({ ok: true, enabled: false, intervalMinutes: 10 }),
        get_recording_settings: () => ({ ok: true }),
        get_recording_state: () => ({ ok: true, isRecording: false }),
        get_recording_devices: () => ({ ok: true, devices: [] }),
        get_recording_apps: () => ({ ok: true, apps: [] }),
        get_export_audio_defaults: () => ({ ok: true }),
        get_gpu_devices: () => ({ ok: true, devices: [] }),
        get_onnx_status: () => ({ ok: true, available: false }),
        get_dml_adapters: () => ({ ok: true, adapters: [] }),
        get_waveform_manifest: () => ({ ok: true, entries: [], items: [] }),
        set_transport: () => ({ ok: true }),
        set_project_length: () => ({ ok: true }),
        select_clip: () => ({ ok: true }),
        select_track: () => ({ ok: true }),
        save_ui_settings: () => ({ ok: true }),
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
                    return existing(...args);
                };
            }
            // 未实现的方法：返回一个恒为 `{ ok: true }` 的假实现。
            return (...args: unknown[]) => {
                calls.push(`${property}(fallback)`);
                void args;
                return { ok: true };
            };
        },
    });

    (window as unknown as { pywebview?: { api: unknown } }).pywebview = { api };
}
