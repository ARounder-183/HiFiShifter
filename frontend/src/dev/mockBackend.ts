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
function buildMockPeaks(peakCount: number, seed: number): {
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
        get_waveform_manifest: (...args: unknown[]) => {
            const sourcePath = String(args[0] ?? "");
            const totalFrames = MOCK_WAVEFORM_DURATION_SEC * MOCK_WAVEFORM_SAMPLE_RATE;
            return {
                sourcePath,
                revision: "mock-1",
                sampleRate: MOCK_WAVEFORM_SAMPLE_RATE,
                totalFrames,
                channels: 1,
                durationSec: MOCK_WAVEFORM_DURATION_SEC,
                tilePeaks: 0,
                levels: MOCK_DIVISION_FACTORS.map((divisionFactor, level) => ({
                    level,
                    divisionFactor,
                    peakCount: Math.floor(totalFrames / divisionFactor),
                    tileCount: 0,
                })),
            };
        },
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
    const holder = window as unknown as { __mockArgs?: Record<string, unknown> };
    const record = holder.__mockArgs ?? {};
    try {
        record[method] = JSON.parse(JSON.stringify(args[0]));
    } catch {
        record[method] = String(args[0]);
    }
    holder.__mockArgs = record;
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
