/**
 * dev-only：一键生成大规模「性能工程」，供真机 Profile 时间线缩放/平移。
 *
 * 【主要内容】
 * - `buildPerfTimeline()`：按给定规模合成一份后端 `TimelineState` 快照
 *   （若干轨道 × 每轨若干 clip，等长、首尾相接）；
 * - `installSyntheticPeakSource()`：把 `waveformMipmapStore` 对 `hs-perf://`
 *   前缀的取数短路到合成 peaks，掐断指向后端的 IPC；
 * - `generatePerfProject()`：dispatch `applyTimelinePayload` 落地并返回规模摘要。
 *
 * 【作用】真机 Performance 录制需要「10 轨 / 400 clip / 全览缩放」这种规模，
 * 而手工导入 400 个音频文件不现实；且在没有真实音频文件时，mipmap store 会
 * 把假路径当作"尚未就绪"、按 3 秒冷却反复重试 IPC。本模块用合成 peaks 让
 * **波形开销与真实数据一致**（峰值密度与切片公式与
 * `waveformMipmapStore.getBestSliceView()` 逐字一致），Profile 结果才可信。
 *
 * 【与其他模块的关系】
 * - 上游：`main.tsx` 在 `import.meta.env.DEV` 下动态 import，生产构建不打包。
 * - 横向：`features/session/sessionSlice.applyTimelinePayload`（唯一写入口）、
 *   `utils/waveformMipmapStore`（运行时猴补丁短路）、
 *   `waveform/perfFixtures.createSyntheticPeakSource`（合成数据源）。
 * - 消费方：无生产消费方。仅经 `window.__hsPerf` 或 `?perf=N` 触发。
 *
 * 【重要约束】生成的工程是**只读观测态**：后端并不认识这些 clip / track，
 * 任何触发 remote thunk 的操作（点选、拖动、Ctrl+Z、播放、录音、fetchTimeline…）
 * 都会用后端快照整体覆写 `state.clips`，把性能工程抹掉。
 * **测量期间只做滚动与缩放，不要编辑。**
 */

import { store } from "../app/store";
import { applyTimelinePayload } from "../features/session/sessionSlice";
import type { TimelineClip, TimelineState, TimelineTrack } from "../types/api";
import { waveformMipmapStore } from "../utils/waveformMipmapStore";
import { startFrameProfiler, stopFrameProfiler } from "./frameProfiler";
import { PERF_GL_CLIP_BODIES_KEY } from "../components/layout/timeline/runtime/timelineClipGlRenderer.js";
import { createSyntheticPeakSource, type SyntheticPeakSource } from "../waveform/perfFixtures";

/** 合成源路径前缀：被本模块短路、绝不打到后端。 */
export const PERF_SOURCE_PREFIX = "hs-perf://";

/** 合成素材采样率，与主流音频一致。 */
const DEFAULT_SAMPLE_RATE = 44100;

/** 规模选择的持久化键：写后重载页面即可自动重建（见 `installPerfProjectDevtools`）。 */
const PERSIST_KEY = "hifishifter.perfProject";

/** 帧率探针开关 key（探针本身也只在 dev 模式加载）。 */
const FRAME_PROFILER_KEY = "hifishifter.frameProfiler";

/** 解析持久化的选项 JSON；非法 JSON 返回 `null` 由调用方忽略。 */
function safeParseJson(raw: string): PerfProjectOptions | null {
    try {
        const parsed: unknown = JSON.parse(raw);
        return typeof parsed === "object" && parsed !== null
            ? (parsed as PerfProjectOptions)
            : null;
    } catch {
        return null;
    }
}

/** 默认轨道配色，与 `TrackList` 的轨道色板同源（取几种便于肉眼区分）。 */
const TRACK_COLORS = [
    "#4f8ef7",
    "#f2994a",
    "#27ae60",
    "#9b51e0",
    "#eb5757",
    "#2d9cdb",
    "#f2c94c",
    "#00b8a9",
    "#f178b6",
    "#6b7280",
];

export interface PerfProjectOptions {
    /** 轨道数，默认 10。 */
    trackCount?: number;
    /** 每轨 clip 数，默认 40（合计 400 clip）。 */
    clipsPerTrack?: number;
    /** 单个 clip 时长（秒），默认 60。 */
    clipLengthSec?: number;
    /** 同轨相邻 clip 之间的空隙（秒），默认 0（首尾相接，重叠检测最省）。 */
    gapSec?: number;
    /** 合成素材采样率，默认 44100。 */
    sampleRate?: number;
    /**
     * 手动淡入淡出长度（秒），默认 0。
     *
     * 设为 >0 可让 `drawFadeCurveStroke` 进入 Profile 样本——它在小缩放下
     * 被压成亚像素（几乎免费），但在中高缩放下单条最多细分 1200 点，是
     * clip 体画布里少数"随缩放急剧变贵"的分支，值得单独测一遍。
     */
    fadeSec?: number;
}

export interface PerfProjectSummary {
    trackCount: number;
    clipCount: number;
    clipLengthSec: number;
    gapSec: number;
    fadeSec: number;
    /** 工程总长（秒）。 */
    projectSec: number;
    /** 让全部内容装进视口所需的 pxPerSec（全览缩放档）。 */
    fitPxPerSec: number;
}

/** 被猴补丁的 store 方法形状（仅本模块内部使用，避免污染生产类型）。 */
interface MipmapStorePatchTarget {
    getBestSliceView: typeof waveformMipmapStore.getBestSliceView;
    batchPreload: (sourcePaths: readonly string[]) => Promise<unknown>;
    preload: (sourcePath: string) => Promise<unknown>;
}

let patchInstalled = false;
let syntheticSource: SyntheticPeakSource | null = null;

/**
 * 把 `hs-perf://` 前缀的取数短路到合成 peaks，并掐断相关预加载入口。
 *
 * 流程：
 * 1. 首次调用时保存 `getBestSliceView` / `batchPreload` / `preload` 的原实现；
 * 2. 前缀命中时直接返回合成切片，否则回落到原实现；
 * 3. 预加载入口过滤掉性能路径——否则 400 个假路径会每 3 秒触发一轮批量 IPC。
 *
 * 特殊说明：这是**运行时猴补丁**而非给生产代码加 hook，目的是把改动完全
 * 关在 dev-only 模块内；重复调用只会替换合成源（供改变素材时长后复用）。
 *
 * @param mediaDurationSec 合成素材时长（秒），须与 clip 长度一致。
 */
function installSyntheticPeakSource(mediaDurationSec: number): void {
    syntheticSource = createSyntheticPeakSource({
        sampleRate: DEFAULT_SAMPLE_RATE,
        mediaDurationSec,
    });
    if (patchInstalled) return;
    patchInstalled = true;

    const target = waveformMipmapStore as unknown as MipmapStorePatchTarget;
    const originalGetBestSliceView = target.getBestSliceView;
    const originalBatchPreload = target.batchPreload;
    const originalPreload = target.preload;

    target.getBestSliceView = function patchedGetBestSliceView(
        sourcePath,
        preferredLevel,
        startSec,
        durationSec,
    ) {
        if (sourcePath.startsWith(PERF_SOURCE_PREFIX) && syntheticSource !== null) {
            return syntheticSource.getPeaks(sourcePath, DEFAULT_SAMPLE_RATE, startSec, durationSec);
        }
        return originalGetBestSliceView.call(
            waveformMipmapStore,
            sourcePath,
            preferredLevel,
            startSec,
            durationSec,
        );
    };

    target.batchPreload = async function patchedBatchPreload(sourcePaths) {
        const realPaths = sourcePaths.filter((path) => !path.startsWith(PERF_SOURCE_PREFIX));
        if (realPaths.length === 0) return;
        return originalBatchPreload.call(waveformMipmapStore, realPaths);
    };

    target.preload = async function patchedPreload(sourcePath) {
        if (sourcePath.startsWith(PERF_SOURCE_PREFIX)) return;
        return originalPreload.call(waveformMipmapStore, sourcePath);
    };
}

/**
 * 合成一份后端 `TimelineState` 快照。
 *
 * 流程：按 `trackCount × clipsPerTrack` 铺开 clip，同轨内以
 * `clipLengthSec + gapSec` 为步距首尾相接，每个 clip 独占一个合成源文件。
 *
 * 特殊说明：字段全部按 `applyTimelineState()`（`sessionSlice.ts:1402`）的
 * 解析口径给出——它把 snake_case 载荷映射为 `ClipInfo`，任何缺失字段都会
 * 走 `?? 默认值` 分支，因此这里只需保证**波形可见的最小集**齐全：
 * `source_path` / `source_sample_rate` / `duration_sec` / `source_start_sec`
 * / `source_end_sec`（缺 `source_path` 会让 `clipToSceneClip()` 直接返回
 * `null`，波形完全不画，Profile 结果也就失去意义）。
 *
 * @param options 规模参数，缺省即 10 轨 × 40 clip × 60 秒。
 * @returns 可直接喂给 `applyTimelinePayload` 的快照。
 */
export function buildPerfTimeline(options: PerfProjectOptions = {}): TimelineState {
    const trackCount = Math.max(1, Math.trunc(options.trackCount ?? 10));
    const clipsPerTrack = Math.max(1, Math.trunc(options.clipsPerTrack ?? 40));
    const clipLengthSec = Math.max(0.1, options.clipLengthSec ?? 60);
    const gapSec = Math.max(0, options.gapSec ?? 0);
    const sampleRate = Number.isFinite(options.sampleRate)
        ? Math.max(1, options.sampleRate as number)
        : DEFAULT_SAMPLE_RATE;
    const fadeSec = Math.max(0, options.fadeSec ?? 0);

    const tracks: TimelineTrack[] = Array.from({ length: trackCount }, (_unused, index) => ({
        id: `perf-track-${index}`,
        name: `Perf ${index + 1}`,
        parent_id: null,
        depth: 0,
        child_track_ids: [],
        muted: false,
        solo: false,
        volume: 1,
        compose_enabled: false,
        pitch_analysis_algo: "nsf_hifigan_onnx",
        color: TRACK_COLORS[index % TRACK_COLORS.length] as string,
    }));

    const clips: TimelineClip[] = [];
    for (let trackIndex = 0; trackIndex < trackCount; trackIndex += 1) {
        for (let clipIndex = 0; clipIndex < clipsPerTrack; clipIndex += 1) {
            const startSec = clipIndex * (clipLengthSec + gapSec);
            clips.push({
                id: `perf-clip-${trackIndex}-${clipIndex}`,
                track_id: `perf-track-${trackIndex}`,
                name: `Perf ${trackIndex + 1}-${clipIndex + 1}`,
                start_sec: startSec,
                length_sec: clipLengthSec,
                color: "blue",
                source_path: `${PERF_SOURCE_PREFIX}media/${trackIndex}-${clipIndex}.wav`,
                duration_sec: clipLengthSec,
                source_sample_rate: sampleRate,
                gain: 1,
                muted: false,
                source_start_sec: 0,
                source_end_sec: clipLengthSec,
                playback_rate: 1,
                clip_playback_rate: 1,
                reversed: false,
                loop_enabled: false,
                snap_offset_sec: 0,
                fade_in_sec: fadeSec,
                fade_out_sec: fadeSec,
                fade_in_shape: 0,
                fade_out_shape: 0,
                fade_in_dir: 0,
                fade_out_dir: 0,
            });
        }
    }

    return {
        tracks,
        clips,
        selected_track_id: tracks[0]?.id ?? null,
        selected_clip_id: null,
        bpm: 120,
        playhead_sec: 0,
        project_sec: clips.length > 0 ? clipLengthSec * clipsPerTrack : 0,
    };
}

/**
 * 生成性能工程并落地到 Redux。
 *
 * 流程：先安装合成 peak 源（保证 dispatch 后第一批绘制就有数据），再 dispatch
 * `applyTimelinePayload` 用快照整体覆写 tracks + clips。
 *
 * 特殊说明：覆写是 `force: true`，会无视交互锁直接生效；若当前正处于拖动
 * 等交互中调用，需先结束交互。
 *
 * @param options 规模参数，缺省即 10 轨 × 40 clip × 60 秒。
 * @returns 规模摘要，含全览缩放所需的 `fitPxPerSec`。
 */
export function generatePerfProject(options: PerfProjectOptions = {}): PerfProjectSummary {
    const clipLengthSec = Math.max(0.1, options.clipLengthSec ?? 60);
    installSyntheticPeakSource(clipLengthSec);

    const timeline = buildPerfTimeline(options);
    store.dispatch(applyTimelinePayload(timeline));

    const projectSec = clipLengthSec * Math.max(1, Math.trunc(options.clipsPerTrack ?? 40));
    // 视口宽度取「窗口宽 − 轨道头 256px」：轨道头是固定 w-64，这里只是给
    // 全览缩放一个可直接使用的估计值，不参与任何渲染换算。
    const viewportWidthPx = Math.max(320, (window.innerWidth || 1280) - 256);

    return {
        trackCount: timeline.tracks.length,
        clipCount: timeline.clips.length,
        clipLengthSec,
        gapSec: Math.max(0, options.gapSec ?? 0),
        fadeSec: Math.max(0, options.fadeSec ?? 0),
        projectSec,
        fitPxPerSec: projectSec > 0 ? viewportWidthPx / projectSec : 1,
    };
}

/**
 * 把全览缩放档写入 localStorage 并重载页面。
 *
 * 流程：写 `hifishifter.pxPerSec`（`useTimelineState.ts:346` 启动时会读它），
 * 再用 sessionStorage 标记避免重载循环，最后 `location.reload()`。
 *
 * 特殊说明：重载会清空 Redux，因此调用方必须在重载后重新生成工程——
 * `?perf=` 分支正是靠这个顺序实现"冷启动即全览"。
 *
 * @param pxPerSec 目标缩放档。
 */
function applyFitZoomAndReload(pxPerSec: number): void {
    const RELOAD_GUARD = "hs-perf-fit-zoom-applied";
    if (sessionStorage.getItem(RELOAD_GUARD) === "1") return;
    sessionStorage.setItem(RELOAD_GUARD, "1");
    localStorage.setItem("hifishifter.pxPerSec", String(pxPerSec));
    location.reload();
}

/**
 * 解析规模描述串为选项。
 *
 * 支持两种写法：`"400"`（clip 总数，按 10 轨均分）与 `"10x40"`（轨数 × 每轨
 * clip 数）；无法解析时返回 `null` 由调用方忽略。
 *
 * @param raw 描述串。
 * @returns 规模选项，解析失败返回 `null`。
 */
function parseScaleSpec(raw: string): PerfProjectOptions | null {
    const trimmed = raw.trim();
    const pair = /^(\d+)\s*[xX]\s*(\d+)$/.exec(trimmed);
    if (pair) {
        return { trackCount: Number(pair[1]), clipsPerTrack: Number(pair[2]) };
    }
    const total = Number.parseInt(trimmed, 10);
    if (!Number.isFinite(total) || total <= 0) return null;
    return { trackCount: 10, clipsPerTrack: Math.max(1, Math.ceil(total / 10)) };
}

/** 生成工程并回写摘要，供全局入口与悬浮面板共用。 */
function generateAndReport(options: PerfProjectOptions): PerfProjectSummary {
    const summary = generatePerfProject(options);
    console.info("[perf] 性能工程已生成", summary);
    return summary;
}

/**
 * 挂载 dev-only 悬浮面板。
 *
 * 流程：直接用原生 DOM 在右下角插一个固定定位的小面板，提供三档预设与一个
 * 「清空」按钮，并把当前选择写入 localStorage 以便重载后自动重建。
 *
 * 特殊说明：
 * - **为什么不是快捷键 / dev 菜单**：新增快捷键要动 `ActionId` 联合、默认键位
 *   表、`ACTION_META`、i18n 与 App.tsx 的 switch 共 4+ 处，还会出现在用户
 *   可见的快捷键设置面板里；本模块的目标是零生产侵入，故用原生 DOM 自建。
 * - Tauri 的 `devUrl` 固定为 `http://localhost:5173`，`?perf=` 传不进去，
 *   面板是 Tauri 开发环境下唯一的零摩擦入口。
 * - 面板不参与任何渲染链路，也不订阅视口，对 Profile 无干扰；不需要时点
 *   「清空」并刷新即可（localStorage 被一并清除）。
 */
function mountPerfPanel(): void {
    const panel = document.createElement("div");
    panel.setAttribute("data-hs-perf-panel", "1");
    Object.assign(panel.style, {
        position: "fixed",
        right: "12px",
        bottom: "12px",
        zIndex: "9999",
        display: "flex",
        flexDirection: "column",
        gap: "4px",
        padding: "8px",
        borderRadius: "6px",
        background: "rgba(17, 24, 39, 0.92)",
        color: "#e5e7eb",
        font: "12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace",
        boxShadow: "0 4px 16px rgba(0, 0, 0, 0.35)",
        userSelect: "none",
    } satisfies Partial<CSSStyleDeclaration>);

    const title = document.createElement("div");
    title.textContent = "PERF (dev)";
    title.style.opacity = "0.6";
    panel.appendChild(title);

    const status = document.createElement("div");
    status.style.minHeight = "16px";
    panel.appendChild(status);

    const makeButton = (label: string, onClick: () => void): HTMLButtonElement => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = label;
        Object.assign(button.style, {
            padding: "3px 8px",
            borderRadius: "4px",
            border: "1px solid rgba(229, 231, 235, 0.25)",
            background: "transparent",
            color: "inherit",
            font: "inherit",
            cursor: "pointer",
        } satisfies Partial<CSSStyleDeclaration>);
        button.addEventListener("click", onClick);
        panel.appendChild(button);
        return button;
    };

    const run = (spec: string) => {
        const options = parseScaleSpec(spec);
        if (options === null) return;
        const summary = generateAndReport(options);
        localStorage.setItem(PERSIST_KEY, spec);
        status.textContent = `${summary.trackCount}×${summary.clipCount / summary.trackCount} = ${summary.clipCount} clip`;
        applyFitZoomAndReload(summary.fitPxPerSec);
    };

    makeButton("400 clip", () => run("400"));
    makeButton("1000 clip", () => run("1000"));
    makeButton("400 clip + fade", () => {
        const summary = generateAndReport({ clipsPerTrack: 40, fadeSec: 0.5 });
        localStorage.setItem(PERSIST_KEY, JSON.stringify({ clipsPerTrack: 40, fadeSec: 0.5 }));
        status.textContent = `${summary.clipCount} clip · fade 0.5s`;
        applyFitZoomAndReload(summary.fitPxPerSec);
    });
    // ── GL clip 体开关（P3）──────────────────────────────────────
    // 默认开启；切到 "0" 即退回 Canvas2D。切换 localStorage 后派发自定义
    // 事件，TimelineCanvasViewport 监听它即时建拆 GL 渲染器，无需刷新。
    const glEnabled = (): boolean => localStorage.getItem(PERF_GL_CLIP_BODIES_KEY) !== "0";
    const glToggle = makeButton(`GL clip: ${glEnabled() ? "on" : "off"}`, () => {
        const next = glEnabled() ? "0" : "1";
        localStorage.setItem(PERF_GL_CLIP_BODIES_KEY, next);
        glToggle.textContent = `GL clip: ${next === "1" ? "on" : "off"}`;
        window.dispatchEvent(new Event(PERF_GL_CLIP_BODIES_KEY));
        status.textContent = `GL clip bodies ${next === "1" ? "ON" : "OFF"}`;
    });

    // ── 帧率探针开关 ────────────────────────────
    // 打开后右上角出现浮层：FPS / 帧间隔 / 各图层绘制耗时 / React 提交耗时。
    let profilerOn = localStorage.getItem(FRAME_PROFILER_KEY) === "1";
    const profilerButton = makeButton(`profiler: ${profilerOn ? "on" : "off"}`, () => {
        profilerOn = !profilerOn;
        localStorage.setItem(FRAME_PROFILER_KEY, profilerOn ? "1" : "0");
        profilerButton.textContent = `profiler: ${profilerOn ? "on" : "off"}`;
        if (profilerOn) {
            startFrameProfiler();
        } else {
            stopFrameProfiler();
        }
        status.textContent = `frame profiler ${profilerOn ? "ON" : "OFF"}`;
    });
    if (profilerOn) {
        startFrameProfiler();
        profilerButton.textContent = "profiler: on";
    }

    makeButton("清空", () => {
        localStorage.removeItem(PERSIST_KEY);
        store.dispatch(
            applyTimelinePayload({
                tracks: [],
                clips: [],
                selected_track_id: null,
                selected_clip_id: null,
                bpm: 120,
                playhead_sec: 0,
            }),
        );
        status.textContent = "已清空";
    });

    document.body.appendChild(panel);
}

/**
 * 安装 dev-only 全局入口。
 *
 * 流程：
 * 1. 挂 `window.__hsStore`（便于在控制台观察 / 手动 dispatch）与
 *    `window.__hsPerf`（以选项对象生成工程）；
 * 2. 若 localStorage 里有上次选择的规模，自动重建（Tauri 的 `devUrl` 固定，
 *    `?perf=` 传不进去，持久化是它唯一的"冷启动即全览"通道）；
 * 3. 解析 `?perf=` 查询参数——仅在直接用浏览器打开 vite dev server 时可达；
 * 4. 挂载右下角悬浮面板。
 */
export function installPerfProjectDevtools(): void {
    const scope = window as unknown as Record<string, unknown>;
    scope.__hsStore = store;
    scope.__hsPerf = (options: PerfProjectOptions = {}) => generatePerfProject(options);

    const persisted = localStorage.getItem(PERSIST_KEY);
    let restored = false;
    if (persisted != null && persisted !== "") {
        const options = persisted.trimStart().startsWith("{")
            ? (safeParseJson(persisted) ?? null)
            : parseScaleSpec(persisted);
        if (options !== null) {
            generateAndReport(options);
            restored = true;
        }
    }

    const raw = new URLSearchParams(location.search).get("perf");
    if (raw != null && !restored) {
        const options = parseScaleSpec(raw);
        if (options !== null) {
            localStorage.setItem(PERSIST_KEY, raw.trim());
            applyFitZoomAndReload(generateAndReport(options).fitPxPerSec);
        }
    }

    mountPerfPanel();
}
