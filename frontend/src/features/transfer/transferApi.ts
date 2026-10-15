/**
 * Transfer 命令的前端门面。
 *
 * 【为什么与 `araApi` 分开】两条通路的目标完全不同：`araApi` 面向**独立 App** 连
 * 宿主，走 Tauri 命令；Transfer 面向**插件自己的窗口**，走插件 webview 的命令通道
 * （`invoke` 在插件模式下就是 postMessage）。放在一起会让"这个命令在哪种形态下可用"
 * 变得要靠读者推断。
 *
 * 【为什么状态是拉取式】录制进度每 100ms 变一次，而宿主不会推送；轮询是唯一可行的
 * 形态，且它的代价只是一个整数读取（后端 `transfer_status` 不取渲染锁）。
 */
import { invoke } from "../../services/invoke";
import type { MessageKey } from "../../i18n/messages";

/** 录制中的读数。 */
export interface TransferRecording {
    capturedFrames: number;
    sealedSegments: number;
    position: number;
    peak: [number, number];
    gaps: number;
    spares: number;
}

/** 录制中正在长出来的工程区间（时间轴据此画实时录制带）。 */
export interface TransferCapture {
    startFrame: number;
    endFrame: number;
    positionFrame: number;
    sampleRate: number;
}

/** 活动素材的摘要。 */
export interface TransferMaterial {
    clips: number;
    capturedFrames: number;
    coveredFrames: number;
    spannedFrames: number;
    sampleRate: number;
    /**
     * 录到了东西，但没有任何有声内容。
     *
     * 【为什么必须有这个判据】全静音的录制不产生片段，于是"面板说已录制、时间轴空白、
     * 播放无声"三个读数互相矛盾。有了它，界面能直接说清是**输入总线没声音**。
     */
    silent: boolean;
}

/** 已 promote 素材的工程跨度。 */
export interface TransferSpan {
    startFrame: number;
    endFrame: number;
    sampleRate: number;
}

/** 一个 take 的清单条目。 */
export interface TransferTake {
    id: number;
    active: boolean;
    clips: number;
    startSec: number | null;
    endSec: number | null;
}

export interface TransferOptions {
    maxDurationSec: number;
    segmentSec: number;
    monitor: boolean;
    /** 素材覆盖范围之外是否直通宿主原声（`false` = 区间外静音）。 */
    outsidePassthrough: boolean;
}

export interface TransferStatus {
    /** `"empty"` / `"recording"` / `"ready"`。 */
    state: "empty" | "recording" | "ready";
    recording: TransferRecording | null;
    capture: TransferCapture | null;
    material: TransferMaterial | null;
    hasMaterial: boolean;
    span: TransferSpan | null;
    takes: TransferTake[];
    options: TransferOptions;
    takeCount: number;
    /** 最近一次录制的 take 号（派生读数；投影取全部 take，见后端 `latest_take`）。 */
    activeTake: number | null;
    /** 宿主给过可用的 `ProcessContext` —— 录制的唯一硬前提。 */
    sawContext: boolean;
    sawInput: boolean;
    sawOffline: boolean;
    canCapture: boolean;
    /**
     * 宿主**此刻**是否在播放。
     *
     * 【为什么要它】Transfer 的用法是"先按录制、再按播放"，所以"按了录制却忘了播放"
     * 是这条通路最常见的失败。实时读数让面板直接说出来，而不是让用户对着一个 0.00 秒的
     * 读数猜。`canCapture` 不依赖它：那条只回答"这个宿主给不给位置"。
     */
    hostPlaying: boolean;
    /**
     * 宿主是否**正在**通过 ARA 供音。
     *
     * 【为什么状态里要有它】Transfer 入口现在无条件可见（形态判定要等 5 秒宽限期，
     * 不能拿它当可见性判据），所以面板必须自己回答"我在这里有没有用"。
     */
    araBound: boolean;
}

export const transferApi = {
    status: () => invoke<TransferStatus>("transfer_status"),
    start: () => invoke<TransferStatus>("transfer_start"),
    stop: () => invoke<{ take: number; status: TransferStatus }>("transfer_stop"),
    cancel: () => invoke<TransferStatus>("transfer_cancel"),
    clear: () => invoke<TransferStatus>("transfer_clear"),
    deleteTake: (takeId: number) => invoke<TransferStatus>("transfer_delete_take", takeId),
    setOptions: (patch: Partial<TransferOptions>) =>
        invoke<TransferStatus>("transfer_set_options", patch),
};

/** 后端错误的可读文本；与 `araError` 同一套解包口径。 */
/**
 * 语言无关的分类：宿主正在通过 ARA 供音，Transfer 不适用。
 *
 * 【为什么分类在后端、文案在前端】与 `araApi` 同一条纪律：后端只给稳定的分类串，
 * 文案按 catalog 本地化。这里因此不解析后端的中文/英文句子。
 */
const TRANSFER_NOT_APPLICABLE_ARA = "transfer_not_applicable_ara";

/** 后端分类 → 本地化文案；其余错误原样透传（由调用方按既有分支处理）。 */
export function transferErrorText(error: unknown, t: (key: MessageKey) => string): string {
    const raw = transferError(error);
    return raw.includes(TRANSFER_NOT_APPLICABLE_ARA) ? t(TRANSFER_NOT_APPLICABLE_ARA) : raw;
}

export function transferError(error: unknown): string {
    if (error instanceof Error)
        return error.cause !== undefined ? transferError(error.cause) : error.message;
    return String(error);
}

/** 把帧数换算成秒的显示串（保留一位小数）。 */
export function framesToSeconds(frames: number, sampleRate: number): string {
    if (!Number.isFinite(sampleRate) || sampleRate <= 0) return "0.0";
    return (frames / sampleRate).toFixed(1);
}

/**
 * 工程帧 → 秒。
 *
 * 【为什么要有它】录制带的几何来自后端（工程帧），而时间轴与视口都以秒为单位。
 * 换算只在这一处发生，避免每个调用点各写一遍"除以哪个采样率"。
 */
export function frameToSec(frame: number, sampleRate: number): number {
    if (!Number.isFinite(sampleRate) || sampleRate <= 0) return 0;
    return frame / sampleRate;
}
