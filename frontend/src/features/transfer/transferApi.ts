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

export interface TransferOptions {
    maxDurationSec: number;
    segmentSec: number;
    monitor: boolean;
}

export interface TransferStatus {
    /** `"empty"` / `"recording"` / `"ready"`。 */
    state: "empty" | "recording" | "ready";
    recording: TransferRecording | null;
    options: TransferOptions;
    takeCount: number;
    activeTake: number | null;
    /** 宿主给过可用的 `ProcessContext` —— 录制的唯一硬前提。 */
    sawContext: boolean;
    sawInput: boolean;
    sawOffline: boolean;
    canCapture: boolean;
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
    selectTake: (takeId: number) => invoke<TransferStatus>("transfer_select_take", takeId),
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
