/**
 * Capture 命令的前端门面。
 *
 * 【为什么与 `hostLinkApi` 分开】两条通路的目标完全不同：`hostLinkApi` 面向
 * **独立 App** 打开宿主导出的工作区，走 Tauri 命令；Capture 面向**插件自己的窗口**，
 * 走插件 webview 的命令通道（`invoke` 在插件模式下就是 postMessage）。放在一起会让
 * "这个命令在哪种形态下可用"变得要靠读者推断。
 *
 * 【为什么状态是拉取式】录制进度每 100ms 变一次，而宿主不会推送；轮询是唯一可行的
 * 形态，且它的代价只是一个整数读取（后端 `capture_status` 不取渲染锁）。
 */
import { invoke } from "../../services/invoke";
import type { MessageKey } from "../../i18n/messages";

/** 录制中的读数。 */
export interface CaptureRecording {
    capturedFrames: number;
    sealedSegments: number;
    position: number;
    peak: [number, number];
    gaps: number;
    spares: number;
}

/** 录制中正在长出来的工程区间（时间轴据此画实时录制带）。 */
export interface CaptureCapture {
    startFrame: number;
    endFrame: number;
    positionFrame: number;
    sampleRate: number;
}

/** 活动素材的摘要。 */
export interface CaptureMaterial {
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
export interface CaptureSpan {
    startFrame: number;
    endFrame: number;
    sampleRate: number;
}

/** 一个 take 的清单条目。 */
export interface CaptureTake {
    id: number;
    active: boolean;
    clips: number;
    startSec: number | null;
    endSec: number | null;
}

export interface CaptureOptions {
    maxDurationSec: number;
    segmentSec: number;
    monitor: boolean;
    /** 素材覆盖范围之外是否直通宿主原声（`false` = 区间外静音）。 */
    outsidePassthrough: boolean;
}

/**
 * 清单里有、但**本机读不出来**的捕获音频。
 *
 * 【为什么这是一条产品信息而不是诊断读数】sidecar 落在本机缓存目录，清单随宿主工程
 * 走 —— 把工程拷到另一台机器时音频不会跟过去。没有这一项，用户只会看到"参数还在、
 * 声音没了"，且不知道为什么。
 */
export interface CaptureMissingTake {
    id: number;
    frames: number;
    /** 该 take 的时长（秒）。 */
    seconds: number;
    startSec: number | null;
    endSec: number | null;
}

export interface CaptureStatus {
    /**
     * `"empty"` / `"standby"` / `"recording"` / `"ready"`。
     *
     * 【为什么"待命"是独立一档】武装之后、宿主播放之前，素材一定还是零 —— 而那与
     * "正在录"是两种不同的等待：前者等用户**去按播放**，后者在积累素材。合成一个
     * "已录 0.0 秒"会让待命看起来像失败（用户反馈里的"点了录制没反应"）。
     */
    state: "empty" | "standby" | "recording" | "ready";
    /** 已武装但宿主还没播过（= `state === "standby"` 的布尔形式）。 */
    standby: boolean;
    recording: CaptureRecording | null;
    capture: CaptureCapture | null;
    material: CaptureMaterial | null;
    hasMaterial: boolean;
    span: CaptureSpan | null;
    takes: CaptureTake[];
    /** 本机缺失的捕获音频（空 = 全都恢复了）。 */
    missingTakes: CaptureMissingTake[];
    options: CaptureOptions;
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
     * 【为什么要它】Capture 的用法是"先按录制、再按播放"，所以"按了录制却忘了播放"
     * 是这条通路最常见的失败。实时读数让面板直接说出来，而不是让用户对着一个 0.00 秒的
     * 读数猜。`canCapture` 不依赖它：那条只回答"这个宿主给不给位置"。
     */
    hostPlaying: boolean;
    /**
     * 宿主在用**非 32-bit 浮点**跑本实例（`kSample64`）。
     *
     * 【为什么要单独一档】那种情况下插件的样本平面是 64 位的，插件读不了 ⇒ 捕获与输出
     * 都按设计静音。它表现为"插上插件就没声音、也录不到东西"，而补救是**用户可执行的**：
     * 把宿主的处理精度切回 32-bit 浮点。
     */
    unsupportedPrecision: boolean;
    /**
     * 距单次录制上限不足 10 秒。
     *
     * 【为什么要预警】上限到点会**自己停**，而那是用户设的一个数值 —— 不预警的话
     * 用户只能看到"录着录着就断了"。
     */
    limitWarning: boolean;
    /**
     * 上一次录制**自己**停下的原因（`null` = 没有过、或那一次是用户按的停）。
     *
     * 取值与后端 `ClosedReason::as_str` 一致：`"limit_reached"` / `"no_spare"` /
     * `"rate_changed"` / `"budget_full"` / `"jump"`。三者的补救完全不同，所以分开报。
     */
    stoppedReason: string | null;
    /**
     * 宿主是否**正在**通过 ARA 供音。
     *
     * 【为什么状态里要有它】Capture 入口现在无条件可见（形态判定要等 5 秒宽限期，
     * 不能拿它当可见性判据），所以面板必须自己回答"我在这里有没有用"。
     */
    araBound: boolean;
    /**
     * **此刻听到的是哪一版**（后端输出仲裁的结论）。
     *
     * 取值与后端 `CaptureOutput::as_str` 一致：`"processed"` / `"raw"` /
     * `"passthrough"` / `"pending_synthesis"` / `"recording"` / `"untouched"`。
     *
     * `pending_synthesis` 与 `passthrough` 的**输出内容**相同（都是宿主原声），
     * 区别是前者"素材已在、结果未出"，后者"还没有素材"。
     *
     * 【为什么它值得占一行】处理结果、原始录音与宿主原声听起来都"像那么回事"，而它们
     * 之间的差别正是"我的编辑生效了没有" —— 用户无法自己判断。后端已经在仲裁，界面
     * 只是把它说出来。
     */
    output: string;
}

export const captureApi = {
    status: () => invoke<CaptureStatus>("capture_status"),
    start: () => invoke<CaptureStatus>("capture_start"),
    stop: () => invoke<{ take: number; status: CaptureStatus }>("capture_stop"),
    cancel: () => invoke<CaptureStatus>("capture_cancel"),
    clear: () => invoke<CaptureStatus>("capture_clear"),
    deleteTake: (takeId: number) => invoke<CaptureStatus>("capture_delete_take", takeId),
    setOptions: (patch: Partial<CaptureOptions>) =>
        invoke<CaptureStatus>("capture_set_options", patch),
};

/** 后端错误的可读文本；与 `hostLinkError` 同一套解包口径。 */
/**
 * 语言无关的分类：宿主正在通过 ARA 供音，Capture 不适用。
 *
 * 【为什么分类在后端、文案在前端】与 `hostLinkApi` 同一条纪律：后端只给稳定的分类串，
 * 文案按 catalog 本地化。这里因此不解析后端的中文/英文句子。
 */
const CAPTURE_NOT_APPLICABLE_ARA = "capture_not_applicable_ara";

/** 后端分类 → 本地化文案；其余错误原样透传（由调用方按既有分支处理）。 */
export function captureErrorText(error: unknown, t: (key: MessageKey) => string): string {
    const raw = captureError(error);
    return raw.includes(CAPTURE_NOT_APPLICABLE_ARA) ? t(CAPTURE_NOT_APPLICABLE_ARA) : raw;
}

export function captureError(error: unknown): string {
    if (error instanceof Error)
        return error.cause !== undefined ? captureError(error.cause) : error.message;
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
