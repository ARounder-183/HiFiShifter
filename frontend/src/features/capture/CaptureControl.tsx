/**
 * 传输栏上的捕获控件（插件形态的主入口）。
 *
 * 【为什么入口在传输栏而不是一个面板】用户对"录制"这件事的心智模型就是传输栏上的
 * 一个键：Melodyne 的 Capture 也长在那里。
 *
 * # 这里只回答"现在按这个键会发生什么"
 *
 * 主工具栏是**按功能组换行**的容器（见 `ui/Toolbar` 的 `AppToolbarGroup`），而插件
 * 形态下它的宽度由宿主的窗口决定 —— 实测可能很窄。所以：
 *
 * - **组内绝不换行**。此前录制中会多出一行读数，那一行把整个工具栏撑高，而当时的条形
 *   只有固定高度、纵向裁切，于是读数被切掉，表现为"点了录制排版就乱"。读数全部收进
 *   图标与状态片，组本身保持单行（`AppToolbarGroup` 的 `whitespace-nowrap`）。
 * - **键用图标而不是文字**。图标的 `tooltip` 同时是 `aria-label`，所以无文字不等于
 *   无名称。
 * - **tooltip 只写一句短语**。"为什么现在不能用"、"峰值多少"、"宿主在不在播放"
 *   这些属于**设置窗口**（右键或菜单进入），不属于悬停气泡 —— 悬停是"我现在要按
 *   这个键，它会做什么"，不是一篇说明书。
 *
 * 【为什么读数取工程秒而不是"帧数"】用户看到的时间轴单位是秒；后端给的是工程帧
 * （那才是唯一无歧义的量），换算只发生在 `framesToSeconds` / `frameToSec` 两处。
 */
import { useEffect, useSyncExternalStore } from "react";
import { Flex } from "@radix-ui/themes";
import {
    Cross2Icon,
    DotFilledIcon,
    ExclamationTriangleIcon,
    StopIcon,
} from "@radix-ui/react-icons";

import { useI18n } from "../../i18n/I18nProvider";
import { AppIconButton, AppStatusChip } from "../../ui";
import { framesToSeconds } from "./captureApi";
import { CaptureSettingsDialog } from "./CaptureSettingsDialog";
import {
    getCaptureSettingsOpen,
    openCaptureSettings,
    setCaptureSettingsOpen,
    subscribeCaptureSettings,
} from "./captureSettingsBus";
import { startCapturePolling, captureActions, useCaptureSnapshot } from "./captureStore";

/** 显示用采样率兜底：后端在还没收到上下文时给 0，取不到就按 48k 显示量级。 */
const FALLBACK_SAMPLE_RATE = 48_000;

export function CaptureControl() {
    const { t, tVars } = useI18n();
    const { status, busy, error } = useCaptureSnapshot();
    const settingsOpen = useSyncExternalStore(
        subscribeCaptureSettings,
        getCaptureSettingsOpen,
        getCaptureSettingsOpen,
    );

    // 控件常驻（插件形态），因此它承担"启动唯一那条轮询"的职责。
    useEffect(() => startCapturePolling(), []);

    const recording = status?.state === "recording";
    const standby = status?.state === "standby";
    // 武装 = 待命或录制中。两者都是"这一次录制还没结束"，键的语义都是"停止"。
    const armed = recording || standby;
    const araBound = status?.araBound ?? false;
    const canCapture = !araBound && (status?.canCapture ?? false);
    const limitWarning = status?.limitWarning ?? false;
    const captured = status?.recording
        ? framesToSeconds(
              status.recording.capturedFrames,
              status.capture?.sampleRate || FALLBACK_SAMPLE_RATE,
          )
        : null;

    /*
     * 唯一那条 tooltip —— **一句短语**，只回答"按下去会怎样"。
     *
     * 【为什么"为什么不能用"不在这里】那句话需要一整句才能说清（宿主没给位置 / 插件没
     * 接在要处理的轨道上 / 宿主精度是 64 位），塞进悬停气泡既读不完也挤爆布局。它现在
     * 住在设置窗口里：用户想知道原因时会右键打开它，而那正是他愿意读长文的时候。
     */
    const tooltip = !canCapture
        ? t("capture_unavailable")
        : recording
          ? t("capture_stop")
          : standby
            ? t("capture_waiting")
            : t("capture_start_hint");

    return (
        <Flex align="center" gap="1" className="shrink-0" data-capture-control>
            {/* 捕获 / 停止：一个图标键切换。武装时用实心 + 危险色 —— 那是"这个键现在
                处于开着的状态"，扫一眼就知道捕获还没结束。

                【右键 = 打开设置】不可用时按钮是禁用的，而禁用按钮收不到点击 —— 右键
                是它唯一还能用的手势，也是"我想知道为什么"最自然的那一下。可用时右键
                同样进设置：用户想调监听/上限，手本来就在这个键上。 */}
            <span
                onContextMenu={(event) => {
                    event.preventDefault();
                    openCaptureSettings();
                }}
            >
                <AppIconButton
                    size="sm"
                    intent={armed ? "danger" : "default"}
                    active={armed}
                    icon={armed ? <StopIcon /> : <DotFilledIcon />}
                    tooltip={tooltip}
                    disabled={busy || (!armed && !canCapture)}
                    onClick={() => void (armed ? captureActions.stop() : captureActions.start())}
                />
            </span>
            {/* 取消：只在武装期出现。捕获键就在工具栏上，误触很便宜 —— 而没有取消的话，
                唯一的出路是"先停止（产生一个 take）再删掉它"。 */}
            {armed ? (
                <AppIconButton
                    size="sm"
                    icon={<Cross2Icon />}
                    tooltip={t("capture_cancel_hint")}
                    disabled={busy}
                    onClick={() => void captureActions.cancel()}
                />
            ) : null}
            {/* 已捕获时长：录制中唯一"在动"的东西。没有它，用户无法判断插件到底有没有在
                收音频 —— 而它是这一行里唯一值得占宽度的文字。
                上限预警折进它的颜色：接近上限时变危险色，不额外占位置。 */}
            {recording ? (
                <AppStatusChip tone={limitWarning ? "danger" : "warning"} tabular tooltip={tooltip}>
                    {tVars("capture_recording_elapsed", { seconds: captured ?? "0.0" })}
                </AppStatusChip>
            ) : null}
            {/* 出错：只给一个图标，**点击进设置**。错误正文可能是后端的一整句话，放进工具栏
                会把整条栏撑开 —— 完整正文与可执行的下一步都在设置窗口里。 */}
            {error ? (
                <button
                    type="button"
                    role="alert"
                    aria-label={error}
                    data-tooltip={t("capture_open_settings")}
                    onClick={openCaptureSettings}
                    style={{
                        color: "var(--qt-danger-text)",
                        display: "inline-flex",
                        alignItems: "center",
                        background: "none",
                        border: "none",
                        padding: 0,
                        cursor: "pointer",
                    }}
                >
                    <ExclamationTriangleIcon width="var(--qt-icon-sm)" height="var(--qt-icon-sm)" />
                </button>
            ) : null}
            <CaptureSettingsDialog
                open={settingsOpen}
                onOpenChange={setCaptureSettingsOpen}
                backendError={error}
            />
        </Flex>
    );
}
