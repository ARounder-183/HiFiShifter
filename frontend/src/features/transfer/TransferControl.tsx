/**
 * 传输栏上的 Transfer 控件（插件形态的主入口）。
 *
 * 【为什么入口在这里而不是一个浮动面板】用户对"录制"这件事的心智模型就是传输栏上的
 * 一个键：Melodyne 的 Transfer 也长在那里。此前面板是 460×320 的浮窗、只有几个调试
 * 读数，用户找不到它、也不知道它和播放有什么关系。
 *
 * # 为什么这里只有**一个图标 + 一个读数**
 *
 * 主工具栏是 `flex-nowrap overflow-x-auto overflow-y-hidden` 的**单行**容器
 * （`ActionBar` 的根节点），而插件形态下它的宽度由宿主的 VST 窗口决定 —— 实测可能很窄。
 * 因此这里的每一像素都要有理由：
 *
 * - **绝不换行**。此前录制中会多出一行读数，而工具栏只有一行高且纵向裁切 —— 那一行被
 *   裁掉，表现为"点了录制排版就乱"。读数全部收进图标与状态片。
 * - **键用图标而不是文字**。"录制"/"停止"/"取消"在任何语言里都比一个图标宽得多，
 *   而它们的含义由图标本身与 tooltip 表达。图标按钮的 `tooltip` 同时是 `aria-label`，
 *   所以无文字不等于无名称。
 * - **诊断读数搬去面板**（`TransferPanel`）。宿主在不在播放、峰值、分段数、为什么没
 *   声音 —— 那些是"刚发生了什么"，属于面板；工具栏只回答"现在按这个键会发生什么"。
 *   这与 ActionBar 里"插件形态不渲染节拍器"是同一条取舍：宿主窗口很窄时，不常用的
 *   东西不该占位置。
 *
 * 【为什么待命要单独说】"按了录制没反应"是这条通路最常见的困惑，而它有两种完全不同的
 * 原因：宿主还没播（等用户去按播放），或宿主压根不给播放位置（这条通路在这个宿主上
 * 不成立）。两者都落在 tooltip 上，但措辞不同 —— 合成一句"不可用"等于让用户自己猜。
 *
 * 【为什么读数取工程秒而不是"帧数"】用户看到的时间轴单位是秒；后端给的是工程帧
 * （那才是唯一无歧义的量），换算只发生在 `framesToSeconds` / `frameToSec` 两处。
 */
import { useEffect } from "react";
import { Flex } from "@radix-ui/themes";
import {
    Cross2Icon,
    DotFilledIcon,
    ExclamationTriangleIcon,
    StopIcon,
} from "@radix-ui/react-icons";

import { useI18n } from "../../i18n/I18nProvider";
import { AppIconButton, AppStatusChip } from "../../ui";
import { framesToSeconds } from "./transferApi";
import { startTransferPolling, transferActions, useTransferSnapshot } from "./transferStore";

/** 显示用采样率兜底：后端在还没收到上下文时给 0，取不到就按 48k 显示量级。 */
const FALLBACK_SAMPLE_RATE = 48_000;

export function TransferControl() {
    const { t, tVars } = useI18n();
    const { status, busy, error } = useTransferSnapshot();

    // 控件常驻（插件形态），因此它承担"启动唯一那条轮询"的职责。
    useEffect(() => startTransferPolling(), []);

    const recording = status?.state === "recording";
    const standby = status?.state === "standby";
    // 武装 = 待命或录制中。两者都是"这一次录制还没结束"，键的语义都是"停止"。
    const armed = recording || standby;
    const araBound = status?.araBound ?? false;
    const canCapture = !araBound && (status?.canCapture ?? false);
    // 录制中但宿主没在播放 ⇒ 一个字节也进不来。这是这条通路最常见的失败
    // （用法是"先按录制、再按播放"）。
    const stalled = recording && !(status?.hostPlaying ?? false);
    // 宿主在播放、却一帧都没到 ⇒ 插件没拿到音频。**必须与"没在播放"分开**：
    // 前者要去检查路由（插件插错轨 / 宿主没把它接进链路），后者只要按播放。
    const silentInput =
        recording &&
        (status?.hostPlaying ?? false) &&
        (status?.recording?.capturedFrames ?? 0) === 0;
    const captured = status?.recording
        ? framesToSeconds(
              status.recording.capturedFrames,
              status.capture?.sampleRate || FALLBACK_SAMPLE_RATE,
          )
        : null;
    // 【为什么"有位置但没输入"排在"不能录"之前】两者都是"点了也没用"，但要做的事
    // 完全不同：前者要去检查宿主的路由（插件是不是插错了轨道），后者是宿主根本不给位置。
    const noInput = canCapture && status?.sawInput === false;
    const limitWarning = status?.limitWarning ?? false;

    /*
     * 唯一那条说明。工具栏没有地方放第二行文字，所以**所有**状态都必须落在这里 ——
     * 这也意味着它的优先级就是"用户此刻最该知道哪一件事"的优先级：
     * 先讲这条通路成不成立（精度 / ARA / 没有位置 / 没有输入），再讲这次录制怎么了
     * （到上限 / 没在播放 / 没拿到音频 / 正在录），最后才是"录好了"与"怎么开始"。
     */
    const tooltip = status?.unsupportedPrecision
        ? t("transfer_unsupported_precision")
        : araBound
          ? t("transfer_not_applicable_ara")
          : !canCapture
            ? t("transfer_cannot_capture")
            : noInput
              ? t("transfer_no_input_audio")
              : limitWarning
                ? t("transfer_limit_warning")
                : standby
                  ? t("transfer_field_waiting_host")
                  : silentInput
                    ? t("transfer_no_audio_arriving")
                    : stalled
                      ? t("transfer_host_not_playing")
                      : recording
                        ? t("transfer_guide_recording")
                        : (status?.takeCount ?? 0) > 0
                          ? t("transfer_guide_ready")
                          : t("transfer_guide_body");

    return (
        <Flex align="center" gap="1" className="shrink-0" data-transfer-control>
            {/* 录制 / 停止：一个图标键切换。武装时用实心 + 危险色 —— 那是"这个键现在
                处于开着的状态"，扫一眼就知道录制还没结束。 */}
            <AppIconButton
                size="sm"
                intent={armed ? "danger" : "default"}
                active={armed}
                icon={armed ? <StopIcon /> : <DotFilledIcon />}
                tooltip={tooltip}
                disabled={busy || (!armed && !canCapture)}
                onClick={() => void (armed ? transferActions.stop() : transferActions.start())}
            />
            {/* 取消：只在武装期出现。录制键就在工具栏上，误触很便宜 —— 而没有取消的话，
                唯一的出路是"先停止（产生一个 take）再删掉它"。 */}
            {armed ? (
                <AppIconButton
                    size="sm"
                    icon={<Cross2Icon />}
                    tooltip={t("transfer_cancel_hint")}
                    disabled={busy}
                    onClick={() => void transferActions.cancel()}
                />
            ) : null}
            {/* 已录时长：录制中唯一"在动"的东西。没有它，用户无法判断插件到底有没有在
                收音频 —— 而它是这一行里唯一值得占宽度的文字。
                上限预警折进它的颜色与 tooltip：接近上限时变危险色，不额外占位置。 */}
            {recording ? (
                <AppStatusChip tone={limitWarning ? "danger" : "warning"} tabular tooltip={tooltip}>
                    {tVars("transfer_recording_elapsed", { seconds: captured ?? "0.0" })}
                </AppStatusChip>
            ) : null}
            {/* 出错：只给一个图标 + tooltip。错误正文可能是后端的一整句话（例如
                "宿主还没声明采样率…"），放进工具栏会把整条栏撑开 —— 完整正文在面板里。 */}
            {error ? (
                <span
                    role="alert"
                    aria-label={error}
                    data-tooltip={error}
                    style={{
                        color: "var(--qt-danger-text)",
                        display: "inline-flex",
                        alignItems: "center",
                    }}
                >
                    <ExclamationTriangleIcon width="var(--qt-icon-sm)" height="var(--qt-icon-sm)" />
                </span>
            ) : null}
        </Flex>
    );
}
