/**
 * 传输栏上的 Transfer 控件（插件形态的主入口）。
 *
 * 【为什么入口在这里而不是一个浮动面板】用户对"录制"这件事的心智模型就是传输栏上的
 * 一个键：Melodyne 的 Transfer 也长在那里。此前面板是 460×320 的浮窗、只有几个调试
 * 读数，用户找不到它、也不知道它和播放有什么关系。
 *
 * 【为什么控件分两级】录制中需要回答的问题（宿主在不在播、录到哪了、峰值多少）比
 * 状态栏一行能装下的多，而没在录制时那些问题一个都不存在。于是：常驻的只有键与状态片，
 * 展开区只在录制中出现 —— 空间花在需要它的时刻。
 *
 * 【为什么待命要单独说】"按了录制没反应"是这条通路最常见的困惑，而它有两种完全不同的
 * 原因：宿主还没播（等用户去按播放），或宿主压根不给播放位置（这条通路在这个宿主上
 * 不成立）。合成一句"不可用"等于让用户自己猜。
 *
 * 【为什么读数取工程秒而不是"帧数"】用户看到的时间轴单位是秒；后端给的是工程帧
 * （那才是唯一无歧义的量），换算只发生在 `framesToSeconds` / `frameToSec` 两处。
 */
import { useEffect } from "react";
import { Flex } from "@radix-ui/themes";

import { useI18n } from "../../i18n/I18nProvider";
import { AppStatusChip, AppButton } from "../../ui";
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
    const araBound = status?.araBound ?? false;
    const canCapture = !araBound && (status?.canCapture ?? false);
    // 录制中但宿主没在播放 ⇒ 一个字节也进不来。这是这条通路最常见的失败
    // （用法是"先按录制、再按播放"），必须直接说出来。
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
    const tooltip = status?.unsupportedPrecision
        ? t("transfer_unsupported_precision")
        : araBound
          ? t("transfer_not_applicable_ara")
          : !canCapture
            ? t("transfer_cannot_capture")
            : noInput
              ? t("transfer_no_input_audio")
              : standby
                ? t("transfer_guide_standby")
                : recording
                  ? t("transfer_guide_recording")
                  : (status?.takeCount ?? 0) > 0
                    ? t("transfer_guide_ready")
                    : t("transfer_guide_body");

    return (
        <Flex direction="column" gap="1" className="shrink-0" data-transfer-control>
            <Flex align="center" gap="1">
                <AppButton
                    intent={recording || standby ? "danger" : "subtle"}
                    emphasis={recording || standby ? "solid" : undefined}
                    size="sm"
                    disabled={busy || (!recording && !canCapture)}
                    data-tooltip={tooltip}
                    onClick={() =>
                        void (recording || standby
                            ? transferActions.stop()
                            : transferActions.start())
                    }
                >
                    {recording || standby ? t("transfer_stop") : t("transfer_start")}
                </AppButton>
                {recording || standby ? (
                    <>
                        {/* 【为什么留着"取消"】录制键就在传输栏上，误触很便宜 —— 而没有取消
                            的话，唯一的出路是"先停止（产生一个 take）再删掉它"。 */}
                        <AppButton
                            intent="subtle"
                            size="sm"
                            disabled={busy}
                            data-tooltip={t("transfer_cancel_hint")}
                            onClick={() => void transferActions.cancel()}
                        >
                            {t("transfer_cancel")}
                        </AppButton>
                    </>
                ) : null}
                {/* 状态片：待命与录制中是两种不同的等待，读数也不同 —— 待命没有读数
                    （素材还没开始长），所以那一片只说"等待播放"。 */}
                {standby ? (
                    <AppStatusChip tone="accent" tooltip={t("transfer_guide_standby")}>
                        {t("transfer_status_standby")}
                    </AppStatusChip>
                ) : null}
                {recording ? (
                    <AppStatusChip tone="warning" tabular tooltip={t("transfer_guide_recording")}>
                        {tVars("transfer_recording_progress", { seconds: captured ?? "0.0" })}
                    </AppStatusChip>
                ) : null}
                {status?.limitWarning ? (
                    <AppStatusChip tone="danger" tooltip={t("transfer_limit_warning")}>
                        {t("transfer_field_limit_soon")}
                    </AppStatusChip>
                ) : null}
                {stalled ? (
                    <span
                        className="hs-type-caption"
                        style={{ color: "var(--qt-warning-text)" }}
                        data-tooltip={t("transfer_host_not_playing")}
                    >
                        {t("transfer_field_waiting_host")}
                    </span>
                ) : null}
                {silentInput ? (
                    <span
                        className="hs-type-caption"
                        style={{ color: "var(--qt-danger-text)" }}
                        data-tooltip={t("transfer_no_audio_arriving")}
                    >
                        {t("transfer_field_no_audio")}
                    </span>
                ) : null}
                {status?.unsupportedPrecision ? (
                    <span
                        className="hs-type-caption"
                        style={{ color: "var(--qt-danger-text)" }}
                        data-tooltip={t("transfer_unsupported_precision")}
                    >
                        {t("transfer_field_precision")}
                    </span>
                ) : null}
                {error ? (
                    <span
                        className="hs-type-caption"
                        role="alert"
                        style={{ color: "var(--qt-danger-text)" }}
                    >
                        {error}
                    </span>
                ) : null}
            </Flex>
            {/* 展开区：只在录制中出现。它装的是"录制中才有意义"的三个读数 ——
                宿主在不在播、峰值、分段数；不录制时它们一个都不该占地方。 */}
            {recording ? (
                <Flex align="center" gap="2" data-transfer-details>
                    <span className="hs-type-caption">
                        {status?.hostPlaying
                            ? t("transfer_host_playing")
                            : t("transfer_field_waiting_host")}
                    </span>
                    <span className="hs-type-caption">
                        {tVars("transfer_field_peak", {
                            left: (status?.recording?.peak?.[0] ?? 0).toFixed(2),
                            right: (status?.recording?.peak?.[1] ?? 0).toFixed(2),
                        })}
                    </span>
                    <span className="hs-type-caption">
                        {tVars("transfer_field_segments_value", {
                            count: String(status?.recording?.sealedSegments ?? 0),
                        })}
                    </span>
                </Flex>
            ) : null}
        </Flex>
    );
}
