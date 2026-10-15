/**
 * 传输栏上的 Transfer 控件（插件形态的主入口）。
 *
 * 【为什么入口在这里而不是一个浮动面板】用户对"录制"这件事的心智模型就是传输栏上的
 * 一个键：Melodyne 的 Transfer 也长在那里。此前面板是 460×320 的浮窗、只有几个调试
 * 读数，用户找不到它、也不知道它和播放有什么关系。
 *
 * 【为什么它同时是状态显示】Transfer 的前提是"宿主先播放，插件才能录"（VST3 里插件
 * **无法**请求宿主播放）。控件因此必须自己回答"现在缺哪一步"：没播放、没素材、
 * 正在录、录好了 —— 而不是让用户对着一个灰按钮猜。
 *
 * 【为什么读数取工程秒而不是"帧数"】用户看到的时间轴单位是秒；后端给的是工程帧
 * （那才是唯一无歧义的量），换算只发生在 `frameToSec` 一处。
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
    const araBound = status?.araBound ?? false;
    const canCapture = !araBound && (status?.canCapture ?? false);
    // 录制中但宿主没在播放 ⇒ 一个字节也进不来。这是这条通路最常见的失败
    // （用法是"先按录制、再按播放"），必须直接说出来。
    const stalled = recording && !(status?.hostPlaying ?? false);
    const captured = status?.recording
        ? framesToSeconds(
              status.recording.capturedFrames,
              status.capture?.sampleRate || FALLBACK_SAMPLE_RATE,
          )
        : null;

    const tooltip = araBound
        ? t("transfer_not_applicable_ara")
        : !canCapture
          ? t("transfer_cannot_capture")
          : recording
            ? t("transfer_guide_recording")
            : (status?.takeCount ?? 0) > 0
              ? t("transfer_guide_ready")
              : t("transfer_guide_body");

    return (
        <Flex align="center" gap="1" className="shrink-0" data-transfer-control>
            <AppButton
                intent={recording ? "danger" : "subtle"}
                emphasis={recording ? "solid" : undefined}
                size="sm"
                disabled={busy || (!recording && !canCapture)}
                data-tooltip={tooltip}
                onClick={() => void (recording ? transferActions.stop() : transferActions.start())}
            >
                {recording ? t("transfer_stop") : t("transfer_start")}
            </AppButton>
            {recording ? (
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
                    <AppStatusChip tone="warning" tabular tooltip={t("transfer_guide_recording")}>
                        {tVars("transfer_recording_progress", { seconds: captured ?? "0.0" })}
                    </AppStatusChip>
                </>
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
    );
}
