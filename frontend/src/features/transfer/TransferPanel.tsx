/**
 * Transfer 的**设置与诊断**面板（仅在不支持 ARA 的宿主里出现）。
 *
 * 【为什么录制键不在这里】主入口是传输栏上的 [`TransferControl`] —— 用户对"录制"的
 * 心智模型就是传输栏上的一个键（Melodyne 的 Transfer 也长在那里）。这个面板回答的是
 * 另外两个问题：**刚发生了什么**（读数、take 列表、为什么没有声音）与**怎么改变行为**
 * （监听、区间外处理）。
 *
 * 【为什么引导语由观察位决定】`sawContext` / `sawInput` / `sawOffline` 描述宿主
 * **实际**做了什么。没见过位置时不能说"还没播放"（可能宿主永远不给），而要说
 * "这个宿主没有把播放位置交给插件" —— 那是两个不同的、都能让用户据以行动的事实。
 *
 * 【为什么状态来自共享 store】它与传输栏上的控件必须看到**同一份**读数；各拉各的会
 * 让两处错开一帧（控件说"录制中"、面板说"已录好"，用户会当成 bug）。
 */
import { useEffect } from "react";
import { Flex } from "@radix-ui/themes";

import { useI18n } from "../../i18n/I18nProvider";
import { AppButton, AppEmptyState, AppField, AppStatusChip, AppSwitchRow } from "../../ui";
import { framesToSeconds, type TransferStatus } from "./transferApi";
import { startTransferPolling, transferActions, useTransferSnapshot } from "./transferStore";

/** 帧→秒显示用的兜底采样率；后端还没收到宿主上下文时给 0。 */
const FALLBACK_SAMPLE_RATE = 48_000;

export function TransferPanel() {
    const { t, tVars } = useI18n();
    const { status, busy, error } = useTransferSnapshot();

    useEffect(() => startTransferPolling(), []);

    const recording = status?.state === "recording";
    const araBound = status?.araBound ?? false;
    const canCapture = !araBound && (status?.canCapture ?? false);
    const rate =
        status?.capture?.sampleRate || status?.material?.sampleRate || FALLBACK_SAMPLE_RATE;

    return (
        <Flex direction="column" gap="3" p="3">
            <Flex align="center" gap="2">
                <AppStatusChip tone={recording ? "warning" : "accent"} tabular>
                    {t(statusKey(status?.state))}
                </AppStatusChip>
                {recording ? (
                    <span>
                        {tVars("transfer_recording_progress", {
                            seconds: framesToSeconds(status?.recording?.capturedFrames ?? 0, rate),
                        })}
                    </span>
                ) : null}
                {status?.hostPlaying ? (
                    <AppStatusChip tone="success">{t("transfer_host_playing")}</AppStatusChip>
                ) : null}
            </Flex>

            {/* 引导：先按播放，再录制。这是这条通路存在的首要理由（见文件头）。 */}
            <p>{guideText(t, status, canCapture)}</p>

            {/* 【为什么"录到了但没有内容"单独一条】全静音的录制不产生片段，于是时间轴
                空白、面板却说"已录制" —— 三个读数互相矛盾。这一条把原因说穿。 */}
            {/* 宿主在播放、却一帧都没到 ⇒ 插件没拿到音频。这条与"没在播放"分开说，
                因为要做的事完全不同（检查路由 vs 去按播放）。 */}
            {status?.state === "recording" &&
            status.hostPlaying &&
            (status.recording?.capturedFrames ?? 0) === 0 ? (
                <span role="alert" style={{ color: "var(--qt-danger-text)" }}>
                    {t("transfer_no_audio_arriving")}
                </span>
            ) : null}

            {status?.material?.silent ? (
                <span role="alert" style={{ color: "var(--qt-warning-text)" }}>
                    {tVars("transfer_recorded_silent", {
                        seconds: framesToSeconds(status.material.capturedFrames, rate),
                    })}
                </span>
            ) : null}

            <Flex gap="2" wrap="wrap">
                <AppButton
                    intent="danger"
                    disabled={busy || (status?.takeCount ?? 0) === 0}
                    onClick={() => void transferActions.clear()}
                >
                    {t("transfer_clear")}
                </AppButton>
            </Flex>

            {recording ? (
                <Flex direction="column" gap="2">
                    <AppField label={t("transfer_field_segments")}>
                        {String(status?.recording?.sealedSegments ?? 0)}
                    </AppField>
                    <AppField label={t("transfer_field_gaps")}>
                        {String(status?.recording?.gaps ?? 0)}
                    </AppField>
                    {/* 备用缓冲见底 = 录制即将停止；这条读数让"为什么停了"有据可查。 */}
                    <AppField label={t("transfer_field_spares")}>
                        {String(status?.recording?.spares ?? 0)}
                    </AppField>
                </Flex>
            ) : null}

            {status && status.takes.length > 0 ? (
                <Flex direction="column" gap="2">
                    <AppField label={t("transfer_field_takes")}>
                        {String(status.takes.length)}
                    </AppField>
                    {status.takes.map((take) => (
                        <Flex key={take.id} align="center" gap="2">
                            <AppStatusChip tone={take.active ? "accent" : "neutral"} tabular>
                                {tVars("transfer_take_label", {
                                    id: String(take.id),
                                    clips: String(take.clips),
                                })}
                            </AppStatusChip>
                            {take.startSec !== null && take.endSec !== null ? (
                                <span className="hs-type-caption">
                                    {tVars("transfer_take_span", {
                                        start: take.startSec.toFixed(2),
                                        end: take.endSec.toFixed(2),
                                    })}
                                </span>
                            ) : null}
                            <AppButton
                                size="sm"
                                intent="subtle"
                                disabled={busy}
                                onClick={() => void transferActions.deleteTake(take.id)}
                            >
                                {t("transfer_take_delete")}
                            </AppButton>
                        </Flex>
                    ))}
                </Flex>
            ) : (
                <AppEmptyState size="compact">{t("transfer_status_empty")}</AppEmptyState>
            )}

            <Flex direction="column" gap="2">
                <AppSwitchRow
                    control="switch"
                    label={t("transfer_option_monitor")}
                    hint={t("transfer_option_monitor_hint")}
                    checked={status?.options.monitor ?? true}
                    disabled={busy}
                    onCheckedChange={(checked) =>
                        void transferActions.setOptions({ monitor: checked })
                    }
                />
                <AppSwitchRow
                    control="switch"
                    label={t("transfer_option_outside")}
                    hint={t("transfer_option_outside_hint")}
                    checked={status?.options.outsidePassthrough ?? true}
                    disabled={busy}
                    onCheckedChange={(checked) =>
                        void transferActions.setOptions({ outsidePassthrough: checked })
                    }
                />
            </Flex>

            {/* 观察位：宿主**实际**做了什么。它是"为什么现在不能录"的唯一判据来源。 */}
            <Flex direction="column" gap="1">
                <span className="hs-type-caption">
                    {tVars("transfer_diagnostics", {
                        context: yesNo(t, status?.sawContext),
                        input: yesNo(t, status?.sawInput),
                        offline: yesNo(t, status?.sawOffline),
                    })}
                </span>
            </Flex>

            {error ? <span role="alert">{error}</span> : null}
        </Flex>
    );
}

function yesNo(t: (key: TransferMessageKey) => string, value: boolean | undefined): string {
    return value ? t("transfer_yes") : t("transfer_no");
}

/** 状态 → 词表键。不拼字符串。 */
function statusKey(state: TransferStatus["state"] | undefined) {
    switch (state) {
        case "recording":
            return "transfer_status_recording" as const;
        case "ready":
            return "transfer_status_ready" as const;
        default:
            return "transfer_status_empty" as const;
    }
}

/**
 * 引导语。
 *
 * 【为什么要分几种】"还不能录"有几个**完全不同**的原因，用户要做的事也不同：
 * 没播放过（去按播放）、宿主不给位置（这条通路在这个宿主上不成立）、已经在录
 * （去按停止）。合成一句"不可用"等于让用户自己猜。
 */
function guideText(
    t: (key: TransferMessageKey) => string,
    status: TransferStatus | null,
    canCapture: boolean,
): string {
    if (status?.unsupportedPrecision) return t("transfer_unsupported_precision");
    if (status?.araBound) return t("transfer_not_applicable_ara");
    if (status?.state === "recording") return t("transfer_guide_recording");
    if (status?.state === "ready") return t("transfer_guide_ready");
    if (status?.sawOffline) return t("transfer_guide_offline");
    if (canCapture) {
        // 有位置、没有输入：这条通路在这个宿主上"通了但没料"，用户要检查的是路由。
        return status?.sawInput ? t("transfer_guide_body") : t("transfer_no_input_audio");
    }
    return t("transfer_cannot_capture");
}

type TransferMessageKey =
    | "transfer_not_applicable_ara"
    | "transfer_host_not_playing"
    | "transfer_host_playing"
    | "transfer_guide_body"
    | "transfer_guide_recording"
    | "transfer_guide_ready"
    | "transfer_guide_offline"
    | "transfer_cannot_capture"
    | "transfer_no_input_audio"
    | "transfer_no_audio_arriving"
    | "transfer_unsupported_precision"
    | "transfer_option_monitor"
    | "transfer_option_monitor_hint"
    | "transfer_option_outside"
    | "transfer_option_outside_hint"
    | "transfer_take_delete"
    | "transfer_yes"
    | "transfer_no";
