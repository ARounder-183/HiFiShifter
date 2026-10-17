/**
 * Transfer 的**设置与诊断**面板（仅在不支持 ARA 的宿主里出现）。
 *
 * 【为什么录制键不在这里】主入口是传输栏上的 [`TransferControl`] —— 用户对"录制"的
 * 心智模型就是传输栏上的一个键（Melodyne 的 Transfer 也长在那里）。这个面板回答的是
 * 另外几个问题：**刚发生了什么**（读数、take 列表、为什么没有声音、为什么自己停了）
 * 与**怎么改变行为**（监听、区间外处理、单次录制上限）。
 *
 * 【为什么引导语由观察位决定】`sawContext` / `sawInput` / `sawOffline` 描述宿主
 * **实际**做了什么。没见过位置时不能说"还没播放"（可能宿主永远不给），而要说
 * "这个宿主没有把播放位置交给插件" —— 那是两个不同的、都能让用户据以行动的事实。
 *
 * 【为什么"为什么自己停了"要单独一条】到上限、内存没跟上、宿主换采样率这三件事都会
 * 让录制自己停下，而补救完全不同（改上限 / 清旧录音 / 什么都不用做）。合成一句
 * "已停止"等于让用户对着一个没有原因的终局猜。
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

/** 单次录制上限的可选档位（秒）。0 = 不限。 */
const LIMIT_CHOICES_SEC = [60, 120, 300, 600, 1800, 0] as const;

export function TransferPanel() {
    const { t, tVars } = useI18n();
    const { status, busy, error } = useTransferSnapshot();

    useEffect(() => startTransferPolling(), []);

    const recording = status?.state === "recording";
    const standby = status?.state === "standby";
    const araBound = status?.araBound ?? false;
    const canCapture = !araBound && (status?.canCapture ?? false);
    const rate =
        status?.capture?.sampleRate || status?.material?.sampleRate || FALLBACK_SAMPLE_RATE;

    return (
        <Flex direction="column" gap="3" p="3">
            <Flex align="center" gap="2">
                <AppStatusChip tone={recording || standby ? "warning" : "accent"} tabular>
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

            {/* 引导：按录制、再播放。这是这条通路存在的首要理由（见文件头）。 */}
            <p>{guideText(t, status, canCapture)}</p>

            {/* 【为什么"录到了但没有内容"单独一条】全静音的录制不产生片段，于是时间轴
                空白、面板却说"已录制" —— 两个读数互相矛盾。这一条把原因说穿。 */}
            {status?.material?.silent ? (
                <span role="alert" style={{ color: "var(--qt-warning-text)" }}>
                    {tVars("transfer_recorded_silent", {
                        seconds: framesToSeconds(status.material.capturedFrames, rate),
                    })}
                </span>
            ) : null}

            {/* 宿主在播放、却一帧都没到 ⇒ 插件没拿到音频。这条与"没在播放"分开说，
                因为要做的事完全不同（检查路由 vs 去按播放）。 */}
            {recording && status.hostPlaying && (status.recording?.capturedFrames ?? 0) === 0 ? (
                <span role="alert" style={{ color: "var(--qt-danger-text)" }}>
                    {t("transfer_no_audio_arriving")}
                </span>
            ) : null}

            {/* 上限预警：上限到点会**自己停**，而那是用户设的一个数值。 */}
            {status?.limitWarning ? (
                <span role="alert" style={{ color: "var(--qt-warning-text)" }}>
                    {t("transfer_limit_warning")}
                </span>
            ) : null}

            {/* 上一次录制为什么自己停了。三件事的补救完全不同，所以分开说。 */}
            {status?.stoppedReason ? (
                <span role="alert" style={{ color: "var(--qt-warning-text)" }}>
                    {stoppedReasonText(t, status.stoppedReason)}
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
                    {/* 【为什么峰值读数在这一栏】它此前长在工具栏上，而工具栏是宿主窗口
                        里的一条单行容器 —— 峰值与分段数、备用缓冲同一性质（都是"刚发生了
                        什么"），所以它们应当在一起（面板），而不是挤在键旁边。 */}
                    <AppField label={t("transfer_field_peak")}>
                        {tVars("transfer_field_peak_value", {
                            left: (status?.recording?.peak?.[0] ?? 0).toFixed(2),
                            right: (status?.recording?.peak?.[1] ?? 0).toFixed(2),
                        })}
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
                {/* 【为什么上限必须是可见选项】它此前是一个**从未被执行**的设置：
                    默认 10 分钟，但没有任何一处比较它与已捕获的帧数。一个不生效的上限
                    比没有上限更糟 —— 用户以为到点会自动停，于是放心去干别的。 */}
                <AppField label={t("transfer_option_limit")}>
                    <Flex gap="1" wrap="wrap">
                        {LIMIT_CHOICES_SEC.map((seconds) => {
                            const current = Math.round(status?.options.maxDurationSec ?? 0);
                            return (
                                <AppButton
                                    key={seconds}
                                    size="sm"
                                    intent={current === seconds ? "primary" : "subtle"}
                                    disabled={busy}
                                    onClick={() =>
                                        void transferActions.setOptions({
                                            maxDurationSec: seconds === 0 ? 1 : seconds,
                                        })
                                    }
                                >
                                    {seconds === 0
                                        ? t("transfer_option_limit_unlimited")
                                        : tVars("transfer_option_limit_seconds", {
                                              seconds: String(seconds),
                                          })}
                                </AppButton>
                            );
                        })}
                    </Flex>
                </AppField>
                <span className="hs-type-caption">{t("transfer_option_limit_hint")}</span>
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
                {/* 【为什么"现在听到的是哪一版"要常驻】处理结果、原始录音与宿主原声
                    听起来都"像那么回事"，而它们之间的差别正是"我的编辑生效了没有" ——
                    用户无法自己判断。后端已经在仲裁，这里只是把它说出来。 */}
                {status?.output ? (
                    <span className="hs-type-caption">
                        {tVars("transfer_now_playing", { what: outputText(t, status.output) })}
                    </span>
                ) : null}
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
        case "standby":
            return "transfer_status_standby" as const;
        case "recording":
            return "transfer_status_recording" as const;
        case "ready":
            return "transfer_status_ready" as const;
        default:
            return "transfer_status_empty" as const;
    }
}

/** 后端 `ClosedReason::as_str` → 词表键。未知取值原样透出（不猜）。 */
function stoppedReasonText(t: (key: TransferMessageKey) => string, reason: string): string {
    switch (reason) {
        case "limit_reached":
            return t("transfer_stopped_limit");
        case "no_spare":
            return t("transfer_stopped_no_spare");
        case "rate_changed":
            return t("transfer_stopped_rate_changed");
        case "budget_full":
            return t("transfer_stopped_budget_full");
        case "jump":
            return t("transfer_stopped_jump");
        default:
            return reason;
    }
}

/** 后端 `TransferOutput::as_str` → 词表键。未知取值原样透出（不猜）。 */
function outputText(t: (key: TransferMessageKey) => string, output: string): string {
    switch (output) {
        case "processed":
            return t("transfer_output_processed");
        // 【为什么要单独一档】它说的是"上一版处理结果，新的还没合成好"。与
        // `processed` 共用一句会让用户以为听到的就是最新编辑 —— 而那正是他无法自己
        // 判断的那一件事（见 `outputText` 上方对"现在听到哪一版"的说明）。
        case "stale_processed":
            return t("transfer_output_stale");
        case "raw":
            return t("transfer_output_raw");
        case "recording":
            return t("transfer_output_recording");
        // 【为什么要单独一档】它说的不是"还没有素材"，而是"素材已经在了，但处理
        // 结果还没算出来"。两者输出内容相同（都是宿主原声），含义却完全不同：
        // 后者是"该有结果却没有"，用户据此知道该等、还是该报障。
        case "pending_synthesis":
            return t("transfer_output_pending");
        case "passthrough":
        case "untouched":
            return t("transfer_output_passthrough");
        default:
            return output;
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
    if (status?.state === "standby") return t("transfer_guide_standby");
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
    | "transfer_guide_standby"
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
    | "transfer_option_limit"
    | "transfer_option_limit_hint"
    | "transfer_option_limit_seconds"
    | "transfer_option_limit_unlimited"
    | "transfer_take_delete"
    | "transfer_limit_warning"
    | "transfer_stopped_limit"
    | "transfer_stopped_no_spare"
    | "transfer_stopped_rate_changed"
    | "transfer_stopped_budget_full"
    | "transfer_stopped_jump"
    | "transfer_now_playing"
    | "transfer_output_processed"
    | "transfer_output_stale"
    | "transfer_output_raw"
    | "transfer_output_recording"
    | "transfer_output_pending"
    | "transfer_output_passthrough"
    | "transfer_status_standby"
    | "transfer_yes"
    | "transfer_no";
