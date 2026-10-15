/**
 * Transfer 面板（仅在不支持 ARA 的宿主里出现）。
 *
 * 【它解决什么】VST3 里插件**无法**请求宿主播放：`IComponentHandler` 的三个版本里
 * 都没有播放请求，工程位置也只有 `ProcessContext` 一个来源。所以用户必须先按播放，
 * 插件才能把音频录进来。这个面板存在的首要理由就是**把这件事说清楚** ——
 * 否则用户面对的是一个按下去没有任何反应的按钮。
 *
 * 【为什么状态是轮询而不是事件】录制进度每 200ms 变一次，宿主不会推送。轮询
 * `transfer_status` 的代价只是一个整数读取（后端那条命令不取渲染锁、不合成）。
 *
 * 【为什么引导语由观察位决定】`sawContext` / `sawInput` / `sawOffline` 描述宿主
 * **实际**做了什么。没见过位置时不能说"还没播放"（可能宿主永远不给），而要说
 * "这个宿主没有把播放位置交给插件" —— 那是两个不同的、都能让用户据以行动的事实。
 */
import { useCallback, useEffect, useState } from "react";
import { Flex } from "@radix-ui/themes";

import { useI18n } from "../../i18n/I18nProvider";
import { AppButton, AppField, AppStatusChip } from "../../ui";
import { pluginBootstrap } from "../../services/hostDetect";
import {
    framesToSeconds,
    transferApi,
    transferErrorText,
    type TransferStatus,
} from "./transferApi";

/** 轮询间隔：与后端录制读数的更新节奏一致（见 `transfer/mod.rs` 的说明）。 */
const POLL_MS = 200;

/** 帧→秒显示用的采样率；后端不把它塞进状态里，取不到时按 48k 显示量级即可。 */
const DISPLAY_SAMPLE_RATE = 48_000;

export function TransferPanel() {
    const { t, tVars } = useI18n();
    const [status, setStatus] = useState<TransferStatus | null>(null);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState("");

    const refresh = useCallback(async () => {
        try {
            setStatus(await transferApi.status());
            setError("");
        } catch (cause) {
            // 【为什么不在这里停止轮询】状态查询失败是暂时的（会话未就绪等），
            // 而面板的职责是持续反映真实状态。终态的判定由 `invoke` 的统一闸门做。
            setError(transferErrorText(cause, t));
        }
    }, []);

    useEffect(() => {
        void refresh();
        const timer = window.setInterval(() => void refresh(), POLL_MS);
        return () => window.clearInterval(timer);
    }, [refresh]);

    const run = useCallback(
        async (action: () => Promise<unknown>) => {
            setBusy(true);
            try {
                await action();
                await refresh();
            } catch (cause) {
                setError(transferErrorText(cause, t));
            } finally {
                setBusy(false);
            }
        },
        [refresh],
    );

    const recording = status?.recording ?? null;
    // 宿主正在通过 ARA 供音 ⇒ Transfer 不适用（后端 `transfer_start` 也会拒）。
    const araBound = status?.araBound ?? false;
    const canCapture =
        !araBound && (status?.canCapture ?? pluginBootstrap()?.transferCaps?.canCapture ?? false);
    const recordingNow = status?.state === "recording";
    // 录制中但宿主没在播放 ⇒ 一个字节也进不来。这是这条通路最常见的失败（用法是
    // "先按录制、再按播放"），必须直接说出来。
    const stalled = recordingNow && !(status?.hostPlaying ?? false);

    return (
        <Flex direction="column" gap="3" p="3">
            <Flex align="center" gap="2">
                <AppStatusChip tone={recordingNow ? "warning" : "accent"} tabular>
                    {t(statusKey(status?.state))}
                </AppStatusChip>
                {recording ? (
                    <span>
                        {tVars("transfer_recording_progress", {
                            seconds: framesToSeconds(recording.capturedFrames, DISPLAY_SAMPLE_RATE),
                        })}
                    </span>
                ) : null}
            </Flex>

            {/* 引导：先按播放，再录制。这是本面板存在的首要理由（见文件头）。 */}
            <p>{araBound ? t("transfer_not_applicable_ara") : guideText(t, canCapture, status)}</p>

            <Flex gap="2" wrap="wrap">
                {recordingNow ? (
                    <>
                        <AppButton
                            intent="primary"
                            onClick={() => void run(transferApi.stop)}
                            disabled={busy}
                        >
                            {t("transfer_stop")}
                        </AppButton>
                        <AppButton
                            intent="subtle"
                            onClick={() => void run(transferApi.cancel)}
                            disabled={busy}
                        >
                            {t("transfer_cancel")}
                        </AppButton>
                    </>
                ) : (
                    <AppButton
                        intent="primary"
                        onClick={() => void run(transferApi.start)}
                        disabled={busy || !canCapture}
                        data-tooltip={canCapture ? undefined : t("transfer_cannot_capture")}
                    >
                        {t("transfer_start")}
                    </AppButton>
                )}
                <AppButton
                    intent="danger"
                    onClick={() => void run(transferApi.clear)}
                    disabled={busy || (status?.takeCount ?? 0) === 0}
                >
                    {t("transfer_clear")}
                </AppButton>
            </Flex>

            {recording ? (
                <Flex direction="column" gap="2">
                    <AppField label={t("transfer_field_segments")}>
                        {String(recording.sealedSegments)}
                    </AppField>
                    <AppField label={t("transfer_field_gaps")}>{String(recording.gaps)}</AppField>
                    {/* 备用缓冲见底 = 录制即将停止；这条读数让"为什么停了"有据可查。 */}
                    <AppField label={t("transfer_field_spares")}>
                        {String(recording.spares)}
                    </AppField>
                </Flex>
            ) : null}

            {status && status.takeCount > 0 ? (
                <AppField label={t("transfer_field_takes")}>{String(status.takeCount)}</AppField>
            ) : null}

            {stalled ? (
                <span className="hs-type-caption" style={{ color: "var(--qt-warning-text)" }}>
                    {t("transfer_host_not_playing")}
                </span>
            ) : null}

            {error ? <span role="alert">{error}</span> : null}
        </Flex>
    );
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
    canCapture: boolean,
    status: TransferStatus | null,
): string {
    if (status?.state === "recording") return t("transfer_guide_recording");
    if (status?.state === "ready") return t("transfer_guide_ready");
    if (status?.sawOffline) return t("transfer_guide_offline");
    if (canCapture) return t("transfer_guide_body");
    return t("transfer_cannot_capture");
}

type TransferMessageKey =
    | "transfer_not_applicable_ara"
    | "transfer_host_not_playing"
    | "transfer_guide_body"
    | "transfer_guide_recording"
    | "transfer_guide_ready"
    | "transfer_guide_offline"
    | "transfer_cannot_capture";
