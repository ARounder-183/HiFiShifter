/**
 * 插件里的「导出到 HiFiShifter」控件（状态栏）。
 *
 * 【为什么入口在状态栏】导出之后用户要去**独立 App** 里编辑 —— 也就是离开这个插件
 * 窗口。它不属于编辑工作流，而是一句"我把工作区放在哪了"。状态栏右侧本来就是插件
 * 自己的读数槽（`PluginApplyStatus` 就在那里）。
 *
 * 【为什么状态来自 store 而不是 useState】这段读数挂在 `AppInner` 上；把 750ms 一次的
 * 轮询 setState 放进去，等于每轮重渲染整棵应用树（见 `hostLinkPluginStore`）。
 *
 * 【为什么要显示目录路径】这是这条通路唯一需要用户手动完成的一步：导出的工作区在磁盘
 * 上，用户必须把它的**路径**交给独立 App。路径因此放在明面上，并配一个复制按钮。
 */
import { useEffect, useRef, useState } from "react";
import { Flex } from "@radix-ui/themes";

import { useI18n } from "../../i18n/I18nProvider";
import { AppButton, AppDialog, AppField, AppStatusChip } from "../../ui";
import { copyTextToClipboard } from "../../utils/copyText";
import { hostLinkStatusText, type HostLinkExportPhase } from "./hostLinkApi";
import {
    exportHostLinkWorkspace,
    releaseHostLinkWorkspace,
    startHostLinkPluginPolling,
    useHostLinkPluginSnapshot,
} from "./hostLinkPluginStore";

export function HostLinkExportControl() {
    const { t, tVars } = useI18n();
    const { status, failure, busy, ended } = useHostLinkPluginSnapshot();
    const [details, setDetails] = useState(false);
    const [copied, setCopied] = useState(false);
    // "已复制"的复位定时器：句柄必须跟踪 —— 连续点击会叠加多个定时器
    // （先复位的那个会把后来那次的高亮提前掐掉），卸载后也会留下对已卸载
    // 组件的 setState。
    const copiedResetTimer = useRef<number | null>(null);

    useEffect(() => startHostLinkPluginPolling(), []);

    useEffect(
        () => () => {
            if (copiedResetTimer.current !== null) window.clearTimeout(copiedResetTimer.current);
        },
        [],
    );

    const phase = status?.attached ? status.phase : undefined;
    const directory = status?.attached ? status.directory : undefined;
    const lastEdit = status?.attached ? status.lastEdit : undefined;

    async function copyDirectory() {
        if (!directory) return;
        if (await copyTextToClipboard(directory)) {
            setCopied(true);
            if (copiedResetTimer.current !== null) window.clearTimeout(copiedResetTimer.current);
            copiedResetTimer.current = window.setTimeout(() => {
                copiedResetTimer.current = null;
                setCopied(false);
            }, 1500);
        }
    }

    return (
        <Flex align="center" gap="1" className="min-w-0">
            <AppButton
                size="sm"
                disabled={busy || phase === "running" || ended}
                onClick={() => void exportHostLinkWorkspace()}
            >
                {t("host_link_export")}
            </AppButton>
            <AppStatusChip tone={phaseTone(phase, failure)} tooltip={failure || undefined}>
                {phaseLabel(phase, t)}
            </AppStatusChip>

            {/* 导出成功之后，路径是用户唯一需要手抄的东西 —— 放在明面上并给复制按钮。 */}
            {directory ? (
                <>
                    <span
                        className="hs-type-caption min-w-0 truncate"
                        style={{ maxWidth: 220, color: "var(--qt-text-muted)" }}
                        data-tooltip={directory}
                    >
                        {directory}
                    </span>
                    <AppButton size="sm" intent="subtle" onClick={() => void copyDirectory()}>
                        {copied ? t("host_link_copied") : t("host_link_copy_path")}
                    </AppButton>
                </>
            ) : null}

            <AppButton size="sm" intent="subtle" onClick={() => setDetails(true)}>
                {t("host_link_details")}
            </AppButton>

            {phase === "ready" ? (
                <AppButton
                    size="sm"
                    intent="subtle"
                    disabled={busy}
                    onClick={() => void releaseHostLinkWorkspace()}
                >
                    {t("host_link_release")}
                </AppButton>
            ) : null}

            {status?.attached && status.phase === "failed" ? (
                <span
                    role="alert"
                    className="hs-type-label min-w-0 truncate"
                    style={{ color: "var(--qt-danger-text)" }}
                >
                    {tVars("host_link_export_error", { error: status.error ?? "" })}
                </span>
            ) : null}

            <AppDialog
                open={details}
                onOpenChange={setDetails}
                title={t("host_link_details_title")}
                size="md"
            >
                <Flex direction="column" gap="3">
                    <AppField label={t("host_link_folder")}>
                        {directory ? (
                            <Flex align="center" gap="2">
                                <span
                                    className="hs-type-mono min-w-0 truncate"
                                    data-tooltip={directory}
                                >
                                    {directory}
                                </span>
                                <AppButton
                                    size="sm"
                                    intent="subtle"
                                    onClick={() => void copyDirectory()}
                                >
                                    {copied ? t("host_link_copied") : t("host_link_copy_path")}
                                </AppButton>
                            </Flex>
                        ) : (
                            <span style={{ color: "var(--qt-text-muted)" }}>
                                {t("host_link_phase_idle")}
                            </span>
                        )}
                    </AppField>
                    {lastEdit ? (
                        <AppField label={t("host_link_last_edit_label")}>
                            <span
                                style={{
                                    color: lastEdit.ok ? undefined : "var(--qt-danger-text)",
                                }}
                            >
                                {lastEdit.ok
                                    ? t("host_link_last_edit_applied")
                                    : hostLinkStatusText(lastEdit, t, tVars)}
                            </span>
                        </AppField>
                    ) : null}
                    <AppField label={t("host_link_status_json_label")}>
                        <pre
                            className="hs-type-mono custom-scrollbar hs-scroll-gutter-flush max-h-64 overflow-auto rounded border border-qt-border bg-qt-window p-qt-2"
                            data-testid="host-link-status-json"
                        >
                            {JSON.stringify(status ?? { attached: false }, null, 2)}
                        </pre>
                    </AppField>
                </Flex>
            </AppDialog>
        </Flex>
    );
}

/** 阶段 → 状态词语表键。不拼字符串。 */
function phaseLabel(
    phase: HostLinkExportPhase | undefined,
    t: (key: HostLinkMessageKey) => string,
): string {
    switch (phase) {
        case "running":
            return t("host_link_phase_running");
        case "failed":
            return t("host_link_phase_failed");
        case "ready":
            return t("host_link_phase_ready");
        default:
            return t("host_link_phase_idle");
    }
}

function phaseTone(
    phase: HostLinkExportPhase | undefined,
    failure: string,
): "neutral" | "accent" | "success" | "danger" {
    if (failure) return "danger";
    switch (phase) {
        case "running":
            return "accent";
        case "failed":
            return "danger";
        case "ready":
            return "success";
        default:
            return "neutral";
    }
}

type HostLinkMessageKey =
    | "host_link_phase_idle"
    | "host_link_phase_running"
    | "host_link_phase_failed"
    | "host_link_phase_ready";
