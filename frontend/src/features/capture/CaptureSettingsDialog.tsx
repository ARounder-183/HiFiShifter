/**
 * 捕获设置。
 *
 * 【为什么是一个对话框，而不是原来那个浮动面板】旧面板（`CapturePanel`）把两件完全
 * 不同的事挤在一张 460×320 的浮窗里：**改变行为**（监听、区间外处理、单次上限）与
 * **刚发生了什么**（分段数、跳跃次数、剩余缓冲、峰值、take 列表、宿主观察位）。
 * 前者是用户会反复回来调的设置，后者是排查时才看的读数 —— 混在一起的结果是两边都
 * 不顺手，而且它用的是自造的排布，与本项目其它设置界面（`RecordingSettingsDialog`、
 * `TimelineDisplaySettingsDialog`）长得完全不一样。
 *
 * 现在：设置走 `AppDialog` + `AppForm`（与全站设置同一套排布），诊断读数搬去
 * 诊断包（`plugin_diagnostics` 的 system_info 里已经有捕获读数）。
 *
 * # 为什么"为什么现在不能用"住在这里
 *
 * 传输栏那个按钮的悬停气泡只能放一句短语 —— 而"宿主没把播放位置交给插件"这件事
 * 需要一整句才能说清，且用户需要据此决定下一步（去按播放？去检查路由？去改宿主
 * 精度？）。这里就是读那句话的地方：三条入口（右键按钮、「视图」菜单、出错图标）
 * 都指向同一个窗口。
 */
import { useEffect } from "react";
import { Flex } from "@radix-ui/themes";

import { useI18n } from "../../i18n/I18nProvider";
import type { MessageKey } from "../../i18n/messages";
import { AppButton, AppField, AppForm, AppSelect, AppStatusChip, AppSwitchRow } from "../../ui";
import { AppDialog } from "../../ui/Dialog";
import type { CaptureStatus } from "./captureApi";
import { captureActions, startCapturePolling, useCaptureSnapshot } from "./captureStore";

/** 单次捕获上限的可选档位（秒）。 */
const LIMIT_CHOICES_SEC = [60, 120, 300, 600, 1800] as const;

interface CaptureSettingsDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    /** 传输栏捕获到的最后一条后端错误（`null` = 没有）。 */
    backendError?: string;
}

export function CaptureSettingsDialog({
    open,
    onOpenChange,
    backendError,
}: CaptureSettingsDialogProps) {
    const { t, tVars } = useI18n();
    const { status, busy } = useCaptureSnapshot();

    // 对话框可能是**唯一**的挂载点（用户直接从菜单打开时，传输栏那条轮询也在跑；
    // 引用计数保证只有一条）。
    useEffect(() => startCapturePolling(), []);

    const options = status?.options ?? null;
    const limitSeconds = Math.round(options?.maxDurationSec ?? 0);
    const reasonKey = captureUnavailableReason(status);
    const reason = reasonKey === null ? null : t(reasonKey);

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={t("capture_settings_title")}
            description={t("capture_settings_desc")}
            size="sm"
            actions={[
                {
                    id: "clear",
                    label: t("capture_clear"),
                    intent: "danger",
                    align: "start",
                    disabled: busy || (status?.takeCount ?? 0) === 0,
                    onClick: () => void captureActions.clear(),
                },
                { id: "close", label: t("close"), onClick: () => onOpenChange(false) },
            ]}
        >
            <AppForm labelWidth="md" booleanRow="aligned">
                {/* 状态：一句话说清"现在能不能用、为什么"。这是旧面板里最有价值的那部分。 */}
                <AppField label={t("capture_status")}>
                    <AppStatusChip tone={reason ? "warning" : "accent"}>
                        {reason ?? t(statusKey(status?.state))}
                    </AppStatusChip>
                </AppField>

                {/* 后端错误原文：只在有错误时出现，且**可复制**（用户要把它贴进 issue）。 */}
                {backendError ? (
                    <AppField label={t("capture_last_error")}>
                        <span
                            role="alert"
                            data-hs-selectable="true"
                            className="hs-type-caption"
                            style={{ color: "var(--qt-danger-text)" }}
                        >
                            {backendError}
                        </span>
                    </AppField>
                ) : null}

                <AppSwitchRow
                    control="switch"
                    label={t("capture_option_monitor")}
                    hint={t("capture_option_monitor_hint")}
                    checked={options?.monitor ?? true}
                    disabled={busy || options === null}
                    onCheckedChange={(checked) =>
                        void captureActions.setOptions({ monitor: checked })
                    }
                />
                <AppSwitchRow
                    control="switch"
                    label={t("capture_option_outside")}
                    hint={t("capture_option_outside_hint")}
                    checked={options?.outsidePassthrough ?? true}
                    disabled={busy || options === null}
                    onCheckedChange={(checked) =>
                        void captureActions.setOptions({ outsidePassthrough: checked })
                    }
                />
                {/* 【为什么上限是一个下拉而不是一排按钮】旧面板用 6 个 `AppButton` 当档位
                    选择器 —— 那是自造的选择器，且占了半屏宽度。`AppSelect` 是全站既有的
                    档位原语，在插件那种窄窗口里也不会换行。 */}
                <AppField label={t("capture_option_limit")} hint={t("capture_option_limit_hint")}>
                    <AppSelect
                        value={String(limitSeconds)}
                        disabled={busy || options === null}
                        ariaLabel={t("capture_option_limit")}
                        options={LIMIT_CHOICES_SEC.map((seconds) => ({
                            value: String(seconds),
                            label: tVars("capture_option_limit_seconds", {
                                seconds: String(seconds),
                            }),
                        }))}
                        onValueChange={(next) =>
                            void captureActions.setOptions({ maxDurationSec: Number(next) })
                        }
                    />
                </AppField>

                {/* **本机缺失的捕获音频**。
                    sidecar 落在本机缓存目录，清单随宿主工程走 —— 把工程拷到另一台
                    机器时音频不会跟过去。这里必须说清"丢了什么、有多少、怎么办"，
                    而不是让用户面对一片空白的时间轴。
                    【为什么用 ?? []】missingTakes 是后加的字段：旧后端回执（以及
                    测试里的"空对象"兜底 mock）没有它 —— 这里不能假设字段存在。 */}
                {status && (status.missingTakes ?? []).length > 0 ? (
                    <AppField label={t("capture_missing_title")}>
                        <span role="alert" style={{ color: "var(--qt-warning-text)" }}>
                            {tVars("capture_missing_body", {
                                count: String((status.missingTakes ?? []).length),
                                seconds: (status.missingTakes ?? [])
                                    .reduce((sum, take) => sum + take.seconds, 0)
                                    .toFixed(1),
                            })}
                        </span>
                    </AppField>
                ) : null}

                {/* take 清单：这是"操作"，不是"读数" —— 用户真的会按那个删除键。
                    takes 同 missingTakes：按缺字段防御（同上）。 */}
                {status && (status.takes ?? []).length > 0 ? (
                    <AppField label={t("capture_field_takes")}>
                        <Flex direction="column" gap="1">
                            {(status.takes ?? []).map((take) => (
                                <Flex key={take.id} align="center" gap="2">
                                    <AppStatusChip
                                        tone={take.active ? "accent" : "neutral"}
                                        tabular
                                    >
                                        {tVars("capture_take_label", {
                                            id: String(take.id),
                                            clips: String(take.clips),
                                        })}
                                    </AppStatusChip>
                                    {take.startSec !== null && take.endSec !== null ? (
                                        <span className="hs-type-caption">
                                            {tVars("capture_take_span", {
                                                start: take.startSec.toFixed(2),
                                                end: take.endSec.toFixed(2),
                                            })}
                                        </span>
                                    ) : null}
                                    <AppButton
                                        size="sm"
                                        intent="subtle"
                                        disabled={busy}
                                        onClick={() => void captureActions.deleteTake(take.id)}
                                    >
                                        {t("capture_take_delete")}
                                    </AppButton>
                                </Flex>
                            ))}
                        </Flex>
                    </AppField>
                ) : null}
            </AppForm>
        </AppDialog>
    );
}

/** 状态 → 词表键。不拼字符串。 */
function statusKey(state: CaptureStatus["state"] | undefined) {
    switch (state) {
        case "standby":
            return "capture_status_standby" as const;
        case "recording":
            return "capture_status_recording" as const;
        case "ready":
            return "capture_status_ready" as const;
        default:
            return "capture_status_empty" as const;
    }
}

/**
 * "这条通路现在成不成立"，不成立时给出**一句话**的原因。
 *
 * 【为什么顺序是这样】先讲通路（精度 / ARA / 没有位置 / 没有输入）—— 那些是"按了也没用"；
 * 再讲这次捕获怎么了（上限预警 / 宿主没在播放）。用户在设置窗口里最想先知道的是前者。
 *
 * 【为什么"没在播放"不算"不可用"】它只说明这一次还没开始，按一下播放就好 —— 把它
 * 报成不可用会让用户以为插件坏了。
 */
function captureUnavailableReason(status: CaptureStatus | null): MessageKey | null {
    if (!status) return null;
    if (status.unsupportedPrecision) return "capture_unsupported_precision";
    if (status.araBound) return "capture_not_applicable_ara";
    if (!status.canCapture) return "capture_cannot_capture";
    if (status.sawInput === false) return "capture_no_input_audio";
    if (status.limitWarning) return "capture_limit_warning";
    return null;
}
