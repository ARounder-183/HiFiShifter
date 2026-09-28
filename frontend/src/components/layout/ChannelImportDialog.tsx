/*
 * 导入声道策略设置对话框。
 *
 * 背景：真立体声源在渲染时会把整条处理器链按声道跑两遍，耗时翻倍。大量
 * "人力"素材实际是单声道内容被混流成双声道（两声道逐样本相同），折叠为
 * 单声道后听感不变、耗时减半。
 *
 * 本对话框编辑的是**导入策略**（作用于无声道权威来源的新建 Take：媒体导入、
 * VocalShifter 导入、v4 及更早工程升级）。REAPER 导入导出自带 CHANMODE，
 * 不受影响。对已导入的素材可用右键菜单的"扫描假立体声"手动重跑。
 *
 * 交互约定（与其它设置对话框一致）：
 * - 设置以草稿形式编辑，点「保存」才落盘；
 * - 草稿只在"打开"这一时机初始化一次，避免保存后 effect 重跑清掉提示；
 * - 下拉框与数字输入框统一走 `AppSelect` / `AppNumberField` 原语，滚轮
 *   步进与「精细调整」修饰键由原语内建（不再逐控件手写）。
 */

import { useEffect, useRef, useState } from "react";
import { Flex, Separator, Text } from "@radix-ui/themes";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { useI18n } from "../../i18n/I18nProvider";
import {
    TOLERANCE_PERCENT_MAX,
    normalizeChannelImportPolicy,
    percentToTolerance,
    toleranceToPercent,
    type ChannelImportMode,
    type ChannelImportPolicy,
} from "../../services/api/settings";
import { persistUiSettings } from "../../features/session/thunks/runtimeThunks";
import { setChannelImportPolicy } from "../../features/session/sessionSlice";
import { AppNumberField, AppSelect } from "../../ui";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm } from "../../ui/Field";

interface ChannelImportDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

const MODE_OPTIONS: ReadonlyArray<{ value: ChannelImportMode; labelKey: string }> = [
    { value: "smart", labelKey: "clip_channel_import_mode_smart" },
    { value: "alwaysMono", labelKey: "clip_channel_import_mode_always_mono" },
    { value: "off", labelKey: "clip_channel_import_mode_off" },
];

/** 转换目标模式（仅"智能/全部"生效）。 */
const TARGET_MODE_OPTIONS: ReadonlyArray<{ value: string; labelKey: string }> = [
    { value: "2", labelKey: "clip_channel_mode_mono_mix" },
    { value: "3", labelKey: "clip_channel_mode_mono_left" },
    { value: "4", labelKey: "clip_channel_mode_mono_right" },
];

export function ChannelImportDialog({ open, onOpenChange }: ChannelImportDialogProps) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const saved = useAppSelector((state) => state.session.channelImportPolicy);

    const [draft, setDraft] = useState<ChannelImportPolicy>(saved);
    const [saving, setSaving] = useState(false);
    const [notice, setNotice] = useState("");
    const [errorText, setErrorText] = useState("");

    // 草稿只在"打开"这一时机初始化一次：保存后 Redux 中的设置会更新，若把
    // `saved` 放进依赖，effect 会立刻重跑并把"设置已保存"的提示清掉。
    const savedRef = useRef(saved);
    savedRef.current = saved;
    useEffect(() => {
        if (!open) {
            setNotice("");
            return;
        }
        setDraft({ ...savedRef.current });
        setNotice("");
        setErrorText("");
    }, [open]);

    function patch(partial: Partial<ChannelImportPolicy>) {
        setDraft((prev) => ({ ...prev, ...partial }));
    }

    async function handleSave() {
        setErrorText("");
        setSaving(true);
        try {
            const normalized = normalizeChannelImportPolicy(draft);
            dispatch(setChannelImportPolicy(normalized));
            await dispatch(persistUiSettings());
            setDraft(normalized);
            setNotice(tAny("render_cache_settings_saved"));
        } catch {
            setErrorText(tAny("render_cache_settings_save_failed"));
        } finally {
            setSaving(false);
        }
    }

    // "智能转换"才需要采样参数；"全部转换"只用到目标模式；"不自动转换"都不需要。
    const isSmart = draft.mode === "smart";
    const isOff = draft.mode === "off";

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tAny("clip_channel_import_dialog_title")}
            description={tAny("clip_channel_import_dialog_desc")}
            size="md"
            actions={[
                { id: "cancel", label: tAny("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "save",
                    label: tAny("clip_channel_import_save"),
                    intent: "primary",
                    disabled: saving,
                    onClick: handleSave,
                },
            ]}
        >
            <AppForm>
                <Separator size="4" />

                {/* ── 总策略 ─────────────────────────────────────────── */}
                <AppField label={tAny("clip_channel_import_mode")}>
                    <AppSelect
                        value={draft.mode}
                        onValueChange={(value) => patch({ mode: value as ChannelImportMode })}
                        options={MODE_OPTIONS.map((option) => ({
                            value: option.value,
                            label: tAny(option.labelKey),
                        }))}
                    />
                </AppField>
                <Text size="1" color="gray">
                    {isOff
                        ? tAny("clip_channel_import_mode_off_hint")
                        : tAny("clip_channel_import_mode_hint")}
                </Text>

                {/* ── 目标模式 ───────────────────────────────────────── */}
                {!isOff && (
                    <AppField label={tAny("clip_channel_import_target_mode")}>
                        <AppSelect
                            value={String(draft.monoTargetMode)}
                            onValueChange={(value) => patch({ monoTargetMode: Number(value) })}
                            options={TARGET_MODE_OPTIONS.map((option) => ({
                                value: option.value,
                                label: tAny(option.labelKey),
                            }))}
                        />
                    </AppField>
                )}

                {/* ── 采样参数（仅智能模式）──────────────────────────── */}
                {isSmart && (
                    <>
                        <Separator size="4" />
                        <Text size="2" weight="medium">
                            {tAny("clip_channel_import_advanced")}
                        </Text>

                        <AppField label={tAny("clip_channel_import_tolerance")}>
                            <Flex align="center" gap="2">
                                <AppNumberField
                                    value={toleranceToPercent(draft.tolerance)}
                                    unit="percentFine"
                                    min={0}
                                    max={TOLERANCE_PERCENT_MAX}
                                    width={120}
                                    suffix="%"
                                    ariaLabel={tAny("clip_channel_import_tolerance")}
                                    onCommit={(next) =>
                                        patch({ tolerance: percentToTolerance(next) })
                                    }
                                />
                            </Flex>
                        </AppField>
                        <Text size="1" color="gray">
                            {tAny("clip_channel_import_tolerance_hint")}
                        </Text>

                        <AppField label={tAny("clip_channel_import_window_sec")}>
                            <AppNumberField
                                value={draft.windowSec}
                                unit="seconds"
                                min={0.05}
                                max={5}
                                width={120}
                                ariaLabel={tAny("clip_channel_import_window_sec")}
                                onCommit={(next) => patch({ windowSec: next })}
                            />
                        </AppField>

                        <AppField label={tAny("clip_channel_import_window_count")}>
                            <AppNumberField
                                value={draft.windowCount}
                                unit="integer"
                                min={0}
                                max={256}
                                width={120}
                                ariaLabel={tAny("clip_channel_import_window_count")}
                                onCommit={(next) => patch({ windowCount: next })}
                            />
                        </AppField>
                        <Text size="1" color="gray">
                            {tAny("clip_channel_import_window_hint")}
                        </Text>
                    </>
                )}

                {notice && (
                    <Text size="1" color="green">
                        {notice}
                    </Text>
                )}
                {errorText && (
                    <Text size="1" color="red">
                        {errorText}
                    </Text>
                )}
            </AppForm>
        </AppDialog>
    );
}
