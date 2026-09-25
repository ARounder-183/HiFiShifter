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
 * 设计要点：
 * - 设置以草稿形式编辑，点「保存」才落盘（与渲染缓存/自动备份对话框一致）；
 * - 草稿只在"打开"这一时机初始化一次，避免保存后 effect 重跑清掉提示；
 * - 采样参数收在"高级"区，默认值对绝大多数素材已足够。
 */

import { useEffect, useRef, useState } from "react";
import {
    Button,
    Checkbox,
    Dialog,
    Flex,
    Select,
    Separator,
    Text,
    TextField,
} from "@radix-ui/themes";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { useI18n } from "../../i18n/I18nProvider";
import {
    normalizeChannelImportPolicy,
    type ChannelImportMode,
    type ChannelImportPolicy,
} from "../../services/api/settings";
import { persistUiSettings } from "../../features/session/thunks/runtimeThunks";
import { setChannelImportPolicy } from "../../features/session/sessionSlice";

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

/** 容差预设（越小越严格）；0 = 逐样本完全相等。 */
const TOLERANCE_PRESETS = ["0", "1e-6", "1e-5", "1e-4", "1e-3"];

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
    const toleranceValue = TOLERANCE_PRESETS.includes(String(draft.tolerance))
        ? String(draft.tolerance)
        : "custom";

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content
                style={{ maxWidth: 640 }}
                onKeyDown={(event) => event.stopPropagation()}
            >
                <Dialog.Title>{tAny("clip_channel_import_dialog_title")}</Dialog.Title>
                <Dialog.Description>
                    {tAny("clip_channel_import_dialog_desc")}
                </Dialog.Description>

                <Flex direction="column" gap="3" mt="3">
                    <Separator size="4" />

                    {/* ── 总策略 ─────────────────────────────────────────── */}
                    <Text size="2" weight="medium">
                        {tAny("clip_channel_import_mode")}
                    </Text>
                    <Flex direction="column" gap="2">
                        {MODE_OPTIONS.map((option) => (
                            <Text
                                key={option.value}
                                size="2"
                                as="label"
                                style={{ display: "flex", alignItems: "center", gap: 8 }}
                            >
                                <input
                                    type="radio"
                                    name="channel-import-mode"
                                    checked={draft.mode === option.value}
                                    onChange={() => patch({ mode: option.value })}
                                />
                                {tAny(option.labelKey)}
                            </Text>
                        ))}
                    </Flex>
                    <Text size="1" color="gray">
                        {isOff
                            ? tAny("clip_channel_import_mode_off_hint")
                            : tAny("clip_channel_import_mode_hint")}
                    </Text>

                    {/* ── 目标模式 ───────────────────────────────────────── */}
                    {!isOff && (
                        <Flex align="center" justify="between" gap="3">
                            <Text size="2">{tAny("clip_channel_import_target_mode")}</Text>
                            <Select.Root
                                value={String(draft.monoTargetMode)}
                                onValueChange={(value) =>
                                    patch({ monoTargetMode: Number(value) })
                                }
                            >
                                <Select.Trigger style={{ minWidth: 180 }} />
                                <Select.Content>
                                    {TARGET_MODE_OPTIONS.map((option) => (
                                        <Select.Item key={option.value} value={option.value}>
                                            {tAny(option.labelKey)}
                                        </Select.Item>
                                    ))}
                                </Select.Content>
                            </Select.Root>
                        </Flex>
                    )}

                    {/* ── 采样参数（仅智能模式）──────────────────────────── */}
                    {isSmart && (
                        <>
                            <Separator size="4" />
                            <Text size="2" weight="medium">
                                {tAny("clip_channel_import_advanced")}
                            </Text>

                            <Flex align="center" justify="between" gap="3">
                                <Text size="2">{tAny("clip_channel_import_window_sec")}</Text>
                                <TextField.Root
                                    type="number"
                                    min={0.05}
                                    max={5}
                                    step={0.05}
                                    style={{ width: 120 }}
                                    value={String(draft.windowSec)}
                                    onChange={(event) =>
                                        patch({ windowSec: Number(event.target.value) })
                                    }
                                />
                            </Flex>

                            <Flex align="center" justify="between" gap="3">
                                <Text size="2">{tAny("clip_channel_import_window_count")}</Text>
                                <TextField.Root
                                    type="number"
                                    min={0}
                                    max={256}
                                    step={1}
                                    style={{ width: 120 }}
                                    value={String(draft.windowCount)}
                                    onChange={(event) =>
                                        patch({ windowCount: Number(event.target.value) })
                                    }
                                />
                            </Flex>
                            <Text size="1" color="gray">
                                {tAny("clip_channel_import_window_hint")}
                            </Text>

                            <Flex align="center" justify="between" gap="3">
                                <Text size="2">{tAny("clip_channel_import_tolerance")}</Text>
                                <Select.Root
                                    value={toleranceValue}
                                    onValueChange={(value) =>
                                        value !== "custom" &&
                                        patch({ tolerance: Number(value) })
                                    }
                                >
                                    <Select.Trigger style={{ minWidth: 180 }} />
                                    <Select.Content>
                                        {TOLERANCE_PRESETS.map((preset) => (
                                            <Select.Item key={preset} value={preset}>
                                                {preset === "0"
                                                    ? tAny("clip_channel_import_tolerance_exact")
                                                    : preset}
                                            </Select.Item>
                                        ))}
                                    </Select.Content>
                                </Select.Root>
                            </Flex>
                            <Text size="1" color="gray">
                                {tAny("clip_channel_import_tolerance_hint")}
                            </Text>
                        </>
                    )}

                    {/* ── 作用范围 ───────────────────────────────────────── */}
                    <Separator size="4" />
                    <Text as="label" size="2" style={{ display: "flex", gap: 8 }}>
                        <Checkbox
                            checked={draft.applyToLegacyTakes}
                            onCheckedChange={(checked) =>
                                patch({ applyToLegacyTakes: checked === true })
                            }
                        />
                        {tAny("clip_channel_import_apply_legacy")}
                    </Text>
                    <Text size="1" color="gray">
                        {tAny("clip_channel_import_apply_legacy_hint")}
                    </Text>

                    <Text as="label" size="2" style={{ display: "flex", gap: 8 }}>
                        <Checkbox
                            checked={draft.applyToAllTakes}
                            onCheckedChange={(checked) =>
                                patch({ applyToAllTakes: checked === true })
                            }
                        />
                        {tAny("clip_bulk_apply_all_takes")}
                    </Text>
                    <Text size="1" color="gray">
                        {tAny("clip_bulk_apply_all_takes_hint")}
                    </Text>

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

                    <Flex justify="end" gap="2" mt="4">
                        <Button variant="soft" color="gray" onClick={() => onOpenChange(false)}>
                            {tAny("render_cache_close")}
                        </Button>
                        <Button disabled={saving} onClick={() => void handleSave()}>
                            {tAny("render_cache_save_settings")}
                        </Button>
                    </Flex>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
}
