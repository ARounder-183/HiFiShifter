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
 * - 下拉框滚轮切换选项（`applySelectWheelChange`）；
 * - 数字输入框滚轮步进，按住「精细调整」修饰键时用更小步长
 *   （`isModifierActive` + `modifier.paramFineAdjust`），并由
 *   `useWheelScrollGuard` 阻止滚轮冒泡去滚动对话框。
 */

import { useEffect, useRef, useState } from "react";
import { Button, Dialog, Flex, Select, Separator, Text, TextField } from "@radix-ui/themes";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { useI18n } from "../../i18n/I18nProvider";
import {
    CHANNEL_TOLERANCE_PRESETS,
    normalizeChannelImportPolicy,
    type ChannelImportMode,
    type ChannelImportPolicy,
} from "../../services/api/settings";
import { persistUiSettings } from "../../features/session/thunks/runtimeThunks";
import { setChannelImportPolicy } from "../../features/session/sessionSlice";
import { isModifierActive, selectKeybinding } from "../../features/keybindings/keybindingsSlice";
import { applySelectWheelChange } from "../../utils/selectWheel";
import { useNonPassiveWheel } from "../../utils/useNonPassiveWheel";

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

/**
 * 容差下拉项的取值字符串。
 *
 * 必须与 `Select.Root` 的 `value` 用**同一套编码**：`String(1e-3)` 得到
 * `"0.001"`，若选项用 `"1e-3"` 当 value，两者永不相等 —— Radix 找不到匹配项
 * 时 Trigger 会显示空白（曾经的实际缺陷）。这里统一走 `String(number)`。
 */
function toleranceValueOf(tolerance: number): string {
    return String(tolerance);
}

export function ChannelImportDialog({ open, onOpenChange }: ChannelImportDialogProps) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const saved = useAppSelector((state) => state.session.channelImportPolicy);
    const paramFineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );

    const [draft, setDraft] = useState<ChannelImportPolicy>(saved);
    const [saving, setSaving] = useState(false);
    const [notice, setNotice] = useState("");
    const [errorText, setErrorText] = useState("");

    // 滚轮守卫：数字输入框与下拉触发按钮自行消费滚轮步进，必须阻止该事件再去
    // 滚动对话框。React 的合成 `onWheel` 是 passive 监听，里面的 preventDefault
    // 是空操作，所以要在这里挂**原生非被动**监听。
    //
    // 用回调 ref（`useNonPassiveWheel`）而不是 `useWheelScrollGuard`：后者在
    // `[selector]` 依赖的 effect 里读 `ref.current`，而 Dialog 内容是条件挂载的
    // —— 首次打开时元素尚不存在，守卫永远挂不上。
    const attachWheelGuard = useNonPassiveWheel<HTMLDivElement>((event) => {
        const target = event.target;
        if (!(target instanceof Element)) return;
        // 只拦截会自行处理滚轮的控件：数字输入框与 Radix 下拉触发按钮。
        if (target.closest("input") || target.closest('[role="combobox"]')) {
            event.preventDefault();
        }
    });

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

    /** 数字输入框滚轮步进：普通步长 / 精细步长。 */
    function stepNumber(
        event: React.WheelEvent<HTMLInputElement>,
        current: number,
        coarse: number,
        fine: number,
        min: number,
        max: number,
        apply: (next: number) => void,
    ) {
        if (!Number.isFinite(event.deltaY) || event.deltaY === 0) return;
        event.preventDefault();
        const step = isModifierActive(paramFineAdjustKb, event.nativeEvent) ? fine : coarse;
        const dir = event.deltaY < 0 ? 1 : -1;
        const next = Math.min(max, Math.max(min, current + dir * step));
        if (next !== current) apply(next);
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
    const toleranceValues = CHANNEL_TOLERANCE_PRESETS.map(toleranceValueOf);
    // 当前值可能不在档位里（手改过配置文件）：退化为"不选中"，而不是显示空白。
    const toleranceValue = toleranceValues.includes(toleranceValueOf(draft.tolerance))
        ? toleranceValueOf(draft.tolerance)
        : undefined;

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content
                style={{ maxWidth: 520 }}
                onKeyDown={(event) => event.stopPropagation()}
            >
                <Dialog.Title>{tAny("clip_channel_import_dialog_title")}</Dialog.Title>
                <Dialog.Description>
                    {tAny("clip_channel_import_dialog_desc")}
                </Dialog.Description>

                <Flex direction="column" gap="3" mt="3" ref={attachWheelGuard}>
                    <Separator size="4" />

                    {/* ── 总策略 ─────────────────────────────────────────── */}
                    <Flex align="center" justify="between" gap="3">
                        <Text size="2">{tAny("clip_channel_import_mode")}</Text>
                        <Select.Root
                            value={draft.mode}
                            onValueChange={(value) => patch({ mode: value as ChannelImportMode })}
                        >
                            <Select.Trigger
                                style={{ minWidth: 220 }}
                                onWheel={(event) =>
                                    applySelectWheelChange({
                                        event,
                                        currentValue: draft.mode,
                                        options: MODE_OPTIONS.map((option) => option.value),
                                        onChange: (next) => patch({ mode: next }),
                                    })
                                }
                            />
                            <Select.Content>
                                {MODE_OPTIONS.map((option) => (
                                    <Select.Item key={option.value} value={option.value}>
                                        {tAny(option.labelKey)}
                                    </Select.Item>
                                ))}
                            </Select.Content>
                        </Select.Root>
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
                                <Select.Trigger
                                    style={{ minWidth: 220 }}
                                    onWheel={(event) =>
                                        applySelectWheelChange({
                                            event,
                                            currentValue: String(draft.monoTargetMode),
                                            options: TARGET_MODE_OPTIONS.map(
                                                (option) => option.value,
                                            ),
                                            onChange: (next) =>
                                                patch({ monoTargetMode: Number(next) }),
                                        })
                                    }
                                />
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
                                <Text size="2">{tAny("clip_channel_import_tolerance")}</Text>
                                <Select.Root
                                    value={toleranceValue}
                                    onValueChange={(value) => patch({ tolerance: Number(value) })}
                                >
                                    <Select.Trigger
                                        placeholder={tAny("clip_channel_import_tolerance_custom")}
                                        style={{ minWidth: 220 }}
                                        onWheel={(event) => {
                                            if (toleranceValue === undefined) return;
                                            applySelectWheelChange({
                                                event,
                                                currentValue: toleranceValue,
                                                options: toleranceValues,
                                                onChange: (next) =>
                                                    patch({ tolerance: Number(next) }),
                                            });
                                        }}
                                    />
                                    <Select.Content>
                                        {CHANNEL_TOLERANCE_PRESETS.map((preset) => (
                                            <Select.Item
                                                key={toleranceValueOf(preset)}
                                                value={toleranceValueOf(preset)}
                                            >
                                                {preset === 0
                                                    ? tAny("clip_channel_import_tolerance_exact")
                                                    : toleranceValueOf(preset)}
                                            </Select.Item>
                                        ))}
                                    </Select.Content>
                                </Select.Root>
                            </Flex>
                            <Text size="1" color="gray">
                                {tAny("clip_channel_import_tolerance_hint")}
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
                                    onWheel={(event) =>
                                        stepNumber(
                                            event,
                                            draft.windowSec,
                                            0.05,
                                            0.01,
                                            0.05,
                                            5,
                                            (next) => patch({ windowSec: next }),
                                        )
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
                                    onWheel={(event) =>
                                        stepNumber(
                                            event,
                                            draft.windowCount,
                                            1,
                                            1,
                                            0,
                                            256,
                                            (next) => patch({ windowCount: next }),
                                        )
                                    }
                                />
                            </Flex>
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
