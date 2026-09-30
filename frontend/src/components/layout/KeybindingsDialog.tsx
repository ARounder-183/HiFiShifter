import React, { useCallback, useEffect, useRef, useState } from "react";
import { Flex, Button, ScrollArea, Separator } from "@radix-ui/themes";
import { Cross2Icon } from "@radix-ui/react-icons";
import { useI18n } from "../../i18n/I18nProvider";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { IS_MAC } from "../../utils/platform";
import {
    selectMergedKeybindings,
    setKeybinding,
    resetKeybinding,
    resetAllKeybindings,
    formatKeybinding,
    findConflicts,
    createModifierOnlyBinding,
    // isNoneBinding, // 已删除未使用变量
} from "../../features/keybindings/keybindingsSlice";
import {
    DEFAULT_KEYBINDINGS,
    ACTION_META,
    ALL_ACTION_IDS,
    ACTION_GROUP_ORDER,
    GROUP_LABEL_KEYS,
} from "../../features/keybindings/defaultKeybindings";
import type { ActionId, ActionMeta, Keybinding } from "../../features/keybindings/types";
import { canonicalKeyFromEvent } from "../../features/keybindings/keybindingMatch";
import { useShortcutSuppression } from "../../ui/shortcutScope";
import { AppDialog } from "../../ui/Dialog";
import { AppSelect, AppStatusChip, type AppStatusTone } from "../../ui";
import type { MessageKey } from "../../i18n/messages";
import {
    KEYBINDING_PRESET_SELECTION_IDS,
    KEYBINDING_PRESETS,
    isKeybindingPresetId,
    type KeybindingPresetSelectionId,
} from "../../features/keybindings/keybindingPresets";

/**
 * 预设名 → 词典键的**显式**映射。
 *
 * 【为什么不用 `kb_preset_${presetId}` 模板拼接】预设 id 是运行期标识符
 * （`spaceReturnPlayhead` / `vegasPro` / `vocalShifter`…），它们会被写进用户设置，
 * 不能为了迎合词典的 snake_case 约定而改名 —— 改了会丢用户的预设选择。反过来，
 * 用模板拼键名意味着**词典里少一个键也不会报错**：`tf` 拿到一个不存在的键会原样
 * 返回键名，界面上就出现 `kb_preset_vegasPro` 这样的英文键。
 *
 * 写成显式表后，两个方向都被类型检查守住：表必须覆盖全部 id（`Record<...>`），
 * 值必须是真实存在的词典键（`MessageKey`）。
 */
const PRESET_LABEL_KEY: Record<KeybindingPresetSelectionId, MessageKey> = {
    custom: "kb_preset_custom",
    default: "kb_preset_default",
    spaceReturnPlayhead: "kb_preset_space_return_playhead",
    touchpad: "kb_preset_touchpad",
    reaper: "kb_preset_reaper",
    vegasPro: "kb_preset_vegas_pro",
    vocalShifter: "kb_preset_vocal_shifter",
};

/** "无" 绑定常量 */
const NONE_BINDING: Keybinding = { key: "__none__" };

type ModifierToken = "control" | "shift" | "alt";

/** 修饰键手势徽章的 i18n key 与色调（四档互不相同，仍可区分手势类型） */
const GESTURE_BADGES: Record<
    NonNullable<ActionMeta["modifierOperationType"]>,
    { labelKey: string; tone: AppStatusTone }
> = {
    drag: { labelKey: "kb_gesture_drag", tone: "accent" },
    click: { labelKey: "kb_gesture_click", tone: "warning" },
    wheel: { labelKey: "kb_gesture_wheel", tone: "success" },
    hold: { labelKey: "kb_gesture_hold", tone: "neutral" },
};

function isPhysicalModifierKey(key: string): boolean {
    const lower = key.toLowerCase();
    return lower === "control" || lower === "shift" || lower === "alt" || lower === "meta";
}

function modifierTokenFromKey(key: string, isMac: boolean): ModifierToken | null {
    const lower = key.toLowerCase();
    if (lower === "shift") return "shift";
    if (lower === "alt") return "alt";
    if (lower === "control") return isMac ? null : "control";
    if (lower === "meta") return isMac ? "control" : null;
    return null;
}

function modifierTokensFromEvent(e: KeyboardEvent, isMac: boolean): Set<ModifierToken> {
    const tokens = new Set<ModifierToken>();
    if (isMac ? e.metaKey : e.ctrlKey) tokens.add("control");
    if (e.shiftKey) tokens.add("shift");
    if (e.altKey) tokens.add("alt");
    return tokens;
}

interface KeybindingsDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

/**
 * 快捷键设置面板
 * 打开时阻塞所有下层交互（通过 Dialog overlay）
 */
export const KeybindingsDialog: React.FC<KeybindingsDialogProps> = ({ open, onOpenChange }) => {
    const dispatch = useAppDispatch();
    const { tf } = useI18n();
    const keybindings = useAppSelector(selectMergedKeybindings);
    const overrides = useAppSelector((s) => s.keybindings.overrides);

    // 打开时抑制全局快捷键与工程编辑。走统一作用域（src/ui/shortcutScope.ts），
    // 取代此前各自的 `data-keybindings-dialog-open` body 属性。
    useShortcutSuppression(open);

    // 当前处于"录入模式"的 actionId
    const [recordingId, setRecordingId] = useState<ActionId | null>(null);
    // 冲突提示
    const [conflict, setConflict] = useState<{
        actionId: ActionId;
        newBinding: Keybinding;
        conflictWith: ActionId[];
    } | null>(null);
    const [selectedPreset, setSelectedPreset] = useState<KeybindingPresetSelectionId>("custom");

    const recordingRef = useRef(recordingId);
    useEffect(() => {
        recordingRef.current = recordingId;
    }, [recordingId]);

    // 录入模式的键盘监听
    useEffect(() => {
        if (!recordingId) return;

        const currentIsModifierOnly = Boolean(DEFAULT_KEYBINDINGS[recordingId]?.modifierOnly);
        const isMac = IS_MAC;
        const pressedModifierTokens = new Set<ModifierToken>();

        function applyModifierTokens(tokens: Set<ModifierToken>) {
            const currentId = recordingRef.current;
            if (!currentId || tokens.size === 0) return;

            const newBinding: Keybinding = createModifierOnlyBinding({
                ctrl: tokens.has("control"),
                shift: tokens.has("shift"),
                alt: tokens.has("alt"),
            });

            const conflicts = findConflicts(overrides, currentId, newBinding);
            if (conflicts.length > 0) {
                setConflict({
                    actionId: currentId,
                    newBinding,
                    conflictWith: conflicts,
                });
                return;
            }

            dispatch(setKeybinding({ actionId: currentId, binding: newBinding }));
            setSelectedPreset("custom");
            recordingRef.current = null;
            setRecordingId(null);
        }

        function onKeyDown(e: KeyboardEvent) {
            e.preventDefault();
            e.stopPropagation();

            // Escape 取消录入
            if (e.key === "Escape") {
                recordingRef.current = null;
                setRecordingId(null);
                setConflict(null);
                return;
            }

            if (currentIsModifierOnly) {
                const tokenFromKey = modifierTokenFromKey(e.key, isMac);
                if (!tokenFromKey) return;

                const snapshotTokens = modifierTokensFromEvent(e, isMac);
                if (snapshotTokens.size === 0) snapshotTokens.add(tokenFromKey);

                pressedModifierTokens.clear();
                snapshotTokens.forEach((token) => pressedModifierTokens.add(token));
                return;
            }

            // 普通模式：忽略单独按下修饰键
            if (isPhysicalModifierKey(e.key)) return;

            // 录入规范化：Shift 会改写标点字符（US 布局 Shift+= 产出 "+"），
            // 按 e.code 还原为物理键位的基础字符（"="），与默认绑定一致，
            // 才能参与冲突检测与运行时匹配（见 keybindingMatch.ts）。
            const key = canonicalKeyFromEvent(e);
            const ctrl = isMac ? e.metaKey : e.ctrlKey;

            const newBinding: Keybinding = {
                key,
                ...(ctrl ? { ctrl: true } : {}),
                ...(e.shiftKey ? { shift: true } : {}),
                ...(e.altKey ? { alt: true } : {}),
            };

            const currentId = recordingRef.current;
            if (!currentId) return;

            // 检测冲突
            const conflicts = findConflicts(overrides, currentId, newBinding);
            if (conflicts.length > 0) {
                setConflict({
                    actionId: currentId,
                    newBinding,
                    conflictWith: conflicts,
                });
                return;
            }

            dispatch(setKeybinding({ actionId: currentId, binding: newBinding }));
            setSelectedPreset("custom");
            recordingRef.current = null;
            setRecordingId(null);
        }

        function onKeyUp(e: KeyboardEvent) {
            if (!currentIsModifierOnly) return;
            e.preventDefault();
            e.stopPropagation();

            const tokenFromKey = modifierTokenFromKey(e.key, isMac);
            if (!tokenFromKey) return;

            const snapshotTokens = modifierTokensFromEvent(e, isMac);
            snapshotTokens.add(tokenFromKey);
            if (pressedModifierTokens.size > 0) {
                pressedModifierTokens.forEach((token) => snapshotTokens.add(token));
            }

            applyModifierTokens(snapshotTokens);
            pressedModifierTokens.clear();
        }

        window.addEventListener("keydown", onKeyDown, true);
        window.addEventListener("keyup", onKeyUp, true);
        return () => {
            window.removeEventListener("keydown", onKeyDown, true);
            window.removeEventListener("keyup", onKeyUp, true);
        };
    }, [recordingId, dispatch, overrides]);

    const handleConfirmConflict = useCallback(() => {
        if (!conflict) return;
        // 先清除冲突的绑定
        for (const cId of conflict.conflictWith) {
            dispatch(resetKeybinding(cId));
        }
        // 再设置新绑定
        dispatch(
            setKeybinding({
                actionId: conflict.actionId,
                binding: conflict.newBinding,
            }),
        );
        setSelectedPreset("custom");
        setConflict(null);
        setRecordingId(null);
    }, [conflict, dispatch]);

    const handleCancelConflict = useCallback(() => {
        setConflict(null);
    }, []);

    const handleApplyPreset = useCallback(
        (value: string) => {
            if (value === "custom") {
                setSelectedPreset("custom");
                return;
            }

            if (value === "default") {
                dispatch(resetAllKeybindings());
                setSelectedPreset("default");
                setConflict(null);
                setRecordingId(null);
                return;
            }

            if (!isKeybindingPresetId(value)) {
                setSelectedPreset("custom");
                return;
            }

            const preset = KEYBINDING_PRESETS[value];
            for (const [actionId, binding] of Object.entries(preset)) {
                dispatch(
                    setKeybinding({
                        actionId: actionId as ActionId,
                        binding,
                    }),
                );
            }

            setSelectedPreset(value);
            setConflict(null);
            setRecordingId(null);
        },
        [dispatch],
    );

    // 按分组组织操作
    const groups = React.useMemo(() => {
        return ACTION_GROUP_ORDER.map((group) => ({
            group,
            actions: ALL_ACTION_IDS.filter((id) => ACTION_META[id].group === group),
        }));
    }, []);

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("kb_dialog_title")}
            description={tf("kb_dialog_desc")}
            size="lg"
            actions={[
                {
                    id: "close",
                    label: tf("close"),
                    // 迁移时丢失的 ✕ 图标（原按钮是 <Cross2Icon /> + 文案）
                    icon: <Cross2Icon />,
                    onClick: () => onOpenChange(false),
                },
            ]}
            /*
             * 正在录入时不允许 Esc / 外部点击关闭：按下的那个键正是要录入的
             * 内容，关掉对话框等于把用户的操作丢掉。
             */
            beforeClose={() => recordingId === null}
        >
            <span className="hs-type-muted" style={{ marginBottom: 4, display: "block" }}>
                {tf("kb_dialog_hint_click")}
            </span>

            <Flex align="center" gap="2" mt="2" mb="2">
                <span className="hs-type-muted" style={{ whiteSpace: "nowrap" }}>
                    {tf("kb_preset_label")}
                </span>
                <AppSelect
                    fullWidth={false}
                    value={selectedPreset}
                    onValueChange={handleApplyPreset}
                    options={KEYBINDING_PRESET_SELECTION_IDS.map((presetId) => ({
                        value: presetId,
                        label: tf(PRESET_LABEL_KEY[presetId]),
                    }))}
                />
            </Flex>

            <ScrollArea style={{ maxHeight: "56vh", marginTop: 8 }} scrollbars="vertical">
                <Flex direction="column" gap="3" py="3">
                    {groups.map(({ group, actions }) => (
                        <Flex direction="column" gap="1" key={group}>
                            <span
                                className="hs-type-label font-semibold"
                                style={{
                                    textTransform: "uppercase",
                                    letterSpacing: "0.05em",
                                    padding: "4px 0",
                                }}
                            >
                                {tf(GROUP_LABEL_KEYS[group])}
                            </span>
                            <Separator size="4" />
                            {actions.map((actionId) => {
                                const meta = ACTION_META[actionId];
                                const currentKb = keybindings[actionId];
                                const defaultKb = DEFAULT_KEYBINDINGS[actionId];
                                const isDefault =
                                    currentKb.key === defaultKb.key &&
                                    Boolean(currentKb.ctrl) === Boolean(defaultKb.ctrl) &&
                                    Boolean(currentKb.shift) === Boolean(defaultKb.shift) &&
                                    Boolean(currentKb.alt) === Boolean(defaultKb.alt) &&
                                    Boolean(currentKb.modifierOnly) ===
                                        Boolean(defaultKb.modifierOnly);
                                const isRecording = recordingId === actionId;

                                return (
                                    <Flex
                                        key={actionId}
                                        align="center"
                                        justify="between"
                                        px="2"
                                        py="1"
                                        style={{
                                            borderRadius: "var(--qt-radius-sm)",
                                            background: isRecording ? "var(--accent-3)" : undefined,
                                            minHeight: 36,
                                        }}
                                    >
                                        <Flex align="center" gap="2" minWidth="0">
                                            {/* 修饰键手势徽章：区分拖拽 / 点击 / 滚轮 / 按住 */}
                                            {meta.modifierOperationType && (
                                                <AppStatusChip
                                                    tone={
                                                        GESTURE_BADGES[meta.modifierOperationType]
                                                            .tone
                                                    }
                                                >
                                                    {tf(
                                                        GESTURE_BADGES[meta.modifierOperationType]
                                                            .labelKey,
                                                    )}
                                                </AppStatusChip>
                                            )}
                                            <span className="hs-type-body" style={{ minWidth: 0 }}>
                                                {tf(meta.labelKey)}
                                            </span>
                                        </Flex>
                                        <Flex align="center" gap="2">
                                            {/* 快捷键显示 / 录入按钮 */}
                                            <Button
                                                variant={isRecording ? "solid" : "soft"}
                                                color={
                                                    isRecording
                                                        ? "blue"
                                                        : !isDefault
                                                          ? "green"
                                                          : "gray"
                                                }
                                                size="1"
                                                style={{
                                                    minWidth: 120,
                                                    fontFamily: "monospace",
                                                }}
                                                onClick={() => {
                                                    if (isRecording) {
                                                        // 录入中左键点击 → 设为"无"
                                                        dispatch(
                                                            setKeybinding({
                                                                actionId,
                                                                binding: NONE_BINDING,
                                                            }),
                                                        );
                                                        setSelectedPreset("custom");
                                                        setRecordingId(null);
                                                        setConflict(null);
                                                    } else {
                                                        setConflict(null);
                                                        setRecordingId(actionId);
                                                    }
                                                }}
                                                onContextMenu={(e) => {
                                                    e.preventDefault();
                                                    // 右键点击 → 直接重置为默认
                                                    dispatch(resetKeybinding(actionId));
                                                    setSelectedPreset("custom");
                                                    setRecordingId(null);
                                                    setConflict(null);
                                                }}
                                            >
                                                {isRecording
                                                    ? tf(
                                                          defaultKb.modifierOnly
                                                              ? "kb_press_modifier"
                                                              : "kb_press_key",
                                                      )
                                                    : formatKeybinding(currentKb, tf("kb_none"))}
                                            </Button>
                                        </Flex>
                                    </Flex>
                                );
                            })}
                        </Flex>
                    ))}
                </Flex>
            </ScrollArea>

            {/* 冲突提示：按所属场景分组展示冲突项 */}
            {conflict && (
                <Flex
                    align="center"
                    gap="2"
                    py="2"
                    px="3"
                    style={{
                        background: "var(--red-3)",
                        borderRadius: "var(--qt-radius-md)",
                        marginTop: 8,
                    }}
                >
                    <Flex direction="column" gap="1" style={{ flex: 1, minWidth: 0 }}>
                        <span className="hs-type-body" style={{ color: "var(--qt-danger-text)" }}>
                            {tf("kb_conflict_msg")}
                        </span>
                        {Array.from(
                            conflict.conflictWith.reduce((map, id) => {
                                const group = ACTION_META[id].group;
                                const ids = map.get(group) ?? [];
                                ids.push(id);
                                map.set(group, ids);
                                return map;
                            }, new Map<ActionMeta["group"], ActionId[]>()),
                        ).map(([group, ids]) => (
                            <span
                                key={group}
                                className="hs-type-body"
                                style={{ color: "var(--qt-danger-text)" }}
                            >
                                {tf(GROUP_LABEL_KEYS[group])}：
                                <strong>
                                    {ids.map((id) => tf(ACTION_META[id].labelKey)).join("、")}
                                </strong>
                            </span>
                        ))}
                    </Flex>
                    <Button size="1" color="red" variant="soft" onClick={handleConfirmConflict}>
                        {tf("kb_conflict_override")}
                    </Button>
                    <Button size="1" color="gray" variant="soft" onClick={handleCancelConflict}>
                        {tf("kb_conflict_cancel")}
                    </Button>
                </Flex>
            )}
        </AppDialog>
    );
};
