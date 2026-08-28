import React, { useCallback, useEffect, useRef, useState } from "react";
import { Dialog, Flex, Text, Button, ScrollArea, Separator, Select, Badge } from "@radix-ui/themes";
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
import {
    KEYBINDING_PRESET_SELECTION_IDS,
    KEYBINDING_PRESETS,
    isKeybindingPresetId,
    type KeybindingPresetSelectionId,
} from "../../features/keybindings/keybindingPresets";

/** "无" 绑定常量 */
const NONE_BINDING: Keybinding = { key: "__none__" };

type ModifierToken = "control" | "shift" | "alt";

/** 修饰键手势徽章的 i18n key 与配色 */
const GESTURE_BADGES: Record<
    NonNullable<ActionMeta["modifierOperationType"]>,
    { labelKey: string; color: "blue" | "orange" | "green" | "purple" }
> = {
    drag: { labelKey: "kb_gesture_drag", color: "blue" },
    click: { labelKey: "kb_gesture_click", color: "orange" },
    wheel: { labelKey: "kb_gesture_wheel", color: "green" },
    hold: { labelKey: "kb_gesture_hold", color: "purple" },
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
    const { t } = useI18n();
    const keybindings = useAppSelector(selectMergedKeybindings);
    const overrides = useAppSelector((s) => s.keybindings.overrides);

    // 打开时在 body 上标记，阻塞全局快捷键和工程编辑
    useEffect(() => {
        if (open) {
            document.body.setAttribute("data-keybindings-dialog-open", "true");
        } else {
            document.body.removeAttribute("data-keybindings-dialog-open");
        }
        return () => {
            document.body.removeAttribute("data-keybindings-dialog-open");
        };
    }, [open]);

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

            const key = e.key === " " ? "space" : e.key.toLowerCase();
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

    const tAny = t as (key: string) => string;

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            {/* Overlay 阻塞下层所有交互 */}
            {open && (
                <div
                    style={{
                        position: "fixed",
                        inset: 0,
                        zIndex: 9998,
                        background: "rgba(0,0,0,0.4)",
                    }}
                    onPointerDown={(e) => e.stopPropagation()}
                    onKeyDown={(e) => {
                        if (!recordingId) return;
                        e.stopPropagation();
                    }}
                />
            )}
            <Dialog.Content
                style={{
                    width: "min(560px, 92vw)",
                    maxWidth: 560,
                    maxHeight: "80vh",
                    zIndex: 9999,
                    overflow: "hidden",
                    display: "flex",
                    flexDirection: "column",
                    minHeight: 0,
                }}
                onPointerDownOutside={(e) => {
                    // 如果正在录入，阻止点击外部关闭
                    if (recordingId) e.preventDefault();
                }}
                onKeyDown={(e) => {
                    // 阻止按键穿透到工程
                    e.stopPropagation();
                }}
            >
                <Dialog.Title>{tAny("kb_dialog_title")}</Dialog.Title>
                <Dialog.Description size="2" color="gray">
                    {tAny("kb_dialog_desc")}
                </Dialog.Description>
                <Text size="1" color="gray" style={{ marginTop: 4, display: "block" }}>
                    {tAny("kb_dialog_hint_click")}
                </Text>

                <Flex align="center" gap="2" mt="2" mb="2">
                    <Text size="2" color="gray" style={{ whiteSpace: "nowrap" }}>
                        {tAny("kb_preset_label")}
                    </Text>
                    <Select.Root value={selectedPreset} onValueChange={handleApplyPreset}>
                        <Select.Trigger style={{ minWidth: 220 }} />
                        <Select.Content>
                            {KEYBINDING_PRESET_SELECTION_IDS.map((presetId) => (
                                <Select.Item key={presetId} value={presetId}>
                                    {tAny(`kb_preset_${presetId}`)}
                                </Select.Item>
                            ))}
                        </Select.Content>
                    </Select.Root>
                </Flex>

                <ScrollArea style={{ flex: 1, minHeight: 0, marginTop: 8 }} scrollbars="vertical">
                    <Flex direction="column" gap="3" py="3">
                        {groups.map(({ group, actions }) => (
                            <Flex direction="column" gap="1" key={group}>
                                <Text
                                    size="1"
                                    weight="bold"
                                    color="gray"
                                    style={{
                                        textTransform: "uppercase",
                                        letterSpacing: "0.05em",
                                        padding: "4px 0",
                                    }}
                                >
                                    {tAny(GROUP_LABEL_KEYS[group])}
                                </Text>
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
                                                borderRadius: 4,
                                                background: isRecording
                                                    ? "var(--accent-3)"
                                                    : undefined,
                                                minHeight: 36,
                                            }}
                                        >
                                            <Flex align="center" gap="2" minWidth="0">
                                                {/* 修饰键手势徽章：区分拖拽 / 点击 / 滚轮 / 按住 */}
                                                {meta.modifierOperationType && (
                                                    <Badge
                                                        size="1"
                                                        variant="soft"
                                                        color={
                                                            GESTURE_BADGES[
                                                                meta.modifierOperationType
                                                            ].color
                                                        }
                                                    >
                                                        {tAny(
                                                            GESTURE_BADGES[
                                                                meta.modifierOperationType
                                                            ].labelKey,
                                                        )}
                                                    </Badge>
                                                )}
                                                <Text size="2" style={{ minWidth: 0 }}>
                                                    {tAny(meta.labelKey)}
                                                </Text>
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
                                                        ? tAny(
                                                              defaultKb.modifierOnly
                                                                  ? "kb_press_modifier"
                                                                  : "kb_press_key",
                                                          )
                                                        : formatKeybinding(
                                                              currentKb,
                                                              tAny("kb_none"),
                                                          )}
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
                            borderRadius: 6,
                            marginTop: 8,
                        }}
                    >
                        <Flex direction="column" gap="1" style={{ flex: 1, minWidth: 0 }}>
                            <Text size="2" color="red">
                                {tAny("kb_conflict_msg")}
                            </Text>
                            {Array.from(
                                conflict.conflictWith.reduce((map, id) => {
                                    const group = ACTION_META[id].group;
                                    const ids = map.get(group) ?? [];
                                    ids.push(id);
                                    map.set(group, ids);
                                    return map;
                                }, new Map<ActionMeta["group"], ActionId[]>()),
                            ).map(([group, ids]) => (
                                <Text key={group} size="2" color="red">
                                    {tAny(GROUP_LABEL_KEYS[group])}：
                                    <strong>
                                        {ids
                                            .map((id) => tAny(ACTION_META[id].labelKey))
                                            .join("、")}
                                    </strong>
                                </Text>
                            ))}
                        </Flex>
                        <Button size="1" color="red" variant="soft" onClick={handleConfirmConflict}>
                            {tAny("kb_conflict_override")}
                        </Button>
                        <Button size="1" color="gray" variant="soft" onClick={handleCancelConflict}>
                            {tAny("kb_conflict_cancel")}
                        </Button>
                    </Flex>
                )}

                {/* 底部按钮 */}
                <Flex justify="end" align="center" pt="3">
                    <Dialog.Close>
                        <Button variant="soft" color="gray" size="2">
                            <Cross2Icon />
                            {tAny("kb_close")}
                        </Button>
                    </Dialog.Close>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
};
