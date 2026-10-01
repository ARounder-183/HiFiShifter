/**
 * 快捷键设置面板。
 *
 * 打开时阻塞所有下层交互（通过 `AppDialog` overlay）。
 *
 * 【布局】搜索框横跨全窗口；下面分左右两栏 —— 左栏是常驻的分类导航（兼作搜索
 * 的范围过滤器），右栏是结果。有查询时右栏按相关度打平排列（每条附分组名作路标），
 * 无查询时按分组排列且组标题粘性吸顶。
 *
 * 数据本身（默认值、录入、冲突、预设）仍在 `features/keybindings` 里；本文件只负责
 * 呈现与编排。检索为什么要单独一栏的设计见
 * `docs/plans/2026-09-30-keybindings-search-design.md`。
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Flex, TextField, IconButton, Button } from "@radix-ui/themes";
import { Cross2Icon, MagnifyingGlassIcon } from "@radix-ui/react-icons";
import { useI18n } from "../../i18n/I18nProvider";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { IS_MAC } from "../../utils/platform";
import {
    selectMergedKeybindings,
    setKeybinding,
    resetKeybinding,
    resetAllKeybindings,
    findConflicts,
    createModifierOnlyBinding,
} from "../../features/keybindings/keybindingsSlice";
import {
    DEFAULT_KEYBINDINGS,
    ACTION_META,
    ALL_ACTION_IDS,
    ACTION_GROUP_ORDER,
    GROUP_LABEL_KEYS,
} from "../../features/keybindings/defaultKeybindings";
import {
    buildKeybindingSearchEntries,
    matchKeybindingEntries,
} from "../../features/keybindings/keybindingSearch";
import type { ActionId, ActionMeta, Keybinding } from "../../features/keybindings/types";
import { canonicalKeyFromEvent } from "../../features/keybindings/keybindingMatch";
import { useShortcutSuppression } from "../../ui/shortcutScope";
import { AppDialog, AppSelect } from "../../ui";
import type { MessageKey } from "../../i18n/messages";
import {
    KEYBINDING_PRESET_SELECTION_IDS,
    KEYBINDING_PRESETS,
    isKeybindingPresetId,
    type KeybindingPresetSelectionId,
} from "../../features/keybindings/keybindingPresets";
import { KeybindingsActionRow } from "./keybindings/KeybindingsActionRow";
import {
    GESTURE_BADGES,
    isDefaultBinding,
    resolveGroupNavLabel,
} from "./keybindings/keybindingRowShared";
import { KeybindingsNavRail, type KeybindingsNavItem } from "./keybindings/KeybindingsNavRail";

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

export const KeybindingsDialog: React.FC<KeybindingsDialogProps> = ({ open, onOpenChange }) => {
    const dispatch = useAppDispatch();
    const { tf, plural } = useI18n();
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
    /** 搜索查询。空串表示不过滤。 */
    const [query, setQuery] = useState("");
    /** 选中的分类；`null` 为"全部"。 */
    const [activeGroup, setActiveGroup] = useState<ActionMeta["group"] | null>(null);

    /*
     * 每次打开都从"不过滤、不选分类"开始。
     *
     * 【为什么在渲染期做，而不是放 effect】这是 React 官方的「按 props 变化调整
     * state」写法：上一次的 `open` 存在 state 里，渲染时与本次比较，变化就**同步**
     * 重置。写在 effect 里会先渲染出带旧查询的一帧、再由第二轮渲染纠正 ——
     * 级联渲染，也正是 `react-hooks/set-state-in-effect` 要拦的写法。
     */
    const [prevOpen, setPrevOpen] = useState(open);
    if (prevOpen !== open) {
        setPrevOpen(open);
        if (open) {
            setQuery("");
            setActiveGroup(null);
        }
    }

    const searchInputRef = useRef<HTMLInputElement | null>(null);
    /** 聚焦搜索框属于"与外部系统同步"（操作 DOM），可以留在 effect 里。 */
    useEffect(() => {
        if (!open) return;
        // Radix 会把焦点交给第一个可聚焦元素；等它完成再夺过来。
        const timer = window.setTimeout(() => searchInputRef.current?.focus(), 0);
        return () => window.clearTimeout(timer);
    }, [open]);

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

    /*
     * 检索索引：只在语系变化时重建。
     *
     * 【为什么依赖 `tf` 而不是直接在模块顶层构建】索引里装的是**本地化后的**文本，
     * 语系一变就要重建。`tf` 由 Provider 提供，引用随语系切换而变，正好作为依赖。
     */
    const searchEntries = useMemo(() => buildKeybindingSearchEntries(tf), [tf]);

    /** 命中 query 的全部条目（有查询时按相关度降序，无查询时为原顺序）。 */
    const matchedEntries = useMemo(
        () => matchKeybindingEntries(searchEntries, query),
        [searchEntries, query],
    );

    /** 再按选中的分类收窄。 */
    const visibleEntries = useMemo(
        () =>
            activeGroup === null
                ? matchedEntries
                : matchedEntries.filter((entry) => entry.group === activeGroup),
        [matchedEntries, activeGroup],
    );

    /** 每组是否被改过 —— 用于导航栏的小圆点。 */
    const customizedGroups = useMemo(() => {
        const set = new Set<ActionMeta["group"]>();
        for (const id of ALL_ACTION_IDS) {
            if (!isDefaultBinding(keybindings[id], DEFAULT_KEYBINDINGS[id])) {
                set.add(ACTION_META[id].group);
            }
        }
        return set;
    }, [keybindings]);

    /** 导航栏条目。 */
    const navItems = useMemo<KeybindingsNavItem[]>(() => {
        const countsByGroup = new Map<ActionMeta["group"], number>();
        const totalsByGroup = new Map<ActionMeta["group"], number>();
        for (const id of ALL_ACTION_IDS) {
            const group = ACTION_META[id].group;
            totalsByGroup.set(group, (totalsByGroup.get(group) ?? 0) + 1);
        }
        if (query.trim()) {
            for (const entry of matchedEntries) {
                countsByGroup.set(entry.group, (countsByGroup.get(entry.group) ?? 0) + 1);
            }
        }

        return [
            {
                group: null,
                label: tf("kb_group_all"),
                total: ALL_ACTION_IDS.length,
                matchCount: query.trim() ? matchedEntries.length : undefined,
                customized: customizedGroups.size > 0,
            },
            ...ACTION_GROUP_ORDER.map((group) => ({
                group,
                label: resolveGroupNavLabel(group, tf),
                total: totalsByGroup.get(group) ?? 0,
                matchCount: query.trim() ? (countsByGroup.get(group) ?? 0) : undefined,
                customized: customizedGroups.has(group),
            })),
        ];
    }, [tf, query, matchedEntries, customizedGroups]);

    /** 无查询时按分组展示；选中某一组时只展示该组。 */
    const groupsToRender = useMemo(
        () =>
            ACTION_GROUP_ORDER.filter((group) => activeGroup === null || group === activeGroup)
                .map((group) => ({
                    group,
                    actions: ALL_ACTION_IDS.filter((id) => ACTION_META[id].group === group),
                }))
                .filter((entry) => entry.actions.length > 0),
        [activeGroup],
    );

    /** 单行渲染所需的回调与文案。抽出来避免每行重建闭包。 */
    const noneLabel = tf("kb_none");
    const pressKeyLabel = tf("kb_press_key");
    const pressModifierLabel = tf("kb_press_modifier");

    const resultListRef = useRef<HTMLDivElement | null>(null);

    /** 把某一行滚进视野并聚焦它的按键按钮。 */
    const focusRow = useCallback((index: number) => {
        const rows = resultListRef.current?.querySelectorAll<HTMLElement>("[data-hs-kb-row]");
        const target = rows?.[index];
        if (!target) return;
        /*
         * 【为什么把 `scrollIntoView` 包在 try 里】jsdom 没实现它（缺少 layout），
         * 直接调用会抛。真实浏览器不受影响，但即便哪天滚动这一步失败，用户也应该
         * 拿到正确的键盘焦点 —— 焦点移动优先于滚动。
         */
        try {
            target.scrollIntoView({ block: "nearest" });
        } catch {
            // 环境不支持滚动定位；焦点仍然移动。
        }
        // 行内可能还有手势徽章里的按钮，只认 `data-hs-kb-bind` 这一个。
        target.querySelector<HTMLButtonElement>("[data-hs-kb-bind]")?.focus();
    }, []);

    const handleSearchKeyDown = useCallback(
        (event: React.KeyboardEvent<HTMLInputElement>) => {
            if (event.key === "Escape") {
                /*
                 * Esc 分级：**有查询先清空查询**，没有才让窗口关闭。
                 *
                 * 【为什么这里只用 `preventDefault`】Radix 在 `document` 上监听 Esc，
                 * React 的 `stopPropagation()` 拦不住已经走到原生的事件；真正决定窗口
                 * 生死的开关在 `beforeClose`（见下）—— Esc 在这里只负责清空查询。
                 */
                event.preventDefault();
                setQuery("");
                return;
            }
            /*
             * ↓ / ↑：从搜索框跳到结果行。
             *
             * 【为什么只在无查询时启用】有查询时结果顺序由相关度决定、用户还没看清，
             * 方向键直接落进行内反而容易误触录入按钮；此时需要的是继续输入，而不是
             * 移动焦点。无查询时列表是稳定的场景分组，跳进去才是无歧义的。
             */
            if ((event.key === "ArrowDown" || event.key === "ArrowUp") && !query.trim()) {
                const rows =
                    resultListRef.current?.querySelectorAll<HTMLElement>("[data-hs-kb-row]");
                if (!rows?.length) return;
                event.preventDefault();
                focusRow(event.key === "ArrowDown" ? 0 : rows.length - 1);
                return;
            }
            if (event.key === "Enter") {
                /*
                 * 【为什么必须 `preventDefault`】`AppDialog` 的 `<form>` 会把输入框里的
                 * Enter 交给默认动作（这里的默认动作是"关闭"）。不拦的话，搜完按回车
                 * 等于关掉整个窗口 —— 而用户只是想确认搜索。见 `Dialog.tsx` 里
                 * 「Enter 的归属」那段注释。
                 */
                event.preventDefault();
                searchInputRef.current?.blur();
            }
        },
        [query, focusRow],
    );

    const renderRow = useCallback(
        (actionId: ActionId, groupLabel?: string) => {
            const meta = ACTION_META[actionId];
            const currentKb = keybindings[actionId];
            const defaultKb = DEFAULT_KEYBINDINGS[actionId];
            const gesture = meta.modifierOperationType
                ? GESTURE_BADGES[meta.modifierOperationType]
                : undefined;

            return (
                <KeybindingsActionRow
                    key={actionId}
                    label={tf(meta.labelKey)}
                    meta={meta}
                    binding={currentKb}
                    isDefault={isDefaultBinding(currentKb, defaultKb)}
                    isRecording={recordingId === actionId}
                    gestureLabel={gesture ? tf(gesture.labelKey) : undefined}
                    isDefaultModifierOnly={Boolean(defaultKb.modifierOnly)}
                    noneLabel={noneLabel}
                    pressKeyLabel={pressKeyLabel}
                    pressModifierLabel={pressModifierLabel}
                    groupLabel={groupLabel}
                    onStartRecording={() => {
                        setConflict(null);
                        setRecordingId(actionId);
                    }}
                    onClearBinding={() => {
                        // 录入中左键点击 → 设为"无"
                        dispatch(setKeybinding({ actionId, binding: NONE_BINDING }));
                        setSelectedPreset("custom");
                        setRecordingId(null);
                        setConflict(null);
                    }}
                    onResetBinding={() => {
                        // 右键点击 → 直接重置为默认
                        dispatch(resetKeybinding(actionId));
                        setSelectedPreset("custom");
                        setRecordingId(null);
                        setConflict(null);
                    }}
                />
            );
        },
        [keybindings, recordingId, dispatch, tf, noneLabel, pressKeyLabel, pressModifierLabel],
    );

    const hasQuery = query.trim().length > 0;

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("kb_dialog_title")}
            description={tf("kb_dialog_desc")}
            size="xl"
            /*
             * 两栏布局的前提：让 body 成为**不滚动的 flex 列容器**，由内部的两个
             * pane 各自滚动。默认的 `"scroll"` 会让 body 自己滚，内层的 `flex-1`
             * 就退化为按内容高度排布 —— 详见 `Dialog.tsx` 里 `bodyLayout` 的注释。
             */
            bodyLayout="pane"
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
             * Esc / 外部点击的护栏。两条：
             *   1. 正在录入时不关 —— 按下的那个键正是要录入的内容，关掉等于丢弃；
             *   2. **有搜索查询时先让这一次 Esc 去清空查询**，窗口保持打开。
             *      Radix 在 document 级处理 Esc，React 侧拦不住；这里是唯一的否决点。
             */
            beforeClose={() => {
                if (recordingId !== null) return false;
                if (query.trim()) {
                    setQuery("");
                    return false;
                }
                return true;
            }}
        >
            <Flex direction="column" className="min-h-0 flex-1 gap-2">
                <span className="hs-type-muted shrink-0">{tf("kb_dialog_hint_click")}</span>

                {/*
                 * 搜索框单独一行、铺满宽度。
                 *
                 * 【为什么不给它和预设下拉同一行】搜索是这个窗口的主入口，和结果列表
                 * 上下对齐比与下拉框左右相邻更可读；挤在一行里它只能拿到剩余宽度
                 * （实测 608/750），视觉重心还被左侧的下拉抢走。
                 */}
                <TextField.Root
                    ref={searchInputRef}
                    size="1"
                    className="shrink-0"
                    placeholder={tf("kb_search_placeholder")}
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    onKeyDown={handleSearchKeyDown}
                    style={{ backgroundColor: "var(--qt-base)" }}
                >
                    <TextField.Slot>
                        <MagnifyingGlassIcon height="12" width="12" />
                    </TextField.Slot>
                    {query && (
                        <TextField.Slot>
                            <IconButton
                                size="1"
                                variant="ghost"
                                color="gray"
                                aria-label={tf("kb_clear_search")}
                                onClick={() => {
                                    setQuery("");
                                    searchInputRef.current?.focus();
                                }}
                                style={{ width: 16, height: 16 }}
                            >
                                <Cross2Icon width="10" height="10" />
                            </IconButton>
                        </TextField.Slot>
                    )}
                </TextField.Root>

                <Flex align="center" gap="2" className="shrink-0">
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

                <Flex className="min-h-0 flex-1 gap-3">
                    <KeybindingsNavRail
                        items={navItems}
                        activeGroup={activeGroup}
                        onSelect={setActiveGroup}
                        ariaLabel={tf("kb_nav_label")}
                        customizedHint={tf("kb_group_customized_hint")}
                    />

                    {/*
                     * 右栏：唯一的滚动区。用 `min-h-0 flex-1` 参与外层 flex 布局，
                     * 而不是写 `max-h-[Npx]` —— 后者会在滚动瓶颈里叠出第二条滚动条
                     * （见 `designSystemGates.test.ts` 的"有界滚动盒"门禁）。
                     */}
                    <div
                        ref={resultListRef}
                        className="hs-scroll-gutter min-h-0 min-w-0 flex-1 overflow-y-auto custom-scrollbar"
                    >
                        {hasQuery ? (
                            visibleEntries.length > 0 ? (
                                <Flex direction="column" gap="1">
                                    {visibleEntries.map((entry) =>
                                        renderRow(entry.id, resolveGroupNavLabel(entry.group, tf)),
                                    )}
                                </Flex>
                            ) : (
                                <span className="hs-type-muted block px-2 py-4 text-center">
                                    {tf("kb_no_results")}
                                </span>
                            )
                        ) : (
                            <Flex direction="column" gap="3">
                                {groupsToRender?.map(({ group, actions }) => (
                                    <section key={group} className="flex flex-col gap-1">
                                        {/*
                                         * 粘性组标题：需要一个不透明背景，否则滚动时行
                                         * 文字会从标题下面透出来。
                                         */}
                                        <h3 className="hs-type-section sticky top-0 z-10 m-0 bg-qt-panel py-1">
                                            {tf(GROUP_LABEL_KEYS[group])}
                                        </h3>
                                        {actions.map((actionId) => renderRow(actionId))}
                                    </section>
                                ))}
                            </Flex>
                        )}
                    </div>
                </Flex>

                {hasQuery && (
                    <span className="hs-type-caption shrink-0">
                        {plural("kb_result_count", visibleEntries.length)}
                    </span>
                )}
            </Flex>

            {/* 冲突提示：按所属场景分组展示冲突项 */}
            {conflict && (
                <Flex
                    align="center"
                    gap="2"
                    py="2"
                    px="3"
                    className="shrink-0"
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
