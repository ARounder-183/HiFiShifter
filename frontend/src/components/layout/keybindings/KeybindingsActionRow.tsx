/**
 * 快捷键设置窗口里的一行：操作名 + 绑定 chip 列表（每个 chip 也是录入触发器）+ 追加按钮。
 *
 * 【为什么单独成文件】`KeybindingsDialog` 原先把这一整块内联在分组地图里
 * （约 100 行 JSX），加上搜索框与分类导航后会逼近 700 行。本组件是纯展示 +
 * 回调，是最自然的切分点：抽出去之后主文件只剩编排。
 *
 * 这里只有"一行长什么样"，不含"哪些行被筛出来" —— 后者在 `keybindingSearch.ts`。
 *
 * 【一个动作可以绑多个键】行内渲染 `bindings` 的**全部** chip：
 * - 左键点某个 chip → 录入进**那个槽位**；
 * - 录入中左键点该 chip → 清空该槽位（多于一个槽位时即删除，只剩一个时设为"无"）；
 * - 右键点某个 chip → 删除该槽位（只剩一个时由调用方改为重置为默认）；
 * - 尾部 `+` → 追加一个槽位并立即录入。
 *
 * 交互刻意只用左右键 + 一个显式的 `+`：这是既有的两种手势（左键录入 / 右键重置）
 * 在多绑定下的自然推广，不需要再引入拖拽或下拉。
 *
 * 【为什么 chip 与 `+` 的尺寸都是固定的】录入时 chip 的文字会从"当前键位"变成
 * "请按键…"，`+` 也会因为不能追加而禁用 —— 若宽度随内容走，整行会在点击的瞬间
 * 左右跳动（用户报告的正是这个"按钮不对齐"）。所以：
 * - chip **固定宽度** + 省略号（完整文本放 `title`，悬停可见）；
 * - `+` 的**位置常驻**（非修饰键动作永远占这一格），只在不能追加时禁用。
 * 于是进入 / 离开录入态、追加 / 删除槽位都不会移动任何已有元素。
 */
import { Box, Flex } from "@radix-ui/themes";
import { PlusIcon } from "@radix-ui/react-icons";

import { AppButton, AppIconButton, AppStatusChip } from "../../../ui";
import type { ActionMeta, Keybinding } from "../../../features/keybindings/types";
import { MAX_BINDINGS_PER_ACTION } from "../../../features/keybindings/types";
import { formatKeybinding } from "../../../features/keybindings/keybindingsSlice";
import { GESTURE_BADGES } from "./keybindingRowShared";

/**
 * chip 的固定宽度（px）。
 *
 * 取值依据：能完整容纳默认表里最长的组合（`Ctrl+Shift+Z`，13 字符）与各语系的
 * "请按键…"提示（最长的是日文 `キーを押してください...`）。更长的用户自定义组合
 * 以省略号收尾，完整文本在 `title` 里。
 */
const CHIP_WIDTH_PX = 132;

/**
 * 追加槽位的边长。
 *
 * **这个值定义的是"槽位"，不是"按钮"**：槽位在每一行都存在（见下方 `Box` 的
 * 说明），按钮只是可能不画。对齐因此由槽位保证，不依赖按钮的渲染与否 ——
 * 取值走 `--qt-ctl-md` 令牌（`AppIconButton` 的 `md` 档就是它），
 * 于是 Radix 内部尺寸变了也不会与按钮失配。
 */
const ADD_SLOT_SIZE = "var(--qt-ctl-md)";

export interface KeybindingsActionRowProps {
    /** 本地化后的操作名。由调用方解析，本组件不接触 i18n。 */
    label: string;
    meta: ActionMeta;
    /** 当前生效的**全部**绑定（已合并用户覆盖）；下标 0 是主绑定。 */
    bindings: readonly Keybinding[];
    /** 正在录入的槽位下标；`null` = 本行未在录入。 */
    recordingSlot: number | null;
    /** 本地化后的手势徽章文案。有 `modifierOperationType` 时必填。 */
    gestureLabel?: string;
    /**
     * 是否为**修饰键手势**动作。
     *
     * 这类动作只有一个槽位：它的"键"就是修饰键本身，绑两个组合没有可解释的
     * 语义（按下哪一个算触发？）。因此既不用修饰键提示文案，也不提供追加按钮。
     */
    isModifierOnly: boolean;
    /** "无" 绑定的本地化占位文案。 */
    noneLabel: string;
    pressKeyLabel: string;
    pressModifierLabel: string;
    /** 追加槽位按钮的 `aria-label`。 */
    addBindingLabel: string;
    onStartRecording: (slot: number) => void;
    /** 录入中点击该槽位 → 清空它。 */
    onClearSlot: (slot: number) => void;
    /** 右键点击槽位 → 删除它（只剩一个时由调用方决定是否重置为默认）。 */
    onRemoveSlot: (slot: number) => void;
    /** 追加一个槽位并开始录入。 */
    onAddBinding: () => void;
    /**
     * 搜索结果（打平展示）时附带的分组名路标。按分组展示时省略。
     *
     * 【为什么需要】打平列表失去了分组标题这个视觉锚点，用户看到"粘贴"分不清
     * 是音频块的还是钢琴卷帘的 —— 两条同名但作用于不同表面。
     */
    groupLabel?: string;
}

export function KeybindingsActionRow({
    label,
    meta,
    bindings,
    recordingSlot,
    gestureLabel,
    isModifierOnly,
    noneLabel,
    pressKeyLabel,
    pressModifierLabel,
    addBindingLabel,
    onStartRecording,
    onClearSlot,
    onRemoveSlot,
    onAddBinding,
    groupLabel,
}: KeybindingsActionRowProps) {
    const isRecording = recordingSlot !== null;
    const recordPrompt = isModifierOnly ? pressModifierLabel : pressKeyLabel;
    const atBindingLimit = bindings.length >= MAX_BINDINGS_PER_ACTION;

    // 追加槽位时该槽位还没有值，但仍要渲染出来 —— 否则用户点了 `+` 之后
    // 界面上没有任何变化，只能靠猜"现在该按键了"。
    const slotCount =
        recordingSlot !== null ? Math.max(bindings.length, recordingSlot + 1) : bindings.length;

    return (
        <Flex
            align="center"
            justify="between"
            px="2"
            py="1"
            data-hs-kb-row={meta.group}
            style={{
                borderRadius: "var(--qt-radius-sm)",
                background: isRecording ? "var(--qt-accent-subtle)" : undefined,
                minHeight: 36,
            }}
        >
            <Flex align="center" gap="2" minWidth="0">
                {/* 修饰键手势徽章：区分拖拽 / 点击 / 滚轮 / 按住 */}
                {meta.modifierOperationType && (
                    <AppStatusChip tone={GESTURE_BADGES[meta.modifierOperationType].tone}>
                        {gestureLabel}
                    </AppStatusChip>
                )}
                <span className="hs-type-body truncate">{label}</span>
                {groupLabel && (
                    <span className="hs-type-caption shrink-0" style={{ opacity: 0.75 }}>
                        {groupLabel}
                    </span>
                )}
            </Flex>
            <Flex align="center" gap="1" className="shrink-0">
                {Array.from({ length: slotCount }, (_, slot) => {
                    const binding = bindings[slot];
                    const slotIsRecording = recordingSlot === slot;
                    const text = slotIsRecording
                        ? recordPrompt
                        : binding
                          ? formatKeybinding(binding, noneLabel)
                          : noneLabel;
                    return (
                        <AppButton
                            /*
                             * `data-hs-kb-bind` 是键盘导航（↑/↓ 从搜索框跳进行内）与测试选中
                             * 这一行的入口。挂在**主绑定**（槽位 0）上，行级定位因此不受槽位
                             * 数量影响；槽位自身用 `data-hs-kb-slot` 标识。
                             */
                            key={`${meta.group}-${label}-${slot}`}
                            {...(slot === 0 ? { "data-hs-kb-bind": label } : {})}
                            data-hs-kb-slot={slot}
                            /*
                             * 只有"正在录入"需要跳出来（实心强调色）。**不再给自定义绑定
                             * 上色**：那原本是写死的绿色，而"这一组里有自定义绑定"已经由
                             * 分组标题的 `kb_group_customized_hint` 说明 —— 同一件事说两遍，
                             * 还用的是设计系统之外的调色板。
                             */
                            intent={slotIsRecording ? "primary" : "default"}
                            size="sm"
                            /*
                             * 固定宽度 + 省略号：录入提示与键位文本宽度不同，不固定就会
                             * 整行跳动（用户报告过"按钮不对齐"）。完整文本放 `title`。
                             */
                            className="hs-type-mono truncate"
                            style={{ width: CHIP_WIDTH_PX, flex: "0 0 auto" }}
                            title={text}
                            onClick={() =>
                                slotIsRecording ? onClearSlot(slot) : onStartRecording(slot)
                            }
                            onContextMenu={(e) => {
                                e.preventDefault();
                                onRemoveSlot(slot);
                            }}
                        >
                            {text}
                        </AppButton>
                    );
                })}
                {/*
                 * 追加**槽位**：每一行都占这一格，与"这一行有没有按钮"无关。
                 *
                 * 【为什么连修饰键行也要占位】整行是 `justify="between"`，右侧
                 * 那组贴着行右缘。修饰键手势不能绑多个键（按下哪一个算触发无法
                 * 解释），因此它那一行**没有**追加按钮 —— 若连槽位也一起省掉，
                 * 它的 chip 就会比其他行往右多出「按钮 + 间距」那一段，搜索把两类
                 * 动作混排时 chip 列参差不齐。占位之后对齐由槽位保证，与按钮画不画
                 * 无关。
                 *
                 * 槽位留空而不是放一个永久禁用的按钮：永远不可能被启用的控件是
                 * 噪声（用户会去猜"怎样才能点它"）。
                 */}
                <Box
                    data-hs-kb-add-slot={label}
                    style={{ width: ADD_SLOT_SIZE, height: ADD_SLOT_SIZE }}
                    className="shrink-0"
                >
                    {!isModifierOnly && (
                        <AppIconButton
                            icon={<PlusIcon />}
                            tooltip={addBindingLabel}
                            data-hs-kb-add={label}
                            /* 录入中或已达上限时禁用，但位置常驻。 */
                            disabled={isRecording || atBindingLimit}
                            onClick={onAddBinding}
                        />
                    )}
                </Box>
            </Flex>
        </Flex>
    );
}
