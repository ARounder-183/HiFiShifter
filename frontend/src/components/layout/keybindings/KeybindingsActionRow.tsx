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
 * - 尾部 `+` → 追加一个槽位并立即录入（达到 `MAX_BINDINGS_PER_ACTION` 时隐藏）。
 *
 * 交互刻意只用左右键 + 一个显式的 `+`：这是既有的两种手势（左键录入 / 右键重置）
 * 在多绑定下的自然推广，不需要再引入拖拽或下拉。
 */
import { Button, Flex, IconButton } from "@radix-ui/themes";
import { PlusIcon } from "@radix-ui/react-icons";

import { AppStatusChip } from "../../../ui";
import type { ActionMeta, Keybinding } from "../../../features/keybindings/types";
import { formatKeybinding } from "../../../features/keybindings/keybindingsSlice";
import { GESTURE_BADGES } from "./keybindingRowShared";

export interface KeybindingsActionRowProps {
    /** 本地化后的操作名。由调用方解析，本组件不接触 i18n。 */
    label: string;
    meta: ActionMeta;
    /** 当前生效的**全部**绑定（已合并用户覆盖）；下标 0 是主绑定。 */
    bindings: readonly Keybinding[];
    /** 是否为默认绑定列表 —— 非默认的行用绿色按钮标出。 */
    isDefault: boolean;
    /** 正在录入的槽位下标；`null` = 本行未在录入。 */
    recordingSlot: number | null;
    /** 本地化后的手势徽章文案。有 `modifierOperationType` 时必填。 */
    gestureLabel?: string;
    /** 默认绑定是否为纯修饰键手势 —— 决定录入提示文案。 */
    isDefaultModifierOnly: boolean;
    /** "无" 绑定的本地化占位文案。 */
    noneLabel: string;
    pressKeyLabel: string;
    pressModifierLabel: string;
    /** 追加槽位按钮的 `aria-label`。 */
    addBindingLabel: string;
    /** 是否还能再追加一个槽位。 */
    canAddBinding: boolean;
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
    isDefault,
    recordingSlot,
    gestureLabel,
    isDefaultModifierOnly,
    noneLabel,
    pressKeyLabel,
    pressModifierLabel,
    addBindingLabel,
    canAddBinding,
    onStartRecording,
    onClearSlot,
    onRemoveSlot,
    onAddBinding,
    groupLabel,
}: KeybindingsActionRowProps) {
    const isRecording = recordingSlot !== null;
    const recordPrompt = isDefaultModifierOnly ? pressModifierLabel : pressKeyLabel;

    return (
        <Flex
            align="center"
            justify="between"
            px="2"
            py="1"
            data-hs-kb-row={meta.group}
            style={{
                borderRadius: "var(--qt-radius-sm)",
                background: isRecording ? "var(--accent-3)" : undefined,
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
            <Flex align="center" gap="1" style={{ flexShrink: 0 }}>
                {/*
                 * 绑定 chip 列表：一个 chip 一个槽位。
                 *
                 * 追加槽位时（`recordingSlot === bindings.length`）该槽位还没有值，
                 * 但**必须**照样渲染出来 —— 否则用户点了 `+` 之后界面上没有任何
                 * 变化，只能靠猜"现在该按键了"。
                 */}
                {Array.from(
                    {
                        length:
                            recordingSlot !== null
                                ? Math.max(bindings.length, recordingSlot + 1)
                                : bindings.length,
                    },
                    (_, slot) => {
                        const binding = bindings[slot];
                        const slotIsRecording = recordingSlot === slot;
                        return (
                            <Button
                                /*
                                 * `data-hs-kb-bind` 是键盘导航（↑/↓ 从搜索框跳进行内）与测试
                                 * 选中这一行的入口。挂在**主绑定**（槽位 0）上，行级定位因此
                                 * 不受槽位数量影响；槽位自身用 `data-hs-kb-slot` 标识。
                                 */
                                key={`${meta.group}-${label}-${slot}`}
                                {...(slot === 0 ? { "data-hs-kb-bind": label } : {})}
                                data-hs-kb-slot={slot}
                                variant={slotIsRecording ? "solid" : "soft"}
                                color={slotIsRecording ? "blue" : !isDefault ? "green" : "gray"}
                                size="1"
                                style={{ minWidth: 96, fontFamily: "monospace" }}
                                onClick={() =>
                                    slotIsRecording ? onClearSlot(slot) : onStartRecording(slot)
                                }
                                onContextMenu={(e) => {
                                    e.preventDefault();
                                    onRemoveSlot(slot);
                                }}
                            >
                                {slotIsRecording
                                    ? recordPrompt
                                    : binding
                                      ? formatKeybinding(binding, noneLabel)
                                      : noneLabel}
                            </Button>
                        );
                    },
                )}
                {/* 追加槽位：唯一可见的"能绑多个键"入口；录入中不出现以免误触 */}
                {canAddBinding && !isRecording && (
                    <IconButton
                        size="1"
                        variant="ghost"
                        color="gray"
                        aria-label={addBindingLabel}
                        data-hs-kb-add={label}
                        onClick={onAddBinding}
                    >
                        <PlusIcon />
                    </IconButton>
                )}
            </Flex>
        </Flex>
    );
}
