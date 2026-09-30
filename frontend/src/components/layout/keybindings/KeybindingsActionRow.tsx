/**
 * 快捷键设置窗口里的一行：操作名 + 快捷键按钮（也是录入触发器）。
 *
 * 【为什么单独成文件】`KeybindingsDialog` 原先把这一整块内联在分组地图里
 * （约 100 行 JSX），加上搜索框与分类导航后会逼近 700 行。本组件是纯展示 +
 * 两个回调，是最自然的切分点：抽出去之后主文件只剩编排。
 *
 * 这里只有"一行长什么样"，不含"哪些行被筛出来" —— 后者在 `keybindingSearch.ts`。
 */
import { Button, Flex } from "@radix-ui/themes";

import { AppStatusChip } from "../../../ui";
import type { ActionMeta, Keybinding } from "../../../features/keybindings/types";
import { formatKeybinding } from "../../../features/keybindings/keybindingsSlice";
import { GESTURE_BADGES } from "./keybindingRowShared";

export interface KeybindingsActionRowProps {
    /** 本地化后的操作名。由调用方解析，本组件不接触 i18n。 */
    label: string;
    meta: ActionMeta;
    /** 当前生效的绑定（已合并用户覆盖）。 */
    binding: Keybinding;
    /** 是否为默认绑定 —— 非默认的行用绿色按钮标出。 */
    isDefault: boolean;
    /** 该操作是否处于录入模式。 */
    isRecording: boolean;
    /** 本地化后的手势徽章文案。有 `modifierOperationType` 时必填。 */
    gestureLabel?: string;
    /** 默认绑定是否为纯修饰键手势 —— 决定录入提示文案。 */
    isDefaultModifierOnly: boolean;
    /** "无" 绑定的本地化占位文案。 */
    noneLabel: string;
    pressKeyLabel: string;
    pressModifierLabel: string;
    onStartRecording: () => void;
    /** 录入中点击按钮 → 设为"无"。 */
    onClearBinding: () => void;
    /** 右键点击 → 重置为默认。 */
    onResetBinding: () => void;
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
    binding,
    isDefault,
    isRecording,
    gestureLabel,
    isDefaultModifierOnly,
    noneLabel,
    pressKeyLabel,
    pressModifierLabel,
    onStartRecording,
    onClearBinding,
    onResetBinding,
    groupLabel,
}: KeybindingsActionRowProps) {
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
            <Flex align="center" gap="2" style={{ flexShrink: 0 }}>
                {/* 快捷键显示 / 录入按钮 */}
                <Button
                    /*
                     * `data-hs-kb-bind` 是键盘导航（↑/↓ 从搜索框跳进行内）与测试选中
                     * 这一行的入口。行内可能还有手势徽章里的按钮，靠索引选不可靠。
                     */
                    data-hs-kb-bind={label}
                    variant={isRecording ? "solid" : "soft"}
                    color={isRecording ? "blue" : !isDefault ? "green" : "gray"}
                    size="1"
                    style={{ minWidth: 120, fontFamily: "monospace" }}
                    onClick={isRecording ? onClearBinding : onStartRecording}
                    onContextMenu={(e) => {
                        e.preventDefault();
                        // 右键点击 → 直接重置为默认
                        onResetBinding();
                    }}
                >
                    {isRecording
                        ? isDefaultModifierOnly
                            ? pressModifierLabel
                            : pressKeyLabel
                        : formatKeybinding(binding, noneLabel)}
                </Button>
            </Flex>
        </Flex>
    );
}
