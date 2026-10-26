// 颤音预设列表行：选中编辑、双击设为当前、启用/停用、拖拽排序与就地重命名。
import React, { useEffect, useRef } from "react";
import { Box, Flex } from "@radix-ui/themes";
import { CheckCircledIcon, CircleIcon, EyeNoneIcon, EyeOpenIcon } from "@radix-ui/react-icons";
import { useI18n } from "../../../i18n/I18nProvider";
import { AppIconButton, AppListRow } from "../../../ui";
import type { VibratoPreset } from "../../../features/vibrato/vibratoTypes";
import {
    vibratoPresetDescription,
    vibratoPresetLabel,
    vibratoPresetSummary,
} from "../vibrato/vibratoDialogLogic";
import { VibratoPresetGlyph } from "../vibrato/VibratoPresetGlyph";

interface PresetRowProps {
    preset: VibratoPreset;
    selected: boolean;
    active: boolean;
    /** 是否启用（停用的预设不出现在工具栏列表里，循环切换也会跳过）。 */
    enabled: boolean;
    /** 可拖拽排序（用户预设；系统预设顺序固定）。 */
    reorderable?: boolean;
    onSelect: () => void;
    onActivate: () => void;
    onToggleEnabled: () => void;
    onContextMenu: (x: number, y: number) => void;
    /** 行上按下指针（拖拽排序的起点）。 */
    onDragPointerDown?: (event: React.PointerEvent<HTMLDivElement>) => void;
    /** 就地重命名：输入框覆盖在名字上。 */
    renaming?: boolean;
    renameValue?: string;
    onRenameChange?: (value: string) => void;
    onRenameCommit?: () => void;
    onRenameCancel?: () => void;
}

/**
 * 列表行：单击选中（编辑它），双击设为当前使用，右侧按钮启用 / 停用；
 * 预设还可以按住行上下拖拽排序。
 *
 * 【为什么分开】"编辑某个预设"与"现在就用某个预设"是两件事：用户可能正在调
 * 一个还没调好的预设，却仍然希望拖拽用着上一个。合起来会让编辑动作顺带改掉
 * 当前音色。
 */
export function PresetRow({
    preset,
    selected,
    active,
    enabled,
    reorderable = false,
    onSelect,
    onActivate,
    onToggleEnabled,
    onContextMenu,
    onDragPointerDown,
    renaming = false,
    renameValue,
    onRenameChange,
    onRenameCommit,
    onRenameCancel,
}: PresetRowProps) {
    const { t, tVars } = useI18n();
    const description = vibratoPresetDescription(preset, t);
    const label = vibratoPresetLabel(preset, t);
    const summary = description ?? vibratoPresetSummary(preset, t);

    /*
     * 聚焦放在 effect 里，而不是用 `autoFocus`。
     *
     * 【为什么】重命名是从右键菜单点开的，而菜单关闭时会把"打开前的焦点"还回去
     * （`AppContextMenu` 的焦点归还）。`autoFocus` 在**挂载那一刻**就抢焦点，紧接着
     * 菜单的被动 effect 清理又把焦点还给了那一行 —— 输入框刚出现就被 blur，而 blur
     * 会提交，于是它立刻消失，看起来"点了重命名没反应"。effect 在清理之后运行，
     * 顺序才对。
     */
    const renameInputRef = useRef<HTMLInputElement | null>(null);
    useEffect(() => {
        if (!renaming) return;
        const input = renameInputRef.current;
        if (!input) return;
        input.focus();
        // 全选：重命名多半是整体替换，直接敲字即可。
        input.select();
    }, [renaming]);

    return (
        <Flex align="center" gap="1" style={{ minWidth: 0, flex: 1 }}>
            {/*
             * 「设为当前使用」——每行一个单选圆点，兼作状态显示：实心即当前使用中。
             *
             * 【为什么必须是一个**看得见的控件**】这一行原来只有一个 `●`：它既不像
             * 按钮，也没说明是什么状态，于是"怎么把某条设成当前使用的"只剩双击与右键
             * 菜单两个藏起来的入口。圆点把状态与操作合成一处 —— 扫一眼知道用的是哪条，
             * 点一下换过去。
             *
             * 【为什么放在行外】与右侧的眼睛按钮同一处理：`AppListRow` 是列表项，
             * 里面再嵌按钮会让"点行 = 选中编辑"与"点按钮 = 设为使用"互相打架。
             */}
            <AppIconButton
                size="sm"
                active={active}
                emphasis="accent"
                icon={active ? <CheckCircledIcon /> : <CircleIcon />}
                tooltip={active ? t("vibrato_manager_in_use") : t("vibrato_manager_set_active")}
                onClick={onActivate}
            />
            <Box style={{ minWidth: 0, flex: 1 }}>
                <AppListRow
                    selected={selected}
                    density="compact"
                    role="option"
                    onClick={onSelect}
                    onDoubleClick={onActivate}
                    onPointerDown={onDragPointerDown}
                    onContextMenu={(event) => {
                        event.preventDefault();
                        onContextMenu(event.clientX, event.clientY);
                    }}
                    // 可拖拽的行给一个"抓得住"的光标；不可拖的（系统预设）保持默认。
                    className={reorderable ? "cursor-grab" : undefined}
                    // 项目自定义气泡（不是浏览器原生 title）：样式与全应用一致。
                    tooltip={
                        enabled ? summary : tVars("vibrato_preset_tooltip_disabled", { summary })
                    }
                >
                    {/* `flex: 1` 让这一行铺满列表行：重命名时输入框才有可用的宽度去
                        撑开，而不是反过来把行撑宽（见输入框上的 `size` 说明）。 */}
                    <Flex align="center" gap="2" style={{ minWidth: 0, flex: 1 }}>
                        <VibratoPresetGlyph preset={preset} width={40} height={14} />
                        {renaming ? (
                            <input
                                ref={renameInputRef}
                                /*
                                 * `size={1}` 是关键：`<input>` 默认 `size=20`，其固有宽度
                                 * 约 170px —— 比列表列（208px）减去字形后还宽，会把整行
                                 * 撑出去、连累整个列表横向位移。压到 1 之后它不再贡献
                                 * 固有宽度，只按 `flex-1` 填满行内剩余空间。
                                 */
                                size={1}
                                className="hs-type-label min-w-0 flex-1 rounded border border-qt-border bg-qt-window px-1 py-0 text-qt-text outline-hidden focus:border-qt-highlight"
                                value={renameValue ?? ""}
                                aria-label={t("vibrato_manager_rename")}
                                // 行本身带点击 / 双击 / 拖拽：输入框内的事件一律不外泄，
                                // 否则在框里拖选文字会触发"拖拽排序"、双击会激活预设。
                                onPointerDown={(event) => event.stopPropagation()}
                                onClick={(event) => event.stopPropagation()}
                                onDoubleClick={(event) => event.stopPropagation()}
                                onChange={(event) => onRenameChange?.(event.target.value)}
                                onKeyDown={(event) => {
                                    // 先于窗口级 Escape 关闭处理（与 Clip 的内联重命名
                                    // 同一约定）。
                                    event.stopPropagation();
                                    if (event.key === "Enter") onRenameCommit?.();
                                    else if (event.key === "Escape") onRenameCancel?.();
                                }}
                                // 失焦即提交：点别处不该丢掉刚敲的名字。
                                onBlur={() => onRenameCommit?.()}
                            />
                        ) : (
                            <span
                                className="hs-type-label"
                                style={{
                                    overflow: "hidden",
                                    textOverflow: "ellipsis",
                                    whiteSpace: "nowrap",
                                    // 停用的预设压暗：一眼看出它不会出现在工具栏里。
                                    opacity: enabled ? undefined : 0.45,
                                }}
                            >
                                {label}
                            </span>
                        )}
                    </Flex>
                </AppListRow>
            </Box>
            {/* 启用 / 停用：停用后不进工具栏列表、拖拽切换也跳过。 */}
            <AppIconButton
                size="sm"
                emphasis={enabled ? "neutral" : "accent"}
                icon={enabled ? <EyeOpenIcon /> : <EyeNoneIcon />}
                // 提示的是**这个按钮点了会做什么**：启用中就说"停用"，反之亦然。
                tooltip={enabled ? t("vibrato_manager_disable") : t("vibrato_manager_enable")}
                onClick={onToggleEnabled}
            />
        </Flex>
    );
}
