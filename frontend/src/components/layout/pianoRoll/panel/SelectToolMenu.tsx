/**
 * 工具栏的**选择族** split button 与模式菜单。
 *
 * 选择族只有一个槽位，内部有两种对象级模式：
 * - 参数线范围选择（range）：在曲线上拖出参数选区；
 * - 参数块选择（block）：命中块层，复用同一份参数选区做块选择 / 拖动。
 *
 * 左键进入选择族并保留当前（或上次）子模式；右键 / 长按打开菜单显式选择。
 * 按钮的强调色只看工具族是否激活，图标看选择族记住的子模式 —— 因此块模式下
 * 选择按钮仍然高亮，不会出现“选择、绘制两个按钮同时熄灭”的空档。
 */
import { Box } from "@radix-ui/themes";
import { CheckIcon, CursorArrowIcon } from "@radix-ui/react-icons";
import {
    formatKeybindingList,
    isNoneBindingList,
} from "../../../../features/keybindings/keybindingsSlice";
import {
    setSelectToolModePersistent,
    setToolModePersistent,
} from "../../../../features/session/thunks/toolModeThunks";
import { AppAnchoredMenu } from "../../../../ui";
import type { PianoRollPanelModel } from "./usePianoRollPanelModel";
import { ToolFamilyButton } from "./ToolFamilyButton";

type Props = {
    m: Pick<
        PianoRollPanelModel,
        | "blockSelectionKb"
        | "currentSelectToolIcon"
        | "dispatch"
        | "paramBlockToolIcon"
        | "s"
        | "selectToolButtonTitle"
        | "selectToolKb"
        | "selectToolMenuAnchorRef"
        | "selectToolMenuOpen"
        | "selectToolMenuRef"
        | "setSelectToolMenuOpen"
        | "tf"
    >;
};

export function SelectToolMenu({ m }: Props) {
    const {
        blockSelectionKb,
        currentSelectToolIcon,
        dispatch,
        paramBlockToolIcon,
        s,
        selectToolButtonTitle,
        selectToolKb,
        selectToolMenuAnchorRef,
        selectToolMenuOpen,
        selectToolMenuRef,
        setSelectToolMenuOpen,
        tf,
    } = m;

    return (
        <Box ref={selectToolMenuAnchorRef} data-hs-context-menu>
            <ToolFamilyButton
                active={s.toolMode === "select"}
                tooltip={`${tf("select_tool_menu")}: ${selectToolButtonTitle}`}
                icon={currentSelectToolIcon}
                onClick={() => void dispatch(setToolModePersistent("select"))}
                onContextMenu={(e) => {
                    e.preventDefault();
                    setSelectToolMenuOpen(true);
                }}
            />
            <AppAnchoredMenu
                open={selectToolMenuOpen}
                anchorRef={selectToolMenuAnchorRef}
                menuRef={selectToolMenuRef}
            >
                {[
                    {
                        mode: "range" as const,
                        label: tf("select_tool_range"),
                        icon: <CursorArrowIcon />,
                        shortcut: selectToolKb,
                    },
                    {
                        mode: "block" as const,
                        label: tf("select_tool_block"),
                        icon: paramBlockToolIcon,
                        shortcut: blockSelectionKb,
                    },
                ].map((item) => {
                    // 勾选表达“选择族记住的是哪个子模式”，与工具族是否激活无关。
                    const active = s.selectToolMode === item.mode;
                    return (
                        <button
                            key={item.mode}
                            type="button"
                            className="hs-menu__item"
                            onClick={() => {
                                void dispatch(setSelectToolModePersistent(item.mode));
                                setSelectToolMenuOpen(false);
                            }}
                            onPointerDown={(e) => e.stopPropagation()}
                        >
                            <span className="flex min-w-0 items-center gap-2">
                                <Box className="hs-menu__icon">{item.icon}</Box>
                                <span className="hs-menu__label-text">{item.label}</span>
                            </span>
                            <span className="hs-menu__trail">
                                {active ? (
                                    <span className="hs-menu__check">
                                        <CheckIcon />
                                    </span>
                                ) : null}
                                {isNoneBindingList(item.shortcut) ? null : (
                                    <span>{formatKeybindingList(item.shortcut, "")}</span>
                                )}
                            </span>
                        </button>
                    );
                })}
            </AppAnchoredMenu>
        </Box>
    );
}
