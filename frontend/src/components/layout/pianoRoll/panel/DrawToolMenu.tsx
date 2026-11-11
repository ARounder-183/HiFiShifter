/**
 * 工具栏的**绘制工具族** split button 与模式菜单。
 *
 * 绘制族包含自由手绘 / 直线 / 颤音三种工具。按钮显示绘制族记住的当前工具，
 * 右下角小三角表示“族内还有别的模式”；右键（或长按）打开菜单切换。菜单里还
 * 挂着颤音预设的子面板（`AppSubMenu`）与“打开预设管理”入口。
 *
 * 【为什么单独成文件】它是工具栏里**带弹出层**的按钮：按钮、锚点外壳、菜单、
 * 子面板四层结构混在工具栏的行内布局里时，横向伸缩（`minWidth: 0`、
 * `hideAt` 让位）与弹出层定位的因果关系读不出来。拆出来后，“按钮怎么排”与
 * “菜单里有什么”各自成立。
 *
 * 【锚点外壳】菜单本体挂到 `document.body`（见 `AppAnchoredMenu`），这里的
 * `Box` 只负责提供“菜单该贴在哪”的矩形，因此它必须始终挂载——不能跟着
 * 工具栏层级一起隐藏。
 */
import { Box, ScrollArea } from "@radix-ui/themes";
import { CheckIcon, Pencil1Icon } from "@radix-ui/react-icons";
import {
    isNoneBindingList,
    formatKeybindingList,
} from "../../../../features/keybindings/keybindingsSlice";
import {
    persistUiSettings,
    setActiveVibratoPreset,
} from "../../../../features/session/sessionSlice";
import { setToolModePersistent } from "../../../../features/session/thunks/toolModeThunks";
import { chooseVibratoPreset } from "../../../../features/vibrato/vibratoPresetList";
import { vibratoPresetLabel } from "../../vibrato/vibratoDialogLogic";
import { AppAnchoredMenu, AppSubMenu } from "../../../../ui";
import { ToolFamilyButton } from "./ToolFamilyButton";
import type { PianoRollPanelModel } from "./usePianoRollPanelModel";
import { VibratoPresetGlyph } from "../../vibrato/VibratoPresetGlyph";

/** 本文件用到的模型字段（AST 扫描得出）。 */
type Props = {
    m: Pick<
        PianoRollPanelModel,
        | "activeVibratoPreset"
        | "activeVibratoPresetId"
        | "currentDrawTool"
        | "currentDrawToolIcon"
        | "dispatch"
        | "drawToolButtonTitle"
        | "drawToolKb"
        | "drawToolMenuAnchorRef"
        | "drawToolMenuOpen"
        | "drawToolMenuRef"
        | "enabledVibratoPresetList"
        | "lineToolIcon"
        | "lineToolKb"
        | "openVibratoDialog"
        | "presetFlyoutMaxHeight"
        | "s"
        | "setDrawToolMenuOpen"
        | "tf"
        | "vibratoToolIcon"
        | "vibratoToolKb"
    >;
};

export function DrawToolMenu({ m }: Props) {
    const {
        activeVibratoPreset,
        activeVibratoPresetId,
        currentDrawTool,
        currentDrawToolIcon,
        dispatch,
        drawToolButtonTitle,
        drawToolKb,
        drawToolMenuAnchorRef,
        drawToolMenuOpen,
        drawToolMenuRef,
        enabledVibratoPresetList,
        lineToolIcon,
        lineToolKb,
        openVibratoDialog,
        presetFlyoutMaxHeight,
        s,
        setDrawToolMenuOpen,
        tf,
        vibratoToolIcon,
        vibratoToolKb,
    } = m;

    return (
        <Box ref={drawToolMenuAnchorRef} data-hs-context-menu>
            <ToolFamilyButton
                active={s.toolMode !== "select"}
                tooltip={drawToolButtonTitle}
                icon={currentDrawToolIcon}
                onClick={() => void dispatch(setToolModePersistent(currentDrawTool))}
                onContextMenu={(e) => {
                    e.preventDefault();
                    setDrawToolMenuOpen(true);
                }}
            />

            <AppAnchoredMenu
                open={drawToolMenuOpen}
                anchorRef={drawToolMenuAnchorRef}
                menuRef={drawToolMenuRef}
                // 菜单里挂着 `AppSubMenu` 的绝对定位子面板：壳一旦
                // `overflow-y: auto` 就会把它裁掉（与 `ClipContextMenu`
                // 同一理由，见 `.hs-menu--submenu` 的说明）。子面板自己滚动。
                className="hs-menu--no-scroll"
            >
                {/* 三种绘制工具并列。直线与颤音在这里是**平级**的两项
                                —— 它们在类型与实现上也是两个工具，只是共用一条
                                "起点 → 终点"的拖拽路径。 */}
                {[
                    {
                        mode: "draw" as const,
                        label: tf("draw_tool"),
                        icon: <Pencil1Icon />,
                        shortcut: drawToolKb,
                    },
                    {
                        mode: "line" as const,
                        label: tf("line_draw_tool"),
                        icon: lineToolIcon,
                        shortcut: lineToolKb,
                    },
                    {
                        mode: "vibrato" as const,
                        label: tf("vibrato_draw_tool"),
                        icon: vibratoToolIcon,
                        shortcut: vibratoToolKb,
                    },
                ].map((item) => {
                    // 选择族激活时，绘制菜单不伪造“当前绘制工具”的勾选：
                    // 勾选只在这三个工具真的属于当前工具族时出现。
                    const active = s.toolMode !== "select" && currentDrawTool === item.mode;
                    return (
                        <button
                            key={item.mode}
                            type="button"
                            className="hs-menu__item"
                            onClick={() => {
                                void dispatch(setToolModePersistent(item.mode));
                                setDrawToolMenuOpen(false);
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

                <Box className="hs-menu__separator shrink-0" />

                {/* 颤音工具的二级菜单 = 预设列表 + 管理器入口。
                                它紧跟在「颤音工具」下面：预设属于这个工具，摆到别处
                                会让"换一个颤音"多出一个说不清归属的入口。

                                行上带活动预设的波形缩略图与名字 —— 预设按钮从工具栏
                                撤掉之后，"现在用的是哪个预设"就靠这一行扫读。

                                【它始终是颤音工具的活动预设，与当前工具无关】这个入口
                                回答的是"**颤音工具**用哪个预设"，那是颤音工具的属性，不是
                                "当前这一笔画什么"。直线工具固定从直线预设起手（见
                                `straightVibratoPreset`），那是它自己的事，不该改掉这里显示的
                                值 —— 用户切回颤音工具时，看到的必须还是他挑的那个。 */}
                <AppSubMenu
                    label={tf("vibrato_toolbar_label")}
                    badge={vibratoPresetLabel(activeVibratoPreset, tf)}
                    icon={
                        <VibratoPresetGlyph preset={activeVibratoPreset} width={15} height={15} />
                    }
                    panelClassName="hs-menu--no-scroll flex flex-col"
                >
                    <ScrollArea
                        className="hs-scroll-area min-h-0"
                        style={{
                            flex: "1 1 auto",
                            maxHeight: presetFlyoutMaxHeight,
                        }}
                        scrollbars="vertical"
                        type="auto"
                    >
                        {enabledVibratoPresetList.map((preset) => (
                            <button
                                key={preset.id}
                                type="button"
                                className="hs-menu__item"
                                onClick={() => {
                                    // 选中「直线」预设 = 切到直线工具 ——
                                    // 两者是同一件事，由 `VibratoChoice`
                                    // 统一裁决，这里不另写判断。
                                    const choice = chooseVibratoPreset(preset.id);
                                    if (choice.tool === "line") {
                                        void dispatch(setToolModePersistent("line"));
                                    } else {
                                        void dispatch(setToolModePersistent("vibrato"));
                                        dispatch(setActiveVibratoPreset(choice.presetId));
                                        void dispatch(persistUiSettings());
                                    }
                                    setDrawToolMenuOpen(false);
                                }}
                                onPointerDown={(e) => e.stopPropagation()}
                            >
                                <span className="flex min-w-0 items-center gap-2">
                                    <VibratoPresetGlyph preset={preset} width={26} height={10} />
                                    <span className="hs-menu__label-text">
                                        {vibratoPresetLabel(preset, tf)}
                                    </span>
                                </span>
                                <span className="hs-menu__trail">
                                    {preset.id === activeVibratoPresetId ? (
                                        <span className="hs-menu__check">
                                            <CheckIcon />
                                        </span>
                                    ) : null}
                                </span>
                            </button>
                        ))}
                    </ScrollArea>
                    <Box className="hs-menu__separator shrink-0" />
                    <button
                        type="button"
                        className="hs-menu__item shrink-0"
                        onClick={() => {
                            setDrawToolMenuOpen(false);
                            openVibratoDialog("manage");
                        }}
                        onPointerDown={(e) => e.stopPropagation()}
                    >
                        <span className="hs-menu__label-text">{tf("vibrato_manager_open")}</span>
                    </button>
                </AppSubMenu>
            </AppAnchoredMenu>
        </Box>
    );
}
