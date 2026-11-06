/**
 * 参数编辑器「块」药丸与下拉菜单。
 *
 * 与参数面板的药丸同构：眼睛控制块层显隐，标签进入块工具，右侧箭头展开
 * 合并 / 解合并动作。菜单仍沿用 `AppAnchoredMenu` 的锚点结构。
 */
import { Box } from "@radix-ui/themes";
import {
    CheckIcon,
    ChevronDownIcon,
    LayersIcon,
    ResetIcon,
    TrashIcon,
} from "@radix-ui/react-icons";
import { setToolModePersistent } from "../../../../features/session/sessionSlice";
import { AppAnchoredMenu } from "../../../../ui";
import type { PianoRollPanelModel } from "./usePianoRollPanelModel";
import type { MessageKey } from "../../../../i18n/messages";
import { ParamToolbarPill } from "./toolbarParts";

type Props = {
    m: Pick<
        PianoRollPanelModel,
        | "blockToolMenuAnchorRef"
        | "blockToolMenuOpen"
        | "blockToolMenuRef"
        | "canMergeSelectedParamBlocks"
        | "canUnmergeSelectedParamBlocks"
        | "canDeleteSelectedParamBlocks"
        | "canRestoreDeletedParamBlocks"
        | "dispatch"
        | "mergeAllOverlappingParamBlocks"
        | "mergeOverlapsActive"
        | "mergeSelectedParamBlocks"
        | "deleteSelectedParamBlocks"
        | "restoreParamBlocks"
        | "paramBlockMessage"
        | "paramBlockToolActive"
        | "paramBlocksVisible"
        | "selectedParamBlockCount"
        | "setBlockToolMenuOpen"
        | "s"
        | "tf"
        | "toggleMergeOverlaps"
        | "toggleParamBlocksVisible"
        | "unmergeSelectedParamBlocks"
    >;
};

export function BlockToolMenu({ m }: Props) {
    const {
        blockToolMenuAnchorRef,
        blockToolMenuOpen,
        blockToolMenuRef,
        canMergeSelectedParamBlocks,
        canUnmergeSelectedParamBlocks,
        canDeleteSelectedParamBlocks,
        canRestoreDeletedParamBlocks,
        dispatch,
        mergeAllOverlappingParamBlocks,
        mergeOverlapsActive,
        mergeSelectedParamBlocks,
        deleteSelectedParamBlocks,
        restoreParamBlocks,
        paramBlockMessage,
        paramBlockToolActive,
        paramBlocksVisible,
        selectedParamBlockCount,
        setBlockToolMenuOpen,
        s,
        tf,
        toggleMergeOverlaps,
        toggleParamBlocksVisible,
        unmergeSelectedParamBlocks,
    } = m;

    const close = () => setBlockToolMenuOpen(false);

    /*
     * 药丸主体进入块工具；再点一次回到进入前的绘制工具。此前只切 `toolMode`，
     * 但 `s.drawToolMode` 在块工具期间仍保存着旧值，用户第二次点击会“没有反应”。
     */
    const toggleBlockTool = () => {
        const next = paramBlockToolActive ? s.drawToolMode || "draw" : "block";
        void dispatch(setToolModePersistent(next));
    };

    return (
        <Box
            ref={blockToolMenuAnchorRef}
            data-hs-context-menu
            onContextMenu={(e) => {
                e.preventDefault();
                setBlockToolMenuOpen(true);
            }}
        >
            <ParamToolbarPill
                label={tf("param_blocks_short")}
                labelKind="block-icon"
                labelAriaLabel={tf("param_blocks_tool")}
                labelTooltip={tf("param_blocks_tool")}
                active={paramBlockToolActive}
                onSelect={toggleBlockTool}
                eyeMode={paramBlocksVisible ? "on" : "off"}
                onToggleEye={toggleParamBlocksVisible}
                eyeTooltip={tf("param_blocks_show_overlay")}
                eyeLabel={tf("param_blocks_show_overlay")}
                dropdown={
                    <button
                        type="button"
                        className="param-pill__seg param-pill__seg--chev"
                        data-tooltip={tf("param_blocks_menu")}
                        aria-label={tf("param_blocks_menu")}
                        tabIndex={-1}
                        onClick={(e) => {
                            e.stopPropagation();
                            setBlockToolMenuOpen(!blockToolMenuOpen);
                        }}
                    >
                        <ChevronDownIcon width="12" height="12" />
                    </button>
                }
            />
            <AppAnchoredMenu
                open={blockToolMenuOpen}
                anchorRef={blockToolMenuAnchorRef}
                menuRef={blockToolMenuRef}
            >
                <button
                    type="button"
                    className="hs-menu__item"
                    disabled={!canMergeSelectedParamBlocks}
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={() => {
                        mergeSelectedParamBlocks();
                        close();
                    }}
                >
                    <span className="flex min-w-0 items-center gap-2">
                        <Box className="hs-menu__icon">
                            <LayersIcon />
                        </Box>
                        <span className="hs-menu__label-text">{tf("param_blocks_merge")}</span>
                    </span>
                </button>
                <button
                    type="button"
                    className="hs-menu__item"
                    disabled={!canDeleteSelectedParamBlocks}
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={() => {
                        deleteSelectedParamBlocks();
                        close();
                    }}
                >
                    <span className="flex min-w-0 items-center gap-2">
                        <Box className="hs-menu__icon">
                            <TrashIcon />
                        </Box>
                        <span className="hs-menu__label-text">{tf("param_blocks_delete")}</span>
                    </span>
                </button>
                <button
                    type="button"
                    className="hs-menu__item"
                    disabled={!canRestoreDeletedParamBlocks}
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={() => {
                        restoreParamBlocks();
                        close();
                    }}
                >
                    <span className="flex min-w-0 items-center gap-2">
                        <Box className="hs-menu__icon">
                            <ResetIcon />
                        </Box>
                        <span className="hs-menu__label-text">{tf("param_blocks_restore")}</span>
                    </span>
                </button>
                <button
                    type="button"
                    className="hs-menu__item"
                    disabled={!canUnmergeSelectedParamBlocks}
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={() => {
                        unmergeSelectedParamBlocks();
                        close();
                    }}
                >
                    <span className="flex min-w-0 items-center gap-2">
                        <Box className="hs-menu__icon">
                            <LayersIcon style={{ transform: "scaleY(-1)" }} />
                        </Box>
                        <span className="hs-menu__label-text">{tf("param_blocks_unmerge")}</span>
                    </span>
                </button>
                <Box className="hs-menu__separator shrink-0" />
                <button
                    type="button"
                    className="hs-menu__item"
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={() => {
                        toggleMergeOverlaps();
                        close();
                    }}
                >
                    <span className="flex min-w-0 flex-col gap-0.5">
                        <span className="hs-menu__label-text">
                            {tf("param_blocks_merge_overlaps")}
                        </span>
                        <span className="hs-type-caption" style={{ color: "var(--qt-text-muted)" }}>
                            {tf("param_blocks_merge_auto_hint")}
                        </span>
                    </span>
                    <span className="hs-menu__trail">
                        {mergeOverlapsActive ? <CheckIcon /> : null}
                    </span>
                </button>
                <button
                    type="button"
                    className="hs-menu__item"
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={() => {
                        mergeAllOverlappingParamBlocks();
                        close();
                    }}
                >
                    <span className="hs-menu__label-text">{tf("param_blocks_merge_do_now")}</span>
                </button>
                {(selectedParamBlockCount > 0 || paramBlockMessage !== null) && (
                    <Box
                        className="hs-type-caption"
                        style={{
                            padding: "4px 8px",
                            color: paramBlockMessage
                                ? "var(--qt-warning-text)"
                                : "var(--qt-text-muted)",
                        }}
                    >
                        {paramBlockMessage === null
                            ? `${selectedParamBlockCount}`
                            : tf(paramBlockMessage as MessageKey)}
                    </Box>
                )}
            </AppAnchoredMenu>
        </Box>
    );
}
