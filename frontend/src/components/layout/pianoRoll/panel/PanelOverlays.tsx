/**
 * 面板上的浮层与模态：MIDI 拖放提示、导入对话框、右键菜单、颤音窗口。
 *
 * 【为什么单独成文件】它们都是"盖在编辑面上"的东西，与画布宿主、工具栏的布局
 * 无关；放在一起也让"面板会弹出什么"可以一屏读完。
 */
import { planParamConversion } from "../paramConversion";
import { VibratoDialog } from "../../VibratoDialog";
import { MidiTrackSelectDialog } from "../../MidiTrackSelectDialog";
import { EditContextMenu } from "../../../editDialogs/EditContextMenu";

import type { PianoRollPanelModel } from "./usePianoRollPanelModel";

/** 本文件用到的模型字段（AST 扫描得出）；`Pick<>` 保证它不会悄悄变宽。 */
type Props = {
    m: Pick<
        PianoRollPanelModel,
        | "clipboardGuid"
        | "closeLeadingGap"
        | "ctxMenu"
        | "currentParamRange"
        | "editParam"
        | "fillGaps"
        | "handleCloseLeadingGapChange"
        | "handleEditOp"
        | "handleExportMidiFromEditor"
        | "handleFillGapsChange"
        | "handleImportAsClip"
        | "handleImportBpmAsProjectChange"
        | "handleImportPositionChange"
        | "handleImportTargetChange"
        | "handleImportTempoMapEnabledChange"
        | "handleImportTempoMapKeySignatureChange"
        | "handleImportTempoMapTempoChange"
        | "handleImportTempoMapTimeSignatureChange"
        | "handleMidiImported"
        | "handleMultiTrackMergeChange"
        | "handleNoteBpmModeChange"
        | "handleRequestEnableCompose"
        | "handleSaveAsPitchRef"
        | "handleSpecifiedBpmChange"
        | "importBpmAsProject"
        | "importPosition"
        | "importTargetParamEditor"
        | "importTargetReaperClipboard"
        | "importTempoMapEnabled"
        | "importTempoMapKeySignature"
        | "importTempoMapTempo"
        | "importTempoMapTimeSignature"
        | "midiDialogOpen"
        | "midiDialogSourceRef"
        | "midiPath"
        | "midiSelRanges"
        | "midiSelectionAvailable"
        | "multiTrackMerge"
        | "noteBpmMode"
        | "openEditDialog"
        | "openVibratoDialog"
        | "paramEditorMidiDragOver"
        | "rootTrack"
        | "s"
        | "setCtxMenu"
        | "setMidiDialogOpen"
        | "setVibratoDialogOpen"
        | "specifiedBpm"
        | "tf"
        | "vibratoApplyTarget"
        | "vibratoDialogMode"
        | "vibratoDialogOpen"
        | "vibratoDialogSession"
        | "vibratoInitialPresetId"
    >;
};

export function PanelOverlays({ m }: Props) {
    const {
        clipboardGuid,
        closeLeadingGap,
        ctxMenu,
        currentParamRange,
        editParam,
        fillGaps,
        handleCloseLeadingGapChange,
        handleEditOp,
        handleExportMidiFromEditor,
        handleFillGapsChange,
        handleImportAsClip,
        handleImportBpmAsProjectChange,
        handleImportPositionChange,
        handleImportTargetChange,
        handleImportTempoMapEnabledChange,
        handleImportTempoMapKeySignatureChange,
        handleImportTempoMapTempoChange,
        handleImportTempoMapTimeSignatureChange,
        handleMidiImported,
        handleMultiTrackMergeChange,
        handleNoteBpmModeChange,
        handleRequestEnableCompose,
        handleSaveAsPitchRef,
        handleSpecifiedBpmChange,
        importBpmAsProject,
        importPosition,
        importTargetParamEditor,
        importTargetReaperClipboard,
        importTempoMapEnabled,
        importTempoMapKeySignature,
        importTempoMapTempo,
        importTempoMapTimeSignature,
        midiDialogOpen,
        midiDialogSourceRef,
        midiPath,
        midiSelRanges,
        midiSelectionAvailable,
        multiTrackMerge,
        noteBpmMode,
        openEditDialog,
        openVibratoDialog,
        paramEditorMidiDragOver,
        rootTrack,
        s,
        setCtxMenu,
        setMidiDialogOpen,
        setVibratoDialogOpen,
        specifiedBpm,
        tf,
        vibratoApplyTarget,
        vibratoDialogMode,
        vibratoDialogOpen,
        vibratoDialogSession,
        vibratoInitialPresetId,
    } = m;

    return (
        <>
            {paramEditorMidiDragOver ? (
                <div className="pointer-events-none absolute left-1/2 top-10 z-40 -translate-x-1/2 rounded border border-qt-snap-source/70 bg-qt-panel/95 px-3 py-1.5 text-qt-sm text-qt-text shadow-lg">
                    {tf("param_editor_drop_midi_hint")}
                </div>
            ) : null}
            <MidiTrackSelectDialog
                open={midiDialogOpen}
                onOpenChange={setMidiDialogOpen}
                midiPath={midiPath}
                importTarget={
                    midiDialogSourceRef.current === "reaperClipboard"
                        ? importTargetReaperClipboard
                        : importTargetParamEditor
                }
                onImportTargetChange={handleImportTargetChange}
                rootTrackComposeEnabled={rootTrack?.composeEnabled ?? true}
                onRequestEnableCompose={handleRequestEnableCompose}
                clipboardGuid={clipboardGuid}
                selectionRanges={midiSelRanges}
                onImported={handleMidiImported}
                onImportAsClip={handleImportAsClip}
                importPosition={importPosition}
                onImportPositionChange={handleImportPositionChange}
                selectionAvailable={midiSelectionAvailable}
                fillGaps={fillGaps}
                onFillGapsChange={handleFillGapsChange}
                projectBpm={s.bpm}
                importBpmAsProject={importBpmAsProject}
                onImportBpmAsProjectChange={handleImportBpmAsProjectChange}
                noteBpmMode={noteBpmMode}
                onNoteBpmModeChange={handleNoteBpmModeChange}
                specifiedBpm={specifiedBpm}
                onSpecifiedBpmChange={handleSpecifiedBpmChange}
                multiTrackMerge={multiTrackMerge}
                onMultiTrackMergeChange={handleMultiTrackMergeChange}
                closeLeadingGap={closeLeadingGap}
                onCloseLeadingGapChange={handleCloseLeadingGapChange}
                importTempoMapEnabled={importTempoMapEnabled}
                onImportTempoMapEnabledChange={handleImportTempoMapEnabledChange}
                importTempoMapTempo={importTempoMapTempo}
                onImportTempoMapTempoChange={handleImportTempoMapTempoChange}
                importTempoMapTimeSignature={importTempoMapTimeSignature}
                onImportTempoMapTimeSignatureChange={handleImportTempoMapTimeSignatureChange}
                importTempoMapKeySignature={importTempoMapKeySignature}
                onImportTempoMapKeySignatureChange={handleImportTempoMapKeySignatureChange}
            />
            {ctxMenu && s.toolMode === "select" && (
                <EditContextMenu
                    x={ctxMenu.x}
                    y={ctxMenu.y}
                    isPitchParam={editParam === "pitch"}
                    onClose={() => setCtxMenu(null)}
                    onCopy={() => void handleEditOp("copy")}
                    onCut={() => void handleEditOp("cut")}
                    onPaste={() => void handleEditOp("paste")}
                    onSelectAll={() => void handleEditOp("selectAll")}
                    onDeselect={() => void handleEditOp("deselect")}
                    onInitialize={() => void handleEditOp("initialize")}
                    onTransposeCents={() => openEditDialog("transposeCents")}
                    onTransposeDegrees={() => openEditDialog("transposeDegrees")}
                    onSetPitch={() => openEditDialog("setPitch")}
                    onAverage={() => openEditDialog("average")}
                    onSmooth={() => openEditDialog("smooth")}
                    onAddVibrato={() => openVibratoDialog("apply")}
                    onQuantize={() => openEditDialog("quantize")}
                    onMeanQuantize={() => openEditDialog("meanQuantize")}
                    onSaveAsPitchRef={() => void handleSaveAsPitchRef()}
                    onExportMidi={() => void handleExportMidiFromEditor()}
                    // 音量 ↔ 动态 互转：参数本身就是这两个之一时始终可用。
                    //
                    // 【为什么不按"有无选区"门控】这里曾经只在有选区时显示（理由是
                    // "无选区时整条互换应由初始化表达"）。现在菜单里所有以选区为作用域
                    // 的操作都遵循同一条规则：**无选区时先隐式全选再执行**（见
                    // `handleEditOp` 开头），因此"整条互换"就是一次明确的全选 + 互换，
                    // 不再需要靠隐藏菜单项来回避。
                    isVolumeParam={
                        editParam === "volume" && planParamConversion(editParam) !== null
                    }
                    isDynParam={editParam === "dyn" && planParamConversion(editParam) !== null}
                    onConvertVolumeToDyn={() => void handleEditOp("convertVolumeToDyn")}
                    onConvertDynToVolume={() => void handleEditOp("convertDynToVolume")}
                />
            )}

            {/*
             * 颤音窗口：预设库 + 套用到选区，同一个窗口（右键菜单 / 快捷键 / 工具栏 /
             * 菜单栏共用这一个组件）。宿主在本面板是因为"套用预览"要选区真实帧值。
             *
             * 每次打开换 key → 重新挂载，草稿按当前活动预设播种、页签与纵轴复位。
             * 两副面孔由 `vibratoDialogMode` 决定：只有「添加颤音」那一次才把选区宿主
             * 交出去（于是才有套用页签 / 提取 / 应用）；「管理预设…」是纯改库，与合并
             * 前的预设管理器一致。
             */}
            <VibratoDialog
                key={vibratoDialogSession}
                open={vibratoDialogOpen}
                onOpenChange={setVibratoDialogOpen}
                editParam={editParam}
                paramRange={currentParamRange}
                initialPresetId={vibratoInitialPresetId ?? undefined}
                applyTarget={vibratoDialogMode === "apply" ? vibratoApplyTarget : undefined}
            />
        </>
    );
}
