/**
 * 以**选区为作用域**的面板动作：提取颤音预设、套用颤音、另存为音高参考、导出 MIDI。
 *
 * 【为什么是独立的一个 hook】这四个动作原本夹在编辑命令分发与工具图标之间（模型的
 * 2535-2642 行）。它们形状一致：都是把"当前选区 + 当前参数 + 根轨道"交给
 * `pitchReferenceOps` 里的实现，再由实现决定怎么读写；共同的前提也是同一条 ——
 * **没有选区时先做一次隐式全选**（作用域 = 整条参数曲线），与右键菜单里其它以选区为
 * 作用域的操作保持同一口径。
 *
 * 【所有权不变】`selectionRef` / `paramView` / `editParam` 仍然属于面板与取数 hook，
 * 这里只是把六个 `useCallback` / `useMemo` 的声明搬进子 hook：调用点、hook 顺序与
 * 依赖数组全部保持原位，因此渲染行为与拆分前逐字节相同。
 */
import { useCallback, useMemo } from "react";
import type { VibratoPreset } from "../../../../features/vibrato/vibratoTypes";
import {
    handleExportMidiFromEditor as handleExportMidiFromEditorImpl,
    handleExtractVibratoPreset as handleExtractVibratoPresetImpl,
    handleApplyVibrato as handleApplyVibratoImpl,
    handleSaveAsPitchRef as handleSaveAsPitchRefImpl,
    loadVibratoApplyOriginal as loadVibratoApplyOriginalImpl,
} from "./pitchReferenceOps";
import type { ModelDeps } from "./modelDeps";

type PianoRollSelectionActionsInputs = Pick<
    ModelDeps,
    | "currentParamRange"
    | "dispatch"
    | "editParam"
    | "handleEditOp"
    | "paramView"
    | "rootTrackId"
    | "s"
    | "selectAllParamRange"
    | "selectionRef"
    | "t"
>;

/**
 * 选区作用域的面板动作。
 *
 * 调用点必须固定在模型里原来的位置：`vibratoApplyTarget` 是交给颤音窗口的稳定对象，
 * 而它依赖另外三个回调的引用 —— 声明位置决定它们的身份，进而决定窗口里的 effect
 * 会不会每帧重跑。
 */
export function usePianoRollSelectionActions({
    currentParamRange,
    dispatch,
    editParam,
    handleEditOp,
    paramView,
    rootTrackId,
    s,
    selectAllParamRange,
    selectionRef,
    t,
}: PianoRollSelectionActionsInputs) {
    /**
     * 「从选区提取颤音预设」。
     *
     * 把用户手绘（或导入）的一段颤音拟合成预设：去趋势 → 自相关测周期 →
     * 折叠出单周期波形 → 测包络与不规则度。成功后入库、设为当前使用，并把这条
     * **返回给调用方**（颤音窗口随即选中它，用户在那里改名字、微调参数）。
     *
     * 【为什么返回预设而不是自己开窗口】合并之后提取就发生在颤音窗口里，宿主只负责
     * "读帧 + 拟合 + 入库"这三件只有它做得了的事；窗口怎么显示结果由窗口决定。
     * 失败返回 `null`，由窗口给出行内提示（弹一层模态会盖住刚刚那个窗口）。
     *
     * 【为什么用第一段选区】多种选区下"提取一个预设"的语义是模糊的；取第一段
     * 最符合直觉（也是 `runPerRange` 之外唯一不需要用户再做选择的解释）。
     */
    const handleExtractVibratoPreset = useCallback(
        () =>
            handleExtractVibratoPresetImpl({
                currentParamRange,
                dispatch,
                editParam,
                paramView,
                rootTrackId,
                selectAllParamRange,
                selectionRef,
                t,
            }),
        [
            rootTrackId,
            selectionRef,
            selectAllParamRange,
            paramView,
            editParam,
            currentParamRange,
            dispatch,
            t,
        ],
    );

    /**
     * 取颤音窗口「套用到选区」页签的预览数据：首个选区段的前 400 帧（≈2s）。
     *
     * 无选区时取曲线头部 —— 与 `handleEditOp` 的"隐式全选"口径一致（那里的作用域
     * 是整条曲线，这里只取开头一段来预览，避免为预览拉整条曲线）。400 帧足够看清
     * 若干周期与渐入，开销可忽略。
     */
    const loadVibratoApplyOriginal = useCallback(
        () => loadVibratoApplyOriginalImpl({ editParam, paramView, rootTrackId, selectionRef }),
        [rootTrackId, selectionRef, paramView, editParam],
    );

    /** 颤音窗口的「应用」：把完整预设交给编辑管线（本地微调随之传入）。 */
    const handleApplyVibrato = useCallback(
        (preset: VibratoPreset) => handleApplyVibratoImpl({ handleEditOp }, preset),
        [handleEditOp],
    );

    /**
     * 交给颤音窗口的选区宿主。
     *
     * 【为什么 memo】窗口用它判断"有没有选区上下文"（页签、页脚动作），并在取数据
     * 的 effect 里盯住 `loadOriginal`。每次渲染新建一个对象会让下游的比较全部失效 ——
     * 窗口里那句"依赖 `loadOriginal` 本身而不是这个对象"正是为了兜住这一层，但这里
     * 仍然应当稳定它：宿主传给子组件的对象不该每帧换新。
     */
    const vibratoApplyTarget = useMemo(
        () => ({
            loadOriginal: loadVibratoApplyOriginal,
            onApply: handleApplyVibrato,
            onExtract: handleExtractVibratoPreset,
        }),
        [loadVibratoApplyOriginal, handleApplyVibrato, handleExtractVibratoPreset],
    );

    /**
     * 「另存为音高参考」：每个选区段生成一个独立的 Pitch Ref clip
     * （不合并断层 —— 合并会把缺口处也填上参考音高）。
     *
     * 无选区时先做一次隐式全选（作用域 = 整条参数曲线），与右键菜单里其它
     * 以选区为作用域的操作一致 —— 否则这个菜单项点下去会毫无反应。
     */
    const handleSaveAsPitchRef = useCallback(
        () =>
            handleSaveAsPitchRefImpl({
                dispatch,
                paramView,
                rootTrackId,
                s,
                selectAllParamRange,
                selectionRef,
            }),
        [selectionRef, rootTrackId, paramView, s, dispatch, selectAllParamRange],
    );

    /**
     * 「导出 MIDI」：每个选区段作为一条独立的导出条目
     * （后端按 track 条目逐条导出，断层因此不会被填上音符）。
     */
    const handleExportMidiFromEditor = useCallback(
        () =>
            handleExportMidiFromEditorImpl({
                paramView,
                rootTrackId,
                s,
                selectAllParamRange,
                selectionRef,
            }),
        [rootTrackId, selectionRef, paramView, s, selectAllParamRange],
    );
    return {
        vibratoApplyTarget,
        handleSaveAsPitchRef,
        handleExportMidiFromEditor,
    };
}
