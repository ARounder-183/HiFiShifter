/**
 * 参数编辑器的**编辑会话状态**：此刻选中了什么、正在画什么、手里拿着什么。
 *
 * 【为什么是独立的一个 hook】这一段原本夹在竖向滚动条换算与取数 hook 之间（模型的
 * 靠近模型前段）。它是一组**互不相干的名字，但同一个语义**：编辑会话的可变状态 ——
 * 多选区、变形预览叠加、画布光标、进行中的笔画与平移手势、参数线剪贴板槽位、以及
 * "笔画进行中"标志。它们全都只在**一次编辑会话内**有意义，也全都在面板卸载时一起
 * 消失。
 *
 * 【所有权不变】状态仍然属于面板（面板重挂载即归零，与拆分前一致），这里只是把
 * 声明搬进了子 hook，并由它交还给面板。hook 的调用点就是原来第一条语句的位置，
 * 因此内部两个 effect 的注册顺序与拆分前**逐条相同**。
 *
 * 【为什么 `liveEditActiveRef` 是带边沿回调的受控标志】写它的有十余处手势收尾分支，
 * 任何 `= false` 的写入都要补触发一次推迟的刷新 —— 做成受控标志后，不需要在十余处
 * 逐一记得调用（漏一处曲线就会停在旧数据上）。
 */
import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { setParamSelectionActive } from "../../../../features/session/sessionSlice";
import { normalizeSelection, type ParamSelection } from "../paramSelection";
import { createLiveEditFlag } from "../liveEditFlag";
import type { ParamClipboardData } from "../paramClipboardMapping";
import type {
    ParamMorphOverlay,
    ParamName,
    StrokeMode,
    StrokePoint,
    ValueViewport,
} from "../types";
import type { ModelDeps } from "./modelDeps";

type EditSessionInputs = Pick<ModelDeps, "dispatch" | "invalidate" | "s">;

/**
 * 编辑会话状态。
 *
 * 调用点必须固定在模型里原来的位置：这里注册的两个 effect（把选区存在性入仓、
 * 应用撤销/重做带回的选区）的相对顺序是行为的一部分。
 */
export function usePianoRollEditSession({ dispatch, invalidate, s }: EditSessionInputs) {
    /** 多选区（升序、互不相交、相邻已合并的**帧**区间列表；null = 无选区） */
    const selectionRef = useRef<ParamSelection | null>(null);
    const [selectionUi, setSelectionUi] = useState<ParamSelection | null>(null);
    // 参数线选区存在性入仓（session.paramSelectionActive）：复制/剪切按
    // "当前选中了什么"路由时以此判定参数侧（selectionRef 的每次变更都
    // 成对经过 setSelectionUi，见 usePianoRollInteractions）。拖拽期间的
    // 重复派发值不变，immer 判定无修改直接跳过。
    useEffect(() => {
        dispatch(setParamSelectionActive(selectionUi !== null));
    }, [dispatch, selectionUi]);
    // 撤销/重做/跳转恢复参数编辑器选区：请求由后端载荷驱动 —— 只有「边缘拉伸」
    // 这类同时改变选区的步骤才在后端记了快照（见 state.rs 的
    // `HistoryRecord::param_selection`），撤销/重做时随载荷带回；其余操作不带该
    // 字段，因此**不会**动用户手动调整过的选区。requestId 单调，按 id 幂等应用一次。
    const appliedParamSelectionRestoreRef = useRef(0);
    useEffect(() => {
        const request = s.pendingParamSelectionRestore;
        if (!request || request.requestId === appliedParamSelectionRestoreRef.current) return;
        appliedParamSelectionRestoreRef.current = request.requestId;
        const next = normalizeSelection(request.selection);
        selectionRef.current = next;
        setSelectionUi(next);
        invalidate();
    }, [invalidate, s.pendingParamSelectionRestore]);
    const [paramMorphOverlays, setParamMorphOverlays] = useState<ParamMorphOverlay[] | null>(null);
    const [canvasCursor, setCanvasCursor] = useState<CSSProperties["cursor"]>(
        s.toolMode === "select" ? "default" : "crosshair",
    );

    const strokeRef = useRef<{
        mode: StrokeMode;
        pointerId: number;
        param: ParamName;
        points: StrokePoint[];
    } | null>(null);

    const panRef = useRef<{
        pointerId: number;
        startClientX: number;
        startClientY: number;
        startScrollLeft: number;
        startView: ValueViewport;
        startRectH: number;
    } | null>(null);

    /**
     * 参数线剪贴板（多选区）：每段带**相对复制起点**的帧偏移，断层以偏移空洞
     * 的形式保留 —— 粘贴/预览按 paramClipboardMapping 的唯一映射规则求交。
     */
    const clipboardRef = useRef<ParamClipboardData | null>(null);

    /**
     * 笔画进行中标志（pointer down 期间为 true）。
     *
     * 读它的是两个取数 hook（`usePianoRollData` / `useLoudnessCurves`）：笔画期间的
     * 回包必须推迟落地，否则换掉 `paramView` 或响度快照都会打断进行中的笔画。
     *
     * 写它的有十余处手势收尾分支（见 `usePianoRollInteractions`），因此这里用
     * **带边沿回调**的受控标志：任何 `= false` 的写入都自动补触发一次推迟的刷新，
     * 不需要在十余处逐一记得调用 —— 漏一处就会让曲线停在旧数据上。
     */
    const liveEditEndHandlerRef = useRef<(() => void) | null>(null);
    // eslint-disable-next-line react-hooks/preserve-manual-memoization -- 这枚标志**刻意**只建一次并保持引用恒定（它进入两个取数 hook 的依赖链）；内部的 ref 读取是惰性的，不参与渲染期求值
    const liveEditActiveRef = useMemo(
        // eslint-disable-next-line react-hooks/refs -- 箭头函数只在"写入标志"时被调用，不在渲染期读取 ref
        () => createLiveEditFlag(() => liveEditEndHandlerRef.current?.()),
        [],
    );
    return {
        selectionRef,
        selectionUi,
        setSelectionUi,
        paramMorphOverlays,
        setParamMorphOverlays,
        canvasCursor,
        setCanvasCursor,
        strokeRef,
        panRef,
        clipboardRef,
        liveEditEndHandlerRef,
        liveEditActiveRef,
    };
}
