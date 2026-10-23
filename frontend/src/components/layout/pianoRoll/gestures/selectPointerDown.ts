import { startPointerGesture, type PointerGesture } from "./pointerGesture";
import { isModifierActive } from "../../../../features/keybindings/keybindingsSlice";
import { armRightDragContextMenuGuard } from "../../../../utils/rightDragContextMenuGuard";
import { clamp } from "../../timeline";
import {
    addPointerRange,
    rangeIndexAtFrame,
    removeRangeAtFrame,
    selectionFromPointers,
} from "../paramSelection";
import { frameFromViewportClientX } from "../seekPlayheadMapping";
import { timeToFrame } from "../utils";
import type { EdgeScrollDriver } from "../kernel/edgeScrollDriver";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { PointerDeps } from "./pointerDeps";
import { handleStretchSelectionEdge } from "./selection/stretchSelectionEdge";
import { handleSelectionEdgeDrag } from "./selection/selectionEdgeDrag";
import { handleSelectionPanDrag } from "./selection/selectionPanDrag";
import { handleParamLineDrag } from "./selection/paramLineDrag";
import { handleMorphPointerDown } from "./selection/morphDrag";

/** `handleCanvasPointerDown` 作用域里的局部量。 */
type PointerDownLocals = {
    edgeScroll: EdgeScrollDriver;
};

/** 选择工具起手所需的全部上下文。 */
type SelectPointerDeps = Pick<
    PointerDeps,
    | "applyDenseToLiveEdit"
    | "applyMorphOverlayPreview"
    | "axisFromRefs"
    | "blendDenseEdges"
    | "buildMorphDense"
    | "buildUploadSegments"
    | "bumpRefreshToken"
    | "canvasRef"
    | "clearActivePointerGestureEnd"
    | "createFineAdjustedPointerState"
    | "disposeFineAdjustedPointerState"
    | "dragDirection"
    | "dynamicProjectSec"
    | "edgeHalfSpanForIndices"
    | "edgeSmoothnessPercent"
    | "editParam"
    | "ensureLiveEditBase"
    | "fetchCommitBaseSource"
    | "findStretchSelectionEdge"
    | "framePeriodMs"
    | "frameToViewportPx"
    | "getFineAdjustedPointerPosition"
    | "installDragDirectionKeyCycler"
    | "invalidate"
    | "isEffectivePitchSnapActive"
    | "isPointerNearDraggableSelection"
    | "liveEditActiveRef"
    | "liveEditOverrideRef"
    | "morphDragRef"
    | "morphModifierDownRef"
    | "morphOverlayRef"
    | "onContextMenu"
    | "onCycleDragDirection"
    | "onParamCommitSucceeded"
    | "onParamValuePreviewChange"
    | "onPitchSnapGestureActiveChange"
    | "paramMultiSelectKb"
    | "paramStretchKb"
    | "paramValuePopupEnabled"
    | "paramViewRef"
    | "pitchSnapUnit"
    | "pointerSec"
    | "pointerValue"
    | "projectScale"
    | "requestWaveformRepaint"
    | "resetLiveEditPreview"
    | "rootTrackId"
    | "scaleAtSec"
    | "scrollerRef"
    | "selectionRef"
    | "setActivePointerGestureEnd"
    | "setCanvasCursor"
    | "setMorphOverlay"
    | "setParamView"
    | "updateSelectionUi"
    | "valueToY"
    | "viewSizeRef"
    | "pointerFrame"
> &
    PointerDownLocals;

/**
 * 处理选择工具的指针按下。
 *
 * @param deps hook 作用域 + 回调局部量的上下文。
 * @param e 触发本次手势的指针事件。
 * @returns `true` = 本次手势已消费、调用方应停止；`false` = 未消费，调用方继续。
 */
export function handleSelectPointerDown(
    deps: SelectPointerDeps,
    e: ReactPointerEvent<HTMLCanvasElement>,
): boolean {
    const {
        applyDenseToLiveEdit,
        applyMorphOverlayPreview,
        axisFromRefs,
        blendDenseEdges,
        buildMorphDense,
        buildUploadSegments,
        bumpRefreshToken,
        canvasRef,
        clearActivePointerGestureEnd,
        createFineAdjustedPointerState,
        disposeFineAdjustedPointerState,
        dragDirection,
        dynamicProjectSec,
        edgeHalfSpanForIndices,
        edgeScroll,
        edgeSmoothnessPercent,
        editParam,
        ensureLiveEditBase,
        fetchCommitBaseSource,
        findStretchSelectionEdge,
        framePeriodMs,
        frameToViewportPx,
        getFineAdjustedPointerPosition,
        installDragDirectionKeyCycler,
        invalidate,
        isEffectivePitchSnapActive,
        isPointerNearDraggableSelection,
        liveEditActiveRef,
        liveEditOverrideRef,
        morphDragRef,
        morphModifierDownRef,
        morphOverlayRef,
        onContextMenu,
        onCycleDragDirection,
        onParamCommitSucceeded,
        onParamValuePreviewChange,
        onPitchSnapGestureActiveChange,
        paramMultiSelectKb,
        paramStretchKb,
        paramValuePopupEnabled,
        paramViewRef,
        pitchSnapUnit,
        pointerSec,
        pointerValue,
        projectScale,
        requestWaveformRepaint,
        resetLiveEditPreview,
        rootTrackId,
        scaleAtSec,
        scrollerRef,
        selectionRef,
        setActivePointerGestureEnd,
        setCanvasCursor,
        setMorphOverlay,
        setParamView,
        updateSelectionUi,
        valueToY,
        viewSizeRef,
        pointerFrame,
    } = deps;

    if (e.button !== 0 && e.button !== 2) return true;

    const existingMorph = morphOverlayRef.current;
    const pvForMorph = paramViewRef.current;
    const canvas = canvasRef.current;
    if (existingMorph && existingMorph.length > 0 && pvForMorph && canvas) {
        if (
            handleMorphPointerDown(
                {
                    applyMorphOverlayPreview,
                    axisFromRefs,
                    buildMorphDense,
                    bumpRefreshToken,
                    canvas,
                    clearActivePointerGestureEnd,
                    createFineAdjustedPointerState,
                    disposeFineAdjustedPointerState,
                    editParam,
                    ensureLiveEditBase,
                    existingMorph,
                    getFineAdjustedPointerPosition,
                    liveEditActiveRef,
                    liveEditOverrideRef,
                    morphDragRef,
                    morphModifierDownRef,
                    morphOverlayRef,
                    onParamCommitSucceeded,
                    onParamValuePreviewChange,
                    paramValuePopupEnabled,
                    paramViewRef,
                    pointerFrame,
                    pointerValue,
                    pvForMorph,
                    rootTrackId,
                    setActivePointerGestureEnd,
                    setCanvasCursor,
                    setMorphOverlay,
                    setParamView,
                    valueToY,
                    viewSizeRef,
                },
                e,
            )
        ) {
            return true;
        }
    }

    const b = pointerFrame(e.clientX);
    const sel = selectionRef.current;

    // 选区构建的指针换算（普通框选、多选追加、边缘拖拽共用）。
    // 允许自动滚动时有 32px 边缘触发区。
    //
    // 【这里只产出**连续帧坐标**】"指针位置 → 选区边界"的两条规则（左 =
    // 最近的那一帧、右 = 所在帧之后）与"工程起点之前不可选"的钳制都在
    // `paramSelection` 里，由 `frameRangeFromPointers` /
    // `resizeFrameRangeEdge` 按边界选用；本函数只负责把指针换成帧坐标。
    //
    // 【只夹右端，**不夹左端**】右端夹到 `maxSelectableFrame - 1`：右边界是
    // "指针所在帧之后"（`floor(f) + 1`），指针落在最后一帧上时它正好等于工程
    // 末端的排他边界，再往右会把选区拖出工程。
    // 左端**照实保留负值**：同步到时间轴时左侧那段与轨道头等宽的额外空间是
    // 负时间，`frameRangeFromPointers` 要靠它判断"这次拖拽是否整段都在工程
    // 起点之前"（是则根本不产生选区）。若在这里夹成 0，那个判断就永远为假。
    const maxSelectableFrame = Math.max(0, timeToFrame(dynamicProjectSec, framePeriodMs));
    const clampSelectionFrame = (frame: number) =>
        Math.min(frame, Math.max(0, maxSelectableFrame - 1));
    const selectionFrameFromClientX = (clientX: number, allowAutoScroll: boolean) =>
        clampSelectionFrame(rawFrameFromClientX(clientX, allowAutoScroll));

    /**
     * 视口 x → **连续帧坐标**（不夹取）。自动滚动与投影都在这里。
     */
    function rawFrameFromClientX(clientX: number, allowAutoScroll: boolean): number {
        const scroller = scrollerRef.current;
        if (!scroller) {
            return pointerFrame(clientX);
        }

        // 本帧的交互投影：**与渲染侧同源**（内核真值，见 `axisFromRefs`）。
        const axis = axisFromRefs();
        const bounds = scroller.getBoundingClientRect();

        if (allowAutoScroll) {
            // 边缘自动滚动收口在 `edgeScroll` 驱动里（几何是
            // `shared/edgeAutoScroll` 的纯函数，驱动本身有单测）。
            // 这里只需"按当前指针位置滚一帧"：滚动量、可达上界、方向
            // 符号、帧时长归一都由它负责，本函数不再自己算一遍。
            //
            // 【为什么选择工具不启用持续滚动】它的语义是"指针到哪选到哪"，
            // 松手即定型 —— 停在边缘不动没有意义。持续滚动只服务绘制类
            // 工具（笔画必须跟着视图继续延伸）。
            edgeScroll.step(clientX);
        }

        const clampedClientX = clamp(clientX, bounds.left, bounds.right);
        // 视口 x → 帧：一律走统一投影的逆函数，**不再**用
        // `(scrollLeftRef + dx) / pxPerBeatRef` 这条独立算式。
        //
        // 【为什么这条算式是缺陷根因】它读的 `scrollLeftRef` 会被渲染期的
        // state→ref 同步覆盖回去，而 state 是 256px 步长的量化提交，滚动后
        // 最多滞后内核 255px——于是框选产生的选区整体偏移同一距离，而画面上
        // 选区块按内核绘制，两者对不上（用户报告"实际产生的选区与鼠标划定的
        // 区域不一致"，且先缩放后滚动才出现）。
        return frameFromViewportClientX({
            clientX: clampedClientX,
            viewportLeft: bounds.left,
            axis,
            framePeriodMs,
        });
    }

    // ── 选区上的手势优先级（高 → 低，唯一定义处）──────────────────
    //   0. Alt 形变控制点（上方已 return；控制点是画面上显式的小圆点，
    //      命中框 8px，比参数线还窄，且只在按住修饰键时出现 → 排最高）；
    //   1. **已选中的参数线**（落在某段内且靠近曲线）；
    //   2. Alt + 选区边缘 → 拉伸选区内的曲线（改数据、进撤销栈）；
    //   3. 多选区追加：⌘/Ctrl + 左键，**或右键落在未选中区域**
    //      （后者与前者等同；原地点击已有段仍是"取消该段"）；
    //   4. 无修饰键 + 边缘 → 只调整选区边界；
    //   5. 右键 + 已选中区域内部 → 左右平移**指针所在的那一段**（只动
    //      选区，不动数据；多段选区下其余段不动）；
    //   6. 普通框选（替换整个选区）。
    //
    // 【右键为什么分两种】右键落在**已选中区域**里 = 平移那一段（选区调整）；
    // 落在**未选中区域**里 = 追加一段（多选区操作）—— 平移对"空处"没有
    // 意义，而"再选一段"正是用户在那里起手时想做的事。
    //
    // 【为什么参数线永远第一】参数线的命中范围最窄（10px 纵向带 + 必须
    // 落在段内），而它在画面里可能只占很短一截，很难瞄准；选区边缘则是两条
    // 贯穿整个高度的明显竖线，用户想抓边缘时会自己去找那两条线。因此**所有
    // 选区自身的交互（边缘 / 多选 / 平移）都必须给参数线让路** —— 否则
    // "抓选区边界处那截曲线"会被选区交互永远吃掉。
    //
    // 下面每一处选区交互都显式让位（`!onSelectedCurve` / 各自的修饰键
    // 取反），因为这里的分支顺序本身就编码了优先级。
    const onSelectedCurve = isPointerNearDraggableSelection(
        e.clientX,
        e.clientY,
        e.nativeEvent.pointerType,
    );

    // ── 多选区追加：⌘/Ctrl + 左键，**或右键落在未选中区域** ─────────
    // 拖动 = 在已有选区上**追加**一段（并集；重叠/相接自动合并）；
    // 原地点击已有段 = 取消该段（与时间轴 ⌘+点击多选切换同源语义）。
    //
    // 【为什么右键也算多选】右键在**未选中区域**起手时，用户想做的通常
    // 就是"再选一段" —— 选区平移只对已选中区域有意义（见下一个分支），
    // 因此这里的右键与按住多选区修饰键**完全等同**。右键在已选中区域内
    // 起手仍是平移、压在参数线上仍是曲线交互，两者都不变。
    //
    // Alt 拉伸修饰键按下时本分支让位（见条件里的取反），否则在选区内侧
    // 边缘处无法拉伸；指针压住**已选中的参数线**时也让位，否则按住修饰键
    // 就再也拖不动选区内的曲线（而这正是该修饰键最不该挡住的交互）。
    const rightDragInUnselectedArea =
        e.button === 2 &&
        !isModifierActive(paramStretchKb, e.nativeEvent) &&
        rangeIndexAtFrame(sel, b) === -1;
    if (
        rightDragInUnselectedArea ||
        (e.button === 0 &&
            !onSelectedCurve &&
            !isModifierActive(paramStretchKb, e.nativeEvent) &&
            isModifierActive(paramMultiSelectKb, e.nativeEvent))
    ) {
        // 起手的指针位置：既用于"点在不在已有段里"，也作为新段的起点。
        const startFrame = selectionFrameFromClientX(e.clientX, false);
        const baseSelection = selectionRef.current ?? [];
        const hitIndex = rangeIndexAtFrame(baseSelection, startFrame);
        const hitExistingRange = hitIndex >= 0;
        const startClientX = e.clientX;
        let moved = false;
        // 起手用的是哪个键：拖动期间按这个掩码判断"键还按着"（左 1 / 右 2）。
        const pressMask = e.button === 2 ? 2 : 1;
        const isRightDrag = e.button === 2;

        // 右键手势的收尾菜单：拖拽期间一律吞掉，**没拖动**时按"右键点击"
        // 处理（手工打开菜单）—— 与选区平移 / 曲线形变两处右键分支同一套。
        const suppressContextMenu = (ev: Event) => {
            ev.preventDefault();
            ev.stopImmediatePropagation();
        };
        /** 手势句柄（onMove 需要主动收尾，见 pan 分支的同名字段）。 */
        let gesture: PointerGesture | null = null;
        const onMove = (ev: globalThis.PointerEvent) => {
            if ((ev.buttons & pressMask) !== pressMask) {
                gesture?.end("up", ev);
                return;
            }
            // 3px 死区：区分「点击切换」与「拖动追加」，避免手抖误删段
            if (!moved && Math.abs(ev.clientX - startClientX) > 3) {
                moved = true;
                // 确认构成拖拽：武装跨表面守卫，松手若落在别的表面上
                // （标尺 / 轨道头）也不会弹出那个表面的菜单。
                if (isRightDrag) armRightDragContextMenuGuard();
            }
            if (!moved) return;
            const bbFrame = selectionFrameFromClientX(ev.clientX, true);
            selectionRef.current = addPointerRange(baseSelection, startFrame, bbFrame);
            updateSelectionUi(selectionRef.current);
            invalidate();
        };
        const onEnd = (reason: "up" | "cancel", ev: globalThis.PointerEvent | null) => {
            // 「原地点击已有段 = 取消该段」只属于**修饰键**路径：右键起手的
            // 分支只在未选中区域触发，右键点击在那里是"交回菜单"（见下），
            // 不应借帧坐标的边界夹取误差误删某一段。
            // 【为什么要求 reason === "up"】pointercancel 意味着手势不可信（OS 打断 /
            // 指针被抢）。把"原地点击 = 取消该段"用在一条被取消的手势上，等于凭一个
            // 不可信的落点删掉用户的选区。
            if (reason === "up" && !moved && hitExistingRange && !isRightDrag) {
                // 切换取消：移除被点击的那一段
                // （removeRangeAtFrame 已覆盖"按点击帧定位该段"的语义）
                selectionRef.current = removeRangeAtFrame(selectionRef.current, startFrame);
                updateSelectionUi(selectionRef.current);
            }
            if (isRightDrag && !moved && reason === "up") {
                // 右键点击（没有拖动）：未选中区域起手时既没有可取消的段，
                // 也没有要追加的范围 —— 原样交回上下文菜单。
                if (onContextMenu && document.hasFocus() && ev) {
                    onContextMenu(ev.clientX, ev.clientY);
                }
            }
            invalidate();
        };
        gesture = startPointerGesture({
            pointerId: e.pointerId,
            captureTarget: e.currentTarget,
            onMove,
            onEnd,
            // 只有右键起手才吞菜单：左键分支不需要，也不该多装一个全局监听。
            extra: isRightDrag
                ? [{ type: "contextmenu", listener: suppressContextMenu, capture: true }]
                : [],
            registerEnd: setActivePointerGestureEnd,
            clearEnd: clearActivePointerGestureEnd,
        });
        return true;
    }

    // 如果已有选区，且鼠标在选区范围内且靠近曲线，则进入拖拽曲线模式
    if (sel) {
        // 本块内四种选区交互的取舍（总表见上方"手势优先级"）：
        // - Alt + 边缘 = **拉伸选区内的曲线**（改参数值、进撤销栈）；
        // - 无修饰键 + 边缘 = **只调整选区边界**（不碰曲线、不进撤销栈）；
        // - 右键 + 选区内部 = **左右平移整个选区**（同样只动选区）；
        // - 已选中的参数线 = 拖动曲线（优先级最高，见 `onSelectedCurve`）。
        //
        // 【"拉伸"与"调整选区"是两件事】用户选定一段后想把范围改大改小，
        // 最直觉的做法就是抓住边缘拖 —— 那时他做的是**选区调整**，不是
        // 拉伸内容：所以无修饰键的边缘拖拽只改边界、不重采样选区内的曲线
        // 值，也不写撤销历史（否则用户按一次撤销，看到的却是"什么都没变"）。
        // Alt 才是"拉伸"这个显式意图。
        //
        // 边缘手势都只作用于**被抓住的那一段**（其余段不动）；命中判定取
        // 所有段中最近的边缘。
        const stretchHit = onSelectedCurve ? null : findStretchSelectionEdge(e);
        if (stretchHit) {
            if (
                handleStretchSelectionEdge(
                    {
                        applyDenseToLiveEdit,
                        blendDenseEdges,
                        bumpRefreshToken,
                        clearActivePointerGestureEnd,
                        createFineAdjustedPointerState,
                        disposeFineAdjustedPointerState,
                        edgeHalfSpanForIndices,
                        edgeSmoothnessPercent,
                        editParam,
                        ensureLiveEditBase,
                        getFineAdjustedPointerPosition,
                        invalidate,
                        liveEditActiveRef,
                        liveEditOverrideRef,
                        onParamCommitSucceeded,
                        paramViewRef,
                        pointerFrame,
                        requestWaveformRepaint,
                        rootTrackId,
                        sel,
                        selectionRef,
                        setActivePointerGestureEnd,
                        setCanvasCursor,
                        setParamView,
                        stretchHit,
                        updateSelectionUi,
                    },
                    e,
                )
            ) {
                return true;
            }
        }

        // ── 边缘拖拽（无修饰键，左键）：只调整选区边界 ───────────────
        //
        // 与上面 Alt 拉伸的区别是**语义**：这里只挪动被抓住的那条边界，
        // 选区内已有的曲线值原样保留（不重采样、不写回后端），因此它既
        // 不需要 live 覆盖层、也不打撤销检查点 —— 撤销栈里应当只有"对参数
        // 的编辑"，把单纯的选区调整写进去会让"撤销"变成一次无变化的空操作。
        //
        // 让位规则：指针同时压住"已选中的参数线"时优先拖曲线（`onSelectedCurve`
        // 已在上方算好）；右键不参与本支（右键是"平移整个选区"，见下一段）。
        const selectionEdgeHit =
            onSelectedCurve || e.button !== 0
                ? null
                : findStretchSelectionEdge(e, { requireModifier: false });
        if (selectionEdgeHit) {
            if (
                handleSelectionEdgeDrag(
                    {
                        clearActivePointerGestureEnd,
                        frameToViewportPx,
                        invalidate,
                        sel,
                        selectionEdgeHit,
                        selectionFrameFromClientX,
                        selectionRef,
                        setActivePointerGestureEnd,
                        setCanvasCursor,
                        updateSelectionUi,
                    },
                    e,
                )
            ) {
                return true;
            }
        }

        // ── 右键拖拽选区内部：左右平移指针所在的那一段 ───────────────
        //
        // 与"边缘调整"（上一段）同属**选区自身的调整**：只平移边界，选区
        // 内的曲线值原样保留，不写后端、不进撤销栈。区别只是这里两条边界
        // **同时**跟着走（整段搬移），而不是只动被抓住的那一条。
        //
        // 【多段选区：只搬被抓住的那一段】平移的作用范围就是指针按下的那
        // 一段（`rangeIndexAtFrame` 定位），其余段纹丝不动 —— 与边缘调整
        // "只作用于被抓住的那一段"同一条规则。若整段一起搬，用户想只挪
        // 其中一段时会连带改掉全部，无法只调整局部。
        //
        // 让位规则（与全局优先级一致）：
        // - 指针压住已选中的参数线 → 由下面的曲线分支处理（右键在曲线上是
        //   形变/幅度变换，属"参数线交互"，优先级更高）；
        // - 形变修饰键按下 → 让位给 Alt 边缘拉伸（那段已在上方 return）。
        const shiftRangeIndex = rangeIndexAtFrame(sel, b);
        if (
            e.button === 2 &&
            !onSelectedCurve &&
            !isModifierActive(paramStretchKb, e.nativeEvent) &&
            shiftRangeIndex !== -1
        ) {
            if (
                handleSelectionPanDrag(
                    {
                        clearActivePointerGestureEnd,
                        invalidate,
                        maxSelectableFrame,
                        onContextMenu,
                        sel,
                        selectionFrameFromClientX,
                        selectionRef,
                        setActivePointerGestureEnd,
                        setCanvasCursor,
                        shiftRangeIndex,
                        updateSelectionUi,
                    },
                    e,
                )
            ) {
                return true;
            }
        }

        if (rangeIndexAtFrame(sel, b) !== -1) {
            if (
                handleParamLineDrag(
                    {
                        applyDenseToLiveEdit,
                        b,
                        buildUploadSegments,
                        bumpRefreshToken,
                        canvasRef,
                        clearActivePointerGestureEnd,
                        createFineAdjustedPointerState,
                        disposeFineAdjustedPointerState,
                        dragDirection,
                        edgeHalfSpanForIndices,
                        editParam,
                        ensureLiveEditBase,
                        fetchCommitBaseSource,
                        getFineAdjustedPointerPosition,
                        installDragDirectionKeyCycler,
                        invalidate,
                        isEffectivePitchSnapActive,
                        liveEditActiveRef,
                        liveEditOverrideRef,
                        onContextMenu,
                        onCycleDragDirection,
                        onParamCommitSucceeded,
                        onParamValuePreviewChange,
                        onPitchSnapGestureActiveChange,
                        paramValuePopupEnabled,
                        paramViewRef,
                        pitchSnapUnit,
                        pointerFrame,
                        pointerSec,
                        pointerValue,
                        projectScale,
                        requestWaveformRepaint,
                        resetLiveEditPreview,
                        rootTrackId,
                        scaleAtSec,
                        sel,
                        selectionRef,
                        setActivePointerGestureEnd,
                        setCanvasCursor,
                        setParamView,
                        updateSelectionUi,
                        valueToY,
                        viewSizeRef,
                    },
                    e,
                )
            ) {
                return true;
            }
        }
    }

    // 默认行为：仅左键创建新选区（**替换**整个选区为单段）；
    // 右键不应在 pointerdown 时清除选区。
    // 指针 → 帧坐标的换算 selectionFrameFromClientX 已在上方定义
    // （与多选追加共用）；"指针 → 边界"的两条规则在 `paramSelection`。
    if (e.button === 0) {
        const startFrame = selectionFrameFromClientX(e.clientX, false);
        // 【按下时**什么都不改**】按下只是"武装"一次可能的框选：用户可能
        // 只是长按（还没决定要拖），此时既不该亮出新选区，也不该动原选区。
        // 选区在**第一次真正拖动**时（越过下面的死区）才建出来。
        //
        // 【为什么不能"先写一段再等拖动覆盖"】那样长按时就会先亮出一段
        // 选区（哪怕只有一帧），而手还没动 —— 反馈与意图不符。
        //
        // 【为什么不需要在这里标脏】没有任何状态被改动：原选区框保持不动
        // 正是"按下还没发生什么"的正确画面。第一次拖动时 onMove 会赋**新
        // 对象**（主画布内容签名走引用比较，见 `mainCanvasSignature.ts`）
        // 并 invalidate，缓存随之失效。
        const entrySelection = selectionRef.current;
        const pid = e.pointerId;
        const finePointerState = createFineAdjustedPointerState(
            e.nativeEvent,
            e.currentTarget as HTMLCanvasElement,
        );
        // 区分"单击"与"框选"的死区（与多选追加分支同一阈值）。
        //
        // 【为什么必须区分】长按与手抖期间也会来 pointermove（1px 抖动、
        // 笔的压力漂移），不做死区判定的话这些抖动就会被当成"拖拽"而建出
        // 选区；松手时的收尾也靠它决定"这一次到底是单击还是框选"。
        const startClientX = e.clientX;
        let moved = false;
        /** 手势句柄（onMove 需要主动收尾，见 pan 分支的同名字段）。 */
        let gesture: PointerGesture | null = null;
        const onMove = (ev: globalThis.PointerEvent) => {
            if ((ev.buttons & 1) !== 1) {
                gesture?.end("up", ev);
                return;
            }
            if (!moved && Math.abs(ev.clientX - startClientX) > 3) moved = true;
            // 拖拽途中选区被外部清除（如 BackSpace / 取消选择）时不再续建。
            // 判据必须带上"按下时本来**有**选区"：本次拖拽正要创建第一个
            // 选区时 `selectionRef.current` 本来就是 null，只判 null 会把它
            // 误当成"被清除"，框选永远建不起来。
            if (entrySelection != null && selectionRef.current == null) return;
            if (!moved) return;
            const adjusted = getFineAdjustedPointerPosition(finePointerState, ev);
            const bbFrame = selectionFrameFromClientX(adjusted.clientX, true);
            // 整段拖拽都落在工程起点之前（左侧那段额外空间）→
            // `frameRangeFromPointers` 给出"无选区"：既不建新选区，也**不动**
            // 原选区（否则在空白处拖一下就会把用户的选区清掉）。
            const next = selectionFromPointers(startFrame, bbFrame);
            if (next == null) return;
            selectionRef.current = next;
            updateSelectionUi(selectionRef.current);
            invalidate(); // 实时重绘选区
        };
        const onEnd = () => {
            disposeFineAdjustedPointerState(finePointerState);
            if (!moved) {
                // 单击（含笔/触摸的抖动）：没有拖动就没有新选区，同时把
                // 按下前的那一段清掉 —— 与"单击空白处 = 取消选择"一致。
                selectionRef.current = null;
                updateSelectionUi(null);
            }
            invalidate();
        };
        gesture = startPointerGesture({
            pointerId: pid,
            captureTarget: e.currentTarget,
            onMove,
            onEnd,
            registerEnd: setActivePointerGestureEnd,
            clearEnd: clearActivePointerGestureEnd,
        });
        return true;
    }
    // 右键：不在 pointerdown 时清除或重建选区，交由 contextmenu 或右键拖拽逻辑处理
    if (e.button === 2) {
        return true;
    }

    return false;
}
