/**
 * 参数编辑器的**编辑命令**：全选、命令分发、以及剪贴板的键盘与外部同步接线。
 *
 * 【为什么是独立的一个 hook】这一段原本夹在交互回调与选区动作之间（模型的
 * 2354-2511 行），但它是一条完整的链路：命令从哪来（右键菜单 / MenuBar 事件 /
 * 键盘 / 宿主转发）、怎么被解析成一次参数写入（`runEditOp` 的路由表）、以及
 * 剪贴板槽位在"本面板复制"与"别的面板/记事本复制"之间如何保持一致。
 *
 * 【为什么 `handleEditOp` 要经 ref 转交】交互层（`usePianoRollInteractions`）在
 * 本 hook **之前**就构造好了，它拿到的是一枚稳定的 `stableEditAction`，真正的实现
 * 通过 `handleEditActionRef` 在渲染期刷新 —— 于是命令的实现可以随渲染更新，而
 * 交互层的回调引用保持恒定（否则每次渲染都会重建整条手势链）。
 *
 * 【所有权不变】选区、剪贴板、参数视图仍属于面板与取数 hook，这里只是把语句搬进
 * 子 hook：调用点、hook 顺序与依赖数组全部保持原位。
 */
import { useCallback, useEffect } from "react";
import type { RefObject } from "react";
import { createClipboardPreviewSync } from "../clipboardPreviewSync";
import { pitchDeltaToDegreeSteps } from "./sceneBuilders";
import { runEditOp } from "./editOperations";
import { selectionFromFrames } from "../paramSelection";
import { timeToFrame } from "../utils";
import { readSystemClipboardObject } from "../../../../utils/systemClipboard";
import type { ModelDeps } from "./modelDeps";

type PianoRollEditCommandInputs = Pick<
    ModelDeps,
    | "activeVibratoPresetId"
    | "bumpRefreshToken"
    | "clipboardRef"
    | "currentParamDefaultValue"
    | "currentParamQuantizeUnit"
    | "currentParamRange"
    | "dispatch"
    | "editParam"
    | "effectiveProjectScale"
    | "invalidate"
    | "midiDialogOpenParamsRef"
    | "midiDialogSourceRef"
    | "paramView"
    | "paramViewRef"
    | "pitchEnabled"
    | "processorParamsRef"
    | "projectScaleAtSec"
    | "resolvedVibratoPresets"
    | "resolveScaleFromToken"
    | "rootTrackId"
    | "s"
    | "scrollerRef"
    | "selectionRef"
    | "setClipboardGuid"
    | "setMidiDialogOpen"
    | "setMidiDialogSelection"
    | "setMidiPath"
    | "setParamView"
    | "setSelectionUi"
    | "store"
    | "vibratoBaseline"
> & {
    /** 可滚域时长（全选的右边界）。 */
    dynamicProjectSec: number;
    /** 交互层持有的稳定命令入口（渲染期刷新，见文件头）。 */
    handleEditActionRef: RefObject<(op: string) => void>;
    /** 打开颤音窗口（命令分发里的一条路由）。 */
    openVibratoDialog: (mode?: "apply" | "manage", initialPresetId?: string) => void;
};

/**
 * 编辑命令的分发与剪贴板接线。
 *
 * 调用点必须固定在模型里原来的位置：`handleEditActionRef.current` 在这里于渲染期
 * 刷新，而交互层在此之前已经取走了 `stableEditAction`。
 */
export function usePianoRollEditCommands({
    activeVibratoPresetId,
    bumpRefreshToken,
    clipboardRef,
    currentParamDefaultValue,
    currentParamQuantizeUnit,
    currentParamRange,
    dispatch,
    dynamicProjectSec,
    editParam,
    effectiveProjectScale,
    handleEditActionRef,
    invalidate,
    midiDialogOpenParamsRef,
    midiDialogSourceRef,
    openVibratoDialog,
    paramView,
    paramViewRef,
    pitchEnabled,
    processorParamsRef,
    projectScaleAtSec,
    resolvedVibratoPresets,
    resolveScaleFromToken,
    rootTrackId,
    s,
    scrollerRef,
    selectionRef,
    setClipboardGuid,
    setMidiDialogOpen,
    setMidiDialogSelection,
    setMidiPath,
    setParamView,
    setSelectionUi,
    store,
    vibratoBaseline,
}: PianoRollEditCommandInputs) {
    /**
     * 参数编辑器「全选」：把整条参数曲线（0 → 工程时长）设为选区。
     *
     * 独立成函数有两个用途：`selectAll` 菜单命令本身，以及 `handleEditOp` 开头的
     * **隐式全选**（无选区时先把作用域铺满整条曲线，再执行原本需要选区的操作）。
     *
     * @returns 是否真的设置了选区；工具模式不是「选择」时为 `false`（与菜单命令
     *   同一守卫：绘制 / 直线 / 颤音工具下不产生选区），调用方据此保持原行为。
     */
    const selectAllParamRange = useCallback((): boolean => {
        if (s.toolMode !== "select") return false;
        // 整条曲线 = 帧 [0, 工程末端)。**数据路径**直接按整数帧构造：全选的边界就是
        // 首帧与末帧，不该被"最近中点"再挪半帧（见 paramSelection 的两个构造器）。
        const totalFrames = Math.max(
            0,
            timeToFrame(dynamicProjectSec, paramView?.framePeriodMs ?? 5),
        );
        selectionRef.current = selectionFromFrames(0, totalFrames);
        setSelectionUi(selectionRef.current);
        invalidate();
        return true;
        // eslint-disable-next-line react-hooks/exhaustive-deps -- selectionRef 是面板的 useRef 结果、setSelectionUi 是面板 useState 的 setter（按构造稳定）
    }, [s.toolMode, dynamicProjectSec, paramView?.framePeriodMs, invalidate]);

    // ── Edit operation handler (shared by context menu + MenuBar events) ──
    const handleEditOp = useCallback(
        async (op: string, data?: Record<string, unknown>) =>
            runEditOp(
                {
                    activeVibratoPresetId,
                    bumpRefreshToken,
                    clipboardRef,
                    currentParamDefaultValue,
                    currentParamQuantizeUnit,
                    currentParamRange,
                    dispatch,
                    editParam,
                    effectiveProjectScale,
                    invalidate,
                    midiDialogOpenParamsRef,
                    midiDialogSourceRef,
                    paramView,
                    paramViewRef,
                    pitchDeltaToDegreeSteps,
                    pitchEnabled,
                    processorParamsRef,
                    projectScaleAtSec,
                    resolveScaleFromToken,
                    resolvedVibratoPresets,
                    rootTrackId,
                    s,
                    scrollerRef,
                    selectAllParamRange,
                    selectionRef,
                    setClipboardGuid,
                    setMidiDialogOpen,
                    setMidiDialogSelection,
                    setMidiPath,
                    setParamView,
                    setSelectionUi,
                    store,
                    vibratoBaseline,
                },
                op,
                data,
            ),
        [
            rootTrackId,
            editParam,
            s.tracks,
            paramView?.framePeriodMs,
            dynamicProjectSec,
            s.edgeSmoothnessPercent,
            effectiveProjectScale,
            projectScaleAtSec,
            resolveScaleFromToken,
            currentParamRange,
            currentParamDefaultValue,
            currentParamQuantizeUnit,
            pitchEnabled,
            bumpRefreshToken,
            invalidate,
            dispatch,
            selectAllParamRange,
            // 颤音预设：`case "addVibrato"` 读这两个值来决定"套用哪个预设"。
            // 漏掉它们会让闭包拿到旧列表，表现为"点了新预设却套用了旧的"。
            resolvedVibratoPresets,
            activeVibratoPresetId,
            vibratoBaseline,
        ],
    );

    // Keep the ref in sync so usePianoRollInteractions can dispatch edit ops
    // eslint-disable-next-line react-hooks/refs -- 渲染期刷新是**刻意**的：交互层在本 hook 之前就取走了稳定的 stableEditAction，命令实现必须随渲染更新（见文件头）
    handleEditActionRef.current = (op: string) => void handleEditOp(op);

    // Listen for edit operations dispatched from MenuBar / 全局路由
    // hifi:editOp 是参数编辑器专属通道（事件名即契约）：全局路由
    // （focusRouting.resolveEditOpRoute）按活动编辑表面把编辑操作定向派发
    // 到这里，消费者信任事件、不再自行判断焦点 —— 旧版在此重猜
    // activeElement / body 属性，与时间轴侧判断互相矛盾，正是复制/剪切/
    // 粘贴冲突的根因。selectAll/deselect 的工具模式守卫在 handleEditOp 内。
    useEffect(() => {
        const handler = (e: Event) => {
            const detail = (e as CustomEvent).detail;
            if (!detail?.op) return;
            const { op, ...data } = detail;
            void handleEditOp(op, data);
        };
        window.addEventListener("hifi:editOp", handler);
        return () => window.removeEventListener("hifi:editOp", handler);
    }, [handleEditOp]);

    /**
     * 快捷键「添加颤音」→ 打开颤音窗口（落在「套用到选区」页签）。
     *
     * `edit.addVibrato` 在交互层被归入"需要弹窗"的一类（派发 `hifi:openEditDialog`），
     * 但该事件由菜单栏消费，而窗口宿主在本面板（预览要选区数据）。这里补上本面板的
     * 消费分支，快捷键与右键菜单才走同一个入口。
     */
    useEffect(() => {
        const handler = (e: Event) => {
            if ((e as CustomEvent).detail?.dialog === "addVibrato") {
                openVibratoDialog("apply");
            }
        };
        window.addEventListener("hifi:openEditDialog", handler);
        return () => window.removeEventListener("hifi:openEditDialog", handler);
    }, [openVibratoDialog]);

    // 单剪贴板纪律：槽位被别的表面整体替换后（时间轴复制/剪切，或记事本暂存块
    // 的「恢复到剪贴板」），内部剪贴板缓存必须**重新对齐槽位** —— 否则
    // "复制 Clip 后在参数编辑器粘贴"会把更早复制、已被替换掉的参数线数据从
    // 内部缓存复活。
    //
    // 【只清不读是不够的】清空只解决了"槽位里不再是参数线数据"的一半；另一半
    // 是"槽位里（重新）是参数线数据"：从记事本恢复一份参数线载荷时，数据明明
    // 可粘贴，预览却始终空白 —— 而预览画的正是"粘贴会落下的数据"，它必须跟着
    // 槽位走。因此这里清空之后立刻回读一次。
    useEffect(() => {
        const sync = createClipboardPreviewSync(() => readSystemClipboardObject("param"));
        const handler = () => {
            // 先清空：待读取期间也绝不展示已经过期的曲线。读取到（或读不到）
            // 参数线数据后，结果统一由 apply 落地。
            clipboardRef.current = null;
            invalidate();
            void sync.sync((data) => {
                clipboardRef.current = data;
                invalidate();
            });
        };
        window.addEventListener("hifi:clipboardReplaced", handler);
        // 面板首次挂载时也对齐一次：槽位可能在面板被打开之前就已经恢复好了
        // （恢复动作发生在记事本，参数编辑器当时还没有挂载）。
        handler();
        return () => {
            sync.cancel();
            window.removeEventListener("hifi:clipboardReplaced", handler);
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps -- clipboardRef 是面板的 useRef 结果（按构造稳定）
    }, [invalidate]);
    return { selectAllParamRange, handleEditOp };
}
