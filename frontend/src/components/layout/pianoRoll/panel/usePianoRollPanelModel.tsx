import React, {
    type CSSProperties,
    useCallback,
    useEffect,
    useLayoutEffect,
    useMemo,
    useRef,
    useState,
} from "react";
import { shallowEqual } from "react-redux";
import { useAppDispatch, useAppSelector, useAppStore } from "../../../../app/hooks";
import type { RootState } from "../../../../app/store";
import { useI18n } from "../../../../i18n/I18nProvider";
import {
    setEditParam,
    setEdgeSmoothnessPercent,
    setTrackStateRemote,
    cycleDragDirection,
    persistUiSettings,
    setParamSelectionActive,
} from "../../../../features/session/sessionSlice";
import { resolveRootTrackId } from "../../../../features/session/trackUtils";
import {
    buildPitchAlgoOptions,
    resolveEffectivePitchAlgo,
} from "../../../../features/tracks/pitchAlgoOptions";
import { useAppTheme } from "../../../../theme/AppThemeProvider";
import { getWaveformColors } from "../../../../theme/waveformColors";
import { isScaleKey } from "../../../../utils/musicalScales";
import {
    firstBinding,
    isModifierActive,
    isNoneBinding,
} from "../../../../features/keybindings/keybindingsSlice";
import {
    SEPARATION_PARAM_ID,
    firstGatedParamId,
    isSeparationEnabled,
} from "../../../../features/pitch/separationGate";
import { useNonPassiveWheel } from "../../../../utils/useNonPassiveWheel";
import type { ScaleLike } from "../../../../utils/musicalScales";
import { clamp } from "../../timeline";
import { effectiveScaleAtSec } from "../../../../utils/tempoMap";
import { publishPianoRollSelection } from "../../../../utils/pianoRollSelectionBus";
import { resolveTimelineScrollRange } from "../../timeline/runtime/timelineScrollRange";
import { TOOLBAR_MAX_TIER } from "../toolbarOverflow";
import { useToolbarOverflowTier } from "../useToolbarOverflowTier";
import { normalizeSelection, selectionBoundingSpan, type ParamSelection } from "../paramSelection";
import { type ParamClipboardData } from "../paramClipboardMapping";
import { usePianoRollData } from "../usePianoRollData";
import { makeLoudnessAmplitudeMap } from "../PianoRollWaveformSurface";
import { formatDbReadout, resolveParamAxisUnit, supportsParamAxisUnit } from "../paramAxisUnits";
import { useLoudnessCurves, type LoudnessSnapshot } from "../useLoudnessCurves";
import { createLiveOverrideReader, type LiveOverrideReader } from "../liveLoudnessOverride";
import { pianoRollViewportBus } from "../pianoRollViewportBus";
import { createLoudnessGeometryWarp, type LoudnessGeometryWarp } from "../loudnessGeometryWarp";
import {
    getClipGeometryPreviewOrigin,
    subscribeClipGeometryPreview,
    type ClipGeometrySnapshot,
} from "../../timeline/clipGeometryPreviewBus";
import { createRenderLoop, type RenderLoop } from "../../renderKernel/renderLoop.js";
import { createTimelineAxis, playheadLineLeftPx } from "../../renderKernel/timelineAxis.js";
import { usePianoRollInteractions } from "../usePianoRollInteractions";
import { resolveMenuMaxHeight } from "../../menuPlacement";
import {
    enabledVibratoPresets,
    resolveActiveVibratoPreset,
    resolveVibratoPresets,
    findVibratoPreset,
    systemVibratoPreset,
} from "../../../../features/vibrato/vibratoPresetList";
import { STRAIGHT_VIBRATO_PRESET_ID } from "../../../../features/vibrato/systemPresets";
import { vibratoPresetLabel } from "../../vibrato/vibratoDialogLogic";
import {
    buildChildPitchOffsetCentsParam,
    buildChildPitchOffsetDegreesParam,
    buildChildFormantOffsetCentsParam,
    childPitchOffsetValueToDisplay,
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
    isChildFormantOffsetCentsParam,
    isChildPitchOffsetParam,
} from "../childPitchOffsetParams";
import { computeAutoFollowScrollLeft } from "../../../../utils/autoFollowScroll";
import { readDevicePixelRatio } from "../../../../utils/devicePixelLine";
import { useVisualPlayhead } from "../../../../hooks/useVisualPlayhead";
import {
    subscribeDevicePixelRatio,
    useDevicePixelRatio,
} from "../../../../hooks/useDevicePixelRatio";
import { toggleSecondaryParamVisibility } from "../secondaryOverlaySelection";
import type {
    ParamMorphOverlay,
    ParamName,
    StrokeMode,
    StrokePoint,
    ValueViewport,
} from "../types";
import {
    selectKeybinding,
    selectKeybindings,
    selectMergedKeybindings,
} from "../../../../features/keybindings/keybindingsSlice";
import { clearPianoRollLoading, setPianoRollLoading } from "../../../../utils/pianoRollStatusBus";
import { createLiveEditFlag } from "../liveEditFlag";
import { parseCustomScaleToken } from "../../../../utils/scaleSelection";
import { type PianoRollKernelHost } from "../kernel/host/pianoRollKernelHost";
import type {
    MutablePianoRollKernelData,
    PianoRollGridSpec,
} from "../kernel/host/pianoRollKernelData";
import { resolvePanelRenderViewport } from "../kernel/viewportSource";
import { NOTE_NAMES_SHARP } from "./panelHelpers";
import { usePianoRollParamGates } from "./usePianoRollParamGates";
import { usePianoRollScrollerWiring } from "./usePianoRollScrollerWiring";
import { usePianoRollSelectionActions } from "./usePianoRollSelectionActions";
import { useTimelineDisplayControls } from "./useTimelineDisplayControls";
import { buildToolIcons } from "./toolIcons";
import { usePianoRollEditCommands } from "./usePianoRollEditCommands";
import { usePianoRollFrameAssembly } from "./usePianoRollFrameAssembly";
import { usePianoRollScrollKernel } from "./usePianoRollScrollKernel";
import { useMidiImport } from "./useMidiImport";
import {
    pitchDeltaToDegreeSteps,
    buildGridSpec as buildGridSpecImpl,
    valueToY as valueToYImpl,
    yToViewportT as yToViewportTImpl,
    yToValue as yToValueImpl,
    clampViewport as clampViewportImpl,
    getParamValueBoundsForScrollbar as getParamValueBoundsForScrollbarImpl,
    getCurrentViewportForScrollbar as getCurrentViewportForScrollbarImpl,
    syncVerticalScrollbarForViewport as syncVerticalScrollbarForViewportImpl,
} from "./sceneBuilders";
import { usePianoRollViewport } from "./usePianoRollViewport";
import { useProcessorParams } from "./useProcessorParams";

/**
 * 装配参数编辑器面板的模型。
 *
 * @param dockFormId 本窗体在停靠布局里的 id（判断是否与时间轴上下堆叠）。
 * @returns 面板视图层需要的全部状态、派生值与回调。
 */
export function usePianoRollPanelModel(dockFormId: string | undefined) {
    const dispatch = useAppDispatch();
    // 事件监听器内同步读取 session（如 selectClipParamRange 的 Clip 查找），
    // 避免闭包快照滞后。
    const store = useAppStore();
    const rafRef = useRef<number | null>(null);
    const visualPlayheadSecRef = useRef(0);
    // 视觉插值播放头读取器（与绘制同源）：缩放锚点必须使用它，不能用 33Hz
    // 轮询的 store 滞后值——播放中缩放以滞后值锚定会让播放头跳变 δ·Δpx。
    const getVisualPlayheadSec = useCallback(() => visualPlayheadSecRef.current, []);
    const rulerPlayheadLineRef = useRef<HTMLDivElement | null>(null);
    /** 标尺上的捕获区间带（几何由内核写，见宿主里的 captureBand）。 */
    const captureBandRef = useRef<HTMLDivElement | null>(null);
    const rulerPlayheadHeadRef = useRef<HTMLDivElement | null>(null);
    /**
     * 内核宿主的稳定引用。
     *
     * 【为什么声明在这里】`invalidate`（紧随其后）要把标脏转交宿主，
     * 因此必须在它之前声明。放在下方原来的位置会让 `invalidate` 的闭包引用一个
     * 尚未初始化的 `const`（TDZ）——首次调用即抛 ReferenceError。
     */
    const hostRef = useRef<PianoRollKernelHost | null>(null);
    const drawRef = useRef<() => void>(() => {});
    /**
     * 请求重绘（所有会改变画面的数据/状态变更都经此入口）。
     *
     * 【为什么要转交宿主】曲线层在 GL 上，画面由**两个**渲染循环驱动：
     * 面板自己的 rAF（Canvas2D 细节层）与内核宿主的 rAF（GL 层）。只调度前者会让
     * GL 层永远停留在旧内容上——典型症状是"曲线数据到了但屏幕上不出现，直到滚动
     * 一下才显示"。因此标脏一律交给宿主：宿主的帧提交里会回调 `onFrame`
     * → `applyScrollLayers` → `drawRef.current()`，Canvas2D 与 GL 同帧一起刷新。
     *
     * 特殊说明：`drawRef.current()` 自身不调用 `invalidate()`，所以这条链不会自激。
     * 宿主尚未创建（挂载前 / 卸载后）时退回面板自己的 rAF，避免丢失首次绘制。
     */
    const invalidate = useCallback(() => {
        const host = hostRef.current;
        if (host != null) {
            host.invalidate();
            return;
        }
        if (rafRef.current != null) return;
        rafRef.current = requestAnimationFrame(() => {
            rafRef.current = null;
            drawRef.current();
        });
    }, []);

    /**
     * 播放头专用标脏：只重绘 GL 叠加层，跳过网格 / 键盘 / 曲线层。
     *
     * 【为什么不复用 `invalidate`】播放头每帧前进时视口没变，曲线几何完全不变；
     * 走完整失效等于每个播放帧白做一次全部曲线几何的重建（本仓实测约 20 万顶点 /
     * 3 MB 分配 / 帧）。宿主侧有专门的轻量入口（与 `timelineKernelHost` 同一机制）。
     *
     * 【无宿主时退回完整路径】回退分支是面板自己的 rAF 画 Canvas2D，没有"只画
     * 叠加层"的粒度；那种情况本就罕见（宿主尚未创建），退回 `invalidate` 不会
     * 造成每帧浪费 —— 该分支下曲线也不走 GL。
     */
    const invalidatePlayhead = useCallback(() => {
        const host = hostRef.current;
        if (host != null) {
            host.invalidatePlayhead();
            return;
        }
        invalidate();
    }, [invalidate]);

    /**
     * DPR 变化必须重绘主画布。
     *
     * `rasterize` 会按新 dpr 重设 backing store 与 CSS 尺寸，但**没有任何东西**
     * 会在纯 dpr 变化时触发重绘：`ResizeObserver` 观察的是 CSS 布局盒（dpr 变了
     * 尺寸也没变），而本组件的 `viewSize` 也保持不变。缺这条订阅时，换显示器或改
     * 系统缩放后主画布会一直用旧 dpr 的光栅化结果（发虚），直到某次滚动才恢复。
     * 标脏转交宿主，与 GL 层同帧刷新（见 `invalidate` 的说明）。
     */
    React.useEffect(() => subscribeDevicePixelRatio(() => invalidate()), [invalidate]);

    /**
     * 标尺播放头元素（竖线 / 倒三角）的挂载回调。
     *
     * 【为什么挂载时要请求一帧】位置只由内核在帧提交里写，而元素可能在**播放头
     * 静止**时被重建（面板重挂载 / 视图切换）：新节点没有任何 `left`，等价于
     * `left: auto` ⇒ 落在静态位置（左缘 = 工程起始处）。此时若没有任何东西弄脏帧
     * （暂停且不滚动），内核不会提交，元素就**停在工程起始处不动**。元素一出现就
     * 请求一帧，写入器随即按当前视口定位它（去重键含元素身份，见
     * `createPlayheadElementWriter`）。
     */
    const attachRulerPlayheadLine = useCallback((element: HTMLDivElement | null) => {
        rulerPlayheadLineRef.current = element;
        if (element !== null) hostRef.current?.invalidate();
    }, []);
    const attachRulerPlayheadHead = useCallback((element: HTMLDivElement | null) => {
        rulerPlayheadHeadRef.current = element;
        if (element !== null) hostRef.current?.invalidate();
    }, []);
    const { t, tf, tVars } = useI18n();
    const s = useAppSelector((state: RootState) => state.session, shallowEqual);

    // 工程会话切换：强制视口总线按当前工程内容重绘一次（跨工程投影保留契约
    // 的强制点，见 utils/timelineViewportBus.invalidate 的说明）。
    const projectSessionPath = s.project.path;
    React.useEffect(() => {
        pianoRollViewportBus.invalidate();
    }, [projectSessionPath]);
    const effectiveProjectScale = useMemo<ScaleLike>(
        () =>
            s.project.useCustomScale && s.project.customScale
                ? s.project.customScale.notes
                : s.project.baseScale,
        [s.project.baseScale, s.project.customScale, s.project.useCustomScale],
    );
    /**
     * 某秒位置生效的“工程音阶”（受 Tempo Map 音阶变化点影响）。
     * 无 Tempo Map 音阶覆盖时即为工程音阶。
     */
    const projectScaleAtSec = useCallback(
        (sec: number): ScaleLike | undefined =>
            effectiveScaleAtSec(s.tempoMap, sec, effectiveProjectScale),
        [s.tempoMap, effectiveProjectScale],
    );
    /**
     * 将音阶 token 解析为 ScaleLike。
     * `__project__` 在提供 `atSec` 时按该时刻的 Tempo Map 生效音阶解析，
     * 否则使用工程音阶（用于全局场景）。
     */
    const resolveScaleFromToken = useCallback(
        (scaleToken: string, atSec?: number): ScaleLike => {
            if (scaleToken === "__project__") {
                return atSec != null ? (projectScaleAtSec(atSec) ?? "C") : effectiveProjectScale;
            }

            const customScaleId = parseCustomScaleToken(scaleToken);
            if (customScaleId) {
                const preset = s.customScalePresets.find((item) => item.id === customScaleId);
                if (preset) {
                    return preset.notes;
                }
            }

            return isScaleKey(scaleToken) ? scaleToken : "C";
        },
        [effectiveProjectScale, projectScaleAtSec, s.customScalePresets],
    );
    const editParam = s.editParam as ParamName;
    /**
     * `editParam` 的 ref 镜像。
     *
     * 【为什么需要】宿主在**挂载时创建一次**（长生命周期），其回调（`onFrame` /
     * `onScrollTopFrame`）里无法直接读到最新的 `editParam`——闭包捕获的是挂载时的
     * 值，切换参数后回调仍按旧参数换算视口，会出现"切了参数但竖向位置按上一个参数
     * 映射"的错位。ref 每次渲染刷新，回调现读即最新。
     */
    const editParamRef = useRef<ParamName>(editParam);
    editParamRef.current = editParam;
    // pitchSnapOpen 已在顶部工具栏 JSX 内声明和使用，无需重复声明
    // pianoRoll.copy/cut/paste 的复制/剪切/粘贴已由全局路由统一派发到
    // handleEditOp，本地不再需要键位匹配（见 useKeybindings/focusRouting）。
    const prVerticalZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.pianoRollVerticalZoom"),
    );
    const horizontalZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.horizontalZoom"),
    );
    const scrollHorizontalKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.scrollHorizontal"),
    );
    const scrollVerticalKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.scrollVertical"),
    );
    const scrollbarZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.scrollbarZoom"),
    );
    const pianoKeysVerticalScrollKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.pianoKeysVerticalScroll"),
    );
    const pianoKeysVerticalZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.pianoKeysVerticalZoom"),
    );
    const paramMorphKb = useAppSelector((state) => selectKeybinding(state, "modifier.paramMorph"));
    // 多选区修饰键（默认 ⌘/Ctrl）：按住拖动追加一段选区，按住点击取消该段
    const paramMultiSelectKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramMultiSelect"),
    );
    const paramFineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );
    // 边缘平滑度滑块的滚轮步进：React 17+ 的根容器 wheel 监听是 passive，
    // JSX onWheel 里的 preventDefault 无效（伴随干预警告），必须走原生
    // 非 passive 监听（与主画布滚轮路径同模式）。
    //
    // 【修饰键必须读 `e` 本身】这里拿到的是**原生** WheelEvent（见
    // `useNonPassiveWheel` 的说明），`ctrlKey/shiftKey/altKey` 就在它自己身上。
    // 曾经写的是 `e.nativeEvent`（那是 React 合成事件的取法），原生事件没有这个
    // 属性，取到 undefined 之后 `isModifierActive` 读 `ctrlKey` 直接抛异常 ——
    // 处理器在第一行就炸，后面的 dispatch 从未执行，表现为**滚轮完全无效**。
    const attachEdgeSmoothnessWheel = useNonPassiveWheel<HTMLInputElement>((e) => {
        e.preventDefault();
        const fine = isModifierActive(paramFineAdjustKb, e);
        const step = fine ? 1 : 5;
        const dir = e.deltaY < 0 ? 1 : -1;
        const next = clamp(Math.round(s.edgeSmoothnessPercent) + dir * step, 0, 100);
        dispatch(setEdgeSmoothnessPercent(next));
        void dispatch(persistUiSettings());
    });
    // 参数选区边缘拉伸的修饰键。与时间轴 clip 边缘的 `modifier.clipStretch`
    // **分离**（两个表面各自独立改绑），只用于选择工具下的选区边缘拉伸。
    const stretchKb = useAppSelector((state) => selectKeybinding(state, "modifier.paramStretch"));
    const vibratoAmplitudeAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.vibratoAmplitudeAdjust"),
    );
    const vibratoFrequencyAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.vibratoFrequencyAdjust"),
    );
    const vibratoDragAmplitudeIncreaseKb = useAppSelector((state) =>
        selectKeybinding(state, "pianoRoll.vibratoDragAmplitudeIncrease"),
    );
    const vibratoDragAmplitudeDecreaseKb = useAppSelector((state) =>
        selectKeybinding(state, "pianoRoll.vibratoDragAmplitudeDecrease"),
    );
    const vibratoDragFrequencyIncreaseKb = useAppSelector((state) =>
        selectKeybinding(state, "pianoRoll.vibratoDragFrequencyIncrease"),
    );
    const vibratoDragFrequencyDecreaseKb = useAppSelector((state) =>
        selectKeybinding(state, "pianoRoll.vibratoDragFrequencyDecrease"),
    );
    // 拖拽颤音时切换预设（默认 `,` / `.`）。只在拖拽期间生效 —— 这两个动作的
    // `scopedContext` 是 `pianoRollVibratoDrag`，全局派发器不会执行它们。
    const vibratoPresetPrevKb = useAppSelector((state) =>
        selectKeybinding(state, "pianoRoll.vibratoPresetPrev"),
    );
    const vibratoPresetNextKb = useAppSelector((state) =>
        selectKeybinding(state, "pianoRoll.vibratoPresetNext"),
    );
    const vibratoPresetUserList = useAppSelector((state) => state.session.vibratoPresets);
    const activeVibratoPresetId = useAppSelector((state) => state.session.activeVibratoPresetId);
    /** 「摆放方式」：添加颤音的参数（存在设置里，与预设无关）。 */
    const vibratoBaseline = useAppSelector((state) => state.session.vibratoBaseline);
    const disabledVibratoPresetIds = useAppSelector(
        (state) => state.session.disabledVibratoPresetIds,
    );
    const builtinVibratoPresetOrder = useAppSelector(
        (state) => state.session.builtinVibratoPresetOrder,
    );
    const resolvedVibratoPresets = useMemo(
        () => resolveVibratoPresets(vibratoPresetUserList, builtinVibratoPresetOrder).all,
        [vibratoPresetUserList, builtinVibratoPresetOrder],
    );
    /**
     * 可选（未被停用）的预设：工具栏下拉只列这些，拖拽中的循环切换也只在这些里绕。
     *
     * 停用名单之外的用途仍走全量 `resolvedVibratoPresets` —— HUD 要按 id 找到
     * 正在用的预设（它可能已被停用），「重置到直线」也要能找到直线预设。
     */
    const enabledVibratoPresetList = useMemo(
        () => enabledVibratoPresets(resolvedVibratoPresets, disabledVibratoPresetIds),
        [resolvedVibratoPresets, disabledVibratoPresetIds],
    );
    const activeVibratoPreset = useMemo(
        () => resolveActiveVibratoPreset(resolvedVibratoPresets, activeVibratoPresetId),
        [resolvedVibratoPresets, activeVibratoPresetId],
    );
    /**
     * 直线工具的固定起手预设。
     *
     * 从全量列表里找，**找不到就用出厂那份兜底**（`systemVibratoPreset` 自带构造）：
     * 直线工具不依赖用户是否停用了直线预设 —— 停用只影响"怎么挑预设"，不该让
     * 一个工具失去它唯一的工作方式。
     */
    const straightVibratoPreset = useMemo(
        () =>
            findVibratoPreset(resolvedVibratoPresets, STRAIGHT_VIBRATO_PRESET_ID) ??
            systemVibratoPreset("straight"),
        [resolvedVibratoPresets],
    );
    /**
     * 拖拽 HUD 的内容。`null` = 没在拖。
     *
     * 用一个本地 state 而不是 ref：HUD 要跟着切换预设 / 滚轮调参实时刷新。
     */
    const [vibratoDragHud, setVibratoDragHud] = useState<{
        presetId: string;
        depthCents: number;
        rateHz: number;
        adjusted: boolean;
        clientX: number;
        clientY: number;
    } | null>(null);
    /**
     * 颤音窗口是否打开。
     *
     * 参数编辑器自己持有这个状态（不走 `hifi:openEditDialog`）：那个通道是给
     * 「一次性参数对话框」用的，而颤音窗口是常驻的预设库，菜单栏的「管理预设…」
     * 与这里的右键 / 工具栏入口共用同一个对话框组件。
     */
    const [vibratoDialogOpen, setVibratoDialogOpen] = useState(false);
    /** 打开时要编辑哪一条；`null` = 按当前活动预设播种。 */
    const [vibratoInitialPresetId, setVibratoInitialPresetId] = useState<string | null>(null);
    /**
     * 这次打开是哪一面：`"apply"` = 「添加颤音」（带套用预览 / 提取 / 应用），
     * `"manage"` = 「管理预设…」（纯预设库）。
     *
     * 【为什么必须有它】窗口有两副面孔，而"从哪个入口进来的"只有宿主知道。窗口自己
     * 不该去猜"我是被打开发动机是什么"—— 于是宿主把结论（有没有选区宿主）直接告诉它：
     * `apply` 才传 `applyTarget`，`manage` 不传，窗口据此收起全部套用专属能力。
     */
    const [vibratoDialogMode, setVibratoDialogMode] = useState<"apply" | "manage">("manage");
    /**
     * 窗口的"会话号"：每次打开自增并作为组件的 `key`。
     *
     * 窗口借此**重新挂载** —— 草稿回到播种值、预览页签与两把纵轴复位、选区帧值重取。
     * 用 key 而不是 effect 内重置，是为了不在 effect 里同步 setState（那会级联渲染，
     * 也被 lint 禁止）。这一条替代了合并前"应用弹窗的会话号 + 管理器的播种闸"两套机制。
     */
    const [vibratoDialogSession, setVibratoDialogSession] = useState(0);

    /**
     * 打开颤音窗口。
     *
     * @param mode 哪一面（决定要不要给选区宿主，见 `vibratoDialogMode`）。
     * @param initialPresetId 打开时编辑哪一条；省略 = 当前活动预设。
     */
    const openVibratoDialog = useCallback(
        (mode: "apply" | "manage" = "manage", initialPresetId?: string) => {
            setVibratoDialogMode(mode);
            setVibratoInitialPresetId(initialPresetId ?? null);
            setVibratoDialogSession((session) => session + 1);
            setVibratoDialogOpen(true);
        },
        [],
    );

    /**
     * 工具菜单里「颤音预设」二级面板中**预设列表**的最大高度。
     *
     * 预设列表从独立的下拉搬进子菜单后，高度仍需按锚点在**本面板内**的位置算
     * （见 `resolveMenuMaxHeight`）：面板可停靠在窗口任意高度，按视口取上限会让
     * 列表伸出面板之外。锚点用绘制类型工具按钮（预设入口就在它的菜单里）。
     */
    const [presetFlyoutMaxHeight, setPresetFlyoutMaxHeight] = useState(320);
    // 拖动方向循环切换键：拖拽进行中按下可即时切换本次拖拽方向（触控板替代右键）。
    const cycleDragDirectionKb = useAppSelector((state) =>
        selectKeybindings(state, "pianoRoll.cycleDragDirection"),
    );
    // 工具菜单里三项各自的快捷键：把绑定显示在菜单上，用户不必去设置里查。
    const drawToolKb = useAppSelector((state) => selectKeybindings(state, "mode.drawTool"));
    const lineToolKb = useAppSelector((state) => selectKeybindings(state, "mode.lineTool"));
    const vibratoToolKb = useAppSelector((state) => selectKeybindings(state, "mode.vibratoTool"));
    const mergedKeybindings = useAppSelector(selectMergedKeybindings);
    // 是否按住切换吸附的修饰键（临时切换吸附时用于高亮显示）
    const [snapToggleHeld, setSnapToggleHeld] = useState(false);
    // 仅在参数编辑实际操作期间（选择拖拽/绘制）参与临时吸附视觉切换
    const [snapGestureActive, setSnapGestureActive] = useState(false);
    const [hoveredReferenceRootTrackId, setHoveredReferenceRootTrackId] = useState<string | null>(
        null,
    );

    useEffect(() => {
        // 修饰键手势只有一个绑定，取主绑定（见 types.ts 的 KeybindingMap）。
        const kb = firstBinding(mergedKeybindings["modifier.clipNoSnap"]);
        if (!kb || isNoneBinding(kb)) return;
        const onKey = (e: KeyboardEvent) => {
            const active = isModifierActive(kb, e);
            setSnapToggleHeld(active);
        };
        window.addEventListener("keydown", onKey as EventListener);
        window.addEventListener("keyup", onKey as EventListener);
        // also track blur to clear state
        const onBlur = () => setSnapToggleHeld(false);
        window.addEventListener("blur", onBlur);
        return () => {
            window.removeEventListener("keydown", onKey as EventListener);
            window.removeEventListener("keyup", onKey as EventListener);
            window.removeEventListener("blur", onBlur);
        };
    }, [mergedKeybindings]);
    const { mode: themeMode, fontFamily } = useAppTheme();
    const waveformColors = useMemo(() => getWaveformColors(themeMode, "piano-roll"), [themeMode]);

    const effectivePitchSnapVisual =
        snapGestureActive && snapToggleHeld ? !s.pitchSnapEnabled : s.pitchSnapEnabled;

    // 右键编辑菜单状态
    const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number } | null>(null);
    const [drawToolMenuOpen, setDrawToolMenuOpen] = useState(false);
    /**
     * 工具栏下拉的**锚点**与**面板** ref 必须分开。
     *
     * 【为什么】面板经 `AppAnchoredMenu` 挂到 `document.body` 下，不再是锚点的
     * DOM 后代 —— 用锚点的 ref 做 `contains` 判定会永远为假，于是"点菜单里的项"
     * 会被当成"点了外面"，菜单在选中前就被关掉。
     */
    const drawToolMenuAnchorRef = useRef<HTMLDivElement | null>(null);
    const drawToolMenuRef = useRef<HTMLDivElement | null>(null);
    const [pitchSnapMenuOpen, setPitchSnapMenuOpen] = useState(false);
    const pitchSnapMenuAnchorRef = useRef<HTMLDivElement | null>(null);
    const pitchSnapMenuRef = useRef<HTMLDivElement | null>(null);
    const [paramValuePreview, setParamValuePreview] = useState<{
        clientX: number;
        clientY: number;
        value: number;
        displayText?: string;
    } | null>(null);
    // 浮窗更新合帧（见 onParamValuePreviewChange 处的说明）：pointermove 在高刷
    // 设备上可达数百 Hz，逐事件 setState 会让整个面板以事件频率重渲。
    const paramValuePreviewRafRef = useRef<number | null>(null);
    const paramValuePreviewPendingRef = useRef<{
        payload: { clientX: number; clientY: number; value: number; displayText?: string } | null;
    } | null>(null);
    useEffect(() => {
        return () => {
            if (paramValuePreviewRafRef.current != null) {
                cancelAnimationFrame(paramValuePreviewRafRef.current);
                paramValuePreviewRafRef.current = null;
            }
            paramValuePreviewPendingRef.current = null;
        };
    }, []);
    /**
     * 纵轴标尺的悬浮读数（`弹出展示参数` 在左轴上的形态）。
     *
     * 【为什么与 `paramValuePreview` 分开】那个浮窗挂**曲线画布**的坐标系
     * （`canvasRef` 的 rect），本浮窗挂轴列（`axisWrapRef` 的 rect）。两者坐标系
     * 不同，混用会让浮窗横向偏出 56px 宽的轴列。
     */
    const [axisValuePreview, setAxisValuePreview] = useState<{
        clientX: number;
        clientY: number;
        text: string;
    } | null>(null);

    /**
     * 当前参数的纵轴展示单位（音量 / 动态支持倍率 ↔ dB 切换）。
     *
     * 一个值同时喂给三处：GL 刻度标签（经 `buildGridSpec`）、参数线悬浮浮窗
     * （`formatParamValuePreview`）、纵轴标尺悬浮浮窗。三者必须同源，否则同一位
     * 置会出现"刻度写 −6、浮窗写 0.501"的自相矛盾读数。
     */
    const editParamAxisUnit = resolveParamAxisUnit(s.paramAxisUnits, editParam);
    /** 当前参数的左轴是否可切换展示单位（决定光标与角标）。 */
    const axisUnitToggleAvailable = supportsParamAxisUnit(editParam);

    const formatParamValuePreview = useCallback(
        (value: number): string => {
            if (!Number.isFinite(value)) return "";
            if (editParam === "pitch") {
                const rounded = Math.round(value);
                const pitchClass = ((rounded % 12) + 12) % 12;
                const octave = Math.floor(rounded / 12) - 1;
                const noteName = `${NOTE_NAMES_SHARP[pitchClass]}${octave}`;
                const cents = Math.round((value - rounded) * 100);
                const signedCents = cents >= 0 ? `+${cents}` : `${cents}`;
                return `${noteName}${signedCents}`;
            }
            if (isChildPitchOffsetDegreesParam(editParam)) {
                const display = childPitchOffsetValueToDisplay(editParam, value);
                if (Math.abs(display) >= 100) return display.toFixed(1);
                if (Math.abs(display) >= 10) return display.toFixed(2);
                return display.toFixed(3);
            }
            // 音量 / 动态切到 dB 读法：1× = 0 dB（见 paramAxisUnits）。带 `dB` 后缀
            // 与倍率读数区分——浮窗有空间写清楚单位，用户才不会把 −6 误读成倍率。
            if (editParamAxisUnit === "db" && supportsParamAxisUnit(editParam)) {
                return formatDbReadout(value);
            }
            if (Math.abs(value) >= 100) return value.toFixed(1);
            if (Math.abs(value) >= 10) return value.toFixed(2);
            return value.toFixed(3);
        },
        [editParam, editParamAxisUnit],
    );

    const currentDrawTool = s.drawToolMode;
    const drawToolButtonTitle =
        currentDrawTool === "vibrato"
            ? // 颤音工具下把当前预设一并报出来：预设按钮已经并进右键菜单，
              // 工具按钮的 tooltip 是"接下来会画出什么"的最后一块可扫读信息。
              `${tf("vibrato_draw_tool")}: ${vibratoPresetLabel(activeVibratoPreset, tf)}`
            : currentDrawTool === "line"
              ? tf("line_draw_tool")
              : tf("draw_tool");
    const activeDragDirection =
        s.toolMode === "select"
            ? s.selectDragDirection
            : currentDrawTool === "draw"
              ? s.drawDragDirection
              : s.lineVibratoDragDirection;
    // 直线与颤音共用同一份拖动方向（`lineVibratoDragDirection`）—— 它们是同一个
    // "起点 → 终点"手势，拆成两份只会让"在直线下调过、切到颤音又不算"成为新的困惑。
    const activeDragDirectionTool =
        s.toolMode === "select"
            ? ("select" as const)
            : currentDrawTool === "draw"
              ? ("draw" as const)
              : ("vibrato" as const);

    useEffect(() => {
        if (!drawToolMenuOpen && !pitchSnapMenuOpen) return;
        const onPointerDown = (e: PointerEvent) => {
            const target = e.target as Node | null;
            if (drawToolMenuRef.current?.contains(target)) return;
            if (pitchSnapMenuRef.current?.contains(target)) return;
            setDrawToolMenuOpen(false);
            setPitchSnapMenuOpen(false);
        };
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                setDrawToolMenuOpen(false);
                setPitchSnapMenuOpen(false);
            }
        };
        window.addEventListener("pointerdown", onPointerDown, true);
        window.addEventListener("keydown", onKeyDown, true);
        return () => {
            window.removeEventListener("pointerdown", onPointerDown, true);
            window.removeEventListener("keydown", onKeyDown, true);
        };
    }, [drawToolMenuOpen, pitchSnapMenuOpen]);

    /** 量一次预设列表的最大高度（锚点 / 面板缺失时返回 null）。 */
    const measurePresetFlyoutMaxHeight = useCallback((): number | null => {
        const anchor = drawToolMenuAnchorRef.current;
        const container = paramEditorRef.current;
        if (!anchor || !container) return null;
        const anchorRect = anchor.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();
        return resolveMenuMaxHeight({
            anchorBottom: anchorRect.bottom,
            containerTop: containerRect.top,
            containerBottom: containerRect.bottom,
        });
    }, []);

    // 工具菜单开着时窗口缩放、或面板被拖动分隔条改变大小，都要重算预设列表的
    // 可用高度。setState 放在回调里（而不是 effect 体内同步调用），避免级联渲染。
    useEffect(() => {
        if (!drawToolMenuOpen) return;
        const update = () => {
            const maxHeight = measurePresetFlyoutMaxHeight();
            if (maxHeight != null) setPresetFlyoutMaxHeight(maxHeight);
        };
        update();
        window.addEventListener("resize", update);
        const observer = new ResizeObserver(update);
        if (paramEditorRef.current) observer.observe(paramEditorRef.current);
        return () => {
            window.removeEventListener("resize", update);
            observer.disconnect();
        };
    }, [drawToolMenuOpen, measurePresetFlyoutMaxHeight]);

    // ── MIDI 拖放到参数编辑器（文件浏览器拖拽 + Tauri 系统文件拖放）────────
    // 与“导入 MIDI”按钮同属参数编辑器场景：导入目标默认 Pitch Param，并持久化
    // 到 midiImportTargetParamEditor（与按钮共用同一设置项）。
    const paramEditorRef = useRef<HTMLDivElement | null>(null);

    const effectiveSelectedTrackId = useMemo(() => {
        if (s.selectedTrackId) return s.selectedTrackId;
        const clipId = s.selectedClipId;
        if (!clipId) return null;
        const clip = s.clips.find((c) => c.id === clipId);
        return clip?.trackId ?? null;
    }, [s.selectedTrackId, s.selectedClipId, s.clips]);

    const selectedTrack = useMemo(() => {
        if (!effectiveSelectedTrackId) return null;
        return s.tracks.find((track) => track.id === effectiveSelectedTrackId) ?? null;
    }, [effectiveSelectedTrackId, s.tracks]);

    const selectedIsChildTrack = Boolean(selectedTrack?.parentId);

    const childPitchOffsetCentsParam = useMemo(() => {
        if (!effectiveSelectedTrackId || !selectedIsChildTrack) return null;
        return buildChildPitchOffsetCentsParam(effectiveSelectedTrackId);
    }, [effectiveSelectedTrackId, selectedIsChildTrack]);

    const childPitchOffsetDegreesParam = useMemo(() => {
        if (!effectiveSelectedTrackId || !selectedIsChildTrack) return null;
        return buildChildPitchOffsetDegreesParam(effectiveSelectedTrackId);
    }, [effectiveSelectedTrackId, selectedIsChildTrack]);
    // 视口（滚动 / 缩放 / 与时间轴同步）整块在 ``usePianoRollViewport`` 里。调用点必须留
    // 在原位：hook 顺序、以及 `useLayoutEffect` 的提交时机（"缩放先提交 state，再由
    // layout effect 落地"）都是行为的一部分。
    //
    // `commitViewportNow` 在下方以**函数声明**定义（hoisted），所以这里能提前把它交出去；
    // 它只在事件期被调用，不存在渲染期读到未初始化绑定的问题。
    const {
        scrollLeft,
        setScrollLeft,
        pxPerSec,
        pxPerBeat,
        dynamicProjectSec,
        scrollLeftRef,
        pxPerSecRef,
        zoomRafRef,
        paramEditorSyncTimelineRef,
        timelineOffsetRef,
        timelineOffsetPx,
        timelineSyncApplyingRef,
        horizontalZoomChainRef,
        lastScrollLeftRef,
        scrollerRef,
        viewSizeRef,
        getScrollLeftLive,
        contentWidthAt,
        handleHorizontalZoom,
    } = usePianoRollViewport({
        s,
        hostRef,
        dockFormId,
        getVisualPlayheadSec,
        commitViewportNow,
    });
    // 副参数独立显示开关，默认全部关闭
    const [secondaryParamVisible, setSecondaryParamVisible] = useState<
        Partial<Record<ParamName, boolean>>
    >({});

    const toggleSecondaryParam = useCallback((param: ParamName) => {
        setSecondaryParamVisible((prev) => toggleSecondaryParamVisibility(prev, param));
    }, []);

    const pitchViewRef = useRef<ValueViewport>({
        center: 72,
        span: 24,
    });
    const setPitchView = useCallback(
        (next: ValueViewport) => {
            pitchViewRef.current = next;
            syncVerticalScrollbarForViewport("pitch", next);
            // 值域视口是**竖轴的手势提交点**：滚动条 thumb 已同步写下 DOM，曲线 /
            // 音高键 / 数值轴（GL）必须在**同一任务**里提交，否则竖轴平移/缩放时
            // 这些线比滚动条慢一帧到几帧（见 `paintNow`）。
            paintNow();
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [invalidate],
    );

    const paramViewsRef = useRef<Record<string, ValueViewport>>({});
    const setParamViewport = useCallback(
        (param: string, next: ValueViewport) => {
            paramViewsRef.current = { ...paramViewsRef.current, [param]: next };
            syncVerticalScrollbarForViewport(param as ParamName, next);
            // 同 `setPitchView`：竖轴的手势提交点，DOM 与 GL 必须同任务。
            paintNow();
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [invalidate],
    );

    const rootTrackId = useMemo(() => {
        return resolveRootTrackId(s.tracks, effectiveSelectedTrackId);
    }, [effectiveSelectedTrackId, s.tracks]);

    const rootTrack = useMemo(() => {
        if (!rootTrackId) return null;
        return s.tracks.find((tr) => tr.id === rootTrackId) ?? null;
    }, [s.tracks, rootTrackId]);

    // 算法下拉的选项与取值：唯一来源（见 pitchAlgoOptions.ts）。不可用时
    // vslib 从列表消失；当前值已是 vslib 时保留并标注"不可用"，避免下拉框
    // 回退显示成 nsf-hifigan 而谎报轨道真实算法。
    const pitchAlgoOptions = useMemo(
        () =>
            buildPitchAlgoOptions({
                noneLabel: t("common_none"),
                vslibAvailable: s.vslibAvailable,
                formatUnavailable: (label) => tVars("algo_unavailable_label", { name: label }),
                currentValue: rootTrack?.pitchAnalysisAlgo,
            }),
        [t, tVars, s.vslibAvailable, rootTrack?.pitchAnalysisAlgo],
    );

    const childFormantOffsetParam = useMemo(() => {
        if (!effectiveSelectedTrackId || !selectedIsChildTrack) return null;
        // 归一后再判：未识别的算法在后端按默认算法（nsf-hifigan）渲染，而
        // nsf-hifigan 链路是支持子轨道共振峰偏移的（见
        // `does_clip_need_processor_render` 的 `has_child_formant_offset`）——
        // 用原始字符串判断会把这个控件藏掉，功能在后端生效、界面却给不了入口。
        const algo = resolveEffectivePitchAlgo(rootTrack?.pitchAnalysisAlgo);
        if (algo !== "nsf_hifigan_onnx" && algo !== "vslib") return null;
        return buildChildFormantOffsetCentsParam(effectiveSelectedTrackId);
    }, [effectiveSelectedTrackId, selectedIsChildTrack, rootTrack?.pitchAnalysisAlgo]);

    const pitchGroupActive =
        editParam === "pitch" ||
        editParam === childPitchOffsetCentsParam ||
        editParam === childPitchOffsetDegreesParam;
    // 工具栏上的音高组按钮使用简写（非中文语系为 PIT），完整名称放入 ToolTip。
    const pitchGroupLabel =
        editParam === childPitchOffsetCentsParam
            ? t("child_pitch_mode_cents")
            : editParam === childPitchOffsetDegreesParam
              ? t("child_pitch_mode_degrees")
              : t("param_btn_pitch");
    const pitchGroupTooltip =
        editParam === childPitchOffsetCentsParam
            ? t("child_pitch_offset_cents_label")
            : editParam === childPitchOffsetDegreesParam
              ? t("child_pitch_offset_degrees_label")
              : t("common_pitch");

    // 声码器参数描述符 + 当前参数元数据 + 三个标签映射。
    const {
        processorParams,
        processorParamsRef,
        processorStaticParams,
        processorStaticValues,
        orderedProcessorParams,
        currentParamRange,
        currentParamDefaultValue,
        currentParamQuantizeUnit,
        handleStaticParamChange,
        getProcessorParamLabel,
        getProcessorParamShortLabel,
        getStaticOptionLabel,
    } = useProcessorParams({
        rootTrack,
        rootTrackId,
        editParam,
        paramViewsRef,
        invalidate,
        t,
    });

    // 当 processorParams 变化时，若 editParam 不在可用集合内，自动回退到 pitch
    useEffect(() => {
        const available = new Set([
            "pitch",
            ...processorParams.map((p) => p.id),
            ...(childPitchOffsetCentsParam ? [childPitchOffsetCentsParam] : []),
            ...(childPitchOffsetDegreesParam ? [childPitchOffsetDegreesParam] : []),
            ...(childFormantOffsetParam ? [childFormantOffsetParam] : []),
        ]);
        if (isChildPitchOffsetParam(editParam)) {
            if (!selectedIsChildTrack || !effectiveSelectedTrackId) {
                dispatch(setEditParam("pitch"));
                return;
            }
            if (isChildPitchOffsetCentsParam(editParam)) {
                const expected = buildChildPitchOffsetCentsParam(effectiveSelectedTrackId);
                if (editParam !== expected) {
                    dispatch(setEditParam(expected));
                    return;
                }
            }
            if (isChildPitchOffsetDegreesParam(editParam)) {
                const expected = buildChildPitchOffsetDegreesParam(effectiveSelectedTrackId);
                if (editParam !== expected) {
                    dispatch(setEditParam(expected));
                    return;
                }
            }
            if (isChildFormantOffsetCentsParam(editParam)) {
                const expected = buildChildFormantOffsetCentsParam(effectiveSelectedTrackId);
                if (editParam !== expected) {
                    dispatch(setEditParam(expected));
                    return;
                }
            }
        }

        if (!available.has(editParam)) {
            dispatch(setEditParam("pitch"));
        }
    }, [
        processorParams,
        editParam,
        dispatch,
        childPitchOffsetCentsParam,
        childPitchOffsetDegreesParam,
        childFormantOffsetParam,
        effectiveSelectedTrackId,
        selectedIsChildTrack,
    ]);

    // ── 气声分离开关状态 ────────────────────────────────────────────────────
    //
    // 开关值为 1 表示"做谐波/噪声分离"。它是气声音量与张力共同的前提：
    // 关闭时两者置灰、不可编辑，且不参与合成（后端在 `ClipProcessContext`
    // 构造点剥离这两条曲线，因此也不会走 HNSEP）。
    // 阈值语义见 `separationGate.ts`（与后端 `extra_param_enabled` 同口径）。
    //
    // `separationDesc` 同时决定开关**是否存在**：该描述符只挂在 nsf-hifigan 的
    // 描述符表里（见 `renderer/chain.rs` 的 `HIFIGAN_PARAM_DESCRIPTORS`），
    // 因此 world / vslib 下开关不渲染 —— 那两种算法本就没有气声 / 张力参数。
    const separationDesc = useMemo(
        () => processorStaticParams.find((sp) => sp.id === SEPARATION_PARAM_ID) ?? null,
        [processorStaticParams],
    );
    const separationEnabled = useMemo(() => {
        const defaultValue =
            separationDesc && separationDesc.kind.type === "static_enum"
                ? separationDesc.kind.default_value
                : 0;
        return isSeparationEnabled(processorStaticValues[SEPARATION_PARAM_ID], defaultValue);
    }, [separationDesc, processorStaticValues]);

    // 分离开关在工具栏上的落点：被门禁组（气声 + 张力）的**组首**。
    //
    // 【为什么是组首而不是"算法下拉之前"】开关只随 nsf-hifigan 出现；若把它放在
    // 工具栏最右端（算法下拉左侧），切换算法时它整块消失 / 出现，会把音量、声像、
    // 算法选项一起推来推去。放在参数组内部后，它右侧紧邻的就是固定的
    // 「音量 → 动态 → 声像」，消失时右侧纹丝不动。
    //
    // 位置由 `firstGatedParamId` 从门禁列表推导（见该函数说明），不写死具体 id。
    const separationSwitchAnchorParamId = useMemo(
        () => (separationDesc ? firstGatedParamId(orderedProcessorParams.map((p) => p.id)) : null),
        [separationDesc, orderedProcessorParams],
    );

    // ── 工具栏横向不足时的分级隐藏 ──────────────────────────────────────────
    //
    // 参数编辑器工具栏是**一整行**（标题 / 工具 / 平滑度在左，参数药丸 / 算法在右，
    // 同属 `h-qt-bar-main` 这一个容器），因此只观察这一个容器：任一处放不下即升级。
    // 完整层级见 `toolbarOverflow.ts` 的 `TOOLBAR_MAX_TIER`：
    //   1 标题 · 2 算法文本 · 3 张力药丸 · 4 气声药丸（开关保留）· 5 平滑度文本 ·
    //   6 平滑度数值 · 7 平滑度滑块 · 8 导入 MIDI · 9 参考轨道组文本
    // 原则是**只砍冗余、不砍入口**：被隐藏项要么是纯装饰文字（旁边控件已表达同一信息），
    // 要么功能另有入口；唯一入口（如参考轨道组、分离开关）只瘦身、不隐藏。
    // 平滑度滑块是**控件**、数值只是读数，故数值（6）先于滑块（7）让位。
    const toolbarRowRef = useRef<HTMLDivElement | null>(null);
    const toolbarTier = useToolbarOverflowTier(toolbarRowRef, TOOLBAR_MAX_TIER);

    // 门禁与可见性（Compose / 分离两道门禁、药丸让位、参考轨道与副参数的眼睛）
    // 整块在 `usePianoRollParamGates` 里（见该文件头）。调用点留在原位：它内部注册的
    // 两个 effect 的相对顺序是行为的一部分。
    const {
        gatedPillHideOrder,
        gatedPillsHidden,
        composeEnabled,
        groupTrackIds,
        referenceRootTrackOptions,
        visibleReferenceRootTrackIds,
        pitchHardDisableReason,
        childPitchHardDisableReason,
        pitchEnabled,
        visibleSecondaryParamIds,
        updateVisibleReferenceRootTrackIds,
    } = usePianoRollParamGates({
        dispatch,
        editParam,
        s,
        rootTrackId,
        rootTrack,
        orderedProcessorParams,
        processorParams,
        separationEnabled,
        toolbarTier,
        secondaryParamVisible,
        setSecondaryParamVisible,
        hoveredReferenceRootTrackId,
        setHoveredReferenceRootTrackId,
        t,
    });

    // 【已删除 `secPerBeat = 60 / bpm`】参数编辑器的选区、剪贴板映射、交互换算、
    // 取数窗口全部改为**帧制**（工程级常量栅格，见 `paramSelection.ts`），BPM 只
    // 剩下一个消费者：网格与标尺（`buildTimelineTicks` 直接读 `s.bpm` + Tempo Map）。
    // 因此这里不再需要"每拍秒数"，改 BPM 也不会再牵动选区或触发曲线重取。
    const contentWidth = contentWidthAt(pxPerSec);

    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const axisCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const axisWrapRef = useRef<HTMLDivElement | null>(null);
    const scrollStateRafRef = useRef<number | null>(null);
    /**
     * **上一次镜像写入 scroller 的原生偏移**（两轴各自记录）。
     *
     * 用途只有一个：判定原生 `scroll` 事件是不是镜像回写造成的**回声**
     * （见 `timeline/scrollEcho` 的 `isMirrorEcho`）。判据必须拿"我上次写下的值"做
     * 基准，而不是拿内核的当前值——连续滚动 / 缩放时事件报的是**上一帧**镜像的值，
     * 与内核此刻已差一整帧位移，用它判会稳定地把回声误收成用户输入，进而把量化误差
     * 推回共享视口，让时间轴跟着抽动。
     *
     * 初值 `NaN` = 从未写过 → 一律判"不是回声"（宁可多采纳一次也不吞掉首帧前的输入）。
     */
    const lastMirroredScrollLeftRef = useRef(Number.NaN);
    const lastMirroredScrollTopRef = useRef(Number.NaN);

    const rulerContentRef = useRef<HTMLDivElement | null>(null);
    const gridLayerRef = useRef<HTMLDivElement | null>(null);

    // ── 渲染内核（滚动 / 视口所有权）────────────────────────────────────
    //
    // 【所有权在谁手里】内核是唯一渲染路径，宿主持有滚动真值，**原生 scroller 只是
    // 被动镜像**——宿主每帧把真值写回它，使尚未迁移的输入代码（中键拖拽、框选自动
    // 滚动等直接读写 `scroller.scrollLeft` 的地方）读到的仍是同一份位置，行为不变。
    // 各图层由宿主的帧回调 `applyScrollLayers` 驱动（取代此前的 `scroll` 事件）。
    //
    /**
     * 内核数据镜像（每次 render 更新字段，宿主每帧现读）。
     *
     * 特殊说明：显式标注为 `PianoRollKernelData` 而不是让它靠初始值推断——推断出的
     * 窄类型会让后续写入 `grid`（可选字段）报错，也会漏掉字段名拼写错误。
     */
    const kernelDataRef = useRef<MutablePianoRollKernelData>({
        projectSec: 1,
        valueDomain: { min: 0, max: 1, span: 1 },
        grid: null,
        overlay: null,
        selectionBand: null,
    });
    /** 自绘滚动条的 thumb（恒挂载）。 */
    const hScrollbarThumbRef = useRef<HTMLDivElement | null>(null);
    const vScrollbarThumbRef = useRef<HTMLDivElement | null>(null);
    /** 自绘滚动条的**轨道**（承接「点空白翻页」，恒挂载）。 */
    const hScrollbarTrackRef = useRef<HTMLDivElement | null>(null);
    const vScrollbarTrackRef = useRef<HTMLDivElement | null>(null);
    /** GL 静态层画布：网格 / 键盘几何，恒挂载（它是这些图层的唯一绘制者）。 */
    const glCanvasRef = useRef<HTMLCanvasElement | null>(null);
    /** 键盘轴 GL 画布（阶段 2/3）：键盘几何 + 音名标签。独立画布：轴列不随横向滚动移动。 */
    const glAxisCanvasRef = useRef<HTMLCanvasElement | null>(null);
    /** 动态叠加层 GL 画布（阶段 2/3）：播放头，位于曲线层之上，恒挂载。 */
    const glOverlayCanvasRef = useRef<HTMLCanvasElement | null>(null);

    const [viewSize, setViewSize] = useState({ w: 1, h: 1 });
    /**
     * 当前 dpr（React 形态）：仅供**渲染期内联样式**里的物理像素线宽使用。
     * 命令式绘制路径每帧现读 `readDevicePixelRatio()`，不经此值。
     */
    const devicePixelRatio = useDevicePixelRatio();
    const [timeDisplaySettingsOpen, setTimeDisplaySettingsOpen] = useState(false);
    // 参数编辑器的内容绘制在 sticky 视口层中，滚动范围由后面的 spacer 提供。
    // 两个子元素按垂直方向堆叠，因此 scrollWidth 取二者宽度最大值；
    // 想让原生最大滚动位置为「工程宽」，spacer 需再加一个视口宽。
    //
    // 【不变量：与轨道视图同一份可滚范围】这里复用轨道视图的同一个函数
    // （`resolveTimelineScrollRange`），两者都传自己的视口宽——于是两个面板的
    // 最大滚动位置**逐值相等**（都 = 内容宽）。
    //
    // 【为什么同步偏移不进这里（用户报告的「工程结尾右侧空白长度不同」）】spacer 的
    // 宽度决定原生最大滚动位置。此前同步开启时会再加一个 `timelineOffsetPx`，参数
    // 编辑器因此比轨道视图多出一段「只有自己能滚」的空白：滚到最右时轨道视图停在
    // 内容宽处（工程结尾落在轨道区左缘），参数编辑器却能再滚一个偏移量（工程结尾落到
    // 轨道区左缘以左），两个面板不再对齐。偏移是**纯投影量**，只改内容画在屏幕上的
    // 位置（见内核宿主的 `horizontalOffsetPx`），不参与可滚范围。
    const timelineScrollRange = useMemo(
        () => resolveTimelineScrollRange({ contentWidth, viewportWidth: viewSize.w }),
        [contentWidth, viewSize.w],
    );
    const paddedContentWidth = timelineScrollRange.paddedContentWidth;

    useLayoutEffect(() => {
        const el = scrollerRef.current;
        if (!el) return;
        const ro = new ResizeObserver(() => {
            // 【为什么用 clientWidth（整数）而不是 getBoundingClientRect（分数）】
            // 这个值同时是下方 sticky 包裹层的 CSS `width`（`style={{ width: viewSize.w }}`）
            // 与各画布 `rasterize` 的 CSS 尺寸入参。包裹层是 `overflow: hidden`，
            // 画布被 `rasterize` 吸附到设备像素后最多比它宽 `0.5 / dpr` CSS px ——
            // 这点亚像素差被裁掉、不产生滚动条；反过来若取分数宽度，包裹层与画布
            // 会各自落在不同的亚像素相位上，投影锚点（left: 0）之外的边缘更容易露缝。
            // 用整数是"最不易出错"的一侧；`Math.floor` 在此是恒等操作
            //（clientWidth 已是整数），保留只为显式表达"这里要的是整数 CSS 宽"。
            const w = Math.max(1, Math.floor(el.clientWidth));
            const h = Math.max(1, Math.floor(el.clientHeight));
            viewSizeRef.current = { w, h };
            setViewSize({ w, h });
            // 视口尺寸（含首次测量）必须立即同步到参数编辑器总线：总线
            // 初始窗口是 1px 占位，PianoRollWaveformSurface 按总线 axis
            // 裁剪（窗口外 clip 直接跳过、不发起取数）——fresh 应用在
            // 首次滚动/缩放之前总线永远不会被 emit，波形因此无法显示。
            pianoRollViewportBus.emit(scrollLeftRef.current, pxPerSecRef.current, w);
        });
        ro.observe(el);
        return () => ro.disconnect();
    }, []);

    // The ruler is React-rendered, but the main graph is canvas-rendered.
    // Ensure playhead changes (seek / playback) trigger a redraw.
    useEffect(() => {
        invalidate();
    }, [s.playheadSec, invalidate]);

    /**
     * 标尺播放头线的**首帧与缩放后**定位。
     *
     * 【为什么位置不再交给 React 渲染】见 `TimeRulerPlayhead.positionFromProps`：
     * React 只持有 30Hz 的已提交播放头，而画布/GL 上的播放头用的是 60Hz 插值
     * 位置，两者同时写 `style.left` 会让标尺上的线落在旧位置 —— 用户看到的就是
     * "标尺的播放头与参数编辑器里的播放头分离"。
     *
     * 这里用 layout effect（绘制前执行）补上首帧与缩放后的位置，播放/滚动期间的
     * 逐帧位置由 `useVisualPlayhead` 的 onFrame 与 `applyScrollLayers` 负责。
     */
    useLayoutEffect(() => {
        // 内核宿主每帧都在写这两个元素（单写者原则，见 4b0bbb40）：此 effect 只在
        // 无宿主的回退路径补首帧/缩放后定位，否则一次一帧过期的 React 提交会把
        // 播放头拉回旧位置（"标尺播放头与主体分离"回归）。
        if (hostRef.current !== null) {
            return;
        }
        // 缩放取**内核真值**（与 GL 播放头、标尺平移同一口径）：面板的 React
        // `pxPerSec` 是请求值，内核可能因视口宽度变化而钳制它，用请求值定位会与
        // 主体播放头差一个 `播放头秒数 × 缩放差`。此分支只在无宿主时执行，
        // 因此没有内核视口可取，直接退回面板值。
        const view = resolvePanelRenderViewport({
            kernelView: null,
            refPxPerSec: pxPerSecRef.current,
            refScrollLeftPx: scrollLeftRef.current,
        });
        const leftPx = playheadLineLeftPx(
            createTimelineAxis({
                pxPerSec: view.pxPerSec,
                scrollLeftPx: view.scrollLeftPx,
                viewportWidthPx: viewSizeRef.current.w,
                dpr: readDevicePixelRatio(),
            }),
            visualPlayheadSecRef.current,
        );
        if (rulerPlayheadLineRef.current) {
            rulerPlayheadLineRef.current.style.left = `${leftPx}px`;
        }
        if (rulerPlayheadHeadRef.current) {
            rulerPlayheadHeadRef.current.style.left = `${leftPx}px`;
        }
    }, [pxPerSec]);

    // 原地等待渲染（位置冻结）期间不得推进视觉插值（见 TimelinePanel 同名注释）。
    const isTransportAdvancing =
        s.runtime.isPlaying &&
        !s.runtime.playbackWaitingForRender &&
        s.runtime.playbackPositionSec > 1e-4;

    useVisualPlayhead({
        syncedPlayheadSec: s.playheadSec,
        syncedAtMs: s.playheadSampledAtMs,
        isTransportAdvancing,
        onFrame: useCallback(
            (visualPlayheadSec: number) => {
                visualPlayheadSecRef.current = visualPlayheadSec;
                // 标尺播放头线**不在这里写**：它由内核在帧提交里按内核视口写，
                // 与 GL 主体播放头同源同帧（见 `sync.rulerPlayheadLine` 的说明）。
                // 这里曾用面板的 `pxPerSecRef` 写同一条线 —— 面板缩放与内核缩放
                // 一旦不一致（面板宽度变化时内核会重新钳制缩放），两条线就会相差
                // `播放头秒数 × 缩放差`，也就是用户报告的"标尺线与主体线分离"。
                let followed = false;
                if (!s.paramEditorSyncTimeline && s.autoScrollEnabled && s.runtime.isPlaying) {
                    const scroller = scrollerRef.current;
                    if (scroller) {
                        const next = computeAutoFollowScrollLeft({
                            playheadSec: visualPlayheadSec,
                            pxPerSec: pxPerSecRef.current,
                            viewportWidth: scroller.clientWidth,
                            contentWidth,
                        });
                        if (Math.abs(scroller.scrollLeft - next) > 0.5) {
                            scroller.scrollLeft = next;
                            lastMirroredScrollLeftRef.current = next;
                            syncScrollLeft(scroller);
                            followed = true;
                        }
                    }
                }
                // 【为什么区分两种失效】自动跟随滚动改变了视口，曲线几何必须重建，
                // 因此那一帧仍走完整失效；而"视口没动的纯播放头前进"帧几何完全不变
                // —— 走 `invalidatePlayhead()` 只重绘叠加层，省掉每帧约 20 万顶点的
                // 曲线几何重建（约 3 MB 分配 / 帧）。见宿主接口处的说明。
                if (followed) {
                    invalidate();
                } else {
                    invalidatePlayhead();
                }
            },
            // syncScrollLeft reads the latest scroll state through refs and is called
            // imperatively; including the plain render-scope function would defeat memoization.
            // eslint-disable-next-line react-hooks/exhaustive-deps
            [
                contentWidth,
                invalidate,
                invalidatePlayhead,
                s.paramEditorSyncTimeline,
                s.autoScrollEnabled,
                s.runtime.isPlaying,
            ],
        ),
    });

    useEffect(() => {
        return () => {
            if (scrollStateRafRef.current != null) {
                cancelAnimationFrame(scrollStateRafRef.current);
                scrollStateRafRef.current = null;
            }
            if (zoomRafRef.current != null) {
                cancelAnimationFrame(zoomRafRef.current);
                zoomRafRef.current = null;
            }
            if (rafRef.current != null) {
                cancelAnimationFrame(rafRef.current);
                rafRef.current = null;
            }
        };
    }, []);

    // 内核宿主 + 各图层的同帧落地整块在 `usePianoRollScrollKernel` 里（见该文件头）。
    // 调用点留在原位：这几个函数的声明位置就是它们的**使用顺序**——`syncScrollLeft` 在
    // 上方 `useVisualPlayhead` 的 onFrame 里被闭包引用（延迟读取，与 `commitViewportNow`
    // 提前交给 `usePianoRollViewport` 是同一模式），其余调用者都在本行之后。
    const {
        applyScrollLayers,
        applyHorizontalScrollPosition,
        applyEdgeScrollLeft,
        syncScrollLeft,
    } = usePianoRollScrollKernel({
        s,
        hostRef,
        scrollerRef,
        scrollLeftRef,
        lastScrollLeftRef,
        lastMirroredScrollLeftRef,
        lastMirroredScrollTopRef,
        pxPerSecRef,
        viewSizeRef,
        paramEditorSyncTimelineRef,
        timelineOffsetRef,
        timelineSyncApplyingRef,
        scrollStateRafRef,
        setScrollLeft,
        visualPlayheadSecRef,
        rulerPlayheadLineRef,
        rulerPlayheadHeadRef,
        captureBandRef,
        rulerContentRef,
        gridLayerRef,
        drawRef,
        kernelDataRef,
        editParamRef,
        pitchViewRef,
        paramViewsRef,
        hScrollbarThumbRef,
        vScrollbarThumbRef,
        hScrollbarTrackRef,
        vScrollbarTrackRef,
        glCanvasRef,
        glAxisCanvasRef,
        glOverlayCanvasRef,
        getCurrentViewportForScrollbar,
        getParamValueBoundsForScrollbar,
    });

    /**
     * **立即**提交一帧（有宿主时同任务提交，否则退回标脏 rAF）。
     *
     * 【为什么需要它：DOM / Canvas2D 与 GL 必须落在同一个任务里】
     * 面板的权威写入点（缩放落地、共享视口应用、值域平移/缩放）会同步写 DOM /
     * Canvas2D。若 GL 侧（参数线 / 原始音高线 / 播放头）等下一帧才跟上，同一份视口
     * 就被两套图层分帧呈现——用户看到的就是"这些线迟缓几帧渲染"。
     *
     * 实测（Chrome，参数编辑器内滚轮缩小，按帧记录两个绘制路径的 scrollLeft）：
     * 面板在 t=6384 写下 DOM/Canvas2D，GL 到 t=6446 才用同一个值重绘（React 提交 +
     * rAF 重新排队，慢一帧到数帧）。改用本方法后两者同任务提交，手感与旧实现
     * （原生滚动 + 在事件里同步重绘）一致。
     *
     * 与 `invalidate()` 的分工：数据变更（编辑结果、主题、音阶）仍走 rAF 合并；
     * **用户手势的视口提交**走本方法。
     */
    function paintNow(): void {
        const host = hostRef.current;
        if (host) {
            host.paintNow();
            return;
        }
        invalidate();
    }

    /**
     * 视口提交的统一入口：内核写入 + 各图层**同任务**提交。
     *
     * 流程：内核写入（含钳制与标脏）→ `paintNow()`（同一任务的 GL + DOM + Canvas2D 提交）。
     *
     * 【缩放必须与位置**一起**写入内核（否则画布层会抽搐一帧）】
     * 只写位置时，内核仍是**旧缩放**：本函数紧接着的同帧绘制（波形 / 参数线 / 播放头 /
     * 选区都在 GL 或 Canvas2D 上）就会用旧缩放画出新位置，而 DOM 侧的标尺 / 网格（由
     * React 的 `pxPerSec` 驱动）已经是新缩放——一屏之内两套缩放，下一帧才被纠正。
     * 实测（Chrome，参数编辑器内滚轮缩放，日志打在绘制前）：`kernelPps=150 targetPps=165`
     * ——画布按 150 画、DOM 按 165 画。传 `pxPerSecAtCommit` 后内核先原子换到新缩放，
     * 同帧绘制即与 DOM 一致（`kernelPps=165 targetPps=165`）。
     *
     * 特殊说明：宿主尚未创建（挂载期）时退回 `applyScrollLayers` 直接写 DOM，与各调用点
     * 原本的兜底语义一致。
     *
     * @param drawingScrollLeft 目标水平位置（绘制坐标）。
     * @param pxPerSecAtCommit 本次提交应生效的缩放；缺省表示缩放不变（纯滚动）。
     */
    function commitViewportNow(drawingScrollLeft: number, pxPerSecAtCommit?: number): void {
        const host = hostRef.current;
        if (host) {
            if (pxPerSecAtCommit !== undefined && Number.isFinite(pxPerSecAtCommit)) {
                // 缩放 + 位置**一次原子写入**：内核用新缩放算上限并钳制，随后同帧绘制
                // 用的就是新缩放（见上方说明）。
                host.setViewport({ pxPerSec: pxPerSecAtCommit, scrollLeft: drawingScrollLeft });
            } else {
                applyHorizontalScrollPosition(drawingScrollLeft);
            }
            host.paintNow();
            return;
        }
        applyHorizontalScrollPosition(drawingScrollLeft);
        applyScrollLayers(drawingScrollLeft);
    }

    // 内核数据镜像：每次 render 更新字段（宿主每帧现读，不触发重建）。
    //
    // 特殊说明 1：`projectSec` 必须是**秒**而不是 `contentWidth`（像素）——宿主内部
    // 会自行乘以 pxPerSec 得到内容宽，传像素会让内容宽度被放大 pxPerSec 倍，
    // 横向滚动条 thumb 缩成一条线。
    //
    // 特殊说明 2：值域（`min/max/span`）必须与面板当前视口同源。宿主用它把像素位置
    // 换算成值域中心；若停留在占位值，竖向换算的比例就错了——表现为钢琴键盘整体
    // 偏移一个八度、竖向滚动条 thumb 位置也不对（曾实际发生）。
    //
    // 特殊说明 3：`span` 用**当前视口**的跨度（而非值域全长）。竖向像素位置 ↔ 值域
    // 中心的映射依赖 span，用错会让滚到同一位置对应的中心值不同。

    /**
     * 构建 GL 场景层的网格输入（阶段 2）。
     *
     * 流程：由 `editParam` 判定网格种类 → 取值域边界与当前视口 → 解析该主题的两条
     * 网格线颜色（以及钢琴背景 / 音阶高亮颜色）为数值 RGBA → 绑定 `valueToY` →
     * pitch 下再补键盘颜色与音阶音级。
     *
     * 特殊说明 1：`valueToY` 传的是**面板自己的**那个函数（只绑定 `editParam`），
     * 与 Canvas2D 路径共用同一份投影——这正是两种渲染模式网格不会错位的原因。
     *
     * 特殊说明 2：颜色经 `parseRgbaColor` 解析为 0..1 浮点，因为 GL 上传统一用浮点；
     * 在渲染热路径上反复解析 CSS 字符串是纯浪费，故在镜像更新（低频）时做。
     * 解析失败（拿到 NaN）时退回**不透明黑**——宁可颜色不对也不要上传 NaN，
     * NaN 会让整个实例属性失效、整层消失。
     *
     * 特殊说明 3（音阶高亮）：条件与旧 Canvas2D 分支**逐字对齐**——`always` 模式且
     * 存在工程音阶时才给音级集合（`off` / 无音阶时给空数组，等价于不高亮）。
     *
     * 【已知限制：Tempo Map 分段音阶未迁移】旧 Canvas2D 路径还支持 Tempo Map 的
     * **分段音阶**（`buildScaleSegments`：不同时间段用不同音阶，按段画不同 x 范围）。
     * GL 网格几何是**视口坐标且不含时间轴**——它不知道自己在哪个时间段上，无法表达
     * 分段。要迁移需要把网格层做成时间相关（或把高亮拆成独立的时间感知图层），
     * 超出本次改动范围。因此这里只实现**单一工程音阶**的均匀路径：有 Tempo Map
     * 音阶变化的工程，高亮会按工程音阶（而非各段音阶）绘制。
     *
     * 特殊说明 4（钢琴背景）：黑键行背景带**恒开**、无开关、无持久化设置；它只是
     * 纹理，与音阶高亮（语义）互不影响，后者在 GL 实例顺序上压在它之上。
     *
     * @returns 网格输入；GL 关闭或无法解析时返回 null（宿主按"没有网格"处理）。
     */
    function buildGridSpec(): PianoRollGridSpec | null {
        return buildGridSpecImpl({
            clampViewport,
            editParam,
            editParamAxisUnit,
            effectiveProjectScale,
            fontFamily,
            getCurrentViewportForScrollbar,
            getParamValueBoundsForScrollbar,
            prAxis,
            processorParamsRef,
            s,
            themeMode,
            valueToY,
        });
    }

    /**
     * 内核数据镜像：项目时长 / 值域 / 网格输入。
     *
     * 特殊说明 1：网格输入必须每次构建——它是 GL 静态层的唯一来源（Canvas2D 侧的
     * 音高网格分支已整体删除，网格只有 GL 一个绘制者）。颜色解析会创建 DOM 探针
     * （`normalizeCssColor`），但这是唯一路径，没有可省的余地。
     *
     * 特殊说明 2（**依赖项必须覆盖 buildGridSpec 读取的全部输入**）：本 effect 是
     * 镜像里 `grid` 的**唯一**写入点，因此 `buildGridSpec` 读到的每一个输入都必须
     * 在依赖里出现，否则该输入变化时镜像不刷新、签名不变、GL 几何不重建——症状
     * 正是"点了按钮但画面纹丝不动"。因此除视口 / 参数外，还必须含：
     * - `themeMode`：决定两套配色（切主题后背景带与网格线都要换色）；
     * - `s.scaleHighlightMode`：决定音阶高亮是否产生音级集合（缺陷 #6）；
     * - `effectiveProjectScale`：换调 / 换音阶后高亮的音级集合变化。
     *
     * 实测（补依赖之前）：点击「音阶高亮」按钮后 Redux 状态确实翻转（按钮 variant
     * `ghost → solid`），但 GL 画布的像素**完全不变**——本 effect 没重跑，镜像里的
     * `grid` 还是旧对象（连带旧签名），几何自然不重建。补上依赖后同样操作能画出
     * 琥珀色强调线（像素级实测：变化行数 = 音阶音级数 × 每线设备像素行数，且再点
     * 一次回到逐像素相同）。
     */
    useLayoutEffect(() => {
        const bounds = getParamValueBoundsForScrollbar(editParam);
        const view = clampViewport(editParam, getCurrentViewportForScrollbar(editParam));
        kernelDataRef.current.projectSec = dynamicProjectSec;
        kernelDataRef.current.valueDomain = {
            min: bounds.min,
            max: bounds.max,
            span: view.span,
        };
        kernelDataRef.current.grid = buildGridSpec();
        // pxPerSec 由面板解析后写入内核（阶段 1 不迁移缩放，见宿主文件头）。
        hostRef.current?.setPxPerSec(pxPerSec);
        // 同步偏移会改变**水平上限**（原生域 = 内容宽 + 偏移），必须按新边界重钳。
        hostRef.current?.reclamp();
        hostRef.current?.invalidate();
        // `pitchViewRef` / `paramViewsRef` 是 ref（变更不触发渲染），故依赖项里无法列出；
        // 值域跨度变化由下面的显式同步点负责（`syncVerticalScrollbarForViewport`）。
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [
        dynamicProjectSec,
        pxPerSec,
        editParam,
        processorParams,
        s.paramEditorSyncTimeline,
        timelineOffsetPx,
        // 见特殊说明 2：buildGridSpec 读到的主题 / 高亮 / 音阶都必须在此列出。
        themeMode,
        s.scaleHighlightMode,
        effectiveProjectScale,
        // 纵轴展示单位（倍率 / dB）：同样是 buildGridSpec 的输入，漏了就是
        // "点了切换但刻度纹丝不动"（签名不变 → GL 几何与文字都不重建）。
        editParamAxisUnit,
        // Tempo Map 分段音阶（缺陷 #6）：换 Tempo Map 会换分段。
        s.tempoMap,
    ]);

    // 渲染期刷新 syncScrollLeft 引用（useCallback 化后引用仅在同步开关变化时更新）：
    // 供只注册一次的原生 `wheel` 监听器调用，避免闭包捕获陈旧实现。
    const syncScrollLeftRef = useRef(syncScrollLeft);
    syncScrollLeftRef.current = syncScrollLeft;

    // 【已删除：每帧对账自愈循环】
    // 该循环的存在前提是「原生 scroller 是滚动/缩放的唯一事实源」。内核是唯一渲染
    // 路径后真值源反转（内核持有、原生是被动镜像），用原生值反向对账会变成两个方向
    // 的自愈互相打架：内核刚写镜像，循环又把镜像读回来当作"权威"重发一次。内核自己
    // 每帧提交、且镜像由内核回写，故该职责整体消失。

    const valueToY = useCallback(
        (param: ParamName, v: number, h: number) =>
            valueToYImpl({ paramViewsRef, pitchViewRef, processorParamsRef }, param, v, h),
        [],
    );

    const yToViewportT = yToViewportTImpl;

    const yToValue = useCallback(
        (param: ParamName, y: number, h: number) =>
            yToValueImpl({ paramViewsRef, pitchViewRef, processorParamsRef }, param, y, h),
        [],
    );

    function clampViewport(param: ParamName, v: ValueViewport): ValueViewport {
        return clampViewportImpl({ processorParamsRef }, param, v);
    }

    function getParamValueBoundsForScrollbar(param: ParamName): { min: number; max: number } {
        return getParamValueBoundsForScrollbarImpl({ processorParamsRef }, param);
    }

    function getCurrentViewportForScrollbar(param: ParamName): ValueViewport {
        return getCurrentViewportForScrollbarImpl(
            { getParamValueBoundsForScrollbar, paramViewsRef, pitchViewRef },
            param,
        );
    }

    /**
     * 把值域视口同步到竖向滚动位置（值域 → 像素）。
     *
     * 流程：钳制视口 → 更新内核数据镜像的值域 → 把 `center` 交给滚动载体。
     *
     * 特殊说明 1：滚动载体是内核宿主。像素域恒为 0..1600，映射由
     * `verticalScrollTopFromCenter` 单一来源保证。宿主内部自带去重，故无需在这里
     * 再判一次差值。宿主尚未创建时（挂载期）退回原生 scroller，语义一致。
     *
     * 特殊说明 2（**必须先刷新镜像的 span**）：竖向缩放（钢琴键区 alt/ctrl+滚轮）
     * 只改 `span`，而视口存在 **ref** 里——ref 变更不触发 React 渲染，于是「内核数据
     * 镜像」那个 effect 不会重跑，内核读到的 `valueDomain.span` 仍是旧值。此时
     * `setValueCenter` 会按**旧 span** 反算像素位置，结果完全错位（实测：缩放后
     * nativeTop 直接掉到 0 且再也动不了）。因此这里在提交前就地刷新镜像。
     *
     * @param param 参数名（决定值域边界）。
     * @param view 目标值域视口。
     */
    function syncVerticalScrollbarForViewport(param: ParamName, view: ValueViewport): void {
        return syncVerticalScrollbarForViewportImpl(
            {
                clampViewport,
                getParamValueBoundsForScrollbar,
                hostRef,
                kernelDataRef,
                lastMirroredScrollTopRef,
                scrollerRef,
            },
            param,
            view,
        );
    }

    /**
     * 【已删除：`applyViewportFromVerticalScrollbar`（像素 → 值域）】
     *
     * 它的唯一调用者是原生 `scroll` 事件里的竖向对账分支，而那段的成立前提是
     * 「原生 scroller 是竖向事实源」。内核是唯一渲染路径后竖向由内核拥有，值域视口
     * 由宿主的 `onScrollTopFrame` **正向**回写（见上方内核宿主创建处的说明），
     * 反向采纳路径失去存在理由，故随分支一并删除。值域 → 像素方向仍由
     * `syncVerticalScrollbarForViewport` 提供。
     */

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
    const liveEditActiveRef = useMemo(
        () => createLiveEditFlag(() => liveEditEndHandlerRef.current?.()),
        [],
    );

    const {
        paramView,
        setParamView,
        secondaryParamViews,
        referencePitchViews,
        bumpRefreshToken,
        refreshToken,
        refreshNow,
        refreshSecondaryNow,
        notifyLiveEditEnded,
        isLoading,
    } = usePianoRollData({
        editParam,
        secondaryParamIds: visibleSecondaryParamIds,
        referenceRootTrackIds: visibleReferenceRootTrackIds,
        pitchEnabled,
        paramsEpoch: (s as unknown as { paramsEpoch?: number }).paramsEpoch ?? 0,
        rootTrackId,
        selectedTrackId: effectiveSelectedTrackId,
        scrollLeft,
        // 视口窗口只需要「秒 ↔ 像素」这一个系数。此前传的是 `pxPerBeat` +
        // `secPerBeat`（两者相乘才等于它），于是**改 BPM 会让取数 effect 重跑**、
        // 整条曲线白取一次；换成 pxPerSec 后 BPM 与取数彻底解耦。
        pxPerSec,
        viewWidth: viewSize.w,
        viewSizeRef,
        scrollLeftRef,
        pxPerSecRef,
        invalidate,
        liveEditActiveRef,
    });

    /**
     * 波形「可听结果」映射所需的响度自动化快照（整条工程的 volume 曲线 +
     * 动态目标/原声基线）。
     *
     * 【为什么独立于 paramView】波形的形变与"当前编辑哪个参数"无关 —— 用户在
     * **任何**参数面板都要实时看到音频波形（画一笔音量/动态曲线波形立刻跟着
     * 动）。paramView 只覆盖当前参数且随视口窗口化，撑不起这个语义；快照按
     * 整工程自适应 stride 拉取，滚动/缩放零重取（见 useLoudnessCurves）。
     */
    const loudnessFpMs = paramView?.framePeriodMs ?? 5;
    const loudnessProjectFrames = Math.max(1, Math.ceil((dynamicProjectSec * 1000) / loudnessFpMs));
    const {
        snapshot: loudnessSnapshot,
        analysisPending: loudnessAnalysisPending,
        snapshotFetchSeq: loudnessSnapshotFetchSeq,
        getLatestFetchSeq: getLatestLoudnessFetchSeq,
        flushPending: flushLoudnessPending,
    } = useLoudnessCurves({
        rootTrackId,
        projectFrames: loudnessProjectFrames,
        framePeriodMs: loudnessFpMs,
        liveEditActiveRef,
        paramsEpoch: (s as unknown as { paramsEpoch?: number }).paramsEpoch ?? 0,
        refreshToken,
    });

    /**
     * live 覆盖读取器（解析按覆盖对象身份缓存，见 `createLiveOverrideReader`）。
     *
     * 【为什么必须缓存】`readLiveOverrideFor` 在波形几何重建的热路径上被反复
     * 调用 —— 几何层按列内增益切片调用幅度映射，一次重建可达数万次（实测
     * 2112 列 × 16 切片 × 2 = 6.8 万次）。若每次询问都 `split` key 再新建视图
     * 对象，仅字符串切分就要吃掉 ~26ms/帧 —— 用户报告的「编辑音量/动态时
     * 卡顿」。读取器把稳态压缩成一次身份比较，零分配。
     */
    const liveOverrideReaderRef = useRef<LiveOverrideReader | null>(null);
    if (liveOverrideReaderRef.current === null) {
        liveOverrideReaderRef.current = createLiveOverrideReader();
    }

    /**
     * 把 live 覆盖读成 LoudnessLiveCurve（按 key 中的参数 id 与窗口对齐）。
     *
     * 键解析与视图对象由 `createLiveOverrideReader` 按覆盖对象身份缓存。
     */
    const readLiveOverrideFor = useCallback((param: "volume" | "dyn") => {
        return liveOverrideReaderRef.current?.read(param, liveEditOverrideRef.current) ?? null;
        // eslint-disable-next-line react-hooks/exhaustive-deps -- 两个 ref 都是稳定引用（reader 在下方惰性创建、live 覆盖由 useLiveParamEditing 持有并就地更新）；加入依赖不会让回调更正确，反而会在每次渲染换掉引用、使幅度映射的几何缓存键失效
    }, []);

    /**
     * 波形响度映射的修订号（ref，**不触发 React 渲染**）。
     *
     * 绘制中的 live 覆盖写在 ref 上，幅度映射内部用延迟取值读它，因此函数引用
     * 不变也能产出不同结果 —— 几何缓存若只按引用比较会误判"没变"，
     * 绘制中的曲线就画不上去。这里用 ref 累加计数、映射通过 `revision()`
     * 惰性读取：若改成 state，拖动时每次 pointermove 都会重渲染整块面板。
     */
    const loudnessWaveformRevisionRef = useRef(0);

    /**
     * 时间轴几何手势的「按下时几何」快照（见 `clipGeometryPreviewBus`）。
     *
     * 【为什么用 state 而不是 ref】手势起止是**离散**事件（每只手势各一次），
     * 而它决定波形的乘数从哪儿来 —— 必须触发一次重渲染，让幅度映射挂上/撤下
     * 几何映射。手势内部的逐帧变化不走 state（走 `loudnessWarpRef` + LUT 的
     * 映射身份比较），否则整块面板会以指针频率重渲染。
     */
    const [clipGeometryPreviewOrigin, setClipGeometryPreviewOrigin] = useState<
        readonly ClipGeometrySnapshot[] | null
    >(() => getClipGeometryPreviewOrigin());

    /**
     * 收尾期的「基线溯源基准」：手势结束**之前**最后落地的那份快照的溯源键。
     *
     * 【为什么不能用取数序号】提交会先 `checkpointHistory()` 递增 `paramsEpoch`
     * （早于后端写入），而取数序号在**发出**时（`useLoudnessCurves` 的
     * `fetchReqIdRef`）就推进 —— 水位与"这份数据反映哪份几何"必然错拍，
     * 于是映射时而提前撤下（闪一下）、时而过晚撤下（新基线被再搬一次）。
     *
     * 溯源键是**事实**：拖拽期间后端几何被冻结 ⇒ 键恒定；提交写回后端后键必变。
     * 于是"键变了"当且仅当"这份基线反映新几何"。判定在**渲染期**完成（见
     * `warpSettled`），撤下与快照落地因此落在同一个 React 提交里，
     * 结构上不存在"新快照 × 旧映射"或"旧快照 × 无映射"的中间帧。
     */
    const warpBasisKeyRef = useRef<string | null>(null);
    /** 键不可用时的退化基准（快照对象身份），见 `warpSettled`。 */
    const warpBasisSnapshotRef = useRef<LoudnessSnapshot | null>(null);
    /** 手势是否已结束、正在等权威快照追上。 */
    const warpSettlingRef = useRef(false);

    useEffect(() => {
        setClipGeometryPreviewOrigin(getClipGeometryPreviewOrigin());
        warpSettlingRef.current = false;
        return subscribeClipGeometryPreview(() => {
            const origin = getClipGeometryPreviewOrigin();
            if (origin !== null) {
                warpSettlingRef.current = false;
                setClipGeometryPreviewOrigin(origin);
                return;
            }
            // 手势结束：**不在这里**撤下映射。提交之后、权威快照回来之前，映射仍是
            // 把基线搬到新位置的唯一正确来源（后端此刻正在算的正是同一件事）。
            // 撤下交给 `warpSettled`（渲染期判定，与快照落地同批）。
            //
            // 【取消手势为什么同样安全】取消会把 Redux 几何还原成按下时的值，
            // 差分随之退化为空 ⇒ 映射为 `null`，撤不撤在画面上没有区别。
            warpSettlingRef.current = true;
        });
    }, []);

    /**
     * 当前生效的几何映射（渲染期同步的 ref 镜像，见文件内既有的同款模式）。
     *
     * 幅度映射以 provider 惰性读取它，因此映射对象的**引用**保持不变（几何缓存与
     * 组件 memo 不会因它而重建），而变化由查表键里的映射身份比较捕捉
     * （见 `makeLoudnessAmplitudeMap` 的 `warp` 说明）。
     */
    const loudnessWarpRef = useRef<LoudnessGeometryWarp | null>(null);
    const readLoudnessWarp = useCallback(() => loudnessWarpRef.current, []);

    /**
     * 波形重绘的**帧内合并**调度（与曲线层同用 `renderKernel/renderLoop` 语义）。
     *
     * 【为什么需要】一次全量重建要重算两千余列 × 16 切片的包络并上传数百 KB
     * 顶点（实测典型窗口 ~2.7ms）。而绘制中的 live 覆盖在**同一帧内可能被更新
     * 多次**：手绘工具会把一个 `pointermove` 的 coalesced 采样全部展开处理
     * （`usePianoRollInteractions` 的 `flushPendingMoves`），高刷鼠标 / 笔一帧
     * 可积 2~8 个采样。此前每个采样都**同步**强制提交一次全量重建 —— 单帧
     * 8~30ms 的几何工作，正是用户报告的"编辑音量/动态时卡顿"。
     *
     * 改为标脏 + rAF 合并后一帧至多重建一次，且与曲线层（面板自己的
     * `invalidate()` → 宿主 renderLoop）落在**同一帧**，两层不再错帧。
     *
     * 【为什么不用同帧 `flush()`】滚动路径要求同帧提交，是因为要与原生滚动的
     * DOM 内容层对齐（见 PianoRollWaveformSurface / viewportBus 的契约）。绘制
     * 参数曲线的路径没有随滚动移动的 DOM 层，与曲线层同帧即可。
     */
    const waveformRepaintLoopRef = useRef<RenderLoop | null>(null);
    if (waveformRepaintLoopRef.current === null) {
        waveformRepaintLoopRef.current = createRenderLoop({
            draw: () => pianoRollViewportBus.invalidate(),
        });
    }
    useEffect(() => {
        const loop = waveformRepaintLoopRef.current;
        loop?.start();
        return () => loop?.stop();
    }, []);

    /**
     * 强制重绘参数面板波形（**仅当绘制中的 live 覆盖会改变波形时**）。
     *
     * 必要性：绘制中的响度曲线只写在 `liveEditOverrideRef`（ref 变更不触发
     * React 渲染），而波形面是 memo 组件 + 几何缓存，既收不到 ref 变更、
     * 也不会因 props 未变而重绘。故显式强制重绘一次：总线以同一份投影 force
     * commit，波形图层命中重绘并因修订号变化而重建几何（其余图层内容未变，
     * 开销可忽略）。
     *
     * 【为什么必须按参数过滤】波形画的是「可听结果」
     * （`源峰值 × clip增益×淡化 × volume(t) × dyn增益(t)`，见
     * `makeLoudnessAmplitudeMap`）—— 它**不依赖音高 / 共振峰 / 齿度**等参数。
     * 而波形面的几何重建在总线驱动下是**每次全量**（`WaveformSurface.draw`：
     * 总线驱动的 `canReuse` 恒为 false，见该函数的说明），一次重建要重算
     * 两千余列 × 16 切片的包络。绘制音高时逐帧触发它纯属浪费 —— 一次重建
     * 在参数面板的典型窗口下约 2.7ms（响度映射），叠加曲线自身重绘后会
     * 明显抬高指针帧成本。因此这里用当前 live 覆盖的参数 id 做闸门：
     * 只有编辑 volume / dyn 时才推进修订号并请求重绘。
     *
     * 【与曲线绘制的关系】曲线（选区、绘制中的参数线）由面板自己的
     * `invalidate()` → 宿主帧提交驱动，**不经过本函数**；因此本闸门只影响波形
     * 面，绘制音高时曲线仍然逐帧更新。
     */
    const requestWaveformRepaint = useCallback(() => {
        // 闸门：只有编辑 volume / dyn 时波形才可能变（波形画的是可听结果，
        // 与音高 / 共振峰等参数无关）。判定复用读取器的解析缓存，零分配。
        if (!liveOverrideReaderRef.current?.affectsWaveform(liveEditOverrideRef.current)) {
            return;
        }
        // 修订号在**请求时**推进（不是绘制时）：同一帧内其它路径触发的绘制也要
        // 看到"上次重建之后 live 覆盖变过"，否则几何缓存会误判为可复用。
        loudnessWaveformRevisionRef.current += 1;
        // 绘制请求帧内合并：同一帧内多次请求只重绘一次（见 waveformRepaintLoop）。
        waveformRepaintLoopRef.current?.invalidate();
        // eslint-disable-next-line react-hooks/exhaustive-deps -- liveOverrideReaderRef / waveformRepaintLoopRef 是稳定引用（惰性创建）；live 覆盖经 ref 读取，故本回调必须保持引用稳定（它进入 usePianoRollInteractions 的依赖链）
    }, []);

    /**
     * 参数面板波形的幅度映射（**所有参数统一**）。
     *
     * 把源波形画成**应用响度自动化之后的可听结果**
     * （`源峰值 × clip增益×淡化 × volume(t) × dyn增益(t)`）：
     * - 编辑任何参数时画一笔音量/动态曲线，波形立刻跟着起伏；
     * - 切换参数时波形保持稳定（映射不依赖 editParam），只有曲线叠加层变化；
     * - 音量↔动态互转前后波形逐像素不变 —— 等效性最直观的佐证。
     *
     * 【引用稳定性】该值参与 `WaveformSurface` 的几何缓存键，因此必须 memo：
     * 只在快照对象真正变化时换引用。绘制中的 live 覆盖不改变本对象
     * （延迟取值 + 修订号），由 `requestWaveformRepaint` 驱动重建。
     *
     * 【恒等快化】volume 恒 1 且动态无基线（未使用响度自动化的工程）时
     * `identity = true` → 不挂映射，与时间线及既有行为逐像素一致。
     */
    const pianoRollAmplitudeMap = useMemo(() => {
        if (!loudnessSnapshot || loudnessSnapshot.identity) return undefined;
        if (loudnessSnapshot.volume.length === 0) return undefined;
        return makeLoudnessAmplitudeMap(
            loudnessSnapshot,
            {
                volume: () => readLiveOverrideFor("volume"),
                dyn: () => readLiveOverrideFor("dyn"),
            },
            () => loudnessWaveformRevisionRef.current,
            // 拖拽期间的几何映射：把基线/曲线搬到 clip 的当前位置，消除
            // 「旧位置的基线 × 新位置的峰值」造成的整段错位（见该模块的文件头）。
            readLoudnessWarp,
        );
    }, [loudnessSnapshot, readLiveOverrideFor, readLoudnessWarp]);

    const refreshSecondaryNowRef = useRef(refreshSecondaryNow);
    useEffect(() => {
        refreshSecondaryNowRef.current = refreshSecondaryNow;
    }, [refreshSecondaryNow]);

    useEffect(() => {
        if (!rootTrackId) {
            invalidate();
            return;
        }
        if (visibleSecondaryParamIds.length > 0 || visibleReferenceRootTrackIds.length > 0) {
            void refreshSecondaryNowRef.current();
            return;
        }
        invalidate();
    }, [invalidate, rootTrackId, visibleReferenceRootTrackIds, visibleSecondaryParamIds]);

    const handleRequestEnableCompose = useCallback(() => {
        const rtId = rootTrackId;
        if (!rtId) return;
        dispatch(
            setTrackStateRemote({
                trackId: rtId,
                composeEnabled: true,
            }),
        );
    }, [dispatch, rootTrackId]);

    /*
     * MIDI 导入：对话框状态、导入参数、拖放接入与「导入为 Clip」都归这一个 hook
     * （见 `useMidiImport.ts`）。它自带 15 个 useState 与一条独立通路，与画布、滚动、
     * 绘制没有交集 —— 是面板里少数几个边界清楚、可以整块搬走的功能。
     */
    const {
        clipboardGuid,
        closeLeadingGap,
        fillGaps,
        handleCloseLeadingGapChange,
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
        handleOpenMidiDialog,
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
        midiDialogOpenParamsRef,
        midiDialogSourceRef,
        midiPath,
        midiSelRanges,
        midiSelectionAvailable,
        multiTrackMerge,
        noteBpmMode,
        paramEditorMidiDragOver,
        setClipboardGuid,
        setMidiDialogOpen,
        setMidiDialogSelection,
        setMidiPath,
        specifiedBpm,
    } = useMidiImport({ dispatch, paramEditorRef, refreshNow, s, selectionRef });

    // 将当前选区发布到总线，供 MenuBar 等判断“工程音阶”是否受 Tempo Map 影响。
    // 多选区发布**包围区间**（首段起点 → 末段终点）：该提示只关心"选区覆盖的
    // 时间跨度里有没有音阶变化点"，包围区间是它的保守上界。
    useEffect(() => {
        const sel = selectionUi;
        const bounding = selectionBoundingSpan(sel);
        if (!bounding) {
            publishPianoRollSelection(null);
            return;
        }
        // 选区已经是帧制：直接发布，不再有任何换算（此前要经 `secPerBeat` 转一道）。
        publishPianoRollSelection({
            startFrame: Math.max(0, bounding.startFrame),
            frameCount: Math.max(1, bounding.frameCount),
            framePeriodMs: paramView?.framePeriodMs ?? 5,
        });
        return () => {
            publishPianoRollSelection(null);
        };
    }, [selectionUi, paramView?.framePeriodMs]);

    // 获取当前 track 下的所 ?clips，用 ?per-clip 波形叠加绘制
    // 获取轨道组内所有 clips（包含 root 轨道及所有子轨道的 clip）
    const trackClips = useMemo(
        () => s.clips.filter((c) => groupTrackIds.has(c.trackId)),
        [s.clips, groupTrackIds],
    );

    /**
     * 拖拽期间的响度时域映射：把「快照所依据的几何」与当前几何的差分，应用到基线 /
     * 曲线（见 `loudnessGeometryWarp` 的文件头）。
     *
     * 【为什么只取本轨道组的 clip】原声基线是**按根轨道组**组装的；别的轨道组的
     * clip 与本组基线无关，把它们的映射算进来会错误地搬移本组曲线。`trackClips`
     * 已按 `groupTrackIds` 过滤，正好是基线的参与集合。
     *
     * 【为什么在渲染期判定收尾】`warpSettled` 只依赖"当前快照的溯源键"这一个事实，
     * 因此它翻转的那次渲染**就是**新快照生效的那次渲染：映射在同一次提交里变成
     * `null`，`pianoRollAmplitudeMap` 也在同一次提交里换新 —— 不存在中间帧。
     */
    const loudnessSnapshotKey = loudnessSnapshot?.baselineKey ?? null;
    // 非收尾期：基准跟随最新快照。拖拽期间后端几何被冻结，期间落地的快照同样是
    // "按下之前"的几何，对它可以安全地继续套用映射（不 re-base 会让映射平白失效）。
    if (!warpSettlingRef.current) {
        warpBasisKeyRef.current = loudnessSnapshotKey;
        warpBasisSnapshotRef.current = loudnessSnapshot;
    }
    /**
     * 收尾是否已完成：手势结束**之后**才落地的一份快照带回了不同的溯源键
     * （= 它反映的是新几何）。
     *
     * 键不可用（后端从未组装过基线）时退化为「快照换了对象即算追上」—— 该路径下
     * 后端没有可比对的几何信息，且基线本身也多半为空（映射在画面上无作用）。
     */
    const warpSettled =
        warpSettlingRef.current &&
        (warpBasisKeyRef.current !== null && loudnessSnapshotKey !== null
            ? loudnessSnapshotKey !== warpBasisKeyRef.current
            : loudnessSnapshot !== warpBasisSnapshotRef.current);

    const loudnessGeometryWarp = useMemo(() => {
        if (clipGeometryPreviewOrigin === null || warpSettled) return null;
        return createLoudnessGeometryWarp({
            origin: clipGeometryPreviewOrigin,
            clips: trackClips,
            framePeriodMs: loudnessFpMs,
            lockParamLines: s.lockParamLinesEnabled,
        });
    }, [
        clipGeometryPreviewOrigin,
        trackClips,
        s.lockParamLinesEnabled,
        loudnessFpMs,
        // 快照对象本身**不是**依赖：映射只由"按下时几何 + 当前几何"决定，快照换了
        // 对象而键没变（拖拽期间后端几何被冻结时的重取）不该让映射重算。
        // 需要跟着快照变化的那一件事 —— 收尾判据 `warpSettled` —— 已在依赖里。
        warpSettled,
    ]);

    // 收尾完成 ⇒ 可以安全丢掉基准几何：此时映射已是 `null`，且下一次手势会发布新的。
    useEffect(() => {
        if (!warpSettled || clipGeometryPreviewOrigin === null) return;
        warpSettlingRef.current = false;
        setClipGeometryPreviewOrigin(null);
    }, [warpSettled, clipGeometryPreviewOrigin]);

    /**
     * 映射变化 ⇒ **渲染期**推进修订号。
     *
     * 【为什么必须在渲染期】映射引用在渲染期写入 `loudnessWarpRef`（读取方在绘制期
     * 惰性读取），而修订号此前在被动 effect 里推进 —— 两者不同步：`WaveformSurface`
     * 的 layout effect 会在同一次提交里同步全量重建，它读到的将是「新映射 / 旧修订号」，
     * 几何缓存据此误判"可复用"从而画出上一帧的顶点。这里把修订号与引用一起在渲染期
     * 更新，任何一次绘制看到的 `(amplitudeMap, revision)` 都与当前映射一致。
     * 重绘**请求**仍交给下面的 effect（那只影响"什么时候画"，不影响"画得对不对"）。
     */
    const warpRevisionRef = useRef<LoudnessGeometryWarp | null>(null);
    if (warpRevisionRef.current !== loudnessGeometryWarp) {
        warpRevisionRef.current = loudnessGeometryWarp;
        loudnessWaveformRevisionRef.current += 1;
    }
    // 渲染期同步镜像（本文件既有的取值新鲜度模式：读取方在绘制期惰性读取）。
    loudnessWarpRef.current = loudnessGeometryWarp;

    /**
     * 映射变化 ⇒ 请求一次重绘。
     *
     * 【为什么不能只靠 rows 变化】拖拽期间 `rows` 确实每帧都换（几何跟着走），
     * 但映射的变化源不止几何：手势结束、锁定参数线开关、以及"等权威快照追上"的
     * 收尾，都可能在 `rows` 不变的情况下换掉映射。少了这一步，那些帧会停在旧映射
     * 上（画面与数据不一致）。
     */
    const lastLoudnessWarpRef = useRef<LoudnessGeometryWarp | null>(null);
    useEffect(() => {
        if (lastLoudnessWarpRef.current === loudnessGeometryWarp) return;
        lastLoudnessWarpRef.current = loudnessGeometryWarp;
        waveformRepaintLoopRef.current?.invalidate();
    }, [loudnessGeometryWarp]);

    // 帧装配（轴投影 → 峰值 → live 覆盖 → 曲线/选区图层 → `drawRef.current`）整块在
    // `usePianoRollFrameAssembly` 里（见该文件头）。调用点必须留在原位：`useClipsPeaksForPianoRoll`
    // 与 `useLiveParamEditing` 由它调用，状态槽与 effect 顺序取决于这个位置。
    const {
        prAxis,
        prTickAxis,
        clipPeaks,
        paramViewRef,
        liveEditOverrideRef,
        ensureLiveEditBase,
        applyDenseToLiveEdit,
        resetLiveEditPreview,
        commitStroke,
        onParamCommitSucceeded,
        detectedPitchCurves,
        referencePitchOverlays,
        buildCurveLayers,
        buildSelectionBandSpec,
        secSpanClipRect,
        selectionClipRect,
    } = usePianoRollFrameAssembly({
        s,
        t,
        themeMode,
        fontFamily,
        pxPerSec,
        pxPerBeat,
        scrollLeft,
        viewSize,
        editParam,
        pitchEnabled,
        pitchHardDisableReason,
        childPitchHardDisableReason,
        effectiveProjectScale,
        trackClips,
        groupTrackIds,
        rootTrackId,
        rootTrack,
        visibleReferenceRootTrackIds,
        visibleSecondaryParamIds,
        hoveredReferenceRootTrackId,
        snapToggleHeld,
        paramView,
        setParamView,
        secondaryParamViews,
        referencePitchViews,
        bumpRefreshToken,
        notifyLiveEditEnded,
        loudnessAnalysisPending,
        loudnessSnapshotFetchSeq,
        getLatestLoudnessFetchSeq,
        flushLoudnessPending,
        selectionRef,
        clipboardRef,
        paramViewsRef,
        pitchViewRef,
        paramMorphOverlays,
        pxPerSecRef,
        scrollLeftRef,
        viewSizeRef,
        visualPlayheadSecRef,
        canvasRef,
        axisCanvasRef,
        hostRef,
        kernelDataRef,
        drawRef,
        liveEditActiveRef,
        liveEditEndHandlerRef,
        valueToY,
        invalidate,
    });

    const handleEditActionRef = useRef<(op: string) => void>(() => {});
    // Stable callback that delegates to the latest handleEditOp via ref
    const stableEditAction = useCallback((op: string) => {
        handleEditActionRef.current(op);
    }, []);

    const interactions = usePianoRollInteractions({
        dispatch,
        rootTrackId,
        editParam,
        pitchEnabled,
        toolMode: s.toolMode,
        framePeriodMs: paramView?.framePeriodMs ?? 5,
        dynamicProjectSec,
        scrollLeftRef,
        pxPerSecRef,
        // 交互侧坐标换算的视口真值：与渲染侧（`resolvePanelRenderViewport`）同源，
        // 避免框选 / 命中测试读到量化提交滞后的 `scrollLeftRef`。见该字段的说明。
        getViewportTruth: useCallback(() => hostRef.current?.getViewport() ?? null, []),
        horizontalZoomChainRef,
        onHorizontalZoom: handleHorizontalZoom,
        syncTimelineEnabled: s.paramEditorSyncTimeline,
        timelineOffsetRef,
        setPitchView,
        setParamViewport,
        pitchViewRef,
        paramViewsRef,
        scrollerRef,
        canvasRef,
        viewSizeRef,
        selectionRef,
        selectionUi,
        setSelectionUi,
        setCanvasCursor,
        strokeRef,
        panRef,
        paramView,
        paramViewRef,
        bumpRefreshToken,
        syncScrollLeft,
        applyEdgeScrollLeft,
        invalidate,
        yToViewportT,
        yToValue,
        valueToY,
        clampViewport,
        ensureLiveEditBase,
        applyDenseToLiveEdit,
        resetLiveEditPreview,
        onParamCommitSucceeded,
        requestWaveformRepaint,
        commitStroke,
        setParamView,
        liveEditOverrideRef,
        liveEditActiveRef,
        prVerticalZoomKb,
        horizontalZoomKb,
        scrollHorizontalKb,
        scrollVerticalKb,
        scrollbarZoomKb,
        paramMorphKb,
        paramMultiSelectKb,
        paramStretchKb: stretchKb,
        vibratoAmplitudeAdjustKb,
        vibratoFrequencyAdjustKb,
        vibratoDragAmplitudeIncreaseKb,
        vibratoDragAmplitudeDecreaseKb,
        vibratoDragFrequencyIncreaseKb,
        vibratoDragFrequencyDecreaseKb,
        vibratoPresetPrevKb,
        vibratoPresetNextKb,
        vibratoPreset: activeVibratoPreset,
        straightVibratoPreset,
        vibratoPresetCycleList: enabledVibratoPresetList,
        onVibratoDragStateChange: setVibratoDragHud,
        onVibratoDragEnd: useCallback(() => {
            setVibratoDragHud(null);
        }, []),
        cycleDragDirectionKb,
        paramFineAdjustKb,
        onContextMenu: useCallback((x: number, y: number) => {
            setCtxMenu({ x, y });
        }, []),
        getPlayheadSec: getVisualPlayheadSec,
        playheadZoomEnabled: s.playheadZoomEnabled,
        paramEditorSeekPlayheadEnabled: s.paramEditorSeekPlayheadEnabled,
        pitchSnapEnabled: s.pitchSnapEnabled,
        pitchSnapUnit: s.pitchSnapUnit,
        projectScale: effectiveProjectScale,
        /** Tempo Map 感知：按帧时刻解析生效音阶。 */
        scaleAtSec: projectScaleAtSec,
        pitchSnapToleranceCents: s.pitchSnapToleranceCents,
        keybindingMap: mergedKeybindings,
        onEditAction: stableEditAction,
        dragDirection: activeDragDirection,
        onCycleDragDirection: useCallback(
            (tool: "select" | "draw" | "vibrato") => {
                dispatch(cycleDragDirection(tool));
                void dispatch(persistUiSettings());
            },
            [dispatch],
        ),
        edgeSmoothnessPercent: s.edgeSmoothnessPercent,
        onMorphOverlayChange: setParamMorphOverlays,
        currentParamRange,
        onPitchSnapGestureActiveChange: useCallback((active: boolean) => {
            setSnapGestureActive(active);
        }, []),
        paramValuePopupEnabled: s.showParamValuePopup,
        onParamValuePreviewChange: useCallback(
            (
                next: {
                    clientX: number;
                    clientY: number;
                    value: number;
                    displayText?: string;
                } | null,
            ) => {
                // 合帧提交：只记最新载荷、每渲染帧至多 setState 一次（与交互层
                // 各手势的 schedulePreview 同一动机）。浮窗是纯读数 UI，滞后一帧
                // 不可感知；不合帧时每次 move 的新对象都无法让 React 跳过重渲。
                paramValuePreviewPendingRef.current = { payload: next };
                if (paramValuePreviewRafRef.current != null) return;
                paramValuePreviewRafRef.current = requestAnimationFrame(() => {
                    paramValuePreviewRafRef.current = null;
                    const pending = paramValuePreviewPendingRef.current;
                    paramValuePreviewPendingRef.current = null;
                    if (pending) setParamValuePreview(pending.payload);
                });
            },
            [],
        ),
    });

    // 参数数据更新后重算当前悬停浮窗：键盘平移参数线（"=" / "-" / "]" / "["
    // 及其 Shift/Ctrl 变体）、撤销/重做、远端写入都只更新 paramView 数据，
    // 不会触发 pointermove —— 悬停值是 pointermove 时的快照，若不在此重算，
    // 浮窗会一直显示旧值直到用户再次移动鼠标。声明顺序在 paramViewRef 同步
    // effect（上方）之后，重算读到的是本次渲染的最新数据。
    // refreshParamValuePreview 先解构出稳定引用：直接依赖 interactions 对象
    // 会让 effect 每次渲染都触发（setState 新对象 → 无限重渲染）。
    const { refreshParamValuePreview } = interactions;
    useEffect(() => {
        refreshParamValuePreview();
    }, [paramView, refreshParamValuePreview]);

    // 滚动容器与轴列的输入接线（原生 scroll 采纳、滚动条滚轮/翻页、标尺滚轮、
    // 轴列两个交互 hook、工具模式的光标与右键菜单复位）整块在
    // `usePianoRollScrollerWiring` 里（见该文件头）。调用点留在原位：轴列两个 hook
    // 的状态槽与 effect 顺序取决于这个位置。
    const {
        onScrollerScroll,
        attachVerticalScrollbarTrack,
        attachHorizontalScrollbarTrack,
        handleRulerWheel,
    } = usePianoRollScrollerWiring({
        s,
        dispatch,
        editParam,
        pxPerSec,
        contentWidth,
        scrollerRef,
        axisWrapRef,
        hScrollbarTrackRef,
        vScrollbarTrackRef,
        viewSizeRef,
        hostRef,
        lastMirroredScrollLeftRef,
        lastMirroredScrollTopRef,
        timelineSyncApplyingRef,
        syncScrollLeftRef,
        visualPlayheadSecRef,
        pitchViewRef,
        paramViewsRef,
        processorParamsRef,
        processorParams,
        invalidate,
        clampViewport,
        getCurrentViewportForScrollbar,
        yToValue,
        setPitchView,
        setParamViewport,
        syncVerticalScrollbarForViewport,
        syncScrollLeft,
        formatParamValuePreview,
        interactions,
        horizontalZoomKb,
        scrollHorizontalKb,
        pianoKeysVerticalScrollKb,
        pianoKeysVerticalZoomKb,
        setCanvasCursor,
        setCtxMenu,
        setAxisValuePreview,
    });

    // 同步数据加载状态到状态栏。
    //
    // 走**外部 store** 而非 Context：这个 boolean 只服务状态栏一处显示，而它随每次
    // 取数（水平缩放 / 滚动 / 编辑提交）翻转 —— 曾经它住在包裹整个 AppInner 的
    // Context 里，于是每次翻转都重渲染整棵应用树，把正在进行的绘制笔画打断。
    // 现在只有状态栏那个小组件会重渲染（见 pianoRollStatusBus）。
    //
    // 按**窗体 id** 记账：参数编辑器可以多开，共用一个 boolean 会让一个实例卸载时
    // 抹掉另一个实例的状态。
    const loadingPublisherId = dockFormId ?? "param-editor";
    useEffect(() => {
        setPianoRollLoading(loadingPublisherId, isLoading);
    }, [isLoading, loadingPublisherId]);

    // 卸载时注销（否则本实例的最后状态会永远粘住，状态栏再也不会收起）。
    useEffect(() => () => clearPianoRollLoading(loadingPublisherId), [loadingPublisherId]);

    // 编辑命令（全选、命令分发、剪贴板键盘与外部同步接线）整块在
    // `usePianoRollEditCommands` 里（见该文件头）。
    const { selectAllParamRange, handleEditOp } = usePianoRollEditCommands({
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
    });

    // Dispatch helper: context menu dialog ops → open MenuBar dialogs
    const openEditDialog = useCallback((dialog: string) => {
        window.dispatchEvent(
            new CustomEvent("hifi:openEditDialog", {
                detail: { dialog },
            }),
        );
    }, []);

    // 以选区为作用域的动作（提取/套用颤音、另存为音高参考、导出 MIDI）整块在
    // `usePianoRollSelectionActions` 里（见该文件头）。调用点留在原位：这几个回调的
    // 引用身份决定 `vibratoApplyTarget` 是否稳定。
    const { vibratoApplyTarget, handleSaveAsPitchRef, handleExportMidiFromEditor } =
        usePianoRollSelectionActions({
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
        });

    // Pitch Snap 设置弹窗状态
    const [pitchSnapOpen, setPitchSnapOpen] = useState(false);

    // 工具图标是纯 JSX 常量（只依赖当前工具），见 `toolIcons.tsx`。
    const {
        vibratoToolIcon,
        lineToolIcon,
        pitchSnapSemitoneIcon,
        pitchSnapScaleIcon,
        currentDrawToolIcon,
    } = buildToolIcons(currentDrawTool);

    // 标尺刻度 / 时间格式 / 单位切换 / 复制播放头时间整块在
    // `useTimelineDisplayControls` 里（见该文件头）。它们只依赖会话切片与刻度轴。
    const {
        timelineTicks,
        timeContext,
        handlePrimaryUnitChange,
        handleSecondaryUnitChange,
        handleTempoMapChange,
        handleTempoMapCommit,
        handleCopyPlayheadTime,
    } = useTimelineDisplayControls({ dispatch, s, prTickAxis });

    /*
     * 【这个返回对象就是视图层的全部输入】字段清单是三个视图文件 AST 扫描结果的并集
     * （另加外框自己用的 `paramEditorRef`）。新增字段 = 视图里用一个新名字，然后在这里补上。
     */
    return {
        activeDragDirection,
        activeDragDirectionTool,
        activeVibratoPreset,
        activeVibratoPresetId,
        attachEdgeSmoothnessWheel,
        childFormantOffsetParam,
        childPitchOffsetCentsParam,
        childPitchOffsetDegreesParam,
        composeEnabled,
        currentDrawTool,
        currentDrawToolIcon,
        cycleDragDirectionKb,
        devicePixelRatio,
        dispatch,
        drawToolButtonTitle,
        drawToolKb,
        drawToolMenuAnchorRef,
        drawToolMenuOpen,
        drawToolMenuRef,
        editParam,
        effectivePitchSnapVisual,
        enabledVibratoPresetList,
        gatedPillHideOrder,
        gatedPillsHidden,
        getProcessorParamLabel,
        getProcessorParamShortLabel,
        getStaticOptionLabel,
        handleOpenMidiDialog,
        handleStaticParamChange,
        lineToolIcon,
        lineToolKb,
        openVibratoDialog,
        orderedProcessorParams,
        pitchAlgoOptions,
        pitchEnabled,
        pitchGroupActive,
        pitchGroupLabel,
        pitchGroupTooltip,
        pitchHardDisableReason,
        pitchSnapMenuAnchorRef,
        pitchSnapMenuOpen,
        pitchSnapMenuRef,
        pitchSnapOpen,
        pitchSnapScaleIcon,
        pitchSnapSemitoneIcon,
        presetFlyoutMaxHeight,
        processorParamsRef,
        processorStaticParams,
        processorStaticValues,
        referenceRootTrackOptions,
        rootTrack,
        rootTrackId,
        s,
        secondaryParamVisible,
        selectedIsChildTrack,
        separationEnabled,
        separationSwitchAnchorParamId,
        setDrawToolMenuOpen,
        setHoveredReferenceRootTrackId,
        setPitchSnapMenuOpen,
        setPitchSnapOpen,
        setTimeDisplaySettingsOpen,
        t,
        tf,
        timeDisplaySettingsOpen,
        toggleSecondaryParam,
        toolbarRowRef,
        toolbarTier,
        updateVisibleReferenceRootTrackIds,
        vibratoToolIcon,
        vibratoToolKb,
        visibleReferenceRootTrackIds,
        attachHorizontalScrollbarTrack,
        attachRulerPlayheadHead,
        attachRulerPlayheadLine,
        attachVerticalScrollbarTrack,
        axisCanvasRef,
        axisUnitToggleAvailable,
        axisValuePreview,
        axisWrapRef,
        canvasCursor,
        canvasRef,
        captureBandRef,
        clipPeaks,
        contentWidth,
        currentParamRange,
        dynamicProjectSec,
        editParamAxisUnit,
        effectiveProjectScale,
        formatParamValuePreview,
        getScrollLeftLive,
        glAxisCanvasRef,
        glCanvasRef,
        glOverlayCanvasRef,
        gridLayerRef,
        hScrollbarThumbRef,
        handleCopyPlayheadTime,
        handlePrimaryUnitChange,
        handleRulerWheel,
        handleSecondaryUnitChange,
        handleTempoMapChange,
        handleTempoMapCommit,
        interactions,
        onScrollerScroll,
        paddedContentWidth,
        paramValuePreview,
        pianoRollAmplitudeMap,
        prTickAxis,
        pxPerBeat,
        pxPerSec,
        resolvedVibratoPresets,
        rulerContentRef,
        scrollLeft,
        scrollerRef,
        timeContext,
        timelineTicks,
        vScrollbarThumbRef,
        vibratoDragHud,
        viewSize,
        waveformColors,
        clipboardGuid,
        closeLeadingGap,
        ctxMenu,
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
        paramEditorMidiDragOver,
        setCtxMenu,
        setMidiDialogOpen,
        setVibratoDialogOpen,
        specifiedBpm,
        // —— 内部依赖（不经视图，供 `editOperations.ts` 的命令分发取用）——
        bumpRefreshToken,
        clipboardRef,
        currentParamDefaultValue,
        currentParamQuantizeUnit,
        invalidate,
        midiDialogOpenParamsRef,
        paramView,
        paramViewRef,
        pitchDeltaToDegreeSteps,
        projectScaleAtSec,
        resolveScaleFromToken,
        selectAllParamRange,
        selectionRef,
        setClipboardGuid,
        setMidiDialogSelection,
        setMidiPath,
        setParamView,
        setSelectionUi,
        store,
        vibratoBaseline,
        // —— 内部依赖（不经视图，供 `sceneBuilders.ts` / `drawFrame.ts` 取用）——
        buildCurveLayers,
        buildSelectionBandSpec,
        childPitchHardDisableReason,
        clampViewport,
        detectedPitchCurves,
        fontFamily,
        getCurrentViewportForScrollbar,
        getParamValueBoundsForScrollbar,
        hostRef,
        kernelDataRef,
        lastMirroredScrollTopRef,
        liveEditOverrideRef,
        loudnessAnalysisPending,
        paramMorphOverlays,
        paramViewsRef,
        pitchViewRef,
        prAxis,
        pxPerSecRef,
        referencePitchOverlays,
        scrollLeftRef,
        secSpanClipRect,
        secondaryParamViews,
        selectionClipRect,
        themeMode,
        valueToY,
        viewSizeRef,
        visibleSecondaryParamIds,
        visualPlayheadSecRef,
        vibratoApplyTarget,
        vibratoDialogMode,
        vibratoDialogOpen,
        vibratoDialogSession,
        vibratoInitialPresetId,
        paramEditorRef,
    };
}

/** 模型对象的类型：由实现推出，视图层据此声明 `Pick<>`。 */
export type PianoRollPanelModel = ReturnType<typeof usePianoRollPanelModel>;
