// VibratoDialog 的草稿、预设库、试听、预览手势与导入导出模型。
// JSX 留在 VibratoDialog.tsx，便于把纯逻辑单独测试/演进。
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { EyeNoneIcon, EyeOpenIcon } from "@radix-ui/react-icons";
import { useAppDispatch, useAppSelector } from "../../../app/hooks";
import { shallowEqual } from "react-redux";
import type { RootState } from "../../../app/store";
import { isModifierActive, selectKeybinding } from "../../../features/keybindings/keybindingsSlice";
import { useI18n } from "../../../i18n/I18nProvider";
import {
    persistUiSettings,
    removeVibratoPreset,
    reorderBuiltinVibratoPreset,
    reorderVibratoPreset,
    setActiveVibratoPreset,
    toggleVibratoPresetEnabled,
    upsertVibratoPreset,
} from "../../../features/session/sessionSlice";
import {
    MAX_VIBRATO_PRESETS,
    VIBRATO_LIMITS,
    createVibratoPresetId,
    duplicateVibratoPreset,
    isBuiltinVibratoPresetId,
    nextDuplicatePresetName,
    sanitizeVibratoPreset,
} from "../../../features/vibrato/vibratoPresets";
import {
    activeIdAfterRemoval,
    findVibratoPreset,
    resolveVibratoPresets,
} from "../../../features/vibrato/vibratoPresetList";
import { SYSTEM_VIBRATO_PRESETS } from "../../../features/vibrato/systemPresets";
import {
    mergeImportedPresets,
    parseVibratoPresets,
    serializeVibratoPresets,
    vibratoPresetFileName,
    vibratoPresetSignature,
} from "../../../features/vibrato/vibratoPresetFile";
import { shapeUsesSkew } from "../../../features/vibrato/vibratoCycle";
import { depthStepUnitFor } from "../../../features/vibrato/vibratoDepth";
import { estimateCycles } from "../../../features/vibrato/vibratoCurve";
import { planVibratoTarget } from "../../../features/vibrato/vibratoPitch";
import { exportVibratoPresetsJson } from "../../../services/api/jsonExport";
import {
    buildAuditionCurve,
    buildContourAuditionPair,
    vibratoAudition,
} from "../../../features/vibrato/vibratoAudition";
import type { CycleSource, VibratoPreset, WaveShape } from "../../../features/vibrato/vibratoTypes";
import { type AppMenuItemSpec } from "../../../ui";
import { useInputModifiers } from "../../../ui/useInputModifiers";
import { createAxisGainState2D, type AxisGainState2D } from "../../../utils/axisGain";
import { registerDragAbort } from "../../../utils/gestureFocusGuard";
import {
    type VibratoPreviewGestureInfo,
    type VibratoPreviewInputState,
} from "../vibrato/VibratoPreviewCanvas";
import {
    type VibratoAppliedStatus,
    type VibratoAuditionKind,
    type VibratoPreviewTab,
} from "../vibrato/VibratoPreviewPane";
import {
    PREVIEW_DEFAULT,
    buildAppliedPreview,
    buildVibratoPreview,
    depthForParam,
    fitPreviewRangeCents,
    vibratoPresetLabel,
} from "../vibrato/vibratoDialogLogic";
import {
    advancePreviewFineDrag,
    applyPreviewGesture,
    createPreviewFineDragState,
    cycleWidthPxFor,
    handleLayoutFor,
    type PreviewFineDragState,
    type PreviewGestureSnapshot,
    type PreviewZone,
} from "../vibrato/vibratoPreviewGestures";
import { tableFromCycle } from "../vibrato/vibratoCycleEdit";
import {
    REORDER_DRAG_THRESHOLD_PX,
    reorderAutoScrollDelta,
    reorderInsertionIndex,
    reorderTargetIndex,
} from "../dragReorder";
import type { PresetGroup, VibratoDialogProps } from "./types";

const selectVibratoDialogSession = (state: RootState) => {
    const session = state.session;
    return {
        activeVibratoPresetId: session.activeVibratoPresetId,
        builtinVibratoPresetOrder: session.builtinVibratoPresetOrder,
        disabledVibratoPresetIds: session.disabledVibratoPresetIds,
        penInput: session.penInput,
        vibratoBaseline: session.vibratoBaseline,
        vibratoPresets: session.vibratoPresets,
    };
};

export function useVibratoDialogModel({
    open,
    onOpenChange,
    editParam = "pitch",
    paramRange,
    initialPresetId,
    applyTarget,
}: VibratoDialogProps) {
    const dispatch = useAppDispatch();
    const { t, plural } = useI18n();
    // 只选取本组件实际消费的字段子集并以 shallowEqual 比较：播放期间 playheadSec
    // 每 ~33ms 变一次、session 对象引用随之失效，直接订阅 state.session 会让整个
    // 对话框（含预设列表 / 周期编辑器 / 预览画布）以 ≥30Hz 空转重渲。
    const session = useAppSelector(selectVibratoDialogSession, shallowEqual);
    /** 「精细调整」修饰键（默认 `Ctrl` / macOS `Command`）：预览拖拽时缩到 1/10。 */
    const paramFineAdjustKb = useAppSelector((state: RootState) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );
    /**
     * 设备手感倍率（压感 / 触摸斜坡 / 触控板补偿）。
     *
     * 与上面的修饰键互补：那个是用户按下的离散键盘信号，这个是设备自带的连续
     * 信号。两者串联，因此"按着 Ctrl 的同时压感也在起作用"是自然成立的。
     */
    const inputModifiers = useInputModifiers();

    const resolved = useMemo(
        () => resolveVibratoPresets(session.vibratoPresets, session.builtinVibratoPresetOrder),
        [session.vibratoPresets, session.builtinVibratoPresetOrder],
    );

    /**
     * 播种用的那条预设：调用方指定的（提取出来的）→ 当前活动预设 → 列表首项。
     *
     * 【为什么在渲染里算一次、而不是在 effect 里播种】宿主每次打开都给本组件换一个
     * `key`（见 PianoRollPanel），于是"打开即播种"由**重新挂载**完成 —— 惰性初值就
     * 够了，不需要 effect 内 setState（那会级联渲染，也被 lint 禁止），也不需要
     * "这次打开是否播种过"的 ref（那个 ref 存在的唯一理由，是"保存 / 切换预设会重算
     * `resolved.all`、从而误触发播种"；重挂载下不存在这种误触发）。
     */
    const seedPreset =
        resolved.all.find((preset) => preset.id === (initialPresetId ?? "")) ??
        resolved.all.find((preset) => preset.id === session.activeVibratoPresetId) ??
        resolved.all[0] ??
        null;
    /** 编辑中的草稿。`null` 表示库里一条预设都没有（`resolved.all` 为空）。 */
    const [draft, setDraft] = useState<VibratoPreset | null>(seedPreset);
    /**
     * 已经为哪一份系统预设草稿尝试过"立刻另存为副本"。
     *
     * 【为什么需要】已达预设上限时副本造不出来，草稿会一直停在"系统预设 + 有改动"
     * 上；每次编辑都提示一遍就成了噪声。记下"草稿 id + 当时的用户预设条数"，
     * 同一种情况只尝试一次；用户腾出空位（条数变化）后自动重试。
     */
    const promoteAttemptRef = useRef<string | null>(null);
    /**
     * 刚创建的自定义预设 id：等它渲染进列表后把列表滚到最底部（见下面的 effect）。
     *
     * 【为什么要滚】新预设一律追加在用户段末尾（`upsertVibratoPreset` 对未知 id 是
     * `push`），列表一长它就落在视野之外 —— 用户看不到自己刚做出来的东西。新建 /
     * 复制为自定义 / 从选区提取 / 导入 / 编辑系统预设这五条路都要走一遍。
     */
    const pendingScrollPresetRef = useRef<string | null>(null);

    /** 记下"这条刚做出来的预设，等它进列表后滚过去"（见上面的 ref）。 */
    function markCreatedPresetForScroll(presetId: string) {
        pendingScrollPresetRef.current = presetId;
    }
    /**
     * 手绘周期编辑器的展开状态。
     *
     * `baseline` = 进入手绘前的周期来源，「复位」回到它（采样成表后停留在编辑器里）。
     * 切换预设 / 改变形状时收起，避免"编辑器还开着、草稿却已经换人"的错位。
     */
    const [handDraw, setHandDraw] = useState<{
        /** 进入手绘前的波形来源（「复位到旧形状」回到它）。 */
        baseline: CycleSource;
        /**
         * 下拉框里当前选中的参数式形状（「复位到{形状}」用它）。
         *
         * 单独记一份：进入手绘后 `cycle` 已经是表，光看草稿推不出"当前形状"——那样
         * 按钮只能退化成固定的"复位到正弦"，与用户看到的形状对不上。
         */
        shape: WaveShape;
        skew: number;
    } | null>(null);
    const [deleteTarget, setDeleteTarget] = useState<VibratoPreset | null>(null);
    /**
     * 预览页签。
     *
     * 初值跟着 `applyTarget` 走：「添加颤音」那一次打开的目的就是"看套上去什么样、
     * 然后应用"，直接落在套用页；"管理预设"那一面根本没有这一页。
     */
    const [previewTab, setPreviewTab] = useState<VibratoPreviewTab>(
        applyTarget ? "applied" : "preset",
    );
    /**
     * 选区帧值（套用页签的输入）。
     *
     * `null` = 还没取到（或取不到）：前者显示"载入中"，后者显示"这段选区里没有…"。
     */
    const [original, setOriginal] = useState<{ values: number[]; framePeriodMs: number } | null>(
        null,
    );
    /**
     * 是否正在取选区帧值。
     *
     * 初值就按"有宿主 = 正在载入"来定：取数据是异步的，若初值为 `false`，第一帧会
     * 先画一句"选中一段后可在此预览效果"，再被结果替换掉 —— 一次可见的闪烁。
     */
    const [originalLoading, setOriginalLoading] = useState(Boolean(applyTarget?.loadOriginal));
    /**
     * 套用预览的纵轴拟合令牌：每点一次「适应」自增，让下面的 `useMemo` 重算标尺。
     *
     * 用令牌而不是"拟合函数 + state"，是为了把**什么时候重算**直接写在依赖数组里
     * （见下），读的人不必去追 effect 的触发条件。预设页签那把标尺是 state
     * （`previewHalfCents`），因为它由离散动作显式拟合；两者形态不同只是因为历史，
     * 语义是同一条："编辑期间标尺不动"。
     */
    const [appliedRefitToken, setAppliedRefitToken] = useState(0);
    /**
     * 预设列表行的右键菜单（视口坐标 + 目标预设）。
     *
     * 列表行能做的动作（启用 / 停用、设为当前、复制、删除）散落在页脚与编辑器里，
     * 右键菜单把它们按"针对这一条预设"收拢到指针处。
     */
    const [presetMenu, setPresetMenu] = useState<{
        x: number;
        y: number;
        preset: VibratoPreset;
    } | null>(null);
    /**
     * 用户预设列表的拖拽排序状态（`null` = 没在拖）。
     *
     * `insertionIndex` 是**当前顺序**里的插入位置，用于画落点指示线；换算成
     * `reorderVibratoPreset` 需要的下标在松手时做（见 `reorderTargetIndex`）。
     */
    const [presetDrag, setPresetDrag] = useState<{
        group: PresetGroup;
        id: string;
        fromIndex: number;
        insertionIndex: number;
    } | null>(null);
    /** 按下的起点（尚未越过阈值）。用 ref 而不是 state：指针移动每帧都读它。 */
    const presetDragStartRef = useRef<{
        group: PresetGroup;
        id: string;
        fromIndex: number;
        pointerY: number;
    } | null>(null);
    /** 进行中的拖拽（与 `presetDrag` 同步，供事件回调读取最新值）。 */
    const presetDragRef = useRef<{
        group: PresetGroup;
        id: string;
        fromIndex: number;
        insertionIndex: number;
    } | null>(null);
    /** 最近一次拖拽指针 Y：自动滚动的 rAF 循环每帧读它（指针不动时也要继续滚）。 */
    const presetDragPointerYRef = useRef<number | null>(null);
    /** 本次指针交互是否已经变成拖拽 —— 用于吞掉拖完那一下的 click。 */
    const suppressPresetClickRef = useRef(false);
    /** 两组列表各自的容器：拖拽落点只看**同组**的行。 */
    const systemListRef = useRef<HTMLDivElement | null>(null);
    const userListRef = useRef<HTMLDivElement | null>(null);
    /**
     * 正在试听的是哪一条（`null` = 没在响）。
     *
     * 用"哪一条"而不是布尔量：三个按钮（预设 / 原参数线 / 新参数线）各自要显示自己的
     * 播放态，而试听器同一时刻只允许一路声音。一个三值 state 与那条**唯一**的不变量
     * 同形；用两个独立 state 会允许"两个按钮同时显示正在播放"这种不可能的状态。
     */
    const [audition, setAudition] = useState<VibratoAuditionKind | null>(null);
    /** 导入 / 导出用的隐藏文件输入（必须常驻挂载，见 `AppFileInput` 的说明）。 */
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    /**
     * 导入 / 导出 / 提取后的一条行内反馈。
     *
     * 【为什么是行内而不是模态】导入是高频小操作，模态打断太重；而「什么都不说」
     * 正是主题导入曾经的缺陷 —— 坏文件静默无反应，用户以为导入不起作用。提取失败
     * 同样走这里：它就在窗口里发生，弹一层模态只会盖住刚刚那个窗口。
     */
    const [ioNotice, setIoNotice] = useState<{ text: string; danger: boolean } | null>(null);
    /**
     * 预设波形页签的纵轴半幅（cents）—— **一次性拟合，编辑期间不动**。
     *
     * 【为什么要稳定】若标尺跟着当前深度自适应，波形永远填满画布：调深度时看到的
     * 只是整幅在竖直方向"抖一下"，读不出幅度大小。把标尺固定下来，波形高度就等于
     * 深度，配合轴上的刻度标签可以直读。换预设 / 打开 / 点「适应」/ 保存时才重新
     * 拟合（那是离散动作，不是每帧都在跳）。
     */
    const [previewHalfCents, setPreviewHalfCents] = useState<number>(() =>
        seedPreset
            ? fitPreviewRangeCents(buildVibratoPreview(seedPreset).peakCents)
            : fitPreviewRangeCents(0),
    );

    /** 按给定草稿重新拟合纵轴。 */
    const fitPreviewAxis = useCallback((preset: VibratoPreset | null) => {
        if (!preset) return;
        setPreviewHalfCents(fitPreviewRangeCents(buildVibratoPreview(preset).peakCents));
    }, []);

    const isBuiltin = Boolean(draft?.builtin);
    /** 正在编辑的这条是否就是当前使用的（编辑器顶部的按钮据此显示状态）。 */
    const draftIsActive = Boolean(draft) && session.activeVibratoPresetId === draft?.id;
    const previewSamples = useMemo(() => (draft ? buildVibratoPreview(draft) : null), [draft]);

    /**
     * 取选区帧值 —— 只在有选区宿主时做，且只取一次（每次打开重挂载）。
     *
     * 依赖的是 `applyTarget?.loadOriginal` 本身而不是 `applyTarget` 对象：后者由宿主
     * 每次渲染新建时，这个 effect 会跟着重跑 → `setOriginal` → 再渲染 → 无限循环。
     * 盯住那个函数（宿主用 `useCallback` 稳定它）才是真正的"数据源变了"。
     */
    const loadOriginal = applyTarget?.loadOriginal;
    useEffect(() => {
        if (!open || !loadOriginal) return;
        let cancelled = false;
        // 异步回调里的 setState 不受"effect 内同步 setState"约束（同应用弹窗旧实现）。
        void loadOriginal()
            .then((result) => {
                if (cancelled) return;
                setOriginal(result);
                setOriginalLoading(false);
            })
            .catch((err) => {
                /*
                 * 取选区帧值走 IPC（`getParamFrames`），失败此前没有 catch：`then` 永不
                 * 执行，`originalLoading` 永远停在 true —— 「套用到选区」页签显示一句
                 * 永久的「载入中」，且是一个 unhandled rejection。这里与"取不到数据"
                 * 同路：清掉 loading、`original` 置 null，落到既有的空占位提示。
                 */
                if (cancelled) return;
                console.error("[VibratoDialog] load selection frames failed", err);
                setOriginal(null);
                setOriginalLoading(false);
            });
        return () => {
            cancelled = true;
        };
    }, [open, loadOriginal]);

    /**
     * 套用预览采样：喂**选区真实帧值**，走与落盘同一条 `buildVibratoCurve`。
     *
     * 只画"相对共同中心的偏移"，不画绝对音高 —— 理由见 `buildAppliedPreview` 的
     * 注释（素材轮廓与颤音幅度差两个数量级，同轴会互相压扁）。
     */
    const appliedPreview = useMemo(() => {
        if (!applyTarget || !draft || !original) return null;
        return buildAppliedPreview({
            preset: draft,
            original: original.values,
            param: editParam,
            framePeriodMs: original.framePeriodMs,
            range: paramRange,
            baseline: session.vibratoBaseline,
        });
    }, [applyTarget, draft, original, editParam, paramRange, session.vibratoBaseline]);

    /**
     * 套用页签的纵轴半幅（cents）。
     *
     * 【什么时候重算】依赖数组就是答案：**换选区**（`original`）、**换预设**
     * （`draft?.id`）、**点「适应」**（令牌）。深度 / 速率的改动刻意不在其中 ——
     * 内容会变，但那是"编辑期间"，标尺不动（否则深度一调整幅就重新缩放）。
     */
    const appliedHalfCents = useMemo(
        () => fitPreviewRangeCents(appliedPreview ? appliedPreview.peakCents : 0),
        // eslint-disable-next-line react-hooks/exhaustive-deps -- 内容变化（深度 / 速率）刻意不重算标尺，见上
        [original, draft?.id, appliedRefitToken],
    );

    /** A/B 试听的两条曲线：同一中心、同一基音，差异只来自颤音。 */
    const auditionCurves = useMemo(() => {
        if (!appliedPreview || !original) return null;
        // 整包传进去：不可信的帧（八度跳 / 过渡段）由它自己按最近的音高持续 ——
        // 按原值合成出来是超低频，试听是合成人声，没有"唱 30 Hz"这回事。
        return buildContourAuditionPair(appliedPreview, original.framePeriodMs);
    }, [appliedPreview, original]);

    /**
     * 选区里**没有可调制的音符段**。
     *
     * 音高参数下这意味着无从调制：值为 0 的未检测帧、以及浊清边界上低而非零的过渡帧
     * （跟踪器给的 20~40 Hz 低估）都不是音符；连续有声帧还要够长（见 `vibratoPitch.ts`）。
     * 预览为空，应用也不会改动任何帧 —— 提交侧对每个选区段各自判定。这里只负责把
     * "为什么没有预览"说清楚：把"没数据"和"这段没有音高"混成同一句提示，用户会以为
     * 是自己没选对区域。
     *
     * 【为什么不据此禁用「应用」】预览只取**第一个**选区段；多选区时其余段可能有音符，
     * 按第一段禁用会把本来能应用的选区挡掉。
     */
    const unvoicedPitch =
        editParam === "pitch" &&
        original !== null &&
        planVibratoTarget(editParam, original.values, original.framePeriodMs) === null;

    /** 套用页签画不出东西时的原因（决定占位文案）。 */
    const appliedStatus: VibratoAppliedStatus = originalLoading
        ? "loading"
        : unvoicedPitch
          ? "unvoiced"
          : "empty";

    // 卸载兜底：整个组件被卸载（而不只是关闭）时也要停 —— 否则试听会在
    // 窗口消失之后继续响。这里不含 setState；关闭时的按钮状态复位在
    // `handleOpenChange`（事件处理器）里做，不违反 effect 的规则。
    useEffect(() => () => vibratoAudition.stop(), []);

    /** 关闭对话框：先停试听、复位按钮，再向上传播。 */
    function handleOpenChange(next: boolean) {
        if (!next) {
            vibratoAudition.stop();
            setAudition(null);
        }
        onOpenChange(next);
    }

    /**
     * 播放 / 停止某一条试听。自然结束后引擎回调把按钮切回「播放」。
     *
     * 【为什么切页签也要停】声音与按钮是两处状态，只切页签不停声会出现"看不见的
     * 播放中"—— 用户在新页签上找不到停止按钮，只能靠关窗口。
     */
    function toggleAudition(kind: VibratoAuditionKind) {
        if (audition === kind) {
            vibratoAudition.stop();
            setAudition(null);
            return;
        }
        if (kind === "preset") {
            if (!previewSamples) return;
            // 不变量「听到的 = 看到的」：直接吃预览的同一份数据，不重算。
            const started = vibratoAudition.play(buildAuditionCurve(previewSamples), () =>
                setAudition(null),
            );
            setAudition(started ? kind : null);
            return;
        }
        if (!auditionCurves) return;
        const started = vibratoAudition.play(auditionCurves[kind], () => setAudition(null));
        setAudition(started ? kind : null);
    }

    /** 切换预览页签：先停声（见 `toggleAudition` 的说明），再换页。 */
    function handlePreviewTabChange(next: VibratoPreviewTab) {
        if (next === previewTab) return;
        vibratoAudition.stop();
        setAudition(null);
        setPreviewTab(next);
    }

    /** 草稿的局部更新（不落盘）。 */
    function patch(changes: Partial<VibratoPreset>) {
        setDraft((prev) => (prev ? { ...prev, ...changes } : prev));
    }

    /**
     * 预览画布手势的活动状态（起点快照 + 区域 + 精细调整累计量）。
     *
     * 【为什么快照】主体拖动的深度换算依赖画布**当前**的纵轴标尺，而深度一改
     * 标尺（按峰值自适应）也跟着变；每帧重取会让拖动变成非线性甚至反向。起点
     * 取一次，整段手势按同一套几何走。
     *
     * 【为什么带着精细调整状态】修饰键可以在拖拽途中按下 / 松开，位移必须**按增量**
     * 缩放（见 `advancePreviewFineDrag`），否则切换修饰键的一瞬间累计位移被整体
     * 重算，数值闪回、拖拽被打断。
     */
    const previewGestureRef = useRef<{
        zone: PreviewZone;
        snapshot: PreviewGestureSnapshot;
        fine: PreviewFineDragState;
        /**
         * 设备手感倍率的累计状态（压感 / 触摸斜坡 / 触控板补偿）。
         *
         * 【为什么与 `fine` 分开】两者都是"按增量缩放"，但来源不同：`fine` 是用户
         * 按下的键盘修饰键（离散），这里是设备自带的连续信号。串联两级（先设备、
         * 后修饰键）即可，各自都满足"只乘增量"，因此整体也满足 —— 见
         * `utils/axisGain.ts` 的论证。
         */
        device: AxisGainState2D;
    } | null>(null);

    /**
     * 预设波形页签的可见窗口时长（ms）：与 `buildVibratoPreview` 的默认几何一致。
     */
    const presetWindowMs = (PREVIEW_DEFAULT.frameCount - 1) * PREVIEW_DEFAULT.framePeriodMs;
    /**
     * 套用页签的可见窗口时长（ms）：**选区自己的时长**。
     *
     * 【为什么不能沿用预设窗口】渐入 / 渐出与相位都以"整段时长"为基准：手柄画在
     * `attackMs / windowMs` 处，水平拖动按 `dx/width × windowMs` 换算。套用预览画的是
     * 选区真实帧数（最多 400 帧、帧周期也来自工程），与预设的 320 帧 × 5ms 不是一回事
     * —— 拿预设窗口去换算，手柄会画错位置，拖一下改的毫秒数也是错的。
     */
    const appliedWindowMs =
        original && original.values.length >= 2
            ? (original.values.length - 1) * original.framePeriodMs
            : presetWindowMs;
    /**
     * **当前页签**的窗口时长 —— 手势换算按它取。
     *
     * 取一次就够：快照在手势起点建立（见 `handlePreviewGestureStart`），整段手势沿用
     * 同一个窗口；而切页签必须先松手（段控在画布之外）。
     */
    const activeWindowMs =
        previewTab === "applied" && applyTarget ? appliedWindowMs : presetWindowMs;

    /**
     * 画布手势开始：登记起点快照。
     *
     * 系统预设同样可拖：拖改的是**本地草稿**，落盘时变成一份副本（见 `commitDraft`）——
     * 出厂预设本身不会被碰。
     *
     * 【窗口时长按当前页签取】渐入 / 渐出的换算是"位移占整段时长的比例"，而两个页签的
     * 时间轴不是一回事：预设波形是固定的 320 帧窗口，套用预览画的是**选区真实帧数**。
     * 用错窗口会让手柄画错位置、拖一下改的毫秒数也是错的（见 `activeWindowMs`）。
     */
    function handlePreviewGestureStart(zone: PreviewZone, info: VibratoPreviewGestureInfo) {
        if (!draft) return;
        const windowMs = activeWindowMs;
        const fineActive = isModifierActive(paramFineAdjustKb, info.modifiers);
        previewGestureRef.current = {
            zone,
            snapshot: {
                attackMs: draft.attackMs,
                releaseMs: draft.releaseMs,
                depthCents: draft.depthCents,
                startPhaseDeg: draft.startPhaseDeg,
                windowMs,
                widthPx: info.width,
                cycleWidthPx: cycleWidthPxFor(draft, info.width, windowMs),
                /*
                 * 纵向灵敏度取**画布自己的**标尺（`info.centsPerPx`），于是两个页签
                 * 遵守同一条规则："拖多少像素，线就走多少像素"。
                 *
                 * 【为什么不改成统一按预设的幅度定标】套用页签的标尺要容下素材轮廓，
                 * 通常比预设波形宽一两个数量级 —— 换算法会让同一次拖动在两个页签里
                 * 改出完全不同的幅度，而"看到的那条线跟着手指走"才是用户预期的手感。
                 * 觉得太粗时按「精细调整」修饰键（1/10），或先点「适应」。
                 */
                centsPerPx: info.centsPerPx,
            },
            fine: createPreviewFineDragState(fineActive),
            device: createAxisGainState2D(),
        };
        // 手势开始：重置压感标定（每段手势重新观察，不把上一次的极端值固化下来）。
        inputModifiers.reset();
    }

    /** 画布手势移动：位移 → 草稿字段（换算规则见 `vibratoPreviewGestures`）。 */
    function handlePreviewGestureMove(
        deltaX: number,
        deltaY: number,
        modifiers: VibratoPreviewInputState,
    ) {
        const gesture = previewGestureRef.current;
        if (!gesture) return;
        /*
         * 两级增量缩放，顺序固定：**先设备倍率、后修饰键**。
         *
         * 设备倍率（压感 / 触摸斜坡）由 `useInputModifiers` 给出，它内部维护了
         * "上一帧的原始累计值"，因此换倍率时已累计的部分原封不动。
         */
        const gained = inputModifiers.advanceXY(
            gesture.device,
            deltaX,
            deltaY,
            modifiers,
            Math.hypot(deltaX - gesture.device.x.raw, deltaY - gesture.device.y.raw),
        );
        // 「精细调整」按增量缩放位移：中途按下 / 松开只改变此后的速度，累计量连续。
        const scaled = advancePreviewFineDrag(
            gesture.fine,
            gained.x,
            gained.y,
            isModifierActive(paramFineAdjustKb, modifiers),
        );
        const next = applyPreviewGesture(
            gesture.zone,
            gesture.snapshot,
            scaled.deltaX,
            scaled.deltaY,
        );
        /*
         * 笔杆倾斜 → 波形偏斜。
         *
         * 【为什么只在形状真的用偏斜时才写】`skew` 字段对所有形状都存在，但
         * `sine` / `saw` / 手绘表完全忽略它。若无条件写入，用户在一个正弦预设上
         * 拖一下就会把偏斜悄悄改掉 —— 直到他之后换成三角波才会发现形状不对。
         * `shapeUsesSkew` 是"会不会用到"的唯一出处。
         */
        const skew = inputModifiers.tiltSkewFor(modifiers);
        if (skew != null && draft?.cycle.kind === "shape" && shapeUsesSkew(draft.cycle.shape)) {
            patch({ ...next, cycle: { ...draft.cycle, skew } });
            return;
        }
        patch(next);
    }

    /** 画布手势结束。 */
    function handlePreviewGestureEnd() {
        previewGestureRef.current = null;
    }

    /**
     * 预览画布捏合调深度。
     *
     * 画布已经把"格"折算成 cents（与纵向拖拽同一条换算，见 `VibratoPreviewCanvas`
     * 的 `PINCH_DRAG_PX_PER_NOTCH`），这里只负责钳制与写入。
     *
     * 【为什么用函数式 `setDraft`】捏合是一串高频事件，可能在同一帧内到达多次；
     * 读闭包里的 `draft` 会拿到同一份旧值，增量因此互相覆盖（表现为"捏了没反应"）。
     */
    function handlePreviewPinchDepth(deltaCents: number) {
        if (!Number.isFinite(deltaCents) || deltaCents === 0) return;
        setDraft((prev) =>
            prev
                ? {
                      ...prev,
                      depthCents: Math.min(
                          VIBRATO_LIMITS.depthCents.max,
                          Math.max(VIBRATO_LIMITS.depthCents.min, prev.depthCents + deltaCents),
                      ),
                  }
                : prev,
        );
    }

    const persistPreset = useCallback(
        (preset: VibratoPreset) => {
            dispatch(upsertVibratoPreset(preset));
            void dispatch(persistUiSettings());
        },
        [dispatch],
    );

    /**
     * 草稿相对"它该等于什么"是否有未保存的改动。
     *
     * - 用户预设：与库里同 id 的那条比；列表里找不到同 id，说明是刚新建、还没入库的
     *   预设，也算有改动。
     * - 系统预设：与**出厂定义**比。
     *
     * 【为什么系统预设也有"改动"】参数是可以改的（改的是本地草稿），只是落盘时不能
     * 覆盖出厂预设 —— 于是"有改动"的含义变成"存下去要生成一份副本"。签名忽略 `id`
     * 与 `builtin`（见 `vibratoPresetSignature`），因此没动过的草稿与定义完全相同，
     * 保存就是个空操作。
     */
    function draftHasUnsavedChanges(): boolean {
        if (!draft) return false;
        const normalized = sanitizeVibratoPreset(draft);
        if (isBuiltin) {
            const definition = SYSTEM_VIBRATO_PRESETS.find((preset) => preset.id === draft.id);
            if (!definition) return false;
            return (
                vibratoPresetSignature(normalized) !==
                vibratoPresetSignature(sanitizeVibratoPreset(definition))
            );
        }
        const stored = resolved.user.find((preset) => preset.id === draft.id);
        if (!stored) return true;
        return vibratoPresetSignature(normalized) !== vibratoPresetSignature(stored);
    }

    /**
     * 把一条预设另存为自定义副本。
     *
     * 名字按**显示名**预填，并避开已占用的名字：系统预设的 `name` 是空的（名字走
     * 词条），拿它当基底会得到一个无名副本；编号避开现有全部显示名，连点两次才不会
     * 叠成"名字 2 2"。
     *
     * @returns 已达上限（`MAX_VIBRATO_PRESETS`）时返回 `null`，由调用方给出提示 ——
     *          静默失败会让用户以为"保存没反应"。
     */
    const duplicateAsCustom = useCallback(
        (source: VibratoPreset): VibratoPreset | null => {
            if (resolved.user.length >= MAX_VIBRATO_PRESETS) return null;
            return duplicateVibratoPreset(
                source,
                nextDuplicatePresetName(
                    vibratoPresetLabel(source, t) || t("vibrato_manager_new"),
                    resolved.all.map((preset) => vibratoPresetLabel(preset, t)),
                ),
            );
        },
        [resolved, t],
    );

    /**
     * 把一条预设**入库为自定义副本**，返回那份副本（已达上限时返回 `null` 并给出提示）。
     *
     * 出厂预设永远不能被覆盖（要可复原），所以"存成自己的"这件事只有这一条路。
     */
    const saveAsCustom = useCallback(
        (source: VibratoPreset): VibratoPreset | null => {
            const copy = duplicateAsCustom(sanitizeVibratoPreset(source));
            if (!copy) {
                setIoNotice({ text: t("vibrato_manager_at_cap"), danger: true });
                return null;
            }
            persistPreset(copy);
            return copy;
        },
        [duplicateAsCustom, persistPreset, t],
    );

    /**
     * 把草稿的**未保存改动**落盘，返回落盘后的那条预设（没有改动时返回 `null`）。
     *
     * 【系统预设的落盘 = 另存为副本】对它的改动落到一份副本上（新 id、`builtin: false`、
     * 名字按显示名预填），出厂预设本身不会被碰。
     *
     * 【只应用、不保存】「应用」走的是另一条路（`handleApply` 直接把草稿交给编辑
     * 管线），因此"基于系统预设调两个参数然后应用"不会在库里留下任何东西。
     *
     * 【"没有改动"时为什么什么都不做】这条路是"离开这条预设 = 落盘"用的：在列表里
     * 点着看一遍就凭空多出十几条副本，比丢掉改动更糟。「保存」不走这条判断 ——
     * 见 `handleSave`。
     */
    function commitDraft(): VibratoPreset | null {
        if (!draft || !draftHasUnsavedChanges()) return null;
        if (isBuiltin) {
            // 应用窗口里草稿只服务这一次套用：切换预设不该顺手在库里留下一条用户
            // 没打算要的副本 —— 那正是管理窗口这次要修掉的旧毛病。管理窗口里系统
            // 预设的改动早在第一次编辑时就变成副本了，走不到这里。
            return applyTarget ? null : saveAsCustom(draft);
        }
        const normalized = sanitizeVibratoPreset(draft);
        persistPreset(normalized);
        return normalized;
    }

    /**
     * 选中一个预设进行编辑。
     *
     * 【切走时先落盘】用户在 A 上改了一半、切到 B 看看，若改动被直接丢弃，
     * "编辑途中不能换预设"就成了硬伤。因此切换前先把 A 的未保存改动落盘（系统预设
     * 落成一份副本）—— 与「保存」同一套路径。
     */
    function selectPreset(preset: VibratoPreset) {
        if (draft && draft.id !== preset.id) commitDraft();
        setHandDraw(null);
        setDraft(preset);
        // 换预设 = 换一段波形，纵轴跟着重新拟合（编辑期间则保持不动）。
        fitPreviewAxis(preset);
    }

    /** 进入手绘：把当前波形采样成表作为起点（改形比从零画顺手）。 */
    function openHandDraw() {
        if (!draft) return;
        const baseline = draft.cycle;
        // 记下"当前形状"：来自参数式就用它，来自表（提取 / 上次手绘）则兜底正弦 ——
        // 与形状下拉的显示口径一致。
        const shape: WaveShape = baseline.kind === "shape" ? baseline.shape : "sine";
        const skew = baseline.kind === "shape" ? baseline.skew : 0.5;
        const table = baseline.kind === "table" ? [...baseline.table] : tableFromCycle(baseline);
        setHandDraw({ baseline, shape, skew });
        patch({ cycle: { kind: "table", table } });
    }

    /**
     * 设为当前使用（列表行上的圆点、右键菜单、编辑器顶部按钮都用它）。
     *
     * 【已经是它就直接返回】列表行上的圆点是个可反复点的控件，重复点击不该产生
     * 一次无谓的设置写入；同时这也让"当前使用中"那个按钮的禁用态与行为一致。
     */
    function activatePreset(preset: VibratoPreset) {
        if (session.activeVibratoPresetId === preset.id) return;
        dispatch(setActiveVibratoPreset(preset.id));
        void dispatch(persistUiSettings());
    }

    /**
     * 启用 / 停用一条预设。
     *
     * 只影响本机的工具栏列表与拖拽中的循环切换 —— 预设本身、以及"当前使用"的
     * 选择都不受影响，因此不需要动活动预设。
     */
    function togglePresetEnabled(preset: VibratoPreset) {
        dispatch(toggleVibratoPresetEnabled(preset.id));
        void dispatch(persistUiSettings());
    }

    /** 复制一份预设（页脚与列表右键菜单共用）。复制的是**入库的那条**，不是草稿。 */
    function duplicatePreset(source: VibratoPreset) {
        const copy = duplicateAsCustom(source);
        if (!copy) {
            setIoNotice({ text: t("vibrato_manager_at_cap"), danger: true });
            return;
        }
        persistPreset(copy);
        selectPreset(copy);
        activatePreset(copy);
        markCreatedPresetForScroll(copy.id);
    }

    /**
     * 「保存」：把草稿落盘，并让草稿跟着落到那份上。
     *
     * 【系统预设：改没改过都另存为副本】用户按下保存就是想要一份**自己能改的**副本 ——
     * "没动过就不给存"只会让人以为按钮坏了（而它明明是可点的）。于是这条路不看
     * `draftHasUnsavedChanges`：按下就存。用户预设仍然只在有改动时写回库（没有改动
     * 就没有可写的东西）。
     *
     * 【为什么草稿要跟着切】对系统预设来说，落盘产生的是**另一条**预设（新 id）。
     * 草稿若还停在原来的 id 上，用户接着调、再按保存就会又生成一份副本 —— 那不是
     * "保存"，是"每次都新建"。切过去之后，第二次保存就是更新那一份。
     *
     * 【为什么顺手设为当前使用】与「复制为自定义」同一套：用户按保存表达的是
     * "我要这一版"，把它设为当前使用才符合意图（拖拽 / 菜单都用当前使用的那条）。
     */
    function handleSave() {
        if (!draft) return;
        const saved = isBuiltin ? saveAsCustom(draft) : commitDraft();
        if (!saved) return;
        setDraft(saved);
        if (isBuiltin) activatePreset(saved);
        // 保存是"这段波形定稿了"的时机：顺势把纵轴重新拟合回六成上下。
        fitPreviewAxis(saved);
    }

    /**
     * 待办的"编辑系统预设 → 立刻另存为副本"。
     *
     * 【为什么算在渲染里、做在 effect 里】造副本是纯计算（草稿 + 现有名字），入库才是
     * 副作用。编辑入口（捏合、手绘）会在同一帧里连续改草稿，若在编辑回调里顺手入库，
     * 一次捏合就会生成一串副本；effect 每次提交只跑一次，dispatch 也不在渲染中，
     * 天然安全。
     */
    const builtinEditPending =
        // 应用窗口除外：那里的草稿只服务"这一次套用"，编辑它不该在库里留下东西
        // （「应用」不写库是那个窗口的契约，见文件头）。
        !applyTarget && draft && draft.builtin && draftHasUnsavedChanges() ? draft : null;

    /**
     * 对系统预设的**任意编辑**：立刻在列表里生成一份自定义副本，并把草稿切过去。
     *
     * 【旧行为的问题】副本原本是在"离开这条预设"（切到别的预设 / 删除 / 导入）时才
     * 生成的。用户改完系统预设 A、点向 B 的那一刻库里才多出一条 C —— 他的注意力全在
     * B 上，C 于是成了"不经意间添加的预设"。改成第一次编辑就生成并选中，用户当场看见
     * 它出现。
     *
     * 【为什么不打断编辑】副本的内容就是"系统预设 + 这次改动"，草稿换的只是 id 与
     * 名字；拖动 / 输入依赖的是草稿的**内容**，因此手势、光标位置与纵轴标尺都不动
     * （刻意不调 `fitPreviewAxis`：编辑期间标尺本就该保持不动）。
     *
     * 【为什么不设为当前使用】激活预设会改掉颤音工具下一笔画什么，那是用户自己的
     * 决定（双击 / 右键「设为当前使用」）；这里只负责让副本可见。
     */
    useEffect(() => {
        if (!builtinEditPending) return;
        const attemptKey = `${builtinEditPending.id}:${resolved.user.length}`;
        if (promoteAttemptRef.current === attemptKey) return;
        promoteAttemptRef.current = attemptKey;
        const copy = saveAsCustom(builtinEditPending);
        if (!copy) return;
        setDraft(copy);
        markCreatedPresetForScroll(copy.id);
    }, [builtinEditPending, resolved.user.length, saveAsCustom]);

    /**
     * 新预设落进列表后，把预设列表滚到最底部 —— 它就在那儿，滚过去用户才看得见。
     *
     * 【为什么单独一个 effect、而不是在创建处顺手滚】创建那一刻只是 dispatch，列表
     * 还没重新渲染，`scrollHeight` 里还没有新行；等 `draft` 切到新预设上的这次提交，
     * 行才真的在 DOM 里。滚的是 Radix 的滚动视口（`closest` 从用户段往上找），
     * 不是列表容器本身。
     */
    useEffect(() => {
        const id = pendingScrollPresetRef.current;
        if (!id || draft?.id !== id) return;
        pendingScrollPresetRef.current = null;
        const viewport = userListRef.current?.closest<HTMLElement>(
            "[data-radix-scroll-area-viewport]",
        );
        if (viewport) viewport.scrollTop = viewport.scrollHeight;
    }, [draft]);

    /**
     * 「应用」：把当前草稿交给编辑管线，然后**关窗**。
     *
     * 【为什么传完整预设而不是 id】本地微调（深度 / 速率 / 波形）必须跟着过去 ——
     * 只传 id 的话落盘的还是库里的旧参数，用户看到的与得到的不一致。
     *
     * 【为什么不写库】写库是「保存」的事（两个动作语义不同，见文件头）。应用落下的
     * 是一次离散的编辑操作，不该顺手改掉库里的预设。
     *
     * 【为什么关窗】「应用」是「添加颤音」这次打开的终点 —— 用户点它就是表达"就这样，
     * 落到选区上"。留着窗口等于把"完成"变成"又一次操作"，用户还得再找一次关闭。
     * （"管理预设"那一面没有这个动作，也就没有这条路径。）
     */
    function handleApply() {
        if (!draft || !applyTarget) return;
        applyTarget.onApply(sanitizeVibratoPreset(draft));
        handleOpenChange(false);
    }

    /**
     * 「从选区提取颤音预设」。
     *
     * 【为什么由宿主做重活】提取要读参数帧（选区 + 轨道 + 参数），只有宿主够得着；
     * 本窗口只负责"把结果显示出来"：成功就选中刚入库的那条（用户接着在这儿改名 /
     * 微调），失败给一条行内提示。
     *
     * 【为什么不再跳窗口】合并前提取完要开一次管理器（因为提取的结果属于"改库"那
     * 一侧）；现在两侧同窗，提取 = 列表里多一条并选中它。
     */
    async function handleExtractFromSelection() {
        if (!applyTarget?.onExtract) return;
        setIoNotice(null);
        let extracted: VibratoPreset | null;
        try {
            extracted = await applyTarget.onExtract();
        } catch (err) {
            // 宿主提取走 IPC（读参数帧），失败此前会变成 unhandled rejection：
            // 按钮复位、对话框停在原地、错误只进控制台。与"提取到空结果"同路
            // 反馈给用户（行内提示，不弹窗）。
            console.error("[VibratoDialog] extract from selection failed", err);
            setIoNotice({ text: t("vibrato_extract_failed"), danger: true });
            return;
        }
        if (!extracted) {
            setIoNotice({ text: t("vibrato_extract_failed"), danger: true });
            return;
        }
        /*
         * 选中新提取的那条。
         *
         * 不走 `selectPreset`：那个函数会先把**当前草稿**的未保存改动落盘（"离开这条
         * 预设 = 落盘"）—— 这里用户并没有离开，是列表多了一条。于是直接换草稿，并把
         * 纵轴按新波形拟合一次（换预设的既定动作）。
         */
        setHandDraw(null);
        setDraft(extracted);
        fitPreviewAxis(extracted);
        markCreatedPresetForScroll(extracted.id);
    }

    function handleDuplicate() {
        if (!draft) return;
        duplicatePreset(draft);
    }

    /**
     * 导出**当前选中的**预设。
     *
     * 系统预设也允许导出 —— 导出只是序列化，不改任何数据；顺带成了查看系统
     * 预设原始参数的口子。
     */
    async function handleExport() {
        if (!draft) return;
        setIoNotice(null);
        const text = serializeVibratoPresets([draft]);
        const result = await exportVibratoPresetsJson(text, vibratoPresetFileName(draft));
        // 用户在原生对话框里取消不是错误 —— 不提示（与其他导出一致）。
        if (result.canceled) return;
        if (!result.ok) {
            setIoNotice({ text: t("vibrato_io_read_failed"), danger: true });
        }
    }

    /** 导入预设文件：净化、去重、入库，并把草稿切到最后导入的那条。 */
    async function handleImport(files: File[]) {
        const file = files[0];
        if (!file) return;
        setIoNotice(null);
        let text: string;
        try {
            text = await file.text();
        } catch {
            setIoNotice({ text: t("vibrato_io_read_failed"), danger: true });
            return;
        }
        const parsed = parseVibratoPresets(text);
        if (!parsed.ok) {
            const key =
                parsed.error === "newerVersion"
                    ? "vibrato_io_newer_version"
                    : parsed.error === "noPresets"
                      ? "vibrato_io_empty"
                      : // badJson 与 wrongKind 对用户是同一句话：这不是我们的文件 /
                        // 读不出来 —— 分开说只会让人更困惑。
                        "vibrato_io_wrong_kind";
            setIoNotice({ text: t(key), danger: true });
            return;
        }
        if (parsed.presets.length === 0) {
            setIoNotice({ text: t("vibrato_io_empty"), danger: true });
            return;
        }

        const capacity = MAX_VIBRATO_PRESETS - session.vibratoPresets.length;
        const { imported, skipped } = mergeImportedPresets(
            parsed.presets,
            session.vibratoPresets,
            capacity,
        );
        if (imported.length === 0) {
            // 全部是重复：不写库，但要把"跳过 n 条"说出来 —— 否则同样是"点了没反应"。
            setIoNotice({
                text: `${plural("vibrato_io_imported", 0)}${plural("vibrato_io_skipped", skipped)}`,
                danger: false,
            });
            return;
        }

        for (const preset of imported) {
            dispatch(upsertVibratoPreset(preset));
        }
        const last = imported[imported.length - 1];
        if (last) {
            dispatch(setActiveVibratoPreset(last.id));
            setDraft(last);
            markCreatedPresetForScroll(last.id);
        }
        void dispatch(persistUiSettings());
        setIoNotice({
            text:
                plural("vibrato_io_imported", imported.length) +
                (skipped > 0 ? plural("vibrato_io_skipped", skipped) : ""),
            danger: false,
        });
    }

    function handleCreate() {
        const created = sanitizeVibratoPreset({
            id: createVibratoPresetId(),
            name: t("vibrato_manager_new"),
        });
        persistPreset(created);
        selectPreset(created);
        markCreatedPresetForScroll(created.id);
    }

    function handleDelete() {
        if (!deleteTarget) return;
        const removedId = deleteTarget.id;
        const deletingDraft = draft?.id === removedId;
        /*
         * 删的若是**正在编辑的那条**，草稿不能就这么置空 —— 那会把编辑器丢进"未选择
         * 任何预设"的空状态（并显示"还没有自定义预设"，而库里明明还有一堆）。活动 id
         * 已由 reducer 迁移到仍然存在的预设上，这里用**同一个** `activeIdAfterRemoval`
         * 算出迁移目标，把草稿切过去，与库里的状态保持一致。
         */
        const nextActiveId = activeIdAfterRemoval(
            resolved.all,
            session.activeVibratoPresetId,
            removedId,
        );
        dispatch(removeVibratoPreset(removedId));
        void dispatch(persistUiSettings());
        setDeleteTarget(null);
        if (!deletingDraft) return;
        const remaining = resolved.all.filter((preset) => preset.id !== removedId);
        const next = findVibratoPreset(remaining, nextActiveId) ?? remaining[0] ?? null;
        setDraft(next);
        fitPreviewAxis(next);
    }

    /**
     * 上移 / 下移一条预设。
     *
     * 右键菜单里的这条是**键盘可达的等价操作** —— 列表靠拖拽排序，只为了拖拽就砍掉
     * 非指针用户的路子代价太大。两组各自在自己的集合里换位。
     */
    function movePreset(preset: VibratoPreset, delta: 1 | -1) {
        const isSystem = isBuiltinVibratoPresetId(preset.id);
        const list = isSystem ? resolved.system : resolved.user;
        const index = list.findIndex((item) => item.id === preset.id);
        if (index < 0) return;
        const next = index + delta;
        if (next < 0 || next >= list.length) return;
        dispatch(
            isSystem
                ? reorderBuiltinVibratoPreset({ id: preset.id, toIndex: next })
                : reorderVibratoPreset({ id: preset.id, toIndex: next }),
        );
        void dispatch(persistUiSettings());
    }

    /** 某一组预设各行的中线（视口 Y），按当前顺序 —— 拖拽落点据此判定。 */
    const presetRowCenters = useCallback((group: PresetGroup): number[] => {
        const container = group === "system" ? systemListRef.current : userListRef.current;
        if (!container) return [];
        return Array.from(container.querySelectorAll<HTMLElement>("[data-preset-row]")).map(
            (row) => {
                const rect = row.getBoundingClientRect();
                return rect.top + rect.height / 2;
            },
        );
    }, []);

    /** 按指针 Y 更新拖拽落点（自动滚动推进后也要重算，指示线才跟得上）。 */
    const applyPresetDragAtPointer = useCallback(
        (clientY: number) => {
            const start = presetDragStartRef.current;
            if (!start) return;
            const next = {
                group: start.group,
                id: start.id,
                fromIndex: start.fromIndex,
                insertionIndex: reorderInsertionIndex(presetRowCenters(start.group), clientY),
            };
            presetDragRef.current = next;
            setPresetDrag(next);
        },
        [presetRowCenters],
    );

    /**
     * 行上按下指针：登记起点。
     *
     * 这里**不**立刻进入拖拽 —— 越过 `REORDER_DRAG_THRESHOLD_PX` 才算，否则单击选中
     * 与双击设为当前都会被拖拽逻辑吃掉。行内的按钮（启用 / 停用、右键菜单触发）不
     * 参与拖拽。
     */
    const beginPresetDrag = useCallback(
        (
            preset: VibratoPreset,
            index: number,
            group: PresetGroup,
            event: { button: number; clientY: number; target: EventTarget | null },
        ) => {
            if (event.button !== 0) return;
            const target = event.target as HTMLElement | null;
            if (target?.closest("button")) return;
            suppressPresetClickRef.current = false;
            presetDragStartRef.current = {
                group,
                id: preset.id,
                fromIndex: index,
                pointerY: event.clientY,
            };
        },
        [],
    );

    // 拖拽排序：指针越过阈值才进入拖拽状态，松手时落盘一次。
    useEffect(() => {
        if (!open) return;

        /*
         * 边缘自动滚动。
         *
         * 【为什么是 rAF 而不是"在 pointermove 里滚"】指针贴住边缘不动时也要持续滚
         * —— pointermove 不会再触发，只有逐帧推进才能把列表卷上来。滚动之后行位置
         * 变了，因此每帧重算落点，指示线跟着内容走。
         *
         * 【为什么滚不动就停】视口已到顶 / 到底时 `scrollTop` 不再变化，继续排帧只会
         * 空转；停下后用户再动一下指针即可重新启动。
         *
         * 【为什么写在 effect 里而不是 `useCallback`】这个循环要自引用（每帧重新排
         * 自己），而 hook 的 lint 不允许 useCallback 访问尚未声明的自身。普通函数声明
         * 没有这个问题，副作用也只属于这个 effect。
         */
        let autoScrollFrame: number | null = null;
        const stopAutoScroll = () => {
            if (autoScrollFrame != null) {
                cancelAnimationFrame(autoScrollFrame);
                autoScrollFrame = null;
            }
            presetDragPointerYRef.current = null;
        };
        const tickAutoScroll = () => {
            autoScrollFrame = null;
            const pointerY = presetDragPointerYRef.current;
            const viewport = userListRef.current?.closest<HTMLElement>(
                "[data-radix-scroll-area-viewport]",
            );
            if (pointerY == null || !presetDragRef.current || !viewport) return;
            const rect = viewport.getBoundingClientRect();
            const delta = reorderAutoScrollDelta({
                pointerY,
                viewportTop: rect.top,
                viewportBottom: rect.bottom,
            });
            if (delta === 0) return;
            const before = viewport.scrollTop;
            viewport.scrollTop = before + delta;
            if (viewport.scrollTop === before) return;
            applyPresetDragAtPointer(pointerY);
            autoScrollFrame = requestAnimationFrame(tickAutoScroll);
        };
        const ensureAutoScroll = () => {
            if (autoScrollFrame == null) autoScrollFrame = requestAnimationFrame(tickAutoScroll);
        };

        const onMove = (event: PointerEvent) => {
            const start = presetDragStartRef.current;
            if (!start) return;
            if (
                !presetDragRef.current &&
                Math.abs(event.clientY - start.pointerY) < REORDER_DRAG_THRESHOLD_PX
            ) {
                return;
            }
            applyPresetDragAtPointer(event.clientY);
            // 已经进入拖拽：把松手时那一下 click 吞掉，避免顺带选中 / 激活。
            suppressPresetClickRef.current = true;
            presetDragPointerYRef.current = event.clientY;
            ensureAutoScroll();
        };
        let unregisterAbort = () => {};
        const onUp = () => {
            const start = presetDragStartRef.current;
            presetDragStartRef.current = null;
            const drag = presetDragRef.current;
            presetDragRef.current = null;
            setPresetDrag(null);
            stopAutoScroll();
            unregisterAbort();
            if (!start || !drag) return;
            const toIndex = reorderTargetIndex(drag.insertionIndex, drag.fromIndex);
            // 两组各自排序：系统预设的顺序以 id 列表持久化，用户预设直接排数组。
            dispatch(
                drag.group === "system"
                    ? reorderBuiltinVibratoPreset({ id: drag.id, toIndex })
                    : reorderVibratoPreset({ id: drag.id, toIndex }),
            );
            void dispatch(persistUiSettings());
        };
        // 失焦（Alt+Tab / 最小化）时 pointerup 不会送回本窗口：不接住它，
        // `presetDrag` 的插入位置指示线会一直留在预设列表上，自动滚动 rAF 也会
        // 空转不停（直到对话框关闭）。
        unregisterAbort = registerDragAbort(onUp);
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
        window.addEventListener("pointercancel", onUp);
        return () => {
            unregisterAbort();
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            window.removeEventListener("pointercancel", onUp);
            // 对话框关闭 / 组件卸载时别把 rAF 循环留着。
            stopAutoScroll();
        };
    }, [open, dispatch, applyPresetDragAtPointer]);

    /** 正在就地重命名的那条预设（输入框覆盖在它的名字上）。 */
    const [renameTarget, setRenameTarget] = useState<{ id: string; value: string } | null>(null);

    /**
     * 提交重命名。
     *
     * 空名不提交（预设名是列表里唯一的辨识信息，清空等于让条目变成空白）；`Esc` 走
     * `onCancel`。与 Clip 的内联重命名同一套约定（Enter 提交 / Esc 取消 / 失焦提交）。
     */
    function commitRename() {
        const target = renameTarget;
        setRenameTarget(null);
        if (!target) return;
        const name = target.value.trim();
        if (!name) return;
        const stored = resolved.user.find((preset) => preset.id === target.id);
        if (!stored || stored.name === name) return;
        const renamed = sanitizeVibratoPreset({ ...stored, name });
        persistPreset(renamed);
        // 正在编辑的就是它时，草稿只更新名字 —— **不能**整体替换成入库版本，否则
        // 草稿里还没保存的参数改动会被这次重命名顺手抹掉。
        setDraft((current) => (current?.id === renamed.id ? { ...current, name } : current));
    }

    /** 就地重命名要摊给列表行的那几个 prop（两组共用）。 */
    const renameProps = (preset: VibratoPreset) => ({
        renaming: renameTarget?.id === preset.id,
        renameValue: renameTarget?.value,
        onRenameChange: (value: string) =>
            setRenameTarget((current) => (current ? { ...current, value } : current)),
        onRenameCommit: commitRename,
        onRenameCancel: () => setRenameTarget(null),
    });

    /** 选中一条预设；刚拖完的那一下 click 不触发选中。 */
    function handlePresetRowClick(preset: VibratoPreset) {
        if (suppressPresetClickRef.current) {
            suppressPresetClickRef.current = false;
            return;
        }
        selectPreset(preset);
    }

    const depthUnit = depthStepUnitFor(editParam);
    const depthValue = draft ? depthForParam(draft.depthCents, editParam, paramRange) : 0;
    const cycleEstimate = draft ? estimateCycles(draft, 320, 5) : 0;
    /**
     * 两个页签各自的渐入 / 渐出手柄位置（归一化 0..1）。
     *
     * 【为什么两把】手柄的横向位置是"占**该页签**整段时长的比例"：预设波形用固定窗口，
     * 套用预览用选区真实时长。同一对 `attackMs` / `releaseMs`，在两个时间轴上落在不同
     * 的像素处 —— 画错位置的手柄比没有手柄更糟（用户会去拖一个不在斜坡上的点）。
     *
     * 系统预设也画：手柄可拖，拖的是本地草稿（保存时落成副本）。
     */
    const previewHandles = draft ? handleLayoutFor(draft, presetWindowMs) : undefined;
    const appliedHandles = draft ? handleLayoutFor(draft, appliedWindowMs) : undefined;

    const customCount = resolved.user.length;
    const atCap = customCount >= MAX_VIBRATO_PRESETS;

    /**
     * 形状下拉**当前显示**的形状与偏斜。
     *
     * 手绘中取 `handDraw.shape`（那是用户在编辑器里看到的形状）；否则表波形兜底显示
     * 正弦 —— 与形状下拉本身同一套口径，按钮文案才不会与下拉值对不上。
     */
    const selectedWaveShape: WaveShape =
        handDraw?.shape ?? (draft?.cycle.kind === "shape" ? draft.cycle.shape : "sine");
    const selectedWaveSkew =
        handDraw?.skew ?? (draft?.cycle.kind === "shape" ? draft.cycle.skew : 0.5);

    /** 列表行右键菜单的条目（针对被右击的那条预设）。 */
    const presetMenuItems: AppMenuItemSpec[] = presetMenu
        ? (() => {
              const target = presetMenu.preset;
              const targetDisabled = session.disabledVibratoPresetIds.includes(target.id);
              const targetIsBuiltin = isBuiltinVibratoPresetId(target.id);
              // 两组各自排序：系统预设排在 resolved.system 里，自定义排在 resolved.user 里。
              const groupList = targetIsBuiltin ? resolved.system : resolved.user;
              const groupIndex = groupList.findIndex((preset) => preset.id === target.id);
              return [
                  {
                      key: "toggle-enabled",
                      label: targetDisabled
                          ? t("vibrato_manager_enable")
                          : t("vibrato_manager_disable"),
                      icon: targetDisabled ? <EyeOpenIcon /> : <EyeNoneIcon />,
                      onSelect: () => togglePresetEnabled(target),
                  },
                  {
                      key: "activate",
                      label: t("vibrato_manager_set_active"),
                      checked: session.activeVibratoPresetId === target.id,
                      onSelect: () => activatePreset(target),
                  },
                  {
                      // 就地重命名：输入框覆盖在列表里那条预设的名字上。
                      // 系统预设的名字来自词条（只读），要改只能先复制为自定义。
                      key: "rename",
                      label: t("vibrato_manager_rename"),
                      disabled: targetIsBuiltin,
                      onSelect: () =>
                          setRenameTarget({ id: target.id, value: vibratoPresetLabel(target, t) }),
                  },
                  // 上移 / 下移：列表靠拖拽排序，这里是**键盘可达的等价操作**
                  // —— 只为了拖拽就砍掉非指针用户的路子，代价太大。两组都可排。
                  ...(groupIndex < 0
                      ? []
                      : [
                            {
                                key: "move-up",
                                label: t("vibrato_manager_move_up"),
                                separatorBefore: true,
                                disabled: groupIndex === 0,
                                onSelect: () => movePreset(target, -1),
                            },
                            {
                                key: "move-down",
                                label: t("vibrato_manager_move_down"),
                                disabled: groupIndex === groupList.length - 1,
                                onSelect: () => movePreset(target, 1),
                            },
                        ]),
                  {
                      key: "duplicate",
                      label: t("vibrato_manager_duplicate"),
                      separatorBefore: true,
                      disabled: atCap,
                      tooltip: atCap ? t("vibrato_manager_at_cap") : undefined,
                      onSelect: () => duplicatePreset(target),
                  },
                  {
                      key: "delete",
                      label: t("vibrato_manager_delete"),
                      danger: true,
                      // 出厂预设不能删：它是"永远可复原"的锚点。
                      disabled: targetIsBuiltin,
                      onSelect: () => setDeleteTarget(target),
                  },
              ];
          })()
        : [];

    return {
        dispatch,
        t,
        plural,
        session,
        resolved,
        draft,
        isBuiltin,
        previewSamples,
        previewTab,
        handlePreviewTabChange,
        previewHalfCents,
        setPreviewHalfCents,
        previewHandles,
        cycleEstimate,
        appliedPreview,
        appliedHalfCents,
        appliedHandles,
        appliedStatus,
        setAppliedRefitToken,
        audition,
        toggleAudition,
        auditionCurves,
        ioNotice,
        systemListRef,
        userListRef,
        presetDrag,
        handlePresetRowClick,
        activatePreset,
        togglePresetEnabled,
        setPresetMenu,
        beginPresetDrag,
        renameProps,
        draftIsActive,
        patch,
        selectedWaveShape,
        selectedWaveSkew,
        handDraw,
        setHandDraw,
        openHandDraw,
        paramFineAdjustKb,
        inputModifiers,
        depthValue,
        depthUnit,
        presetMenu,
        presetMenuItems,
        handleImport,
        deleteTarget,
        setDeleteTarget,
        handleDelete,
        handleExtractFromSelection,
        handleExport,
        handleCreate,
        handleDuplicate,
        handleSave,
        handleApply,
        handleOpenChange,
        fileInputRef,
        atCap,
        fitPreviewAxis,
        handlePreviewGestureStart,
        handlePreviewGestureMove,
        handlePreviewGestureEnd,
        handlePreviewPinchDepth,
        customCount,
    };
}
