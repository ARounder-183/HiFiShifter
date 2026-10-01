/**
 * 颤音窗口 —— 预设库 + 套用到选区，**同一个窗口**。
 *
 * 【为什么合并成一个】曾经是两扇窗：「颤音预设管理」回答"这个预设长什么样"（改库），
 * 「添加颤音」回答"套到这段选区上长什么样"（用库）。它们各有一份预设列表、一块预览
 * 画布、一套试听，中间靠宿主侧一层跳转（把选中的 id 带过去、返回时再带回来）缝合。
 * 那层缝合的代价是真实的：跳转即重新挂载 —— 纵轴重拟合、试听停掉、滚动回到顶部；
 * 而且"改预设"与"看选区效果"分成两处，用户要"提取 → 改名 → 应用"必须跨窗口走一圈。
 *
 * 合并的关键是：两者本来就在算同一件事（同一条 `buildVibratoCurve`、同一条
 * `planVibratoTarget`、同一条试听链），分裂的只是外壳。于是外壳合成一个，内核一行
 * 不改 —— 这也是这次合并最有力的证据（见 `vibratoDialogLogic.ts`）。
 *
 * 【结构】左列预设列表、右侧预览 + 编辑器（照 `CustomScaleDialog` 的分工：列表负责
 * "选哪一个"，编辑器负责"捏成什么样"）。预览区有两页（`VibratoPreviewPane`）：
 * 「预设波形」与「套用到选区」；后者只在宿主给了选区数据入口（`applyTarget`）时才出现。
 *
 * 【只读系统预设】系统预设禁用一切字段，只有「复制为自定义」可用。这样出厂
 * 预设永远可复原，而"改坏了"不会变成不可逆 —— 比"允许改但提供重置"更简单，
 * 也不会让用户对着一个与出厂说明不符的"自然"预设困惑。
 *
 * 【草稿-保存】编辑改的是局部草稿，点「保存」才 dispatch + 落盘。**离开这条预设 =
 * 落盘**（切换预设、删除、导入都先写回库），只有关闭窗口才丢弃未保存的改动 —— 这
 * 一条与合并前一致，是"用户改了一半去看看别的"不会丢东西的保证。
 *
 * 【一体两面】窗口有两副面孔，由 `applyTarget` 一票决定：
 *
 * - **有宿主**（「添加颤音」）：多出「套用到选区」页签、「从选区提取」与「应用」——
 *   既能改库，也能看着选区把这份参数落下去。
 * - **没有宿主**（工具栏 / 菜单栏的「管理预设…」）：只有「预设波形」一页，页脚也只
 *   有改库的动作 —— 与合并前的预设管理器完全一致。改库这件事不该因为窗口变大了就
 *   多出一堆用不上的按钮。
 *
 * 【应用】「应用」把**当前草稿**（含未保存的微调）交给编辑管线，不写库，然后**关窗**：
 * 它是「添加颤音」这次打开的终点。它与「保存」是两件事（写选区 vs 写库），因此两个
 * 按钮并存，`defaultActionId` 在有宿主时选「应用」。
 */

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ShuffleIcon, EyeNoneIcon, EyeOpenIcon } from "@radix-ui/react-icons";
import { Box, Flex, ScrollArea, TextField } from "@radix-ui/themes";

import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { isModifierActive, selectKeybinding } from "../../features/keybindings/keybindingsSlice";
import { useI18n } from "../../i18n/I18nProvider";
import {
    persistUiSettings,
    removeVibratoPreset,
    reorderBuiltinVibratoPreset,
    reorderVibratoPreset,
    setActiveVibratoPreset,
    toggleVibratoPresetEnabled,
    upsertVibratoPreset,
} from "../../features/session/sessionSlice";
import {
    MAX_VIBRATO_PRESETS,
    VIBRATO_LIMITS,
    createVibratoPresetId,
    duplicateVibratoPreset,
    isBuiltinVibratoPresetId,
    nextDuplicatePresetName,
    sanitizeVibratoPreset,
} from "../../features/vibrato/vibratoPresets";
import {
    activeIdAfterRemoval,
    findVibratoPreset,
    resolveVibratoPresets,
} from "../../features/vibrato/vibratoPresetList";
import {
    mergeImportedPresets,
    parseVibratoPresets,
    serializeVibratoPresets,
    vibratoPresetFileName,
    vibratoPresetSignature,
} from "../../features/vibrato/vibratoPresetFile";
import { shapeUsesSkew } from "../../features/vibrato/vibratoCycle";
import { randomVibratoSeed } from "../../features/vibrato/vibratoSeed";
import { depthStepUnitFor } from "../../features/vibrato/vibratoDepth";
import { estimateCycles } from "../../features/vibrato/vibratoCurve";
import { planVibratoTarget } from "../../features/vibrato/vibratoPitch";
import { exportVibratoPresetsJson } from "../../services/api/jsonExport";
import {
    buildAuditionCurve,
    buildContourAuditionPair,
    vibratoAudition,
} from "../../features/vibrato/vibratoAudition";
import type {
    BaselineMode,
    CycleSource,
    EnvelopeCurve,
    VibratoPreset,
    VibratoRateMode,
    WaveShape,
} from "../../features/vibrato/vibratoTypes";
import {
    AppButton,
    AppContextMenu,
    AppConfirmDialog,
    AppDialog,
    AppField,
    AppForm,
    AppFormSection,
    AppIconButton,
    AppListRow,
    AppNumberField,
    AppSegmentedControl,
    AppSelect,
    AppSlider,
    AppSliderReadout,
    AppSwitchRow,
    type AppMenuItemSpec,
} from "../../ui";
import { AppFileInput } from "../../ui/FileInput";
import { VibratoPresetGlyph } from "./vibrato/VibratoPresetGlyph";
import {
    type VibratoPreviewGestureInfo,
    type VibratoPreviewModifiers,
} from "./vibrato/VibratoPreviewCanvas";
import {
    VibratoPreviewPane,
    type VibratoAppliedStatus,
    type VibratoAuditionKind,
    type VibratoPreviewTab,
} from "./vibrato/VibratoPreviewPane";
import {
    BASELINE_MODE_KEYS,
    BASELINE_MODE_ORDER,
    ENVELOPE_CURVE_KEYS,
    ENVELOPE_CURVE_ORDER,
    PREVIEW_DEFAULT,
    RATE_MODE_KEYS,
    WAVE_SHAPE_KEYS,
    WAVE_SHAPE_ORDER,
    buildAppliedPreview,
    buildVibratoPreview,
    depthToCents,
    depthForParam,
    fitPreviewRangeCents,
    formatNumber,
    vibratoPresetDescription,
    vibratoPresetLabel,
    vibratoPresetSummary,
} from "./vibrato/vibratoDialogLogic";
import {
    advancePreviewFineDrag,
    applyPreviewGesture,
    createPreviewFineDragState,
    cycleWidthPxFor,
    handleLayoutFor,
    type PreviewFineDragState,
    type PreviewGestureSnapshot,
    type PreviewZone,
} from "./vibrato/vibratoPreviewGestures";
import { VibratoCycleEditor } from "./vibrato/VibratoCycleEditor";
import { tableFromCycle } from "./vibrato/vibratoCycleEdit";
import {
    REORDER_DRAG_THRESHOLD_PX,
    reorderAutoScrollDelta,
    reorderInsertionIndex,
    reorderTargetIndex,
} from "./dragReorder";

/**
 * 选区宿主：给了这三样，窗口就多出「套用到选区」页签与「应用」「提取」两个动作。
 *
 * 【为什么合成一个 prop 而不是三个可选的】它们同生共死 —— 能画套用预览的地方就能
 * 应用，反之亦然。三个独立的可选 prop 允许"只给一个"的中间状态，而那个状态没有
 * 意义（预览画得出来却应用不了？），却要在组件里到处判空。
 */
export interface VibratoApplyTarget {
    /**
     * 取选区（无选区时取曲线头部）的原始帧值。
     *
     * 由宿主提供：只有面板手里有 `rootTrackId` / `editParam` / 选区。返回 `null`
     * 表示取不到数据（例如刚打开还没有参数曲线）。
     */
    loadOriginal: () => Promise<{ values: number[]; framePeriodMs: number } | null>;
    /** 应用：把（可能已微调的）完整预设交给编辑管线。 */
    onApply: (preset: VibratoPreset) => void;
    /**
     * 从选区提取颤音预设。
     *
     * 【为什么由宿主做】提取要读参数帧（选区 + 轨道 + 参数），只有宿主够得着。成功
     * 返回**已入库**的那条预设（本窗口随即选中它），失败返回 `null`（本窗口给出
     * 行内提示）。入库动作留在宿主，是因为它还要顺带更新"当前使用"。
     */
    onExtract?: () => Promise<VibratoPreset | null>;
}

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    /** 预设编辑时的"当前参数"：深度按它换算成原生单位显示与编辑。 */
    editParam?: string;
    paramRange?: { min: number; max: number };
    /**
     * 打开时**编辑**哪个预设；省略则用当前活动预设。
     *
     * 从选区提取出来的那条预设走这里：用户刚把它提出来，当然是要接着编辑它。
     */
    initialPresetId?: string;
    /**
     * 选区宿主；省略 = 纯预设库。
     *
     * 【这一个 prop 决定窗口的两副面孔】给了它才有「套用到选区」页签、「从选区提取」
     * 与「应用」—— 也就是「添加颤音」那一次打开的全部专属能力；不给就是"改库"那一面
     * （工具栏 / 菜单栏的「管理预设…」），与合并前的预设管理器一致。
     *
     * 打开时落在哪个预览页签也由它决定：有宿主就落在「套用到选区」（那次打开的目的
     * 就是"看套上去什么样、然后应用"），没有就落在「预设波形」。
     */
    applyTarget?: VibratoApplyTarget;
}

/** 列表列的宽度（CSS 像素）。 */
const LIST_WIDTH = 208;

/**
 * 预设列表的两个分组。
 *
 * 拖拽只在**同组内**换位：系统预设与用户预设是两份独立的有序集合（前者顺序存在设置
 * 里的 id 列表，后者就是数组本身），跨组拖拽没有明确的语义，也会让"系统预设只读"
 * 的边界变糊。
 */
type PresetGroup = "system" | "user";
/**
 * 对话框内容的定高。
 *
 * 【为什么必须定高】对话框正文区自身是 `overflow-y-auto`；只要内容总高超过它，
 * 外层就会出现第二条竖直滚动条（内层两栏各一条 + 外层一条）。选到系统预设时
 * 多出的只读提示行、导入后的反馈行，都曾把总高顶过上限 —— 那正是"双竖直
 * 滚动条"只在特定选择下出现的指纹。
 *
 * 定高之后语义反转：条件行不再"加高总内容"，而是**压缩栏高**（两栏是
 * `flex-1 min-h-0`，被谁挤都只是各自变矮），外层永不溢出 —— 结构上不可能再
 * 出现双层滚动，不依赖任何 vh 阈值的运气。
 */
const CONTENT_HEIGHT = "min(60vh, 560px)";

export function VibratoDialog({
    open,
    onOpenChange,
    editParam = "pitch",
    paramRange,
    initialPresetId,
    applyTarget,
}: Props) {
    const dispatch = useAppDispatch();
    const { t, plural } = useI18n();
    const session = useAppSelector((state: RootState) => state.session);
    /** 「精细调整」修饰键（默认 `Ctrl` / macOS `Command`）：预览拖拽时缩到 1/10。 */
    const paramFineAdjustKb = useAppSelector((state: RootState) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );

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
        void loadOriginal().then((result) => {
            if (cancelled) return;
            setOriginal(result);
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
        });
    }, [applyTarget, draft, original, editParam, paramRange]);

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
        return buildContourAuditionPair(
            appliedPreview.contour,
            appliedPreview.wave,
            original.framePeriodMs,
        );
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
    } | null>(null);

    /** 画布手势开始：登记起点快照（窗口时长取预览默认几何）。 */
    function handlePreviewGestureStart(zone: PreviewZone, info: VibratoPreviewGestureInfo) {
        if (!draft || isBuiltin) return;
        const windowMs = (PREVIEW_DEFAULT.frameCount - 1) * PREVIEW_DEFAULT.framePeriodMs;
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
                centsPerPx: info.centsPerPx,
            },
            fine: createPreviewFineDragState(fineActive),
        };
    }

    /** 画布手势移动：位移 → 草稿字段（换算规则见 `vibratoPreviewGestures`）。 */
    function handlePreviewGestureMove(
        deltaX: number,
        deltaY: number,
        modifiers: VibratoPreviewModifiers,
    ) {
        const gesture = previewGestureRef.current;
        if (!gesture) return;
        // 「精细调整」按增量缩放位移：中途按下 / 松开只改变此后的速度，累计量连续。
        const scaled = advancePreviewFineDrag(
            gesture.fine,
            deltaX,
            deltaY,
            isModifierActive(paramFineAdjustKb, modifiers),
        );
        patch(applyPreviewGesture(gesture.zone, gesture.snapshot, scaled.deltaX, scaled.deltaY));
    }

    /** 画布手势结束。 */
    function handlePreviewGestureEnd() {
        previewGestureRef.current = null;
    }

    function persistPreset(preset: VibratoPreset) {
        dispatch(upsertVibratoPreset(preset));
        void dispatch(persistUiSettings());
    }

    /**
     * 草稿相对库中同 id 预设是否有未保存的改动。
     *
     * 系统预设永远不算"可保存的改动"（只读）；列表里找不到同 id，说明是刚新建、
     * 还没入库的预设，也算有改动。
     */
    function draftHasUnsavedChanges(): boolean {
        if (!draft || isBuiltin) return false;
        const stored = resolved.user.find((preset) => preset.id === draft.id);
        if (!stored) return true;
        return (
            vibratoPresetSignature(sanitizeVibratoPreset(draft)) !== vibratoPresetSignature(stored)
        );
    }

    /**
     * 选中一个预设进行编辑。
     *
     * 【切走时先落盘】用户在 A 上改了一半、切到 B 看看，若改动被直接丢弃，
     * "编辑途中不能换预设"就成了硬伤。因此切换前先把 A 的未保存改动写回库 ——
     * 与「保存」同一套净化 / 入库路径。没有改动（或系统预设）时什么都不做。
     */
    function selectPreset(preset: VibratoPreset) {
        if (draft && draft.id !== preset.id && draftHasUnsavedChanges()) {
            persistPreset(sanitizeVibratoPreset(draft));
        }
        setHandDraw(null);
        setDraft(preset);
        // 换预设 = 换一段波形，纵轴跟着重新拟合（编辑期间则保持不动）。
        fitPreviewAxis(preset);
    }

    /** 进入手绘：把当前波形采样成表作为起点（改形比从零画顺手）。 */
    function openHandDraw() {
        if (!draft || isBuiltin) return;
        const baseline = draft.cycle;
        // 记下"当前形状"：来自参数式就用它，来自表（提取 / 上次手绘）则兜底正弦 ——
        // 与形状下拉的显示口径一致。
        const shape: WaveShape = baseline.kind === "shape" ? baseline.shape : "sine";
        const skew = baseline.kind === "shape" ? baseline.skew : 0.5;
        const table = baseline.kind === "table" ? [...baseline.table] : tableFromCycle(baseline);
        setHandDraw({ baseline, shape, skew });
        patch({ cycle: { kind: "table", table } });
    }

    /** 设为当前使用（拖拽 / 菜单都用它）。 */
    function activatePreset(preset: VibratoPreset) {
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

    /** 复制一份预设（管理器页脚与列表右键菜单共用）。 */
    function duplicatePreset(source: VibratoPreset) {
        /*
         * 名字按**显示名**预填：系统预设的 `name` 是空的（名字走词条），拿它当基底会
         * 得到一个无名副本。编号还要避开现有全部显示名，否则连点两次就是"名字 2 2"。
         */
        const name = nextDuplicatePresetName(
            vibratoPresetLabel(source, t) || t("vibrato_manager_new"),
            resolved.all.map((preset) => vibratoPresetLabel(preset, t)),
        );
        const copy = duplicateVibratoPreset(source, name);
        persistPreset(copy);
        selectPreset(copy);
        activatePreset(copy);
    }

    function handleSave() {
        if (!draft || isBuiltin) return;
        const normalized = sanitizeVibratoPreset(draft);
        persistPreset(normalized);
        setDraft(normalized);
        // 保存是"这段波形定稿了"的时机：顺势把纵轴重新拟合回六成上下。
        fitPreviewAxis(normalized);
    }

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
        const extracted = await applyTarget.onExtract();
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
        const onUp = () => {
            const start = presetDragStartRef.current;
            presetDragStartRef.current = null;
            const drag = presetDragRef.current;
            presetDragRef.current = null;
            setPresetDrag(null);
            stopAutoScroll();
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
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
        window.addEventListener("pointercancel", onUp);
        return () => {
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
    /** 预览窗口时长（ms）：与 `buildVibratoPreview` 的默认几何一致。 */
    const previewWindowMs = (PREVIEW_DEFAULT.frameCount - 1) * PREVIEW_DEFAULT.framePeriodMs;
    /** 渐入 / 渐出手柄的归一化位置。系统预设只读，不画手柄。 */
    const previewHandles =
        draft && !isBuiltin ? handleLayoutFor(draft, previewWindowMs) : undefined;

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
                      tooltip: targetIsBuiltin ? t("vibrato_manager_readonly") : undefined,
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
                      // 系统预设只读：要删只能删副本。
                      disabled: targetIsBuiltin,
                      tooltip: targetIsBuiltin ? t("vibrato_manager_readonly") : undefined,
                      onSelect: () => setDeleteTarget(target),
                  },
              ];
          })()
        : [];

    return (
        <>
            <AppDialog
                open={open}
                onOpenChange={handleOpenChange}
                title={t("vibrato_manager_title")}
                size="xl"
                actions={[
                    /*
                     * 【「从选区提取」排在页脚最左】它是作用于**选区**的动作，与右边
                     * 那些"改这条预设 / 改这个库"的动作不同类；而且它与「删除」同属
                     * 低频、非主动作的那一类（`align: "start"` 的既有语义）。
                     * 只在有选区宿主时出现 —— 菜单栏那条路径背后没有选区。
                     */
                    ...(applyTarget?.onExtract
                        ? [
                              {
                                  id: "extract",
                                  label: t("vibrato_extract_action"),
                                  align: "start" as const,
                                  autoClose: false,
                                  onClick: async () => {
                                      await handleExtractFromSelection();
                                  },
                              },
                          ]
                        : []),
                    /*
                     * 【为什么每个动作都显式写 `autoClose: false`】
                     * `AppDialog` 对**同步**动作默认 `autoClose: true`（异步动作默认
                     * false）。本对话框里没有"点一下就完事"的动作：导入要挑文件、
                     * 新建 / 复制 / 删除都要接着在同一个窗口里继续编辑。少了这个字段，
                     * 一个同步动作就会顺手把窗口关掉 —— 而且这条规则很容易被一次
                     * "把 async 去掉"的重构悄悄改回去，所以宁可逐个写明。
                     */
                    {
                        id: "delete",
                        label: t("vibrato_manager_delete"),
                        intent: "danger",
                        align: "start",
                        disabled: !draft || isBuiltin,
                        // 删除走二次确认，不关闭主对话框。
                        autoClose: false,
                        onClick: async () => {
                            setDeleteTarget(draft);
                        },
                    },
                    {
                        id: "import",
                        label: t("vibrato_io_import"),
                        autoClose: false,
                        onClick: () => fileInputRef.current?.click(),
                    },
                    {
                        id: "export",
                        label: t("vibrato_io_export"),
                        disabled: !draft,
                        autoClose: false,
                        onClick: async () => {
                            await handleExport();
                        },
                    },
                    {
                        id: "new",
                        label: t("vibrato_manager_new"),
                        disabled: atCap,
                        autoClose: false,
                        onClick: handleCreate,
                    },
                    {
                        id: "duplicate",
                        label: t("vibrato_manager_duplicate"),
                        disabled: !draft,
                        autoClose: false,
                        onClick: handleDuplicate,
                    },
                    {
                        id: "save",
                        label: t("vibrato_manager_save"),
                        /*
                         * 【为什么「保存」不再是 primary】有选区宿主时页脚有两个
                         * "往前走"的动作（保存写库、应用写选区），而一张对话框里
                         * 只该有一个主按钮 —— 主的是「应用」（用户开这扇窗的目的）。
                         * 没有宿主时它就是唯一的主按钮（见 defaultActionId）。
                         */
                        intent: applyTarget ? "default" : "primary",
                        disabled: !draft || isBuiltin,
                        // 保存**不关闭**对话框：用户常要"先存一版、接着调"，
                        // 存完就把窗口收掉等于逼他重新打开。
                        autoClose: false,
                        onClick: handleSave,
                    },
                    ...(applyTarget
                        ? [
                              {
                                  id: "apply",
                                  label: t("vibrato_apply_apply"),
                                  intent: "primary" as const,
                                  disabled: !draft,
                                  /*
                                   * 关窗在 `handleApply` 里显式做（而不是靠 `autoClose`
                                   * 的默认值）：先落到选区、再关，两步的顺序是这段
                                   * 逻辑的一部分，不该藏在"同步动作默认关窗"这条规则里。
                                   */
                                  autoClose: false,
                                  onClick: handleApply,
                              },
                          ]
                        : []),
                    {
                        // 显式关闭按钮：保存不再关闭窗口之后，页脚里没有"退出"的
                        // 去处，只剩 Esc / 点外部 —— 两者都不显眼。
                        id: "close",
                        label: t("close"),
                        autoClose: false,
                        onClick: () => handleOpenChange(false),
                    },
                ]}
                /*
                 * 默认动作：有选区宿主时是「应用」（用户开这扇窗的目的就是应用它），
                 * 否则是「保存」（与合并前一致）。关闭排在它们右边（页脚最右是关闭的
                 * 常见排布），但 Enter 不该变成"关掉窗口"。
                 */
                defaultActionId={applyTarget ? "apply" : "save"}
            >
                <Flex
                    direction="column"
                    gap="3"
                    data-vibrato-content
                    className="min-h-0"
                    style={{ height: CONTENT_HEIGHT }}
                >
                    {/* ---- 预览（整行置顶，不参与任何滚动） ----
                        放在两栏之上而不是塞进参数流的头部：整行宽度读波形更清楚，
                        且它不属于任何滚动区，调参数时**永远**不会滚出视野。
                        两个页签共用这一块（见 `VibratoPreviewPane`）。 */}
                    {draft && previewSamples ? (
                        <>
                            <VibratoPreviewPane
                                tab={previewTab}
                                onTabChange={handlePreviewTabChange}
                                hasSelection={Boolean(applyTarget)}
                                presetSamples={previewSamples}
                                presetHalfCents={previewHalfCents}
                                presetHandles={previewHandles}
                                onGestureStart={handlePreviewGestureStart}
                                onGestureMove={handlePreviewGestureMove}
                                onGestureEnd={handlePreviewGestureEnd}
                                cyclesEstimate={cycleEstimate}
                                appliedSamples={appliedPreview}
                                appliedHalfCents={appliedHalfCents}
                                appliedStatus={appliedStatus}
                                onFit={() => {
                                    // 「适应」作用于**当前页签**：两页各有一把标尺，
                                    // 一次点击只该动用户正看着的那把。
                                    if (previewTab === "applied" && applyTarget) {
                                        setAppliedRefitToken((token) => token + 1);
                                    } else {
                                        fitPreviewAxis(draft);
                                    }
                                }}
                                fitDisabled={isBuiltin}
                                audition={audition}
                                onAudition={toggleAudition}
                                appliedAuditionDisabled={!auditionCurves}
                            />
                            {isBuiltin ? (
                                <span className="hs-type-caption">
                                    {t("vibrato_manager_readonly")}
                                </span>
                            ) : null}
                        </>
                    ) : null}

                    {ioNotice ? (
                        <span
                            className="hs-type-caption"
                            style={{
                                color: ioNotice.danger
                                    ? "var(--qt-danger-text)"
                                    : "var(--qt-text-muted)",
                            }}
                            role="status"
                        >
                            {ioNotice.text}
                        </span>
                    ) : null}

                    <Flex gap="4" align="stretch" className="min-h-0 flex-1">
                        {/* ---- 预设列表 ---- */}
                        <Flex
                            direction="column"
                            gap="2"
                            className="min-h-0"
                            style={{ width: LIST_WIDTH, flexShrink: 0 }}
                        >
                            <ScrollArea
                                className="hs-scroll-area"
                                style={{ height: "100%" }}
                                scrollbars="vertical"
                                type="auto"
                            >
                                <Flex direction="column" gap="1" pr="2">
                                    <span className="hs-type-muted">
                                        {t("vibrato_manager_group_system")}
                                    </span>
                                    {/* 系统预设同样可拖拽排序（顺序以 id 列表持久化）。
                                        只在本组内换位，不会与下面的自定义预设混排。 */}
                                    <Flex direction="column" gap="1" ref={systemListRef}>
                                        {resolved.system.map((preset, index) => (
                                            <Fragment key={preset.id}>
                                                {presetDrag?.group === "system" &&
                                                presetDrag.insertionIndex === index ? (
                                                    <Box
                                                        aria-hidden
                                                        className="rounded-full bg-qt-accent"
                                                        style={{ height: 2 }}
                                                    />
                                                ) : null}
                                                <Box
                                                    data-preset-row={preset.id}
                                                    style={{
                                                        opacity:
                                                            presetDrag?.group === "system" &&
                                                            presetDrag.id === preset.id
                                                                ? 0.4
                                                                : undefined,
                                                    }}
                                                >
                                                    <PresetRow
                                                        preset={preset}
                                                        selected={draft?.id === preset.id}
                                                        active={
                                                            session.activeVibratoPresetId ===
                                                            preset.id
                                                        }
                                                        enabled={
                                                            !session.disabledVibratoPresetIds.includes(
                                                                preset.id,
                                                            )
                                                        }
                                                        reorderable
                                                        onSelect={() =>
                                                            handlePresetRowClick(preset)
                                                        }
                                                        onActivate={() => activatePreset(preset)}
                                                        onToggleEnabled={() =>
                                                            togglePresetEnabled(preset)
                                                        }
                                                        onContextMenu={(x, y) =>
                                                            setPresetMenu({ x, y, preset })
                                                        }
                                                        onDragPointerDown={(event) =>
                                                            beginPresetDrag(
                                                                preset,
                                                                index,
                                                                "system",
                                                                event,
                                                            )
                                                        }
                                                        {...renameProps(preset)}
                                                    />
                                                </Box>
                                            </Fragment>
                                        ))}
                                        {presetDrag?.group === "system" &&
                                        presetDrag.insertionIndex >= resolved.system.length ? (
                                            <Box
                                                aria-hidden
                                                className="rounded-full bg-qt-accent"
                                                style={{ height: 2 }}
                                            />
                                        ) : null}
                                    </Flex>

                                    <Box pt="2">
                                        <span className="hs-type-muted">
                                            {t("vibrato_manager_group_user")}
                                        </span>
                                    </Box>
                                    {resolved.user.length === 0 ? (
                                        <span className="hs-type-caption">
                                            {t("vibrato_manager_empty")}
                                        </span>
                                    ) : (
                                        /* 用户预设可**拖拽排序**：按住行上下拖，越过别行中线
                                           即换位，松手落盘。落点用一条指示线表达；被拖的那行
                                           压暗。键盘可达的等价操作在行的右键菜单里
                                           （上移 / 下移）。 */
                                        <Flex direction="column" gap="1" ref={userListRef}>
                                            {resolved.user.map((preset, index) => (
                                                <Fragment key={preset.id}>
                                                    {presetDrag?.group === "user" &&
                                                    presetDrag.insertionIndex === index ? (
                                                        <Box
                                                            aria-hidden
                                                            className="rounded-full bg-qt-accent"
                                                            style={{ height: 2 }}
                                                        />
                                                    ) : null}
                                                    <Box
                                                        data-preset-row={preset.id}
                                                        style={{
                                                            opacity:
                                                                presetDrag?.group === "user" &&
                                                                presetDrag.id === preset.id
                                                                    ? 0.4
                                                                    : undefined,
                                                        }}
                                                    >
                                                        <PresetRow
                                                            preset={preset}
                                                            selected={draft?.id === preset.id}
                                                            active={
                                                                session.activeVibratoPresetId ===
                                                                preset.id
                                                            }
                                                            enabled={
                                                                !session.disabledVibratoPresetIds.includes(
                                                                    preset.id,
                                                                )
                                                            }
                                                            reorderable
                                                            onSelect={() =>
                                                                handlePresetRowClick(preset)
                                                            }
                                                            onActivate={() =>
                                                                activatePreset(preset)
                                                            }
                                                            onToggleEnabled={() =>
                                                                togglePresetEnabled(preset)
                                                            }
                                                            onContextMenu={(x, y) =>
                                                                setPresetMenu({ x, y, preset })
                                                            }
                                                            onDragPointerDown={(event) =>
                                                                beginPresetDrag(
                                                                    preset,
                                                                    index,
                                                                    "user",
                                                                    event,
                                                                )
                                                            }
                                                            {...renameProps(preset)}
                                                        />
                                                    </Box>
                                                </Fragment>
                                            ))}
                                            {presetDrag?.group === "user" &&
                                            presetDrag.insertionIndex >= resolved.user.length ? (
                                                <Box
                                                    aria-hidden
                                                    className="rounded-full bg-qt-accent"
                                                    style={{ height: 2 }}
                                                />
                                            ) : null}
                                        </Flex>
                                    )}
                                </Flex>
                            </ScrollArea>
                        </Flex>

                        {/* ---- 编辑器 ---- */}
                        <Box className="min-h-0 flex flex-col" style={{ minWidth: 0, flex: 1 }}>
                            {draft && previewSamples ? (
                                <Flex direction="column" gap="3" className="min-h-0 flex-1">
                                    <ScrollArea
                                        className="hs-scroll-area min-h-0 flex-1"
                                        scrollbars="vertical"
                                        type="auto"
                                    >
                                        <Box pr="2">
                                            <AppForm>
                                                {!isBuiltin ? (
                                                    <AppField label={t("vibrato_manager_name")}>
                                                        <TextField.Root
                                                            size="2"
                                                            value={draft.name}
                                                            aria-label={t("vibrato_manager_name")}
                                                            onChange={(event) =>
                                                                patch({ name: event.target.value })
                                                            }
                                                        />
                                                    </AppField>
                                                ) : null}
                                                <AppFormSection title={t("vibrato_section_wave")}>
                                                    <AppField label={t("vibrato_shape_label")}>
                                                        <Flex align="center" gap="2" wrap="wrap">
                                                            <AppSelect
                                                                value={selectedWaveShape}
                                                                disabled={isBuiltin}
                                                                onValueChange={(value) => {
                                                                    const shape =
                                                                        value as WaveShape;
                                                                    if (handDraw) {
                                                                        // 手绘中改形状：记下"当前形状"并即时套用，
                                                                        // **不**退出编辑器 —— 否则「复位到{形状}」
                                                                        // 永远只能指向兜底的正弦。
                                                                        setHandDraw({
                                                                            ...handDraw,
                                                                            shape,
                                                                        });
                                                                        patch({
                                                                            cycle: {
                                                                                kind: "table",
                                                                                table: tableFromCycle(
                                                                                    {
                                                                                        kind: "shape",
                                                                                        shape,
                                                                                        skew: handDraw.skew,
                                                                                    },
                                                                                ),
                                                                            },
                                                                        });
                                                                        return;
                                                                    }
                                                                    // 换成参数形状即退出"手绘表"，避免编辑器与草稿错位。
                                                                    setHandDraw(null);
                                                                    patch({
                                                                        cycle: {
                                                                            kind: "shape",
                                                                            shape,
                                                                            skew:
                                                                                draft.cycle.kind ===
                                                                                "shape"
                                                                                    ? draft.cycle
                                                                                          .skew
                                                                                    : 0.5,
                                                                        },
                                                                    });
                                                                }}
                                                                options={WAVE_SHAPE_ORDER.map(
                                                                    (shape) => ({
                                                                        value: shape,
                                                                        label: t(
                                                                            WAVE_SHAPE_KEYS[shape],
                                                                        ),
                                                                    }),
                                                                )}
                                                            />
                                                            <AppButton
                                                                size="sm"
                                                                emphasis="soft"
                                                                disabled={isBuiltin}
                                                                onClick={openHandDraw}
                                                            >
                                                                {t("vibrato_handdraw_open")}
                                                            </AppButton>
                                                        </Flex>
                                                    </AppField>
                                                    <AppField label={t("vibrato_skew")}>
                                                        <Flex align="center" gap="2" wrap="wrap">
                                                            <AppSlider
                                                                unit="percent"
                                                                min={2}
                                                                max={98}
                                                                /*
                                                                 * 偏斜只对**参数式形状**有效：手绘（以及从选区提取）
                                                                 * 出来的表波形里没有"上升段占比"这个东西 —— 表就是
                                                                 * 用户画的那条曲线本身。
                                                                 *
                                                                 * 【为什么必须禁用而不是"允许调但没效果"】参数式形状的
                                                                 * 偏斜在 `sampleCycle` 里参与取样；表的取样完全不看它。
                                                                 * 留着能拖会让用户以为拖了会变，实际毫无反应。
                                                                 */
                                                                disabled={
                                                                    isBuiltin ||
                                                                    draft.cycle.kind !== "shape" ||
                                                                    !shapeUsesSkew(
                                                                        selectedWaveShape,
                                                                    )
                                                                }
                                                                value={Math.round(
                                                                    selectedWaveSkew * 100,
                                                                )}
                                                                ariaLabel={t("vibrato_skew")}
                                                                onChange={(next) =>
                                                                    patch({
                                                                        cycle: {
                                                                            kind: "shape",
                                                                            shape: selectedWaveShape,
                                                                            skew: next / 100,
                                                                        },
                                                                    })
                                                                }
                                                            />
                                                            <AppSliderReadout>
                                                                {`${formatNumber(selectedWaveSkew * 100)}%`}
                                                            </AppSliderReadout>
                                                        </Flex>
                                                    </AppField>
                                                    {/* 手绘周期编辑器：波形分区下方展开（只在草稿是 table 时）。 */}
                                                    {handDraw && draft.cycle.kind === "table" ? (
                                                        <VibratoCycleEditor
                                                            table={draft.cycle.table}
                                                            disabled={isBuiltin}
                                                            // 画布的 aria-label 讲"这块画布能干什么"，
                                                            // 与展开按钮的"手绘…"（讲动作）分开两个键。
                                                            ariaLabel={t("vibrato_handdraw_canvas")}
                                                            readoutLabels={{
                                                                phase: t("vibrato_handdraw_phase"),
                                                                scale: t("vibrato_handdraw_scale"),
                                                            }}
                                                            // 「精细调整」与预览画布同一个键位。
                                                            fineAdjustKb={paramFineAdjustKb}
                                                            smoothLabel={t(
                                                                "vibrato_handdraw_smooth",
                                                            )}
                                                            resetLabel={t("vibrato_handdraw_reset")}
                                                            resetShapeLabel={t(
                                                                "vibrato_handdraw_reset_shape",
                                                            ).replace(
                                                                "{shape}",
                                                                t(
                                                                    WAVE_SHAPE_KEYS[
                                                                        selectedWaveShape
                                                                    ],
                                                                ),
                                                            )}
                                                            onChange={(table) =>
                                                                patch({
                                                                    cycle: { kind: "table", table },
                                                                })
                                                            }
                                                            onReset={() =>
                                                                patch({
                                                                    cycle: {
                                                                        kind: "table",
                                                                        table: tableFromCycle(
                                                                            handDraw.baseline,
                                                                        ),
                                                                    },
                                                                })
                                                            }
                                                            // 复位到下拉框里当前选中的形状：
                                                            // 手绘画歪了想从头来，或把提取出的
                                                            // 波形换成规整形状，都靠它。
                                                            onResetToShape={() =>
                                                                patch({
                                                                    cycle: {
                                                                        kind: "table",
                                                                        table: tableFromCycle({
                                                                            kind: "shape",
                                                                            shape: selectedWaveShape,
                                                                            skew: selectedWaveSkew,
                                                                        }),
                                                                    },
                                                                })
                                                            }
                                                        />
                                                    ) : null}
                                                </AppFormSection>

                                                <AppFormSection title={t("vibrato_section_depth")}>
                                                    <AppField label={t("vibrato_depth_label")}>
                                                        <AppNumberField
                                                            value={depthValue}
                                                            unit={depthUnit}
                                                            disabled={isBuiltin}
                                                            min={VIBRATO_LIMITS.depthCents.min}
                                                            ariaLabel={t("vibrato_depth_label")}
                                                            onChange={(next) =>
                                                                patch({
                                                                    depthCents: depthToCents(
                                                                        next,
                                                                        editParam,
                                                                        paramRange,
                                                                    ),
                                                                })
                                                            }
                                                            onCommit={() => undefined}
                                                        />
                                                    </AppField>
                                                    <AppField label={t("vibrato_depth_ramp")}>
                                                        <Flex align="center" gap="2" wrap="wrap">
                                                            <AppNumberField
                                                                value={draft.depthRamp.start}
                                                                unit="percentFine"
                                                                disabled={isBuiltin}
                                                                min={0}
                                                                max={2}
                                                                suffix={t(
                                                                    "vibrato_depth_ramp_start",
                                                                )}
                                                                ariaLabel={`${t("vibrato_depth_ramp")} ${t("vibrato_depth_ramp_start")}`}
                                                                onCommit={(next) =>
                                                                    patch({
                                                                        depthRamp: {
                                                                            ...draft.depthRamp,
                                                                            start: next,
                                                                        },
                                                                    })
                                                                }
                                                            />
                                                            <AppNumberField
                                                                value={draft.depthRamp.end}
                                                                unit="percentFine"
                                                                disabled={isBuiltin}
                                                                min={0}
                                                                max={2}
                                                                suffix={t("vibrato_depth_ramp_end")}
                                                                ariaLabel={`${t("vibrato_depth_ramp")} ${t("vibrato_depth_ramp_end")}`}
                                                                onCommit={(next) =>
                                                                    patch({
                                                                        depthRamp: {
                                                                            ...draft.depthRamp,
                                                                            end: next,
                                                                        },
                                                                    })
                                                                }
                                                            />
                                                        </Flex>
                                                    </AppField>
                                                    <AppField label={t("vibrato_bias")}>
                                                        <AppNumberField
                                                            value={draft.biasCents}
                                                            unit="cents"
                                                            disabled={isBuiltin}
                                                            ariaLabel={t("vibrato_bias")}
                                                            onCommit={(next) =>
                                                                patch({ biasCents: next })
                                                            }
                                                        />
                                                    </AppField>
                                                    <AppField label={t("vibrato_irregularity")}>
                                                        <Flex align="center" gap="2" wrap="wrap">
                                                            <AppSlider
                                                                unit="percent"
                                                                min={0}
                                                                max={100}
                                                                disabled={isBuiltin}
                                                                value={Math.round(
                                                                    draft.irregularity,
                                                                )}
                                                                ariaLabel={t(
                                                                    "vibrato_irregularity",
                                                                )}
                                                                onChange={(next) =>
                                                                    patch({ irregularity: next })
                                                                }
                                                            />
                                                            <AppSliderReadout>
                                                                {`${formatNumber(draft.irregularity)}%`}
                                                            </AppSliderReadout>
                                                            <AppIconButton
                                                                icon={<ShuffleIcon />}
                                                                /*
                                                                 * 不规则度为 0 时**没有图案可换** —— 噪声整个被乘掉了，
                                                                 * 换种子不会改变任何一帧。此时禁用并说明原因，而不是
                                                                 * 让用户点了半天看不出变化（新建的预设默认就是 0）。
                                                                 */
                                                                tooltip={
                                                                    draft.irregularity > 0
                                                                        ? t("vibrato_seed_roll")
                                                                        : t(
                                                                              "vibrato_seed_roll_needs_irregularity",
                                                                          )
                                                                }
                                                                disabled={
                                                                    isBuiltin ||
                                                                    !(draft.irregularity > 0)
                                                                }
                                                                onClick={() =>
                                                                    patch({
                                                                        seed: randomVibratoSeed(),
                                                                    })
                                                                }
                                                            />
                                                        </Flex>
                                                    </AppField>
                                                </AppFormSection>

                                                <AppFormSection title={t("vibrato_section_rate")}>
                                                    <AppField label={t("vibrato_rate_mode")}>
                                                        {isBuiltin ? (
                                                            <span className="hs-type-label">
                                                                {t(RATE_MODE_KEYS[draft.rateMode])}
                                                            </span>
                                                        ) : (
                                                            <AppSegmentedControl<VibratoRateMode>
                                                                value={draft.rateMode}
                                                                ariaLabel={t("vibrato_rate_mode")}
                                                                onChange={(next) =>
                                                                    patch({ rateMode: next })
                                                                }
                                                                options={(
                                                                    ["hz", "cycles"] as const
                                                                ).map((mode) => ({
                                                                    value: mode,
                                                                    label: t(RATE_MODE_KEYS[mode]),
                                                                }))}
                                                            />
                                                        )}
                                                    </AppField>
                                                    {draft.rateMode === "hz" ? (
                                                        <AppField label={t("vibrato_rate_label")}>
                                                            <AppNumberField
                                                                value={draft.rateHz}
                                                                unit="vibratoHz"
                                                                disabled={isBuiltin}
                                                                min={0.1}
                                                                max={20}
                                                                ariaLabel={t("vibrato_rate_label")}
                                                                onCommit={(next) =>
                                                                    patch({ rateHz: next })
                                                                }
                                                            />
                                                        </AppField>
                                                    ) : (
                                                        <AppField label={t("vibrato_cycles")}>
                                                            <AppNumberField
                                                                value={draft.cycles}
                                                                unit="integer"
                                                                disabled={isBuiltin}
                                                                min={0.5}
                                                                max={128}
                                                                ariaLabel={t("vibrato_cycles")}
                                                                onCommit={(next) =>
                                                                    patch({ cycles: next })
                                                                }
                                                            />
                                                        </AppField>
                                                    )}
                                                    <AppField label={t("vibrato_rate_ramp")}>
                                                        <AppNumberField
                                                            value={draft.rateRampEnd}
                                                            unit="percentFine"
                                                            disabled={isBuiltin}
                                                            min={0.25}
                                                            max={4}
                                                            ariaLabel={t("vibrato_rate_ramp")}
                                                            onCommit={(next) =>
                                                                patch({ rateRampEnd: next })
                                                            }
                                                        />
                                                    </AppField>
                                                    <AppSwitchRow
                                                        label={t("vibrato_align_cycles")}
                                                        checked={draft.alignCycles}
                                                        disabled={isBuiltin}
                                                        onCheckedChange={(checked) =>
                                                            patch({ alignCycles: checked })
                                                        }
                                                    />
                                                </AppFormSection>

                                                <AppFormSection
                                                    title={t("vibrato_section_envelope")}
                                                >
                                                    <AppField label={t("vibrato_attack")}>
                                                        <Flex align="center" gap="2" wrap="wrap">
                                                            <AppNumberField
                                                                value={draft.attackMs}
                                                                unit="milliseconds"
                                                                disabled={isBuiltin}
                                                                min={0}
                                                                ariaLabel={t("vibrato_attack")}
                                                                onCommit={(next) =>
                                                                    patch({ attackMs: next })
                                                                }
                                                            />
                                                            <AppSelect
                                                                value={draft.attackCurve}
                                                                disabled={isBuiltin}
                                                                ariaLabel={t("vibrato_curve")}
                                                                onValueChange={(value) =>
                                                                    patch({
                                                                        attackCurve:
                                                                            value as EnvelopeCurve,
                                                                    })
                                                                }
                                                                options={ENVELOPE_CURVE_ORDER.map(
                                                                    (curve) => ({
                                                                        value: curve,
                                                                        label: t(
                                                                            ENVELOPE_CURVE_KEYS[
                                                                                curve
                                                                            ],
                                                                        ),
                                                                    }),
                                                                )}
                                                            />
                                                        </Flex>
                                                    </AppField>
                                                    <AppField label={t("vibrato_release")}>
                                                        <Flex align="center" gap="2" wrap="wrap">
                                                            <AppNumberField
                                                                value={draft.releaseMs}
                                                                unit="milliseconds"
                                                                disabled={isBuiltin}
                                                                min={0}
                                                                ariaLabel={t("vibrato_release")}
                                                                onCommit={(next) =>
                                                                    patch({ releaseMs: next })
                                                                }
                                                            />
                                                            <AppSelect
                                                                value={draft.releaseCurve}
                                                                disabled={isBuiltin}
                                                                ariaLabel={t("vibrato_curve")}
                                                                onValueChange={(value) =>
                                                                    patch({
                                                                        releaseCurve:
                                                                            value as EnvelopeCurve,
                                                                    })
                                                                }
                                                                options={ENVELOPE_CURVE_ORDER.map(
                                                                    (curve) => ({
                                                                        value: curve,
                                                                        label: t(
                                                                            ENVELOPE_CURVE_KEYS[
                                                                                curve
                                                                            ],
                                                                        ),
                                                                    }),
                                                                )}
                                                            />
                                                        </Flex>
                                                    </AppField>
                                                    <AppField label={t("vibrato_phase")}>
                                                        <AppNumberField
                                                            value={draft.startPhaseDeg}
                                                            unit="integer"
                                                            disabled={isBuiltin}
                                                            min={0}
                                                            max={360}
                                                            ariaLabel={t("vibrato_phase")}
                                                            onCommit={(next) =>
                                                                patch({ startPhaseDeg: next })
                                                            }
                                                        />
                                                    </AppField>
                                                </AppFormSection>

                                                <AppFormSection
                                                    title={t("vibrato_section_baseline")}
                                                >
                                                    <AppField label={t("vibrato_baseline")}>
                                                        <AppSelect
                                                            value={draft.baseline}
                                                            disabled={isBuiltin}
                                                            onValueChange={(value) =>
                                                                patch({
                                                                    baseline: value as BaselineMode,
                                                                })
                                                            }
                                                            options={BASELINE_MODE_ORDER.map(
                                                                (mode) => ({
                                                                    value: mode,
                                                                    label: t(
                                                                        BASELINE_MODE_KEYS[mode],
                                                                    ),
                                                                }),
                                                            )}
                                                        />
                                                    </AppField>
                                                    <AppField label={t("vibrato_blend")}>
                                                        <Flex align="center" gap="2" wrap="wrap">
                                                            <AppSlider
                                                                unit="percent"
                                                                min={0}
                                                                max={100}
                                                                disabled={
                                                                    isBuiltin ||
                                                                    draft.baseline !== "existing"
                                                                }
                                                                value={Math.round(draft.blend)}
                                                                ariaLabel={t("vibrato_blend")}
                                                                onChange={(next) =>
                                                                    patch({ blend: next })
                                                                }
                                                            />
                                                            <AppSliderReadout>
                                                                {`${formatNumber(draft.blend)}%`}
                                                            </AppSliderReadout>
                                                        </Flex>
                                                    </AppField>
                                                </AppFormSection>
                                            </AppForm>
                                        </Box>
                                    </ScrollArea>
                                </Flex>
                            ) : (
                                <span className="hs-type-caption">
                                    {t("vibrato_manager_empty")}
                                </span>
                            )}
                        </Box>
                    </Flex>

                    {/* 预设行右键菜单：把"针对这一条预设"的动作收拢到指针处。
                        **必须渲染在对话框内部** —— 手写菜单的层级（--qt-z-menu: 999）
                        低于对话框（--qt-z-dialog: 1000），挂在对话框外面会被整个盖住、
                        点不到；而且它一旦跑到对话框的 DOM 之外，Radix 还会把点击当成
                        "点到了弹窗外面"从而顺手把窗口关掉。放在对话框子树里，层级归属
                        对话框自己的 stacking context，就不会被压住。
                        `position: fixed` 仍按视口定位（对话框没有常驻 transform），
                        也不会被祖先的 `overflow: auto` 裁掉。 */}
                    {presetMenu ? (
                        <AppContextMenu
                            x={presetMenu.x}
                            y={presetMenu.y}
                            items={presetMenuItems}
                            ariaLabel={t("vibrato_manager_row_menu")}
                            onClose={() => setPresetMenu(null)}
                        />
                    ) : null}
                </Flex>
            </AppDialog>

            {/* 文件输入必须由**常驻**组件渲染：文件选择期间 change 的接收方要活着
                （用户第一次选 a.json、第二次又选同一文件时，不清空 value 就不会再
                触发 —— 这条由原语处理，这里只保证宿主不随菜单卸载）。 */}
            <AppFileInput
                inputRef={fileInputRef}
                accept=".json,application/json"
                onFiles={(files) => void handleImport(files)}
            />

            <AppConfirmDialog
                open={deleteTarget !== null}
                onOpenChange={(next) => {
                    if (!next) setDeleteTarget(null);
                }}
                title={t("vibrato_manager_delete")}
                message={t("vibrato_manager_delete_confirm").replace(
                    "{name}",
                    deleteTarget ? vibratoPresetLabel(deleteTarget, t) : "",
                )}
                confirmLabel={t("vibrato_manager_delete")}
                cancelLabel={t("cancel")}
                intent="danger"
                onConfirm={handleDelete}
            />
        </>
    );
}

interface PresetRowProps {
    preset: VibratoPreset;
    selected: boolean;
    active: boolean;
    /** 是否启用（停用的预设不出现在工具栏列表里，循环切换也会跳过）。 */
    enabled: boolean;
    /** 可拖拽排序（用户预设；系统预设顺序固定）。 */
    reorderable?: boolean;
    onSelect: () => void;
    onActivate: () => void;
    onToggleEnabled: () => void;
    onContextMenu: (x: number, y: number) => void;
    /** 行上按下指针（拖拽排序的起点）。 */
    onDragPointerDown?: (event: React.PointerEvent<HTMLDivElement>) => void;
    /** 就地重命名：输入框覆盖在名字上。 */
    renaming?: boolean;
    renameValue?: string;
    onRenameChange?: (value: string) => void;
    onRenameCommit?: () => void;
    onRenameCancel?: () => void;
}

/**
 * 列表行：单击选中（编辑它），双击设为当前使用，右侧按钮启用 / 停用；
 * 预设还可以按住行上下拖拽排序。
 *
 * 【为什么分开】"编辑某个预设"与"现在就用某个预设"是两件事：用户可能正在调
 * 一个还没调好的预设，却仍然希望拖拽用着上一个。合起来会让编辑动作顺带改掉
 * 当前音色。
 */
function PresetRow({
    preset,
    selected,
    active,
    enabled,
    reorderable = false,
    onSelect,
    onActivate,
    onToggleEnabled,
    onContextMenu,
    onDragPointerDown,
    renaming = false,
    renameValue,
    onRenameChange,
    onRenameCommit,
    onRenameCancel,
}: PresetRowProps) {
    const { t } = useI18n();
    const description = vibratoPresetDescription(preset, t);
    const label = vibratoPresetLabel(preset, t);
    const summary = description ?? vibratoPresetSummary(preset, t);

    /*
     * 聚焦放在 effect 里，而不是用 `autoFocus`。
     *
     * 【为什么】重命名是从右键菜单点开的，而菜单关闭时会把"打开前的焦点"还回去
     * （`AppContextMenu` 的焦点归还）。`autoFocus` 在**挂载那一刻**就抢焦点，紧接着
     * 菜单的被动 effect 清理又把焦点还给了那一行 —— 输入框刚出现就被 blur，而 blur
     * 会提交，于是它立刻消失，看起来"点了重命名没反应"。effect 在清理之后运行，
     * 顺序才对。
     */
    const renameInputRef = useRef<HTMLInputElement | null>(null);
    useEffect(() => {
        if (!renaming) return;
        const input = renameInputRef.current;
        if (!input) return;
        input.focus();
        // 全选：重命名多半是整体替换，直接敲字即可。
        input.select();
    }, [renaming]);

    return (
        <Flex align="center" gap="1" style={{ minWidth: 0, flex: 1 }}>
            <Box style={{ minWidth: 0, flex: 1 }}>
                <AppListRow
                    selected={selected}
                    density="compact"
                    role="option"
                    onClick={onSelect}
                    onDoubleClick={onActivate}
                    onPointerDown={onDragPointerDown}
                    onContextMenu={(event) => {
                        event.preventDefault();
                        onContextMenu(event.clientX, event.clientY);
                    }}
                    // 可拖拽的行给一个"抓得住"的光标；不可拖的（系统预设）保持默认。
                    className={reorderable ? "cursor-grab" : undefined}
                    // 项目自定义气泡（不是浏览器原生 title）：样式与全应用一致。
                    tooltip={enabled ? summary : `${summary} · ${t("vibrato_manager_disabled")}`}
                >
                    {/* `flex: 1` 让这一行铺满列表行：重命名时输入框才有可用的宽度去
                        撑开，而不是反过来把行撑宽（见输入框上的 `size` 说明）。 */}
                    <Flex align="center" gap="2" style={{ minWidth: 0, flex: 1 }}>
                        {active ? <span aria-hidden="true">●</span> : null}
                        <VibratoPresetGlyph preset={preset} width={40} height={14} />
                        {renaming ? (
                            <input
                                ref={renameInputRef}
                                /*
                                 * `size={1}` 是关键：`<input>` 默认 `size=20`，其固有宽度
                                 * 约 170px —— 比列表列（208px）减去字形后还宽，会把整行
                                 * 撑出去、连累整个列表横向位移。压到 1 之后它不再贡献
                                 * 固有宽度，只按 `flex-1` 填满行内剩余空间。
                                 */
                                size={1}
                                className="hs-type-label min-w-0 flex-1 rounded border border-qt-border bg-qt-window px-1 py-0 text-qt-text outline-none focus:border-qt-highlight"
                                value={renameValue ?? ""}
                                aria-label={t("vibrato_manager_rename")}
                                // 行本身带点击 / 双击 / 拖拽：输入框内的事件一律不外泄，
                                // 否则在框里拖选文字会触发"拖拽排序"、双击会激活预设。
                                onPointerDown={(event) => event.stopPropagation()}
                                onClick={(event) => event.stopPropagation()}
                                onDoubleClick={(event) => event.stopPropagation()}
                                onChange={(event) => onRenameChange?.(event.target.value)}
                                onKeyDown={(event) => {
                                    // 先于窗口级 Escape 关闭处理（与 Clip 的内联重命名
                                    // 同一约定）。
                                    event.stopPropagation();
                                    if (event.key === "Enter") onRenameCommit?.();
                                    else if (event.key === "Escape") onRenameCancel?.();
                                }}
                                // 失焦即提交：点别处不该丢掉刚敲的名字。
                                onBlur={() => onRenameCommit?.()}
                            />
                        ) : (
                            <span
                                className="hs-type-label"
                                style={{
                                    overflow: "hidden",
                                    textOverflow: "ellipsis",
                                    whiteSpace: "nowrap",
                                    // 停用的预设压暗：一眼看出它不会出现在工具栏里。
                                    opacity: enabled ? undefined : 0.45,
                                }}
                            >
                                {label}
                            </span>
                        )}
                    </Flex>
                </AppListRow>
            </Box>
            {/* 启用 / 停用：停用后不进工具栏列表、拖拽切换也跳过。 */}
            <AppIconButton
                size="sm"
                emphasis={enabled ? "neutral" : "accent"}
                icon={enabled ? <EyeOpenIcon /> : <EyeNoneIcon />}
                // 提示的是**这个按钮点了会做什么**：启用中就说"停用"，反之亦然。
                tooltip={enabled ? t("vibrato_manager_disable") : t("vibrato_manager_enable")}
                onClick={onToggleEnabled}
            />
        </Flex>
    );
}
