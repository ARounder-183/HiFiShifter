/**
 * 颤音预设管理器。
 *
 * 【结构】左侧预设列表、右侧编辑器（波形预览 + 参数），照 `CustomScaleDialog`
 * 的分工：列表负责"选哪一个"，编辑器负责"捏成什么样"。
 *
 * 【只读系统预设】系统预设禁用一切字段，只有「复制为自定义」可用。这样出厂
 * 预设永远可复原，而"改坏了"不会变成不可逆 —— 比"允许改但提供重置"更简单，
 * 也不会让用户对着一个与出厂说明不符的"自然"预设困惑。
 *
 * 【草稿-保存】编辑改的是局部草稿，点「保存」才 dispatch + 落盘。与自动备份 /
 * 录音 / 渲染缓存几个对话框一致；预设是"调完再定"的东西，逐键落盘只会让撤销
 * 与失败恢复都变复杂。
 */

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PlayIcon, ShuffleIcon, StopIcon, EyeNoneIcon, EyeOpenIcon } from "@radix-ui/react-icons";
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
    sanitizeVibratoPreset,
} from "../../features/vibrato/vibratoPresets";
import { resolveVibratoPresets } from "../../features/vibrato/vibratoPresetList";
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
import { exportVibratoPresetsJson } from "../../services/api/jsonExport";
import { buildAuditionCurve, vibratoAudition } from "../../features/vibrato/vibratoAudition";
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
    AppIconButton,
    AppConfirmDialog,
    AppDialog,
    AppField,
    AppForm,
    AppFormSection,
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
    VibratoPreviewCanvas,
    type VibratoPreviewGestureInfo,
    type VibratoPreviewModifiers,
} from "./vibrato/VibratoPreviewCanvas";
import {
    BASELINE_MODE_KEYS,
    BASELINE_MODE_ORDER,
    ENVELOPE_CURVE_KEYS,
    ENVELOPE_CURVE_ORDER,
    PREVIEW_DEFAULT,
    RATE_MODE_KEYS,
    WAVE_SHAPE_KEYS,
    WAVE_SHAPE_ORDER,
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
    applyPreviewGesture,
    cycleWidthPxFor,
    handleLayoutFor,
    scalePreviewGestureDeltas,
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

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    /** 预设编辑时的"当前参数"：深度按它换算成原生单位显示与编辑。 */
    editParam?: string;
    paramRange?: { min: number; max: number };
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

export function VibratoPresetDialog({
    open,
    onOpenChange,
    editParam = "pitch",
    paramRange,
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
     * 编辑中的草稿。`null` 表示尚未选过 —— 打开时按活动预设播种。
     *
     * 【为什么不用 `useDialogDraft`】那个钩子播种的是"打开那一刻的值"，而这里
     * 草稿会随列表点选反复替换；用普通 state + 打开时的 effect 更直白。
     */
    const [draft, setDraft] = useState<VibratoPreset | null>(null);
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
    /** 试听是否在响（驱动播放 / 停止按钮的图标与文案）。 */
    const [auditionPlaying, setAuditionPlaying] = useState(false);
    /** 导入 / 导出用的隐藏文件输入（必须常驻挂载，见 `AppFileInput` 的说明）。 */
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    /**
     * 导入 / 导出后的一条行内反馈。
     *
     * 【为什么是行内而不是模态】导入是高频小操作，模态打断太重；而「什么都不说」
     * 正是主题导入曾经的缺陷 —— 坏文件静默无反应，用户以为导入不起作用。
     */
    const [ioNotice, setIoNotice] = useState<{ text: string; danger: boolean } | null>(null);
    /**
     * 预览纵轴的半幅（cents）—— **一次性拟合，编辑期间不动**。
     *
     * 【为什么要稳定】若标尺跟着当前深度自适应，波形永远填满画布：调深度时看到的
     * 只是整幅在竖直方向"抖一下"，读不出幅度大小。把标尺固定下来，波形高度就等于
     * 深度，配合轴上的刻度标签可以直读。换预设 / 打开 / 点「适应」/ 保存时才重新
     * 拟合（那是离散动作，不是每帧都在跳）。
     */
    const [previewHalfCents, setPreviewHalfCents] = useState<number>(() => fitPreviewRangeCents(0));

    /** 按给定草稿重新拟合纵轴。 */
    const fitPreviewAxis = useCallback((preset: VibratoPreset | null) => {
        if (!preset) return;
        setPreviewHalfCents(fitPreviewRangeCents(buildVibratoPreview(preset).peakCents));
    }, []);

    /**
     * 本次"打开"是否已经播种过草稿。
     *
     * 【为什么需要它】这个 effect 依赖 `resolved.all`，而它会在**保存 / 切换预设**时
     * 变化（切换前会先把未保存的改动写回库）。若不设闸，保存动作触发的重算会把草稿
     * 重新播种回"当前活动预设"—— 于是用户点了另一条预设，改动保存了、选中却跳了回去，
     * 得再点一次。播种只在"打开"这一沿做一次即可。
     */
    const seededForOpenRef = useRef(false);

    // 打开时播种：优先用活动预设，找不到就回落到列表首项。
    useEffect(() => {
        if (!open) {
            seededForOpenRef.current = false;
            return;
        }
        if (seededForOpenRef.current) return;
        seededForOpenRef.current = true;
        const active =
            resolved.all.find((preset) => preset.id === session.activeVibratoPresetId) ??
            resolved.all[0];
        if (active) {
            // 对话框打开是一次离散动作：按当前活动预设播种局部草稿，并把预览纵轴
            // 拟合一次（编辑期间保持不动）。
            // eslint-disable-next-line react-hooks/set-state-in-effect -- 打开时按活动预设播种草稿（既有模式）
            setDraft(active);
            fitPreviewAxis(active);
        }
    }, [open, resolved.all, session.activeVibratoPresetId, fitPreviewAxis]);

    const isBuiltin = Boolean(draft?.builtin);
    const previewSamples = useMemo(() => (draft ? buildVibratoPreview(draft) : null), [draft]);

    // 卸载兜底：整个组件被卸载（而不只是关闭）时也要停 —— 否则试听会在
    // 管理器消失之后继续响。这里不含 setState；关闭时的按钮状态复位在
    // `handleOpenChange`（事件处理器）里做，不违反 effect 的规则。
    useEffect(() => () => vibratoAudition.stop(), []);

    /** 关闭对话框：先停试听、复位按钮，再向上传播。 */
    function handleOpenChange(next: boolean) {
        if (!next) {
            vibratoAudition.stop();
            setAuditionPlaying(false);
        }
        onOpenChange(next);
    }

    /** 播放 / 停止当前草稿的试听。自然结束后引擎回调把按钮切回「播放」。 */
    function toggleAudition() {
        if (auditionPlaying || !previewSamples) {
            vibratoAudition.stop();
            setAuditionPlaying(false);
            return;
        }
        // 不变量「听到的 = 看到的」：直接吃预览的同一份数据，不重算。
        const started = vibratoAudition.play(buildAuditionCurve(previewSamples), () =>
            setAuditionPlaying(false),
        );
        setAuditionPlaying(started);
    }

    /** 草稿的局部更新（不落盘）。 */
    function patch(changes: Partial<VibratoPreset>) {
        setDraft((prev) => (prev ? { ...prev, ...changes } : prev));
    }

    /**
     * 预览画布手势的活动状态（起点快照 + 区域）。
     *
     * 【为什么快照】主体拖动的深度换算依赖画布**当前**的纵轴标尺，而深度一改
     * 标尺（按峰值自适应）也跟着变；每帧重取会让拖动变成非线性甚至反向。起点
     * 取一次，整段手势按同一套几何走。
     */
    const previewGestureRef = useRef<{
        zone: PreviewZone;
        snapshot: PreviewGestureSnapshot;
    } | null>(null);

    /** 画布手势开始：登记起点快照（窗口时长取预览默认几何）。 */
    function handlePreviewGestureStart(zone: PreviewZone, info: VibratoPreviewGestureInfo) {
        if (!draft || isBuiltin) return;
        const windowMs = (PREVIEW_DEFAULT.frameCount - 1) * PREVIEW_DEFAULT.framePeriodMs;
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
        // 按住「精细调整」时位移缩到 1/10 —— 与滚轮调参的精细调整同一比例，
        // 大深度预设也能一像素一像素地捏。
        const scaled = scalePreviewGestureDeltas(
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
        const copy = duplicateVibratoPreset(source);
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
        dispatch(removeVibratoPreset(deleteTarget.id));
        void dispatch(persistUiSettings());
        const removedId = deleteTarget.id;
        setDeleteTarget(null);
        setDraft((prev) => (prev?.id === removedId ? null : prev));
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
                        intent: "primary",
                        disabled: !draft || isBuiltin,
                        // 保存**不关闭**对话框：用户常要"先存一版、接着调"，
                        // 存完就把窗口收掉等于逼他重新打开。
                        autoClose: false,
                        onClick: handleSave,
                    },
                    {
                        // 显式关闭按钮：保存不再关闭窗口之后，页脚里没有"退出"的
                        // 去处，只剩 Esc / 点外部 —— 两者都不显眼。
                        id: "close",
                        label: t("close"),
                        autoClose: false,
                        onClick: () => handleOpenChange(false),
                    },
                ]}
                // 默认动作仍是「保存」：关闭排在它右边（页脚最右是关闭的常见排布），
                // 但 Enter 不该变成"关掉窗口"。
                defaultActionId="save"
            >
                <Flex
                    direction="column"
                    gap="3"
                    data-vibrato-content
                    className="min-h-0"
                    style={{ height: CONTENT_HEIGHT }}
                >
                    {/* ---- 波形预览（整行置顶，不参与任何滚动） ----
                        放在两栏之上而不是塞进参数流的头部：整行宽度读波形更清楚，
                        且它不属于任何滚动区，调参数时**永远**不会滚出视野。 */}
                    {draft && previewSamples ? (
                        <>
                            <Box className="rounded border border-qt-border bg-qt-panel p-2">
                                <VibratoPreviewCanvas
                                    samples={previewSamples}
                                    ariaLabel={t("vibrato_preview")}
                                    halfCents={previewHalfCents}
                                    handles={previewHandles}
                                    onGestureStart={handlePreviewGestureStart}
                                    onGestureMove={handlePreviewGestureMove}
                                    onGestureEnd={handlePreviewGestureEnd}
                                />
                                <Flex justify="between" align="center" mt="1" gap="2">
                                    <Flex gap="2" align="center" style={{ minWidth: 0 }}>
                                        <span className="hs-type-caption">
                                            {`±${formatNumber(previewSamples.peakCents)} ${t("vibrato_unit_cents")}`}
                                        </span>
                                        <span className="hs-type-caption">
                                            {t("vibrato_cycles_estimate").replace(
                                                "{count}",
                                                formatNumber(cycleEstimate),
                                            )}
                                        </span>
                                    </Flex>
                                    <Flex gap="1" align="center">
                                        {/* 适应：把纵轴重新拟合到当前波形。标尺在编辑期间
                                            刻意保持不动（这样高度才等于深度），拖到超出量程
                                            或想重新看清形状时点它。 */}
                                        <AppButton
                                            size="sm"
                                            emphasis="soft"
                                            disabled={isBuiltin}
                                            onClick={() => fitPreviewAxis(draft)}
                                        >
                                            {t("vibrato_preview_fit")}
                                        </AppButton>
                                        {/* 试听：合成音色（钢琴卷帘琴键同款），不跑声码器。
                                            系统预设也能试听 —— 试听是只读操作。 */}
                                        <AppIconButton
                                            tooltip={
                                                auditionPlaying
                                                    ? t("vibrato_audition_stop")
                                                    : t("vibrato_audition_play")
                                            }
                                            size="sm"
                                            aria-pressed={auditionPlaying}
                                            onClick={toggleAudition}
                                            icon={
                                                auditionPlaying ? (
                                                    <StopIcon width="15" height="15" />
                                                ) : (
                                                    <PlayIcon width="15" height="15" />
                                                )
                                            }
                                        />
                                    </Flex>
                                </Flex>
                            </Box>
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
                                        className="min-h-0 flex-1"
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
                                                                disabled={
                                                                    isBuiltin ||
                                                                    (draft.cycle.kind !== "shape" &&
                                                                        !handDraw) ||
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
                                                            ariaLabel={t("vibrato_handdraw_open")}
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
                                                                tooltip={t("vibrato_seed_roll")}
                                                                disabled={isBuiltin}
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
                    <Flex align="center" gap="2" style={{ minWidth: 0 }}>
                        {active ? <span aria-hidden="true">●</span> : null}
                        <VibratoPresetGlyph preset={preset} width={40} height={14} />
                        {renaming ? (
                            <input
                                ref={renameInputRef}
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
