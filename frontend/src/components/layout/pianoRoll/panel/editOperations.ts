/**
 * 参数编辑器的命令执行体：`handleEditOp` 的那一千行 `switch`。
 *
 * 【为什么单独成文件】它是面板里最大的一块**纯执行**逻辑：一条 op 进来，算出要写
 * 哪些帧、走哪个编排（选区变换 / 剪贴板 / 颤音 / 转换），然后把结果交给后端。
 * 它不看 JSX、不摆布局，与"面板怎么排"毫无关系 —— 却和视图、滚动、绘制挤在
 * 同一个 hook 里，翻到它要滚过七千行。
 *
 * 【依赖怎么拿】33 个捕获值（store 快照、ref、setState、派生值）由调用方打包成
 * 一个对象传进来，类型是 `Pick<ModelDeps, …>` —— 字段清单由 AST 扫描
 * 得出，不是手抄。函数体一字未改：开头把那 33 个名字解构回局部变量，下面就是
 * 原样搬过来的实现。
 *
 * 【边界】只做"op → 编辑指令"的翻译与派发。它不写 React 状态之外的任何东西，
 * 也不绘制。撤销 / 重做的登记、快照刷新仍由模型侧的回调负责。
 */
import { setEditParam } from "../../../../features/session/sessionSlice";
import { resolveRootTrackId } from "../../../../features/session/trackUtils";
import { paramsApi } from "../../../../services/api/params";
import { webApi } from "../../../../services/webviewApi";
import type { ParamFramesPayload } from "../../../../types/api";
import {
    degreeInputToScaleSteps,
    snapToScale,
    snapToSemitone,
    transposePitchByScaleSteps,
} from "../../../../utils/musicalScales";
import { setActiveSurfaceExplicit } from "../../../../features/uiFocus/focusSurface";
import type { ScaleLike } from "../../../../utils/musicalScales";
import {
    pasteReaperClipboard,
    pasteVocalShifterClipboard,
} from "../../../../features/session/thunks/audioThunks";
import { clamp } from "../../timeline";
import {
    averageSelectionValues,
    smoothSelectionValues,
    smoothContextPadFrames,
} from "../selectionTransforms";
import { applySelectionEditOverRanges, type SelectionEditExtension } from "../selectionEditApply";
import {
    addFrameRange,
    addFrameRanges,
    frameRangeEnd,
    selectionBoundingSpan,
    selectionFromFrames,
    selectionToFrameRanges,
    subtractFrameRange,
    toggleFrameRange,
    type FrameRange,
    type ParamSelection,
} from "../paramSelection";
import {
    mapClipboardToTargetRanges,
    pasteTargetSelectionFromClipboard,
    toParamClipboardPayload,
    type ParamClipboardData,
    type ParamClipboardSegment,
} from "../paramClipboardMapping";
import { uploadFullResCurveSegments } from "../selectionEditData";
import { planParamConversion } from "../paramConversion";
import { editablePitchValue } from "../paramSmoothing";
import {
    DYN_FOLLOW_ORIG,
    dynMultiplicativeFactor,
    isDynParam,
    restoreDynSentinels,
} from "../paramRanges";
import { timeToFrame } from "../utils";
import { buildVibratoCurve } from "../../../../features/vibrato/vibratoCurve";
import {
    planVibratoTarget,
    suppressNonTargetFrames,
} from "../../../../features/vibrato/vibratoPitch";
import { sanitizeVibratoPreset } from "../../../../features/vibrato/vibratoPresets";
import type { VibratoPreset } from "../../../../features/vibrato/vibratoTypes";
import { resolveActiveVibratoPreset } from "../../../../features/vibrato/vibratoPresetList";
import { getParamShiftStep, parseParamShiftMagnitude } from "../paramShiftStep";
import {
    beginSelectionParamEdit,
    endSelectionParamEdit,
} from "../../../../features/session/selectionEditInFlight";
import {
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
    parseChildPitchOffsetParam,
} from "../childPitchOffsetParams";
import { buildChildOffsetPasteValues as buildChildOffsetPasteValuesHelper } from "../childPitchOffsetPaste";
import {
    readSystemClipboardObject,
    writeSystemClipboardObject,
} from "../../../../utils/systemClipboard";
import { SELECTION_SCOPE_EXEMPT_OPS, clipTimeRangeToFrameBounds } from "./panelHelpers";

import type { ModelDeps } from "./modelDeps";

/** 执行一条编辑命令所需的全部捕获值（字段清单由 AST 扫描得出）。 */
type EditOpDeps = Pick<
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
    | "pitchDeltaToDegreeSteps"
    | "pitchEnabled"
    | "processorParamsRef"
    | "projectScaleAtSec"
    | "resolveScaleFromToken"
    | "resolvedVibratoPresets"
    | "rootTrackId"
    | "s"
    | "scrollerRef"
    | "selectAllParamRange"
    | "selectionRef"
    | "setClipboardGuid"
    | "setMidiDialogOpen"
    | "setMidiDialogSelection"
    | "setMidiPath"
    | "setParamView"
    | "setSelectionUi"
    | "store"
    | "vibratoBaseline"
>;

/**
 * 执行一条编辑命令。
 *
 * @param deps 捕获值（store 快照 / ref / setState / 派生值）。
 * @param op 命令名（见 `keybindings` 与右键菜单的 op 表）。
 * @param data 命令参数（选区、目标参数、幅度等），各 op 自解释。
 */
export async function runEditOp(
    deps: EditOpDeps,
    op: string,
    data?: Record<string, unknown>,
): Promise<void> {
    const {
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
    } = deps;

    if (!rootTrackId) return;
    const fp = paramView?.framePeriodMs ?? 5;

    // ── 无选区时的隐式全选 ────────────────────────────────────────
    // 本编辑器里的操作几乎都以**参数选区**为作用域（见
    // SELECTION_SCOPE_EXEMPT_OPS 的说明）。没有选区时它们原本静默什么都
    // 不做 —— 用户得先"全选"再点一次，而右键菜单里点下去毫无反应更容易被
    // 当成功能坏了。这里统一改为：**先全选，再执行**，作用域 = 整条参数曲线
    // （与用户手动全选完全等价：同样的 `selectionFromFrames(0, 工程末端)`）。
    //
    // 非「选择」工具下全选不生效（与菜单命令同一守卫），此时行为与改动前
    // 逐字一致：这些操作在绘制类工具里本就没有意义。
    const hadSelectionAtEntry = selectionRef.current;
    if (
        !SELECTION_SCOPE_EXEMPT_OPS.has(op) &&
        (!hadSelectionAtEntry || hadSelectionAtEntry.length === 0)
    ) {
        selectAllParamRange();
    }

    if (op === "selectAll") {
        selectAllParamRange();
        return;
    }
    if (op === "deselect") {
        if (s.toolMode !== "select") return;
        selectionRef.current = null;
        setSelectionUi(null);
        invalidate();
        return;
    }

    // 时间轴右键手势（按住 `modifier.clipRangeToParamSelection`，默认 Alt，
    // 右键单击 Clip；`TimelinePanel` 派发）：按 Clip 起止范围在参数编辑器
    // 内创建选区，并把交互焦点切到参数编辑器侧 ——
    // 复制/剪切路由（resolveCopyCutRoute 依据 selectionContext，经由下方
    // selectionUi 同步派发 setParamSelectionActive 标记）与活动表面
    // （focusSurface，外来源粘贴兜底等）随之指向参数编辑器。
    //
    // mode（来自时间轴的右键手势）：
    //   - "replace"（缺省）：替换为该块范围，与旧行为逐字一致；
    //   - "add"：把该块范围并入（重叠/相接自动合并）；
    //   - "toggle"：该块范围已被完整覆盖则挖掉，否则并入 —— 同一个块
    //     连按两次回到原状（`modifier.clipRangeToParamSelection` 手势）。
    if (op === "selectClipParamRange") {
        const clipId = typeof data?.clipId === "string" ? data.clipId : "";
        const clip = store.getState().session.clips.find((entry) => entry.id === clipId);
        if (!clip) return;
        const { startBound, endBound } = clipTimeRangeToFrameBounds(clip, fp);
        const rawMode = typeof data?.mode === "string" ? data.mode : "replace";
        const mode: "replace" | "add" | "toggle" =
            rawMode === "add" || rawMode === "toggle" ? rawMode : "replace";
        selectionRef.current =
            mode === "add"
                ? addFrameRange(selectionRef.current, startBound, endBound)
                : mode === "toggle"
                  ? toggleFrameRange(selectionRef.current, startBound, endBound)
                  : selectionFromFrames(startBound, endBound - startBound);
        setSelectionUi(selectionRef.current);
        setActiveSurfaceExplicit("pianoRoll");
        // 单素材切入参数选区与批量入口一致，也交付画布本地快捷键的DOM焦点。
        scrollerRef.current?.focus({ preventScroll: true });
        invalidate();
        return;
    }

    // 音频块范围 → 参数编辑器选区（批量入口；单个音频块的右键手势见
    // selectClipParamRange 的 add/toggle 模式）。
    //
    // 只取**当前参数编辑器所属根轨道组**内的音频块：参数编辑器一次只
    // 展示一条根轨道的参数，跨轨道的块范围对它没有意义（静默忽略，避免
    // 用户以为"加进去了"）。多段求并/相减交给 paramSelection 归一化
    // （相邻自动合并；相减可能把一段切成两段 —— 断层即数据）。
    if (op === "addClipsToParamSelection" || op === "removeClipsFromParamSelection") {
        const session = store.getState().session;
        const requestedIds = Array.isArray(data?.clipIds)
            ? (data.clipIds as unknown[]).filter((id): id is string => typeof id === "string")
            : session.multiSelectedClipIds.length > 0
              ? session.multiSelectedClipIds
              : session.selectedClipId
                ? [session.selectedClipId]
                : [];
        const ranges: FrameRange[] = [];
        for (const id of requestedIds) {
            const clip = session.clips.find((entry) => entry.id === id);
            if (!clip) continue;
            if (resolveRootTrackId(session.tracks, clip.trackId) !== rootTrackId) continue;
            const { startBound, endBound } = clipTimeRangeToFrameBounds(clip, fp);
            ranges.push({ startFrame: startBound, frameCount: endBound - startBound });
        }
        if (ranges.length === 0) return;

        let next: ParamSelection | null = selectionRef.current;
        if (op === "addClipsToParamSelection") {
            next = addFrameRanges(next, ranges);
        } else {
            for (const range of ranges) {
                next = subtractFrameRange(next, range.startFrame, frameRangeEnd(range));
            }
        }
        selectionRef.current = next;
        setSelectionUi(next);
        setActiveSurfaceExplicit("pianoRoll");
        // 素材范围已转入参数编辑，DOM焦点也须转交，否则本地移调快捷键收不到事件。
        scrollerRef.current?.focus({ preventScroll: true });
        invalidate();
        return;
    }

    // VocalShifter clipboard paste stays a dedicated menu action
    // (file-based clipboard). 多选区：把全部选区段交给后端，落盘时只写
    // 落在这些段内的帧（断层不会被填充），偏移基准为首段起点。
    if (op === "pasteVocalShifter") {
        const sel2 = selectionRef.current;
        const selRanges = sel2 ? selectionToFrameRanges(sel2) : [];
        void dispatch(
            pasteVocalShifterClipboard({
                selectionRanges: selRanges.length > 0 ? selRanges : undefined,
                activeParam: editParam,
            }),
        );
        bumpRefreshToken();
        return;
    }

    // REAPERMedia fallback used by the normal paste operation when no
    // HiFiShifter param clipboard data is available. 多选区下这个回退只
    // 关心"往哪里贴"，用选区包围区间（首段起点 → 末段终点）。
    const pasteReaperClipboardFallback = () => {
        const sel2 = selectionRef.current;
        let selArgs:
            | {
                  selectionStartFrame?: number;
                  selectionMaxFrames?: number;
              }
            | undefined;
        const bounding = selectionBoundingSpan(sel2);
        if (bounding) {
            selArgs = {
                selectionStartFrame: Math.max(0, bounding.startFrame),
                selectionMaxFrames: Math.max(1, bounding.frameCount),
            };
        }
        void (async () => {
            try {
                // Standard MIDI File on the clipboard opens the unified
                // MIDI import dialog.
                const midiCheck = await paramsApi.readMidiClipboardToMemory();
                if (midiCheck.ok && midiCheck.guid) {
                    midiDialogSourceRef.current = "reaperClipboard";
                    setClipboardGuid(midiCheck.guid);
                    setMidiPath(null);
                    setMidiDialogSelection(sel2 ? sel2.map((range) => ({ ...range })) : null);
                    midiDialogOpenParamsRef.current = {
                        editParam: s.editParam,
                        toolMode: s.toolMode,
                    };
                    setMidiDialogOpen(true);
                    return;
                }
            } catch {
                // Check failed; fall back to ordinary REAPERMedia paste.
            }
            try {
                // Avoid surfacing a paste error when the system
                // clipboard does not contain REAPERMedia data at all.
                const reaperCheck = await webApi.hasReaperClipboard();
                if (!reaperCheck?.ok || !reaperCheck?.available) return;
            } catch {
                return;
            }
            void dispatch(pasteReaperClipboard(selArgs));
        })();
        bumpRefreshToken();
    };

    /**
     * 解析参数线剪贴板：内部缓存 → 系统剪贴板（后者优先，保持"最后复制的
     * 胜出"：时间轴复制过 Clip 后系统槽位已换，内部缓存随之失效）。
     */
    const readParamClipboardForPaste = async (): Promise<ParamClipboardData | null> => {
        let clip = clipboardRef.current;
        try {
            const fromSystem = await readSystemClipboardObject("param");
            if (fromSystem) {
                clip = fromSystem;
                clipboardRef.current = clip;
            }
        } catch {
            // 系统剪贴板不可用 → 退回内部缓存。
        }
        return clip ?? null;
    };

    // ── 粘贴：剪贴板优先；无选区时按剪贴板 + 播放光标推导选区 ──────────
    // 粘贴与其它操作不同：它的"作用对象"完全由**剪贴板**决定（从哪里开始、
    // 铺多宽、中间有几个断层）。因此无选区时**不做全选**，而是以当前播放
    // 光标为复制起点，按剪贴板自己的段布局重建选区（段数 / 段长 / 断层照搬），
    // 再把数据贴进去 —— 选区就是"粘贴会落到哪里"。
    //
    // 【推导出的选区**先不发布**】它要与粘贴后的曲线**同一次提交**落地：
    // 先亮出一个空选区、隔几毫秒再填上数据，会让用户看到"先划选区、再粘贴"
    // 两步；而用户的心智是"选区出现时粘贴就已经完成了"。因此这里只把它交给
    // 下面的目标帧换算，真正的 ref/UI 更新发生在 paste 分支里（与本地曲线
    // 更新同一次 setState 批处理）。
    //
    // 剪贴板里没有参数线数据时保持既有语义：落到 REAPERMedia / MIDI 剪贴板
    // 回退（那个回退的目标不是参数选区）；音高编辑不可用时同理。
    let pasteClipboard: ParamClipboardData | null = null;
    let derivedPasteSelection: ParamSelection | null = null;
    if (op === "paste") {
        if (!pitchEnabled) {
            pasteReaperClipboardFallback();
            return;
        }
        pasteClipboard = await readParamClipboardForPaste();
        if (!pasteClipboard) {
            pasteReaperClipboardFallback();
            return;
        }
        const selAtEntry = selectionRef.current;
        if (!selAtEntry || selAtEntry.length === 0) {
            derivedPasteSelection = pasteTargetSelectionFromClipboard({
                clipboard: pasteClipboard,
                // 锚点 = 播放光标所在帧（与粘贴的帧制口径一致）。
                // 【调用时现读】播放头高频前进，`s` 是回调重建时的快照
                // （依赖数组刻意不含它，见下方 eslint-disable 注释），
                // 从闭包里取会把粘贴锚定到上一次渲染时的旧光标位置。
                anchorFrame: timeToFrame(store.getState().session.playheadSec, fp),
            });
            if (!derivedPasteSelection) {
                pasteReaperClipboardFallback();
                return;
            }
        }
    }

    // 目标选区：现有选区，或（粘贴且无选区时）刚推导出的那一份。
    const sel =
        (selectionRef.current?.length ? selectionRef.current : null) ?? derivedPasteSelection;
    if (!sel || sel.length === 0) return;
    if (!pitchEnabled) return;

    // 选区的帧区间集合（逐段独立）。选区本来就是帧制，这里只做起点夹取、
    // 帧数钳制与越界截断（见 selectionToFrameRanges）。
    const selFrameRanges = selectionToFrameRanges(sel);
    if (selFrameRanges.length === 0) return;
    const firstRange = selFrameRanges[0];
    const startFrame = firstRange.startFrame;

    /**
     * 逐段执行「取数 → 变换 → 回写」。
     *
     * 撤销点纪律：**整个批次只在第一个真正写入的段上打一次撤销点**
     * （回调返回 false 表示该段未写入，撤销点顺延到下一段）。
     *
     * @param writeRange (段, 段号, 本次是否应打撤销点) => 是否实际写入
     */
    const runPerRange = async (
        writeRange: (
            range: FrameRange,
            rangeIndex: number,
            isFirstWrite: boolean,
        ) => Promise<boolean>,
    ): Promise<boolean> => {
        let wrote = false;
        for (let i = 0; i < selFrameRanges.length; i += 1) {
            const didWrite = await writeRange(selFrameRanges[i], i, !wrote);
            if (didWrite) wrote = true;
        }
        return wrote;
    };

    /**
     * 粘贴的「本地先落」：把写入值先写进本地 paramView，并把**推导出的选区**
     * 在同一次 React 提交里发布。
     *
     * 【为什么需要】后端回写 + 重新取数要走若干个 IPC 往返。若只发布选区、
     * 等取数回来才显示曲线，用户看到的是"先出现一个空选区、隔一会儿才填上
     * 数据"两步；而用户的心智是"选区一出现，粘贴就已经完成"。这里与其它提交
     * 路径同一范式（先本地、后后端；失败时由 `bumpRefreshToken` 的取数纠正），
     * 于是选区与曲线在**同一次 setState 批处理**里落地 —— 感知上是一步。
     *
     * 已有选区时（`derivedPasteSelection` 为 null）只更新曲线，行为不变。
     *
     * @param writes 绝对帧号 + 逐帧值（全分辨率）。
     */
    const applyPasteLocally = (
        writes: readonly { startFrame: number; values: number[] }[],
    ): void => {
        if (derivedPasteSelection) {
            selectionRef.current = derivedPasteSelection;
            setSelectionUi(derivedPasteSelection);
        }
        const pv = paramViewRef.current;
        if (pv && writes.length > 0) {
            const step = Math.max(1, Math.floor(pv.stride));
            const nextEdit = pv.edit.slice();
            for (const write of writes) {
                for (let i = 0; i < write.values.length; i += 1) {
                    const idx = Math.round((write.startFrame + i - pv.startFrame) / step);
                    if (idx >= 0 && idx < nextEdit.length) {
                        nextEdit[idx] = write.values[i];
                    }
                }
            }
            setParamView({ ...pv, edit: nextEdit });
        }
        invalidate();
    };

    // 音量 ↔ 动态 曲线互转（后端单事务：基线补偿换算 + 源参数归位）。
    //
    // 【为什么换算在后端】两者最终增益的语义不同：volume 是乘性增益，
    // dyn 是「目标电平 / 原声基线」。等效互转需要逐帧基线补偿
    // （dyn_target = volume × orig、volume = target/orig），而权威基线与
    // "曲线哪些帧有数据"只有后端知道 —— 前端只传选区（旧的前端纯搬迁
    // 已被证伪：gain = v/orig ≠ v，且会把未画帧物化成显式基线值）。
    //
    // 撤销点在后端单次打点：Ctrl+Z 一次回退整个互转（含源归位）。
    if (op === "convertVolumeToDyn" || op === "convertDynToVolume") {
        // 菜单项只在选中 volume / dyn 时出现，但键盘/程序化触发仍要复核，
        // 避免用一个不匹配的方向覆盖掉用户真正在编辑的参数。
        const fromNarrowed: "volume" | "dyn" = op === "convertVolumeToDyn" ? "volume" : "dyn";
        if (editParam !== fromNarrowed) return;
        const plan = planParamConversion(fromNarrowed);
        if (!plan) return;

        const res = await paramsApi.convertMixParam(
            rootTrackId,
            fromNarrowed,
            selFrameRanges.map((range) => ({
                startFrame: range.startFrame,
                frameCount: range.frameCount,
            })),
        );
        if (!res?.ok) {
            // 基线分析未就绪是最常见的失败原因：保持静默（与其它操作对
            // not-ok 的处理一致），dyn_orig_updated 事件后用户重试即可。
            return;
        }
        bumpRefreshToken();
        // 目标参数可能刚获得第一个非默认值（尤其动态）→ 切换显示，
        // 让用户立刻看到互转结果而不是停在空白的源参数上。
        dispatch(setEditParam(plan.targetParam));
        return;
    }

    // 选区编辑统一入口：取数/编辑/边缘淡化/回写全部在
    // selectionEditApply 模块内完成（delta 空间交叉淡化 + 毫秒定标）。
    // 多选区**逐段独立**执行，整批只打一个撤销点。
    //
    // 回调收到**当前段**：移调度数 / 量化 / 均值量化要按本段起始帧解析
    // Tempo Map 音阶（见 applySelectionEditOverRanges 的说明）。
    const runSelectionEdit = async (
        editSelection: (currentSelectionVals: number[], range: FrameRange) => number[],
        extension?: SelectionEditExtension,
        options?: { preserveDynSentinels?: boolean },
    ) => {
        const ok = await applySelectionEditOverRanges({
            ranges: selFrameRanges,
            trackId: rootTrackId,
            param: editParam,
            framePeriodMs: fp,
            smoothnessPercent: clamp(
                Number(
                    (data?.edgeSmoothnessPercent as number | undefined) ?? s.edgeSmoothnessPercent,
                ) || 0,
                0,
                100,
            ),
            editSelection,
            extension,
            isEditable: editParam === "pitch" ? editablePitchValue : undefined,
            ...options,
        });
        if (ok) bumpRefreshToken();
    };

    switch (op) {
        case "copy": {
            // 逐段取全分辨率数据；段偏移 = 该段起点 − 首段起点，断层以
            // 偏移空洞的形式进入剪贴板（这是"不合并断层"的根本保障）。
            const segments: ParamClipboardSegment[] = [];
            let framePeriodMsFromBackend = fp;
            for (const range of selFrameRanges) {
                const res = await paramsApi.getParamFrames(
                    rootTrackId,
                    editParam,
                    range.startFrame,
                    range.frameCount,
                    1,
                    true,
                    isDynParam(editParam),
                );
                if (!res?.ok) continue;
                const payload = res as ParamFramesPayload;
                if (segments.length === 0) {
                    framePeriodMsFromBackend = Number(payload.frame_period_ms ?? fp) || fp;
                }
                // dyn：未画帧在复制时就编码回哨兵（负值）。后端
                // set_param_frames 入口原样接受负值 = 沿用原声，因此
                // 粘贴路径无需任何特判 —— "未画"语义跨复制/粘贴存活。
                const sentinels = payload.edit_sentinel;
                const values = (payload.edit ?? []).map((v, i) =>
                    sentinels?.[i] === true ? DYN_FOLLOW_ORIG : Number(v) || 0,
                );
                if (values.length === 0) continue;
                segments.push({
                    startFrame: range.startFrame - startFrame,
                    values,
                });
            }
            if (segments.length === 0) return;
            const clipboardData: ParamClipboardData = {
                param: editParam,
                framePeriodMs: framePeriodMsFromBackend,
                segments,
            };
            clipboardRef.current = clipboardData;
            try {
                await writeSystemClipboardObject(toParamClipboardPayload(clipboardData));
            } catch {
                // ignore clipboard write failures
            }
            // 刷新剪贴板预览
            invalidate();
            break;
        }
        case "cut": {
            // 复制部分与 copy 完全同构（多段 + 偏移），随后把各段恢复为
            // 原始值；恢复与复制在同一批次内完成，撤销一次整体回退。
            const segments: ParamClipboardSegment[] = [];
            let framePeriodMsFromBackend = fp;
            let restoredAny = false;
            for (const range of selFrameRanges) {
                const res = await paramsApi.getParamFrames(
                    rootTrackId,
                    editParam,
                    range.startFrame,
                    range.frameCount,
                    1,
                    true,
                    isDynParam(editParam),
                );
                if (!res?.ok) continue;
                const payload = res as ParamFramesPayload;
                if (segments.length === 0) {
                    framePeriodMsFromBackend = Number(payload.frame_period_ms ?? fp) || fp;
                }
                // dyn：未画帧编码回哨兵（与 copy 同口径，见该处说明）。
                const sentinels = payload.edit_sentinel;
                const values = (payload.edit ?? []).map((v, i) =>
                    sentinels?.[i] === true ? DYN_FOLLOW_ORIG : Number(v) || 0,
                );
                if (values.length > 0) {
                    segments.push({
                        startFrame: range.startFrame - startFrame,
                        values,
                    });
                }
                // 第一个真正写入的段打撤销点
                await paramsApi.restoreParamFrames(
                    rootTrackId,
                    editParam,
                    range.startFrame,
                    range.frameCount,
                    !restoredAny,
                );
                restoredAny = true;
            }
            if (segments.length > 0) {
                const clipboardData: ParamClipboardData = {
                    param: editParam,
                    framePeriodMs: framePeriodMsFromBackend,
                    segments,
                };
                clipboardRef.current = clipboardData;
                try {
                    await writeSystemClipboardObject(toParamClipboardPayload(clipboardData));
                } catch {
                    // ignore clipboard write failures
                }
            }
            invalidate();
            bumpRefreshToken();
            break;
        }
        case "paste": {
            // 剪贴板已在上方（进入通用守卫之前）解析过 —— 无选区时的目标
            // 选区就是据它推导的，这里直接复用同一份，避免再读一次系统剪贴板。
            const clip = pasteClipboard;
            if (!clip) return;

            // 剪贴板 → 目标选区的映射（预览与粘贴同源）：交集之外不写，
            // 断层两侧都保持原值。
            const writes = mapClipboardToTargetRanges({
                targetRanges: selFrameRanges,
                clipboard: clip,
            });
            if (writes.length === 0) return;

            if (clip.param !== editParam) {
                // 跨参数粘贴：仅支持 pitch → 子轨音高偏移参数（逐段转换）。
                const canConvert =
                    clip.param === "pitch" &&
                    (isChildPitchOffsetCentsParam(editParam) ||
                        isChildPitchOffsetDegreesParam(editParam));
                if (!canConvert) return;
                const targetParam = parseChildPitchOffsetParam(editParam);
                if (!targetParam) return;
                const resolvedRootTrackId = resolveRootTrackId(s.tracks, targetParam.trackId);
                if (!resolvedRootTrackId || resolvedRootTrackId !== rootTrackId) {
                    return;
                }

                const convertedWrites: Array<{ startFrame: number; values: number[] }> = [];
                for (const write of writes) {
                    const converted = await buildChildOffsetPasteValuesHelper({
                        tracks: s.tracks,
                        rootTrackId,
                        targetTrackId: targetParam.trackId,
                        startFrame: write.startFrame,
                        frameCount: write.values.length,
                        clipboardPitch: write.values,
                        mode: targetParam.mode as "cents" | "degrees",
                        paramsApi,
                        pitchDeltaToDegreeSteps: pitchDeltaToDegreeSteps,
                        projectScale: effectiveProjectScale,
                        // Tempo Map 感知：按帧时刻解析生效音阶。
                        scaleAtFrame: (frame: number) =>
                            projectScaleAtSec((frame * fp) / 1000) ?? effectiveProjectScale,
                    });
                    if (!converted) continue;
                    convertedWrites.push({
                        startFrame: write.startFrame,
                        values: converted.slice(0, write.values.length),
                    });
                }
                if (convertedWrites.length === 0) return;
                // 选区（若为新推导）+ 本地曲线：同一次提交落地，再走后端回写。
                applyPasteLocally(convertedWrites);
                try {
                    await uploadFullResCurveSegments({
                        trackId: rootTrackId,
                        param: editParam,
                        segments: convertedWrites,
                    });
                } catch (err) {
                    console.error("[pianoRoll] paste (converted) failed", err);
                } finally {
                    // 无论成败都重新取数：失败时把上面乐观写入的曲线纠正回后端真值
                    // （与拖拽提交路径同一纪律）。
                    bumpRefreshToken();
                }
                break;
            }

            const pasteWrites = writes.map((write) => ({
                startFrame: write.startFrame,
                values: write.values,
            }));
            // 选区（若为新推导）+ 本地曲线：同一次提交落地，再走后端回写。
            applyPasteLocally(pasteWrites);
            try {
                await uploadFullResCurveSegments({
                    trackId: rootTrackId,
                    param: editParam,
                    segments: pasteWrites,
                });
            } catch (err) {
                console.error("[pianoRoll] paste failed", err);
            } finally {
                // 同拖拽提交路径：失败时用重新取数把乐观写入的曲线纠正回来。
                bumpRefreshToken();
            }
            break;
        }
        case "initialize": {
            await runPerRange(async (range, _index, isFirstWrite) => {
                const restored = await paramsApi.restoreParamFrames(
                    rootTrackId,
                    editParam,
                    range.startFrame,
                    range.frameCount,
                    isFirstWrite,
                );
                return Boolean(restored?.ok);
            });
            bumpRefreshToken();
            break;
        }
        case "average": {
            const strengthPercent = clamp(Number(data?.strength ?? 100) || 0, 0, 100);
            if (strengthPercent <= 0) return;
            // 多选区逐段独立：各段取自己的值、求自己的均值（与 smooth 同构），
            // 整批只打一个撤销点。旧实现只处理 selFrameRanges[0]，多段选区
            // 会静默地只拉平第一段。
            await runPerRange(async (range, _index, isFirstWrite) => {
                const res = await paramsApi.getParamFrames(
                    rootTrackId,
                    editParam,
                    range.startFrame,
                    range.frameCount,
                    1,
                    true,
                    isDynParam(editParam),
                );
                if (!res?.ok) return false;
                const payload = res as ParamFramesPayload;
                const vals = (payload.edit ?? []).map((v) => Number(v) || 0);
                if (vals.length === 0) return false;
                const result = averageSelectionValues(vals, editParam, strengthPercent);
                // dyn：未画帧写回哨兵（防止"沿用原声"被物化成显式目标电平）。
                if (isDynParam(editParam)) {
                    restoreDynSentinels(result, payload.edit_sentinel);
                }
                const written = await paramsApi.setParamFrames(
                    rootTrackId,
                    editParam,
                    range.startFrame,
                    result,
                    isFirstWrite,
                );
                return Boolean(written?.ok);
            });
            bumpRefreshToken();
            break;
        }
        case "transposeCents": {
            const cents = Number(data?.cents ?? 0);
            if (cents === 0) return;
            const delta = cents / 100;
            await runSelectionEdit(
                (vals) =>
                    editParam === "pitch"
                        ? vals.map((v) => (v === 0 ? 0 : v + delta))
                        : vals.map((v) => v + delta),
                { kind: "deltaAt", deltaAt: () => delta },
            );
            break;
        }
        case "transposeDegrees": {
            const degrees = Number(data?.degrees ?? 0);
            const scaleToken = String(data?.scale ?? "__project__");
            // “工程音阶”受 Tempo Map 影响：按每个帧的时刻取生效音阶。
            const fixedScale: ScaleLike | null =
                scaleToken === "__project__" ? null : resolveScaleFromToken(scaleToken);
            const degreeSteps = degreeInputToScaleSteps(degrees);
            if (degreeSteps === 0) return;
            const fpMs = Number(paramView?.framePeriodMs ?? fp) || fp;
            await runSelectionEdit(
                (vals, range) => {
                    return editParam === "pitch"
                        ? vals.map((midi, i) => {
                              if (midi === 0) return 0;
                              const scale =
                                  fixedScale ??
                                  projectScaleAtSec(((range.startFrame + i) * fpMs) / 1000) ??
                                  "C";
                              return transposePitchByScaleSteps(midi, degreeSteps, scale);
                          })
                        : vals.map((midi, i) => {
                              const scale =
                                  fixedScale ??
                                  projectScaleAtSec(((range.startFrame + i) * fpMs) / 1000) ??
                                  "C";
                              return transposePitchByScaleSteps(midi, degreeSteps, scale);
                          });
                },
                // 选区外的延拓 delta：按延拓帧自己的生效音阶逐帧计算
                {
                    kind: "deltaAt",
                    deltaAt: (frame, baseValue) => {
                        const scale = fixedScale ?? projectScaleAtSec((frame * fpMs) / 1000) ?? "C";
                        return (
                            transposePitchByScaleSteps(baseValue, degreeSteps, scale) - baseValue
                        );
                    },
                },
            );
            break;
        }
        case "setPitch": {
            const parsed = Number(data?.value ?? data?.midiNote);
            const midiNote = Number.isFinite(parsed) ? parsed : 60;
            await runSelectionEdit(
                (vals) =>
                    editParam === "pitch"
                        ? vals.map((v) => (v === 0 ? 0 : midiNote))
                        : vals.map(() => midiNote),
                // 选区外延拓 = 目标值本身：向目标的自然滑移
                { kind: "editedAt", editedAt: () => midiNote },
                // 显式写常量：用户意图是覆盖整个选区 → 不保留未画哨兵。
                { preserveDynSentinels: false },
            );
            break;
        }
        case "shiftParamUpSelection":
        case "shiftParamDownSelection": {
            // 长按重复的节拍守卫：上一拍（后端读写仍在途）未完成时
            // 跳过本拍 —— 在途标记经 selectionEditInFlight 与 App
            // 端 fire 共享，两端双重检查避免重复事件堆积。
            if (!beginSelectionParamEdit()) return;
            try {
                const descriptor = processorParamsRef.current.find(
                    (param) => param.id === editParam,
                );
                const magnitude = parseParamShiftMagnitude(data?.magnitude);
                // dyn 是倍率域（0 = 静音）：**档位命令**上下移动用乘性 ——
                // 一次 ±shift = ×2^±magnitude（默认 ×2 / ×0.5 = ±6 dB），
                // 与 DAW 的"增益 ±6 dB"同一语义；0 帧保持 0。选区外延拓同样
                // 按乘性表达：`deltaAt = base × (factor − 1)`，边缘淡化在
                // delta 空间里得到的就是"同一系数作用下的差值"。
                //
                // ⚠ 拖拽**不走**这条法则：拖拽一律是值域内线性偏移，因为
                // 只有线性偏移能让被抓住的那一点始终停在光标下（见
                // `paramRanges.shiftValueForDrag`）。命令是离散的增益档位，
                // 不涉及"跟手"，两者语义不同、不应互相"对齐"。
                if (isDynParam(editParam)) {
                    // 档位（对齐 pitch 的 fine/normal/coarse 节奏）：
                    // fine ≈ +0.6 dB、normal = ×2（+6 dB）、coarse = ×4（+12 dB）。
                    const dynStepDelta =
                        magnitude === "fine" ? 0.05 : magnitude === "coarse" ? 1.0 : 0.5;
                    const factor = dynMultiplicativeFactor(
                        op === "shiftParamUpSelection" ? dynStepDelta : -dynStepDelta,
                    );
                    await runSelectionEdit((vals) => vals.map((v) => v * factor), {
                        kind: "deltaAt",
                        deltaAt: (_f, base) => base * (factor - 1),
                    });
                    break;
                }
                const step = getParamShiftStep(editParam, descriptor, magnitude);
                const delta = op === "shiftParamUpSelection" ? step : -step;
                // 不透传 data?.edgeSmoothnessPercent：键盘路径的事件
                // detail 不携带该值，直接沿用 store 的边缘平滑设置
                // （runSelectionEdit 内部的解析顺序已保证该行为）。
                await runSelectionEdit((vals) => vals.map((v) => v + delta), {
                    kind: "deltaAt",
                    deltaAt: () => delta,
                });
            } finally {
                endSelectionParamEdit();
            }
            break;
        }
        case "smooth": {
            const strength = clamp((Number(data?.strength ?? 50) || 0) / 100, 0, 1);
            if (strength <= 0) return;
            const fpMs = Number(paramView?.framePeriodMs ?? fp) || fp;
            // 多取两侧各 3σ 帧上下文：高斯平滑用真实延拓做边界，
            // 平滑结果与选区外曲线无缝（旧实现只取选区内、边界处
            // 会产生新台阶）。多选区逐段独立取上下文与 σ。
            const pad = smoothContextPadFrames(strength, fpMs);
            await runPerRange(async (range, _index, isFirstWrite) => {
                const ctxStart = Math.max(0, range.startFrame - pad);
                const leftLen = range.startFrame - ctxStart;
                const res = await paramsApi.getParamFrames(
                    rootTrackId,
                    editParam,
                    ctxStart,
                    leftLen + range.frameCount + pad,
                    1,
                    true,
                    isDynParam(editParam),
                );
                if (!res?.ok) return false;
                const payload = res as ParamFramesPayload;
                const all = (payload.edit ?? []).map((v) => Number(v));
                const vals = all.slice(leftLen, leftLen + range.frameCount);
                if (vals.length === 0) return false;
                const result = smoothSelectionValues(vals, editParam, strength, {
                    framePeriodMs: fpMs,
                    leftContext: all.slice(0, leftLen),
                    rightContext: all.slice(leftLen + range.frameCount),
                });
                // dyn：未画帧写回哨兵（防止"沿用原声"被物化成显式目标电平）。
                if (isDynParam(editParam)) {
                    restoreDynSentinels(
                        result,
                        payload.edit_sentinel?.slice(leftLen, leftLen + range.frameCount),
                    );
                }
                const written = await paramsApi.setParamFrames(
                    rootTrackId,
                    editParam,
                    range.startFrame,
                    result,
                    isFirstWrite,
                );
                return Boolean(written?.ok);
            });
            bumpRefreshToken();
            break;
        }
        case "addVibrato": {
            /**
             * 套用颤音预设。
             *
             * 【与旧实现的区别】旧版把整段替换成「起点→终点的直线 + 正弦」，
             * 会抹平歌手原本的音高运动。现在预设默认 `baseline: "existing"`
             * （见系统预设表），在**已有曲线**上叠加颤音，原曲线被保留。
             * 波形、包络、速率渐变、不规则度全部来自预设。
             */
            const rawPreset = data?.preset;
            // 应用弹窗直接给完整预设对象（本地微调的深度 / 速率要传过来）；
            // 菜单 / 快捷键路径仍可只给 presetId，缺失时回落到活动预设。
            const preset: VibratoPreset =
                rawPreset && typeof rawPreset === "object"
                    ? sanitizeVibratoPreset(rawPreset as Partial<VibratoPreset>)
                    : resolveActiveVibratoPreset(
                          resolvedVibratoPresets,
                          typeof data?.presetId === "string"
                              ? data.presetId
                              : activeVibratoPresetId,
                      );
            // 多选区：包络按**每段自身时长**定标，各段独立。
            await runPerRange(async (range, _index, isFirstWrite) => {
                const res = await paramsApi.getParamFrames(
                    rootTrackId,
                    editParam,
                    range.startFrame,
                    range.frameCount,
                    1,
                    true,
                    isDynParam(editParam),
                );
                if (!res?.ok) return false;
                const payload = res as ParamFramesPayload;
                const vals = (payload.edit ?? []).map((v) => Number(v) || 0);
                if (vals.length === 0) return false;
                const fpMs = Number(payload.frame_period_ms ?? fp) || fp;
                /*
                 * 锚点与"哪些帧可调制"一次规划出来（见 `vibratoPitch.ts`）。
                 *
                 * pitch 的 0 是"未检测"哨兵，而浊清边界上还会有**低而非零**的
                 * 过渡帧（跟踪器给的 20~40 Hz 低估）—— 两类都不是音符。拿首末帧
                 * 当锚点时，选区两端只要落在气口上（选一整句几乎必然如此），
                 * 直线摆放就会把整段拉向那个错值，中间真实唱出来的音高被整段
                 * 抹掉。整段都没有音符段时没有可调制的对象，放弃。
                 */
                const plan = planVibratoTarget(editParam, vals, fpMs);
                if (!plan) return false;
                const built = buildVibratoCurve({
                    startFrame: range.startFrame,
                    startValue: plan.anchors.startValue,
                    endFrame: range.startFrame + vals.length - 1,
                    endValue: plan.anchors.endValue,
                    original: vals,
                    preset,
                    param: editParam,
                    framePeriodMs: fpMs,
                    range: currentParamRange,
                    // 摆放方式是**这次操作的参数**（存在设置里），与预设无关：
                    // 换预设不该把它换掉。
                    baseline: vibratoBaseline,
                });
                const result = built.dense;
                // 非音符帧写回哨兵（"不改这一帧"，与 dyn 的哨兵还原同位）。
                suppressNonTargetFrames(result, plan);
                // dyn：未画帧写回哨兵（"沿用原声"不被颤音物化）。
                if (isDynParam(editParam)) {
                    restoreDynSentinels(result, payload.edit_sentinel);
                }
                const written = await paramsApi.setParamFrames(
                    rootTrackId,
                    editParam,
                    range.startFrame,
                    result,
                    isFirstWrite,
                );
                return Boolean(written?.ok);
            });
            bumpRefreshToken();
            break;
        }
        case "quantize": {
            if (editParam !== "pitch") {
                const fallbackUnit = currentParamQuantizeUnit;
                const quantizeUnit = Math.abs(
                    Number(data?.quantizeUnit ?? fallbackUnit) || fallbackUnit,
                );
                if (!Number.isFinite(quantizeUnit) || quantizeUnit <= 0) return;
                const tolerance = Math.abs(
                    Number(data?.tolerance ?? data?.toleranceCents ?? 0) || 0,
                );
                const defaultValue = currentParamDefaultValue;
                // 走统一选区编辑编排（含边缘淡化，平滑度取对话框/全局设置；
                // 强度为 0 时与旧的纯选区写入逐字节一致）。量化 delta 在选区
                // 外的延拓用缺省规则：边界帧实际 delta 常数延拓。
                await runSelectionEdit((vals) =>
                    vals.map((v) => {
                        const stepCount = Math.round((v - defaultValue) / quantizeUnit);
                        const snapped = defaultValue + stepCount * quantizeUnit;
                        if (Math.abs(v - snapped) <= tolerance) return v;
                        return snapped + (v > snapped ? 1 : -1) * tolerance;
                    }),
                );
                break;
            }

            const unit = (data?.unit as string) ?? "semitone";
            const scaleToken = String(data?.scale ?? "__project__");
            // “工程音阶”受 Tempo Map 影响：按每个帧的时刻取生效音阶。
            const fixedScale: ScaleLike | null =
                scaleToken === "__project__" ? null : resolveScaleFromToken(scaleToken);
            const toleranceCents = Math.abs(Math.round(Number(data?.toleranceCents ?? 0) || 0));
            const toleranceSemitone = toleranceCents / 100;
            // project base scale is controlled from toolbar; do not change it here
            const fpMs = Number(paramView?.framePeriodMs ?? fp) || fp;
            const scaleAt = (i: number, range: FrameRange): ScaleLike =>
                fixedScale ?? projectScaleAtSec(((range.startFrame + i) * fpMs) / 1000) ?? "C";
            await runSelectionEdit((vals, range) =>
                unit === "semitone"
                    ? vals.map((v) =>
                          editParam === "pitch" && v === 0
                              ? 0
                              : (() => {
                                    const snapped = snapToSemitone(v);
                                    return Math.abs(v - snapped) <= toleranceSemitone
                                        ? v
                                        : snapped + (v - snapped > 0 ? 1 : -1) * toleranceSemitone;
                                })(),
                      )
                    : vals.map((v, i) =>
                          editParam === "pitch" && v === 0
                              ? 0
                              : (() => {
                                    const snapped = snapToScale(v, scaleAt(i, range));
                                    return Math.abs(v - snapped) <= toleranceSemitone
                                        ? v
                                        : snapped + (v - snapped > 0 ? 1 : -1) * toleranceSemitone;
                                })(),
                      ),
            );
            break;
        }
        case "meanQuantize": {
            if (editParam !== "pitch") {
                const fallbackUnit = currentParamQuantizeUnit;
                const quantizeUnit = Math.abs(
                    Number(data?.quantizeUnit ?? fallbackUnit) || fallbackUnit,
                );
                if (!Number.isFinite(quantizeUnit) || quantizeUnit <= 0) return;
                const tolerance = Math.abs(
                    Number(data?.tolerance ?? data?.toleranceCents ?? 0) || 0,
                );
                const defaultValue = currentParamDefaultValue;
                // 均值量化是整体平移：delta 在 editSelection 内计算并捕获，
                // 选区外按同一 delta 延拓（保持均值量化语义）。
                let valueMeanDelta = 0;
                await runSelectionEdit(
                    (vals) => {
                        if (vals.length === 0) return vals;
                        const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
                        const stepCount = Math.round((avg - defaultValue) / quantizeUnit);
                        const quantizedAvg = defaultValue + stepCount * quantizeUnit;
                        valueMeanDelta = quantizedAvg - avg;
                        return vals.map((v) => {
                            const moved = v + valueMeanDelta;
                            if (Math.abs(moved - v) <= tolerance) return v;
                            return moved + (v > moved ? 1 : -1) * tolerance;
                        });
                    },
                    { kind: "deltaAt", deltaAt: () => valueMeanDelta },
                );
                break;
            }

            const unit = (data?.unit as string) ?? "semitone";
            const scaleToken = String(data?.scale ?? "__project__");
            // “工程音阶”受 Tempo Map 影响：均值吸附使用选区中点时刻的生效音阶，
            // 整体平移量保持统一（均值量化语义）。
            const fixedScale: ScaleLike | null =
                scaleToken === "__project__" ? null : resolveScaleFromToken(scaleToken);
            const toleranceCents = Math.abs(Math.round(Number(data?.toleranceCents ?? 0) || 0));
            const toleranceSemitone = toleranceCents / 100;
            const fpMs = Number(paramView?.framePeriodMs ?? fp) || fp;
            let meanDelta = 0;
            await runSelectionEdit(
                (vals, range) => {
                    // pitch=0 视为未编辑，不参与均值；全部未浊时 delta=0，
                    // 结果与输入逐帧相同（相比旧的直接 return 会多一个
                    // 无变化的撤销点，无副作用，可接受）。
                    const nonZero = vals.filter((v) => v !== 0);
                    if (nonZero.length === 0) return vals.slice();
                    const avg = nonZero.reduce((a, b) => a + b, 0) / nonZero.length;
                    const midScale =
                        fixedScale ??
                        projectScaleAtSec(
                            ((range.startFrame + Math.floor(vals.length / 2)) * fpMs) / 1000,
                        ) ??
                        "C";
                    const quantizedAvg =
                        unit === "semitone" ? snapToSemitone(avg) : snapToScale(avg, midScale);
                    meanDelta = quantizedAvg - avg;
                    return vals.map((v) => {
                        if (v === 0) return 0;
                        const moved = v + meanDelta;
                        return Math.abs(moved - v) <= toleranceSemitone
                            ? v
                            : moved + (v - moved > 0 ? 1 : -1) * toleranceSemitone;
                    });
                },
                { kind: "deltaAt", deltaAt: () => meanDelta },
            );
            break;
        }
    }
}
