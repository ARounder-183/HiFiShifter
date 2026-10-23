/**
 * 参数编辑器主画布的**一帧**：把当前视口、曲线、选区、剪贴板预览交给渲染内核。
 *
 * 【为什么单独成文件】它是 `drawRef.current` 的函数体，每帧被内核的 rAF 调一次。
 * 它只读捕获值、写画布、不 setState —— 是"绘制"这件事的完整定义。留在模型里时，
 * 读它要先翻过命令分发与滚动同步。
 *
 * 【依赖怎么拿】捕获值由模型打包传入，类型 `Pick<ModelDeps, …>`。函数体原样搬过来，
 * 开头解构回局部变量。
 *
 * 【边界】只画一帧；调度（何时画、脏标记、滚动对齐）仍在 `pianoRollKernelHost`。
 */
import { drawPianoRoll } from "../render";
import type { MainCanvasSignature } from "../mainCanvasSignature";
import { isDynParam } from "../paramRanges";
import { createTimelineAxis } from "../../renderKernel/timelineAxis.js";
import { getCaptureBands } from "../../../../features/capture/captureRange";
import { resolvePanelRenderViewport } from "../kernel/viewportSource";
import { resolvePlayheadRgba } from "./panelHelpers";
import type { ModelDeps } from "./modelDeps";

/** 画一帧所需的全部捕获值（字段清单由 AST 扫描得出）。 */
type DrawFrameDeps = Pick<
    ModelDeps,
    | "axisCanvasRef"
    | "buildCurveLayers"
    | "buildSelectionBandSpec"
    | "canvasRef"
    | "childPitchHardDisableReason"
    | "clipboardRef"
    | "detectedPitchCurves"
    | "editParam"
    | "fontFamily"
    | "hostRef"
    | "kernelDataRef"
    | "liveEditOverrideRef"
    | "loudnessAnalysisPending"
    | "paramMorphOverlays"
    | "paramView"
    | "paramViewsRef"
    | "pitchEnabled"
    | "pitchHardDisableReason"
    | "pitchViewRef"
    | "pxPerSecRef"
    | "referencePitchOverlays"
    | "s"
    | "scrollLeftRef"
    | "secondaryParamViews"
    | "selectionRef"
    | "t"
    | "themeMode"
    | "valueToY"
    | "viewSize"
    | "viewSizeRef"
    | "visibleSecondaryParamIds"
    | "visualPlayheadSecRef"
>;

/** 画一帧。内核每帧调用一次；只读捕获值、只写画布。 */
export function drawPianoRollFrame(deps: DrawFrameDeps): void {
    const {
        axisCanvasRef,
        buildCurveLayers,
        buildSelectionBandSpec,
        canvasRef,
        childPitchHardDisableReason,
        clipboardRef,
        detectedPitchCurves,
        editParam,
        fontFamily,
        hostRef,
        kernelDataRef,
        liveEditOverrideRef,
        loudnessAnalysisPending,
        paramMorphOverlays,
        paramView,
        paramViewsRef,
        pitchEnabled,
        pitchHardDisableReason,
        pitchViewRef,
        pxPerSecRef,
        referencePitchOverlays,
        s,
        scrollLeftRef,
        secondaryParamViews,
        selectionRef,
        t,
        themeMode,
        valueToY,
        viewSize,
        viewSizeRef,
        visibleSecondaryParamIds,
        visualPlayheadSecRef,
    } = deps;

    // 滚动热路径的投影：**必须以内核视口为准**（`resolvePanelRenderViewport`）。
    //
    // 【为什么不能用渲染期的 refs】`pxPerSecRef` / `scrollLeftRef` 在渲染期就被同步成
    // React state 的新值，而并发渲染允许"渲染但尚未提交"（被更高优先级更新打断并丢弃）
    // ——那一瞬间 ref 已是新值、内核与 DOM 还是旧值。用 refs 当投影源会让面板的
    // Canvas2D 与曲线**可见段选择**落在新视口、宿主 GL（网格 / 曲线 / 播放头）与 DOM
    // 落在旧视口：同一屏两套视口，表现为"这些线偏移了"。时间轴的 `livePxPerSec` 是同一
    // 约定的先例（逐帧发布一律取内核真值）。
    const kernelView = hostRef.current?.getViewport() ?? null;
    const viewport = resolvePanelRenderViewport({
        kernelView,
        refPxPerSec: pxPerSecRef.current,
        refScrollLeftPx: scrollLeftRef.current,
    });
    const drawAxis = createTimelineAxis({
        pxPerSec: viewport.pxPerSec,
        scrollLeftPx: viewport.scrollLeftPx,
        viewportWidthPx: viewSizeRef.current.w,
        dpr: window.devicePixelRatio || 1,
    });
    // 曲线图层（阶段 3）：每帧重建描述符列表。
    //
    // 【为什么要每帧构建】滚动/缩放会改变可见段与 `axis`，而 GL 侧要在绘制时
    // 才投影；描述符里的 `values` 引用与视口无关（数据没变），因此这里的成本
    // 只是一次浅层数组构建，不复制采样值。
    //
    // 【必须是唯一来源】曲线已归 GL（Canvas2D 侧 `skipCurves` 恒为 true），
    // 置 null 等于曲线无人绘制。
    kernelDataRef.current.curves = buildCurveLayers(drawAxis);

    // 叠加层镜像（播放头）必须**每帧**更新：播放头用的是插值的
    // 视觉值 `visualPlayheadSecRef`，它不由 React 渲染驱动（见下方注释），
    // 因此不能在 render 期写入镜像——那样播放头会停在旧的提交值上。
    // 选区块不在这里，而是走 `selectionBand` 字段（它是曲线**之下**的一层，
    // 由 GL 场景层在曲线之前绘制，见 `PianoRollSelectionBandSpec` 的层序说明）。
    //
    // 【颜色必须显式喂进去，不能靠宿主兜底】宿主在 `playheadRgba` 缺省时用
    // 一个硬编码兜底色（`[0,0,0,0.2]`），而参数编辑器的播放头在 Canvas2D 侧的
    // 绘制已被 `skipPlayhead: true` 永久跳过——也就是说 GL 这里是**唯一**绘制者，
    // 不喂颜色等于画布播放头与上方标尺（`--qt-playhead`）各是一套颜色。
    // 用户报告："播放线在底下和上方标尺的颜色不一致"。
    //
    // 取值经 `normalizeCssColor` 归一化：GL 不认 `var(...)`，必须借浏览器把
    // CSS 变量解析成 `rgb()/rgba()` 再交给 `parseRgbaColor`。归一化会触发样式
    // 重算，但本块是每帧执行的——因此按解析后的字符串做一次模块级缓存，
    // 主题切换时字符串本身会变，缓存自然失效（键就是字符串）。
    kernelDataRef.current.overlay = {
        playheadSec: visualPlayheadSecRef.current,
        playheadRgba: resolvePlayheadRgba(themeMode),
        // 捕获区间：读**非订阅**快照。订阅会随录制每秒重渲染十几次整个面板，
        // 而带子只需要内核在帧提交里读到最新值（见下方的 subscribeCapture）。
        // 【为什么取 .bands】捕获可能是多段（跳播再继续录），并集会把中间没录到的
        // 部分也画成"已录"。见 `features/capture/captureRange`。
        captureBands: getCaptureBands()?.bands ?? null,
    };
    // 选区块镜像（每帧刷新）：选区是**拍**为单位，宿主只认秒，因此这里先换算。
    // 空选区时喂 null，宿主据此清空本帧的实例（否则旧带子会留在画布上）。
    kernelDataRef.current.selectionBand = buildSelectionBandSpec(selectionRef.current);
    /**
     * 主画布上绘制的中央提示文字（"音高被硬禁用"的原因）。
     *
     * 【为什么提取成局部变量】它同时是**签名项**与 `drawPianoRoll` 的入参。
     * 就地写两遍表达式（此前签名里根本没有它）正是本文件反复警告的
     * "签名里写了 A、实际喂给绘制的是 B"的漂移来源：一旦两处写法分叉，
     * 文字就会停在旧值或漏绘。提取一次、两处共用，漂移在结构上不可能发生。
     *
     * 【为什么非 pitch 参数用 childPitchHardDisableReason】子音高偏移参数
     * （cents / degrees / formant）走的是 `childPitchHardDisableReason`，
     * 与 `pitchEnabled` 的判定分支保持一一对应（见上方 `pitchEnabled`）。
     *
     * 【动态基线分析中】dyn 面板的虚线基线依赖后台电平分析；未就绪时基线是
     * 静默的 1.0 平线，用户无从知道"稍后会变"。就绪事件（dyn_orig_updated）
     * 会刷新快照并清掉本提示（useLoudnessCurves 的 analysisPending）。
     */
    const overlayText = !pitchEnabled
        ? editParam === "pitch"
            ? pitchHardDisableReason
            : childPitchHardDisableReason
        : isDynParam(editParam) && loudnessAnalysisPending
          ? t("dyn_analysis_pending")
          : null;

    /**
     * 主画布的内容签名（阶段 2 Task 6）。
     *
     * 【必须包含什么】主画布上绘制的**全部输入**：
     * - 绘图资源：各条曲线数据、参考线、检测曲线、副参数视口、morph 叠加、
     *   剪贴板预览、选区块、live 编辑覆盖、中央提示文字 `overlayText`；
     * - 视口：`viewSize`、`pxPerSec`、`scrollLeft`、`dpr`（滚动/缩放会改变投影）；
     * - 主题与字体（颜色解析与文字宽度都会影响像素结果）。
     *
     * 【音阶高亮为什么不在签名里】它已迁到 **GL 网格层**（主画布不再绘制它），
     * 其输入（音级集合 / 强调线颜色）由 `PianoRollGridSpec` 的几何签名负责。
     * 此前这里编入的 `effectiveProjectScale` / `s.tempoMap` / `s.pitchSnapUnit` /
     * `s.scaleHighlightMode` / `s.toolMode` / `snapToggleHeld` 都随该分支的删除
     * 一并移除——它们对应的绘制入参已不存在，"编进来"只会让缓存无谓失效。
     *
     * 【必须**不**包含什么】播放头位置——它已由 GL 叠加层绘制。把它编进签名会让
     * 播放帧的签名每帧变化、缓存失效，那就退回"每帧重绘曲线"。
     *
     * 【比较语义】这是一个**数组**，交给 `isSameMainCanvasSignature` 逐项
     * `Object.is` 比较：**对象 / 数组项按引用参与**（Redux 只在内容变化时换引用，
     * 比引用既快又准），原始值按数值。**绝不拼接字符串**——此前后缀是
     * `.join("|")`，而 `join` 会把每个对象 / 数组元素串成字面量
     * `"[object Object]"`，两个内容完全不同的选区因此得到同一个签名，缓存命中、
     * 主画布在清屏前就 return，旧选区框留在画布上不消失（缺陷 #4，详见
     * `pianoRoll/mainCanvasSignature.ts`）。引用比较之所以能逐帧失效，是因为
     * 选区在变化时被赋**新对象**；绘制中的 live 覆盖改成了原地更新，故它
     * 不参与引用比较，而是以**显式版本号**参与（见下方签名项）。
     *
     * 【必须与绘制入参一一对应】下面每一项都刻意对应 `drawPianoRoll` 的某个
     * 入参（或影响其投影的视口量），避免"签名里写了 A、实际喂给绘制的是 B"。
     * 漏项的代价是"该图层不再更新"，因此这里**宁可多编**：低频变化的项一并纳入，
     * 成本只是偶尔多一次重绘。最后再列一遍 `secondaryParamViews` 并非冗余——
     * 它与 `paramViewsRef.current` 同为曲线数据源，两者都必须进签名。
     */
    const mainContentSignature: MainCanvasSignature = [
        viewSize.w,
        viewSize.h,
        pxPerSecRef.current,
        scrollLeftRef.current,
        Math.round((window.devicePixelRatio || 1) * 100),
        editParam,
        themeMode,
        fontFamily,
        pitchEnabled ? 1 : 0,
        // 帧 → sec 的换算系数：选区框与剪贴板预览的 x 由它投影。它与 BPM
        // **无关**（工程级常量栅格），因此改 BPM 不再需要重绘主画布；签名里
        // 保留它是为了"帧周期一旦变化，选区几何必须跟着失效"这条契约。
        paramView?.framePeriodMs ?? 5,
        // 数据与几何（按引用比较）。刻意与传给 drawPianoRoll 的字段一一对应，
        // 避免"签名里写了 A、实际喂给绘制的是 B"这种漂移。
        detectedPitchCurves,
        referencePitchOverlays,
        secondaryParamViews,
        visibleSecondaryParamIds,
        paramMorphOverlays,
        s.showClipboardPreview ? clipboardRef.current : null,
        selectionRef.current,
        // 绘制中的 live 覆盖：用**显式版本号**而非对象引用参与签名。
        // 覆盖层改为原地更新后引用不再变化，比引用即可正确失效（且"无覆盖"
        // → 0 与"新覆盖"→ 非 0 天然可分）。见 mainCanvasSignature 的约束说明。
        liveEditOverrideRef.current?.version ?? 0,
        // 中央提示文字（"音高被硬禁用"的原因）：本面板**新增**的字符串签名项，
        // 切换参数 / 轨道组时会变（禁用原因出现或消失）。
        // 注意 `drawPianoRoll` 另有多项字符串入参（editParam / fontFamily 等），
        // 它们在上方各自的位置参与签名——本项并非"唯一"字符串项。
        overlayText,
        // 视口中心/跨度（用 ref 值，避免依赖 React 渲染时机）
        pitchViewRef.current.center,
        pitchViewRef.current.span,
        paramViewsRef.current,
        secondaryParamViews,
    ];

    drawPianoRoll({
        axisCanvas: axisCanvasRef.current,
        canvas: canvasRef.current,
        viewSize: viewSizeRef.current,
        editParam,
        pitchView: pitchViewRef.current,
        paramViews: paramViewsRef.current,
        valueToY,
        paramView: pitchEnabled ? paramView : null,
        secondaryParamViews: pitchEnabled ? secondaryParamViews : {},
        secondaryParamIds: pitchEnabled ? visibleSecondaryParamIds : [],
        showSecondaryParam: pitchEnabled && visibleSecondaryParamIds.length > 0,
        // 与上面签名里的 `overlayText` 是**同一个变量**（不重复写表达式）。
        overlayText,
        liveEditOverride: liveEditOverrideRef.current,
        selection: selectionRef.current,
        axis: drawAxis,
        framePeriodMs: paramView?.framePeriodMs ?? 5,
        // 画布每帧重绘（onFrame invalidate），播放头必须用插值的视觉值：
        // 用 Redux 提交值会让 60fps 的重绘画着同一个旧播放头（且与标尺
        // 的 DOM 插值播放头节奏不一致、短暂错位）。
        playheadSec: visualPlayheadSecRef.current,
        referencePitchOverlays,
        detectedPitchCurves,
        isDark: themeMode === "dark",
        // ── 下列五个 `skip*` 恒为 true：这些图层**只有 GL 一个绘制者** ──────
        //
        // 内核是唯一渲染路径，键盘几何 / 轴文字 / 轴画布 / 播放头 / 曲线
        // 全归 GL。恒 true 意味着 Canvas2D 侧**永久跳过**这些图层。
        //
        // 【已知并接受的限制】WebGL2 在**运行期**失败时，这些图层无人绘制（实测：
        // 贯穿宽度的横向网格线 24 → 0 条，音高网格整片消失）。参数编辑器在无
        // WebGL2 的环境下本就无法工作，补一套 Canvas2D 回退等于把已迁走的渲染层
        // 再实现一遍，收益与成本完全不成比例——用户已明确决定不修。详见设计文档
        // `docs/superpowers/specs/2026-09-13-timeline-single-path-design.md` §2.4。
        //
        // 特殊说明 1：跳过的都是 **GL 已完全接管**的图层；选区块、morph 手柄等仍在
        // 下面由主画布绘制（见 render.ts 对 skipPlayhead / skipCurves 的说明）。
        //
        // 特殊说明 2：**没有 `skipGrid`**——网格的 Canvas2D 分支已被整体删除
        // （它原本就因本处恒传 `true` 而不可达）。音阶高亮随网格一起迁到 GL。
        skipKeyboardGeometry: true,
        skipAxisText: true,
        // 轴画布全部内容归 GL -> 整张跳过（含清屏）。
        skipAxisCanvas: true,
        skipPlayhead: true,
        // 曲线归 GL（阶段 3）。morph 手柄不在曲线层内，仍由主画布绘制。
        skipCurves: true,
        // 主画布内容缓存（Task 6）：签名只含**主画布自己绘制的内容**与视口，
        // 不含播放头（它已归 GL 叠加层）——这正是播放帧能跳过曲线重绘的原因。
        mainContentSignature,
        fontFamily,
        clipboardPreview: s.showClipboardPreview ? clipboardRef.current : null,
        // 形变控制线：每段选区一条（选区已统一为多区间 `ParamSelection`）。
        // 【合并说明】feature/tools 分支此处另行传过 `pitchSnapUnit` /
        // `projectScale` / `scaleHighlightMode` / `scaleSegments` / `toolMode` /
        // `snapToggleHeld`——这些入参在本分支已随「Canvas2D 音高网格分支整体
        // 删除」一并移除（网格与音阶高亮恒由 GL 绘制，见 `render.ts` 的说明），
        // 继续传只会被 `drawPianoRoll` 静默忽略。
        paramMorphOverlays,
    });
}
