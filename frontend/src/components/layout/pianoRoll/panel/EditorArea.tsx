/**
 * 参数编辑器的主编辑面：标尺行 + 卷帘画布 + 自绘滚动条。
 *
 * 【为什么单独成文件】这一段是"画布宿主"——标尺、三层 GL 画布、键盘轴、滚动条的
 * DOM 结构全在这里。它与工具栏、对话框没有任何交集，却和它们挤在同一个函数里。
 *
 * 【ref 全部来自模型】DOM 节点由模型里的 ref 持有（内核通过同一批 ref 绘制），
 * 本文件只负责把它们摆到正确的层级上。
 */
import { Flex, Box } from "@radix-ui/themes";
import { BackgroundGrid, TimeRuler } from "../../timeline";
import { timeRulerHeightPx } from "../../timeline/rulerHeight";
import { TempoMapCornerButton } from "../../timeline/TempoMapCornerButton";
import { AXIS_W, PARAM_EDITOR_BOTTOM_BAR_PX } from "../constants";
import { PianoRollWaveformSurface } from "../PianoRollWaveformSurface";
import { pianoRollViewportBus } from "../pianoRollViewportBus";
import { pointerOverlayStyle } from "../../pointerOverlayStyle";
import { VibratoPresetGlyph } from "../../vibrato/VibratoPresetGlyph";
import { findVibratoPreset } from "../../../../features/vibrato/vibratoPresetList";
import {
    depthForParam,
    depthUnitLabelKey,
    formatNumber,
    vibratoPresetLabel,
} from "../../vibrato/vibratoDialogLogic";
import { PARAM_EDITOR_VERTICAL_SCROLL_RANGE_PX } from "./panelHelpers";

import type { PianoRollPanelModel } from "./usePianoRollPanelModel";

/** 本文件用到的模型字段（AST 扫描得出）；`Pick<>` 保证它不会悄悄变宽。 */
type Props = {
    m: Pick<
        PianoRollPanelModel,
        | "attachHorizontalScrollbarTrack"
        | "attachRulerPlayheadHead"
        | "attachRulerPlayheadLine"
        | "attachVerticalScrollbarTrack"
        | "axisCanvasRef"
        | "axisUnitToggleAvailable"
        | "axisValuePreview"
        | "axisWrapRef"
        | "canvasCursor"
        | "canvasRef"
        | "captureBandsRef"
        | "clipPeaks"
        | "contentWidth"
        | "currentParamRange"
        | "dynamicProjectSec"
        | "editParam"
        | "editParamAxisUnit"
        | "effectiveProjectScale"
        | "formatParamValuePreview"
        | "getScrollLeftLive"
        | "glAxisCanvasRef"
        | "glCanvasRef"
        | "glOverlayCanvasRef"
        | "gridLayerRef"
        | "hScrollbarThumbRef"
        | "handleCopyPlayheadTime"
        | "handlePrimaryUnitChange"
        | "handleRulerWheel"
        | "handleSecondaryUnitChange"
        | "handleTempoMapChange"
        | "handleTempoMapCommit"
        | "interactions"
        | "onScrollerScroll"
        | "paddedContentWidth"
        | "paramValuePreview"
        | "pianoRollAmplitudeMap"
        | "prTickAxis"
        | "pxPerBeat"
        | "pxPerSec"
        | "resolvedVibratoPresets"
        | "rulerContentRef"
        | "s"
        | "scrollLeft"
        | "scrollerRef"
        | "setTimeDisplaySettingsOpen"
        | "t"
        | "tf"
        | "timeContext"
        | "timelineTicks"
        | "vScrollbarThumbRef"
        | "vibratoDragHud"
        | "viewSize"
        | "waveformColors"
    >;
};

export function EditorArea({ m }: Props) {
    const {
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
        captureBandsRef,
        clipPeaks,
        contentWidth,
        currentParamRange,
        dynamicProjectSec,
        editParam,
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
        s,
        scrollLeft,
        scrollerRef,
        setTimeDisplaySettingsOpen,
        t,
        tf,
        timeContext,
        timelineTicks,
        vScrollbarThumbRef,
        vibratoDragHud,
        viewSize,
        waveformColors,
    } = m;

    return (
        <Flex className="flex-1 overflow-hidden relative">
            {/* Left axis + corner */}
            <Flex direction="column" className="shrink-0">
                <Box
                    className="bg-qt-window border-b border-qt-border relative"
                    style={{
                        width: AXIS_W,
                        height: timeRulerHeightPx(
                            Boolean(
                                s.tempoMap && s.tempoMap.points.length > 0 && s.tempoMapVisible,
                            ),
                        ),
                    }}
                >
                    {/* 速度映射小按钮（右下角）：显示/创建 或 清空/隐藏。 */}
                    <TempoMapCornerButton />
                </Box>
                <div
                    ref={axisWrapRef}
                    className="bg-qt-window border-r border-qt-border relative"
                    // 支持切换展示单位的参数（音量 / 动态）整列可点：光标给
                    // pointer 作为"这里可点"的提示，否则该交互完全不可发现。
                    style={{
                        width: AXIS_W,
                        flex: 1,
                        cursor: axisUnitToggleAvailable ? "pointer" : undefined,
                    }}
                >
                    {/* 键盘轴 GL 层（阶段 2/3）：铺在 Canvas2D 轴画布**下面**
                        （DOM 顺序在前、无 z-index），画键盘几何与音名标签。
                        下方那块 Canvas2D 轴画布仍保留在 DOM 中，但绘制已被
                        `skipAxisCanvas: true` **整张跳过（含清屏）**——轴上一切都
                        归 GL。宿主自己按 `axisWidthPx` 给 GL 轴画布定尺寸，不依赖
                        这块画布，故不留它也不影响 GL；保留它只是为了不牵动
                        `drawPianoRoll` 的入参契约（见下方 `skip*` 说明）。 */}
                    <canvas
                        ref={glAxisCanvasRef}
                        className="absolute inset-0 pointer-events-none"
                        aria-hidden
                    />
                    <canvas ref={axisCanvasRef} className="absolute inset-0" />

                    {/* 纵轴展示单位角标（仅音量 / 动态）：显示当前读数单位，同时
                        是"点击可切换"的可见提示（整列光标为 pointer）。
                        刻意**不**挂 onClick —— 点击会冒泡到上面的轴列处理器，
                        再挂一个会切换两次（净效果为"点了没反应"）。 */}
                    {axisUnitToggleAvailable ? (
                        <div
                            className="absolute top-0 right-0 z-10 px-1 text-qt-3xs leading-[14px] text-qt-text-muted"
                            aria-hidden
                        >
                            {editParamAxisUnit === "db"
                                ? tf("param_axis_unit_db")
                                : tf("param_axis_unit_ratio")}
                        </div>
                    ) : null}

                    {/* 纵轴浮动读数（`弹出展示参数` 的轴列形态）：挂在轴列坐标系里，
                        底边对齐光标向上展开——与曲线浮窗同一观感。 */}
                    {s.showParamValuePopup && axisValuePreview
                        ? (() => {
                              const rect = axisWrapRef.current?.getBoundingClientRect();
                              if (!rect) return null;
                              return (
                                  <div
                                      className="absolute z-20 pointer-events-none bg-qt-panel border border-qt-border rounded px-2 py-1 text-qt-xs leading-none text-qt-text"
                                      style={{
                                          // 位移走合成层 transform：`left/top` 上的小数坐标会让
                                          // 文字逐帧重栅格化而抖动（见 `pointerOverlayStyle`）。
                                          ...pointerOverlayStyle(axisValuePreview, rect),
                                          whiteSpace: "nowrap",
                                      }}
                                  >
                                      {axisValuePreview.text}
                                  </div>
                              );
                          })()
                        : null}
                </div>
            </Flex>

            {/* Right: ruler + scrollable canvas */}
            <Flex direction="column" className="flex-1 min-w-0 select-none">
                {/* 标尺 + 捕获区间指示。
                    【为什么包一层 relative】捕获带要贴在标尺下缘、并随横向滚动平移，
                    而 `TimeRuler` 本身是共享组件（时间轴也用它）—— 在它内部加一层
                    只服务参数编辑器的覆盖层会把两个面板的职责搅在一起。
                    包一层之后，覆盖层是这个容器的绝对定位子元素，TimeRuler 一行不改。 */}
                <div className="relative">
                    <TimeRuler
                        scrollLeft={scrollLeft}
                        scrollLeftLive={getScrollLeftLive}
                        tickWindowAnchorPx={prTickAxis.anchorPx}
                        ticks={timelineTicks}
                        pxPerSec={pxPerSec}
                        viewportWidth={viewSize.w}
                        playheadSec={s.playheadSec}
                        positionPlayheadFromProps={false}
                        playheadLineRef={attachRulerPlayheadLine}
                        playheadHeadRef={attachRulerPlayheadHead}
                        contentRef={rulerContentRef}
                        timeContext={timeContext}
                        primaryUnit={s.primaryTimeUnit}
                        secondaryUnit={s.secondaryTimeUnit}
                        onPrimaryUnitChange={handlePrimaryUnitChange}
                        onSecondaryUnitChange={handleSecondaryUnitChange}
                        onOpenSettings={() => setTimeDisplaySettingsOpen(true)}
                        onCopyPlayheadTime={() => void handleCopyPlayheadTime()}
                        t={tf}
                        tempoMap={s.tempoMap}
                        tempoMapVisible={s.tempoMapVisible}
                        projectSec={dynamicProjectSec}
                        grid={s.grid}
                        snapEnabled={s.snapEnabled}
                        timelineSnap={s.timelineSnap}
                        projectScale={effectiveProjectScale}
                        projectScaleName={
                            s.project.useCustomScale
                                ? (s.project.customScale?.name ?? undefined)
                                : undefined
                        }
                        fallbackDenominator={s.project.timeSignatureDenominator}
                        customScalePresets={s.customScalePresets}
                        onTempoMapChange={handleTempoMapChange}
                        onTempoMapCommit={handleTempoMapCommit}
                        subscribeViewport={pianoRollViewportBus.subscribe}
                        onRulerWheel={handleRulerWheel}
                        onMouseDown={(e) => {
                            interactions.onRulerMouseDown(e);
                        }}
                    />
                    {/* 捕获区间带：几何由**内核**在帧提交里写（与标尺平移同源同帧），
                        这里只留一个空容器。此前是一个 React 组件、读 state 的量化
                        scrollLeft —— 滚动时带子几乎不动，开同步时还差一个偏移量。
                        【为什么是容器】捕获可以是**多段**（用户跳播再继续录），
                        内核按段数池化增删子元素，见宿主里的 `syncCaptureBands`。 */}
                    <div
                        ref={captureBandsRef}
                        data-capture-bands
                        aria-hidden
                        style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
                    />
                </div>

                {/* 自绘滚动条的定位容器：滚动条必须是**滚动容器之外**
                    的兄弟节点。放在滚动容器内部会被内容一起滚走（绝对定位在滚动
                    容器里仍随内容平移），这是自绘滚动条最经典的错位根因。 */}
                <div className="flex-1 min-w-0 relative">
                    <div
                        ref={scrollerRef}
                        // 隐藏原生滚动条（自绘条取代），但**保留 `overflow: scroll`**
                        // ——原生 scroller 是被动镜像，必须保留滚动范围才能接受宿主
                        // 每帧的程序化回写，也让尚未迁移的输入代码（中键平移等）
                        // 继续可读可写。`.custom-scrollbar` 不再需要：原生条恒不可见。
                        //
                        // 底部预留 8px（bottom-2）：给自绘水平滚动条独占一行。滚动
                        // 条若叠加在内容上，会挡住贴底的参数线；让 scroller 在水平
                        // 条上方收边，二者互不重叠（竖直条同步缩短，见下方轨道）。
                        className="absolute left-0 right-0 top-0 bg-qt-graph-bg overflow-x-scroll overflow-y-scroll hide-scrollbar outline-hidden focus:outline-hidden focus-visible:outline-hidden"
                        // 底部预留一行给自绘水平滚动条（见 PARAM_EDITOR_BOTTOM_BAR_PX）
                        style={{ bottom: PARAM_EDITOR_BOTTOM_BAR_PX }}
                        data-piano-roll-scroller
                        tabIndex={0}
                        onAuxClick={interactions.onScrollerAuxClick}
                        onScroll={onScrollerScroll}
                        onContextMenu={interactions.onScrollerContextMenu}
                        onKeyDown={interactions.onScrollerKeyDown}
                    >
                        {/* Sticky viewport overlay: grid + canvas do not physically scroll */}
                        <div
                            className="sticky left-0 top-0 h-full"
                            style={{ width: viewSize.w, overflow: "hidden", zIndex: 1 }}
                        >
                            <div className="relative h-full" style={{ width: viewSize.w }}>
                                <BackgroundGrid
                                    contentWidth={contentWidth}
                                    contentHeight={viewSize.h}
                                    viewportWidth={viewSize.w}
                                    scrollLeft={scrollLeft}
                                    pxPerBeat={pxPerBeat}
                                    grid={s.grid}
                                    beatsPerBar={Math.max(1, Math.round(s.beats || 4))}
                                    visible={s.timelineSnap.gridVisible}
                                    minSpacingPx={s.timelineSnap.gridMinSpacingPx}
                                    swingPercent={
                                        s.timelineSnap.swingEnabled
                                            ? s.timelineSnap.swingPercent
                                            : 0
                                    }
                                    layerRef={gridLayerRef}
                                    ticks={timelineTicks}
                                    // 【必须提供视口总线：否则网格会画在滞后的偏移上】
                                    // 不传总线时网格的绘制偏移取自 `scrollLeft` prop，而它是
                                    // **量化提交**的 React state（256px 死区，见
                                    // `gridDrawViewport.ts`）：参数编辑器自己的滚动（滚轮 /
                                    // 拖 thumb / 触摸）只改内核与镜像，**不**逐帧提交 state
                                    // ——最后一次不足 256px 的位移永远不会提交。此后任何一次
                                    // React 重绘（例如时间轴缩放带来的 `pxPerBeat` 变化）都会
                                    // 用这个滞后值画网格，网格就停在错误偏移上，直到下一次滚动。
                                    // 实测（Chrome，先小幅滚动参数编辑器再滚轮缩放时间轴）：
                                    // 网格与自身标尺相差 **59px**（基线只有 8px 的标签内缩）。
                                    // 传总线后偏移一律取总线快照（内核真值），与标尺 / 画布 /
                                    // 波形 / 曲线取同一份视口。
                                    //
                                    // 不传 `layerOrder`：参数编辑器没有统一帧提交器，注册会被
                                    // 跳过；网格仍由 `gridRedrawBridge` 的命令式路径重绘（宿主
                                    // 每帧调用），这里只需要「偏移取总线」这一条契约。
                                    viewportBus={pianoRollViewportBus}
                                    sticky
                                />

                                <PianoRollWaveformSurface
                                    clips={clipPeaks}
                                    widthPx={viewSize.w}
                                    heightPx={viewSize.h}
                                    scrollLeftPx={scrollLeft}
                                    pxPerSec={pxPerSec}
                                    colors={waveformColors}
                                    amplitudeMap={pianoRollAmplitudeMap}
                                />

                                {/* GL 静态层（阶段 2）：网格等静态图层。
                                    层级说明：它是**最底层**——DOM 顺序在 Canvas2D
                                    主画布之前，且不设 z-index（Canvas2D 主画布用
                                    `absolute inset-0` 覆盖其上）。因此 GL 层只画
                                    静态底图，曲线 / 选区 / 播放头由 Canvas2D 与 GL
                                    叠加层画在上层。
                                    恒挂载：GL 是网格的**唯一**绘制者（Canvas2D
                                    侧的整段音高网格分支已被删除，不是"跳过"而是
                                    不存在了），不挂就等于网格消失。 */}
                                <canvas
                                    ref={glCanvasRef}
                                    data-piano-roll-gl-scene
                                    className="absolute inset-0 pointer-events-none"
                                    aria-hidden
                                />

                                {/* `data-piano-roll-canvas`：主曲线画布的稳定选择器。
                                    浏览器自动化验证（scripts/dev-shot.mjs）需要按元素
                                    截图逐像素比对曲线，而 canvas 本身没有可锚定的属性；
                                    页面里有多个同尺寸 canvas，按尺寸猜会命中错误的那一个。 */}
                                <canvas
                                    ref={canvasRef}
                                    data-piano-roll-canvas
                                    className="absolute inset-0"
                                    style={{
                                        cursor: canvasCursor,
                                        // 阻止 WebView 把笔/触摸手势截走做原生
                                        // 滚动（会产生 pointercancel 打断笔画）。
                                        // scroller 自身不加：触摸屏单指滚动依赖原生。
                                        touchAction: "none",
                                    }}
                                    onPointerMove={interactions.onCanvasPointerMove}
                                    onPointerLeave={interactions.onCanvasPointerLeave}
                                    onPointerDown={interactions.onCanvasPointerDown}
                                />

                                {/* 动态叠加层（阶段 2/3）：播放头与选区。
                                    层序：DOM 顺序在曲线画布**之后** => 覆盖其上。
                                    指针事件全部穿透（interactions 挂在曲线画布上），
                                    否则会挡住参数编辑的命中测试。
                                    恒挂载：播放头归 GL 叠加层（Canvas2D 侧
                                    `skipPlayhead: true`）。 */}
                                <canvas
                                    ref={glOverlayCanvasRef}
                                    className="absolute inset-0 pointer-events-none"
                                    aria-hidden
                                />
                                {/* 指针旁的实时读数：**参数值 + 颤音拖拽状态合成一个
                                    气泡**。两者曾经各弹一个，颤音那个还压在值上面，
                                    用户只能看到后弹出的那个 —— 合成之后第一行是参数值
                                    （"这一笔落在什么值上"），其下才是预设 / 深度 / 速率。
                                    共用「弹出展示参数」开关。 */}
                                {s.showParamValuePopup &&
                                    (paramValuePreview || vibratoDragHud) &&
                                    (() => {
                                        const rect = canvasRef.current?.getBoundingClientRect();
                                        if (!rect) return null;
                                        // 位置以值读数为准（它本就贴着指针）；只有 HUD
                                        // 时（例如不产生值读数的路径）用 HUD 的位置。
                                        const anchor = paramValuePreview ?? vibratoDragHud;
                                        if (!anchor) return null;
                                        const hudPreset = vibratoDragHud
                                            ? findVibratoPreset(
                                                  resolvedVibratoPresets,
                                                  vibratoDragHud.presetId,
                                              )
                                            : undefined;
                                        const depthUnitKey = depthUnitLabelKey(editParam);
                                        return (
                                            <div
                                                className="absolute z-20 pointer-events-none bg-qt-panel border border-qt-border rounded px-2 py-1 text-qt-xs leading-none text-qt-text"
                                                style={{
                                                    // 位移走合成层 transform：`clientX/Y` 在 HiDPI 屏上
                                                    // 带小数，直接当 `left/top` 会让气泡里的文字每帧换一个
                                                    // 次像素相位、被反复重栅格化 —— 看起来就是"文字在抖"
                                                    // （见 `pointerOverlayStyle`）。
                                                    ...pointerOverlayStyle(anchor, rect),
                                                    whiteSpace: "nowrap",
                                                }}
                                            >
                                                {paramValuePreview ? (
                                                    <div>
                                                        {paramValuePreview.displayText ??
                                                            formatParamValuePreview(
                                                                paramValuePreview.value,
                                                            )}
                                                    </div>
                                                ) : null}
                                                {vibratoDragHud ? (
                                                    <div
                                                        className={
                                                            paramValuePreview ? "mt-1" : undefined
                                                        }
                                                    >
                                                        <div className="flex items-center gap-1.5">
                                                            {hudPreset ? (
                                                                <VibratoPresetGlyph
                                                                    preset={hudPreset}
                                                                    width={20}
                                                                    height={10}
                                                                />
                                                            ) : null}
                                                            <span>
                                                                {hudPreset
                                                                    ? vibratoPresetLabel(
                                                                          hudPreset,
                                                                          t,
                                                                      )
                                                                    : ""}
                                                            </span>
                                                            {vibratoDragHud.adjusted
                                                                ? ` · ${t("vibrato_adjusted")}`
                                                                : ""}
                                                        </div>
                                                        <div className="text-qt-text-muted tabular-nums">
                                                            {`${formatNumber(depthForParam(vibratoDragHud.depthCents, editParam, currentParamRange))}${depthUnitKey ? ` ${t(depthUnitKey)}` : ""} · ${formatNumber(vibratoDragHud.rateHz)} ${t("vibrato_unit_hz")}`}
                                                        </div>
                                                    </div>
                                                ) : null}
                                            </div>
                                        );
                                    })()}
                            </div>
                        </div>

                        {/* Spacer：提供横向内容宽度与竖向滚动范围，实际绘制仍固定在 sticky 视口层。 */}
                        <div
                            className="relative"
                            style={{
                                width: paddedContentWidth,
                                height: PARAM_EDITOR_VERTICAL_SCROLL_RANGE_PX,
                                pointerEvents: "none",
                            }}
                            aria-hidden
                        />
                    </div>

                    {/* 自绘滚动条：几何由宿主每帧写入。
                        样式对齐原生滚动条（`.custom-scrollbar`，本面板已不再使用）：
                        - thumb 取 `--qt-scrollbar-thumb`（浅色主题下才看得见），
                          而不是固定半透明黑；
                        - 轨道**透明**，加底色会多出一条灰带；
                        - 8px 厚 + 胶囊圆角，对应 macOS 的 overlay thin 滚动条。
                        - 水平条**独占一行**：scroller 已在 bottom-2 收边（见上），
                          轨道落在预留行内，不再叠加内容；竖直条同步在 bottom-2
                          收边，右下角让位给水平条（与原生滚动条的角落行为一致）。
                        - 外层即**轨道**：承接「点空白翻页」。宿主的 thumb 处理器
                          会 `stopPropagation`，因此到达轨道的按下必然不在 thumb 上。
                        恒挂载：原生滚动条已被 `.hide-scrollbar` 隐藏，自绘条是
                        用户可见的**唯一**滚动条。 */}
                    {/* `data-hs-scrollbar` + 光标类：与时间轴自绘滚动条同一
                        约定（见 TimelineKernelView 的 JSX 说明）——轨道区域
                        的光标由自身声明，不继承内容手势。 */}
                    <div
                        ref={attachVerticalScrollbarTrack}
                        data-hs-scrollbar="1"
                        className="absolute right-0 top-0 w-2 z-20 cursor-default"
                        style={{ bottom: PARAM_EDITOR_BOTTOM_BAR_PX }}
                    >
                        <div
                            ref={vScrollbarThumbRef}
                            data-hs-scrollbar="1"
                            className="absolute left-0 w-full cursor-grab rounded-full bg-(--qt-scrollbar-thumb)"
                        />
                    </div>
                    <div
                        ref={attachHorizontalScrollbarTrack}
                        data-hs-scrollbar="1"
                        className="absolute bottom-0 left-0 right-0 z-20 cursor-default"
                        style={{ height: PARAM_EDITOR_BOTTOM_BAR_PX }}
                    >
                        <div
                            ref={hScrollbarThumbRef}
                            data-hs-scrollbar="1"
                            className="absolute top-0 h-full cursor-grab rounded-full bg-(--qt-scrollbar-thumb)"
                        />
                    </div>
                </div>
            </Flex>
        </Flex>
    );
}
