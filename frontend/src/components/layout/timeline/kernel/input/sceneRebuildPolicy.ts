/**
 * 时间轴内核 · 场景（GPU 几何）重建判定
 *
 * 【主要内容】`shouldRebuildScene()`：把"本帧要不要重建网格 / 行分界线 / clip 实例"
 * 的全部条件收敛成一个纯函数。
 *
 * 【作用：为什么必须抽出来】宿主的绘制路径（WebGL2 + 容器量测）在 node 环境的 vitest
 * 里构造不出来，而本判定里**恰好藏着一条曾经出错的承重不变式**：
 *
 * > 判据里的行高必须与**几何构建所用**的行高同源。
 *
 * 几何是用内核行高（`ScrollKernel` 的 `view.rowHeight`）构建的，但历史上这里比较的是
 * **React 镜像行高**（`data().rowHeight`）。两者在内核先行的那一帧不同：内核当帧就
 * 改了行高，镜像要等下一次提交。于是该帧判为"不用重建"，`draw()` 便用**旧行高的几何**
 * 配上**新的视口偏移**重绘一帧 —— 用户看到的就是"竖直缩放之后竖直方向抽动一下"
 * （以指针为锚时，指针处的行号被算成 `anchorRowUnit × 新/旧`，偏差随指针行号线性增大）。
 * 抽出来单测，才能把"判据不含 React 镜像"这条不变式钉死（与 `scrollEcho`、
 * `wheelZoomIntent`、`verticalZoomSettle` 同一模式）。
 *
 * 【与其他模块的关系】
 * - 上游：宿主 `ensureScene` 每帧调用；入参里的标量都在调用处现读。
 * - 下游：为 true 时宿主调用 `rebuildInstances`。
 * - 独立性：纯函数，不依赖 DOM / React / GL。
 */

/** 场景重建判定入参（全部为标量，调用处现读）。 */
export interface SceneRebuildInputs {
    /** 显式标脏（内容 / 主题 / 数据镜像变化时由外部置位）。 */
    readonly sceneDirty: boolean;
    /** 构建当前几何时所用的水平缩放。 */
    readonly builtPxPerSec: number;
    /** 当前视口的水平缩放。 */
    readonly viewPxPerSec: number;
    /** 构建当前几何时所用的主题。 */
    readonly builtDarkMode: boolean;
    /** 当前主题。 */
    readonly darkMode: boolean;
    /**
     * 内容引用是否变化（clip / 轨道 / 工程长度 / 网格参数 / 选中态等）。
     *
     * 由宿主用**引用比较**算好传入（Redux/Immer 未变更时引用稳定），
     * 本函数不关心它由哪些字段组成。
     */
    readonly contentChanged: boolean;
    /** 构建当前几何时的水平位置。 */
    readonly builtScrollLeftPx: number;
    /** 当前水平位置。 */
    readonly viewScrollLeft: number;
    /** 构建当前几何时的竖直位置。 */
    readonly builtScrollTopPx: number;
    /** 当前竖直位置。 */
    readonly viewScrollTop: number;
    /**
     * **构建几何所用**的行高（内核真值 `view.rowHeight`）。
     *
     * 特殊说明：**不得**传 React 镜像行高。镜像会滞后一帧，用它当判据会让"内核已换行高、
     * 几何还是旧行高"的那一帧逃过重建（见文件头）。
     */
    readonly viewRowHeight: number;
    /** 构建当前几何时所用的行高（同源口径，取内核值）。 */
    readonly builtRowHeight: number;
    /** 水平方向"免重建"余量（CSS px）。 */
    readonly horizontalMarginPx: number;
    /** 竖直方向"免重建"余量（行数）。 */
    readonly verticalOverscanRows: number;
}

/**
 * 判定本帧是否需要重建场景几何。
 *
 * 流程（任一成立即重建）：
 * 1. 显式标脏；
 * 2. 水平缩放变化（网格 / clip 的横向投影变了）；
 * 3. 主题变化（配色进了实例缓冲）；
 * 4. 内容引用变化（含**行高**：行高变了所有行的内容坐标都变，必须重建）；
 * 5. 水平位置超出余量；
 * 6. 竖直位置超出余量（轨道窗口按 `scrollTop` 构建，不重建会缺行）。
 *
 * 特殊说明：竖直余量按 `viewRowHeight` 换算（与几何的竖直步长同源）。
 * 水平余量是绝对像素（内容宽度与行高无关），竖直余量是行数。
 *
 * @param input 见 {@link SceneRebuildInputs}。
 * @returns 需要重建时为 true。
 */
export function shouldRebuildScene(input: SceneRebuildInputs): boolean {
    return (
        input.sceneDirty ||
        input.builtPxPerSec !== input.viewPxPerSec ||
        input.builtDarkMode !== input.darkMode ||
        input.contentChanged ||
        // 行高：判据与几何同源（都用内核值）。镜像值不参与，见文件头。
        input.builtRowHeight !== input.viewRowHeight ||
        Math.abs(input.viewScrollLeft - input.builtScrollLeftPx) > input.horizontalMarginPx ||
        Math.abs(input.viewScrollTop - input.builtScrollTopPx) >
            input.viewRowHeight * input.verticalOverscanRows
    );
}
