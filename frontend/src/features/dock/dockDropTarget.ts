/*
 * 拖拽落点判定 —— 纯几何。
 *
 * 渲染层（DockDropOverlay）与拖拽逻辑（useDockDrag）共用这里的函数：如果两边
 * 各算一次，"看到的高亮"与"实际落点"必然会在某些尺寸下漂移，而这类漂移用户
 * 一眼就能看出、却极难复现描述。
 *
 * 落点规则（对齐 REAPER / VEGAS 的直觉）：
 * - 指针在某个 Zone 的**中央区**（距四边都超过感应带）→ 并入该 Zone 的标签组；
 * - 否则落到**最近的那条边** → 在该边方向拆出新组；
 * - 指针不在任何 Zone 内 → 由调用方决定浮动。
 *
 * 分割产生的标签组 Zone 矩形互不重叠（分割即划分）；根级边缘带是有意叠加在
 * 外缘之上的合成 Zone，重叠时按"面积最小者优先"取最具体的落点（见
 * `pickDropTarget`）。
 */

import type { DockDropZone, DockRect } from "./dockTypes";

export function pointInRect(rect: DockRect, x: number, y: number): boolean {
    return x >= rect.x && x <= rect.x + rect.w && y >= rect.y && y <= rect.y + rect.h;
}

/** 每条边各自的感应带厚度（像素）。缺省侧回落到统一的 `edgeBandPx`。 */
export interface DockSideBands {
    left: number;
    right: number;
    top: number;
    bottom: number;
}

/**
 * 判定指针在给定矩形内的落点。
 *
 * 感应带会被钳制到矩形短边的 40%：窄条（比如折叠后的标签条）如果按固定
 * 像素算感应带，四条带会互相吃掉，中央区消失，用户就再也无法"并入标签组"。
 *
 * `sideBands` 允许四条边各用不同的厚度（缺省侧用 `edgeBandPx`）：与根级
 * 边缘带重合的那一侧需要**向外扩展**（见 `tabsetSideBands`），把被根带
 * 遮住的局部拆分感应区在根带内侧补回来。判定规则不变：落在某条边的带内
 * → 该侧拆分；四条带都不在 → 中央并入；多条带同时命中取距离最近者，
 * 同距时水平边优先（与逐维比较的历史行为一致）。
 */
export function resolveDropZone(
    rect: DockRect,
    pointer: { x: number; y: number },
    edgeBandPx: number,
    sideBands?: Partial<DockSideBands>,
): DockDropZone | null {
    if (rect.w <= 0 || rect.h <= 0) return null;
    if (!pointInRect(rect, pointer.x, pointer.y)) return null;

    const cap = Math.min(rect.w, rect.h) * 0.4;
    const clampBand = (px: number) => Math.max(8, Math.min(px, cap));
    const bands: DockSideBands = {
        left: clampBand(sideBands?.left ?? edgeBandPx),
        right: clampBand(sideBands?.right ?? edgeBandPx),
        top: clampBand(sideBands?.top ?? edgeBandPx),
        bottom: clampBand(sideBands?.bottom ?? edgeBandPx),
    };

    const left = pointer.x - rect.x;
    const right = rect.x + rect.w - pointer.x;
    const top = pointer.y - rect.y;
    const bottom = rect.y + rect.h - pointer.y;

    const candidates: Array<{ zone: DockDropZone; dist: number; horizontal: boolean }> = [];
    if (left <= bands.left) candidates.push({ zone: "left", dist: left, horizontal: true });
    if (right <= bands.right) candidates.push({ zone: "right", dist: right, horizontal: true });
    if (top <= bands.top) candidates.push({ zone: "top", dist: top, horizontal: false });
    if (bottom <= bands.bottom) {
        candidates.push({ zone: "bottom", dist: bottom, horizontal: false });
    }
    if (candidates.length === 0) return "center";

    let best = candidates[0];
    for (const candidate of candidates.slice(1)) {
        const closer = candidate.dist < best.dist;
        const tieBreaksHorizontal =
            candidate.dist === best.dist && candidate.horizontal && !best.horizontal;
        if (closer || tieBreaksHorizontal) best = candidate;
    }
    return best.zone;
}

/** 一个可停靠目标：Zone 及其当前视口矩形。 */
export interface DockZoneRect {
    zoneId: string;
    rect: DockRect;
    /**
     * 这条 Zone 属于**哪棵布局根**。
     *
     * 多根之后，zone id 只在根内唯一，同一份 DOM 里可能有多个根的边缘带与
     * 标签组 —— 提交时必须知道改哪棵树。浮窗组合目标没有根（它创建根）。
     */
    rootId?: string;
    /**
     * 非空 = 这是一条**浮窗组合目标**：拖一个浮窗到这个浮窗上 → 组合成面板
     * （目标已是面板则停入它的树）。
     */
    floatFormId?: string;
    /**
     * 非空 = 这条 Zone 在某个浮窗内部（浮动面板的边缘带 / 标签组）。
     *
     * 【为什么需要】浮窗盖在主停靠区之上：指针落在浮窗里时，被浮窗遮住的
     * 主区落点必须让位（用户看到的上层是谁，落点就是谁）。按"最上层浮窗"
     * 过滤候选就是这条规则的实现（见 `resolveTarget`）。
     */
    floatOwnerFormId?: string;
    /**
     * 预解析的落点部位。普通标签组 Zone 省略它（落点按指针在矩形内的位置
     * 现场解析）；**合成 Zone**（根级边缘带，见 `buildRootEdgeZones`）整个
     * 矩形只对应一个部位，必须在这里固定 —— 对着一条"右侧带"跑 `resolveDropZone`
     * 会因为带内位置不同解析出 left/top 等错误结果。
     */
    fixedZone?: DockDropZone;
    /**
     * 提交给覆盖层/提交逻辑的矩形（缺省 = `rect`）。根级边缘带的命中矩形是
     * 一条细带，但预览与提交语义都以**整个根矩形**为基准 —— "贯通整侧"的
     * 半边高亮必须从根矩形算出。
     */
    previewRect?: DockRect;
    /**
     * 四条边各自的感应带厚度（缺省 = 统一 `edgeBandPx`）。与根级边缘带重合
     * 的侧边由 `tabsetSideBands` 向外扩展，保证局部拆分在根带内侧仍有完整的
     * 感应宽度（否则默认布局里"只拆时间轴右侧"会被根带整个吃掉 —— 用户报告）。
     */
    sideBands?: Partial<DockSideBands>;
}

/**
 * 根级边缘带的合成 Zone id。四个方向的带共用它：提交与提示只需要知道
 * "这是根级落点"，具体侧向从 `zone`（由 `fixedZone` 解析而来）读取，
 * 具体改哪棵树由 `rootId` 读取（多根之后每个布局根都有自己的边缘带）。
 */
export const DOCK_ROOT_ZONE_ID = "__dock_root__";

/**
 * 根级边缘带的实际厚度：与 `buildRootEdgeZones` 内部同一套钳制。
 *
 * `tabsetSideBands` 计算重合侧的扩展宽度时必须用**这**个值（而不是裸的
 * `edgeBandPx`）—— 停靠区极小时根带会被钳短，扩展量跟着收缩，"根带占外侧、
 * 局部带占内侧"的划分才始终严丝合缝。
 */
export function rootEdgeBandThickness(rootRect: DockRect, edgeBandPx: number): number {
    if (rootRect.w <= 0 || rootRect.h <= 0) return 0;
    return Math.max(8, Math.min(edgeBandPx, Math.min(rootRect.w, rootRect.h) * 0.25));
}

/**
 * 计算一个标签组四条边的感应带厚度。
 *
 * 【为什么要扩展】根级边缘带（厚度 = `rootEdgeBandThickness`）叠在停靠区
 * 外缘上，"面积最小者优先"会让它赢下与标签组侧边感应带重叠的部分。默认
 * 布局（上下分布）里时间轴的右缘就是停靠区的右缘 —— 不补偿的话，"只拆
 * 时间轴右侧"整条感应区都被根带吃掉，用户只剩"并入标签组"和"贯通整侧"
 * 两种落点（用户报告）。
 *
 * 补偿规则：标签组某条边与根缘**重合**（±1px，容忍浮点取整）时，该侧的
 * 局部感应带向外扩展 `edgeBandPx + 根带厚度`。于是沿这条边从外到内依次是：
 * 根带 `[0, 根带]` → 局部带 `(根带, 根带 + edgeBandPx]` → 中央区。两种意图
 * 各自保有完整的 `edgeBandPx` 感应宽度，谁也不吃掉谁；不重合的内侧边不受
 * 影响，维持原有的单一厚度。
 */
export function tabsetSideBands(
    rect: DockRect,
    rootRect: DockRect,
    rootBandPx: number,
    edgeBandPx: number,
): DockSideBands {
    const coincident = (a: number, b: number) => Math.abs(a - b) <= 1;
    const expanded = edgeBandPx + rootBandPx;
    return {
        left: coincident(rect.x, rootRect.x) ? expanded : edgeBandPx,
        right: coincident(rect.x + rect.w, rootRect.x + rootRect.w) ? expanded : edgeBandPx,
        top: coincident(rect.y, rootRect.y) ? expanded : edgeBandPx,
        bottom: coincident(rect.y + rect.h, rootRect.y + rootRect.h) ? expanded : edgeBandPx,
    };
}

/**
 * 构造根级边缘带的四个合成 Zone：贴着停靠区外缘的一圈感应带，命中即表示
 * "把窗体拆到整个停靠区的这一侧"（贯通全高/全宽），而不是拆开指针恰好
 * 悬停的那个最内层标签组。
 *
 * 【为什么需要】嵌套布局里标签组只铺满自己所在的分支。默认布局（上下分布）
 * 拆出的"右侧新组"若以标签组为参照，只能贴着上块或下块的半高右侧；用户
 * 想要的"两者共同的右侧"必须以**根**为参照拆分。外缘感应带就是为这个意图
 * 预留的：贴边越狠，拆得越"外"。
 *
 * 【与标签组感应带的关系】两组带在"标签组边缘恰好贴着停靠区边缘"时必然
 * 重叠，这是位置判定模型无法消除的物理歧义。划分方式：外侧让给根级带
 * （贴到应用最边缘 = 贯通整侧），内侧还给标签组 —— 重合侧的标签组感应带由
 * `tabsetSideBands` 向外扩展，宽度不受挤占。
 *
 * 四条带互不重叠（左右带贯通全高、上下带让出左右两角），任一指针位置最多
 * 命中一条；`pickDropTarget` 的"面积最小者优先"规则恰好让细带压过下方
 * 标签组的大矩形，不需要额外优先级逻辑。
 */
export function buildRootEdgeZones(
    rootRect: DockRect,
    edgeBandPx: number,
    rootId?: string,
): DockZoneRect[] {
    if (rootRect.w <= 0 || rootRect.h <= 0) return [];
    // 与 resolveDropZone 同样的钳制思路：停靠区极小时按短边收缩，保证中央
    // 区域（并入标签组）永远还有立足之地。
    const band = rootEdgeBandThickness(rootRect, edgeBandPx);
    const innerX = rootRect.x + band;
    const innerW = Math.max(0, rootRect.w - band * 2);
    const left: DockZoneRect = {
        zoneId: DOCK_ROOT_ZONE_ID,
        rootId,
        fixedZone: "left",
        previewRect: rootRect,
        rect: { x: rootRect.x, y: rootRect.y, w: band, h: rootRect.h },
    };
    const right: DockZoneRect = {
        zoneId: DOCK_ROOT_ZONE_ID,
        rootId,
        fixedZone: "right",
        previewRect: rootRect,
        rect: { x: rootRect.x + rootRect.w - band, y: rootRect.y, w: band, h: rootRect.h },
    };
    const top: DockZoneRect = {
        zoneId: DOCK_ROOT_ZONE_ID,
        rootId,
        fixedZone: "top",
        previewRect: rootRect,
        rect: { x: innerX, y: rootRect.y, w: innerW, h: band },
    };
    const bottom: DockZoneRect = {
        zoneId: DOCK_ROOT_ZONE_ID,
        rootId,
        fixedZone: "bottom",
        previewRect: rootRect,
        rect: { x: innerX, y: rootRect.y + rootRect.h - band, w: innerW, h: band },
    };
    return [left, right, top, bottom];
}

/**
 * 空面板占位井的合成 Zone：整块井面就是一个"并入"落点。
 *
 * 【为什么需要】空面板没有标签组（没有 `data-dock-zone`），而根级边缘带只
 * 覆盖外缘一圈 —— 井的正中央会变成"无落点"，用户把窗体拖到空面板正中却
 * 什么都发生。补一条盖满井面的 center 合成 Zone，空面板从建好那一刻起
 * 全面积可停靠。
 */
export function buildEmptyRootZone(rootRect: DockRect, rootId: string): DockZoneRect {
    return {
        zoneId: DOCK_ROOT_ZONE_ID,
        rootId,
        fixedZone: "center",
        previewRect: rootRect,
        rect: { ...rootRect },
    };
}

/**
 * 从全部 Zone 中选出指针所在的那个。
 *
 * 标签组 Zone 互不重叠；根级边缘带（`buildRootEdgeZones`）会有意叠在标签组
 * 的外缘之上 —— 取"面积最小者优先"：细带压过大矩形，贴边的指针表达的是
 * "拆整个停靠区"而不是"拆这个标签组"。
 */
export function pickDropTarget(
    zones: readonly DockZoneRect[],
    pointer: { x: number; y: number },
): DockZoneRect | null {
    let best: DockZoneRect | null = null;
    for (const zone of zones) {
        if (!pointInRect(zone.rect, pointer.x, pointer.y)) continue;
        if (!best || zone.rect.w * zone.rect.h < best.rect.w * best.rect.h) best = zone;
    }
    return best;
}

/**
 * 把落点换算成"新组在目标组的哪一侧"。
 *
 * 与 `resolveDropZone` 分开是因为语义不同：`center` 不是"某一侧"，而是
 * "并入同一组"。调用方据此分派到 `{kind:"tab"}` 或 `{kind:"split"}`。
 */
export function dropZoneToSide(zone: DockDropZone): "left" | "right" | "top" | "bottom" | null {
    return zone === "center" || zone === "float" ? null : zone;
}

/**
 * 拖拽落点预览矩形（用于画半透明提示框）。
 *
 * 与真实落点同源，因此预览框与实际结果必然一致 —— 这是本模块存在的理由。
 */
export function dropPreviewRect(
    target: DockRect,
    zone: DockDropZone,
    splitterPx: number,
): DockRect | null {
    if (zone === "float") return null;
    if (zone === "center") return { ...target };

    const half = splitterPx / 2;
    switch (zone) {
        case "left":
            return { x: target.x, y: target.y, w: Math.max(0, target.w / 2 - half), h: target.h };
        case "right":
            return {
                x: target.x + target.w / 2 + half,
                y: target.y,
                w: Math.max(0, target.w / 2 - half),
                h: target.h,
            };
        case "top":
            return { x: target.x, y: target.y, w: target.w, h: Math.max(0, target.h / 2 - half) };
        default:
            return {
                x: target.x,
                y: target.y + target.h / 2 + half,
                w: target.w,
                h: Math.max(0, target.h / 2 - half),
            };
    }
}

/**
 * 浮动窗落点：夹紧到主窗口可用区内。
 *
 * 只保证"标题栏可见"而不是整个窗口可见 —— 允许窗体部分伸出视口是 DAW 的
 * 常见用法（把长面板推到边上），但标题栏必须留着，否则用户再也抓不回来。
 */
export function clampFloatRect(
    rect: DockRect,
    viewport: { w: number; h: number },
    titleBarPx: number,
): DockRect {
    const w = Math.min(rect.w, Math.max(200, viewport.w));
    const h = Math.min(rect.h, Math.max(120, viewport.h));
    const minVisible = 48;
    const maxX = viewport.w - minVisible;
    const minX = -(w - minVisible);
    const maxY = viewport.h - Math.max(titleBarPx, minVisible);
    const minY = 0;
    return {
        x: Math.round(Math.min(maxX, Math.max(minX, rect.x))),
        y: Math.round(Math.min(maxY, Math.max(minY, rect.y))),
        w: Math.round(w),
        h: Math.round(h),
    };
}

/**
 * 把浮动几何解析成具体矩形：带锚点时按**当前视口**推导位置。
 *
 * 锚点每帧按视口解析，因此主窗口缩放后浮窗仍在右下角；用户手动移动过之后锚点已
 * 被清除（见 `setFloatGeometry`），此处只是原样返回。
 */
export function resolveFloatRect(
    geometry: DockRect & {
        anchor?: string | null;
        anchorMarginPx?: number;
        anchorOffsetX?: number;
        anchorOffsetY?: number;
    },
    viewport: { w: number; h: number },
): DockRect {
    if (geometry.anchor !== "bottom-right" && geometry.anchor !== "center") {
        return { x: geometry.x, y: geometry.y, w: geometry.w, h: geometry.h };
    }
    const margin = geometry.anchorMarginPx ?? 24;
    const offsetX = Number.isFinite(geometry.anchorOffsetX)
        ? (geometry.anchorOffsetX as number)
        : 0;
    const offsetY = Number.isFinite(geometry.anchorOffsetY)
        ? (geometry.anchorOffsetY as number)
        : 0;

    /*
     * 居中：设置类面板（外观设置）的落点。
     *
     * 与右下角同样**按当前视口每帧解析**，因此主窗口缩放后它仍在正中；
     * 同样把结果夹进视口 —— 视口比窗体还小时宁可盖住内容，也不要跑到屏幕外
     * （夹紧后它退化为"贴着边距原点"，与右下角锚点的兜底一致）。
     */
    const x =
        geometry.anchor === "center"
            ? Math.round((viewport.w - geometry.w) / 2 + offsetX)
            : Math.round(viewport.w - geometry.w - margin + offsetX);
    const y =
        geometry.anchor === "center"
            ? Math.round((viewport.h - geometry.h) / 2 + offsetY)
            : Math.round(viewport.h - geometry.h - margin + offsetY);

    return {
        x: Math.max(margin, x),
        y: Math.max(margin, y),
        w: geometry.w,
        h: geometry.h,
    };
}

/**
 * 把"从某个控件打开"的浮窗放到该控件附近（默认正下方，下方放不下则翻到上方）。
 *
 * 【为什么需要】由界面内的按钮打开的辅助面板（如从撤销/重做按钮打开"操作记录"），
 * 用户刚点的按钮就是他的注意力所在 —— 窗口出现在按钮旁边最自然；直接丢到屏幕角落
 * 会让人以为没打开。水平与控件左缘对齐，四边都夹进视口内。
 *
 * @param near 触发打开的控件矩形（客户区坐标）。
 * @param size 浮窗期望尺寸（放不下时会缩小）。
 * @param viewport 主窗口客户区尺寸。
 * @param gapPx 与控件之间的间距。
 * @param marginPx 与视口边缘的最小间距。
 */
export function resolveFloatNearRect(
    near: DockRect,
    size: { w: number; h: number },
    viewport: { w: number; h: number },
    gapPx = 8,
    marginPx = 8,
): DockRect {
    const w = Math.min(size.w, Math.max(200, viewport.w - marginPx * 2));
    const h = Math.min(size.h, Math.max(120, viewport.h - marginPx * 2));
    const below = near.y + near.h + gapPx;
    const above = near.y - gapPx - h;
    const fitsBelow = below + h <= viewport.h - marginPx;
    const fitsAbove = above >= marginPx;
    // 两侧都放不下（控件几乎占满视口高度）：贴视口底部，至少让标题条可见。
    const y = fitsBelow ? below : fitsAbove ? above : viewport.h - h - marginPx;
    return {
        x: Math.round(
            Math.min(Math.max(marginPx, near.x), Math.max(marginPx, viewport.w - w - marginPx)),
        ),
        y: Math.round(
            Math.min(Math.max(marginPx, y), Math.max(marginPx, viewport.h - h - marginPx)),
        ),
        w: Math.round(w),
        h: Math.round(h),
    };
}

/**
 * 浮动窗吸附：靠近视口边缘或其它浮动窗边时对齐。
 *
 * 返回吸附后的位置；`threshold` 为 0 时等价于不吸附（直接返回原值）。
 */
export function snapFloatPosition(
    position: { x: number; y: number; w: number; h: number },
    others: readonly DockRect[],
    viewport: { w: number; h: number },
    threshold: number,
): { x: number; y: number } {
    if (threshold <= 0) return { x: position.x, y: position.y };

    // 候选吸附线：视口边缘 + 其它窗体的边缘。
    const xs: number[] = [0, viewport.w];
    const ys: number[] = [0, viewport.h];
    for (const other of others) {
        xs.push(other.x, other.x + other.w, other.x - position.w, other.x + other.w - position.w);
        ys.push(other.y, other.y + other.h, other.y - position.h, other.y + other.h - position.h);
    }

    let bestX = position.x;
    let bestDx = threshold;
    for (const candidate of xs) {
        const delta = Math.abs(candidate - position.x);
        if (delta < bestDx) {
            bestDx = delta;
            bestX = candidate;
        }
    }

    let bestY = position.y;
    let bestDy = threshold;
    for (const candidate of ys) {
        const delta = Math.abs(candidate - position.y);
        if (delta < bestDy) {
            bestDy = delta;
            bestY = candidate;
        }
    }

    return { x: bestX, y: bestY };
}
