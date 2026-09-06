/**
 * focusSurface.ts — 活动编辑表面（activeSurface）的单一事实源
 *
 * 剪贴板等编辑快捷键（Ctrl+C/X/V、Ctrl+A…）的归属裁决依据：用户最后一次
 * pointerdown 落在哪个编辑表面，该表面就拥有这些按键 —— 与 DAW「最后点击
 * 的编辑器上下文」惯例一致；点击表面之外的区域（播放条、浮动工具窗、
 * 对话框等）不改变归属，编辑上下文得以保留。
 *
 * 为什么不依赖 document.activeElement：时间轴/轨道列刻意在 pointerdown 里
 * preventDefault 自行管理焦点（且表面无 tabIndex），点击它们后 DOM 焦点会
 * 停留在上一个可聚焦元素（如参数编辑器 scroller）上，用它路由会把事件送错
 * 执行者 —— 这正是参数编辑器与时间轴复制/剪切/粘贴互相冲突的根因。
 *
 * 为什么用 document 捕获阶段监听：capture 在传播链最前端触发，天然免疫
 * 子元素的 preventDefault / stopPropagation，任何点击都逃不过解析。
 */

/** 编辑表面的规范类型；与 keybindings/focusRouting 的焦点域一一对应。 */
export type EditSurfaceId = "timeline" | "pianoRoll" | "trackHeader";

/** 各编辑表面根元素声明的规范属性（closest 就近解析，允许嵌套）。 */
export const EDIT_SURFACE_ATTR = "data-hs-surface";

let activeSurface: EditSurfaceId | null = null;

export function getActiveSurface(): EditSurfaceId | null {
    return activeSurface;
}

/**
 * 编程式设置活动表面：用于不经过 pointer 事件的聚焦场景 —— 例如双击
 * Clip 在参数编辑器内创建选区后，把交互焦点（复制/剪切路由、外来源
 * 粘贴兜底等）切到参数编辑器侧。下一次 pointerdown / focusin 仍会照常
 * 覆盖本值。
 */
export function setActiveSurfaceExplicit(surface: EditSurfaceId): void {
    activeSurface = surface;
}

/** 由事件目标解析所属编辑表面；不在任何表面内时返回 null。 */
export function resolveSurfaceFromTarget(target: EventTarget | null): EditSurfaceId | null {
    const el = target as HTMLElement | null;
    const value = el?.closest?.(`[${EDIT_SURFACE_ATTR}]`)?.getAttribute(EDIT_SURFACE_ATTR);
    if (value === "timeline" || value === "pianoRoll" || value === "trackHeader") {
        return value;
    }
    return null;
}

/** pointerdown / focusin 落点更新：落在表面内才更新，表面之外保持原归属。 */
export function updateActiveSurfaceFrom(target: EventTarget | null): void {
    const surface = resolveSurfaceFromTarget(target);
    if (surface && surface !== activeSurface) {
        activeSurface = surface;
    }
}

/**
 * 安装全局 pointerdown / focusin（均捕获阶段）监听，驱动 activeSurface。
 * App 一次性调用；返回清理函数。
 */
export function installFocusSurfaceTracking(): () => void {
    const onPointerDown = (e: PointerEvent) => updateActiveSurfaceFrom(e.target);
    const onFocusIn = (e: FocusEvent) => updateActiveSurfaceFrom(e.target);
    document.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("focusin", onFocusIn, true);
    return () => {
        document.removeEventListener("pointerdown", onPointerDown, true);
        document.removeEventListener("focusin", onFocusIn, true);
    };
}

/** 仅测试用：重置归属。 */
export function resetActiveSurfaceForTests(): void {
    activeSurface = null;
}
