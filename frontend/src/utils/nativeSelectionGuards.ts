/*
 * WebView 原生选择/拖拽的准入判定。
 *
 * ## 背景
 *
 * 这是一个 DAW 风格的应用：默认**整页都不可选中文本**（见 index.css 的
 * `body * { user-select: none }`），只对可编辑控件与显式声明
 * `data-hs-selectable="true"` 的表面放行。App 在 document 上装了三个守卫：
 *
 * - `selectstart` → 不允许则 `preventDefault()`；
 * - `pointerdown` / `mouseup` → 不允许则清空已形成的选区；
 * - `dragstart` → 不允许则阻止原生拖拽。
 *
 * ## 为什么这里要专门处理"文本节点"
 *
 * `selectstart` 的 **target 是文本节点**（nodeType 3），不是元素 —— 这是
 * 浏览器规范行为。早先的实现直接把 target 当 Element 用：
 *
 * ```js
 * const tag = (el.tagName ?? "").toLowerCase();   // 文本节点 → ""
 * if (el.isContentEditable) return true;          // 文本节点 → undefined
 * return el.closest?.(...) != null;               // 文本节点没有 closest → false
 * ```
 *
 * 于是"在 contenteditable 里选中文字"被判成"不可选中"，`selectstart` 被
 * preventDefault，选区被 mouseup 守卫清掉 —— 表现为**在富文本编辑器里既不能
 * 拖选文字、点击文字也不落光标**。
 *
 * 旧的记事本用 `<textarea>`，它的 `selectstart` target 就是 textarea 元素本身
 * （命中 `tag === "textarea"` 分支），因此一直没暴露这个问题；富文本编辑器是
 * 应用里**第一个 contenteditable**，正好踩中。
 *
 * 修法：所有判定先把 target 归一成元素（文本节点取 `parentElement`），再做
 * 后续判断。这样与引擎无关，也不依赖 contenteditable 在各引擎里的特殊行为。
 */

/** 把事件目标归一成元素：文本节点（含注释等）取其父元素。 */
export function elementFromEventTarget(target: EventTarget | null): Element | null {
    if (!target) return null;
    if (target instanceof Element) return target;
    const parent = (target as Node).parentElement;
    return parent ?? null;
}

/** 目标是否位于可编辑控件内（input / textarea / select / contenteditable）。 */
export function isEditableTarget(target: EventTarget | null): boolean {
    const el = elementFromEventTarget(target);
    if (!el) return false;
    const tag = (el.tagName ?? "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return true;
    // `isContentEditable` 只声明在 HTMLElement 上（SVG 元素没有）。
    if ((el as HTMLElement).isContentEditable) return true;
    return el.closest?.('input,textarea,select,[contenteditable="true"]') != null;
}

/** 计算样式里的 user-select（含 WebKit 前缀），读不到时返回空串。 */
function userSelectOf(element: Element): string {
    try {
        const style = window.getComputedStyle(element) as CSSStyleDeclaration & {
            webkitUserSelect?: string;
        };
        return style.userSelect || style.webkitUserSelect || "";
    } catch {
        return "";
    }
}

/**
 * 是否允许 WebView 的原生文本选择。
 *
 * 放行条件（任一）：
 * - 目标在可编辑控件内；
 * - 目标自身或任一祖先显式声明 `data-hs-selectable="true"`；
 * - 目标自身或任一祖先的计算 `user-select` 是 `text` / `all`。
 *
 * 命中 `user-select: none` 的祖先即判为不可选中（这是 index.css 的
 * "整页不可选中"基线所用的信号）。
 */
export function allowsNativeTextSelection(target: EventTarget | null): boolean {
    if (isEditableTarget(target)) return true;

    let node: Element | null = elementFromEventTarget(target);
    while (node) {
        if (node.getAttribute?.("data-hs-selectable") === "true") return true;
        const userSelect = userSelectOf(node);
        if (userSelect === "text" || userSelect === "all") return true;
        if (userSelect === "none") return false;
        node = node.parentElement;
    }
    return false;
}
