/**
 * 「点击面板留白时，焦点该不该收回列表」的判定。
 *
 * 【要修的问题】工具条背景、搜索栏留白、路径栏留白这些地方没有可聚焦元素，
 * 浏览器会把焦点丢回 `<body>` —— 于是面板的键盘模型整个失效：keydown 处理器挂在
 * 面板根上，而 `body` 不是它的后代，事件根本不经过它。用户点一下工具条背景再打字，
 * 既不会跳转、也不触发任何面板快捷键。
 *
 * 【为什么抽成纯函数】判定只看"点中的元素是不是自己管焦点的控件"，与 Redux、
 * 面板状态都无关；抽出来就能直接单测，而不必渲染整个面板（面板依赖 store / Tauri /
 * 音频上下文，本仓的键盘测试刻意不渲染它）。
 */

/**
 * 点击这些元素时焦点归它们自己 —— 面板不抢。
 *
 * - `[tabindex]:not([tabindex="-1"])` 兜住其它可 Tab 进入的控件；
 * - `[tabindex="-1"]` 的容器（列表容器自己）**不算** —— 点它的留白正是要聚焦它；
 * - `.rt-TextFieldRoot`：带插槽的文本框是**一个**控件，点它的插槽或内边距（例如
 *   放大镜图标那一带）不该被当成"留白"。点它时由 `takePanelFocus` 把光标送进
 *   里面的 `<input>`。
 */
export const FOCUS_OWNING_SELECTOR = [
    "input",
    "textarea",
    "select",
    "button",
    "a[href]",
    '[contenteditable="true"]',
    '[role="option"]',
    '[role="menuitem"]',
    '[role="slider"]',
    '[role="combobox"]',
    '[tabindex]:not([tabindex="-1"])',
    ".rt-TextFieldRoot",
].join(",");

/** Radix 带插槽的文本框容器（点它任意一处都该进输入框）。 */
const TEXT_FIELD_SELECTOR = ".rt-TextFieldRoot";

/**
 * 这次点击是否应当由面板把焦点收回列表。
 *
 * @param target 指针按下的目标元素（`event.target`）。
 * @returns 目标是留白（不在任何"自己管焦点"的元素内）时返回 `true`。
 */
export function shouldPanelTakeFocus(target: Element | null): boolean {
    if (!target?.closest) return false;
    return target.closest(FOCUS_OWNING_SELECTOR) === null;
}

/** 处理器需要的最小事件形状（便于单测，不必构造完整的 React 合成事件）。 */
export interface PanelFocusEvent {
    target: EventTarget | null;
    button: number;
    preventDefault: () => void;
}

/**
 * 把焦点从留白收回到列表容器。
 *
 * 【为什么必须 `preventDefault()`】浏览器在**按下**时执行"把焦点移到被点元素"这个
 * 默认动作：点在非可聚焦的留白上时，它会把焦点挪到 `<body>`（并 blur 掉此前聚焦的
 * 元素）。只在按下时 `focus()` 而不阻止默认动作，等于刚设好的焦点立刻被浏览器收回，
 * 表现就是"点了工具条背景之后依然打不了字"——这正是上一版没修好的原因。
 *
 * 【为什么只处理主键】右键点留白要弹背景菜单；对右键 `preventDefault` 有可能连带
 * 抑制 `contextmenu`。中键同理，不参与焦点归属。
 *
 * @param event 按下事件（pointerdown / mousedown 都可，两个都接是为了不依赖
 *   "取消 pointerdown 会不会连带取消 mousedown"这一引擎差异）。
 * @param target 要聚焦的列表容器。
 * @returns 是否接手了这次焦点（调用方一般不需要用它）。
 */
export function takePanelFocus(event: PanelFocusEvent, target: HTMLElement | null): boolean {
    if (event.button !== 0) return false;
    const source = event.target as Element | null;

    /*
     * 点在输入框 / 文本域**本身**上：完全交给浏览器。
     *
     * 【为什么必须提前退出】对 mousedown 做 `preventDefault()` 会连"点选光标位置"
     * 一起挡掉 —— 在文本中间点一下想移动插入点会变成什么都不发生。
     */
    if (source?.closest?.("input, textarea")) return false;

    /*
     * 点在带插槽的文本框上（插槽、内边距，例如放大镜那一带）：把光标送进里面的输入框。
     *
     * 【为什么需要】这类字段是"一个控件"，点它任意一处都应当能开始打字 —— 输入框的
     * 通行行为。不做这一步的话，点插槽既不进输入框、也不会被当作留白
     * （`.rt-TextFieldRoot` 在 FOCUS_OWNING_SELECTOR 里），焦点会掉到 `<body>`。
     */
    const field = source?.closest?.(TEXT_FIELD_SELECTOR);
    if (field) {
        const input = field.querySelector("input, textarea");
        if (input instanceof HTMLElement) {
            event.preventDefault();
            input.focus({ preventScroll: true });
            return true;
        }
    }

    if (!shouldPanelTakeFocus(source)) return false;
    event.preventDefault();
    target?.focus({ preventScroll: true });
    return true;
}
