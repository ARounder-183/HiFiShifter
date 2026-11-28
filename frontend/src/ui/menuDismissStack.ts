/**
 * 菜单的 dismiss 层栈。
 *
 * 【它解决的问题】"菜单里住着另一个菜单的触发器"不是假想场景：参数编辑器工具栏
 * 的「⋯」溢出菜单里放着 `ReferenceTrackGroup` / `ChildPitchGroup` 两个组件，
 * 而它们各自带一个下拉。点开里层下拉再点它的条目时，**外层菜单会先关掉** ——
 * 因为外层判"点在菜单外面"只看自己的 DOM 子树，而里层面板是 portal 到 body 的
 * 独立节点。
 *
 * Radix 靠 `DismissableLayer` 的层栈解决这件事。我们的原语没有层栈，于是"嵌套
 * 菜单"在迁移后必定回归。本模块补上这一层，语义与 Radix 一致：
 *
 *   **点在某处时，只有"比包含该处的最深一层更浅"的层被关掉。**
 *
 * 也就是：点在里层菜单里 → 外层不动；点在所有菜单之外 → 全关。
 *
 * 【为什么是模块级而不是 Context】层栈是**跨 portal** 的：里层面板挂在 body 下、
 * 与外层菜单没有任何 DOM 或 React 父子关系（那正是问题所在）。Context 传不过去，
 * 而"谁在上面"是全局事实。与 `useMenuKeyboard` 的 `menuStack`（键盘归属）互为
 * 镜像：一个管按键给谁，一个管点击关谁。
 */

interface MenuDismissLayer {
    /** 这一层的表面元素。 */
    element: HTMLElement;
    /** 关闭这一层。必须幂等（重复调用不应产生副作用）。 */
    dismiss: () => void;
}

/** 按打开顺序排列；**栈顶 = 最后打开 = 最内层**。 */
const layers: MenuDismissLayer[] = [];

/**
 * 登记一层。
 *
 * @returns 注销函数。调用方在卸载 / 关闭时调用它 —— 不注销的层会永远留在栈里，
 * 之后每一次"点外面"都会试图关一个已经关掉的菜单。
 */
export function pushMenuDismissLayer(element: HTMLElement, dismiss: () => void): () => void {
    const layer: MenuDismissLayer = { element, dismiss };
    layers.push(layer);
    return () => {
        const at = layers.lastIndexOf(layer);
        if (at >= 0) layers.splice(at, 1);
    };
}

/**
 * 处理一次"菜单外部的 pointerdown"。
 *
 * 找到**最深的、包含事件目标**的那一层；它及其更浅的层全部保留，更深（后打开）
 * 的层全部关闭。没有任何层包含目标时全关。
 *
 * 【为什么从栈顶往下找】最深的层就是最后打开的那一层；从栈顶找第一个命中的即可，
 * 不必比较 DOM 包含关系（嵌套菜单的两层面板互不包含）。
 */
export function handleMenuDismissPointerDown(target: Node | null): void {
    let deepest = -1;
    for (let index = layers.length - 1; index >= 0; index -= 1) {
        if (target !== null && layers[index].element.contains(target)) {
            deepest = index;
            break;
        }
    }
    // 先拷一份：dismiss 会让调用方卸载并**就地**注销自己，直接遍历原数组会漏层。
    const doomed = layers.slice(deepest + 1);
    for (let index = doomed.length - 1; index >= 0; index -= 1) {
        doomed[index].dismiss();
    }
}

/**
 * 同一个 `Escape` 事件只处理一次。
 *
 * 【为什么需要】每个菜单表面都在 `document` 上挂着 keydown。一次 Escape 会让
 * 所有打开的层各收到一次 —— 若每层都"关掉自己"，一次按键就关掉整棵菜单树。
 * 用事件对象本身做去重键：同一个 KeyboardEvent 的第二、三次调用直接返回 true。
 */
let lastEscapeEvent: KeyboardEvent | null = null;

/**
 * 处理一次 Escape：**只关最内层**。
 *
 * @returns 有层被关掉（或本事件已被处理）时为 true；调用方据此决定要不要退回
 * "关掉自己"。栈为空时返回 false。
 */
export function handleMenuDismissEscape(event: KeyboardEvent): boolean {
    if (lastEscapeEvent === event) return true;
    lastEscapeEvent = event;
    const top = layers[layers.length - 1];
    if (!top) return false;
    top.dismiss();
    return true;
}

/** 当前层数（测试用；生产代码不需要知道）。 */
export function menuDismissLayerCount(): number {
    return layers.length;
}

/** 清空层栈（仅测试用：模块级状态必须在用例之间归零）。 */
export function resetMenuDismissStack(): void {
    layers.length = 0;
    lastEscapeEvent = null;
}
