/**
 * 菜单的 typeahead（打字跳项）。
 *
 * 【为什么菜单需要它】菜单栏下拉此前只能靠方向键一项项走：从「文件」到「帮助」
 * 之间隔着几十项，键盘用户按十几次 ↓ 才能到。原生菜单、VS Code、Windows 资源
 * 管理器都支持"直接打首字母跳过去" —— 这是菜单的标准能力，不是增强。
 *
 * 【为什么判据在 DOM 上而不在 items 上】菜单有两族实现：声明式 `items`（标签是
 * `ReactNode`，可能是图标 + 文本 + 角标）与手写 `children`（标签就是任意 JSX）。
 * 从 props 里取"可匹配的文本"对两族都要各写一遍特例，而且 `ReactNode` 根本
 * 取不出干净字符串。菜单项渲染出来的 `.hs-menu__label-text` 是**唯一**权威文本。
 */

/**
 * 可导航的项：真实禁用（`disabled`）的项不参与，与鼠标行为一致。
 *
 * 【为什么三种角色都要收】勾选态让项渲染成 `menuitemcheckbox`，单选态渲染成
 * `menuitemradio`（`aria-checked` 只允许挂在这两个角色上）。只认 `menuitem`
 * 的话，"选项菜单里全是勾选项"的那类菜单方向键一格都不动。
 *
 * 导出：`useMenuKeyboard` 与本模块必须**同一个口径** —— 两处各写一份选择器，
 * 迟早出现"方向键能到、打字跳不到"（或反过来）的项。
 */
export const MENU_ITEM_SELECTOR =
    '[role="menuitem"]:not([disabled]), [role="menuitemcheckbox"]:not([disabled]), [role="menuitemradio"]:not([disabled])';

/**
 * 本层菜单的项（按 DOM 顺序）。
 *
 * 【为什么按 `closest('[role="menu"]')` 过滤】子菜单的面板在 DOM 上住在父菜单的
 * 子树里，`querySelectorAll` 会把两层的项一起捞出来 —— 于是外层按方向键会一路
 * 走进子菜单，而子菜单是**另一个**菜单表面，由它自己的处理负责。
 */
export function menuItemsOf(container: HTMLElement): HTMLElement[] {
    return Array.from(container.querySelectorAll<HTMLElement>(MENU_ITEM_SELECTOR)).filter(
        (item) => item.closest('[role="menu"]') === container,
    );
}

/** 项的匹配文本：优先取标签列，退回整行文本。 */
export function menuItemTypeaheadText(element: HTMLElement): string {
    const label = element.querySelector(".hs-menu__label-text");
    return (label?.textContent ?? element.textContent ?? "").trim();
}

/** 两次按键间隔超过它就重新开始拼前缀（与原生菜单同一量级）。 */
export const MENU_TYPEAHEAD_RESET_MS = 500;

/** 打字缓冲。由每个菜单表面各持一份（`useRef`），跨菜单不串。 */
export interface MenuTypeaheadState {
    /** 已拼好的前缀（小写）。 */
    query: string;
    /** 上一次按键的时间戳（`performance.now()` / `Date.now()` 均可，只用差值）。 */
    at: number;
}

export const EMPTY_MENU_TYPEAHEAD: MenuTypeaheadState = { query: "", at: 0 };

/**
 * 把一次按键推进缓冲。
 *
 * 三条规则：
 * 1. 超时 → 重新开始；
 * 2. 同一个字符连按 → **不要**拼成 "cc"，而是保持在 "c" 上（调用侧从"当前项
 *    之后"开始找，于是它表现为"在几个以 c 开头的项之间循环"）；
 * 3. 其余 → 追加。
 */
export function pushMenuTypeahead(
    state: MenuTypeaheadState,
    char: string,
    now: number,
): MenuTypeaheadState {
    if (char.length === 0) return state;
    if (now - state.at > MENU_TYPEAHEAD_RESET_MS) return { query: char, at: now };
    if (state.query.length > 0 && [...state.query].every((existing) => existing === char)) {
        return { query: char, at: now };
    }
    return { query: state.query + char, at: now };
}

/**
 * 从 `from + 1` 开始环形查找第一个以 `query` 开头的标签，返回**在传入数组里的
 * 下标**；找不到返回 -1（调用侧据此"不动"，而不是跳到第一项）。
 *
 * `from` 传 -1 表示"当前没有选中项"，从第一个开始找。
 */
export function matchMenuTypeahead(labels: readonly string[], query: string, from: number): number {
    const count = labels.length;
    if (count === 0) return -1;
    const needle = query.trim().toLowerCase();
    if (needle.length === 0) return -1;
    const start = from < 0 ? 0 : (from + 1) % count;
    for (let step = 0; step < count; step += 1) {
        const index = (start + step) % count;
        if (labels[index].trim().toLowerCase().startsWith(needle)) return index;
    }
    return -1;
}

/**
 * 在某个菜单容器里按前缀找项，返回 `menuItemsOf` 数组里的下标（找不到 -1）。
 */
export function matchMenuTypeaheadIn(container: HTMLElement, query: string, from: number): number {
    const items = menuItemsOf(container);
    return matchMenuTypeahead(items.map(menuItemTypeaheadText), query, from);
}

/**
 * 这次按键是否可以进打字缓冲。
 *
 * 【为什么排除这些】空格是"激活当前项"（菜单的通用约定），而带修饰键的组合是
 * 快捷键而不是打字。单字符判定必须放在 `event.key` 上而不是 `keyCode`：后者在
 * 非拉丁输入法 / 移动端键盘上不可靠。
 */
export function menuTypeaheadChar(
    key: string,
    modifiers: { ctrlKey?: boolean; metaKey?: boolean; altKey?: boolean },
): string | null {
    if (key.length !== 1) return null;
    if (key === " ") return null;
    if (modifiers.ctrlKey || modifiers.metaKey || modifiers.altKey) return null;
    return key.toLowerCase();
}
