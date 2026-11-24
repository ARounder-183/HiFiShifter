/**
 * 移动端「菜单」sheet 的关闭判据：点完一项之后该不该收起整层。
 *
 * 【为什么需要一条判据而不是"点了就关"】菜单里有三种条目，行为必须不同：
 * 1. **普通项**（打开对话框、切换显示项）—— 结果在别处（portal 的对话框或画布），
 *    继续盖着一层全屏 sheet 只会挡住用户刚打开的东西，必须收起；
 * 2. **子菜单触发器**（视图 → 窗口 / 布局、最近工程、拉伸算法…）—— 它只是展开
 *    下一级。收起 sheet 会把刚展开的那一级一起藏掉，用户再也点不到；
 * 3. **禁用项** —— 点了什么也不会发生，收起 sheet 等于"点一下把菜单弄没了"，
 *    用户会以为是自己点错了。
 *
 * 【为什么用 DOM 判据而不是让每个菜单项自己声明】菜单项来自 \`MenuBar\`（几十处
 * \`DropdownMenu.Item\`），给它加参数意味着每一处都要改一次，且新增项默认漏掉。
 * Radix 已经在 DOM 上写明了语义：\`role="menuitem*"\` 是条目，\`aria-haspopup="menu"\`
 * 是子菜单触发器，\`data-disabled\` 是禁用。读这三条即可，无需改动菜单本身。
 *
 * 【为什么监听 document 而不是 sheet 的 onClick】菜单内容由 Radix **portal 到
 * body**，根本不在 sheet 的 DOM 子树里 —— 挂在 sheet 上的处理器收不到二级菜单
 * 里的点击（"视图 → 布局 → 重置布局"正是这种三级路径）。
 */

/** 菜单条目的 ARIA 角色（Radix 的三种 item 形态）。 */
const MENU_ITEM_SELECTOR = '[role="menuitem"], [role="menuitemradio"], [role="menuitemcheckbox"]';

/**
 * 这次点击是否意味着"菜单用完了，可以收起"。
 *
 * @param target 事件目标（\`event.target\`）。非元素（文本节点等）一律返回 \`false\`。
 */
export function menuSelectionClosesSheet(target: EventTarget | null): boolean {
    if (target === null || typeof target !== "object") return false;
    const element = target as Element;
    if (typeof element.closest !== "function") return false;
    const item = element.closest(MENU_ITEM_SELECTOR);
    if (item === null) return false;
    // 子菜单触发器：还要继续选下一级。
    if (item.getAttribute("aria-haspopup") === "menu") return false;
    // 禁用项：没有结果可看，别把菜单收走。
    if (item.hasAttribute("data-disabled")) return false;
    if (item.getAttribute("aria-disabled") === "true") return false;
    return true;
}
