/**
 * 文件浏览器列表的键盘导航算术。
 *
 * 【为什么抽成独立模块】`FileBrowserPanel` 的列表行此前只有指针事件
 * （`onPointerDown` / `onClick` / `onDoubleClick`），键盘用户无法到达、聚焦或
 * 打开任何一项。补上方向键导航需要"下一个活动行下标"这段纯算术，而面板本身
 * 依赖 Redux 与后端 API，整块渲染进单测既重又脆。把这段算术留在这里，
 * 面板只负责把结果映射到焦点与激活，逻辑即可被直接单测。
 *
 * 约定：活动行下标以 `-1` 表示"尚无活动行"。列表为空时任何按键都返回 `-1`。
 */

/** 参与活动行移动的按键。 */
export type FileListNavKey = "ArrowDown" | "ArrowUp" | "Home" | "End";

const NAV_KEYS: readonly FileListNavKey[] = ["ArrowDown", "ArrowUp", "Home", "End"];

/** 该按键是否用于移动活动行。 */
export function isFileListNavKey(key: string): key is FileListNavKey {
    return (NAV_KEYS as readonly string[]).includes(key);
}

/**
 * 该按键是否用于激活当前活动行。
 *
 * Enter 与空格等价（列表项没有"仅选中"的独立语义，两者都执行主操作），
 * 与 `AppContextMenu` 的键盘模型保持一致。
 */
export function isFileListActivationKey(key: string): boolean {
    return key === "Enter" || key === " ";
}

/**
 * 计算按键之后的活动行下标。
 *
 * 夹紧（clamp）而非环绕：从末行再按 ArrowDown 停在末行，从首行再按
 * ArrowUp 停在首行。尚无活动行（`current < 0`）时，ArrowDown / Home 落到
 * 首行，ArrowUp / End 落到末行。未知按键与空列表分别返回 `current` 与 `-1`。
 */
export function nextActiveIndex(current: number, key: string, count: number): number {
    if (count <= 0) return -1;
    switch (key) {
        case "ArrowDown":
            return current < 0 ? 0 : Math.min(current + 1, count - 1);
        case "ArrowUp":
            return current < 0 ? count - 1 : Math.max(current - 1, 0);
        case "Home":
            return 0;
        case "End":
            return count - 1;
        default:
            return current;
    }
}
