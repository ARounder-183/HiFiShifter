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

// ── 输入字母快速跳转（type-ahead，与资源管理器一致） ─────────────────────

/**
 * 连续击键被视为同一次输入的时间窗（毫秒）。窗口内的击键逐字符累积成前缀
 * （`f` → `fa`），超时后的击键重新从单字符开始。资源管理器的同款行为约为 1 秒。
 */
export const TYPE_AHEAD_RESET_MS = 1000;

/**
 * 从 `start`（含）开始环绕查找第一个以 `query` 开头的名字（不区分大小写）。
 *
 * 环绕意味着「从当前活动行的下一行开始找，找不到再回到列表开头」——这正是
 * 资源管理器的行为：已有选中项时输入字母跳到**下一个**匹配项；没有选中项
 * （`start = 0`）时跳到**第一个**匹配项。`start` 超界自动回绕；空列表、空
 * 查询或无匹配返回 `-1`。
 */
export function findTypeAheadIndex(names: readonly string[], query: string, start: number): number {
    const needle = query.toLowerCase();
    if (!needle || names.length === 0) return -1;
    for (let step = 0; step < names.length; step++) {
        const index = (start + step) % names.length;
        if (names[index].toLowerCase().startsWith(needle)) return index;
    }
    return -1;
}

export interface TypeAheadResult {
    /** 命中的行下标；`null` = 无匹配，活动行保持原地不动。 */
    index: number | null;
    /** 本次击键之后的输入缓冲（命中为累积前缀；无匹配保持原值）。 */
    buffer: string;
}

/**
 * 计算一次击键之后的跳转目标与新输入缓冲。
 *
 * 算法与资源管理器的增量搜索一致：
 * 1. 先拿累积前缀（`buffer + key`）从活动行之后环绕查找；
 * 2. 命中即跳转；未命中且**本次缓冲恰好是同一个字母连按**（`f` 再按 `f` →
 *    `ff` 无匹配）时，退回单字母再找一遍 —— 这就是「连续按同一个字母在
 *    同前缀文件间循环」的行为；
 * 3. 仍无匹配则不跳转（`index: null`），缓冲**保持原值**：失败的前缀不可能靠
 *    继续延长复活（`fa` 无匹配则 `fab…` 亦无匹配），保留旧缓冲让下一次击键
 *    从上一个有效前缀重新组合，失败按键不会污染后续输入。
 *
 * 时间窗（超时清空缓冲）由调用方负责：这里只做纯算术，才能脱离计时器单测。
 */
export function nextTypeAhead(
    names: readonly string[],
    buffer: string,
    key: string,
    activeIndex: number,
): TypeAheadResult {
    const candidate = buffer + key;
    const start = activeIndex + 1;
    const index = findTypeAheadIndex(names, candidate, start);
    if (index >= 0) return { index, buffer: candidate };
    if (candidate.length > 1 && buffer === key) {
        const retry = findTypeAheadIndex(names, key, start);
        if (retry >= 0) return { index: retry, buffer: key };
    }
    return { index: null, buffer };
}
