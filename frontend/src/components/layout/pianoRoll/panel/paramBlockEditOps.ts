/**
 * 块命令的**唯一执行表**。
 *
 * 【为什么必须只有一张】键盘通道（hifi:paramBlockEditOp）与右键菜单
 * （ParamBlockContextMenu）是同一个动作集合的两个入口。上一版各写一份 switch，
 * 结果 selectAll / deselect 只在键盘侧存在；而键盘侧因为通道判据写错整条失效，
 * 于是「菜单能用、快捷键不行」—— 两边都各自「看起来对」。
 *
 * 【op 的来源】ParamBlockEditOp 是快捷键通道的契约（见 focusRouting）；
 * createFromSelection 只有菜单入口（没有默认快捷键），因此并进本表但不出现在
 * PARAM_BLOCK_ACTION_TO_OP 里。
 */
import type { ParamBlockEditOp } from "../../../../features/keybindings/focusRouting";

/** 键盘通道与右键菜单的并集：菜单多一个「用当前选区新建块」。 */
export type ParamBlockMenuOp = ParamBlockEditOp | "createFromSelection";

/** 每个 op 的实现。调用方（面板）用 ref 转发，避免声明顺序依赖。 */
export interface ParamBlockOpHandlers {
    readonly split: (context: { readonly snapToggleHeld: boolean }) => void;
    readonly createFromSelection: () => void;
    readonly group: () => void;
    readonly ungroup: () => void;
    readonly delete: () => void;
    readonly selectAll: () => void;
    readonly deselect: () => void;
    readonly restoreGame: () => void;
    readonly restoreDeleted: () => void;
    readonly normalizePitch: () => void;
}

/** 表里登记的全部 op（测试与菜单构造用；顺序即菜单顺序）。 */
export const PARAM_BLOCK_MENU_OPS: readonly ParamBlockMenuOp[] = [
    "split",
    "createFromSelection",
    "group",
    "ungroup",
    "delete",
    "selectAll",
    "deselect",
    "restoreGame",
    "restoreDeleted",
    "normalizePitch",
];

/**
 * 执行一个块命令。
 *
 * 【这是全仓唯一的 op → 实现 映射】任何新命令都必须加在这里；漏了会在
 * `const exhaustive: never = op` 处编译失败。
 */
export function runParamBlockMenuOp(
    op: ParamBlockMenuOp,
    handlers: ParamBlockOpHandlers,
    context: { readonly snapToggleHeld: boolean },
): void {
    switch (op) {
        case "split":
            handlers.split(context);
            return;
        case "createFromSelection":
            handlers.createFromSelection();
            return;
        case "group":
            handlers.group();
            return;
        case "ungroup":
            handlers.ungroup();
            return;
        case "delete":
            handlers.delete();
            return;
        case "selectAll":
            handlers.selectAll();
            return;
        case "deselect":
            handlers.deselect();
            return;
        case "restoreGame":
            handlers.restoreGame();
            return;
        case "restoreDeleted":
            handlers.restoreDeleted();
            return;
        case "normalizePitch":
            handlers.normalizePitch();
            return;
    }
    /* 穷举闸门：新增一个 op 而忘了实现，这里编译失败。 */
    const exhaustive: never = op;
    void exhaustive;
}
