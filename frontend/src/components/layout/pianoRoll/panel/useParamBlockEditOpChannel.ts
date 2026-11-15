/**
 * 块命令的**键盘通道**。
 *
 * 【职责边界】只做两件事：
 * 1. 把 stroke 折成「临时反转吸附」这一个手势事实（只有 split 需要它）；
 * 2. 把 op 交给调用方的 runOp。
 *
 * 【为什么不在这里 switch op】块命令的执行表只有一张（paramBlockEditOps.ts）。
 * 键盘与右键菜单都必须走它 —— 上一版在这里和菜单各写一份 switch，于是
 * selectAll / deselect 只在键盘侧存在，而整个键盘侧因为通道判据写错而静默失效。
 */
import { useEffect } from "react";
import {
    firstBinding,
    isModifierActive,
    isNoneBinding,
} from "../../../../features/keybindings/keybindingsSlice";
import type { ParamBlockEditOp } from "../../../../features/keybindings/focusRouting";
import type { KeybindingMap } from "../../../../features/keybindings/types";

/** 键盘通道订阅的事件名（App.tsx 的块动作分支派发到它）。 */
export const PARAM_BLOCK_EDIT_OP_EVENT = "hifi:paramBlockEditOp";

export function useParamBlockEditOpChannel(args: {
    readonly keybindings: KeybindingMap;
    readonly runOp: (op: ParamBlockEditOp, context: { readonly snapToggleHeld: boolean }) => void;
}): void {
    const { keybindings, runOp } = args;
    useEffect(() => {
        const handler = (event: Event) => {
            const detail = (
                event as CustomEvent<{
                    op?: ParamBlockEditOp;
                    stroke?: {
                        ctrl: boolean;
                        shift: boolean;
                        alt: boolean;
                        meta: boolean;
                    };
                }>
            ).detail;
            if (detail?.op === undefined) return;
            /*
             * 切分不是拖拽手势，但"临时反转吸附"对磁吸同样有意义，
             * 而且与拖动块边缘 / 新建块的 XOR 语义一致。
             */
            const noSnapKb = firstBinding(keybindings["modifier.clipNoSnap"]);
            const stroke = detail.stroke;
            const snapToggleHeld =
                noSnapKb !== undefined &&
                !isNoneBinding(noSnapKb) &&
                stroke !== undefined &&
                isModifierActive(noSnapKb, {
                    ctrlKey: stroke.ctrl,
                    shiftKey: stroke.shift,
                    altKey: stroke.alt,
                    metaKey: stroke.meta,
                });
            runOp(detail.op, { snapToggleHeld });
        };
        window.addEventListener(PARAM_BLOCK_EDIT_OP_EVENT, handler);
        return () => window.removeEventListener(PARAM_BLOCK_EDIT_OP_EVENT, handler);
    }, [keybindings, runOp]);
}
