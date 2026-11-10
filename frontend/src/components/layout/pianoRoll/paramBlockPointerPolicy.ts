/**
 * 参数块指针意图的纯策略层。
 *
 * 这个模块只回答「按钮 + 命中目标 + 修饰键」应该落到哪一种手势，
 * 不读 DOM、不读 React、不修改任何状态。把它从 1500 行的 pointerdown
 * 回调里抽出来后，Alt 的四种组合（块上点击 / 空白拖拽 / Ctrl+Alt
 * 拉平 / Ctrl 多选）可以用一张表穷举测试。
 */

export type ParamBlockPointerTarget =
    | {
          readonly kind: "transition";
          readonly leftId: string;
          readonly rightId: string;
          readonly logicalId?: string;
      }
    | {
          readonly kind: "edge";
          readonly blockId: string;
          readonly logicalId?: string;
          readonly edge: "left" | "right";
      }
    | { readonly kind: "block"; readonly blockId: string; readonly logicalId: string }
    | { readonly kind: "blank" };

export type ParamBlockBlockTarget = Extract<ParamBlockPointerTarget, { kind: "block" }>;
export type ParamBlockEdgeTarget = Extract<ParamBlockPointerTarget, { kind: "edge" }>;
export type ParamBlockTransitionTarget = Extract<ParamBlockPointerTarget, { kind: "transition" }>;
export type ParamBlockClickMode = "replace" | "toggle" | "range";

export type ParamBlockPointerIntent =
    | { readonly kind: "block-context-menu"; readonly target: ParamBlockBlockTarget }
    | { readonly kind: "block-morph"; readonly target: ParamBlockBlockTarget }
    | {
          readonly kind: "block-split";
          readonly target: ParamBlockBlockTarget;
          readonly frame: number;
      }
    | {
          readonly kind: "block-select-or-drag";
          readonly target: ParamBlockBlockTarget;
          readonly mode: ParamBlockClickMode;
      }
    | { readonly kind: "edge-resize"; readonly target: ParamBlockEdgeTarget }
    | { readonly kind: "transition-resize"; readonly target: ParamBlockTransitionTarget }
    | { readonly kind: "background-context-menu" }
    | { readonly kind: "background-range"; readonly mode: "replace" | "add" }
    | { readonly kind: "background-create"; readonly flatten: boolean }
    | { readonly kind: "ignore" };

export interface ParamBlockPointerPolicyInput {
    readonly button: number;
    readonly target: ParamBlockPointerTarget;
    /** 按下时「在点击位置分割」修饰键是否生效。 */
    readonly splitAtClick: boolean;
    /** 按下时「空白拖拽新建」修饰键是否生效。 */
    readonly createBlock: boolean;
    /** 按下时「Ctrl+Alt 空白拖拽新建并拉平」是否生效。 */
    readonly createFlat: boolean;
    /** 按下时多选修饰键是否生效。 */
    readonly multiSelect: boolean;
    /** 按下时范围选择修饰键是否生效。 */
    readonly rangeSelect: boolean;
    readonly frame: number;
}

function blockTarget(target: ParamBlockPointerTarget): ParamBlockBlockTarget | null {
    if (target.kind === "block") return target;
    if (target.kind === "edge") {
        return {
            kind: "block",
            blockId: target.blockId,
            logicalId: target.logicalId ?? target.blockId,
        };
    }
    if (target.kind === "transition") {
        return {
            kind: "block",
            blockId: target.leftId,
            logicalId: target.logicalId ?? target.leftId,
        };
    }
    return null;
}

/**
 * 优先级（与计划 3.3 的矩阵一致）：
 *
 * 1. 右键块/边缘/过渡 → 幅度手势（手势内部再分单击菜单 / 拖动幅度）；
 * 2. 右键空白 → 追加选区（手势内部再分单击菜单 / 拖动范围）；
 * 3. 左键块 + createFlat → 按 Ctrl 选择（createFlat 只对空白生效）；
 * 4. 左键块/边缘 + splitAtClick → 分割候选；
 * 5. 左键过渡 / 边缘 → 手柄手势；
 * 6. 左键块 → 选择或拖动；
 * 7. 左键空白 + createFlat/create → 创建（createFlat 优先）；
 * 8. 左键空白 → 框选（Ctrl 追加，否则替换）。
 */
export function resolveParamBlockPointerIntent(
    input: ParamBlockPointerPolicyInput,
): ParamBlockPointerIntent {
    const { button, target } = input;
    if (button === 2) {
        if (target.kind === "blank") return { kind: "background-range", mode: "add" };
        const block = blockTarget(target);
        return block === null ? { kind: "ignore" } : { kind: "block-morph", target: block };
    }
    if (button !== 0) return { kind: "ignore" };

    if (target.kind === "blank") {
        if (input.createFlat) return { kind: "background-create", flatten: true };
        if (input.createBlock) return { kind: "background-create", flatten: false };
        return { kind: "background-range", mode: input.multiSelect ? "add" : "replace" };
    }

    if (target.kind === "block" && input.createFlat) {
        return {
            kind: "block-select-or-drag",
            target,
            mode: input.rangeSelect ? "range" : input.multiSelect ? "toggle" : "replace",
        };
    }

    if ((target.kind === "block" || target.kind === "edge") && input.splitAtClick) {
        const block = blockTarget(target)!;
        return { kind: "block-split", target: block, frame: input.frame };
    }

    if (target.kind === "transition") return { kind: "transition-resize", target };
    if (target.kind === "edge") return { kind: "edge-resize", target };

    return {
        kind: "block-select-or-drag",
        target,
        mode: input.rangeSelect ? "range" : input.multiSelect ? "toggle" : "replace",
    };
}
