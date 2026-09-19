import type { ParamName } from "./types.js";

/**
 * 切换某个副参数曲线的显示状态（键存在 = 显示）。
 *
 * @param current 当前可见性表。
 * @param param 目标副参数。
 * @returns 新的可见性表（不改原对象）。
 */
export function toggleSecondaryParamVisibility(
    current: Partial<Record<ParamName, boolean>>,
    param: ParamName,
): Partial<Record<ParamName, boolean>> {
    if (current[param]) {
        const next = { ...current };
        delete next[param];
        return next;
    }
    return { ...current, [param]: true };
}

/**
 * 列出应当显示为**副曲线**的参数 id。
 *
 * 音高模式下只能有处理器参数；其余参数模式下音高本身可以当副曲线显示。
 * 当前正在编辑的参数一律排除（它就是主曲线）。
 *
 * @param args 当前编辑参数、处理器参数列表与可见性表。
 * @returns 需要显示的副参数 id 序列（保持候选顺序）。
 */
export function getVisibleSecondaryParamIds(args: {
    editParam: ParamName;
    processorParamIds: ParamName[];
    secondaryParamVisible: Partial<Record<ParamName, boolean>>;
}): ParamName[] {
    const { editParam, processorParamIds, secondaryParamVisible } = args;
    const candidateIds =
        editParam === "pitch" ? processorParamIds : ["pitch", ...processorParamIds];

    return candidateIds.filter(
        (paramId) => paramId !== editParam && secondaryParamVisible[paramId] === true,
    );
}

/**
 * 合并「原始」与「编辑」两条曲线为一条叠加显示序列（编辑值优先，缺失处回退原始值）。
 *
 * 【为什么不能复用共享数组（这是一个真实缺陷的根因）】
 * 本函数此前把结果写进模块级数组并**原样返回引用**，注释称之为"零分配优化"。
 * 但调用方（`PianoRollPanel.buildCurveLayers`）是把返回值**存进图层描述符**、
 * 稍后才消费的——GL 路径的投影发生在 `drawGlCurves` 里，晚于整个图层列表构建。
 * 于是所有参考线 / 副参数图层共享同一个数组，投影时读到的**全是最后一次写入的
 * 内容**：屏幕上表现为"多条未被选中的参数线画出了同一条别的线的数据"，
 * 且随图层构建顺序变化（用户报告为随机的渲染错乱）。
 *
 * 调用方里只有 `render.ts` 的 Canvas2D 路径是**立即消费**的，共享数组在那里恰好
 * 无害——这正是缺陷能长期潜伏的原因：一条路径掩盖了另一条路径的错误。
 *
 * 因此改为每次返回**新数组**。这是渲染层每帧几次的小数组分配，与"投影一条上万点
 * 的曲线"相比可以忽略；用正确性换这点分配是不划算的。
 *
 * @param args 原始曲线与编辑曲线（长度可不同，短的一侧按缺失处理）。
 * @returns 新数组，长度 = `max(orig.length, edit.length)`；每个位置取有限的编辑值，
 *   否则取有限的原始值，两者都非有限时取 0。
 */
export function resolveSecondaryOverlayValues(args: { orig: number[]; edit: number[] }): number[] {
    const { orig, edit } = args;
    const length = Math.max(orig.length, edit.length);
    const values = new Array<number>(length);

    for (let index = 0; index < length; index += 1) {
        const editValue = edit[index];
        const origValue = orig[index];
        const hasEditValue = Number.isFinite(editValue);
        const hasOrigValue = Number.isFinite(origValue);

        if (hasEditValue) {
            values[index] = editValue;
        } else if (hasOrigValue) {
            values[index] = origValue;
        } else {
            values[index] = 0;
        }
    }

    return values;
}
