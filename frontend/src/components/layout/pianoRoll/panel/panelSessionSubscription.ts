/**
 * 面板的会话订阅口径：高频字段剔除表 + 引用比较器。
 *
 * 【为什么单独成文件】usePianoRollPanelModel 已在超大文件预算内（见
 * tests/fileSizeBudget.test.ts），这块与模型主体无关的纯订阅口径先搬出来。
 */
import type { RootState } from "../../../../app/store";

/** 空选中集合的共享实例：避免每次渲染换引用让 `paramBlockSpec` 的 memo 白跑一次。 */
export const EMPTY_BLOCK_ID_SET: ReadonlySet<string> = new Set<string>();

/**
 * 高频会话字段：它们由 `PianoRollPlayheadBridge` 单独订阅，面板主体不因它们重渲染。
 *
 * 【为什么必须从面板选择器里剔除】`playheadSec` 与 `runtime` 在播放时约 30Hz 更新；
 * 面板有数百个 hook / 子组件，若随每次轮询重渲染，60Hz 的光标与自动滚屏会被 React
 * 提交拖慢。这里只忽略它们，其余会话字段仍按引用浅比较。
 */
const PIANO_ROLL_VOLATILE_SESSION_KEYS = new Set<keyof RootState["session"]>([
    "playheadSec",
    "playheadSampledAtMs",
    "runtime",
]);

export function pianoRollSessionEqual(
    left: RootState["session"],
    right: RootState["session"],
): boolean {
    if (left === right) return true;
    for (const key of Object.keys(left) as Array<keyof RootState["session"]>) {
        if (PIANO_ROLL_VOLATILE_SESSION_KEYS.has(key)) continue;
        if (left[key] !== right[key]) return false;
    }
    return true;
}
