/**
 * 参数编辑器「笔画进行中」标志（带 **true→false 边沿回调**）。
 *
 * ## 为什么不用普通的 `useRef(false)`
 * 这个标志有两个消费者契约：
 *
 * 1. **读**：`usePianoRollData` / `useLoudnessCurves` 在取数**落地**前读它，决定这份
 *    回包要不要推迟（笔画期间换掉 `paramView` 或响度快照都会打断笔画）；
 * 2. **写**：`usePianoRollInteractions` 的十余处手势收尾分支把它置回 `false`
 *    （pointerup、pointercancel、各工具自己的中止路径）。
 *
 * 契约 1 的推迟必须在契约 2 的**每一次**收尾上被补触发 —— 漏掉任何一处，曲线就会停在
 * 旧数据上，直到下一次触发才恢复。把"记得在十余处逐一调用补取"当成人肉约定，迟早会漏。
 *
 * 因此这里把标志做成**受控属性**：任何 `current = false` 的写入都会自动触发一次
 * `onEnd`。补取逻辑只注册一次，收尾点有多少个都不再重要 —— 不变量被收口到写入本身。
 *
 * ## 为什么回调放在写入之后
 * 先落值再回调：`onEnd` 触发的补取链路里若有代码回读本标志，应当看到"已结束"的
 * 新值，而不是即将过期的旧值。
 */
export interface LiveEditFlag {
    current: boolean;
}

/**
 * 创建标志。
 *
 * @param onEnd `true → false` 边沿回调（笔画结束）。同一时刻只会在边沿触发一次；
 *   重复写 `false`、或写 `true` 均不触发。
 */
export function createLiveEditFlag(onEnd: () => void): LiveEditFlag {
    let value = false;
    return {
        get current(): boolean {
            return value;
        },
        set current(next: boolean) {
            if (value && !next) {
                // 先落值，再回调（见模块头说明）。
                value = false;
                onEnd();
                return;
            }
            value = next;
        },
    };
}
