/**
 * 笔画期间**被推迟的刷新**台账（`paramView` 的后端驱动变更）。
 *
 * ## 为什么要有这份台账
 *
 * 笔画进行中绝不能换掉 `paramView`：live 覆盖层一旦失去窗口，正在画的轨迹就会
 * 消失（见 docs/plans/2026-09-26-volume-dyn-drag-trail-fix.md）。因此所有
 * 后端驱动的变更入口（取数发起 / 落地、显式立即取数、`paramsEpoch` 触发的重取）
 * 都改为"**只记待办、不做动作**"，等 pointer-up 由
 * `usePianoRollData.notifyLiveEditEnded()` 统一补触发。
 *
 * ## 为什么要区分两类待办
 *
 * 补触发的力度不同，混为一谈会修错一半：
 *
 * - `force`：**后端数据本身变了**（音高 / 动态基线分析完成、任何 timeline 更新）。
 *   补触发必须**强制**取数 —— 取数入口有"视口已覆盖就不取"的短路，它会误判
 *   "没变"而跳过，于是画面停在旧数据上；
 * - `plain`：**视口 / 令牌变了**（滚动、缩放）。普通重取即可，短路会正确地
 *   判断"这份窗口不需要重取"。
 *
 * ## 为什么独立成模块
 *
 * 与 `liveEditFlag` 同一理由：这份契约靠"每个调用点都记得"是守不住的，必须
 * 能被单测钉住（见 liveEditDeferral.test.ts）。台账本身不依赖 React / DOM，
 * 活跃判定由调用方以 thunk 注入。
 */
export interface DeferredRefresh {
    /** 后端数据变了 → 补触发时必须强制取数。 */
    force: boolean;
    /** 视口 / 令牌变了 → 普通重取即可。 */
    plain: boolean;
}

export interface LiveEditDeferral {
    /**
     * 记一次"笔画期间发生的变更"。
     *
     * @param opts.force 见模块头说明（数据变了 vs 视口变了）。
     * @returns `true` = 已推迟，调用方**必须立即 return**、不做任何实际动作；
     *   `false` = 不在笔画中，调用方照常执行。
     */
    defer(opts?: { force?: boolean }): boolean;
    /**
     * 笔画结束：取出并清空待办。
     *
     * @returns `null` = 没有待办（不需要补触发）；否则是两类待办的并集。
     */
    take(): DeferredRefresh | null;
}

/**
 * 创建台账。
 *
 * @param isLiveEditActive 笔画活跃判定（每次 `defer` 时读取，故不必关心闭包时效）。
 */
export function createLiveEditDeferral(isLiveEditActive: () => boolean): LiveEditDeferral {
    let force = false;
    let plain = false;
    return {
        defer(opts) {
            if (!isLiveEditActive()) return false;
            if (opts?.force === true) force = true;
            else plain = true;
            return true;
        },
        take() {
            if (!force && !plain) return null;
            const pending: DeferredRefresh = { force, plain };
            // 取出即清空：一次补触发覆盖全部待办，重复补触发是无意义的重复取数。
            force = false;
            plain = false;
            return pending;
        },
    };
}
