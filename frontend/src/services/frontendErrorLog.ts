/**
 * frontendErrorLog — 兼容门面。
 *
 * 实现已迁入 `services/frontendLog.ts`（四个级别 + 用户档位 + Off 本地短路）。
 * 保留这些 re-export 是为了不动既有 26 个 `reportFrontendError` 调用点与它们的单测；
 * 新代码应直接 import `frontendLog`。
 */
export {
    frontendLog,
    installGlobalErrorReporting,
    installFrontendLogPolicy,
    reportFrontendError,
    resetFrontendErrorDedupForTests,
    setFrontendLogPolicy,
    getFrontendLogPolicy,
    policyFromView,
    toDetailText,
    dedupDecision,
} from "./frontendLog";
export type { FrontendLevel, FrontendLogPolicy } from "./frontendLog";
