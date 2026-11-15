/**
 * 波纹编辑（自动跟进）的**模式词汇表** —— 纯函数，无 DOM、无 React。
 *
 * 【主要内容】
 * - `RippleScope` / `RIPPLE_SCOPES`：用户可选的两个**方向**（按轨道 / 全部轨道）；
 * - `RippleMode`：引擎认的**生效模式**三值（off / track / all）；
 * - `resolveRippleMode`：把"开关 × 方向"折算成生效模式；
 * - `hydrateRippleState`：把持久化设置（含旧版三值编码）折算回开关与方向。
 *
 * 【为什么要区分"方向"与"模式"】
 * 旧版把开关与方向塞进同一个三值字符串 `rippleMode: "off" | "track" | "all"`，
 * 于是工具栏左键只能在三者之间循环 —— 想从"全部轨道"关掉，必须先绕到"按轨道"。
 * 现在拆成两根**正交**的轴（与相邻的"自动滚屏"同构）：
 * - 开关（`rippleEnabled`）决定"要不要跟进"；
 * - 方向（`rippleScope`）决定"跟进谁"，且**关闭时仍然记得**上次选的方向。
 * 引擎那一侧不需要知道这个拆分：它只吃 `RippleMode`，由 `resolveRippleMode` 折算。
 *
 * 【与其他模块的关系】
 * - 上游：`ActionBar` 的波纹按钮（左键开关 / 右键选方向）；
 * - 下游：`ripplePreview.buildRippleFollowers`（拖拽预览）与后端 `ripple_settings`
 *   （提交时的权威行为）—— 两者必须拿到**同一个**生效模式，否则预览与提交会跳变；
 * - 持久化：`SessionState.rippleEnabled` / `rippleScope` → `UiSettings` 同名字段。
 */

/**
 * 波纹编辑的**生效模式**（三值）。
 *
 * 【为什么保留 `off`】它是后端内核的词汇表（`state::RippleMode` 的三个变体，
 * 对应 REAPER 的 Ripple Editing），也是持久化里"关闭"的唯一表达。UI 上它不再是一个
 * 可选项 —— 那是开关的职责。
 */
export type RippleMode = "off" | "track" | "all";

/** 用户可选的两个方向。取值顺序即菜单顺序。 */
export const RIPPLE_SCOPES = ["track", "all"] as const;
export type RippleScope = (typeof RIPPLE_SCOPES)[number];

/**
 * 出厂方向：按轨道。
 *
 * 【为什么是它】旧版三值循环里"关闭"的下一个状态就是"按轨道"，用户点开开关时
 * 期待的就是这个最小作用范围；"全部轨道"会一次性推动所有轨道，属于需要显式选择的
 * 更大动作，不该是默认。
 */
export const DEFAULT_RIPPLE_SCOPE: RippleScope = "track";

/** 取值是否合法（reducer 的写入白名单与设置水合共用）。 */
export function isRippleScope(value: unknown): value is RippleScope {
    return (RIPPLE_SCOPES as readonly unknown[]).includes(value);
}

/**
 * 归一化方向值。
 *
 * 【为什么非法值回落而不是抛错】设置来自磁盘 / 插件形态 / 手改配置，脏值必须让应用
 * 照常启动；回落"按轨道"等价于"没设过"。
 */
export function normalizeRippleScope(value: unknown): RippleScope {
    return isRippleScope(value) ? value : DEFAULT_RIPPLE_SCOPE;
}

/** 波纹编辑的两根轴。 */
export interface RippleState {
    enabled: boolean;
    scope: RippleScope;
}

/**
 * 把持久化设置折算成"开关 + 方向"。
 *
 * 【流程】
 * 1. `rippleEnabled` 存在 → 它就是开关；方向取 `rippleMode`（非法值回落按轨道）。
 * 2. `rippleEnabled` 缺失 → **旧版三值编码**：`"track"` / `"all"` 说明当时是开着的，
 *    `"off"` / 未知值说明是关着的；方向由 `"all"` 决定。
 *
 * 【为什么必须自己认旧编码】后端 `normalize_ripple_mode` 已经做了同一件事，但前端
 * 还会遇到第三种来源：浏览器 mock、旧版插件形态、以及测试夹具。水合这一层是
 * "设置进 Redux 之前的最后一道闸"，它认不出旧编码就会把老用户的"开着"读成"关着"。
 */
export function hydrateRippleState(input: {
    rippleEnabled?: unknown;
    rippleMode?: unknown;
}): RippleState {
    const scope = input.rippleMode;
    if (input.rippleEnabled == null) {
        return {
            enabled: scope === "track" || scope === "all",
            scope: scope === "all" ? "all" : DEFAULT_RIPPLE_SCOPE,
        };
    }
    return { enabled: input.rippleEnabled === true, scope: normalizeRippleScope(scope) };
}

/**
 * 折算成引擎认的生效模式。
 *
 * 【唯一入口】拖拽预览与后端提交必须走同一条折算，否则会出现"预览按全部轨道推、
 * 提交按按轨道推"这种两边都自洽、合起来错位的结果。
 */
export function resolveRippleMode(enabled: boolean, scope: RippleScope): RippleMode {
    return enabled ? scope : "off";
}
