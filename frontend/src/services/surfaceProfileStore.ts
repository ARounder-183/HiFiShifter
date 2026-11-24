/**
 * 形态剖面的进程内副本：**同步可读、异步初始化**。
 *
 * 【为什么需要一份"副本"而不是每次问后端】形态决定了太多**模块加载期**就要定型的
 * 东西：停靠布局的出厂值、控件密度、CSS 令牌层、输入剖面。这些读取点全是同步的
 * （见 `mount.tsx` 的说明），改成异步会迫使它们全部重排时序 —— 而那正是本仓在
 * `uiStorage` 上已经吃过一次的亏。
 *
 * 因此形态与偏好走同一条路：`main.tsx` 在**动态 import 应用之前**把数据灌进来，
 * 之后任何模块读到的都是已经就位的值。
 *
 * 【为什么后备值是"桌面"】这是零回归的锚点：后端不可用（纯浏览器预览、mock 后端
 * 未安装、命令失败）时，一切按桌面行为运行 —— 与引入本模块之前**逐字节相同**。
 * 反过来（后备成移动）会让开发者在浏览器里莫名其妙地看到移动外壳。
 */

/** 后端声明的形态。插件形态由 `hostDetect` 叠加，不在这里。 */
export type DeclaredForm = "desktop" | "mobile";

export type SurfaceOs = "windows" | "macos" | "linux" | "android" | "ios";

/** 能力判定结果。与后端 `CapabilityDecision` 同形。 */
export interface SurfaceDecision {
    allowed: boolean;
    /** 不可用时的理由键；可用时为 `null`。 */
    reason: string | null;
}

/** 后端 `get_surface_profile` 的返回形状（见 `platform/mod.rs`）。 */
export interface SurfaceProfile {
    form: DeclaredForm;
    os: SurfaceOs;
    input: {
        primary: "pointer" | "touch";
        hover: boolean;
        wheel: boolean;
        secondary_button: boolean;
        modifier_keys: boolean;
        pressure: boolean;
        multi_touch: boolean;
    };
    viewport: {
        safe_area: boolean;
        /**
         * 真实的系统栏内边距（CSS px）。
         *
         * 【为什么后端要给这个】Android WebView 的 `env(safe-area-inset-*)` 只反映
         * 刘海，不反映状态栏/导航栏 —— 而 `enableEdgeToEdge()` 让内容画到它们底下。
         * 真实值只能从原生的 `WindowInsets` 取（见 `services/surfaceInsets.ts`）。
         */
        insets: { top: number; right: number; bottom: number; left: number };
        resizable: boolean;
        multi_window: boolean;
    };
    storage: {
        path_access: boolean;
        sandboxed: boolean;
        import_copies: boolean;
        reveal_in_folder: boolean;
        share_sheet: boolean;
    };
    audio: {
        output: boolean;
        microphone: boolean;
        background: boolean;
        focus_events: boolean;
    };
    capabilities: Record<string, SurfaceDecision>;
}

/**
 * 后备剖面 = **桌面**。
 *
 * 【为什么手写一份而不是留 `null`】`null` 会让每个读取点都要判空，而"判空后的
 * 默认值"仍然要写一份 —— 只是散在十几处。集中成一份常量，语义只有一条：
 * 没有剖面信息时按桌面处理。
 */
export const DESKTOP_FALLBACK_PROFILE: SurfaceProfile = {
    form: "desktop",
    os: "windows",
    input: {
        primary: "pointer",
        hover: true,
        wheel: true,
        secondary_button: true,
        modifier_keys: true,
        pressure: true,
        multi_touch: true,
    },
    viewport: {
        safe_area: false,
        insets: { top: 0, right: 0, bottom: 0, left: 0 },
        resizable: true,
        multi_window: true,
    },
    storage: {
        path_access: true,
        sandboxed: false,
        import_copies: false,
        reveal_in_folder: true,
        share_sheet: true,
    },
    audio: { output: true, microphone: true, background: false, focus_events: false },
    capabilities: {},
};

let current: SurfaceProfile = DESKTOP_FALLBACK_PROFILE;
let hydrated = false;
const listeners = new Set<() => void>();

/** 当前剖面。**永远返回一个对象**，未初始化时是桌面后备值。 */
export function surfaceProfile(): SurfaceProfile {
    return current;
}

/** 是否已经拿到后端剖面（后备值不算）。 */
export function isSurfaceProfileHydrated(): boolean {
    return hydrated;
}

/**
 * 覆盖剖面。
 *
 * 【为什么允许覆盖】两条真实用途：单测要模拟移动形态；`?form=mobile` 调试入口要在
 * 桌面上复现移动端（见 `resolveFormOverride`）。生产路径上只有 `hydrate` 调它。
 */
export function setSurfaceProfile(next: SurfaceProfile): void {
    current = next;
    hydrated = true;
    for (const listener of listeners) listener();
}

/** 订阅（`useSyncExternalStore` 的形状）。 */
export function subscribeSurfaceProfile(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
        listeners.delete(listener);
    };
}

/** 测试用：把剖面重置回"未初始化"。 */
export function resetSurfaceProfileForTests(): void {
    current = DESKTOP_FALLBACK_PROFILE;
    hydrated = false;
}

/**
 * `?form=mobile` 调试覆盖。
 *
 * 【为什么需要它】移动端的绝大多数行为差异（外壳、令牌、停靠剖面、命中半径）
 * 与"设备"无关，只与"形态"有关。有了这个入口，桌面上就能用同一套代码路径跑移动
 * 形态 —— 否则每一处改动都要等真机，回归周期会从分钟级变成小时级。
 *
 * 只接受 `mobile`；其它值（含 `desktop`）一律不覆盖，避免"一个拼错的参数把界面
 * 变成另一种形态"这种只在某人机器上复现的故障。
 */
export function formOverrideFromLocation(search: string): DeclaredForm | null {
    const value = new URLSearchParams(search).get("form");
    return value === "mobile" ? "mobile" : null;
}

/**
 * 灌入后端剖面。失败时**保持桌面后备值**并留一条 debug 日志。
 *
 * 【为什么失败不抛】这是启动路径上的一步：抛出去会让整个应用挂不起来，而"形态拿不到"
 * 的合理降级恰恰就是"按桌面跑"。真正的移动端一定有后端，因此这条降级只影响
 * 浏览器预览与 mock 环境。
 */
export async function hydrateSurfaceProfile(fetcher: () => Promise<unknown>): Promise<void> {
    try {
        const raw = await fetcher();
        const profile = normalizeSurfaceProfile(raw);
        const override = formOverrideFromLocation(window.location.search);
        setSurfaceProfile(override ? { ...profile, form: override } : profile);
    } catch (error) {
        // 不 console.warn：浏览器预览下每次启动都会有一条，而它不是问题。
        void error;
        const override = formOverrideFromLocation(window.location.search);
        if (override) setSurfaceProfile({ ...DESKTOP_FALLBACK_PROFILE, form: override });
    }
}

/**
 * 把后端的原始 JSON 收窄成 `SurfaceProfile`。
 *
 * 【为什么逐字段校验而不是直接断言】这份数据跨越 IPC 边界，形状由后端决定。断言
 * 意味着"后端哪天改了字段名，前端会在某个 `.hover` 上读到 `undefined`" —— 而
 * `undefined` 在布尔上下文里是 `false`，症状会是"桌面上的悬停忽然不工作了"，
 * 与根因（一个字段名）之间隔着整个 UI。逐字段兜底到后备值，症状就变成"形态没生效"，
 * 一眼能看出是剖面没拿到。
 */
export function normalizeSurfaceProfile(raw: unknown): SurfaceProfile {
    if (!raw || typeof raw !== "object") return DESKTOP_FALLBACK_PROFILE;
    const value = raw as Partial<SurfaceProfile>;
    const form: DeclaredForm = value.form === "mobile" ? "mobile" : "desktop";
    const os: SurfaceOs =
        value.os === "macos" ||
        value.os === "linux" ||
        value.os === "android" ||
        value.os === "ios" ||
        value.os === "windows"
            ? value.os
            : "windows";
    const fallback = DESKTOP_FALLBACK_PROFILE;
    const capabilities: Record<string, SurfaceDecision> = {};
    const rawCaps = (value.capabilities ?? {}) as Record<string, unknown>;
    for (const [id, entry] of Object.entries(rawCaps)) {
        if (!entry || typeof entry !== "object") continue;
        const decision = entry as { allowed?: unknown; reason?: unknown };
        capabilities[id] = {
            allowed: decision.allowed === true,
            reason: typeof decision.reason === "string" ? decision.reason : null,
        };
    }
    /*
     * `insets` 逐字段兜底而不是整体替换：后端少给一个边（或旧后端根本没有这个
     * 字段）时，其余三边仍然是有效值。整体替换会让"缺一个字段"退化成"安全区全丢"。
     */
    const rawInsets = (value.viewport?.insets ?? {}) as Record<string, unknown>;
    const inset = (candidate: unknown): number => {
        const number = typeof candidate === "number" ? candidate : Number(candidate);
        return Number.isFinite(number) && number > 0 ? number : 0;
    };
    const insets = {
        top: inset(rawInsets.top),
        right: inset(rawInsets.right),
        bottom: inset(rawInsets.bottom),
        left: inset(rawInsets.left),
    };
    return {
        form,
        os,
        input: { ...fallback.input, ...(value.input ?? {}) },
        viewport: { ...fallback.viewport, ...(value.viewport ?? {}), insets },
        storage: { ...fallback.storage, ...(value.storage ?? {}) },
        audio: { ...fallback.audio, ...(value.audio ?? {}) },
        capabilities,
    };
}
