/**
 * 用户界面偏好的持久化出口。
 *
 * 【为什么需要它】这些值原先只存在 WebView 的 `localStorage` 里。独立 App 的
 * WebView2 用户数据目录是稳定的，所以那边一直"能用"；而插件窗口的用户数据目录
 * 是**每进程**的临时目录，退出 REAPER 即丢 —— 用户改的语言、快捷键、外观、
 * 时间轴缩放每次都回到出厂默认。把同一批键同时写进后端的配置文件后，两个形态
 * 还能共用一份偏好：在 App 里调好的外观，进插件就是调好的样子。
 *
 * 【为什么读仍然走 `localStorage`】读取点大量是同步的，而且有些发生在**模块加载期**
 * （`keybindingsSlice` 建 store 时读覆盖项、`themeStorage` 读外观）。把读改成异步会
 * 迫使这些调用点重排时序。因此启动时先 `hydrateUiStorage()` 把后端快照**灌回**
 * `localStorage`，之后所有同步读取照旧；写入则同时落 `localStorage`（同步缓存）
 * 与后端（耐久 + 共享）。
 *
 * 【为什么只同步列在白名单里的键】`localStorage` 里还有一堆调试开关
 * （`hifishifter.debugDnd` / `frameProfiler` / `perfProject`…）。它们属于本机、
 * 本次调试，不该跟着用户配置走。
 *
 * 【为什么本模块不 import `hostCapabilities`】它间接依赖 i18n 提供者（React 组件
 * 链），而本模块会被 `keybindingStorage` 这类**叶子模块**引用 —— 那些模块的 node
 * 单测不装 jsdom，多引一层就会在 `import` 那一刻炸掉。插件模式直接看
 * `window.__HFS_PLUGIN_BOOTSTRAP__` 即可，判据与 `isPluginMode()` 完全相同。
 */

/** 前端偏好的键前缀；与后端 `frontendPrefs` 的键名约定一致。 */
const PREFIX = "hifishifter.";

/**
 * 需要同步到后端的键。
 *
 * 【维护须知】新增"用户会感知、且希望换形态后仍然生效"的偏好时，把键名加进来。
 * 只是本机调试用的开关不要加。
 */
export const PERSISTED_UI_KEYS: readonly string[] = [
    // 界面语言
    "hifishifter.locale",
    // 快捷键覆盖项
    "hifishifter.keybindings",
    // 外观（主题 / 字号 / 密度）与用户自定义主题
    "hifishifter.appearance",
    "hifishifter.customThemes",
    // 时间轴与参数编辑器的缩放、行高（按形态分键，见 `modeKey`）
    "hifishifter.pxPerSec",
    "hifishifter.rowHeight",
    "hifishifter.paramPxPerSec",
    // 文件浏览器上次所在目录
    "hifishifter.fileBrowser.lastPath",
];

/**
 * 与**视口尺寸**相关的偏好键 —— 它们在插件形态下另存一份。
 *
 * 【为什么只有这三个】缩放与行高的正确取值直接取决于窗口有多少像素：为 2560px
 * 调好的 px/sec 在 640×400 的 ARA 窗口里什么都看不清，反之则过于粗放。
 * locale / 快捷键 / 主题 / 路径 / 设备选择这些"取决于用户是谁"的仍然共用同一个键
 * —— 判据与 `UiSettings::dock_plugin` 相同。
 */
const PLUGIN_SCOPED_KEYS: ReadonlySet<string> = new Set([
    "hifishifter.pxPerSec",
    "hifishifter.rowHeight",
    "hifishifter.paramPxPerSec",
]);

/** 插件形态的后缀。用一个**不在** `PREFIX` 之外的普通字符，键名仍以 `hifishifter.` 开头。 */
const PLUGIN_SUFFIX = ".plugin";

/** 当前是否跑在 ARA 插件里（判据与 `hostCapabilities.isPluginMode()` 一致）。 */
function isPluginHost(): boolean {
    return typeof window !== "undefined" && window.__HFS_PLUGIN_BOOTSTRAP__?.version === 1;
}

/**
 * 把偏好键映射到**当前形态**的名字。
 *
 * 【为什么在这里而不是让调用方自己拼】读写路径有四条（`readUiValue` /
 * `writeUiValue` / `removeUiValue` / 启动灌回），散开改必漏一处 —— 而漏掉的那条
 * 会安静地把两个形态的值混在一起，正是本次要修的故障。
 */
export function modeKey(name: string): string {
    if (!isPluginHost() || !PLUGIN_SCOPED_KEYS.has(name)) return name;
    return `${name}${PLUGIN_SUFFIX}`;
}

/** 白名单要同时认下分形态的键名，否则灌回与迁移会跳过它们。 */
const PERSISTED = new Set<string>([
    ...PERSISTED_UI_KEYS,
    ...Array.from(PLUGIN_SCOPED_KEYS, (key) => `${key}${PLUGIN_SUFFIX}`),
]);

/** 写入后延迟提交到后端的时间（毫秒）。 */
const FLUSH_DELAY_MS = 200;

/** 后端读写接口。抽出类型是为了让测试注入替身，而不是真的去调后端。 */
export interface UiStorageTransport {
    dump(): Promise<Record<string, string>>;
    put(patch: Record<string, string>): Promise<void>;
    remove(keys: string[]): Promise<void>;
}

let transport: UiStorageTransport | null = null;
let transportResolved = false;

/** 后端是否具备偏好存储能力：插件与独立 App 都有，纯浏览器 dev 模式没有。 */
function resolveTransport(): UiStorageTransport | null {
    if (transportResolved) return transport;
    transportResolved = true;
    if (typeof window === "undefined") return null;

    // 判据与 `hostCapabilities.isPluginMode()` 一致：只看显式 bootstrap，
    // 不把"存在 WebView2 对象"当成插件（见该模块的说明）。
    const plugin = isPluginHost();
    const tauriInvoke = window.__TAURI__?.core?.invoke ?? window.__TAURI__?.invoke;
    if (!plugin && typeof tauriInvoke !== "function") return null;

    // 延迟 import 避免纯浏览器环境下把后端模块拉进包里。
    transport = {
        async dump() {
            const { settingsApi } = await import("./api/settings");
            return (await settingsApi.uiKvDump()) ?? {};
        },
        async put(patch) {
            const { settingsApi } = await import("./api/settings");
            await settingsApi.uiKvPut(patch);
        },
        async remove(keys) {
            const { settingsApi } = await import("./api/settings");
            await settingsApi.uiKvDelete(keys);
        },
    };
    return transport;
}

/** 仅供测试：注入替身（传 `null` 恢复自动探测）。 */
export function setUiStorageTransportForTest(next: UiStorageTransport | null): void {
    transport = next;
    transportResolved = next !== null;
}

// ── localStorage 访问（不可用时静默降级）───────────────────────────────

function localGet(key: string): string | null {
    try {
        return typeof localStorage === "undefined" ? null : localStorage.getItem(key);
    } catch {
        return null;
    }
}

function localSet(key: string, value: string): void {
    try {
        localStorage?.setItem(key, value);
    } catch {
        // 隐私模式 / 配额耗尽：偏好落不下不该打断交互。
    }
}

function localRemove(key: string): void {
    try {
        localStorage?.removeItem(key);
    } catch {
        // 同上。
    }
}

/** 枚举本机缓存里全部 `hifishifter.*` 键（迁移用）。 */
function localSnapshot(): Record<string, string> {
    const snapshot: Record<string, string> = {};
    try {
        if (typeof localStorage === "undefined") return snapshot;
        for (let index = 0; index < localStorage.length; index += 1) {
            const key = localStorage.key(index);
            if (!key || !key.startsWith(PREFIX) || !PERSISTED.has(key)) continue;
            const value = localStorage.getItem(key);
            if (value != null) snapshot[key] = value;
        }
    } catch {
        // 同上。
    }
    return snapshot;
}

// ── 待提交队列 ─────────────────────────────────────────────────────────

let pendingPut: Record<string, string> = {};
const pendingDelete = new Set<string>();
let flushTimer: ReturnType<typeof setTimeout> | null = null;

function scheduleFlush(): void {
    if (flushTimer !== null) clearTimeout(flushTimer);
    flushTimer = setTimeout(() => {
        flushTimer = null;
        void flushUiStorage();
    }, FLUSH_DELAY_MS);
}

/**
 * 立即把待提交的偏好写入后端。
 *
 * 失败只记日志：偏好存不下不该让界面报错，本次会话仍按内存/`localStorage`
 * 里的值运行。失败项不重新入队 —— 否则一个持续失败的后端会变成无限重试。
 */
export async function flushUiStorage(): Promise<void> {
    if (flushTimer !== null) {
        clearTimeout(flushTimer);
        flushTimer = null;
    }
    const active = resolveTransport();
    if (!active) return;

    const patch = pendingPut;
    const keys = Array.from(pendingDelete);
    pendingPut = {};
    pendingDelete.clear();
    if (Object.keys(patch).length === 0 && keys.length === 0) return;

    try {
        if (Object.keys(patch).length > 0) await active.put(patch);
        if (keys.length > 0) await active.remove(keys);
    } catch (error) {
        console.warn("Persisting UI preferences failed", error);
    }
}

// ── 对外接口 ───────────────────────────────────────────────────────────

/** 读取一个偏好值（同步，走本机缓存）。 */
export function readUiValue(key: string): string | null {
    return localGet(modeKey(key));
}

/**
 * 写入一个偏好值：同步更新本机缓存，并按需排队提交到后端。
 *
 * 非白名单键退化为普通 `localStorage` 写入 —— 调用方不必区分"这个键要不要同步"。
 * 调用方始终传**基础键名**；分形态的改名由 [`modeKey`] 统一处理。
 */
export function writeUiValue(key: string, value: string): void {
    const stored = modeKey(key);
    localSet(stored, value);
    if (!PERSISTED.has(stored)) return;
    // 同一轮里先删后写（或先写后删）时，最后一次操作说了算。
    pendingDelete.delete(stored);
    pendingPut[stored] = value;
    scheduleFlush();
}

/** 删除一个偏好值（"重置为默认"）。 */
export function removeUiValue(key: string): void {
    const stored = modeKey(key);
    localRemove(stored);
    if (!PERSISTED.has(stored)) return;
    delete pendingPut[stored];
    pendingDelete.add(stored);
    scheduleFlush();
}

/**
 * 启动时把后端偏好灌回本机缓存。
 *
 * 【为什么"灌回"而不是"改成异步读"】同步读取点太多（含模块加载期的），把读改成
 * 异步会迫使它们重排时序。灌回之后所有既有同步读取照旧，只是值已经是后端那份。
 *
 * 【一次性迁移】本机缓存里有、后端没有的键，视为升级前留下的值，提交上去。
 * 反方向不迁移：后端有值时以后端为准（它才是跨形态共享的那一份）。
 *
 * 任何失败都静默降级为本机缓存 —— 后端不可用不该让界面起不来。
 */
export async function hydrateUiStorage(): Promise<void> {
    const active = resolveTransport();
    if (!active) return;

    let remote: Record<string, string>;
    try {
        remote = (await active.dump()) ?? {};
    } catch (error) {
        console.warn("Loading UI preferences failed; using the local cache", error);
        return;
    }

    const local = localSnapshot();
    for (const [key, value] of Object.entries(remote)) {
        if (!PERSISTED.has(key)) continue;
        localSet(key, value);
    }

    // 升级迁移：本机有、后端没有的键提交上去。
    const migration: Record<string, string> = {};
    for (const [key, value] of Object.entries(local)) {
        if (!(key in remote)) migration[key] = value;
    }
    if (Object.keys(migration).length > 0) {
        try {
            await active.put(migration);
        } catch (error) {
            console.warn("Migrating local UI preferences failed", error);
        }
    }
}

/**
 * 安装"关窗前补交"钩子。
 *
 * 【为什么必须有】写入是去抖的：用户改完缩放立刻关掉插件窗口时，最后一次变更
 * 可能还排在队列里。`pagehide` 是浏览器保证会触发的最后一个时机。
 */
export function installUiStorageFlushHooks(): void {
    if (typeof window === "undefined") return;
    window.addEventListener("pagehide", () => void flushUiStorage());
    if (typeof document === "undefined") return;
    document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "hidden") void flushUiStorage();
    });
}
