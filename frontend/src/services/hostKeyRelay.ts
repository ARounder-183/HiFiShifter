/**
 * 宿主专属快捷键的让路判定（Host Key Relay）。
 *
 * 【要解决的问题】在不支持 ARA 的宿主里，插件拿不到 `ARAPlaybackControllerInterface`，
 * 于是 `playback.toggle` / `playback.stop` 必然被能力矩阵拒绝，只弹一句
 * 「当前宿主没有提供这个接口」（`cap_reason_unsupported_by_host`）。可用户按 Space
 * 的本意就是**让宿主播放** —— 而 Windows 的键盘消息不向父窗口冒泡，宿主根本收不到这个键。
 * 结果是死角：插件没能力做，宿主没收到键。
 *
 * 【本模块只回答一半】"这一次按键，插件是不是根本用不上、可以还给宿主"。至于
 * "怎么还"（Windows 重投消息 / macOS 让它沿响应链继续走 / X11 合成事件）与"还成了没有"，
 * 都是原生侧的事（见 `backend/hifishifter-plugin/src/editor/host_key_relay/`）。
 * 两边各答一半，接口只有三个字段，判据不重复。
 *
 * 【铁律：只转交按下的那个键，绝不翻译意图】插件不猜宿主的键位表 —— 它只是"不吞掉"。
 * 用户按 Space，宿主收到的就是 Space。因此本模块也**不做**"给宿主发某个动作 ID"的映射。
 */
import { canControlHostTransport, capabilityForAction, isPluginMode } from "./hostCapabilities";
import { hostKeysApi, type HostKeyModifiers } from "./api/hostKeys";
import { getPluginHost, PluginInvokeError } from "./pluginHost";
import { backendDetail, invokeErrorKind } from "./invoke";
import { firstBinding, isNoneBindingList } from "../features/keybindings/keybindingsSlice";
import type { ActionId, KeybindingMap } from "../features/keybindings/types";
import type { EditSurfaceId } from "../features/uiFocus/focusSurface";

/** 一次按键（已归一化）。 */
export type HostKeyStroke = { key: string } & HostKeyModifiers;

/** 判定所需的全部外部事实。由调用方一次读出，本模块不碰 `window`（保持纯函数可测）。 */
export interface HostKeyRelayEnv {
    /** 是否插件形态。独立 App 里键盘归自己，没有"宿主"可让。 */
    plugin: boolean;
    /** 插件是否已拿到 ARA 播放请求 —— 有它插件自己就能做，让位没有收益。 */
    transportControl: boolean;
    /** 本会话里原生已明确表示"让不成立"（旧版本 / 本平台没有这条路），不再重复打扰。 */
    unsupported: boolean;
}

/** 一次让路请求。 */
export interface HostKeyRelayRequest {
    action: ActionId;
    stroke: HostKeyStroke;
}

/** 原生告知"让路没成"的通知。 */
export interface HostKeyRelayNotice {
    action: string;
    reason: "no-response" | "refused" | "unavailable";
}

/**
 * 可让路的按键 token。
 *
 * 【为什么是一张白名单，而不是"什么键都行"】只有**与键盘布局无关**的具名键才在表内。
 * 字母/数字要靠 `VkKeyScanW`（Windows）、`charactersIgnoringModifiers`（macOS）、
 * `XStringToKeysym`（X11）换算，三套换算在非美式布局上都会给出不同结果 —— 而
 * "让路"的全部价值在于**用户按什么、宿主就收到什么**，一个被布局改写的键恰好违背它。
 *
 * 【为什么这张表要跨语言比对】它同时存在于 Rust（`host_key_relay/strokes.rs` 的
 * `RELAYABLE_TOKENS`）。两处漂移的表现是"按了没反应且没有任何提示"，最难查的一类。
 * `hostKeyRelay.table.test.ts` 逐项比对，因此这里改动必须同步改那边。
 */
export const RELAYABLE_KEYS: ReadonlySet<string> = new Set([
    "space",
    "enter",
    "tab",
    "escape",
    "backspace",
    "insert",
    "delete",
    "home",
    "end",
    "pageup",
    "pagedown",
    "arrowleft",
    "arrowup",
    "arrowright",
    "arrowdown",
]);

/** 键位描述是否"裸键"：无任何修饰。修饰状态无法忠实重放（见方案 §8.5）。 */
export function isBareStroke(stroke: HostKeyStroke): boolean {
    return !stroke.ctrl && !stroke.shift && !stroke.alt && !stroke.meta;
}

/** 该键位能否让路：裸键 + 在具名白名单内。 */
export function isRelayableStroke(stroke: HostKeyStroke): boolean {
    return isBareStroke(stroke) && RELAYABLE_KEYS.has(stroke.key);
}

/** 键位的稳定标识（用于去重与集合比对）。 */
function strokeId(stroke: HostKeyStroke): string {
    return stroke.key;
}

/**
 * 本会话里原生是否已经表示过"让不成立"。
 *
 * 【为什么是模块级闩锁而不是 Redux】它是**这一次进程**学到的事实，与界面状态无关；
 * 放进 store 只会让它在换工程时被重置，而"这个宿主不认投递键"不会因为换工程而改变。
 */
let relayUnsupported = false;

/** 读出判定所需的当前事实。 */
export function hostKeyRelayEnv(): HostKeyRelayEnv {
    return {
        plugin: isPluginMode(),
        transportControl: canControlHostTransport(),
        unsupported: relayUnsupported,
    };
}

/** 仅供测试：清掉会话闩锁。 */
export function resetHostKeyRelayLatchForTests(): void {
    relayUnsupported = false;
}

/**
 * 纯判定：这一次按键是否应该让给宿主；不该让返回 `null`。
 *
 * 六条规则，每一条都对应一类**不能**让的情形。改这里的任何一条都要同时改
 * `hostKeyRelay.test.ts` 的真值表 —— 那是本模块唯一的行为契约。
 */
export function classifyHostKeyRelay(
    action: string,
    surface: EditSurfaceId | null,
    stroke: HostKeyStroke,
    env: HostKeyRelayEnv,
): HostKeyRelayRequest | null {
    // 1. 独立 App：键盘归自己，没有"宿主"可让。
    if (!env.plugin) return null;
    // 2. 插件已能控制宿主播放（ARA 播放请求）：自己做就是最好的结果，让位只会双触发。
    if (env.transportControl) return null;
    // 3. 只让"播放"这一族。剪辑类被拒时**绝不能**外泄：Delete 到了宿主就是删宿主的片段。
    if (capabilityForAction(action, surface) !== "transport.control") return null;
    // 4. 裸键才让：修饰状态无法忠实重放，宿主会看到"没按 Ctrl 的 S"。
    if (!isBareStroke(stroke)) return null;
    // 5. 这个会话里已经明确让不成了，不再重复打扰。
    if (env.unsupported) return null;
    // 6. 只让具名、与布局无关的键（见 RELAYABLE_KEYS 的说明）。
    if (!RELAYABLE_KEYS.has(stroke.key)) return null;
    return { action: action as ActionId, stroke };
}

/**
 * 收集"属于宿主"的全部按键，供原生在 PassThrough 平台上做**事前**判定。
 *
 * 取法与 `classifyHostKeyRelay` 的规则 3/6 同源（同一个 `capabilityForAction` 与
 * 同一张 `RELAYABLE_KEYS`），因此不存在"实时判定放行、集合却没收录"的分叉。
 *
 * 【为什么取全部绑定而不是只取主绑定】替补绑定是"等价替补"，它们同样该让。多收一个
 * 不改变语义（它本来就该让），少收一个会让那个替补键失效。去重后按 key 排序输出，
 * 稳定顺序让日志与测试可比。
 */
export function collectHostKeyEscapeStrokes(
    keybindings: KeybindingMap,
    env: HostKeyRelayEnv,
): HostKeyStroke[] {
    if (!env.plugin || env.transportControl) return [];
    const seen = new Set<string>();
    const strokes: HostKeyStroke[] = [];
    for (const action of Object.keys(keybindings) as ActionId[]) {
        if (capabilityForAction(action, null) !== "transport.control") continue;
        const bindings = keybindings[action];
        if (!bindings || isNoneBindingList(bindings)) continue;
        for (const binding of [firstBinding(bindings), ...bindings.slice(1)]) {
            if (binding.modifierOnly) continue;
            const stroke: HostKeyStroke = {
                key: binding.key,
                ctrl: Boolean(binding.ctrl),
                shift: Boolean(binding.shift),
                alt: Boolean(binding.alt),
                meta: false,
            };
            if (!isRelayableStroke(stroke)) continue;
            const id = strokeId(stroke);
            if (seen.has(id)) continue;
            seen.add(id);
            strokes.push(stroke);
        }
    }
    return strokes.sort((left, right) => (strokeId(left) < strokeId(right) ? -1 : 1));
}

/**
 * 把这一次按键交回宿主。
 *
 * 返回 `"unsupported"` 仅当"连让都没让成"（键不在原生表里 / 带修饰 / 旧后端没有这条
 * 命令）—— 调用方此时应回落旧文案。其余情况一律 `"accepted"`：不接手也是**正常**的
 * （宿主本来就会处理、上一拍还没走完），用户不该因此看到任何提示。
 */
export async function relayHostKey(
    request: HostKeyRelayRequest,
): Promise<"accepted" | "unsupported"> {
    try {
        const reply = await hostKeysApi.relay(request.stroke.key, request.action, {
            ctrl: request.stroke.ctrl,
            shift: request.stroke.shift,
            alt: request.stroke.alt,
            meta: request.stroke.meta,
        });
        if (reply?.accepted !== false) return "accepted";
        if (reply.reason === "unsupported-key" || reply.reason === "modified-key") {
            // 让不成立 ⇒ 本会话不再尝试。继续尝试只会每次多一次 IPC 和一次旧文案。
            relayUnsupported = true;
            return "unsupported";
        }
        return "accepted";
    } catch (cause) {
        // 【为什么按 kind 分流】只有**终态**失败才值得烧闩锁：旧后端没有这条命令
        // （method not wired yet）、会话已终态。`invoke` 的 30s 超时、pending≥128
        // 的背压、一次瞬态 IPC 抖动都会 reject —— 把暂态失败也闩死，Space/Enter 的
        // 宿主让路就在余下整个会话里静默失效且不可自愈（第 5 轮审查 P3-3）。
        // 暂态失败保持可重试：用户下一拍按键就是重试。
        //
        // 【为什么不能只判 PluginInvokeError】生产路径走 `invoke()`：它把底层
        // `PluginInvokeError` 重包装成 `BackendInvokeError` 再抛（分类随 cause 保留），
        // 而"旧后端没这条命令"更是裸 `Error`。这里必须按 `invoke()` 实际抛出的形态
        // 判定；`instanceof PluginInvokeError` 只覆盖测试直调 `hostKeysApi.relay` 的路径。
        const notWired = backendDetail(cause)?.includes("not wired") ?? false;
        if (
            (cause instanceof PluginInvokeError && cause.kind === "terminal") ||
            invokeErrorKind(cause) === "terminal" ||
            notWired
        ) {
            relayUnsupported = true;
        }
        return "unsupported";
    }
}

/**
 * 声明"属于宿主"的按键集合（发一次即可，键位或能力位变化时重发）。
 *
 * 失败**静默**：这是 PassThrough 平台的内部接缝，Windows 上按 Space 根本不经过它；
 * 为一次后台同步失败弹提示只会制造噪声。
 */
export function publishHostKeyEscapeSet(strokes: readonly HostKeyStroke[]): void {
    void hostKeysApi
        .publishEscapeSet(
            strokes.map((stroke) => ({
                key: stroke.key,
                ctrl: stroke.ctrl,
                shift: stroke.shift,
                alt: stroke.alt,
                meta: stroke.meta,
            })),
        )
        .catch(() => {
            // 见函数注释：没有可操作的用户动作。
        });
}

/**
 * 把键盘焦点交还宿主 —— 让路的**确定**那条路。
 *
 * 【什么时候用它】投递窗口消息要看宿主脸色：有的宿主只看自己的前台焦点窗口，有的把
 * 编辑器放进桥接进程。那些情况下"让路"不会生效，而用户仍需要一个能立刻用宿主键位的
 * 办法。交还焦点一定会生效，代价是焦点真的走了（之后要点击插件才能回来）—— 因此它
 * 只由用户显式触发（视图菜单），从不自动发生。
 *
 * 返回是否成功；失败不弹提示（调用点就在菜单上，用户看得到焦点有没有走）。
 */
export async function handKeyboardToHost(): Promise<boolean> {
    try {
        const reply = await hostKeysApi.focusHost();
        return reply?.ok === true;
    } catch {
        return false;
    }
}

/**
 * 订阅"让路没成"的通知；返回退订函数。
 *
 * 【为什么不用 `listen` 的 Promise】调用点是一个 React effect，卸载时必须能**同步**
 * 退订。这里把异步订阅的完成与退订排队：若订阅在卸载之后才完成，立即退订。
 */
export function subscribeHostKeyRelayNotice(
    listener: (notice: HostKeyRelayNotice) => void,
): () => void {
    let unsubscribe: (() => void) | null = null;
    let disposed = false;
    void (async () => {
        try {
            const host = getPluginHost();
            if (!host) return;
            const off = await host.listen<HostKeyRelayNotice>("host_key_relay_notice", (event) => {
                const payload = event.payload;
                if (!payload || typeof payload.action !== "string") return;
                const reason = payload.reason;
                if (reason !== "no-response" && reason !== "refused" && reason !== "unavailable")
                    return;
                listener({ action: payload.action, reason });
            });
            if (disposed) off();
            else unsubscribe = off;
        } catch {
            // 独立 App / 消息口不可用：没有通知可订阅，也不需要提示。
        }
    })();
    return () => {
        disposed = true;
        unsubscribe?.();
        unsubscribe = null;
    };
}
