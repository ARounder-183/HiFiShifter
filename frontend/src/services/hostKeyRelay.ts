/**
 * 宿主专属快捷键的让路判定（Host Key Relay）。
 *
 * 【要解决的问题】在不支持 ARA 的宿主里，插件拿不到 `ARAPlaybackControllerInterface`，
 * 于是 `playback.toggle` / `playback.stop` 必然被能力矩阵拒绝，只弹一句
 * 「当前宿主没有提供这个接口」。可用户按 Space 的本意就是**让宿主播放** —— 而 Windows
 * 的键盘消息不向父窗口冒泡，宿主根本收不到这个键。结果是死角：插件没能力做，宿主没收到键。
 *
 * 【本模块只回答"这一次按键该不该让"】"怎么让"（重投窗口消息 / 合成事件）与
 * "让成了没有"都在原生侧（见 `backend/hifishifter-plugin/src/editor/host_key_relay/`）。
 * 两边各答一半，判据不重复。
 *
 * 【铁律：只转交按下的那个键，绝不翻译意图】插件不猜宿主的键位表 —— 它只是"不吞掉"。
 * 用户按 Space，宿主收到的就是 Space。因此本模块也**不做**"给宿主发某个动作 ID"的映射。
 *
 * 【一条规则，两个入口】
 * - 入口 A（`denied`）：插件**认识**这个键（命中过绑定），但能力矩阵说它不归插件做；
 * - 入口 B（`unbound`）：一条绑定都没命中 —— 插件本来就不处理它。
 * 两者合起来就是那条规则：**插件只拥有它的键位表，其余是宿主的。**
 */
import {
    canControlHostTransport,
    capabilityForAction,
    isPluginMode,
    type CapabilityId,
} from "./hostCapabilities";
import { hostKeysApi, type HostKeyModifiers } from "./api/hostKeys";
import { PluginInvokeError } from "./pluginHost";
import { backendDetail, invokeErrorKind } from "./invoke";
import { isNoneBindingList } from "../features/keybindings/keybindingsSlice";
import type { ActionId, KeybindingMap } from "../features/keybindings/types";
import type { EditSurfaceId } from "../features/uiFocus/focusSurface";

/** 一次按键（已归一化到 DOM `code`）。 */
export type HostKeyStroke = { code: string } & HostKeyModifiers;

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
    /** 命中的动作；入口 B（未绑定的键）为 `null` —— 原生只拿它写日志。 */
    action: ActionId | null;
    stroke: HostKeyStroke;
}

/** 判定结论。 */
export type RelayVerdict =
    | { kind: "none" }
    | { kind: "relay"; request: HostKeyRelayRequest; reason: "denied" | "unbound" };

/**
 * 可让路的按键，取值为 DOM `KeyboardEvent.code`。
 *
 * 【为什么是 `code` 而不是 `key`】`key` 随键盘布局与 Shift 变（AZERTY 上按 Q 得到 "a"），
 * `code` 是**与布局无关的物理键名**，与 Windows 的 VK 一一对应。而"让路"的全部价值在于
 * **用户按什么、宿主就收到什么** —— 一个被布局改写的键恰好违背它。
 *
 * 【为什么只有具名键】可打印字符是**打字**的领地：未绑定的字母/数字如果也让，插件窗口
 * 就变成一个透明的键盘转发器 —— 用户在一个没有输入框的表面敲字，宿主要么没反应，要么
 * 执行一个他完全没预期的动作。具名键没有这个歧义：没人会"在时间轴上打字"。
 *
 * 【为什么这张表要跨语言比对】它同时存在于 Rust（`host_key_relay/strokes.rs` 的
 * `CODES`）。两处漂移的表现是"按了没反应且没有任何提示"，最难查的一类。
 * `hostKeyRelay.table.test.ts` 逐项比对，因此这里改动必须同步改那边。
 */
export const RELAYABLE_CODES: ReadonlySet<string> = new Set([
    "Space",
    "Enter",
    "Tab",
    "Escape",
    "Backspace",
    "Insert",
    "Delete",
    "Home",
    "End",
    "PageUp",
    "PageDown",
    "ArrowLeft",
    "ArrowUp",
    "ArrowRight",
    "ArrowDown",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
    "F11",
    "F12",
    "F13",
    "F14",
    "F15",
    "F16",
    "F17",
    "F18",
    "F19",
    "F20",
    "F21",
    "F22",
    "F23",
    "F24",
    "Numpad0",
    "Numpad1",
    "Numpad2",
    "Numpad3",
    "Numpad4",
    "Numpad5",
    "Numpad6",
    "Numpad7",
    "Numpad8",
    "Numpad9",
    "NumpadEnter",
    "NumpadAdd",
    "NumpadSubtract",
    "NumpadMultiply",
    "NumpadDivide",
    "NumpadDecimal",
]);

/**
 * 即使"未绑定"也不让的键。
 *
 * 【为什么需要这份短名单】入口 B 的默认答案是"让"，但这两个键在宿主里几乎总是
 * **关掉当前窗口 / 移动焦点**：插件里一次误触会关掉宿主的 FX 窗口，而用户根本不
 * 知道发生了什么。它们仍然留在 `RELAYABLE_CODES` 里 —— 入口 A（用户自己把它绑到
 * 播放键上）依然成立。
 */
export const NON_RELAYABLE_CODES: ReadonlySet<string> = new Set(["Escape", "Tab"]);

/**
 * 只有"宿主自己那一族"的动作才走入口 A。
 *
 * 【为什么不是"所有被拒绝的动作"】被拒的动作里有一大类是**插件自己的**（编辑、片段
 * 几何、工程文件）。把它们让给宿主是灾难：插件里按 `Ctrl+Z` 会变成撤销**宿主**的编辑，
 * 按 `Delete` 会删宿主的片段。只有传输类的三族，宿主那边的语义与插件这边一致。
 */
const HOST_OWNED_CAPABILITIES: ReadonlySet<CapabilityId> = new Set<CapabilityId>([
    "transport.control",
    "playback.metronome",
    "recording.local",
]);

/** 键位描述是否"裸键"：无任何修饰。 */
export function isBareStroke(stroke: HostKeyStroke): boolean {
    return !stroke.ctrl && !stroke.shift && !stroke.alt && !stroke.meta;
}

/**
 * 该键位能否让路：在具名表内，且不含 `Meta`。
 *
 * 【为什么带修饰的组合**可以**让】上一轮把"带修饰"一律拒掉，理由是 `PostMessage`
 * 不更新线程的异步键状态。那条限制是真的，但它只影响"直接读 `GetKeyState` 的宿主"；
 * 按**消息流**维护修饰状态的宿主收到"修饰按下 → 键按下 → 键抬起 → 修饰抬起"这一串
 * 就是忠实的。而宿主快捷键绝大多数是带修饰的组合（`Ctrl+S`、`Shift+Space`）——
 * 一律拒掉等于让这条路只对空格有用。
 *
 * 【为什么 `Meta` 仍然拒】Win 键带系统级副作用：外送一次就可能把开始菜单弹出来。
 */
export function isRelayableStroke(stroke: HostKeyStroke): boolean {
    return !stroke.meta && RELAYABLE_CODES.has(stroke.code);
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
 * 入口 A：命中了一条绑定，但能力矩阵拒绝。
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
    // 3. 只让"宿主自己那一族"。编辑类被拒时**绝不能**外泄（见 HOST_OWNED_CAPABILITIES）。
    const capability = capabilityForAction(action, surface);
    if (capability === null || !HOST_OWNED_CAPABILITIES.has(capability)) return null;
    // 4. `Meta` 不让（系统级副作用）；其余修饰位由原生侧按消息流补齐。
    if (stroke.meta) return null;
    // 5. 这个会话里已经明确让不成了，不再重复打扰。
    if (env.unsupported) return null;
    // 6. 只让具名、与布局无关的键（见 RELAYABLE_CODES 的说明）。
    if (!RELAYABLE_CODES.has(stroke.code)) return null;
    return { action: action as ActionId, stroke };
}

/**
 * 入口 B：一条绑定都没命中。
 *
 * 【为什么这条规则是安全的】"插件不认识这个键"是**它自己刚判定完**的事实：全局派发器
 * 在捕获阶段跑过一遍 `resolveActionByFocus`，没命中才走到这里。用户改键之后，判据自动
 * 跟着变 —— 不需要维护任何名单。
 *
 * 【为什么不看 `e.key.length`】`RELAYABLE_CODES` 已经把范围收窄到具名键；再加一条
 * 字符长度判断只会多一处可能与表不一致的判据。
 */
export function classifyUnboundHostKey(
    stroke: HostKeyStroke,
    env: HostKeyRelayEnv,
): HostKeyRelayRequest | null {
    if (!env.plugin) return null;
    if (env.unsupported) return null;
    if (!isRelayableStroke(stroke)) return null;
    // 【为什么这份短名单只在**裸键**时生效】`Escape` / `Tab` 在宿主里几乎总是
    // "关窗 / 换焦点"，一次误触就会关掉宿主窗口。而 `Ctrl+Tab`、`Shift+Escape` 是明确
    // 的命令，没有人会把它们当成"随手一按" —— 它们照常外送。
    if (isBareStroke(stroke) && NON_RELAYABLE_CODES.has(stroke.code)) return null;
    return { action: null, stroke };
}

/**
 * 收集"插件自己占用"的按键，供原生在投递前做一致性闸。
 *
 * 【为什么是"占用"而不是"可让"】可让的是**未绑定**的那些键 —— 那是一个没法枚举的补集。
 * 而"插件占用了什么"恰好是有限的、且这里本来就算得出来的（合并后的键位表）。
 *
 * 【为什么要做 key → code 的换算】键位表存的是 `e.key` 归一化后的小写名（`space`、
 * `arrowleft`），而投递用的是物理 `code`。两者只差大小写 —— 但只对**具名键**成立，
 * 因此这里只在 `RELAYABLE_CODES` 里查，表外（字母、数字、标点）直接跳过：它们本来
 * 也不在可让范围里。
 */
export function collectOwnedStrokes(
    keybindings: KeybindingMap,
    env: HostKeyRelayEnv = hostKeyRelayEnv(),
): HostKeyStroke[] {
    if (!env.plugin) return [];
    const seen = new Set<string>();
    const strokes: HostKeyStroke[] = [];
    for (const action of Object.keys(keybindings) as ActionId[]) {
        const bindings = keybindings[action];
        if (!bindings || isNoneBindingList(bindings)) continue;
        for (const binding of bindings) {
            if (binding.modifierOnly) continue;
            const code = codeForBindingKey(binding.key);
            if (code === null) continue;
            const stroke: HostKeyStroke = {
                code,
                ctrl: Boolean(binding.ctrl),
                shift: Boolean(binding.shift),
                alt: Boolean(binding.alt),
                meta: false,
            };
            const id = strokeId(stroke);
            if (seen.has(id)) continue;
            seen.add(id);
            strokes.push(stroke);
        }
    }
    return strokes.sort((left, right) => (strokeId(left) < strokeId(right) ? -1 : 1));
}

/** 键位表的小写键名 → DOM `code`（仅具名键；表外返回 `null`）。 */
function codeForBindingKey(key: string): string | null {
    for (const code of RELAYABLE_CODES) {
        if (code.toLowerCase() === key) return code;
    }
    return null;
}

/** 键位的稳定标识（用于去重与集合比对）。 */
function strokeId(stroke: HostKeyStroke): string {
    return [
        stroke.code,
        stroke.ctrl ? "c" : "",
        stroke.shift ? "s" : "",
        stroke.alt ? "a" : "",
        stroke.meta ? "m" : "",
    ].join("|");
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
        const reply = await hostKeysApi.relay(request.stroke.code, request.action, {
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
        // 宿主让路就在余下整个会话里静默失效且不可自愈。
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
 * 声明"插件自己占用了哪些键"（键位或能力位变化时重发一次）。
 *
 * 失败**静默**：这是一条内部一致性闸，用户按 Space 根本不经过它；
 * 为一次后台同步失败弹提示只会制造噪声。
 */
export function publishOwnedStrokes(strokes: readonly HostKeyStroke[]): void {
    void hostKeysApi
        .publishOwnedStrokes(
            strokes.map((stroke) => ({
                code: stroke.code,
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

/** 一次"把键盘交给宿主"的结果。 */
export interface HostKeyFocusOutcome {
    ok: boolean;
    /**
     * 失败原因（`"no-host-window"` / `"post-failed"` / `"unavailable"`）。
     *
     * 【为什么要把它带出来】这个功能此前失败时是**完全静默**的：调用点把布尔丢掉，
     * 用户看到的就是"按了没反应"。一个没有反馈的功能，与一个坏掉的功能在体验上
     * 无法区分。
     */
    reason: string | null;
}

/**
 * 把键盘焦点交还宿主 —— 让路的**确定**那条路。
 *
 * 【什么时候用它】投递窗口消息要看宿主脸色：有的宿主只看自己的前台焦点窗口，有的把
 * 编辑器放进桥接进程。那些情况下"让路"不会生效，而用户仍需要一个能立刻用宿主键位的
 * 办法。交还焦点一定会生效，代价是焦点真的走了（之后要点击插件才能回来）—— 因此它
 * 只由用户显式触发（快捷键），从不自动发生。
 *
 * 【为什么直接返回结果，不再广播 DOM 事件】它曾经只有一个菜单项入口，而"说一句话"的
 * 实现在 App 里 —— 事件是那时绕开组件边界的手段。现在入口是 App 自己的快捷键分支，
 * 事件绕一圈又回到原点，只剩一层没人受益的间接。
 */
export async function handKeyboardToHost(): Promise<HostKeyFocusOutcome> {
    try {
        const reply = await hostKeysApi.focusHost();
        if (reply?.ok === true) return { ok: true, reason: null };
        return {
            ok: false,
            reason: typeof reply?.reason === "string" ? reply.reason : "post-failed",
        };
    } catch {
        return { ok: false, reason: "unavailable" };
    }
}
