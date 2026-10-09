/**
 * 宿主音频读数（插件模式）的解析与判定。
 *
 * 【为什么单独一个模块】这份读数有两个消费点 —— 时间轴上方的一次性提示条
 * （`HostAudioNotice`）与片段本身的"等待宿主音频"样式。两处必须对**同一个分类名**
 * 得出同一个结论，否则会出现"提示条说挂错了轨道、片段却看起来正常"这种自相矛盾。
 * 判定逻辑集中在这里，两边都从这里取。
 *
 * 【为什么必须校验分类名】后端将来可能加新的分类。旧前端若把不认识的字符串当成
 * 已知状态渲染，就会给出一条**错误的指引**（用户照着改工程，问题更糟）。所以只认
 * 白名单里的分类名，其余一律当作"本次没带该字段"，沿用上一次已知值。
 */
import type { HostAudioPayload, HostAudioState } from "../../types/api";

/** 已知分类名白名单；与后端 `render::extension::HostAudioState::as_str` 一一对应。 */
export const HOST_AUDIO_STATES: ReadonlySet<string> = new Set<HostAudioState>([
    "ready",
    "awaiting_regions",
    "folder_parent_without_regions",
]);

/**
 * 解析后端载荷。返回 `null` 表示"没有可用读数"（字段缺失、分类未知、类型不对），
 * 调用方应沿用上一次已知值而不是清空或猜测。
 */
export function parseHostAudio(raw: unknown): HostAudioPayload | null {
    if (typeof raw !== "object" || raw === null) return null;
    const candidate = raw as { state?: unknown; waiting_clips?: unknown };
    if (typeof candidate.state !== "string" || !HOST_AUDIO_STATES.has(candidate.state)) {
        return null;
    }
    const count = Number(candidate.waiting_clips);
    return {
        state: candidate.state as HostAudioState,
        waiting_clips: Number.isFinite(count) && count > 0 ? Math.floor(count) : 0,
    };
}

/**
 * 是否需要"插件挂在轨道组父轨上"这条提示。
 *
 * 【为什么只有这一种情形提示】`awaiting_regions` 是加载过程中的正常中间态（状态栏
 * 已有的 `等待宿主音频` 读数覆盖了它），为它弹一条横条只会在每次打开工程时打扰用户。
 * 而 folder 父轨是**用法问题**：症状（轨道和片段都看得见、能拖、但没内容）极具
 * 误导性，不给提示用户根本无从判断。
 */
export function needsFolderTrackNotice(status: HostAudioPayload | null): boolean {
    return status?.state === "folder_parent_without_regions";
}

/**
 * 该片段是否"还在等宿主音频"。
 *
 * 【判据沿用既有的 `source_path` 缺失】占位片段没有 `source_path`（见后端
 * `present_host_inventory` 与 `retain_display_waveforms`：未经 ARA 授权不得读 PCM）。
 *
 * MIDI / 音高参考片段同样没有 `source_path`，但它有音符内容可画，不该被标成"等待音频"。
 *
 * 【与 `hostMediaState` 的关系】本函数只回答"要不要画占位"，不回答"为什么"。
 * 悬停文案必须用 [`hostMediaState`] —— 一个布尔承载不了四种成因。
 */
export function isAwaitingHostAudio(clip: {
    sourcePath?: string | null;
    midiNoteCount?: number | null;
}): boolean {
    return !clip.sourcePath && clip.midiNoteCount == null;
}

/** 后端给出的逐 clip 宿主媒体分类；与 `render::editor::workspace` 一一对应。 */
export const HOST_MEDIA_STATES: ReadonlySet<string> = new Set([
    "ready",
    "pending",
    "unavailable",
    "reversed",
]);

export type HostMediaState = "ready" | "pending" | "unavailable" | "reversed";

/**
 * 该片段为什么没有（或已有）音频。
 *
 * 【为什么不能只用一个布尔】"没有 `source_path`"同时命中的情形里，只有一种是故障：
 * - `pending` —— 刚分割/裁切/粘贴，宿主还在分配 region。**在途**，会自己好。
 * - `unavailable` —— FX 挂在 folder 父轨上，本实例**永远**拿不到组内音频。**用法问题**。
 * - `reversed` —— 倒放片段被隔离：ARA 不给反向 PCM。**宿主在处理**。
 * - `ready` —— 正常。
 *
 * 四者此前共用一句"等待 REAPER 提供音频（未分配 ARA 区域）"，于是"我刚分割了一下"
 * 看起来像"插件坏了"。分类名由后端给（语言无关），文案在这里查 catalog。
 *
 * 缺省（独立 App、或旧后端不带该字段）时按 `source_path` 推断 —— 沿用旧行为，
 * 不猜一个新状态。
 */
export function hostMediaState(clip: {
    sourcePath?: string | null;
    midiNoteCount?: number | null;
    hostMedia?: string | null;
}): HostMediaState {
    const raw = clip.hostMedia;
    if (typeof raw === "string" && HOST_MEDIA_STATES.has(raw)) {
        return raw as HostMediaState;
    }
    return isAwaitingHostAudio(clip) ? "unavailable" : "ready";
}

/**
 * `unavailable` 的**原因码**；与后端 `decorate_host_media_locked` 一一对应。
 *
 * 【为什么要有原因码】`unavailable` 单独一句"等待 REAPER 提供音频"回答不了用户最需要
 * 的那个问题：该等、该改用法、还是该撤销。原因码把三种完全不同的处置分开：
 * - `take_switched` —— 宿主换了这一条的当前 Take，而 ARA 尚未重新认领
 *   （典型来源：REAPER 的"倒放 Item 为新 Take"）。**可操作**：撤销或再编辑一次。
 * - `unclaimed` —— 这个 item 从未被本实例的 ARA region 认领。
 * - `folder_parent` —— FX 挂在 folder 父轨，本实例永远拿不到组内音频。**用法问题**。
 * - `awaiting_region` —— 等超时了仍没有 region；多半是宿主侧出了别的岔子。
 * - `direction_unknown` —— 连这一条**是不是倒放**都读不出来（`PCM_Source_GetSectionInfo`
 *   不可用）。再等也不会变好；如实说明，不宣称"是正放"。
 */
export const HOST_MEDIA_REASONS: ReadonlySet<string> = new Set([
    "take_switched",
    "unclaimed",
    "folder_parent",
    "awaiting_region",
    "direction_unknown",
]);

export type HostMediaReason =
    | "take_switched"
    | "unclaimed"
    | "folder_parent"
    | "awaiting_region"
    | "direction_unknown";

/**
 * 该片段 `unavailable` 的原因码；未知或缺失返回 `null`（调用方退回通用文案）。
 */
export function hostMediaReason(clip: { hostMediaReason?: string | null }): HostMediaReason | null {
    const raw = clip.hostMediaReason;
    if (typeof raw === "string" && HOST_MEDIA_REASONS.has(raw)) {
        return raw as HostMediaReason;
    }
    return null;
}
