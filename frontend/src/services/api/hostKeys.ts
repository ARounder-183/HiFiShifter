import { invoke } from "../invoke";

/**
 * 宿主专属快捷键的让路通道（仅插件形态有意义）。
 *
 * 【为什么单独一个分组】这两条命令是**成对**的：`relay` 是"这一次按键我不要了"，
 * `publishOwnedStrokes` 是"以下这些键我自己要用"。分成两处登记会让"哪一侧是权威"
 * 说不清；放在一起，读者一眼看到它们共用同一套按键词汇（`code` + 四个修饰位）。
 */

/** 按键的修饰位。全部为假 = 裸键，也是唯一允许让路的形态（见 hostKeyRelay）。 */
export interface HostKeyModifiers {
    ctrl: boolean;
    shift: boolean;
    alt: boolean;
    meta: boolean;
}

/** 一次让路请求的回执。 */
export interface HostKeyRelayReply {
    accepted: boolean;
    /**
     * 不接手的原因（`null` = 已接手）。
     *
     * 分三类，调用方据此决定要不要回落旧文案：
     * - `host-native` / `in-flight` / `rate-limited`：**正常不接手**（宿主本来就会处理、
     *   上一拍的探针还没走完、连按太快）。用户什么都没做错，不该看到任何提示。
     * - `unsupported-key` / `modified-key`：让路**没成立**，等于回到"插件做不到"的
     *   旧行为，调用方应回落旧文案。
     * - `owned-key`：原生一致性闸拒绝投递 —— 这是入口 B（未绑定键）的投递撞上了
     *   插件自己的键位表（两侧键表漂移或发布竞态）。对调用方等价于"已接手"：
     *   未绑定键没有可回落的文案，证据在诊断读数的 `refused["owned-key"]` 里。
     */
    reason:
        | null
        | "host-native"
        | "in-flight"
        | "rate-limited"
        | "unsupported-key"
        | "modified-key"
        | "owned-key";
}

export const hostKeysApi = {
    /**
     * 把这一次按键交回宿主。
     *
     * 【为什么 `action` 也传】它只用于日志，以及"这一次要不要先探针"：
     * - 非空 = 插件**认识**这个键（命中过绑定、被能力矩阵拒绝）⇒ 先探针，防双触发；
     * - 空串 = 一条绑定都没命中 ⇒ 直接投，插件本来就不处理它。
     *
     * 原生**不**按 `action` 查表 —— 宿主的动作表不可枚举，插件也绝不替用户翻译键。
     */
    relay: (code: string, action: string | null, mods: HostKeyModifiers) =>
        invoke<HostKeyRelayReply>("host_key_relay", code, action ?? "", mods),

    /**
     * 声明"插件自己占用了哪些键"。
     *
     * 【为什么是"占用"而不是"可让"】可让的是**未绑定**的那些键 —— 那是一个没法枚举
     * 的补集（用户随时改键）。而"插件占用了什么"恰好是有限的、前端本来就算得出来的
     * 那份。原生据此拒绝"页面借宿主的手按一个它自己也在用的键"（最坏的情形是
     * `Delete`：插件删片段，宿主删它自己的对象）。
     */
    publishOwnedStrokes: (strokes: ReadonlyArray<{ code: string } & HostKeyModifiers>) =>
        invoke<{ ok: boolean; count: number }>("host_key_owned_set", strokes),

    /**
     * 把键盘焦点交还宿主。
     *
     * 【为什么与"让路"分开成一条命令】让路是"这个键我不要，你替我用"；交还焦点是
     * "整块键盘我都不要了"。后者一定会生效（宿主自己的键位表直接生效），代价是焦点
     * 真的走了 —— 所以它只由用户显式触发，从不自动发生。
     */
    focusHost: () => invoke<{ ok: boolean; reason?: string }>("host_key_focus"),
};
