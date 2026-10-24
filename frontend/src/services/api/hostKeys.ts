import { invoke } from "../invoke";

/**
 * 宿主专属快捷键的让路通道（仅插件形态有意义）。
 *
 * 【为什么单独一个分组】这两条命令是**成对**的：`relay` 是"这一次按键我不要了"，
 * `publishEscapeSet` 是"以下这些键我以后都不要"。分成两处登记会让"哪一侧是权威"
 * 说不清；放在一起，读者一眼看到它们共用同一套按键词汇（`key` + 四个修饰位）。
 */

/** 按键的修饰位。全部为假 = 裸键，也是唯一允许让路的形态。 */
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
     * 分两类，调用方据此决定要不要说话：
     * - `host-native` / `in-flight` / `rate-limited`：**正常不接手**（宿主本来就会处理、
     *   上一拍还没走完、连按太快）。用户什么都没做错，不该看到任何提示。
     * - `unsupported-key` / `modified-key` / `not-foreground` / `no-host-window`：
     *   让路**没成立**。前两者等于回到"插件做不到"的旧行为，调用方应回落旧文案；
     *   后两者由原生另行发 `host_key_relay_notice` 事件说话。
     */
    reason:
        | null
        | "host-native"
        | "in-flight"
        | "rate-limited"
        | "unsupported-key"
        | "modified-key"
        | "not-foreground"
        | "no-host-window";
}

export const hostKeysApi = {
    /**
     * 把这一次按键交回宿主。
     *
     * 【为什么 `action` 也传】它只用于日志与"每个动作每次会话只提示一次"的记账；
     * 原生**不**按它查表 —— 宿主的动作表不可枚举，插件也绝不替用户翻译键（见方案 R1）。
     */
    relay: (key: string, action: string, mods: HostKeyModifiers) =>
        invoke<HostKeyRelayReply>("host_key_relay", key, action, mods),

    /**
     * 声明"哪些键属于宿主"。
     *
     * 【为什么必需】`PassThrough` 平台要在**页面看到按键之前**决定让不让，而原生不知道
     * 用户的键位（键位在 Redux / localStorage 里）。集合由前端算好推过去，
     * 两端因此只有一套"哪些键该让"的判据（见 `collectHostKeyEscapeStrokes`）。
     */
    publishEscapeSet: (strokes: ReadonlyArray<{ key: string } & HostKeyModifiers>) =>
        invoke<{ ok: boolean; count: number }>("host_key_escape_set", strokes),

    /**
     * 把键盘焦点交还宿主。
     *
     * 【为什么与"让路"分开成一条命令】让路是"这个键我不要，你替我用"；交还焦点是
     * "整块键盘我都不要了"。后者一定会生效（宿主自己的键位表直接生效），代价是焦点
     * 真的走了 —— 所以它只由用户显式触发，从不自动发生。
     */
    focusHost: () => invoke<{ ok: boolean; reason?: string }>("host_key_focus"),
};
