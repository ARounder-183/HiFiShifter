/*
 * 剪贴板预览缓存的「对齐槽位」策略。
 *
 * ## 为什么需要它
 *
 * 参数编辑器的剪贴板预览画的就是**粘贴会落下的数据**（见 `clipboardPreviewSpans`），
 * 因此它的取数口径必须与粘贴一致：以**系统槽位**为准，而不是"本面板最近一次
 * 复制过什么"。槽位被别的表面整体替换时会派发 `hifi:clipboardReplaced`
 * （时间轴复制 Clip、记事本暂存块的「恢复到剪贴板」），此时内部缓存要重新对齐
 * 槽位。
 *
 * 【只清不读曾是一个缺口】彼时收到该事件只把缓存清空：时间轴复制 Clip 的场景
 * 看着没问题（槽位里确实不再有参数线数据），但从记事本恢复一份**参数线**载荷
 * 时，槽位里明明躺着可粘贴的数据，预览却始终空白 —— 用户能贴进去，却看不见
 * 会贴成什么样。
 *
 * ## 为什么是"工厂"而不是一个纯函数
 *
 * 对齐要读系统剪贴板（异步），而槽位可能被连续替换：先发起的读取若晚于后发起的
 * 返回，就会用**已经过期的槽位内容**覆盖预览，直接违反"剪贴板只保留最后一份"
 * 的纪律。因此这里持有单调令牌，只认最后一次请求的结果，并允许卸载时取消 ——
 * 这段竞态规则是纯逻辑，独立成模块才可被单测覆盖（面板组件本身无法单测）。
 */

import type { ParamClipboardData } from "./paramClipboardMapping";

/** 读取槽位里的参数线载荷；无数据/不可读时返回 null。 */
export type ParamClipboardReader = () => Promise<ParamClipboardData | null>;

export interface ClipboardPreviewSync {
    /**
     * 对齐一次：读取槽位，并把结果交给 `apply`（读到参数线数据则给出数据，
     * 否则给出 null —— 调用方据此清空预览，不残留上一次的曲线）。
     */
    sync(apply: (data: ParamClipboardData | null) => void): Promise<void>;
    /** 取消：之后到达的读取结果不再交给 `apply`（组件卸载时调用）。 */
    cancel(): void;
}

export function createClipboardPreviewSync(read: ParamClipboardReader): ClipboardPreviewSync {
    let token = 0;
    return {
        async sync(apply) {
            const mine = (token += 1);
            let data: ParamClipboardData | null = null;
            try {
                data = await read();
            } catch {
                // 槽位不可读（外来数据 / 平台不支持该格式）等价于"没有参数线数据"：
                // 预览应当清空，而不是留着上一次的曲线骗人。
                data = null;
            }
            // 期间又对齐过（或已取消）：本次结果已过期，丢弃。
            if (mine !== token) return;
            apply(data);
        },
        cancel() {
            token += 1;
        },
    };
}
