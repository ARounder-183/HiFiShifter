/**
 * 「历史位置被整体改变」的统一入口。
 *
 * 【什么时候算】撤销 / 重做 / 跳转到「操作记录」里的某一步，以及打开 / 新建工程
 * —— 它们都会把时间线整体换成另一份快照（而不是在上面做一处小改动）。
 *
 * 【为什么必须收口到一个函数】这些入口都要做同样两件事，而两件事都必须做：
 *   ① 否决在途的多步骤导入（`importCancellation`）—— 否则循环会继续往已消失的
 *      trackId 上灌 clip，把撤销栈写坏；
 *   ② 通知 UI 层复位那些"描述旧时间线"的瞬时状态（音高分析进度）—— 否则状态栏
 *      会一直停在"正在分析音高"，而分析的对象已经不存在了。
 * 散在各处各写一遍，漏一处就是"撤销之后状态栏卡住"。
 */

import { cancelActiveImportsAndDrain } from "./thunks/importCancellation";

/** 历史位置被整体改变时广播的事件名。 */
export const HISTORY_JUMP_EVENT = "hifi:history-jump";

/**
 * 否决在途导入（**等它们收尾**）+ 通知 UI 复位旧时间线的瞬时状态。
 *
 * 调用方必须 `await` 之后再执行真正的历史跳转：导入的每条后端命令都是独立 IPC，
 * 不等它收尾就跳转，那条命令会在跳转**之后**才落地（轨道组重新出现 / clip 写进
 * 已消失的轨道），撤销看起来就"没撤干净"。
 */
export async function notifyHistoryJump(): Promise<void> {
    await cancelActiveImportsAndDrain();
    window.dispatchEvent(new CustomEvent(HISTORY_JUMP_EVENT));
}
