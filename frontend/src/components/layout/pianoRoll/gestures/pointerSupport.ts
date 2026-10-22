/**
 * 指针手势拆出去之后仍然共用的几个小件。
 *
 * 【为什么单独成文件】这四个声明原本是 `usePianoRollInteractions.ts` 的**文件级**成员：
 * 光标类型、点击死区、以及两个只被指针回调用到的小工具。把回调搬进
 * `canvasPointerDown.ts` 之后，它们两边都要用 —— 留在原文件会让两个模块互相 import
 * （运行时循环）。搬到这里，谁用谁 import。
 *
 * 【`CanvasCursor` 为什么也在这里】hook 的 `setCanvasCursor` 参数类型要用它，回调里
 * 也要用；它是纯类型，跟着使用者走最省事。
 */
import type { ParamViewSegment } from "../types";
import type { ParamSelection } from "../paramSelection";
import { paramsApi } from "../../../../services/api/params";

export type CanvasCursor = "default" | "crosshair" | "grab" | "grabbing" | "ew-resize";

/**
 * pv（可能被降采样）的逐帧读取器。
 *
 * 只用于**即时预览**：pv 是显示数据（低缩放下按画布宽度降采样），拿去回写会
 * 覆盖全分辨率曲线，因此提交路径必须另取全分辨率数据。预览与提交共用同一套
 * 变换函数（见 selectionEditData），所以两者结果一致。
 */
export function makePvValueSource(pv: ParamViewSegment): (frame: number) => number {
    const step = Math.max(1, Math.floor(pv.stride));
    return (frame: number) => {
        const idx = Math.round((frame - pv.startFrame) / step);
        return idx >= 0 && idx < pv.edit.length ? pv.edit[idx] : 0;
    };
}

/**
 * 把一次「改数据并连带移动选区」的拖动，登记到它**刚产生的那个历史步骤**上。
 *
 * 【为什么需要】这类手势（选区拉伸 / 选区整体拖动）在改曲线的同时也改了选区
 * 范围，撤销时曲线会回退 —— 若选区留在拖后的位置，用户看到的就是"曲线回来了、
 * 选区没回来"，也就无法"回到原范围再拉一次 / 再拖一次"。
 *
 * 【为什么由后端持有】选区快照记在后端的历史记录上（与记事本同一套路），撤销/
 * 重做时随载荷带回（`param_selection_restore`），前端不再按撤销深度推断"我这一步
 * 是第几步" —— 那条路会因深度镜像滞后、写回被抑制、以及"新检查点丢弃重做分支"
 * 而错位（详见 state.rs 的说明）。
 *
 * 【调用时机】必须在曲线回写成功之后（`set_param_frames` 首块已打检查点、位置
 * 已指向新步）。这就是全部约束：没有位置推断，也就没有需要等待的时序信号。
 *
 * 失败只记日志：后端拒绝意味着这次回写没有产生可撤销的步骤（被抑制等），此时
 * 本就没有步骤可挂，重试无意义。
 */
export async function recordGestureParamSelectionStep(args: {
    before: ParamSelection | null;
    after: ParamSelection | null;
}): Promise<void> {
    // 线上形状是 `[startFrame, frameCount]`（帧制，与 `FrameRange` 同形）；后端把它
    // 存进该步的历史记录，撤销/重做时随载荷带回。
    const toPairs = (selection: ParamSelection | null): [number, number][] =>
        (selection ?? [])
            .filter(
                (range) => Number.isFinite(range.startFrame) && Number.isFinite(range.frameCount),
            )
            .map((range) => [range.startFrame, range.frameCount]);
    try {
        await paramsApi.recordParamSelectionStep(toPairs(args.before), toPairs(args.after));
    } catch (err) {
        console.warn("[pianoRoll] record param selection step failed", err);
    }
}

/**
 * **数据写入型**拖动（拖动参数线 / 拉伸选区内曲线）的点击死区（CSS px，逐轴判定）。
 *
 * 【为什么必须有】这两条路径的收尾都会**回写后端**并打一个撤销检查点。用户只是
 * 在参数线上点一下（或按住拉伸修饰键在边缘点一下）时，指针没有任何位移，变换结果
 * 与原始数据逐帧相同 —— 于是"什么都没做"的一次点击会写回一份等值数据，还多出
 * 一条按下去毫无变化的撤销记录。
 *
 * 【阈值为什么逐轴判定】"选择"工具的默认拖动方向是 `y-only`（见 selectDragDirection），
 * 只判水平位移会漏掉竖直方向的手抖；取两轴较大的那个更贴合"用户到底动了没有"。
 *
 * 只用于**会写数据**的手势；纯选区调整（边缘调整 / 右键平移 / 框选）不写任何东西，
 * 因此不需要死区。
 */
export const PARAM_DRAG_DEAD_ZONE_PX = 3;
