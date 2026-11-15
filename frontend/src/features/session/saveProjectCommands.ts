/**
 * 「另存为」各入口的调度。
 *
 * 【为什么集中在这里】同一条流程有五个入口：文件菜单的子项、三个可绑定快捷键、
 * `project.saveAs`、首次保存（还没有工程路径）、版本冲突里的「另存为」。
 * 让每个入口各自拼一遍 `getState().session.project.lastSaveFormat ?? "hshp"`
 * 与派发，就会出现"某个入口忘了带格式"这一类静默分叉 —— 而格式正是这条流程
 * 唯一的必填参数。
 */
import type { AppDispatch, RootState } from "../../app/store";
import { saveProjectAsFormat } from "./thunks/projectThunks";
import { saveAsFormatForAction } from "./saveProjectAsMenu";
import type { ProjectSaveFormat } from "../../services/api/project";

/**
 * 按「上次保存格式」直接另存（`project.saveAs` / 首次保存）。
 *
 * 返回 thunk 自己的 promise：不需要结果的调用方 `void` 掉即可，需要判断
 * "是否取消 / 是否版本冲突"的调用方（未保存确认框）直接 `await ...unwrap()`。
 */
export function saveProjectAsWithLastFormat(dispatch: AppDispatch, getState: () => RootState) {
    return dispatch(saveProjectAsFormat(getState().session.project.lastSaveFormat ?? "hshp"));
}

/** 按指定格式另存（版本冲突重试：保留冲突时已知的格式）。 */
export function saveProjectAsWithFormat(
    dispatch: AppDispatch,
    format: ProjectSaveFormat | undefined,
): void {
    void dispatch(saveProjectAsFormat(format ?? "hshp"));
}

/** 三个「另存为某格式」动作：命中则派发并返回 `true`，未命中返回 `false`。 */
export function runSaveAsFormatAction(dispatch: AppDispatch, actionId: string): boolean {
    const format = saveAsFormatForAction(actionId);
    if (!format) return false;
    void dispatch(saveProjectAsFormat(format));
    return true;
}
