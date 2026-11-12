/**
 * 「目标已存在版本不一致工程文件」确认框的三个动作。
 *
 * 【为什么单独一个 hook】这段逻辑与另存为流程强耦合（另存为要保留冲突时的格式，
 * 继续保存要带上同一格式强制覆盖），却散落在 `App` 的十几个 `useCallback` 中间。
 * 抽出来之后 `App` 只留一行调用，也避免"三个入口各写一遍格式回落"的分叉。
 */
import { useCallback } from "react";
import type { AppDispatch } from "../../app/store";
import { closeSaveVersionConflictDialog } from "./sessionSlice";
import { saveProjectToPathRemote } from "./thunks/projectThunks";
import { saveProjectAsWithFormat } from "./saveProjectCommands";
import type { ProjectSaveFormat } from "../../services/api/project";

/** 冲突信号里携带的"目标路径 + 当时选定的格式"。 */
export interface SaveVersionConflictTarget {
    readonly path: string;
    readonly format?: ProjectSaveFormat;
}

export function useSaveVersionConflictActions(
    dispatch: AppDispatch,
    target: SaveVersionConflictTarget | null | undefined,
) {
    const cancel = useCallback(() => {
        dispatch(closeSaveVersionConflictDialog());
    }, [dispatch]);

    // 「另存为」：关闭确认框，按**原路径已选定的格式**重新选路径保存。
    const saveAs = useCallback(() => {
        dispatch(closeSaveVersionConflictDialog());
        saveProjectAsWithFormat(dispatch, target?.format);
    }, [dispatch, target?.format]);

    // 「继续保存」：向已选路径执行强制覆盖保存。
    const overwrite = useCallback(() => {
        dispatch(closeSaveVersionConflictDialog());
        if (!target?.path) return;
        void dispatch(saveProjectToPathRemote({ path: target.path, format: target.format }));
    }, [dispatch, target?.path, target?.format]);

    return { cancel, saveAs, overwrite };
}
