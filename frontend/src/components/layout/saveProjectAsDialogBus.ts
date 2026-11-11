import type { ProjectSaveFormat } from "../../services/api/project";

export const OPEN_SAVE_PROJECT_AS_EVENT = "hifi:saveProjectAs";

/** 从任意命令入口请求打开应用内“另存为格式”对话框。 */
export function requestSaveProjectAsDialog(format?: ProjectSaveFormat): void {
    window.dispatchEvent(new CustomEvent(OPEN_SAVE_PROJECT_AS_EVENT, { detail: { format } }));
}
