/**
 * 「另存为指定格式」的唯一映射表：格式 → 文案 → 快捷键动作。
 *
 * 【为什么必须只有一张表】同一件事有三个消费方：文件菜单的子项文案、子项右侧的
 * 已绑定快捷键、以及全局键盘分派。三处各写一份 `match` 的结果是"菜单里能点、
 * 快捷键没反应"这一类静默分叉；新增一种格式时也必然漏改其中一处。
 *
 * 格式说明（`descKey`）与标签分开：说明要出现在子项的 tooltip 里（尤其
 * "zip 需解压后打开"是必须保留的教育信息），标签只写"另存为 XX"。
 */
import type { MessageKey } from "../../i18n/messages";
import type { ProjectSaveFormat } from "../../services/api/project";
import type { ActionId } from "../keybindings/types";

export interface SaveProjectAsFormatEntry {
    /** 写入文件的格式（由后端按此参数选序列化器）。 */
    readonly format: ProjectSaveFormat;
    /** 菜单子项标签。 */
    readonly labelKey: MessageKey;
    /** 子项 tooltip：这个格式适合什么、有什么代价。 */
    readonly descKey: MessageKey;
    /** 与之绑定的可配置动作（默认不占键位）。 */
    readonly actionId: ActionId;
}

/** 菜单顺序 = 默认格式优先（`.hshp` 是本应用的工程格式）。 */
export const SAVE_PROJECT_AS_FORMATS: readonly SaveProjectAsFormatEntry[] = [
    {
        format: "hshp",
        labelKey: "menu_save_project_as_hshp",
        descKey: "save_as_desc_hshp",
        actionId: "project.saveAsHshp",
    },
    {
        format: "json",
        labelKey: "menu_save_project_as_json",
        descKey: "save_as_desc_json",
        actionId: "project.saveAsJson",
    },
    {
        format: "zip",
        labelKey: "menu_save_project_as_zip",
        descKey: "save_as_desc_zip",
        actionId: "project.saveAsZip",
    },
];

/**
 * 动作 → 格式。
 *
 * 返回 `null` 表示这个动作不是"另存为指定格式"。用 `string` 而不是 `ActionId`
 * 入参：键盘分派拿到的就是动作名，调用点不必先做一次类型断言。
 */
export function saveAsFormatForAction(actionId: string): ProjectSaveFormat | null {
    const entry = SAVE_PROJECT_AS_FORMATS.find((item) => item.actionId === actionId);
    return entry ? entry.format : null;
}
