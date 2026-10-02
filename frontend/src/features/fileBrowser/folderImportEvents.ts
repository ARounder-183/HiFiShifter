/**
 * 「导入文件夹」请求事件。
 *
 * 【为什么是事件而不是 Redux 字段】与 `IMPORT_MIDI_PATH_EVENT` 同一条理由：导入
 * 选项对话框的状态（扫描结果、展开计划、落点）只有一处能持有 —— 全局挂载的那一个
 * 宿主组件。三个入口（系统拖放 / 文件浏览器拖拽 / 右键菜单）各自只发一条请求，
 * 宿主接住并用自己的默认值打开对话框；比让每个入口各持一份对话框状态副本更不容易
 * 分叉（三个副本 = 三套默认值、三种关闭时机）。
 */

export const FOLDER_IMPORT_REQUEST_EVENT = "hifi:folder-import-request";

export interface FolderImportRequestDetail {
    /** 被拖入的目录绝对路径（可多个）。 */
    dirs: string[];
    /** 同时拖入的散文件（不属于任何目录）。 */
    looseFiles: string[];
    /** 落点轨道 id；`null` 表示没落在具体轨道上。 */
    trackId: string | null;
    /** 落点时间（秒）。 */
    startSec: number;
    /**
     * 新建根轨道的**根级**插入下标（`rootIndexAtDrop` 的结果）。
     *
     * 【为什么由发送方算】只有它知道落点命中哪一行、当时的轨道列表是什么；
     * 宿主只负责"放进去"。
     */
    insertIndex?: number | null;
    /**
     * 强制弹选项对话框。
     *
     * 用于"用户明确要求选"的入口：右键菜单的「导入文件夹…」，以及按住修饰键拖入。
     * 没有子目录、也没被截断时，其余入口会直接用记住的选项执行、不打扰用户。
     */
    force?: boolean;
}

export function emitFolderImportRequest(detail: FolderImportRequestDetail): void {
    const dirs = detail.dirs.filter((dir) => dir.trim().length > 0);
    if (dirs.length === 0) return;
    window.dispatchEvent(
        new CustomEvent<FolderImportRequestDetail>(FOLDER_IMPORT_REQUEST_EVENT, {
            detail: { ...detail, dirs },
        }),
    );
}
