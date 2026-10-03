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
     * 本次请求是用户**显式发起**的（右键菜单的「导入文件夹…」、右键拖入）。
     *
     * 【它现在唯一的作用：无媒体时的表现】目录导入一律弹选项对话框，因此这个标记
     * 不再决定"弹不弹"。它只决定"这个文件夹里一个媒体文件都没有"时怎么办：
     * 显式请求仍然弹窗（用「没有媒体文件」+ 禁用的「导入」按钮给出解释，比什么都
     * 不发生更好），隐式拖入则直接返回，不建出一条空轨道。
     */
    fromExplicitRequest?: boolean;
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
