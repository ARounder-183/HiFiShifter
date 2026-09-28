/*
 * 附件索引的读写。
 *
 * 索引只存摘要（id / 类型 / 大小 / 元数据），字节永远留在后端磁盘上 ——
 * 正文里可能有几十个引用，把图片数据全部读进内存只为回答"存在吗"是浪费。
 *
 * 独立成模块而不是放在 NodeView 里：多个入口（面板、暂存块、附件管理器）
 * 都要刷新索引，放在组件文件里既会造成 Fast Refresh 失效，也会让组件之间
 * 互相 import。
 */

import type { AppDispatch } from "../../../app/store";
import { setNotebookAssetIndex } from "../../../features/notebook/notebookSlice";
import { notebookApi } from "../../../services/api/notebook";

/** 从后端拉取附件清单并写入 Redux。插入/删除附件后调用。 */
export async function refreshAssetIndex(dispatch: AppDispatch): Promise<void> {
    const result = await notebookApi.listAssets();
    if (result.ok) dispatch(setNotebookAssetIndex(result.assets));
}
