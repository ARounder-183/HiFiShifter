/**
 * 导入媒体文件时，为"因此新建的轨道"取的名字。
 *
 * 【为什么要单独一个模块】此前 `importThunks.fileStem` 与 `notebookPaths.stemName`
 * 各写了一份"去掉最后一个扩展名"；而"导入建轨以首个媒体文件命名"这条规则是
 * **业务语义**，不该藏在某个 thunk 的私有函数里。抽出来之后，规则只有一处。
 *
 * 【为什么 `dot > 0` 而不是 `dot >= 0`】`.gitignore` 这类以点开头的文件没有"主名"
 * 概念，去掉后缀会得到空串 —— 保留原名。无扩展名的文件同理。
 */

/**
 * 取路径的主名（去掉最后一个扩展名，保留目录部分之外的裸文件名）。
 *
 * 同时接受完整路径与裸文件名：`C:\a\vocal take 01.wav` 与 `vocal take 01.wav`
 * 都得到 `vocal take 01`。分隔符按 `/` 与 `\` 两种都认（Windows 路径混用两种）。
 */
export function trackNameForMedia(path: string): string {
    const base = path.split(/[\\/]/).pop() ?? path;
    const dot = base.lastIndexOf(".");
    return dot > 0 ? base.slice(0, dot) : base;
}
