// 外部文件动作事件定义。
//
// 用于在组件间传递外部文件触发的动作请求（打开工程/导入工程/导入音频）。

export const OPEN_PROJECT_PATH_EVENT = "hifi:open-project-path";

export type ExternalFileActionKind =
    | "openProject"
    | "importVocalShifter"
    | "importReaper"
    | "importAudio";

export type ExternalFileActionDetail = {
    kind: ExternalFileActionKind;
    path: string;
};

export function emitOpenProjectPath(path: string) {
    emitExternalFileAction("openProject", path);
}

export function emitExternalFileAction(
    kind: ExternalFileActionKind,
    path: string,
) {
    const normalized = String(path ?? "").trim();
    if (!normalized) return;
    window.dispatchEvent(
        new CustomEvent<ExternalFileActionDetail>(OPEN_PROJECT_PATH_EVENT, {
            detail: { kind, path: normalized },
        }),
    );
}
