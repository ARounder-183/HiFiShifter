export function createWindowCloseGuard(args: {
    isDisposed: () => boolean;
    consumeAllowClose: () => boolean;
    isProjectDirty: () => boolean;
    getProjectMeta: () => Promise<{ dirty: boolean }>;
    promptSave: () => void;
    closeWindow: () => Promise<void>;
}) {
    let checking = false;
    return async (event: { preventDefault: () => void }): Promise<void> => {
        if (args.isDisposed()) {
            event.preventDefault();
            return;
        }
        if (args.consumeAllowClose()) return;
        event.preventDefault();
        if (checking) return;
        checking = true;
        try {
            const dirty =
                args.isProjectDirty() ||
                (await args.getProjectMeta()).dirty ||
                args.isProjectDirty();
            if (args.isDisposed()) return;
            if (dirty) args.promptSave();
            else await args.closeWindow();
        } catch {
            // 无法确认工程已保存时，保留窗口并交给保存确认流程。
            if (!args.isDisposed()) args.promptSave();
        } finally {
            checking = false;
        }
    };
}
