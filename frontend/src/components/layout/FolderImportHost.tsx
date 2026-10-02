/**
 * 「导入文件夹」的宿主：监听请求、扫描、决定要不要问、执行导入。
 *
 * 【为什么集中在一个全局挂载的组件里】三个入口（系统拖放 / 文件浏览器拖拽 /
 * 右键菜单）都可能发起目录导入，而它们分布在不同的组件树位置。把"扫描 → 决定 →
 * 对话框 → 导入"这一整条链放在这里，入口只需发一条请求事件：
 *   - 只有一份对话框状态，不会出现"两个面板各弹一个"；
 *   - 默认值、重新扫描的时机、记住选项的写入点都只有一处。
 *
 * 【为什么"没有子目录就不弹窗"】递归选项只在有子目录时才有意义，此时弹窗等于用一次
 * 点击换一个用户没得选的确认。但**截断必须弹** —— 那是"你要导入的东西比上限还多"，
 * 不告知就等于静默少导入。判断集中在 `shouldPromptFolderImport`，可单测。
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { FolderMediaScan } from "../../services/api/fileBrowser";
import { fileBrowserApi } from "../../services/api";
import {
    FOLDER_IMPORT_REQUEST_EVENT,
    type FolderImportRequestDetail,
} from "../../features/fileBrowser/folderImportEvents";
import {
    buildFolderImportPlan,
    shouldPromptFolderImport,
} from "../../features/fileBrowser/folderImportPlan";
import {
    normalizeFolderImportOptions,
    type FolderImportOptions,
} from "../../features/fileBrowser/folderImportOptions";
import {
    importFolderAtPosition,
    type FolderImportTreeNode,
} from "../../features/session/thunks/importThunks";
import { persistUiSettings, setFolderImportOptions } from "../../features/session/sessionSlice";
import { FolderImportDialog } from "./FolderImportDialog";

/** 计划节点 → thunk 载荷（去掉 `dir`：导入不需要它）。 */
function toPayloadNode(node: {
    name: string;
    files: string[];
    children: unknown[];
}): FolderImportTreeNode {
    return {
        name: node.name,
        files: node.files,
        children: (node.children as Parameters<typeof toPayloadNode>[0][]).map(toPayloadNode),
    };
}

export function FolderImportHost() {
    const dispatch = useAppDispatch();
    const options = useAppSelector((state) => state.session.folderImportOptions);
    const showHiddenFiles = useAppSelector(
        (state) => state.session.fileBrowserView.showHiddenFiles,
    );

    const [request, setRequest] = useState<FolderImportRequestDetail | null>(null);
    const [scanning, setScanning] = useState(false);
    const [scan, setScan] = useState<FolderMediaScan | null>(null);
    const [draft, setDraft] = useState<FolderImportOptions>(options);

    /*
     * 事件监听只在挂载时注册一次，因此它必须读到**最新**的选项与隐藏开关，而不是
     * 注册那一刻的闭包值。用 ref 承载"最新值"，写入放在 effect 里 —— 渲染期写 ref
     * 是 React 明确不建议的（并发渲染下可能被丢弃）。
     */
    const optionsRef = useRef(options);
    const hiddenRef = useRef(showHiddenFiles);
    useEffect(() => {
        optionsRef.current = options;
    }, [options]);
    useEffect(() => {
        hiddenRef.current = showHiddenFiles;
    }, [showHiddenFiles]);

    const runImport = useCallback(
        (detail: FolderImportRequestDetail, result: FolderMediaScan, opts: FolderImportOptions) => {
            const plan = buildFolderImportPlan(result.groups, detail.looseFiles);
            void dispatch(
                importFolderAtPosition({
                    roots: plan.roots.map(toPayloadNode),
                    looseFiles: plan.looseFiles,
                    orderedFiles: plan.orderedFiles,
                    mode: opts.mode,
                    createFolderTracks: opts.createFolderTracks,
                    trackId: detail.trackId,
                    startSec: detail.startSec,
                    insertIndex: detail.insertIndex ?? null,
                }),
            );
        },
        [dispatch],
    );

    const collect = useCallback(
        (dirs: string[], recursive: boolean) =>
            fileBrowserApi.collectFolderMedia(dirs, {
                recursive,
                // 与目录列表同口径：列表里被隐藏的项，导入时也不该悄悄进来。
                includeHidden: hiddenRef.current,
            }),
        [],
    );

    useEffect(() => {
        const onRequest = (event: Event) => {
            const detail = (event as CustomEvent<FolderImportRequestDetail>).detail;
            if (!detail || detail.dirs.length === 0) return;
            void (async () => {
                const opts = optionsRef.current;
                let result: FolderMediaScan;
                try {
                    result = await collect(detail.dirs, opts.recursive);
                } catch {
                    // 扫描失败（IPC 异常）：什么都不做，而不是拿半截结果去导入。
                    return;
                }
                const prompt = shouldPromptFolderImport({
                    hasSubdirs: result.groups.some((group) => group.hasSubdirs),
                    truncated: result.truncated,
                    force: detail.force,
                });
                if (!prompt) {
                    runImport(detail, result, opts);
                    return;
                }
                setDraft(opts);
                setScan(result);
                setScanning(false);
                setRequest(detail);
            })();
        };
        window.addEventListener(FOLDER_IMPORT_REQUEST_EVENT, onRequest);
        return () => window.removeEventListener(FOLDER_IMPORT_REQUEST_EVENT, onRequest);
    }, [collect, runImport]);

    const plan = useMemo(
        () => buildFolderImportPlan(scan?.groups ?? [], request?.looseFiles ?? []),
        [request, scan],
    );

    const handleOptionsChange = useCallback(
        (patch: Partial<FolderImportOptions>) => {
            const next = normalizeFolderImportOptions({ ...draft, ...patch });
            setDraft(next);
            // 记住这次的选择：没有子目录的目录下次直接按这些值执行、不再打扰。
            dispatch(setFolderImportOptions(patch));
            void dispatch(persistUiSettings());
            if (patch.recursive === undefined || !request) return;
            // 递归与否决定"要导入的文件集合是什么"，只能重新枚举。
            setScanning(true);
            void collect(request.dirs, next.recursive)
                .then((result) => {
                    setScan(result);
                    setScanning(false);
                })
                .catch(() => setScanning(false));
        },
        [collect, dispatch, draft, request],
    );

    const handleConfirm = useCallback(() => {
        if (!request || !scan) return;
        runImport(request, scan, draft);
        setRequest(null);
    }, [draft, request, runImport, scan]);

    const handleOpenChange = useCallback((next: boolean) => {
        if (!next) {
            setRequest(null);
            setScan(null);
            setScanning(false);
        }
    }, []);

    // 未打开时不渲染任何东西（对话框自身也认 `open`，这里只是省掉一次挂载）。
    if (!request) return null;
    return (
        <FolderImportDialog
            open
            onOpenChange={handleOpenChange}
            scan={scan}
            scanning={scanning}
            plan={plan}
            options={draft}
            onOptionsChange={handleOptionsChange}
            onConfirm={handleConfirm}
        />
    );
}
