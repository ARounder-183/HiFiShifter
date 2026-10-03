/**
 * 「导入文件夹」的宿主：监听请求、扫描、弹对话框、执行导入。
 *
 * 【为什么集中在一个全局挂载的组件里】三个入口（系统拖放 / 文件浏览器拖拽 /
 * 右键菜单）都可能发起目录导入，而它们分布在不同的组件树位置。把"扫描 → 弹窗 →
 * 对话框 → 导入"这一整条链放在这里，入口只需发一条请求事件：
 *   - 只有一份对话框状态，不会出现"两个面板各弹一个"；
 *   - 默认值、重新扫描的时机、记住选项的写入点都只有一处。
 *
 * 【为什么目录导入一律弹窗（而多文件导入不弹）】多文件的选择只有"怎么排"一件事，
 * 三个选项一句话说得完，用菜单点一下最快。目录导入多出两个**正交**的问题（要不要
 * 下钻子目录、要不要为文件夹建轨道组），后者还只在「跨轨道添加」下有效 —— 这不是
 * 菜单能表达的形状。文件多少不改变这个形状：即使只有一个文件，用户拖的是**一个
 * 文件夹**，他期待的是"这个文件夹怎么进来"。
 *
 * 【为什么无媒体的文件夹要单独挡一道】用户拖入一个不含任何媒体文件的目录时，
 * 弹窗没有意义（导入按钮必然是禁用的）。此时按入口来源区分：显式请求（右键菜单 /
 * 右键拖入）照旧弹窗，用「没有媒体文件」这句解释代替静默；隐式拖入则直接返回 ——
 * 否则会静默建出一条空轨道。
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
    hasImportableMedia,
    needsRecursiveAdmissionProbe,
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
    /*
     * 重新扫描的序号：递归开关每次切换都自增，在途的旧扫描据此作废。
     *
     * 【为什么必须有】来回切两次"递归"会并发两次 `collect`，而先发的（旧口径）结果
     * 可能后到，把 `scan` 覆盖成与当前 `draft.recursive` 不符的列表 —— 摘要计数看着
     * 自洽，「导入」却只进了顶层文件。只认最新序号的结果。
     */
    const scanSeqRef = useRef(0);
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
                /*
                 * 准入判定必须**递归**。
                 *
                 * 【为什么不能直接用当前扫描结果】`recursive` 选项默认关闭，而一个
                 * 文件夹的媒体文件可能全在子目录里 —— 用非递归的扫描结果判定会把这种
                 * 文件夹挡在对话框之外，用户连"递归导入"都选不到。需求正是"除非该
                 * 文件夹的子文件夹（递归判定）包含媒体文件，否则不可导入"。
                 *
                 * 【为什么只在必要时多扫一次】非递归扫描已找到媒体、或递归本来就开着、
                 * 或压根没有子目录时，都不需要第二趟。
                 */
                let admitted = hasImportableMedia(
                    buildFolderImportPlan(result.groups, detail.looseFiles),
                );
                if (
                    !admitted &&
                    needsRecursiveAdmissionProbe({
                        hasSubdirs: result.groups.some((group) => group.hasSubdirs),
                        recursive: opts.recursive,
                    })
                ) {
                    try {
                        const deep = await collect(detail.dirs, true);
                        admitted = hasImportableMedia(
                            buildFolderImportPlan(deep.groups, detail.looseFiles),
                        );
                    } catch {
                        // 递归扫描失败：保守当作"没有媒体"（与扫描失败同一种处理）。
                    }
                }
                // 无媒体可导：显式请求仍然弹窗（让用户看见"没有媒体文件"这句解释，
                // 比什么都不发生更好），隐式拖入直接返回 —— 否则会静默建出一条空轨道。
                if (!admitted && !detail.fromExplicitRequest) {
                    return;
                }
                // 新请求落地：作废上一个请求可能还在途的递归重扫。
                scanSeqRef.current += 1;
                setDraft(opts);
                setScan(result);
                setScanning(false);
                setRequest(detail);
            })();
        };
        window.addEventListener(FOLDER_IMPORT_REQUEST_EVENT, onRequest);
        return () => window.removeEventListener(FOLDER_IMPORT_REQUEST_EVENT, onRequest);
    }, [collect]);

    const plan = useMemo(
        () => buildFolderImportPlan(scan?.groups ?? [], request?.looseFiles ?? []),
        [request, scan],
    );

    const handleOptionsChange = useCallback(
        (patch: Partial<FolderImportOptions>) => {
            const next = normalizeFolderImportOptions({ ...draft, ...patch });
            setDraft(next);
            // 记住这次的选择：它是**对话框的预填**，用户多数时候只需按回车。
            dispatch(setFolderImportOptions(patch));
            void dispatch(persistUiSettings());
            if (patch.recursive === undefined || !request) return;
            // 递归与否决定"要导入的文件集合是什么"，只能重新枚举。
            // 序号作废在途的旧扫描（快速来回切两次时旧口径可能后到）。
            const seq = ++scanSeqRef.current;
            setScanning(true);
            void collect(request.dirs, next.recursive)
                .then((result) => {
                    if (seq !== scanSeqRef.current) return;
                    setScan(result);
                    setScanning(false);
                })
                .catch(() => {
                    // 只有最新扫描能收敛 loading，否则 spinner 可能被旧请求提前关掉 / 卡住。
                    if (seq !== scanSeqRef.current) return;
                    setScanning(false);
                });
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
            // 关闭即作废在途扫描：否则它可能在下次打开后落地，覆盖新请求的 scan。
            scanSeqRef.current += 1;
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
