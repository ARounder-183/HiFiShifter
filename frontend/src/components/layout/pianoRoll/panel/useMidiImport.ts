/**
 * 参数编辑器的 **MIDI 导入**：对话框状态、导入参数、拖放接入与"导入为 Clip"。
 *
 * 【为什么是一个独立的 hook】这块功能有一整套自己的状态（导入位置、填缝、BPM 来源、
 * 多轨合并、速度/拍号/调号映射、目标参数…共 15 个 useState）和一整条自己的通路
 * （工具栏按钮 → 文件对话框 → 拖放 → 预览选区 → 写入）。它与画布、滚动、绘制
 * 没有交集，却把这些状态和面板的其它 380 个绑定摊在同一层里。
 *
 * 【它自己拥有状态】与前面几刀（把纯函数搬走）不同，这里 `useState` 也一起搬进来了。
 * 这是可以的：hook 的调用顺序只要求**跨渲染稳定**，不要求与原文件里的位置一致；
 * 本 hook 在模型里只被调用一次，位置固定。
 *
 * 【依赖怎么拿】5 个输入由模型传入（`Pick<ModelDeps, …>`），其余全靠自己的状态与
 * import。函数体原样搬过来，开头解构回局部变量。
 *
 * 【边界】只负责"导入这件事"的状态与指令；真正落库的 `importMidiAsClip` 走后端。
 */
import { loadStandaloneWindowApi } from "../../../../services/hostWindow";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { importMidiAsClip } from "../../../../features/session/sessionSlice";
import { findFirstExternalPathAction } from "../../timeline/dnd";
import { selectionToFrameRanges, type ParamSelection } from "../paramSelection";
import { settingsApi } from "../../../../services/api/settings";

import type { ModelDeps } from "./modelDeps";

/** 导入对话框需要的模型输入（字段清单由 AST 扫描得出）。 */
type MidiImportDeps = Pick<
    ModelDeps,
    "dispatch" | "paramEditorRef" | "refreshNow" | "s" | "selectionRef"
>;

/** MIDI 导入的状态与指令。 */
export function useMidiImport(deps: MidiImportDeps) {
    const { dispatch, paramEditorRef, refreshNow, s, selectionRef } = deps;

    // MIDI 导入弹窗状态
    const [midiDialogOpen, setMidiDialogOpen] = useState(false);

    const [midiPath, setMidiPath] = useState<string | null>(null);

    const [clipboardGuid, setClipboardGuid] = useState<string | null>(null);

    // 导入位置选项（持久化到软件设置）
    const [importPosition, setImportPosition] = useState<string>("selection");

    // 填补空隙选项（持久化到软件设置）
    const [fillGaps, setFillGaps] = useState<boolean>(false);

    // BPM 选项（持久化到软件设置）
    const [importBpmAsProject, setImportBpmAsProject] = useState(false);

    const [noteBpmMode, setNoteBpmMode] = useState<string>("midi");

    const [specifiedBpm, setSpecifiedBpm] = useState<number>(120);

    const [multiTrackMerge, setMultiTrackMerge] = useState<boolean>(true);

    const [closeLeadingGap, setCloseLeadingGap] = useState<boolean>(true);

    const [importTempoMapEnabled, setImportTempoMapEnabled] = useState(false);

    const [importTempoMapTempo, setImportTempoMapTempo] = useState(true);

    const [importTempoMapTimeSignature, setImportTempoMapTimeSignature] = useState(true);

    const [importTempoMapKeySignature, setImportTempoMapKeySignature] = useState(false);

    const [importTargetReaperClipboard, setImportTargetReaperClipboard] =
        useState<string>("pitchParam");

    const [importTargetParamEditor, setImportTargetParamEditor] = useState<string>("pitchParam");

    const midiDialogSourceRef = useRef<"reaperClipboard" | "paramEditor">("paramEditor");

    // 启动时从设置加载
    useEffect(() => {
        settingsApi
            .getUiSettings()
            .then((s) => {
                if (s?.midiImportPosition) {
                    setImportPosition(s.midiImportPosition);
                }
                if (s?.midiFillGaps != null) {
                    setFillGaps(s.midiFillGaps);
                }
                if (s?.midiImportBpmAsProject != null) {
                    setImportBpmAsProject(s.midiImportBpmAsProject);
                }
                if (s?.midiNoteBpmMode != null) {
                    setNoteBpmMode(s.midiNoteBpmMode);
                }
                if (s?.midiSpecifiedBpm != null) {
                    setSpecifiedBpm(s.midiSpecifiedBpm);
                }
                if (s?.midiMultiTrackMerge != null) {
                    setMultiTrackMerge(s.midiMultiTrackMerge);
                }
                if (s?.midiCloseLeadingGap != null) {
                    setCloseLeadingGap(s.midiCloseLeadingGap);
                }
                if (s?.midiImportAsTempoMap != null) {
                    setImportTempoMapEnabled(Boolean(s.midiImportAsTempoMap));
                }
                if (s?.midiImportTempoMapTempo != null) {
                    setImportTempoMapTempo(Boolean(s.midiImportTempoMapTempo));
                }
                if (s?.midiImportTempoMapTimeSignature != null) {
                    setImportTempoMapTimeSignature(Boolean(s.midiImportTempoMapTimeSignature));
                }
                if (s?.midiImportTempoMapKeySignature != null) {
                    setImportTempoMapKeySignature(Boolean(s.midiImportTempoMapKeySignature));
                }
                if (s?.midiImportTargetReaperClipboard != null) {
                    setImportTargetReaperClipboard(s.midiImportTargetReaperClipboard);
                }
                if (s?.midiImportTargetParamEditor != null) {
                    setImportTargetParamEditor(s.midiImportTargetParamEditor);
                }
            })
            // 传输层失败（invoke 拒绝）按"保持出厂默认"降级：本 effect 是
            // fire-and-forget，不接住会把拒绝漏成 unhandledrejection。
            .catch(() => {});
    }, []);

    // 记录打开弹窗时的选区（拍数，多段），用于后续计算帧偏移。
    // 注意：MIDI 导入的「选区约束」只有单个时间窗接口，因此用包围区间
    // （首段起点 → 末段终点）—— 见 midiSelArgs。
    const [midiDialogSelection, setMidiDialogSelection] = useState<ParamSelection | null>(null);

    /** 打开“导入到参数编辑器”的 MIDI 导入对话框（编辑器按钮 / 拖放到编辑器内共用）。
     *  midiPath 为 null 时由用户在文件选择器中挑选文件；非 null 时直接导入该文件。 */
    const openParamEditorMidiImport = useCallback(
        (midiPath: string | null) => {
            midiDialogSourceRef.current = "paramEditor";
            // 快照当前选区（拍为单位）
            const sel = selectionRef.current;
            setMidiDialogSelection(sel ? sel.map((range) => ({ ...range })) : null);
            // 快照当前的 editParam 和 toolMode，保证异步加载轨道期间 selectionAvailable 不变
            midiDialogOpenParamsRef.current = {
                editParam: s.editParam,
                toolMode: s.toolMode,
            };
            setMidiPath(midiPath);
            setClipboardGuid(null);
            setMidiDialogOpen(true);
        },
        [s.editParam, s.toolMode],
    );

    const handleOpenMidiDialog = useCallback(() => {
        openParamEditorMidiImport(null);
    }, [openParamEditorMidiImport]);

    const [paramEditorMidiDragOver, setParamEditorMidiDragOver] = useState(false);

    const isPointOverParamEditor = useCallback((clientX: number, clientY: number) => {
        const el = paramEditorRef.current;
        if (!el) return false;
        const rect = el.getBoundingClientRect();
        return (
            clientX >= rect.left &&
            clientX <= rect.right &&
            clientY >= rect.top &&
            clientY <= rect.bottom
        );
    }, []);

    useEffect(() => {
        // 从拖拽载荷中取第一个 MIDI 文件路径（无则 null）。
        const firstMidiPath = (
            paths: string[] | null | undefined,
            primary: string | null | undefined,
        ): string | null => {
            const all = Array.isArray(paths) && paths.length > 0 ? paths : primary ? [primary] : [];
            const found = findFirstExternalPathAction(all);
            return found && found.kind === "importMidi" ? found.path : null;
        };

        // 文件浏览器面板的自定义拖拽事件（无 leave/end 事件，drop 即结束）。
        const onHifiFileDrag = (e: Event) => {
            const detail = (e as CustomEvent).detail as {
                type?: string;
                filePath?: string;
                filePaths?: string[];
                clientX?: number;
                clientY?: number;
            } | null;
            if (!detail) return;
            const clientX = Number(detail.clientX);
            const clientY = Number(detail.clientY);
            const x = Number.isFinite(clientX) ? clientX : undefined;
            const y = Number.isFinite(clientY) ? clientY : undefined;
            const over = x !== undefined && y !== undefined && isPointOverParamEditor(x, y);
            if (detail.type === "start" || detail.type === "move") {
                setParamEditorMidiDragOver(
                    Boolean(over && firstMidiPath(detail.filePaths, detail.filePath)),
                );
                return;
            }
            if (detail.type === "cancel") {
                // 拖拽被打断 / 取消：清掉落点高亮，不触发任何导入。
                setParamEditorMidiDragOver(false);
                return;
            }
            if (detail.type === "drop") {
                const midiPath = firstMidiPath(detail.filePaths, detail.filePath);
                setParamEditorMidiDragOver(false);
                if (over && midiPath) {
                    openParamEditorMidiImport(midiPath);
                }
            }
        };
        window.addEventListener("hifi-file-drag", onHifiFileDrag);

        // Tauri 系统文件拖放：与时间轴的 useTimelineDragDrop 各自独立监听，
        // 按坐标区域互斥处理（落在参数编辑器内的 MIDI 才在此导入）。
        // ⚠️ 必须先持有 Window 实例再调用方法：onDragDropEvent 依赖 `this`
        // （内部 this.listen），直接取方法引用会丢失绑定并静默失效。
        let disposed = false;
        let unlisten: null | (() => void) = null;
        void loadStandaloneWindowApi()
            .then((mod) => mod.getCurrentWindow())
            .then((win) =>
                win.onDragDropEvent((event: unknown) => {
                    if (disposed) return;
                    const payload = (
                        event && typeof event === "object" && "payload" in event
                            ? (event as { payload?: unknown }).payload
                            : event
                    ) as
                        | {
                              type?: string;
                              event?: string;
                              paths?: string[];
                              position?: { x?: number; y?: number };
                              pos?: { x?: number; y?: number };
                              cursorPosition?: { x?: number; y?: number };
                          }
                        | undefined;
                    if (!payload) return;
                    const type = String(payload.type ?? payload.event ?? "");
                    const paths: string[] = Array.isArray(payload.paths) ? payload.paths : [];
                    const pos = (payload.position ?? payload.pos ?? payload.cursorPosition) as
                        { x?: number; y?: number } | undefined;
                    const dpr = window.devicePixelRatio || 1;
                    const clientX = typeof pos?.x === "number" ? pos.x / dpr : undefined;
                    const clientY = typeof pos?.y === "number" ? pos.y / dpr : undefined;
                    const over =
                        clientX !== undefined &&
                        clientY !== undefined &&
                        isPointOverParamEditor(clientX, clientY);
                    if (type === "enter" || type === "over") {
                        setParamEditorMidiDragOver(Boolean(over && firstMidiPath(paths, null)));
                        return;
                    }
                    if (type === "leave") {
                        setParamEditorMidiDragOver(false);
                        return;
                    }
                    if (type === "drop") {
                        const midiPath = firstMidiPath(paths, null);
                        setParamEditorMidiDragOver(false);
                        if (over && midiPath) {
                            openParamEditorMidiImport(midiPath);
                        }
                    }
                }),
            )
            .then((fn) => {
                if (disposed) {
                    // 卸载竞态兜底：注册完成后组件已卸载，立即解绑。
                    fn();
                    return;
                }
                unlisten = fn;
            })
            .catch((err) => {
                console.warn(
                    "[param-editor-midi-drop] Failed to attach Tauri drag-drop listener",
                    err,
                );
            });

        return () => {
            disposed = true;
            window.removeEventListener("hifi-file-drag", onHifiFileDrag);
            if (unlisten) unlisten();
        };
    }, [isPointOverParamEditor, openParamEditorMidiImport]);

    // 记录打开 MIDI 弹窗时的 editParam / toolMode 快照，避免异步加载轨道期间 Redux 状态变化导致 selectionAvailable 跳变
    const midiDialogOpenParamsRef = useRef<{
        editParam: string;
        toolMode: string;
    }>({ editParam: "pitch", toolMode: "select" });

    const handleMidiImported = useCallback(() => {
        refreshNow();
    }, [refreshNow]);

    const handleImportAsClip = useCallback(
        (result: {
            trackIndices: number[];
            notesCount: number;
            midiPath: string;
            fillGaps: boolean;
            multiTrackMerge?: boolean;
            noteBpmMode?: string;
            specifiedBpm?: number;
            importBpmAsProject?: boolean;
            importAsTempoMap?: boolean;
            importTempo?: boolean;
            importTimeSignature?: boolean;
            importKeySignature?: boolean;
            clipboardGuid?: string;
            closeLeadingGap?: boolean;
        }) => {
            void dispatch(
                importMidiAsClip({
                    midiPath: result.midiPath,
                    trackIndices: result.trackIndices,
                    trackId: s.selectedTrackId,
                    startSec: s.playheadSec,
                    fillGaps: result.fillGaps || undefined,
                    multiTrackMerge: result.multiTrackMerge,
                    noteBpmMode: result.noteBpmMode,
                    specifiedBpm: result.specifiedBpm,
                    importBpmAsProject: result.importBpmAsProject,
                    clipboardGuid: result.clipboardGuid,
                    closeLeadingGap: result.closeLeadingGap,
                    importAsTempoMap: result.importAsTempoMap,
                    importTempo: result.importTempo,
                    importTimeSignature: result.importTimeSignature,
                    importKeySignature: result.importKeySignature,
                }),
            );
        },
        [dispatch, s.selectedTrackId, s.playheadSec],
    );

    // 导入位置变更时持久化保存
    const handleImportPositionChange = useCallback((position: string) => {
        setImportPosition(position);
        void settingsApi.saveUiSettings({ midiImportPosition: position });
    }, []);

    // 填补空隙选项变更时持久化保存
    const handleFillGapsChange = useCallback((value: boolean) => {
        setFillGaps(value);
        void settingsApi.saveUiSettings({ midiFillGaps: value });
    }, []);

    // BPM 选项变更时持久化保存
    const handleImportBpmAsProjectChange = useCallback((v: boolean) => {
        setImportBpmAsProject(v);
        void settingsApi.saveUiSettings({ midiImportBpmAsProject: v });
    }, []);

    const handleNoteBpmModeChange = useCallback((v: string) => {
        setNoteBpmMode(v);
        void settingsApi.saveUiSettings({ midiNoteBpmMode: v });
    }, []);

    const handleSpecifiedBpmChange = useCallback((v: number) => {
        setSpecifiedBpm(v);
        void settingsApi.saveUiSettings({ midiSpecifiedBpm: v });
    }, []);

    const handleMultiTrackMergeChange = useCallback((v: boolean) => {
        setMultiTrackMerge(v);
        void settingsApi.saveUiSettings({ midiMultiTrackMerge: v });
    }, []);

    const handleCloseLeadingGapChange = useCallback((v: boolean) => {
        setCloseLeadingGap(v);
        void settingsApi.saveUiSettings({ midiCloseLeadingGap: v });
    }, []);

    const handleImportTempoMapEnabledChange = useCallback((v: boolean) => {
        setImportTempoMapEnabled(v);
        void settingsApi.saveUiSettings({ midiImportAsTempoMap: v });
    }, []);

    const handleImportTempoMapTempoChange = useCallback((v: boolean) => {
        setImportTempoMapTempo(v);
        void settingsApi.saveUiSettings({ midiImportTempoMapTempo: v });
    }, []);

    const handleImportTempoMapTimeSignatureChange = useCallback((v: boolean) => {
        setImportTempoMapTimeSignature(v);
        void settingsApi.saveUiSettings({ midiImportTempoMapTimeSignature: v });
    }, []);

    const handleImportTempoMapKeySignatureChange = useCallback((v: boolean) => {
        setImportTempoMapKeySignature(v);
        void settingsApi.saveUiSettings({ midiImportTempoMapKeySignature: v });
    }, []);

    const handleImportTargetChange = useCallback((v: string) => {
        if (midiDialogSourceRef.current === "reaperClipboard") {
            setImportTargetReaperClipboard(v);
            void settingsApi.saveUiSettings({ midiImportTargetReaperClipboard: v });
        } else {
            setImportTargetParamEditor(v);
            void settingsApi.saveUiSettings({ midiImportTargetParamEditor: v });
        }
    }, []);

    // 计算 MIDI 导入的多选区帧约束：与参数编辑器选区逐段一致，导入只写入
    // 落在任一段内的帧（断层保持原值），对齐基准为首段起点。
    const midiSelRanges = useMemo(() => {
        if (!midiDialogSelection) return undefined;
        const ranges = selectionToFrameRanges(midiDialogSelection);
        return ranges.length > 0 ? ranges : undefined;
    }, [midiDialogSelection]);

    // selection 导入模式是否可用（基于弹窗打开时的快照，避免异步加载轨道时状态变化）
    const midiSelectionAvailable = useMemo(() => {
        if (!midiDialogSelection) return false;
        const p = midiDialogOpenParamsRef.current;
        return p.editParam === "pitch" && p.toolMode === "select";
    }, [midiDialogSelection]);

    return {
        clipboardGuid,
        closeLeadingGap,
        fillGaps,
        handleCloseLeadingGapChange,
        handleFillGapsChange,
        handleImportAsClip,
        handleImportBpmAsProjectChange,
        handleImportPositionChange,
        handleImportTargetChange,
        handleImportTempoMapEnabledChange,
        handleImportTempoMapKeySignatureChange,
        handleImportTempoMapTempoChange,
        handleImportTempoMapTimeSignatureChange,
        handleMidiImported,
        handleMultiTrackMergeChange,
        handleNoteBpmModeChange,
        handleOpenMidiDialog,
        handleSpecifiedBpmChange,
        importBpmAsProject,
        importPosition,
        importTargetParamEditor,
        importTargetReaperClipboard,
        importTempoMapEnabled,
        importTempoMapKeySignature,
        importTempoMapTempo,
        importTempoMapTimeSignature,
        midiDialogOpen,
        midiDialogOpenParamsRef,
        midiDialogSourceRef,
        midiPath,
        midiSelRanges,
        midiSelectionAvailable,
        multiTrackMerge,
        noteBpmMode,
        paramEditorMidiDragOver,
        setClipboardGuid,
        setMidiDialogOpen,
        setMidiDialogSelection,
        setMidiPath,
        specifiedBpm,
    };
}
