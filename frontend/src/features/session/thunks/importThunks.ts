import { createAsyncThunk } from "@reduxjs/toolkit";
import { webApi } from "../../../services/webviewApi";
import { fileBrowserApi } from "../../../services/api";
import { isVideoFilePath } from "../../../components/layout/timeline/dnd";
import { checkpointHistory, applyTimelinePayload, type SessionState } from "../sessionSlice";

import { addTrackRemote, setClipStateRemote } from "./timelineThunks";
import {
    currentImportGeneration,
    isImportCancelled,
    registerImportRun,
} from "./importCancellation";
import { computeAutoCrossfadeFromPayload } from "../../../components/layout/timeline/hooks/autoCrossfade";
import { computeClipNormalizationGain } from "../clipNormalization";
import { trackNameForMedia } from "../mediaTrackName";
import { waveformMipmapStore } from "../../../utils/waveformMipmapStore";
import { appStatusProgressBus } from "../../../utils/appStatusProgressBus";
import {isPluginMode} from "../../../services/hostCapabilities";
import {getPluginHost} from "../../../services/pluginHost";
import type {TimelineResult} from "../../../types/api";

/** 插件多文件依原GUI模式导入；每次确认宿主ARA返回的确切new clip，不按位置猜目标。 */
async function importHostBatch(args:{inputs:Array<string|File>;mode:"across-time"|"across-tracks";trackId?:string|null;startSec:number;
    getSession:()=>SessionState;publish:(timeline:TimelineResult)=>void;cancelled:()=>boolean}) {
    if(args.inputs.length>512) throw new Error("Host import batch exceeds 512 clips");
    const roots=args.getSession().tracks.filter(track=>!track.parentId);
    const first=args.trackId?Math.max(0,roots.findIndex(track=>track.id===args.trackId)):0;
    let track:string|null|undefined=args.trackId;let cursor=args.startSec;let last:TimelineResult|undefined;const newClipIds:string[]=[];
    await webApi.beginUndoGroup("import_media");
    try {for(let i=0;i<args.inputs.length;i++) {
        if(args.cancelled()) break;
        if(args.mode==="across-tracks") track=args.trackId===null?null:roots[first+i]?.id??null;
        if(i===0&&roots.length===0&&track===null) track=undefined;
        const input=args.inputs[i];const before=new Set(args.getSession().clips.map(clip=>clip.id));
        const timeline=typeof input==="string"?await webApi.importAudioItem(input,track,cursor):
            await getPluginHost()!.invoke<TimelineResult>("import_native_audio_file",{trackId:track,startSec:cursor},[input]);
        if(!timeline.ok) throw new Error("host audio import failed");
        const exact=(timeline as TimelineResult&{imported_clip_id?:string}).imported_clip_id;
        const clip=timeline.clips?.find(clip=>exact?clip.id===exact:!before.has(clip.id));
        if(!clip) throw new Error("host import returned no matching clip");
        newClipIds.push(clip.id);last=timeline;args.publish(timeline);
        if(args.mode==="across-time") {track=clip.track_id;cursor=clip.start_sec+clip.length_sec;}
    }} finally {await webApi.endUndoGroup();}
    return args.cancelled()?{ok:true,canceled:true,newClipIds}:{ok:true,imported:last,newClipIds,preservePlayhead:true};
}

type RawTimelineClip = {
    id?: string;
    track_id?: string;
    start_sec?: number;
    length_sec?: number;
    fade_in_sec?: number;
    fade_out_sec?: number;
};

/** `addTrackRemote` 的返回形状（只取"解析新轨道 id"需要的字段）。 */
interface AddedTrackResult {
    tracks: Array<{ id: string }>;
    selected_track_id?: string | null;
}

/** `createTrackForImport` 需要的派发能力（与既有 thunk 内部 dispatch 的用法一致）。 */
type TrackDispatch = (action: ReturnType<typeof addTrackRemote>) => {
    unwrap: () => Promise<AddedTrackResult>;
};

/**
 * 从"调用前后的 track id 差集"里解析出真正新建的那条轨道的 id。
 *
 * 三级回落与各处既有写法一致：差集 → 后端选中的新轨道 → 列表末位。
 */
function resolveCreatedTrackId(
    added: AddedTrackResult,
    beforeIds: ReadonlySet<string>,
): string | null {
    return (
        added.tracks.find((track) => !beforeIds.has(track.id))?.id ??
        added.selected_track_id ??
        added.tracks[added.tracks.length - 1]?.id ??
        null
    );
}

/**
 * 为一次**媒体文件导入**新建一条根轨道。
 *
 * 【为什么命名在这里】需求：导入媒体文件需要新建轨道时，轨道名 = 落到这条轨道上的
 * **第一个**媒体文件的主名（去扩展名）。把命名收进这个辅助函数，六处建轨点就
 * 不可能各写各的（此前一律 `name: undefined`，落成后端的 "Track"）。
 *
 * 【为什么只服务导入】工程里其它建轨行为（Ctrl+T、时间轴「新建轨道」、轨道复制、
 * clip 拖拽到空白处新建、粘贴、MIDI 导入）不经过这里，仍沿用后端的 "Track"。
 *
 * @param nameFromFile 落到这条轨道上的第一个媒体文件的路径或裸文件名；缺省时
 *   沿用后端默认名（目前只有不需要命名的路径会这样调）。
 * @returns 新轨道的 id；派发失败或拿不到 id 时为 `null`（由调用方决定拒绝还是跳过）。
 */
async function createTrackForImport(args: {
    dispatch: TrackDispatch;
    getState: () => unknown;
    nameFromFile?: string | null;
}): Promise<string | null> {
    const { dispatch, getState, nameFromFile } = args;
    const beforeIds = new Set(
        (getState() as { session: SessionState }).session.tracks.map((track) => track.id),
    );
    try {
        const added = await dispatch(
            addTrackRemote({
                name: nameFromFile ? trackNameForMedia(nameFromFile) : undefined,
                parentTrackId: null,
            }),
        ).unwrap();
        return resolveCreatedTrackId(added, beforeIds);
    } catch {
        return null;
    }
}

async function syncAutoCrossfadeFromLatestTimeline(args: {
    dispatch: (action: unknown) => Promise<unknown> & { unwrap: () => Promise<unknown> };
    getState: () => unknown;
    newClipIds: string[];
}) {
    const { dispatch, getState, newClipIds } = args;
    if (newClipIds.length === 0) {
        return null;
    }

    const session = (getState() as { session: SessionState }).session;
    if (!session.autoCrossfadeEnabled) {
        return null;
    }

    const latestTimeline = await webApi.getTimelineState();
    const allClips = (latestTimeline as { clips?: RawTimelineClip[] }).clips ?? [];
    const fadeUpdates = computeAutoCrossfadeFromPayload(allClips, newClipIds);
    if (fadeUpdates.length > 0) {
        const fadePromises = fadeUpdates.map((u) =>
            dispatch(
                setClipStateRemote({
                    clipId: u.clipId,
                    autoFadeInSec: u.autoFadeInSec,
                    autoFadeOutSec: u.autoFadeOutSec,
                    checkpoint: false,
                }),
            ).unwrap(),
        );
        await Promise.allSettled(fadePromises);
        // 淡化更新完成后再重新拉取：上面这份 latestTimeline 是淡化前的
        // 快照，若直接返回，fulfilled reducer 会把刚应用的自动淡化从
        // UI 上回滚掉（后端仍保留，前后端就此分叉）。
        return await webApi.getTimelineState();
    }
    return latestTimeline;
}

const setAudioPathAction = (path: string) => ({
    type: "session/setAudioPath" as const,
    payload: path,
});
export const importAudioFromDialog = createAsyncThunk(
    "session/importAudioFromDialog",
    async (_, { dispatch, rejectWithValue, getState }) => {
        const picked = await webApi.openAudioDialogMultiple();
        if (!picked.ok) {
            return rejectWithValue("open_audio_dialog_failed");
        }
        const pickedPaths = Array.isArray(picked.paths)
            ? picked.paths.filter((p): p is string => Boolean(p))
            : [];
        if (picked.canceled || pickedPaths.length === 0) {
            return { ok: true, canceled: true };
        }

        // Use current playhead position as import start and preserve playhead.
        const state = getState() as { session: SessionState };
        const startSec = state.session.playheadSec ?? 0;
        const trackId = state.session.selectedTrackId ?? null;
        const firstPath = pickedPaths[0];

        dispatch(setAudioPathAction(firstPath));

        if (pickedPaths.length > 1) {
            return {
                ok: true,
                canceled: false,
                path: firstPath,
                requiresModeChoice: true,
                audioPaths: pickedPaths,
                trackId,
                startSec,
            };
        }

        // 多音轨视频：让用户在“文件”菜单导入流程中选择要抽取的音轨。
        if (isVideoFilePath(firstPath)) {
            try {
                const streams = await fileBrowserApi.getMediaAudioStreams(firstPath);
                if (Array.isArray(streams) && streams.length > 1) {
                    return {
                        ok: true,
                        canceled: false,
                        path: firstPath,
                        requiresStreamChoice: true,
                        mediaAudioStreams: streams,
                        trackId,
                        startSec,
                    };
                }
            } catch {
                // 流枚举失败时退回默认音轨导入。
            }
        }

        // Delegate to importAudioAtPosition so imported clips start at playhead
        // and selection/undo handling is consistent with other import flows.
        try {
            const res = await dispatch(
                importAudioAtPosition({
                    audioPath: firstPath,
                    trackId,
                    startSec,
                }),
            ).unwrap();
            return {
                ok: true,
                canceled: false,
                path: firstPath,
                imported: res.imported ?? res,
                newClipIds: res.newClipIds,
            };
        } catch (err) {
            return rejectWithValue(err instanceof Error ? err.message : "import_audio_item_failed");
        }
    },
);

export const importAudioFromPath = createAsyncThunk(
    "session/importAudioFromPath",
    async (audioPath: string, { dispatch, rejectWithValue, getState }) => {
        dispatch(setAudioPathAction(audioPath));
        // 在发起导入前捕获现有 clip id 集合：await 期间其他 thunk 的
        // fulfilled 可能已把新 clip 写进 state，事后取差集会得到空集。
        const beforeClipIds = new Set(
            (getState() as { session: SessionState }).session.clips.map((c) => c.id),
        );
        const imported = await webApi.importAudioItem(audioPath);
        if (!(imported as { ok?: boolean }).ok) {
            const failure = imported as {
                error?: { message?: string };
                missing_files?: string[];
            };
            return rejectWithValue(
                failure.error?.message ?? failure.missing_files?.[0] ?? "import_audio_item_failed",
            );
        }
        const result = imported as { clips?: Array<{ id?: string }> };
        const newClipIds = (result.clips ?? [])
            .map((c) => c.id)
            .filter((id): id is string => !!id && !beforeClipIds.has(id));
        return {
            ok: true,
            path: audioPath,
            imported,
            newClipIds,
        };
    },
);

export const importAudioAtPosition = createAsyncThunk(
    "session/importAudioAtPosition",
    async (
        payload: {
            audioPath: string;
            trackId?: string | null;
            startSec?: number;
            normalizeAfterImport?: boolean;
            mediaAudioStreamIndex?: number;
        },
        { dispatch, rejectWithValue, getState },
    ) => {
        dispatch(setAudioPathAction(payload.audioPath));

        await webApi.beginUndoGroup("import_media");
        try {
            let targetTrackId: string | null | undefined;
            if (payload.trackId === null && !isPluginMode()) {
                // "插入到新轨道"：新轨道以这个文件命名。
                const createdId = await createTrackForImport({
                    dispatch: dispatch as unknown as TrackDispatch,
                    getState,
                    nameFromFile: payload.audioPath,
                });
                if (!createdId) {
                    return rejectWithValue("add_track_failed");
                }
                targetTrackId = createdId;
            } else {
                targetTrackId = isPluginMode()&&payload.trackId===null&&(getState() as {session:SessionState}).session.tracks.length>0?null:payload.trackId??undefined;
            }

            const beforeClipIds = new Set(
                (getState() as { session: SessionState }).session.clips.map((c) => c.id),
            );

            const imported = await webApi.importAudioItem(
                payload.audioPath,
                targetTrackId,
                payload.startSec,
                payload.mediaAudioStreamIndex,
            );
            if (!(imported as { ok?: boolean }).ok) {
                const failure = imported as {
                    error?: { message?: string };
                    missing_files?: string[];
                };
                return rejectWithValue(
                    failure.error?.message ??
                        failure.missing_files?.[0] ??
                        "import_audio_item_failed",
                );
            }

            const result = imported as { clips?: Array<{ id?: string }> };
            const newClipIds = (result.clips ?? [])
                .map((c) => c.id)
                .filter((id): id is string => !!id && !beforeClipIds.has(id));

            let latestTimeline = await syncAutoCrossfadeFromLatestTimeline({
                dispatch: dispatch as unknown as (
                    action: unknown,
                ) => Promise<unknown> & { unwrap: () => Promise<unknown> },
                getState,
                newClipIds,
            });

            if (payload.normalizeAfterImport && newClipIds.length > 0) {
                const timelineForNormalization = (latestTimeline ?? imported) as {
                    clips?: Array<{
                        id?: string;
                        source_path?: string;
                        duration_sec?: number;
                        length_sec?: number;
                        source_start_sec?: number;
                        source_end_sec?: number;
                        playback_rate?: number;
                    }>;
                };
                for (const clipId of newClipIds) {
                    const clip = timelineForNormalization.clips?.find(
                        (entry) => entry.id === clipId,
                    );
                    if (!clip) continue;
                    const gain = computeClipNormalizationGain(
                        {
                            sourcePath: clip.source_path,
                            durationSec: Number(clip.duration_sec ?? 0) || undefined,
                            lengthSec: Math.max(0, Number(clip.length_sec ?? 0) || 0),
                            sourceStartSec: Number(clip.source_start_sec ?? 0) || 0,
                            sourceEndSec: Number(clip.source_end_sec ?? 0) || 0,
                            playbackRate: Number(clip.playback_rate ?? 1) || 1,
                        },
                        {
                            getInterleavedSlice: (
                                sourcePath,
                                _channel,
                                sourceStartSec,
                                sourceSpanSec,
                            ) =>
                                waveformMipmapStore.getInterleavedSlice(
                                    sourcePath,
                                    0,
                                    sourceStartSec,
                                    sourceSpanSec,
                                ),
                            releaseInterleaved: (data) =>
                                waveformMipmapStore.releaseInterleaved(data as Float32Array),
                        },
                    );
                    if (gain == null) continue;
                    latestTimeline = (await dispatch(
                        setClipStateRemote({
                            clipId,
                            gain,
                            checkpoint: false,
                        }),
                    ).unwrap()) as typeof latestTimeline;
                }
            }

            // 导入后将光标定位到第一个音频块的起始位置。必须以独立字段交给
            // reducer 显式采纳：playhead_sec 是传输层字段，全量快照应用默认
            // 不采纳（防播放中编辑的光标跳变），塞进 imported 里会被静默丢弃。
            const importedResult = latestTimeline ?? imported;

            return {
                ok: true,
                imported: importedResult,
                newClipIds,
                playheadSec: typeof payload.startSec === "number" ? payload.startSec : undefined,
            };
        } finally {
            void webApi.endUndoGroup();
        }
    },
);

export const importAudioFileAtPosition = createAsyncThunk(
    "session/importAudioFileAtPosition",
    async (
        payload: { file: File; trackId?: string | null; startSec?: number },
        { dispatch, rejectWithValue, getState },
    ) => {
        await webApi.beginUndoGroup("import_media");
        try {
            let targetTrackId: string | null | undefined;
            if (payload.trackId === null&&!isPluginMode()) {
                // "插入到新轨道"：新轨道以这个文件命名。
                const createdId = await createTrackForImport({
                    dispatch: dispatch as unknown as TrackDispatch,
                    getState,
                    nameFromFile: payload.file.name,
                });
                if (!createdId) {
                    return rejectWithValue("add_track_failed");
                }
                targetTrackId = createdId;
            } else {
                targetTrackId = isPluginMode()&&payload.trackId===null&&(getState() as {session:SessionState}).session.tracks.length>0?null:payload.trackId??undefined;
            }

            const beforeClipIds = new Set(
                (getState() as { session: SessionState }).session.clips.map((c) => c.id),
            );

            const fileName = String(payload.file.name ?? "dropped-audio");
            const host=getPluginHost();
            const dataUrl = host?null:await new Promise<string>((resolve, reject) => {
                const reader = new FileReader();
                reader.onerror = () => reject(new Error("read_failed"));
                reader.onload = () => resolve(String(reader.result ?? ""));
                reader.readAsDataURL(payload.file);
            });

            const commaIdx = dataUrl?.indexOf(",")??-1;
            const base64 = dataUrl?(commaIdx !== -1 ? dataUrl.substring(commaIdx + 1) : dataUrl):"";

            const imported = host?await host.invoke<TimelineResult>("import_native_audio_file",{trackId:targetTrackId,startSec:payload.startSec},[payload.file]):await webApi.importAudioBytes(
                fileName,
                base64,
                targetTrackId,
                payload.startSec,
            );
            if (!(imported as { ok?: boolean }).ok) {
                return rejectWithValue(
                    (imported as { error?: { message?: string } }).error?.message ??
                        "import_audio_bytes_failed",
                );
            }

            const result = imported as { clips?: Array<{ id?: string }> };
            const newClipIds = (result.clips ?? [])
                .map((c) => c.id)
                .filter((id): id is string => !!id && !beforeClipIds.has(id));

            const latestTimeline = await syncAutoCrossfadeFromLatestTimeline({
                dispatch: dispatch as unknown as (
                    action: unknown,
                ) => Promise<unknown> & { unwrap: () => Promise<unknown> },
                getState,
                newClipIds,
            });

            // 导入后将光标定位到第一个音频块的起始位置（经独立字段交 reducer 采纳）。
            const importedResult = latestTimeline ?? imported;

            return {
                ok: true,
                imported: importedResult,
                newClipIds,
                playheadSec: typeof payload.startSec === "number" ? payload.startSec : undefined,
            };
        } catch (err) {
            return rejectWithValue(
                err instanceof Error ? err.message : "import_audio_bytes_failed",
            );
        } finally {
            void webApi.endUndoGroup();
        }
    },
);

/**
 * 多文件导入，支持三种模式:
 * - "across-time": 在同一轨道依次排列（按顺序首尾相连）
 * - "across-tracks": 每个文件分配到不同的新轨道，起始位置相同
 * - "as-takes": 所有文件合并为一个 Clip 的多个 Take，长度取最长媒体
 */
export const importMultipleAudioAtPosition = createAsyncThunk(
    "session/importMultipleAudioAtPosition",
    async (
        payload: {
            audioPaths: string[];
            mode: "across-time" | "across-tracks" | "as-takes";
            trackId?: string | null;
            startSec?: number;
        },
        { dispatch, rejectWithValue, getState },
    ) => {
        const { audioPaths, mode, startSec = 0 } = payload;
        if (audioPaths.length === 0) return { ok: true };
        if(isPluginMode()) {
            if(mode==="as-takes") return rejectWithValue("Alternative take batch import is not implemented in plugin mode");
            const generation=currentImportGeneration();const run=registerImportRun();
            try {return await importHostBatch({inputs:audioPaths,mode,trackId:payload.trackId,startSec,
                getSession:()=> (getState() as {session:SessionState}).session,
                publish:timeline=>{dispatch(applyTimelinePayload(timeline));},cancelled:()=>isImportCancelled(generation)});
            } catch(error) {return rejectWithValue(error instanceof Error?error.message:"host batch import failed");} finally {run.finish();}
        }

        // Single file → delegate to importAudioAtPosition
        if (audioPaths.length === 1) {
            return dispatch(
                importAudioAtPosition({
                    audioPath: audioPaths[0],
                    trackId: payload.trackId,
                    startSec,
                }),
            ).unwrap();
        }

        // Create a single undo checkpoint for the entire batch
        dispatch(checkpointHistory());

        await webApi.beginUndoGroup("import_media");
        // 取消闸门：本批可能是目录导入委托过来的长循环（见 importCancellation）。
        const importGen = currentImportGeneration();
        // 登记这次导入：历史跳转前会等它收尾（见 registerImportRun）。
        const importRun = registerImportRun();
        try {
            const beforeClipIds = new Set(
                (getState() as { session: SessionState }).session.clips.map((c) => c.id),
            );

            let lastImported: unknown = null;
            const accumulatedNewClipIds: string[] = [];

            if (mode === "as-takes") {
                const imported = await webApi.importMediaFilesAsTakes({
                    paths: audioPaths,
                    trackId: payload.trackId,
                    startSec,
                });
                if ((imported as { ok?: boolean }).ok) {
                    lastImported = imported;
                    const result = imported as { clips?: Array<{ id?: string }> };
                    for (const c of result.clips ?? []) {
                        if (c.id) accumulatedNewClipIds.push(c.id);
                    }
                } else {
                    // 单次全有全无调用：失败必须显式拒绝而不是返回 ok:true
                    // 的空结果（missing_files 携带后端给出的具体原因）。
                    const missing = (imported as { missing_files?: string[] }).missing_files;
                    return rejectWithValue(
                        missing?.join("; ") || "import_media_files_as_takes_failed",
                    );
                }
            } else if (mode === "across-time") {
                // Import files sequentially on the same track
                let cursor = startSec;
                let targetTrackId: string | undefined;

                if (payload.trackId === null) {
                    // 整批落在同一条新轨道上 → 以**这批的第一个**文件命名。
                    const createdId = await createTrackForImport({
                        dispatch: dispatch as unknown as TrackDispatch,
                        getState,
                        nameFromFile: audioPaths[0],
                    });
                    if (!createdId) {
                        return rejectWithValue("add_track_failed");
                    }
                    targetTrackId = createdId;
                } else {
                    targetTrackId = payload.trackId ?? undefined;
                }

                for (const audioPath of audioPaths) {
                    if (isImportCancelled(importGen)) break;
                    const imported = await webApi.importAudioItem(audioPath, targetTrackId, cursor);
                    if (!(imported as { ok?: boolean }).ok) continue;
                    lastImported = imported;
                    const result = imported as {
                        clips?: Array<{ id?: string; start_sec?: number; length_sec?: number }>;
                    };
                    const allClips = result.clips ?? [];
                    for (const c of allClips) {
                        if (c.id) accumulatedNewClipIds.push(c.id);
                    }
                    const newClip = allClips.find(
                        (c) => Math.abs((c.start_sec ?? 0) - cursor) < 0.01,
                    );
                    cursor += newClip?.length_sec ?? 0;
                }
            } else {
                // "across-tracks" — start from current track, then use subsequent existing tracks,
                // only creating new tracks when we run out of existing ones.
                const state = getState() as { session: SessionState };
                // Get root-level tracks sorted by order/index
                const rootTracks = state.session.tracks
                    .filter((t) => !t.parentId)
                    .sort((a, b) => {
                        // Use the index in the tracks array as order proxy
                        const ai = state.session.tracks.indexOf(a);
                        const bi = state.session.tracks.indexOf(b);
                        return ai - bi;
                    });

                // Find the starting index: the track the user dropped onto
                let startIdx = 0;
                if (payload.trackId) {
                    const idx = rootTracks.findIndex((t) => t.id === payload.trackId);
                    if (idx >= 0) startIdx = idx;
                }

                for (let i = 0; i < audioPaths.length; i++) {
                    if (isImportCancelled(importGen)) break;
                    const audioPath = audioPaths[i];
                    const trackIdx = startIdx + i;
                    let targetTrackId: string | undefined;

                    if (trackIdx < rootTracks.length) {
                        // Use existing track
                        targetTrackId = rootTracks[trackIdx].id;
                    } else {
                        // 需要新建轨道：across-tracks 下每条新轨道各自以**它承载的
                        // 那个文件**命名（不是这批的第一个）—— 循环变量 `audioPath` 就是它。
                        const createdId = await createTrackForImport({
                            dispatch: dispatch as unknown as TrackDispatch,
                            getState,
                            nameFromFile: audioPath,
                        });
                        if (!createdId) continue;
                        targetTrackId = createdId;
                    }

                    try {
                        const imported = await webApi.importAudioItem(
                            audioPath,
                            targetTrackId,
                            startSec,
                        );
                        if ((imported as { ok?: boolean }).ok) {
                            lastImported = imported;
                            const result = imported as {
                                clips?: Array<{
                                    id?: string;
                                    start_sec?: number;
                                    length_sec?: number;
                                }>;
                            };
                            for (const c of result.clips ?? []) {
                                if (c.id) accumulatedNewClipIds.push(c.id);
                            }
                        }
                    } catch {
                        // Continue with remaining files
                    }
                }
            }

            // 用户撤销 / 跳转历史 → 循环提前退出。**不带**时间线快照：带回去会把
            // 用户刚撤销掉的东西又画回来（`imported: null` 让 reducer 跳过套用）。
            if (isImportCancelled(importGen)) {
                return { ok: true, canceled: true, imported: null, newClipIds: [] as string[] };
            }

            // Detect new clips from all import responses
            const newClipIds = accumulatedNewClipIds.filter((id) => !!id && !beforeClipIds.has(id));

            const latestTimeline = await syncAutoCrossfadeFromLatestTimeline({
                dispatch: dispatch as unknown as (
                    action: unknown,
                ) => Promise<unknown> & { unwrap: () => Promise<unknown> },
                getState,
                newClipIds,
            });

            // 导入后将光标定位到起始位置（经独立字段交 reducer 采纳，
            // 塞进 imported.playhead_sec 会被快照应用静默丢弃）。
            const importedResult = latestTimeline ?? lastImported;

            return { ok: true, imported: importedResult, newClipIds, playheadSec: startSec };
        } finally {
            importRun.finish();
            void webApi.endUndoGroup();
        }
    },
);

/** 目录导入计划里的一棵子树（`FolderImportPlanNode` 去掉 `dir`：导入不需要它）。 */
export interface FolderImportTreeNode {
    /** 轨道名（目录名的最后一段）。 */
    name: string;
    /** 该目录**直属**的媒体文件（已排序）。 */
    files: string[];
    children: FolderImportTreeNode[];
}

export interface ImportFolderAtPositionPayload {
    /** 目录树（仅"创建轨道组"模式需要）。 */
    roots: FolderImportTreeNode[];
    /** 直接拖入的散文件（不属于任何目录）。 */
    looseFiles: string[];
    /** 扁平文件顺序（其余三种落位方式用）。 */
    orderedFiles: string[];
    mode: "across-time" | "across-tracks" | "as-takes";
    /** 为每个文件夹创建轨道组（仅 `across-tracks` 有效）。 */
    createFolderTracks: boolean;
    trackId?: string | null;
    startSec?: number;
    /** 新建根轨道的落点（根级下标）。 */
    insertIndex?: number | null;
}

/**
 * 目录导入进度条的出现阈值（文件数）。
 *
 * 【为什么需要阈值】单个文件导入几十毫秒，为一个 3 个文件的目录闪一下状态栏只是噪音。
 * 超过这个量级才值得让用户看到"还要多久"。
 */
const FOLDER_IMPORT_PROGRESS_THRESHOLD = 12;

/**
 * 目录导入。
 *
 * 【与 `importMultipleAudioAtPosition` 的关系】不创建轨道组时**直接委托**给它 ——
 * 目录的贡献就是"它里面的媒体文件"，三种排布方式的语义、撤销分组、自动交叉淡化
 * 全部照旧，一行都不用重写。只有"创建轨道组"是一条新路径。
 *
 * 【为什么"创建轨道组"必须自己走一遍】它要求：每个文件夹一条**空白**的根轨道、
 * 每个媒体文件一条挂在它下面的子轨道。这与 `across-tracks` 的"复用已有轨道、
 * 不够才新建"是相反的语义（用户明确要求"必定新建，不要利用旧轨道"），没法用
 * 参数表达，只能另写。
 */
export const importFolderAtPosition = createAsyncThunk(
    "session/importFolderAtPosition",
    async (payload: ImportFolderAtPositionPayload, { dispatch, rejectWithValue, getState }) => {
        const { roots, looseFiles, orderedFiles, mode, createFolderTracks, startSec = 0 } = payload;
        const useFolderTracks = mode === "across-tracks" && createFolderTracks && roots.length > 0;

        if (!useFolderTracks) {
            return dispatch(
                importMultipleAudioAtPosition({
                    audioPaths: orderedFiles,
                    mode,
                    trackId: payload.trackId,
                    startSec,
                }),
            ).unwrap();
        }

        // 展平成"父在子前"的下标序列（`add_track_tree` 的要求）。
        interface TreeSpec {
            name: string;
            parentIndex: number | null;
            files: string[];
        }
        const specs: TreeSpec[] = [];
        const walk = (node: FolderImportTreeNode, parentIndex: number | null) => {
            const index = specs.length;
            // 文件夹轨道自身**空白** —— 它代表这个文件夹，媒体文件挂在它的子轨道上。
            specs.push({ name: node.name, parentIndex, files: [] });
            // 子目录整棵在前，然后才是本目录的媒体文件：与文件浏览器的
            // `foldersFirst` 默认一致（列表里看到的顺序就是导入后的顺序）。
            for (const child of node.children) walk(child, index);
            for (const file of node.files) {
                specs.push({ name: trackNameForMedia(file), parentIndex: index, files: [file] });
            }
        };
        for (const root of roots) walk(root, null);
        // 散文件各建一条根轨道：与"每个目录一条根轨道"同构，不引入第三种规则。
        for (const file of looseFiles) {
            specs.push({ name: trackNameForMedia(file), parentIndex: null, files: [file] });
        }

        // 没有任何可建的轨道（既无目录也无散文件）：没什么可做。
        if (specs.length === 0) {
            return { ok: true, imported: null, newClipIds: [] as string[], playheadSec: startSec };
        }

        dispatch(checkpointHistory());
        // 进度：逐个文件导入，含上千个文件的目录会有肉眼可见的等待。低于阈值时不显示，
        // 免得为一次 200ms 的操作闪一下状态栏。走总线而不是 Redux —— 见其文件头。
        const totalFiles = specs.reduce((count, spec) => count + spec.files.length, 0);
        let processed = 0;
        const reportProgress = (active: boolean) =>
            appStatusProgressBus.setFolderImport({
                active: active && totalFiles >= FOLDER_IMPORT_PROGRESS_THRESHOLD,
                done: processed,
                total: totalFiles,
            });
        reportProgress(true);
        // 取消闸门：用户在本循环进行中撤销 / 跳转历史 / 打开别的工程时，下一轮就退出。
        const importGen = currentImportGeneration();
        // 登记这次导入：历史跳转前会等它收尾（见 registerImportRun）。
        const importRun = registerImportRun();
        try {
            const beforeClipIds = new Set(
                (getState() as { session: SessionState }).session.clips.map((c) => c.id),
            );

            /*
             * 打点在**建树之前**：整个导入（轨道组 + 全部 clip）是一个不可分割的
             * 用户操作，一次撤销就该把它整体撤掉。
             *
             * 【为什么不放在建树之后】放在建树之后时，撤销的落点是"有轨道树、无
             * clip"——用户按一次 Ctrl+Z 只撤掉文件，因导入新建的轨道还留在时间线上，
             * 与"我撤销了这次导入"的直觉不符（用户明确要求：撤销导入必须连轨道一起撤）。
             * 放在建树之前，落点就是导入前的那个状态；配合下面的取消闸门，循环不会
             * 再往已消失的 trackId 上灌 clip（那正是撤销栈被写坏的根因）。
             */
            await webApi.beginUndoGroup("import_folder");
            if (isImportCancelled(importGen)) {
                return { ok: true, canceled: true, imported: null, newClipIds: [] as string[] };
            }

            const created = await webApi.addTrackTree({
                nodes: specs.map((spec) => ({
                    name: spec.name,
                    parentIndex: spec.parentIndex,
                })),
                insertIndex: payload.insertIndex ?? null,
            });
            const trackIds = created.createdTrackIds ?? [];
            if (trackIds.length !== specs.length) {
                // 轨道树没建全就继续导入，会把 clip 落到错误的轨道上 —— 宁可不导。
                return rejectWithValue("add_track_tree_failed");
            }
            if (isImportCancelled(importGen)) {
                // 撤销恰好落在"建树在途"的那几毫秒里：`notifyHistoryJump` 会等这一步
                // 收尾之后才真正跳转历史，因此这棵树随后会被那次撤销一并还原 ——
                // 这里只需停止继续导入。
                return { ok: true, canceled: true, imported: null, newClipIds: [] as string[] };
            }
            // 轨道树先落地：即使随后一个文件都没导成（全部不可解码），用户也该看到
            // 自己刚建出来的轨道组，而不是一片空白。
            if (created.timeline) {
                dispatch(applyTimelinePayload(created.timeline));
            }

            const accumulatedNewClipIds: string[] = [];
            const failedFiles: string[] = [];
            let attempted = 0;
            let lastImported: unknown = null;
            for (let index = 0; index < specs.length; index += 1) {
                const trackId = trackIds[index];
                for (const file of specs[index].files) {
                    // 每一轮先问一句"我还在被期待吗"：用户撤销后时间线已回到导入前
                    // （轨道组也没了），继续灌只会把撤销栈写坏。
                    if (isImportCancelled(importGen)) break;
                    attempted += 1;
                    try {
                        const imported = await webApi.importAudioItem(file, trackId, startSec);
                        if (!(imported as { ok?: boolean }).ok) {
                            // 不可解码 / 文件已消失：记下来汇总报告，不中断整批。
                            failedFiles.push(file);
                            continue;
                        }
                        lastImported = imported;
                        const result = imported as { clips?: Array<{ id?: string }> };
                        for (const clip of result.clips ?? []) {
                            if (clip.id) accumulatedNewClipIds.push(clip.id);
                        }
                    } catch {
                        failedFiles.push(file);
                    } finally {
                        processed += 1;
                        reportProgress(true);
                    }
                }
                if (isImportCancelled(importGen)) break;
            }

            if (isImportCancelled(importGen)) {
                // 取消：**不带**时间线快照 —— 用户的撤销已经让后端回到导入前，
                // 带回去只会把刚撤销掉的东西又画回来。复用既有取消语义
                // （`sessionSlice` 翻成"已取消导入"），不走 rejectWithValue
                // （那会把状态栏写成"导入失败"，而用户做的是撤销）。
                return { ok: true, canceled: true, imported: null, newClipIds: [] as string[] };
            }

            const newClipIds = accumulatedNewClipIds.filter((id) => !!id && !beforeClipIds.has(id));
            const latestTimeline = await syncAutoCrossfadeFromLatestTimeline({
                dispatch: dispatch as unknown as (
                    action: unknown,
                ) => Promise<unknown> & { unwrap: () => Promise<unknown> },
                getState,
                newClipIds,
            });

            return {
                ok: true,
                imported: latestTimeline ?? lastImported,
                newClipIds,
                playheadSec: startSec,
                /** 未成功导入的文件（汇总报告用；空数组表示全部成功）。 */
                failedFiles,
                /** 尝试导入的媒体文件总数（成功数 = 它减去失败数）。 */
                attempted,
            };
        } finally {
            reportProgress(false);
            importRun.finish();
            void webApi.endUndoGroup();
        }
    },
);

export const importMultipleAudioFilesAtPosition = createAsyncThunk(
    "session/importMultipleAudioFilesAtPosition",
    async (
        payload: {
            files: File[];
            mode: "across-time" | "across-tracks";
            trackId?: string | null;
            startSec?: number;
        },
        { dispatch, rejectWithValue, getState },
    ) => {
        const { files, mode, startSec = 0 } = payload;
        if (!files || files.length === 0) return { ok: true };
        if(isPluginMode()) {
            const generation=currentImportGeneration();const run=registerImportRun();
            try {return await importHostBatch({inputs:files,mode,trackId:payload.trackId,startSec,
                getSession:()=> (getState() as {session:SessionState}).session,
                publish:timeline=>{dispatch(applyTimelinePayload(timeline));},cancelled:()=>isImportCancelled(generation)});
            } catch(error) {return rejectWithValue(error instanceof Error?error.message:"host File batch import failed");} finally {run.finish();}
        }

        // Single file → delegate to importAudioFileAtPosition
        if (files.length === 1) {
            return dispatch(
                importAudioFileAtPosition({ file: files[0], trackId: payload.trackId, startSec }),
            ).unwrap();
        }

        dispatch(checkpointHistory());

        await webApi.beginUndoGroup();
        // 取消闸门：本批可能是目录导入委托过来的长循环（见 importCancellation）。
        const importGen = currentImportGeneration();
        // 登记这次导入：历史跳转前会等它收尾（见 registerImportRun）。
        const importRun = registerImportRun();
        try {
            const beforeClipIds = new Set(
                (getState() as { session: SessionState }).session.clips.map((c) => c.id),
            );

            const accumulatedNewClipIds: string[] = [];

            let lastImported: unknown = null;

            if (mode === "across-time") {
                let cursor = startSec;
                let targetTrackId: string | undefined;

                if (payload.trackId === null) {
                    // 整批落在同一条新轨道上 → 以**这批的第一个**文件命名。
                    const createdId = await createTrackForImport({
                        dispatch: dispatch as unknown as TrackDispatch,
                        getState,
                        nameFromFile: files[0]?.name,
                    });
                    if (!createdId) {
                        return rejectWithValue("add_track_failed");
                    }
                    targetTrackId = createdId;
                } else {
                    targetTrackId = payload.trackId ?? undefined;
                }

                for (const file of files) {
                    if (isImportCancelled(importGen)) break;
                    const fileName = String(file.name ?? "dropped-audio");
                    const dataUrl = await new Promise<string>((resolve, reject) => {
                        const reader = new FileReader();
                        reader.onerror = () => reject(new Error("read_failed"));
                        reader.onload = () => resolve(String(reader.result ?? ""));
                        reader.readAsDataURL(file);
                    });
                    const base64 = dataUrl.includes(",")
                        ? dataUrl.split(",").slice(1).join(",")
                        : dataUrl;

                    const imported = await webApi.importAudioBytes(
                        fileName,
                        base64,
                        targetTrackId,
                        cursor,
                    );
                    if (!(imported as { ok?: boolean }).ok) continue;
                    lastImported = imported;
                    const result = imported as {
                        clips?: Array<{ id?: string; start_sec?: number; length_sec?: number }>;
                    };
                    for (const c of result.clips ?? []) {
                        if (c.id) accumulatedNewClipIds.push(c.id);
                    }
                    const newClip = result.clips?.find(
                        (c) => Math.abs((c.start_sec ?? 0) - cursor) < 0.01,
                    );
                    cursor += newClip?.length_sec ?? 0;
                }
            } else {
                // across-tracks: similar to importMultipleAudioAtPosition
                const state = getState() as { session: SessionState };
                const rootTracks = state.session.tracks
                    .filter((t) => !t.parentId)
                    .sort(
                        (a, b) => state.session.tracks.indexOf(a) - state.session.tracks.indexOf(b),
                    );
                let startIdx = 0;
                if (payload.trackId) {
                    const idx = rootTracks.findIndex((t) => t.id === payload.trackId);
                    if (idx >= 0) startIdx = idx;
                }

                for (let i = 0; i < files.length; i++) {
                    if (isImportCancelled(importGen)) break;
                    const file = files[i];
                    const trackIdx = startIdx + i;
                    let targetTrackId: string | undefined;
                    if (trackIdx < rootTracks.length) {
                        targetTrackId = rootTracks[trackIdx].id;
                    } else {
                        // 每条新轨道各自以**它承载的那个文件**命名（循环变量 `file`）。
                        const createdId = await createTrackForImport({
                            dispatch: dispatch as unknown as TrackDispatch,
                            getState,
                            nameFromFile: file.name,
                        });
                        if (!createdId) continue;
                        targetTrackId = createdId;
                    }

                    const fileName = String(file.name ?? "dropped-audio");
                    const dataUrl = await new Promise<string>((resolve, reject) => {
                        const reader = new FileReader();
                        reader.onerror = () => reject(new Error("read_failed"));
                        reader.onload = () => resolve(String(reader.result ?? ""));
                        reader.readAsDataURL(file);
                    });
                    const base64 = dataUrl.includes(",")
                        ? dataUrl.split(",").slice(1).join(",")
                        : dataUrl;

                    try {
                        const imported = await webApi.importAudioBytes(
                            fileName,
                            base64,
                            targetTrackId,
                            startSec,
                        );
                        if ((imported as { ok?: boolean }).ok) {
                            lastImported = imported;
                            const result = imported as {
                                clips?: Array<{
                                    id?: string;
                                    start_sec?: number;
                                    length_sec?: number;
                                }>;
                            };
                            for (const c of result.clips ?? []) {
                                if (c.id) accumulatedNewClipIds.push(c.id);
                            }
                        }
                    } catch {
                        // continue
                    }
                }
            }

            // 用户撤销 / 跳转历史 → 循环提前退出，且**不带**时间线快照。
            if (isImportCancelled(importGen)) {
                return { ok: true, canceled: true, imported: null, newClipIds: [] as string[] };
            }

            const newClipIds = accumulatedNewClipIds.filter((id) => !!id && !beforeClipIds.has(id));

            const latestTimeline = await syncAutoCrossfadeFromLatestTimeline({
                dispatch: dispatch as unknown as (
                    action: unknown,
                ) => Promise<unknown> & { unwrap: () => Promise<unknown> },
                getState,
                newClipIds,
            });

            // 导入后将光标定位到起始位置（经独立字段交 reducer 采纳，
            // 塞进 imported.playhead_sec 会被快照应用静默丢弃）。
            const importedResult = latestTimeline ?? lastImported;

            return { ok: true, imported: importedResult, newClipIds, playheadSec: startSec };
        } finally {
            importRun.finish();
            void webApi.endUndoGroup();
        }
    },
);

export const importMidiAsClip = createAsyncThunk(
    "session/importMidiAsClip",
    async (
        payload: {
            midiPath: string;
            trackIndices: number[];
            trackId?: string | null;
            startSec?: number;
            fillGaps?: boolean;
            multiTrackMerge?: boolean;
            noteBpmMode?: string;
            specifiedBpm?: number;
            importBpmAsProject?: boolean;
            clipboardGuid?: string;
            closeLeadingGap?: boolean;
            importAsTempoMap?: boolean;
            importTempo?: boolean;
            importTimeSignature?: boolean;
            importKeySignature?: boolean;
        },
        { dispatch, rejectWithValue, getState },
    ) => {
        await webApi.beginUndoGroup("import_vocalshifter");
        try {
            // 必须在发起导入前捕获现有 clip id 集合（契约见 openVocalShifterFromDialog）：
            // await 期间其他 thunk 的 fulfilled 可能已把新 clip 写进 state，
            // 事后取差集会漏掉真正新增的 clip。
            const beforeClipIds = new Set(
                (getState() as { session: SessionState }).session.clips.map((c) => c.id),
            );
            let targetTrackId: string | undefined;
            if (payload.trackId === null || payload.trackId === undefined) {
                const state = getState() as { session: SessionState };
                const beforeIds = new Set(state.session.tracks.map((t) => t.id));
                const added = await dispatch(
                    addTrackRemote({ name: undefined, parentTrackId: null }),
                ).unwrap();
                targetTrackId =
                    added.tracks.find((t) => !beforeIds.has(t.id))?.id ??
                    added.selected_track_id ??
                    added.tracks[added.tracks.length - 1]?.id ??
                    undefined;
            } else {
                targetTrackId = payload.trackId;
            }

            const imported = await webApi.importMidiAsClip(
                payload.midiPath,
                payload.trackIndices,
                targetTrackId,
                payload.startSec ?? 0,
                payload.fillGaps,
                payload.multiTrackMerge,
                payload.noteBpmMode,
                payload.specifiedBpm,
                payload.importBpmAsProject,
                payload.clipboardGuid,
                payload.closeLeadingGap,
                payload.importAsTempoMap,
                payload.importTempo,
                payload.importTimeSignature,
                payload.importKeySignature,
            );
            if (!(imported as { ok?: boolean }).ok) {
                const errMsg =
                    (imported as { missing_files?: string[] }).missing_files?.[0] ??
                    "import_midi_clip_failed";
                return rejectWithValue(errMsg);
            }
            const result = imported as { clips?: Array<{ id?: string }> };
            const newClipIds = (result.clips ?? [])
                .map((c) => c.id)
                .filter((id): id is string => !!id && !beforeClipIds.has(id));
            return { ok: true, imported, newClipIds };
        } catch (err) {
            return rejectWithValue(err instanceof Error ? err.message : "import_midi_clip_failed");
        } finally {
            void webApi.endUndoGroup();
        }
    },
);
