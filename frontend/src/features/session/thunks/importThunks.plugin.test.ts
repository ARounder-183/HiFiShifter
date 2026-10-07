// 插件多文件流程使用真实thunk/store与宿主确认的clip ID；桩只替代IPC，不按光标位置认新clip。
// @vitest-environment jsdom
import {configureStore} from "@reduxjs/toolkit";
import {afterEach,beforeEach,expect,test,vi} from "vitest";
vi.mock("../../../services/webviewApi",()=>({webApi:{
    beginUndoGroup:vi.fn(async()=>({ok:true})),endUndoGroup:vi.fn(async()=>({ok:true})),
    importAudioItem:vi.fn(),addTrackNested:vi.fn(),
}}));
import {webApi} from "../../../services/webviewApi";
import reducer from "../sessionSlice";
import {importMultipleAudioAtPosition} from "./importThunks";
beforeEach(()=>{vi.clearAllMocks();window.__HFS_PLUGIN_BOOTSTRAP__={version:1,viewId:"imports",audioImport:true};});
afterEach(()=>{delete window.__HFS_PLUGIN_BOOTSTRAP__;});

function backend() {
    const clips:Array<{id:string;track_id:string;name:string;start_sec:number;length_sec:number}>=[];
    const tracks:Array<{id:string;name:string;order:number}>=[];
    vi.mocked(webApi.importAudioItem).mockImplementation(async(path,track,start)=>{
        const target=track??`native-${tracks.length+1}`;
        if(!tracks.some(t=>t.id===target)) tracks.push({id:target,name:target,order:tracks.length});
        const id=`clip-${clips.length+1}`;clips.push({id,track_id:target,name:path,start_sec:start??0,length_sec:2});
        return {ok:true,tracks:[...tracks],clips:[...clips],selected_track_id:target,selected_clip_id:id,
            bpm:120,playhead_sec:0,project_sec:10,imported_clip_id:id} as never;
    });
    return configureStore({reducer:{session:reducer}});
}
test("empty plugin track imports through direct parent then reuses its returned ID and actual clip length",async()=>{
    const store=backend();
    await store.dispatch(importMultipleAudioAtPosition({audioPaths:["E:/a.wav","E:/b.wav"],mode:"across-time",trackId:null,startSec:3})).unwrap();
    expect(webApi.addTrackNested).not.toHaveBeenCalled();
    expect(webApi.importAudioItem).toHaveBeenNthCalledWith(1,"E:/a.wav",undefined,3);
    expect(webApi.importAudioItem).toHaveBeenNthCalledWith(2,"E:/b.wav","native-1",5);
    expect(webApi.beginUndoGroup).toHaveBeenCalledTimes(1);expect(webApi.endUndoGroup).toHaveBeenCalledTimes(1);
});
test("across-tracks uses empty plugin parent once then requests new host tracks",async()=>{
    const store=backend();
    await store.dispatch(importMultipleAudioAtPosition({audioPaths:["E:/a.wav","E:/b.wav"],mode:"across-tracks",trackId:null,startSec:1})).unwrap();
    expect(webApi.importAudioItem).toHaveBeenNthCalledWith(1,"E:/a.wav",undefined,1);
    expect(webApi.importAudioItem).toHaveBeenNthCalledWith(2,"E:/b.wav",null,1);
    expect(webApi.addTrackNested).not.toHaveBeenCalled();
});
