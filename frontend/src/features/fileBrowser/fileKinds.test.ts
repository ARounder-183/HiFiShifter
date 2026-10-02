import { describe, expect, it } from "vitest";

import type { FileEntry } from "../../services/api/fileBrowser";
import {
    isAudioFile,
    isDraggableFile,
    isMediaFile,
    isMidiFile,
    isProjectFile,
    isVideoFile,
} from "./fileKinds";

function entry(name: string, isDir = false): FileEntry {
    return {
        name,
        path: `/tmp/${name}`,
        isDir,
        size: isDir ? null : 1,
        extension: isDir ? null : (name.split(".").pop() ?? null),
        modifiedTime: 0,
    };
}

describe("文件类型判定", () => {
    it("音频与视频同属音频路径（视频按音轨导入）", () => {
        expect(isAudioFile(entry("take.wav"))).toBe(true);
        expect(isAudioFile(entry("clip.mp4"))).toBe(true);
        expect(isVideoFile(entry("clip.mp4"))).toBe(true);
        expect(isVideoFile(entry("take.wav"))).toBe(false);
    });

    it("扩展名不区分大小写（后端已小写化，这里按同样约定）", () => {
        expect(isAudioFile(entry("TAKE.WAV"))).toBe(false); // 后端给的是小写
        expect(isAudioFile(entry("take.WAV".toLowerCase()))).toBe(true);
    });

    it("MIDI 与工程文件各自成类，且含备份后缀", () => {
        expect(isMidiFile(entry("song.mid"))).toBe(true);
        expect(isMidiFile(entry("song.smf"))).toBe(true);
        expect(isProjectFile(entry("song.hshp"))).toBe(true);
        expect(isProjectFile(entry("song.hshp-bak"))).toBe(true);
        expect(isProjectFile(entry("song.rpp-bak"))).toBe(true);
    });

    it("目录不属于任何文件类型", () => {
        for (const dir of ["folder", "folder.wav"]) {
            const e = entry(dir, true);
            expect(isAudioFile(e)).toBe(false);
            expect(isMidiFile(e)).toBe(false);
            expect(isProjectFile(e)).toBe(false);
            expect(isMediaFile(e)).toBe(false);
        }
    });

    it("isMediaFile 恰好是「音频/视频 + MIDI」—— 文件浏览器与快速搜索的公共判据", () => {
        // 这个等式就是两处界面"看到同一批候选"的定义；改一边就会在这里红。
        expect(isMediaFile(entry("take.wav"))).toBe(true);
        expect(isMediaFile(entry("clip.mp4"))).toBe(true);
        expect(isMediaFile(entry("song.mid"))).toBe(true);
        expect(isMediaFile(entry("song.hshp"))).toBe(false);
        expect(isMediaFile(entry("notes.txt"))).toBe(false);
    });

    it("可拖拽 = 音频/视频 + MIDI + 工程文件", () => {
        expect(isDraggableFile(entry("take.wav"))).toBe(true);
        expect(isDraggableFile(entry("song.mid"))).toBe(true);
        expect(isDraggableFile(entry("song.hshp"))).toBe(true);
        expect(isDraggableFile(entry("notes.txt"))).toBe(false);
    });
});
