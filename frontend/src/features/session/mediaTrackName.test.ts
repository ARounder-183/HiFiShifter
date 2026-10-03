/**
 * 导入建轨的命名：`trackNameForMedia`。
 *
 * 【要钉住的边界】"去掉最后一个扩展名"在 Windows 路径上有几个坑：盘符里的点、
 * 以点开头的隐藏文件、无扩展名、名字里本来就有点。此前 `importThunks.fileStem`
 * 与 `notebookPaths.stemName` 各写一份，分叉风险正是本模块要消除的。
 */

import { describe, expect, it } from "vitest";

import { trackNameForMedia } from "./mediaTrackName";

describe("trackNameForMedia", () => {
    it("完整路径取主名（去最后一个扩展名）", () => {
        expect(trackNameForMedia("C:\\music\\vocal take 01.wav")).toBe("vocal take 01");
        expect(trackNameForMedia("/home/u/Drums/Bass/kick.mp3")).toBe("kick");
    });

    it("裸文件名同样适用", () => {
        expect(trackNameForMedia("vocal take 01.wav")).toBe("vocal take 01");
    });

    it("无扩展名时保留原名", () => {
        expect(trackNameForMedia("C:\\a\\README")).toBe("README");
    });

    it("以点开头的隐藏文件保留原名（dot > 0 的边界）", () => {
        expect(trackNameForMedia("C:\\proj\\.gitignore")).toBe(".gitignore");
        expect(trackNameForMedia(".env")).toBe(".env");
    });

    it("名字里有多个点时只去最后一个", () => {
        expect(trackNameForMedia("a.b.c.wav")).toBe("a.b.c");
        expect(trackNameForMedia("C:\\x\\take.01.final.flac")).toBe("take.01.final");
    });

    it("目录名里的点不影响（只看最后一段）", () => {
        expect(trackNameForMedia("C:\\my.folder\\clip.wav")).toBe("clip");
    });

    it("视频容器同样去扩展名（导入建轨不分音频/视频）", () => {
        expect(trackNameForMedia("C:\\v\\take.mp4")).toBe("take");
    });
});
