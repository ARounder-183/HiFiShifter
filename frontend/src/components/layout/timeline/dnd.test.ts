/**
 * 拖放准入（`timeline/dnd`）行为自检。
 *
 * 【本测试要钉住的核心不变量】
 * 拖放此前**接受任意扩展名**并一律按音频导入：`detectExternalPathAction` 对未知
 * 扩展名返回 `null`，而各 drop 分支只对"已知的非音频种类"做了特殊处理，`null` 于是
 * **穿透**到音频导入的默认分支，被当成 Clip 添加。后端只做内容嗅探，于是任何能被
 * 解码的文件——甚至扩展名完全无关的文件——都会被接受。
 *
 * 因此这里逐类验证准入白名单，以及"未知类型必须被拒绝"这条判据本身。
 */
import { describe, expect, it } from "vitest";

import {
    detectExternalPathAction,
    isAcceptedDropFile,
    isAcceptedDropPath,
    partitionDroppedPaths,
} from "./dnd";

/** 应当被接受的路径（按需求列出的四类）。 */
const ACCEPTED: readonly { path: string; kind: string }[] = [
    // ① 媒体文件（音频）
    { path: "C:/audio/vocal.wav", kind: "importAudio" },
    { path: "C:/audio/vocal.FLAC", kind: "importAudio" },
    { path: "C:/audio/vocal.mp3", kind: "importAudio" },
    { path: "C:/audio/vocal.m4a", kind: "importAudio" },
    { path: "C:/audio/vocal.aiff", kind: "importAudio" },
    { path: "C:/audio/vocal.opus", kind: "importAudio" },
    { path: "C:/audio/vocal.ape", kind: "importAudio" },
    // ① 媒体文件（视频容器——会读取其中的音频轨）
    { path: "C:/video/take.mp4", kind: "importAudio" },
    { path: "C:/video/take.mov", kind: "importAudio" },
    { path: "C:/video/take.mkv", kind: "importAudio" },
    { path: "C:/video/take.webm", kind: "importAudio" },
    // `.ts` 是 MPEG Transport Stream（视频容器），与 `.tsx`（源码）不同——后端
    // `VIDEO_EXTENSIONS` 与媒体文件对话框都支持它，因此必须接受。
    { path: "C:/video/capture.ts", kind: "importAudio" },
    // ② 本工程文件
    { path: "C:/proj/song.hshp", kind: "openProject" },
    { path: "C:/proj/song.hsp", kind: "openProject" },
    { path: "C:/proj/song.json", kind: "openProject" },
    // ② 本工程文件的 -bak 备份（与正本同格式）
    { path: "C:/proj/song.hshp-bak", kind: "openProject" },
    { path: "C:/proj/song.hsp-bak", kind: "openProject" },
    // ③ 可导入的外部工程格式
    { path: "C:/proj/song.rpp", kind: "importReaper" },
    { path: "C:/proj/song.rpp-bak", kind: "importReaper" },
    { path: "C:/proj/song.vshp", kind: "importVocalShifter" },
    { path: "C:/proj/song.vsp", kind: "importVocalShifter" },
    // ④ MIDI
    { path: "C:/midi/melody.mid", kind: "importMidi" },
    { path: "C:/midi/melody.midi", kind: "importMidi" },
    { path: "C:/midi/melody.smf", kind: "importMidi" },
];

/** 必须被拒绝的路径（曾经的缺陷：它们会被当成音频导入）。 */
const REJECTED: readonly string[] = [
    "C:/docs/readme.txt",
    "C:/docs/notes.pdf",
    "C:/docs/notes.docx",
    "C:/docs/notes.xlsx",
    "C:/img/cover.png",
    "C:/img/cover.jpg",
    "C:/img/cover.webp",
    "C:/img/logo.svg",
    // 注：`.ts` **不在**此列——它是 MPEG Transport Stream（视频容器），后端
    // `VIDEO_EXTENSIONS` 与文件对话框的媒体过滤器都显式支持它，因此必须接受。
    // 这里刻意保留这个"看起来像源码"的样本，以防有人日后把它当笔误删掉。
    "C:/src/main.tsx",
    "C:/archive/data.zip",
    "C:/archive/data.7z",
    "C:/app/program.exe",
    "C:/app/program.dll",
    "C:/bin/data.bin",
    "C:/bin/data.dat",
    "C:/data/table.csv",
    "C:/data/table.tsv",
    "C:/page/index.html",
    "C:/style/site.css",
    "C:/script/run.js",
    "C:/script/run.py",
    "C:/misc/noextension",
    "C:/misc/trailingdot.",
    "C:/misc/just.wav.txt", // 真正的扩展名是 .txt
    "C:/misc/space in name.tar.gz",
    "",
    "   ",
];

describe("拖放准入：接受白名单内的类型", () => {
    it("四类文件全部被接受，且落在本该落的动作种类上", () => {
        for (const { path, kind } of ACCEPTED) {
            expect(isAcceptedDropPath(path), `应接受：${path}`).toBe(true);
            expect(detectExternalPathAction(path), `种类：${path}`).toBe(kind);
        }
    });

    it("空值 / 空白一律不接受", () => {
        for (const bad of [null, undefined, "", "   "]) {
            expect(isAcceptedDropPath(bad)).toBe(false);
        }
    });
});

describe("拖放准入：拒绝其它所有类型（缺陷回归）", () => {
    it("无关扩展名不得被接受", () => {
        for (const path of REJECTED) {
            expect(isAcceptedDropPath(path), `应拒绝：${path}`).toBe(false);
            // 关键判据：未知类型不得返回 `importAudio`（否则会穿透到建 Clip 的分支）。
            expect(detectExternalPathAction(path), `种类：${path}`).toBeNull();
        }
    });

    it("判据与 detectExternalPathAction 同源（不分叉）", () => {
        // 两条路径若各写一份正则，"预览接受但落下被拒"这类不一致只能靠肉眼发现。
        // 这里对所有样本交叉验证：接受 ⟺ 能识别出动作种类。
        for (const { path } of ACCEPTED) {
            expect(isAcceptedDropPath(path)).toBe(detectExternalPathAction(path) !== null);
        }
        for (const path of REJECTED) {
            expect(isAcceptedDropPath(path)).toBe(detectExternalPathAction(path) !== null);
        }
    });
});

describe("DOM File 的准入（无本地路径时的兜底）", () => {
    it("按扩展名判定", () => {
        expect(isAcceptedDropFile({ name: "take.wav", type: "" })).toBe(true);
        expect(isAcceptedDropFile({ name: "melody.mid", type: "" })).toBe(true);
        expect(isAcceptedDropFile({ name: "song.hshp-bak", type: "" })).toBe(true);
    });

    it("扩展名未知但 MIME 表态时接受", () => {
        expect(isAcceptedDropFile({ name: "stream", type: "audio/wav" })).toBe(true);
        expect(isAcceptedDropFile({ name: "stream", type: "video/mp4" })).toBe(true);
        // MIDI 的标准 MIME 属 audio/*
        expect(isAcceptedDropFile({ name: "melody", type: "audio/midi" })).toBe(true);
    });

    it("扩展名与 MIME 都无关时拒绝", () => {
        expect(isAcceptedDropFile({ name: "notes.pdf", type: "application/pdf" })).toBe(false);
        expect(isAcceptedDropFile({ name: "cover.png", type: "image/png" })).toBe(false);
        expect(isAcceptedDropFile({ name: "data.zip", type: "application/zip" })).toBe(false);
        // `type` 为空是常态（系统不知道 MIME），此时不能据此放行
        expect(isAcceptedDropFile({ name: "notes.pdf", type: "" })).toBe(false);
    });

    it("空对象 / null 拒绝", () => {
        expect(isAcceptedDropFile(null)).toBe(false);
        expect(isAcceptedDropFile(undefined)).toBe(false);
        expect(isAcceptedDropFile({})).toBe(false);
    });
});

describe("partitionDroppedPaths：多文件按类型分类", () => {
    it("媒体 / MIDI / 工程各归其位，未知类型被丢弃", () => {
        const result = partitionDroppedPaths([
            "a.wav",
            "notes.txt", // 拒绝
            "melody.mid",
            "b.mp3",
            "cover.png", // 拒绝
            "song.hshp-bak",
        ]);
        expect(result.mediaPaths).toEqual(["a.wav", "b.mp3"]);
        expect(result.midiPaths).toEqual(["melody.mid"]);
        expect(result.projectPath).toBe("song.hshp-bak");
        expect(result.rejectedPaths).toEqual(["notes.txt", "cover.png"]);
    });

    it("媒体文件保持原顺序（拖放顺序即落点顺序）", () => {
        const result = partitionDroppedPaths(["c.flac", "a.wav", "b.ogg"]);
        expect(result.mediaPaths).toEqual(["c.flac", "a.wav", "b.ogg"]);
    });

    it("多个工程只取第一个（打开会替换当前会话，语义唯一）", () => {
        const result = partitionDroppedPaths(["first.hshp", "second.hshp"]);
        expect(result.projectPath).toBe("first.hshp");
        expect(result.mediaPaths).toEqual([]);
    });

    it("全部未知时不产出任何可导入项", () => {
        const result = partitionDroppedPaths(["a.txt", "b.pdf", "c.exe"]);
        expect(result.mediaPaths).toEqual([]);
        expect(result.midiPaths).toEqual([]);
        expect(result.projectPath).toBeNull();
        expect(result.rejectedPaths).toEqual(["a.txt", "b.pdf", "c.exe"]);
    });

    it("空输入安全返回", () => {
        const result = partitionDroppedPaths([]);
        expect(result.mediaPaths).toEqual([]);
        expect(result.midiPaths).toEqual([]);
        expect(result.projectPath).toBeNull();
        expect(result.rejectedPaths).toEqual([]);
    });

    it("跳过空项与 null", () => {
        const result = partitionDroppedPaths([null, "  ", "a.wav", undefined]);
        expect(result.mediaPaths).toEqual(["a.wav"]);
        expect(result.rejectedPaths).toEqual([]);
    });
});

// ── 目录准入 ────────────────────────────────────────────────────────────
//
// 【为什么 isDir 必须优先于扩展名】目录名可以带任何后缀。一个叫 `take.wav` 的
// 文件夹不是音频，一个叫 `song.hshp` 的文件夹不是工程。判据本身是纯字符串函数，
// 无法 stat，所以目录信息由调用方提供；提供时它压过所有扩展名判断。

describe("目录准入：isDir 优先于扩展名", () => {
    it("同一路径，作为目录一律是 importFolder", () => {
        expect(detectExternalPathAction("C:/audio/take.wav")).toBe("importAudio");
        expect(detectExternalPathAction("C:/audio/take.wav", { isDir: true })).toBe(
            "importFolder",
        );
        // 工程后缀同理：不能因为目录叫 song.hshp 就去打开它。
        expect(detectExternalPathAction("C:/proj/song.hshp", { isDir: true })).toBe(
            "importFolder",
        );
    });

    it("没有扩展名的目录也能被认出来（这正是扩展名判据的盲区）", () => {
        expect(detectExternalPathAction("C:/misc/noextension")).toBeNull();
        expect(detectExternalPathAction("C:/misc/noextension", { isDir: true })).toBe(
            "importFolder",
        );
    });

    it("不提供 isDir 时行为与改动前逐字节一致（目录仍被拒）", () => {
        for (const path of ["C:/music/Takes", "C:/misc/noextension", "C:/audio/take.wav/"]) {
            expect(isAcceptedDropPath(path), `应被拒：${path}`).toBe(false);
        }
    });
});

describe("partitionDroppedPaths：目录单独成类", () => {
    it("已知是目录的路径进 folderPaths，不再落进 rejectedPaths", () => {
        const result = partitionDroppedPaths(["C:/music/Takes", "a.wav"], {
            directories: new Set(["C:/music/Takes"]),
        });
        expect(result.folderPaths).toEqual(["C:/music/Takes"]);
        expect(result.mediaPaths).toEqual(["a.wav"]);
        expect(result.rejectedPaths).toEqual([]);
    });

    it("不传 directories 时目录仍被拒（既有调用方行为不变）", () => {
        const result = partitionDroppedPaths(["C:/music/Takes", "a.wav"]);
        expect(result.folderPaths).toEqual([]);
        expect(result.rejectedPaths).toEqual(["C:/music/Takes"]);
    });

    it("目录与媒体混拖时各自保持拖放顺序", () => {
        const result = partitionDroppedPaths(["C:/c", "x.wav", "C:/a", "y.wav", "C:/b"], {
            directories: new Set(["C:/a", "C:/b", "C:/c"]),
        });
        expect(result.folderPaths).toEqual(["C:/c", "C:/a", "C:/b"]);
        expect(result.mediaPaths).toEqual(["x.wav", "y.wav"]);
    });
});
