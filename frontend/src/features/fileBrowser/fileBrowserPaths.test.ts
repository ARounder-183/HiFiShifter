import { describe, expect, it } from "vitest";

import { FILE_BROWSER_COMPUTER_PATH } from "./fileBrowserSlice";
import { locationLabel, parentDirOf } from "./fileBrowserPaths";

describe("parentDirOf", () => {
    it("Windows：取上一级，盘符根保留尾部反斜杠", () => {
        expect(parentDirOf("D:\\music\\takes\\a.wav")).toBe("D:\\music\\takes");
        expect(parentDirOf("D:\\music")).toBe("D:\\");
        expect(parentDirOf("D:\\")).toBeNull();
    });

    it("POSIX：取上一级，根目录没有上级", () => {
        expect(parentDirOf("/home/me/takes/a.wav")).toBe("/home/me/takes");
        expect(parentDirOf("/home")).toBeNull();
        expect(parentDirOf("/")).toBeNull();
    });

    it("尾部斜杠不影响结果", () => {
        expect(parentDirOf("/home/me/takes/")).toBe("/home/me");
        expect(parentDirOf("D:\\music\\takes\\")).toBe("D:\\music");
    });

    it("正斜杠形式的 Windows 路径同样处理", () => {
        expect(parentDirOf("D:/music/takes")).toBe("D:/music");
        expect(parentDirOf("D:/music")).toBe("D:/");
    });
});

describe("locationLabel", () => {
    it("取路径最后一段", () => {
        expect(locationLabel("D:\\music\\takes", "This PC")).toBe("takes");
        expect(locationLabel("/home/me/takes/", "This PC")).toBe("takes");
    });

    it("盘符根显示成裸盘符", () => {
        expect(locationLabel("D:\\", "This PC")).toBe("D:");
        expect(locationLabel("D:/", "This PC")).toBe("D:");
    });

    it("「计算机」用传入的本地化名", () => {
        expect(locationLabel(FILE_BROWSER_COMPUTER_PATH, "计算机")).toBe("计算机");
    });

    it("无法切分时原样返回", () => {
        expect(locationLabel("relative", "This PC")).toBe("relative");
    });
});
