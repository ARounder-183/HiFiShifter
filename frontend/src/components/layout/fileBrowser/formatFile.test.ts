import { expect, test } from "vitest";

import { formatSize } from "./formatFile";

const KB = 1024;
const MB = KB * 1024;
const GB = MB * 1024;
const TB = GB * 1024;

test("formatSize 覆盖 B / KB / MB / GB / TB 各档", () => {
    expect(formatSize(null)).toBe("");
    expect(formatSize(512)).toBe("512 B");
    expect(formatSize(2 * KB)).toBe("2 KB");
    expect(formatSize(MB)).toBe("1.0 MB");
    expect(formatSize(1.5 * MB)).toBe("1.5 MB");
});

test("formatSize 对 ≥1 GiB 的文件给出 GB / TB 档，而不是 4096.0 MB", () => {
    expect(formatSize(GB)).toBe("1.0 GB");
    expect(formatSize(4 * GB)).toBe("4.0 GB");
    expect(formatSize(TB)).toBe("1.0 TB");
});
