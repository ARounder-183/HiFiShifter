/**
 * `uiStorage` 的回归测试。
 *
 * 【要钉死什么】
 * 1. 白名单里的键写入后端（否则"设置换形态就丢"这个 bug 会悄悄回来）；
 * 2. 白名单外的键**不**写入后端（调试开关不该跟着用户配置走）；
 * 3. 去抖合并：连续多次写同一键只提交一次；
 * 4. hydrate 以后端为准，但把本机独有的键迁上去（升级路径）；
 * 5. 后端不可用时静默降级，不抛错 —— 界面起不来比设置丢更糟。
 */
// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
    PERSISTED_UI_KEYS,
    flushUiStorage,
    hydrateUiStorage,
    modeKey,
    readUiValue,
    removeUiValue,
    setUiStorageTransportForTest,
    writeUiValue,
    type UiStorageTransport,
} from "./uiStorage";

const LOCALE_KEY = "hifishifter.locale";
const KEYBINDINGS_KEY = "hifishifter.keybindings";
const DEBUG_KEY = "hifishifter.debugDnd";

interface Recording extends UiStorageTransport {
    readonly puts: Record<string, string>[];
    readonly deletes: string[][];
    dumped: Record<string, string>;
}

function recordingTransport(dumped: Record<string, string> = {}): Recording {
    const puts: Record<string, string>[] = [];
    const deletes: string[][] = [];
    const transport = {
        puts,
        deletes,
        dumped,
        async dump() {
            return transport.dumped;
        },
        async put(patch: Record<string, string>) {
            puts.push(patch);
        },
        async remove(keys: string[]) {
            deletes.push(keys);
        },
    };
    return transport;
}

beforeEach(() => {
    localStorage.clear();
    setUiStorageTransportForTest(null);
});

afterEach(() => {
    setUiStorageTransportForTest(null);
    vi.useRealTimers();
});

describe("同步缓存", () => {
    test("读写走 localStorage，读取立即可见", () => {
        writeUiValue(LOCALE_KEY, "ja-JP");
        expect(readUiValue(LOCALE_KEY)).toBe("ja-JP");
    });

    test("删除后读不到", () => {
        writeUiValue(LOCALE_KEY, "ko-KR");
        removeUiValue(LOCALE_KEY);
        expect(readUiValue(LOCALE_KEY)).toBeNull();
    });

    test("没有 localStorage 时不抛错", () => {
        const original = globalThis.localStorage;
        // @ts-expect-error 故意制造"没有 localStorage"的环境
        delete globalThis.localStorage;
        try {
            expect(() => writeUiValue(LOCALE_KEY, "en-US")).not.toThrow();
            expect(readUiValue(LOCALE_KEY)).toBeNull();
        } finally {
            Object.defineProperty(globalThis, "localStorage", {
                value: original,
                configurable: true,
                writable: true,
            });
        }
    });
});

describe("后端同步", () => {
    test("白名单键写入后端", async () => {
        const transport = recordingTransport();
        setUiStorageTransportForTest(transport);

        writeUiValue(LOCALE_KEY, "zh-TW");
        await flushUiStorage();

        expect(transport.puts).toEqual([{ [LOCALE_KEY]: "zh-TW" }]);
    });

    test("白名单外的键不写入后端", async () => {
        const transport = recordingTransport();
        setUiStorageTransportForTest(transport);

        writeUiValue(DEBUG_KEY, "1");
        await flushUiStorage();

        expect(transport.puts).toEqual([]);
        // 但本机缓存仍然写进去了：调试开关照旧生效。
        expect(readUiValue(DEBUG_KEY)).toBe("1");
    });

    test("连续写同一键只提交最后一次", async () => {
        const transport = recordingTransport();
        setUiStorageTransportForTest(transport);

        writeUiValue(KEYBINDINGS_KEY, "v1");
        writeUiValue(KEYBINDINGS_KEY, "v2");
        writeUiValue(KEYBINDINGS_KEY, "v3");
        await flushUiStorage();

        expect(transport.puts).toEqual([{ [KEYBINDINGS_KEY]: "v3" }]);
    });

    test("删除只提交键名", async () => {
        const transport = recordingTransport();
        setUiStorageTransportForTest(transport);

        removeUiValue(LOCALE_KEY);
        await flushUiStorage();

        expect(transport.deletes).toEqual([[LOCALE_KEY]]);
        expect(transport.puts).toEqual([]);
    });

    test("同一轮里先写后删，以删除为准", async () => {
        const transport = recordingTransport();
        setUiStorageTransportForTest(transport);

        writeUiValue(LOCALE_KEY, "ja-JP");
        removeUiValue(LOCALE_KEY);
        await flushUiStorage();

        expect(transport.puts).toEqual([]);
        expect(transport.deletes).toEqual([[LOCALE_KEY]]);
    });

    test("后端失败只告警，不向调用方抛出", async () => {
        const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
        setUiStorageTransportForTest({
            async dump() {
                return {};
            },
            async put() {
                throw new Error("backend down");
            },
            async remove() {
                throw new Error("backend down");
            },
        });

        writeUiValue(LOCALE_KEY, "ja-JP");
        await expect(flushUiStorage()).resolves.toBeUndefined();
        expect(warn).toHaveBeenCalled();
        warn.mockRestore();
    });
});

describe("hydrate", () => {
    test("后端值覆盖本机缓存", async () => {
        localStorage.setItem(LOCALE_KEY, "en-US");
        setUiStorageTransportForTest(recordingTransport({ [LOCALE_KEY]: "ko-KR" }));

        await hydrateUiStorage();

        expect(readUiValue(LOCALE_KEY)).toBe("ko-KR");
    });

    test("本机独有的键被迁到后端（升级路径）", async () => {
        localStorage.setItem(KEYBINDINGS_KEY, '{"a":1}');
        const transport = recordingTransport({ [LOCALE_KEY]: "ja-JP" });
        setUiStorageTransportForTest(transport);

        await hydrateUiStorage();

        expect(transport.puts).toEqual([{ [KEYBINDINGS_KEY]: '{"a":1}' }]);
    });

    test("两端都有的键不重复上传", async () => {
        localStorage.setItem(LOCALE_KEY, "en-US");
        const transport = recordingTransport({ [LOCALE_KEY]: "ko-KR" });
        setUiStorageTransportForTest(transport);

        await hydrateUiStorage();

        expect(transport.puts).toEqual([]);
    });

    test("后端读取失败时保留本机缓存，不抛错", async () => {
        const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
        localStorage.setItem(LOCALE_KEY, "en-US");
        setUiStorageTransportForTest({
            async dump() {
                throw new Error("backend down");
            },
            async put() {},
            async remove() {},
        });

        await expect(hydrateUiStorage()).resolves.toBeUndefined();
        expect(readUiValue(LOCALE_KEY)).toBe("en-US");
        warn.mockRestore();
    });

    test("非白名单键不会被后端的值污染", async () => {
        setUiStorageTransportForTest(recordingTransport({ [DEBUG_KEY]: "1" }));
        await hydrateUiStorage();
        expect(readUiValue(DEBUG_KEY)).toBeNull();
    });
});

describe("白名单", () => {
    test("覆盖语言、快捷键、外观、缩放与上次目录", () => {
        for (const key of [
            "hifishifter.locale",
            "hifishifter.keybindings",
            "hifishifter.appearance",
            "hifishifter.customThemes",
            "hifishifter.pxPerSec",
            "hifishifter.rowHeight",
            "hifishifter.paramPxPerSec",
            "hifishifter.fileBrowser.lastPath",
        ]) {
            expect(PERSISTED_UI_KEYS).toContain(key);
        }
    });

    test("调试开关不在白名单里", () => {
        for (const key of [
            "hifishifter.debugDnd",
            "hifishifter.debugPianoRoll",
            "hifishifter.frameProfiler",
            "hifishifter.perfProject",
        ]) {
            expect(PERSISTED_UI_KEYS).not.toContain(key);
        }
    });
});

describe("按形态分键", () => {
    const ZOOM_KEY = "hifishifter.pxPerSec";

    afterEach(() => {
        delete window.__HFS_PLUGIN_BOOTSTRAP__;
    });

    /** 独立 App 用基础键名，插件用 `.plugin` 后缀 —— 两者的值互不可见。 */
    test("plugin mode stores viewport-dependent keys under their own name", () => {
        expect(modeKey(ZOOM_KEY)).toBe(ZOOM_KEY);
        expect(modeKey(LOCALE_KEY)).toBe(LOCALE_KEY);

        window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "zoom" };
        expect(modeKey(ZOOM_KEY)).toBe(`${ZOOM_KEY}.plugin`);
        // 语言这类"取决于用户是谁"的偏好必须仍然共用同一个键。
        expect(modeKey(LOCALE_KEY)).toBe(LOCALE_KEY);
        expect(modeKey(KEYBINDINGS_KEY)).toBe(KEYBINDINGS_KEY);
    });

    /** 两个形态的缩放互不覆盖：这正是本次要修的故障。 */
    test("the two modes keep separate zoom values", () => {
        writeUiValue(ZOOM_KEY, "120");
        window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "zoom" };
        expect(readUiValue(ZOOM_KEY)).toBeNull();
        writeUiValue(ZOOM_KEY, "40");
        expect(readUiValue(ZOOM_KEY)).toBe("40");

        delete window.__HFS_PLUGIN_BOOTSTRAP__;
        expect(readUiValue(ZOOM_KEY)).toBe("120");
    });

    /** 分形态的键也要进白名单，否则既不同步后端、灌回时也会被跳过。 */
    test("both variants are persisted", async () => {
        // 待提交队列是模块级的，前一条测试的写入还排在队里 —— 先排空，
        // 否则这条断言会把它的键也算进来（而它测的是别的事）。
        setUiStorageTransportForTest(recordingTransport());
        await flushUiStorage();

        expect(PERSISTED_UI_KEYS).toContain(ZOOM_KEY);
        window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "zoom" };
        const transport = recordingTransport();
        setUiStorageTransportForTest(transport);
        writeUiValue(ZOOM_KEY, "40");
        await flushUiStorage();
        expect(transport.puts).toEqual([{ [`${ZOOM_KEY}.plugin`]: "40" }]);
    });

    /** 灌回时后端的两份值各归各位。 */
    test("hydrate restores both variants without mixing them", async () => {
        const transport = recordingTransport({
            [ZOOM_KEY]: "120",
            [`${ZOOM_KEY}.plugin`]: "40",
        });
        setUiStorageTransportForTest(transport);
        await hydrateUiStorage();

        expect(readUiValue(ZOOM_KEY)).toBe("120");
        window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "zoom" };
        expect(readUiValue(ZOOM_KEY)).toBe("40");
    });
});
