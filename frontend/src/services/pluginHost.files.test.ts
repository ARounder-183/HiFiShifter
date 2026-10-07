// 原生File通道保持实例/Promise隔离；不把音频读成base64，也不给其它命令附加文件。
// @vitest-environment jsdom
import { expect, test } from "vitest";
import { createPluginHost, type WebViewMessagePort } from "./pluginHost";

test("disk File is sent as an additional object without putting its bytes in JSON", async () => {
    const received = new Set<(event: { data: unknown }) => void>();
    let message: unknown;
    let objects: File[] = [];
    const port: WebViewMessagePort = {
        postMessage: () => {
            throw new Error("wrong JSON transport");
        },
        postMessageWithAdditionalObjects: (m, files) => {
            message = m;
            objects = files;
        },
        addEventListener: (_name, fn) => {
            received.add(fn);
        },
        removeEventListener: (_name, fn) => {
            received.delete(fn);
        },
    };
    const host = createPluginHost(port, { version: 1, viewId: "file-view" });
    const file = new File([new Uint8Array(2 * 1024 * 1024)], "元音.wav");
    const pending = host.invoke("import_native_audio_file", { trackId: null, startSec: 3 }, [file]);
    expect(objects).toEqual([file]);
    expect(message).toEqual({
        version: 1,
        viewId: "file-view",
        id: 1,
        command: "import_native_audio_file",
        args: { trackId: null, startSec: 3 },
    });
    expect(JSON.stringify(message).length).toBeLessThan(300);
    received.forEach((fn) =>
        fn({ data: { version: 1, viewId: "file-view", id: 1, ok: true, value: { ok: true } } }),
    );
    await expect(pending).resolves.toEqual({ ok: true });
    host.dispose();
});
test("missing native File bridge rejects without falling back to base64 or leaking a pending call", async () => {
    let sent = 0;
    const host = createPluginHost(
        {
            postMessage: () => {
                sent++;
            },
            addEventListener: () => {},
            removeEventListener: () => {},
        },
        { version: 1, viewId: "old" },
    );
    await expect(
        host.invoke("import_native_audio_file", {}, [new File([], "test.wav")]),
    ).rejects.toThrow("File menu");
    await expect(host.invoke("set_track_state", {}, [new File([], "test.wav")])).rejects.toThrow(
        "File menu",
    );
    expect(sent).toBe(0);
    host.dispose();
});
