/**
 * 「按指定格式另存」的端到端测试（thunk 层）。
 *
 * 【为什么要到这一层】第一轮的格式选择住在模态组件里，测试只能覆盖"点按钮以后
 * 状态变了没有"；真正决定用户拿到什么文件的，是这条链路：
 * 记住格式 → `pick_project_save_path(format)` → `save_project_to_path(..., format)`。
 * 任何一环丢了 `format`，用户就会得到"选 JSON 却写出 .hshp"的结果，而组件测试
 * 仍然全绿。
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { configureStore } from "@reduxjs/toolkit";

import reducer, { setProjectLastSaveFormat } from "../sessionSlice";
import { saveProjectAsFormat } from "./projectThunks";
import { webApi } from "../../../services/webviewApi";
import { settingsApi } from "../../../services/api/settings";
import {
    resetSaveProjectAsFailureForTests,
    subscribeSaveProjectAsFailure,
} from "../saveProjectAsNotice";
import type { ProjectSaveFormat } from "../../../services/api/project";

function makeStore(lastSaveFormat: ProjectSaveFormat = "hshp") {
    const store = configureStore({ reducer: { session: reducer } });
    store.dispatch(setProjectLastSaveFormat(lastSaveFormat));
    return store;
}

const EXT: Record<ProjectSaveFormat, string> = { hshp: "hshp", json: "json", zip: "zip" };

beforeEach(() => {
    // 落盘免真实 IPC：这是另存为的副产物，与格式选择无关。
    vi.spyOn(settingsApi, "saveUiSettings").mockResolvedValue({ ok: true } as never);
});

afterEach(() => {
    vi.restoreAllMocks();
    resetSaveProjectAsFailureForTests();
});

describe("saveProjectAsFormat", () => {
    it.each(["hshp", "json", "zip"] as ProjectSaveFormat[])(
        "选路径与写文件都带上 %s 格式",
        async (format) => {
            const path = `D:/Projects/example-project.${EXT[format]}`;
            const pick = vi
                .spyOn(webApi, "pickProjectSavePath")
                .mockResolvedValue({ ok: true, canceled: false, path });
            const save = vi
                .spyOn(webApi, "saveProjectToPath")
                .mockResolvedValue({ ok: true, canceled: false, path });

            const store = makeStore("hshp");
            const result = await store.dispatch(saveProjectAsFormat(format)).unwrap();

            expect(result).toMatchObject({ ok: true, path });
            // 过滤器的格式 = 最终写入的格式，否则 macOS 上会出现"预填 .json 却写 hshp"。
            expect(pick).toHaveBeenCalledWith(format, "Untitled");
            expect(save).toHaveBeenCalledWith(path, "", false, format);
            // 上次格式被记住（`project.saveAs` 与首次保存都读它）。
            expect(store.getState().session.project.lastSaveFormat).toBe(format);
        },
    );

    it("取消只记住格式，不写文件", async () => {
        vi.spyOn(webApi, "pickProjectSavePath").mockResolvedValue({ ok: true, canceled: true });
        const save = vi.spyOn(webApi, "saveProjectToPath");

        const store = makeStore("hshp");
        const result = await store.dispatch(saveProjectAsFormat("json")).unwrap();

        expect(result).toEqual({ ok: true, canceled: true });
        expect(store.getState().session.project.lastSaveFormat).toBe("json");
        expect(save).not.toHaveBeenCalled();
    });

    it("扩展名与格式不一致时把错误码发到提示总线", async () => {
        vi.spyOn(webApi, "pickProjectSavePath").mockResolvedValue({
            ok: true,
            canceled: false,
            path: "D:/Projects/example-project.hshp",
        });
        vi.spyOn(webApi, "saveProjectToPath").mockResolvedValue({
            ok: false,
            error: "format_extension_mismatch",
        });

        const seen: string[] = [];
        subscribeSaveProjectAsFailure((code) => seen.push(code));

        const store = makeStore();
        await expect(store.dispatch(saveProjectAsFormat("json")).unwrap()).rejects.toBe(
            "format_extension_mismatch",
        );
        // 模态删除后错误只走这条通道；不发就等于静默失败。
        expect(seen).toEqual(["format_extension_mismatch"]);
    });

    it("选路径失败也走同一条提示通道", async () => {
        vi.spyOn(webApi, "pickProjectSavePath").mockResolvedValue({
            ok: false,
            error: "unsupported_project_format",
        });
        const seen: string[] = [];
        subscribeSaveProjectAsFailure((code) => seen.push(code));

        const store = makeStore();
        await expect(store.dispatch(saveProjectAsFormat("zip")).unwrap()).rejects.toBe(
            "unsupported_project_format",
        );
        expect(seen).toEqual(["unsupported_project_format"]);
    });

    it("版本冲突作为结果返回而不是失败（确认框链路依赖它）", async () => {
        vi.spyOn(webApi, "pickProjectSavePath").mockResolvedValue({
            ok: true,
            canceled: false,
            path: "D:/Projects/demo-project.hshp",
        });
        vi.spyOn(webApi, "saveProjectToPath").mockResolvedValue({
            ok: false,
            canceled: false,
            versionConflict: true,
            path: "D:/Projects/demo-project.hshp",
            format: "hshp",
            existingVersion: 9,
            currentVersion: 7,
            existingIsNewer: true,
        });

        const seen: string[] = [];
        subscribeSaveProjectAsFailure((code) => seen.push(code));

        const store = makeStore();
        const result = (await store.dispatch(saveProjectAsFormat("hshp")).unwrap()) as {
            versionConflict?: boolean;
        };
        expect(result.versionConflict).toBe(true);
        expect(seen).toEqual([]);
        // reducer 必须把冲突落到确认框状态（用户在这里选「另存为」或「覆盖」）。
        expect(store.getState().session.saveVersionConflictDialog?.path).toBe(
            "D:/Projects/demo-project.hshp",
        );
    });
});
