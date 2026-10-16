/**
 * 宿主链接错误分类的契约。
 *
 * 【为什么值得测】后端只回**语言无关**的分类前缀（`ara_host_fields:` / `dirty_project:`
 * / …），文案在这里本地化。分类一旦漏掉一条，用户看到的就是一段英文原文；而
 * `ara_host_fields:` 漏掉时更糟 —— 用户根本不知道是哪些宿主字段对不上。
 */
import { describe, expect, test } from "vitest";

import { enUS } from "../../i18n/en-US";
import type { MessageKey } from "../../i18n/messages";
import {
    hostLinkError,
    hostLinkErrorText,
    hostLinkStatusText,
    isDirtyProjectError,
} from "./hostLinkApi";

const t = (key: MessageKey): string => enUS[key];
const tVars = (key: MessageKey, vars: Record<string, string | number>): string =>
    enUS[key].replace(/\{(\w+)\}/g, (_, name: string) => String(vars[name] ?? `{${name}}`));

describe("hostLinkErrorText", () => {
    test("几何门禁拒绝时指名字段，而不是给一句泛泛的「被拒绝」", () => {
        const raw = "ara_host_fields: clips[2].start_sec: 0.0 -> 0.25";
        const text = hostLinkErrorText(new Error(raw), t, tVars);
        expect(text).toBe(
            enUS.host_link_submit_blocked_host_fields.replace(
                "{fields}",
                "clips[2].start_sec: 0.0 -> 0.25",
            ),
        );
        // 不得把机器可读前缀或键名摊给用户。
        expect(text).not.toContain("ara_host_fields:");
        expect(text).not.toContain("host_link_submit_blocked_host_fields");
    });

    test("前缀后没有字段时退回「提交被拒绝」", () => {
        expect(hostLinkErrorText(new Error("ara_host_fields:"), t, tVars)).toBe(
            enUS.host_link_submit_blocked,
        );
    });

    test("脏工程 / 已打开 / 未打开 / 不接受编辑各自本地化", () => {
        expect(
            hostLinkErrorText(
                new Error("dirty_project: the current project has unsaved changes"),
                t,
                tVars,
            ),
        ).toBe(enUS.host_link_dirty_project);
        expect(
            hostLinkErrorText(new Error("this workspace is already open: held by 1234"), t, tVars),
        ).toBe(enUS.host_link_already_open);
        expect(hostLinkErrorText(new Error("no host workspace is open"), t, tVars)).toBe(
            enUS.host_link_not_open,
        );
        expect(hostLinkErrorText(new Error("this host does not accept edits"), t, tVars)).toBe(
            enUS.host_link_commit_refused,
        );
    });

    test("未知错误原样透传（由调用方按既有分支处理）", () => {
        expect(hostLinkErrorText(new Error("some other failure"), t, tVars)).toBe(
            "some other failure",
        );
    });
});

describe("hostLinkError", () => {
    test("剥开 invoke 包装的 cause", () => {
        const wrapped = new Error("invoke failed", {
            cause: new Error("dirty_project: unsaved"),
        });
        expect(hostLinkError(wrapped)).toBe("dirty_project: unsaved");
        expect(isDirtyProjectError(wrapped)).toBe(true);
    });

    test("非脏工程错误不触发确认流程", () => {
        expect(isDirtyProjectError(new Error("no host workspace is open"))).toBe(false);
    });
});

describe("hostLinkStatusText", () => {
    const at = 0;

    test("字段路径优先于错误码", () => {
        const text = hostLinkStatusText(
            {
                ok: false,
                code: 13,
                error: "ara_host_fields: clips[0].start_sec: 0.0 -> 0.5",
                atUnixMs: at,
            },
            t,
            tVars,
        );
        expect(text).toContain("clips[0].start_sec: 0.0 -> 0.5");
    });

    test("按码给出可行动的中文说明", () => {
        expect(hostLinkStatusText({ ok: false, code: 12, atUnixMs: at }, t, tVars)).toBe(
            enUS.host_link_code_conflict,
        );
        expect(hostLinkStatusText({ ok: false, code: 11, atUnixMs: at }, t, tVars)).toBe(
            enUS.host_link_code_model_not_ready,
        );
        expect(hostLinkStatusText({ ok: false, code: 15, atUnixMs: at }, t, tVars)).toBe(
            enUS.host_link_code_link_mismatch,
        );
        expect(hostLinkStatusText({ ok: false, code: 10, atUnixMs: at }, t, tVars)).toBe(
            enUS.host_link_code_unsupported_format,
        );
        expect(hostLinkStatusText({ ok: false, code: 1, atUnixMs: at }, t, tVars)).toBe(
            enUS.host_link_code_document_closed,
        );
    });

    test("INVALID_EDIT 带上后端原因", () => {
        const text = hostLinkStatusText(
            { ok: false, code: 14, error: "track 3 is unknown", atUnixMs: at },
            t,
            tVars,
        );
        expect(text).toBe(
            enUS.host_link_code_invalid_edit.replace("{error}", "track 3 is unknown"),
        );
    });
});
