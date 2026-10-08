/**
 * ARA 提交门禁错误的展示层。
 *
 * 【为什么值得单独测】门禁拒绝是用户唯一能看到"为什么没提交成功"的地方。
 * 后端只给语言无关的分类与字段路径，任何一处拼错前缀都会让用户退回看到
 * 一串原始英文（甚至退回旧行为：无论实际原因是什么都指控"几何/增益/名字"）。
 * 这里把三种分类逐一钉住。
 */
import { describe, expect, test } from "vitest";

import { formatTemplate } from "../../i18n/format";
import { translateOutsideReact } from "../../i18n/I18nProvider";
import { araError, araErrorText } from "./araApi";

const t = (key: string) => translateOutsideReact(key);
const tVars = (key: string, vars: Record<string, string | number>) =>
    formatTemplate(translateOutsideReact(key), vars);

describe("ARA 错误文案", () => {
    test("门禁拒绝时指名漂移字段，而不是笼统指控几何", () => {
        const text = araErrorText(
            new Error("ara_host_fields: clips[0].start_sec: 0.0 -> 0.25"),
            t,
            tVars,
        );
        // 具体路径必须出现在用户可见文案里 —— 否则用户不知道去 REAPER 改什么。
        expect(text).toContain("clips[0].start_sec: 0.0 -> 0.25");
        // 且不是"查不到就回显键名"的退化形态。
        expect(text).not.toContain("ara_submit_blocked_host_fields");
        expect(text).not.toContain("ara_host_fields:");
    });

    test("分类前缀后没有字段时退化为通用文案", () => {
        const text = araErrorText(new Error("ara_host_fields:"), t, tVars);
        expect(text).toBe(t("ara_submit_blocked"));
    });

    test("宿主时间线改变要求重新连接，也走本地化", () => {
        const text = araErrorText(new Error("ara_reconnect_required"), t, tVars);
        expect(text).toBe(t("ara_submit_reconnect_required"));
    });

    test("其余错误原样透传（Conflict 与脏工程诊断不能被吞掉）", () => {
        expect(araErrorText(new Error("Conflict: host model changed; refresh"), t, tVars)).toBe(
            "Conflict: host model changed; refresh",
        );
        expect(araErrorText(new Error("dirty_project: confirm replacement"), t, tVars)).toBe(
            "dirty_project: confirm replacement",
        );
    });

    test("解包调用包装器的 cause", () => {
        const wrapped = new Error("invoke failed", { cause: new Error("ara_reconnect_required") });
        expect(araError(wrapped)).toBe("ara_reconnect_required");
        expect(araErrorText(wrapped, t, tVars)).toBe(t("ara_submit_reconnect_required"));
    });
});
