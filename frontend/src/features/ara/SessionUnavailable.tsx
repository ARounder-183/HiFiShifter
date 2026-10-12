/**
 * 插件会话不可用 —— 整页提示。
 *
 * 【为什么需要一整页，而不是状态栏里一个片】
 * 会话在**启动前**就已经是终态时（宿主从未把本插件接入 ARA），界面此前会正常挂载，
 * 然后每一批初始化命令同时失败：实测 Cubase 上那一秒里有 15 条 `Invoke failed` 进日志，
 * 用户看到的是一个空时间线加状态栏里一闪而过的错误 —— 既不知道发生了什么，也不知道
 * 下一步该做什么。
 *
 * 这里在挂载 App **之前**做一次探活（见 `mount.tsx`），终态就直接渲染本页：一条能懂的
 * 说明 + 一个可执行的下一步（打开日志目录）。副作用是那 15 条日志收敛成 1 条。
 *
 * 【为什么按协议码分支，不解析错误文本】与 `pluginHost.ts` / `sessionGate.ts` 同一纪律。
 * 码是协议承诺（只增不改），文本只做展示。旧后端不带码时退回"会话已结束"那条通用说明
 * —— 它对"真的关闭了"是准确的。
 *
 * 【为什么用 `AppButton` 与原生 `<span>` 而不是 Radix 的 `Button`/`Text`】
 * `src/ui/araConformanceGates.test.ts` 与 `src/ui/typography.test.ts` 是两道棘轮门禁：
 * 直用 Radix 表单原语的文件数有上限，`<Text>` 在 `src/ui` 之外一律禁止。排版走
 * `hs-type-*` 角色类（与 `HostAudioNotice` 同一套）。
 */
import { Flex } from "@radix-ui/themes";
import { useState, useSyncExternalStore } from "react";

import { useI18n } from "../../i18n/I18nProvider";
import { openLogFolder } from "../../services/api/diagnostics";
import {
    ERROR_CODE,
    getSessionGateSnapshot,
    subscribeSessionGate,
} from "../../services/sessionGate";
import { AppButton } from "../../ui";

export function SessionUnavailable() {
    const { t } = useI18n();
    const [opening, setOpening] = useState(false);
    const terminal = useSyncExternalStore(
        subscribeSessionGate,
        getSessionGateSnapshot,
        getSessionGateSnapshot,
    );
    // "宿主从未接入 ARA"与"会话已结束"需要完全不同的下一步：前者关窗重开无用
    //（宿主根本不走 ARA 路径），后者才有用。混用一句会把用户引向必然失败的动作。
    const message =
        terminal?.code === ERROR_CODE.ARA_NOT_BOUND
            ? t("plugin_ara_not_bound")
            : t("plugin_apply_session_ended");

    return (
        <Flex
            role="alert"
            direction="column"
            align="center"
            justify="center"
            gap="3"
            className="h-screen w-screen bg-qt-window px-qt-6 text-center select-none"
        >
            <span className="hs-type-display font-bold">{t("session_unavailable_title")}</span>
            <span className="hs-type-body max-w-prose" style={{ color: "var(--qt-text-muted)" }}>
                {message}
            </span>
            <AppButton
                intent="primary"
                disabled={opening}
                onClick={() => {
                    setOpening(true);
                    // 打不开日志目录不该把这个页面变成另一个错误现场：失败就静默，
                    // 用户仍然可以从「帮助 → 打开日志目录」走同一条路。
                    void openLogFolder()
                        .catch(() => undefined)
                        .finally(() => setOpening(false));
                }}
            >
                {t("session_unavailable_open_log")}
            </AppButton>
        </Flex>
    );
}
