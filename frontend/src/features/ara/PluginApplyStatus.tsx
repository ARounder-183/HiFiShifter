/**
 * 状态栏里的插件宿主「自动应用」状态片。
 *
 * 【为什么从整条横条改成状态片】这段内容的本质是**状态读数**（已应用 / 等待
 * 宿主音频 / 正在自动应用），不是操作区。它此前占着 `ActionBar` 与工作区之间的
 * 一整行 —— 在插件那个小窗口里，一整行高度是很贵的。状态读数的既有归属就是
 * 状态栏，而状态栏右侧本来留着一个空槽（`justify="between"`）。
 *
 * 【为什么单独成一个组件】见 `pluginApplyStore.ts` 的说明：订阅装在状态栏所在的
 * `AppInner` 上，250 ms 一次的轮询会把整棵应用树拖着重渲染。重渲染必须被限制在
 * 这一个片 + 一个按钮上。
 */
import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { Flex } from "@radix-ui/themes";

import { useI18n } from "../../i18n/I18nProvider";
import { ERROR_CODE } from "../../services/sessionGate";
import { AppButton, AppConfirmDialog, AppStatusChip } from "../../ui";
import {
    getPluginApplySnapshot,
    reloadPluginHost,
    startPluginApplyPolling,
    subscribePluginApply,
} from "./pluginApplyStore";

export function PluginApplyStatus({
    onTimelineChanged,
}: {
    onTimelineChanged: () => Promise<unknown>;
}) {
    const { t, tVars } = useI18n();
    const { state, failure, reloading, sessionEnded, sessionCode } = useSyncExternalStore(
        subscribePluginApply,
        getPluginApplySnapshot,
        getPluginApplySnapshot,
    );
    const [confirm, setConfirm] = useState(false);
    /*
     * 轮询要一个稳定的回调：把最新的 `onTimelineChanged` 放 ref 里，轮询本身
     * 只在挂载时启动一次。写入发生在 **effect** 里而不是渲染期 —— 渲染期写 ref
     * 会被 React 的引用规则拒绝，且本次提交的值本来就要等副作用跑完才可读
     * （与 `ui/useFrameCommit.ts` 处理提交回调的方式一致）。
     */
    const refreshRef = useRef(onTimelineChanged);
    useEffect(() => {
        refreshRef.current = onTimelineChanged;
    });
    useEffect(() => startPluginApplyPolling(() => refreshRef.current()), []);

    const refreshTimeline = () => refreshRef.current();
    const error = failure || state?.error || "";
    // 【为什么终态要单独一条】会话结束不是"尚未应用"的一种：重试、点"重新载入"
    // 都不会成功。把后端原始串直接摊给用户（`plugin_apply_error`）会让人以为
    // 再等等就好。这里给出**可操作的**说明，并且不再显示重载按钮。
    //
    // 【为什么还要按码再分一层】"编辑文档已关闭"与"宿主从未把本插件接入 ARA"是两种
    // 终态，**下一步完全不同**：前者关窗重开有用，后者关窗重开无用（宿主根本不走 ARA
    // 路径，重开会走完全相同的流程）。共用一句会把用户引向一个必然失败的补救动作 ——
    // 这正是 Cubase 报障里那条提示的问题。按协议码分支，不解析错误文本。
    const terminal = sessionEnded;
    const terminalLabel =
        sessionCode === ERROR_CODE.ARA_NOT_BOUND
            ? t("plugin_ara_not_bound")
            : t("plugin_apply_session_ended");
    const tone = terminal
        ? "danger"
        : error
          ? "danger"
          : reloading || state?.pending
            ? "accent"
            : state?.ready
              ? "success"
              : "neutral";
    const label = terminal
        ? terminalLabel
        : !state?.ready
          ? t("plugin_apply_waiting_host")
          : state.pending
            ? t("plugin_apply_pending")
            : t("plugin_apply_applied");
    /*
     * 修订数与长说明都进 `title`：它们在状态栏里没有位置，但"为什么还没应用"
     * 恰恰要靠它们判断 —— 塞进片里会把状态栏撑成一条跑道。
     */
    const title = state
        ? [
              tVars("plugin_apply_generations", {
                  edits: state.generation,
                  audio: state.applied_generation,
              }),
              t("plugin_apply_hint"),
          ].join("\n")
        : undefined;

    function requestReload() {
        if (state?.pending) {
            setConfirm(true);
            return;
        }
        void reloadPluginHost(false, refreshTimeline);
    }

    return (
        <Flex align="center" gap="1" className="min-w-0">
            <AppStatusChip tone={tone} title={title}>
                {label}
            </AppStatusChip>
            {error && !terminal ? (
                <span
                    role="alert"
                    className="hs-type-label min-w-0 truncate"
                    style={{ color: "var(--qt-danger-text)" }}
                    // 【为什么套 `plugin_apply_error`】原先直接渲染后端原始串，
                    // 用户看到的是一句没有上下文的外文/技术文本；而词表里
                    // `plugin_apply_error`（"尚未应用：{error}"）就是为此准备的，
                    // 却一直零引用（一处 i18n 绕过 + 一个死键）。错误内容本身来自
                    // 后端，保持原样；**前缀**走词表。
                    title={tVars("plugin_apply_error", { error })}
                >
                    {tVars("plugin_apply_error", { error })}
                </span>
            ) : null}
            {state?.ready && !terminal ? (
                <AppButton size="sm" disabled={reloading} onClick={requestReload}>
                    {t("plugin_apply_reload_host")}
                </AppButton>
            ) : null}
            <AppConfirmDialog
                open={confirm}
                onOpenChange={setConfirm}
                title={t("plugin_apply_reload_host")}
                message={t("plugin_apply_reload_message")}
                confirmLabel={t("plugin_apply_reload_confirm")}
                cancelLabel={t("cancel")}
                intent="danger"
                tone="danger"
                onConfirm={() => {
                    setConfirm(false);
                    void reloadPluginHost(true, refreshTimeline);
                }}
            />
        </Flex>
    );
}
