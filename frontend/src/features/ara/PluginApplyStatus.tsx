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
    /*
     * 【色调也只跟着 `audible_state` 走】旧判据里的 `state?.pending` 会让状态片在
     * **每一次**播放开始都亮成"进行中"，而那一刻后台跑的只是素材物化 —— 用户看到的
     * 是一盏与自己的操作无关的灯。`waiting` 同理：它不是"在应用你的编辑"。
     */
    // 【旁通用中性色】它不是"在忙"，也不是"出错"：宿主把这个 FX 从信号链里摘了，
    // 插件的行为（不渲染）是正确的。用强调色会让用户以为它在算东西。
    const tone = terminal
        ? "danger"
        : error
          ? "danger"
          : reloading
            ? "accent"
            : state?.host_bypass === "bypassed"
              ? "neutral"
              : state?.audible_state === "preparing" ||
                  state?.audible_state === "stale" ||
                  state?.audible_state === "awaiting-analysis"
                ? "accent"
                : state?.ready
                  ? "success"
                  : "neutral";
    /*
     * 【只读 `audible_state`】它的语义是「你听到的是不是当前编辑」，与状态栏要回答的
     * 问题**同一个**。逐个读 `ready` / `analysis_pending` / `rendering_active` /
     * `pending` 是把"后台在忙"当成"你听到的是旧的"—— 那正是「正在自动应用…」的依据，
     * 而它覆盖的其实是素材物化与后台刷新，用户当时什么都没改。
     *
     * 【为什么还留 `ready` 那一档】后端没给 `audible_state`（旧版本后端 / 首帧）时，
     * 「有没有可服务的输出」仍然只有 `ready` 能回答。它是兜底，不是主判据。
     */
    const audibleState = state?.audible_state;
    /*
     * 【百分比不在这里】它只由左侧 `AppStatusProgressChips` 显示。
     *
     * 旧实现在这里拼「正在合成 42%」，条件是 `audible_state === "stale"` —— 而那一档
     * 在 ARA 下**恒不成立**（一旦发布过快照，`has_serviceable_snapshot()` 就永远为真，
     * 于是 `stale` 永远为假）。也就是说这条百分比是死代码：用户永远看不到它。
     * 两处各拼一份、判据还分叉，只会得到"左边 40%、右边 100%"。
     *
     * 现在：这一片只回答「你听到的是不是当前编辑」与「宿主是否把本插件旁通了」。
     */
    /*
     * 【旁通优先于「已应用」】旁通时宿主根本不来取输出，`audible_state` 会落在
     * `current`（后端把 `applied` 收口到当前代次），于是这一片会说「已应用」——
     * 而用户刚做的编辑**一次都没被渲染**。说「已旁通 · 不渲染」才是实话：
     * 这不是故障，是宿主的选择，用户解除旁通后会自动补上。
     */
    const bypassed = state?.host_bypass === "bypassed";
    const label = terminal
        ? terminalLabel
        : bypassed
          ? t("plugin_apply_bypassed")
          : audibleState === "preparing"
            ? t("plugin_apply_waiting_host")
            : audibleState === "stale"
              ? t("plugin_apply_rendering")
              : audibleState === "awaiting-analysis"
                ? t("plugin_apply_analyzing")
                : audibleState === "current"
                  ? t("plugin_apply_applied")
                  : !state?.ready
                    ? t("plugin_apply_waiting_host")
                    : t("plugin_apply_applied");
    /*
     * 修订数与长说明都进 tooltip：它们在状态栏里没有位置，但"为什么还没应用"
     * 恰恰要靠它们判断 —— 塞进片里会把状态栏撑成一条跑道。
     *
     * 【为什么用 \n 而不是原生 title】气泡是 `white-space: pre-wrap`
     * （`index.css:1594`），换行会如实渲染成两行；而原生 `title` 会把 `\n`
     * 折成一个空格，且样式与明暗主题都不受本项目控制（`dockTooltips.ts:58-61`
     * 对"不要改用原生 title"有同样的结论）。此前这里正是把两行 `join("\n")`
     * 塞给 `title` 的 —— 两行挤成一行，且用的是原生气泡。
     */
    const tooltip = state
        ? [
              tVars("plugin_apply_generations", {
                  edits: state.generation,
                  audio: state.applied_generation,
              }),
              // 分析在途时先说清"这一步本来就这么慢"，否则用户会以为它卡死了。
              state.analysis_pending ? t("plugin_apply_analyzing_hint") : null,
              t("plugin_apply_hint"),
          ]
              .filter((line): line is string => line !== null)
              .join("\n")
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
            <AppStatusChip tone={tone} tooltip={tooltip}>
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
                    //
                    // 【为什么不再另挂一个提示】这行文字本身就是完整的
                    // "尚未应用：{原因}"，再给它加 tooltip 只会重复同样的话 ——
                    // 此前它挂的原生 `title` 与可见文字逐字相同，纯噪声。
                    // 需要更多细节的用户可以看日志（后端已限流记录同一条错误）。
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
