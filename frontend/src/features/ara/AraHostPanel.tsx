/**
 * ARA 宿主会话面板（独立 App）。
 *
 * 【为什么是面板而不是常驻横条】连接宿主是一次长活会话：连上之后要反复
 * 提交 / 刷新 / 断开，所以它需要一个稳定的容器，而不是"点一次就没了"的
 * 对话框。但它在独立 App 里又是低频入口 —— 常驻一整行横条会一直占着工作区，
 * 而多数用户根本不连宿主。因此做成**默认关闭的浮出面板**：入口在「视图」菜单，
 * 需要时打开，不用时不占任何空间。
 *
 * 【为什么状态用 `AppStatusChip` 而不是自绘文字】状态读数与修订号是同一类
 * 信息，全应用的状态片已经有一套语义色（见 `ui/StatusChip.tsx`）；自绘一套
 * 会让它在暗色主题下与其余状态片不一致。
 */
import { useEffect, useState } from "react";
import { Flex } from "@radix-ui/themes";

import { useI18n } from "../../i18n/I18nProvider";
import { AppButton, AppConfirmDialog, AppField, AppForm, AppSelect, AppStatusChip } from "../../ui";
import { araApi, araError, type AraInstance, type AraResult } from "./araApi";

/** 连接成功后展示的状态词。都是词表里的既有键，不拼字符串。 */
type AraStatusKey =
    | "ara_status_connected"
    | "ara_status_refreshed"
    | "ara_status_submitted"
    | "ara_status_disconnected";

export function AraHostPanel({
    dirty,
    onTimelineChanged,
}: {
    dirty: boolean;
    onTimelineChanged: () => Promise<unknown>;
}) {
    const { t, tVars } = useI18n();
    const [instances, setInstances] = useState<AraInstance[]>([]);
    const [selected, setSelected] = useState("");
    const [session, setSession] = useState<AraResult | null>(null);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState("");
    const [status, setStatus] = useState<AraStatusKey | null>(null);
    const [replacement, setReplacement] = useState<"connect" | "refresh" | null>(null);

    async function run(action: () => Promise<void>) {
        setBusy(true);
        setError("");
        setStatus(null);
        try {
            await action();
        } catch (err) {
            setError(araError(err));
        } finally {
            setBusy(false);
        }
    }

    /** 重扫实例；当前选中的实例消失时回落到第一个（或清空）。 */
    async function list() {
        const found = await araApi.list();
        setInstances(found);
        setSelected((value) =>
            found.some((item) => item.instance_id === value)
                ? value
                : (found[0]?.instance_id ?? ""),
        );
    }

    useEffect(() => {
        void run(list);
    }, []);

    /**
     * 连接 / 刷新宿主。
     *
     * `force` 之前先确认：工程有未保存修改时，宿主快照会覆盖本地编辑。脏工程
     * 也可能只被**后端**发现（前端状态过期），因此后端返回 `dirty_project:` 时
     * 同样要回到确认态，而不是把错误直接抛给用户。
     */
    async function operate(kind: "connect" | "refresh", force = false) {
        if (dirty && !force) {
            setReplacement(kind);
            return;
        }
        setReplacement(null);
        await run(async () => {
            let result: AraResult;
            try {
                result =
                    kind === "connect"
                        ? await araApi.connect(selected, force)
                        : await araApi.refresh(force);
                if (!result.ok) {
                    throw new Error(result.error ?? t("ara_error_connect"));
                }
            } catch (err) {
                if (!force && araError(err).startsWith("dirty_project:")) setReplacement(kind);
                throw err;
            }
            setSession(result);
            await onTimelineChanged();
            setStatus(kind === "connect" ? "ara_status_connected" : "ara_status_refreshed");
        });
    }

    async function submit() {
        await run(async () => {
            const result = await araApi.submit();
            if (!result.ok) throw new Error(result.error ?? t("ara_error_submit"));
            setSession(result);
            await onTimelineChanged();
            setStatus("ara_status_submitted");
        });
    }

    async function disconnect() {
        await run(async () => {
            const result = await araApi.disconnect();
            if (!result.ok) throw new Error(result.error ?? t("ara_error_disconnect"));
            setSession(null);
            setStatus("ara_status_disconnected");
        });
    }

    const instanceOptions = instances.map((instance) => ({
        value: instance.instance_id,
        label: `${instance.name} (${instance.pid})`,
    }));

    return (
        <div className="hs-scroll-gutter-flush custom-scrollbar flex h-full flex-col gap-qt-4 overflow-y-auto p-qt-5">
            <AppForm labelWidth="auto">
                <AppField label={t("ara_instance_label")}>
                    {/*
                     * 没有实例时**不渲染下拉**：Radix 的 `Select.Item` 不接受空串值
                     * （空串被它保留给"未选择"），用哨兵值去凑只会让取值多一个
                     * 需要处处过滤的伪状态。直接给一句说明更诚实，也让"连接"按钮
                     * 的禁用原因一眼可见。
                     */}
                    {instances.length ? (
                        <AppSelect
                            value={selected}
                            onValueChange={setSelected}
                            options={instanceOptions}
                            disabled={busy || !!session}
                            density="compact"
                            minWidth={150}
                            ariaLabel={t("ara_instance_label")}
                        />
                    ) : (
                        <span className="hs-type-label" style={{ color: "var(--qt-text-muted)" }}>
                            {t("ara_no_instances")}
                        </span>
                    )}
                </AppField>
            </AppForm>

            <Flex align="center" gap="2" wrap="wrap">
                <AppButton size="sm" disabled={busy} onClick={() => void run(list)}>
                    {t("ara_refresh_instances")}
                </AppButton>
                <AppButton
                    size="sm"
                    disabled={busy || !selected || !!session}
                    onClick={() => void operate("connect")}
                >
                    {t("ara_connect")}
                </AppButton>
                <AppButton
                    size="sm"
                    intent="primary"
                    disabled={busy || !session}
                    onClick={() => void submit()}
                >
                    {t("ara_submit")}
                </AppButton>
                <AppButton
                    size="sm"
                    disabled={busy || !session}
                    onClick={() => void operate("refresh")}
                >
                    {t("ara_refresh_host")}
                </AppButton>
                <AppButton size="sm" disabled={busy || !session} onClick={() => void disconnect()}>
                    {t("ara_disconnect")}
                </AppButton>
            </Flex>

            <Flex align="center" gap="2" wrap="wrap">
                <AppStatusChip
                    tone={error ? "danger" : busy ? "accent" : session ? "success" : "neutral"}
                    title={
                        session
                            ? tVars("ara_revisions", {
                                  revision: session.revision ?? 0,
                                  model: session.model_revision ?? 0,
                              })
                            : undefined
                    }
                >
                    {busy ? t("ara_busy") : status ? t(status) : t("ara_status_idle")}
                </AppStatusChip>
                <span className="hs-type-caption">{t("ara_reverse_unsupported")}</span>
            </Flex>

            {error && (
                <span
                    role="alert"
                    className="hs-type-caption"
                    style={{ color: "var(--qt-danger-text)" }}
                >
                    {error}
                </span>
            )}

            <AppConfirmDialog
                open={replacement !== null}
                onOpenChange={(open) => {
                    if (!open) setReplacement(null);
                }}
                title={t("ara_replace_dirty_confirm")}
                message={t("ara_replace_dirty_message")}
                confirmLabel={t("ara_replace_dirty_confirm")}
                cancelLabel={t("cancel")}
                intent="danger"
                tone="danger"
                onConfirm={() => {
                    if (replacement) void operate(replacement, true);
                }}
            />
        </div>
    );
}
