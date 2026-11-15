import { useEffect, useState } from "react";
import { Flex, Text } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import { mcpApi, type McpStatus } from "../../services/api/mcp";
import { AppButton, AppDialog } from "../../ui";

export function McpDialog({
    open,
    onOpenChange,
}: {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}) {
    const { t } = useI18n();
    const [status, setStatus] = useState<McpStatus | null>(null);
    const [busy, setBusy] = useState(false);
    const [message, setMessage] = useState("");
    useEffect(() => {
        if (!open) return;
        let disposed = false;
        void mcpApi
            .status()
            .then((value) => {
                if (!disposed) setStatus(value);
            })
            .catch((error: unknown) => {
                if (!disposed) setMessage(String(error));
            });
        return () => {
            disposed = true;
        };
    }, [open]);
    async function toggle() {
        setBusy(true);
        setMessage("");
        try {
            const next = await mcpApi.setEnabled(!status?.enabled);
            setStatus(next);
            if (!next.ok) setMessage(next.message ?? "MCP service failed");
        } catch (error) {
            setMessage(String(error));
        } finally {
            setBusy(false);
        }
    }
    async function copy() {
        try {
            await navigator.clipboard.writeText(JSON.stringify(status?.client_config, null, 2));
            setMessage(t("mcp_copied"));
        } catch (error) {
            setMessage(String(error));
        }
    }
    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={t("mcp_title")}
            description={t("mcp_description")}
            size="md"
        >
            <Flex direction="column" gap="3">
                <Text size="2">{status?.enabled ? t("mcp_enabled") : t("mcp_disabled")}</Text>
                {status?.endpoint ? <Text size="2">{status.endpoint}</Text> : null}
                <Flex gap="2">
                    <AppButton disabled={busy || !status} onClick={() => void toggle()}>
                        {status?.enabled ? t("mcp_disable") : t("mcp_enable")}
                    </AppButton>
                    <AppButton disabled={!status?.enabled || busy} onClick={() => void copy()}>
                        {t("mcp_copy_config")}
                    </AppButton>
                </Flex>
                {status?.client_config ? (
                    <>
                        <Text size="2">{t("mcp_config_label")}</Text>
                        <textarea
                            readOnly
                            value={JSON.stringify(status.client_config, null, 2)}
                            rows={9}
                            aria-label={t("mcp_config_label")}
                            className="w-full rounded bg-qt-base p-2 font-mono text-xs select-text"
                        />
                    </>
                ) : null}
                {message ? (
                    <Text size="2" role="status">
                        {message}
                    </Text>
                ) : null}
            </Flex>
        </AppDialog>
    );
}
