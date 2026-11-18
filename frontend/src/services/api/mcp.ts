import { invoke } from "../invoke";
import type { TimelineResult } from "../../types/api";

export interface McpStatus {
    ok: boolean;
    enabled: boolean;
    port?: number;
    endpoint?: string;
    instance_id?: string;
    client_config?: Record<string, unknown>;
    message?: string;
}

export const mcpApi = {
    status: () => invoke<McpStatus>("get_mcp_status"),
    setEnabled: (enabled: boolean) => invoke<McpStatus>("set_mcp_enabled", enabled),
    view: () => invoke<{ ok: boolean; revision: string; timeline: TimelineResult }>("get_mcp_view"),
    acknowledgeView: (revision: string) =>
        invoke<{ ok: boolean }>("acknowledge_mcp_view", revision),
    replyView: (requestId: string, result: unknown) =>
        invoke<{ ok: boolean }>("reply_mcp_view", requestId, result),
};
