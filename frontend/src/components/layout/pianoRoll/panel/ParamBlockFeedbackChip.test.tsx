// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test } from "vitest";
import { ParamBlockFeedbackChip } from "./ParamBlockFeedbackChip";
import { publishParamBlockFeedback } from "../paramBlockFeedback";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
    publishParamBlockFeedback(null);
});

afterEach(() => {
    act(() => root.unmount());
    document.body.innerHTML = "";
    publishParamBlockFeedback(null);
});

test("renders danger feedback with tooltip", async () => {
    await act(async () => {
        root.render(<ParamBlockFeedbackChip />);
    });
    await act(async () => {
        publishParamBlockFeedback({
            text: "参数块编辑被拒绝",
            tone: "danger",
            tooltip: "backend detail",
            holdMs: 0,
        });
    });
    expect(document.body.textContent).toContain("参数块编辑被拒绝");
    expect(document.querySelector("[data-tooltip='backend detail']")).not.toBeNull();
});
