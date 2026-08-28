import { test } from "vitest";

import { isValidElement } from "react";
import { buildReferenceRootTrackTriggerElement } from "./referenceRootTrackTrigger.ts";

test("components/layout/pianoRoll/referenceRootTrackTrigger.test.ts scripted checks", async () => {
    function assert(condition: unknown, label: string): void {
        if (!condition) {
            throw new Error(label);
        }
    }

    const trigger = buildReferenceRootTrackTriggerElement("参考轨道组 (2)");
    const triggerProps = trigger.props as { children?: unknown };

    assert(isValidElement(trigger), "reference root track trigger should be a React element");
    assert(trigger.type === "span", "reference root track trigger should use a span wrapper");
    assert(
        triggerProps.children === "参考轨道组 (2)",
        "reference root track trigger should keep label text",
    );

});
