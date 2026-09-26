import { describe, expect, it, vi } from "vitest";

import { createLiveEditFlag } from "./liveEditFlag";

describe("createLiveEditFlag", () => {
    it("starts idle and tracks writes", () => {
        const flag = createLiveEditFlag(() => {});
        expect(flag.current).toBe(false);
        flag.current = true;
        expect(flag.current).toBe(true);
        flag.current = false;
        expect(flag.current).toBe(false);
    });

    /** 补取只在 `true → false` 的边沿触发一次 —— 这正是收口十余处收尾点的依据。 */
    it("fires onEnd only on the true-to-false edge", () => {
        const onEnd = vi.fn();
        const flag = createLiveEditFlag(onEnd);

        flag.current = false; // 已是 false：无边沿
        expect(onEnd).not.toHaveBeenCalled();

        flag.current = true;
        expect(onEnd).not.toHaveBeenCalled();

        flag.current = true; // 保持 true：无边沿
        expect(onEnd).not.toHaveBeenCalled();

        flag.current = false;
        expect(onEnd).toHaveBeenCalledTimes(1);

        flag.current = false; // 重复置 false：不得重复补取
        expect(onEnd).toHaveBeenCalledTimes(1);
    });

    /**
     * 边沿回调期间标志必须已经是 `false`：补取链路上的代码若回读它，应当看到
     * "已结束"，而不是即将过期的旧值。
     */
    it("reports the cleared value while onEnd runs", () => {
        let seenDuringCallback: boolean | null = null;
        const flag = createLiveEditFlag(() => {
            seenDuringCallback = flag.current;
        });

        flag.current = true;
        flag.current = false;

        expect(seenDuringCallback).toBe(false);
    });

    /** 连续多笔：每一笔结束都要补一次，不能被"只触发一次"吃掉。 */
    it("fires once per stroke across repeated strokes", () => {
        const onEnd = vi.fn();
        const flag = createLiveEditFlag(onEnd);

        for (let stroke = 0; stroke < 3; stroke += 1) {
            flag.current = true;
            flag.current = false;
        }

        expect(onEnd).toHaveBeenCalledTimes(3);
    });
});
