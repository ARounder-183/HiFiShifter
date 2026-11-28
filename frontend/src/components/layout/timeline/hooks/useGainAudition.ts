import React from "react";

import {
    GainAuditionController,
    type GainAudition,
} from "../../../../features/session/gainAudition";
import { timelineApi } from "../../../../services/api";
import { gainLayersAreHostOwned } from "../../../../services/hostCapabilities";

let shared: GainAuditionController | null = null;

function controller(): GainAuditionController {
    if (shared === null) {
        shared = new GainAuditionController(
            async (payload) => {
                // 宿主拥有增益层时（ARA），试听只会把宿主已经施加的增益再乘一遍。
                if (gainLayersAreHostOwned()) return;
                await timelineApi.setGainPreview(payload);
            },
            async (gestureId) => {
                if (gainLayersAreHostOwned()) return;
                await timelineApi.clearGainPreview({ gestureId });
            },
        );
    }
    return shared;
}

/**
 * 整个 WebView 共用一个试听控制器：同一时刻只可能有一个指针手势，共享它可
 * 避免“轨道滚轮防抖提交”与“Clip 旋钮拖动”各自持有一个 gestureId 互相覆盖。
 */
export function useGainAudition(): GainAudition {
    return React.useMemo(() => controller(), []);
}
