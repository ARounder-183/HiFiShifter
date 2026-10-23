/**
 * 参数编辑器的**门禁与可见性**：哪些参数此刻可以编辑、哪些曲线该被画出来。
 *
 * 【为什么是独立的一个 hook】这一段原本夹在工具栏溢出分级与视口测量之间（模型的
 * 1003-1159 行），但它描述的是同一个判断链：轨道级的 Compose 开关与分离开关
 * 决定"参数是否生效"，工具栏横向放不下决定"药丸是否让位"，而用户的"眼睛"开关
 * 决定"曲线是否绘制"。三者曾被混成一件（把让位当成可见），导致画布上凭空多出
 * 一条用户从未打开过的副参数曲线——文件里那段说明就是在钉住这条界线。
 *
 * 【所有权不变】所有状态仍属于面板（`editParam` / `secondaryParamVisible` /
 * `hoveredReferenceRootTrackId` 的声明位置、两个 `useEffect` 的注册顺序全部留在
 * 原相对位置），这里只是把语句搬进了子 hook，因此 hook 顺序与依赖数组与拆分前
 * 逐条相同。
 */
import { useCallback, useEffect, useMemo, type Dispatch, type SetStateAction } from "react";
import {
    persistUiSettings,
    setEditParam,
    setVisibleReferenceRootTrackIds,
} from "../../../../features/session/sessionSlice";
import {
    findBlockedEditParam,
    gatedParamHideOrder,
    paramNeedingVisibilityOnGate,
} from "../../../../features/pitch/separationGate";
import type { ProcessorParamDescriptor } from "../../../../types/api";
import { isChildPitchOffsetParam } from "../childPitchOffsetParams";
import {
    cleanupVisibleReferenceRootTrackIds,
    listReferenceRootTracks,
} from "../referenceRootTracks";
import { getVisibleSecondaryParamIds } from "../secondaryOverlaySelection";
import type { ParamName } from "../types";
import type { ModelDeps } from "./modelDeps";
import { sameStringArray } from "./panelHelpers";

type PianoRollParamGatesInputs = {
    /** Redux dispatch。 */
    dispatch: ModelDeps["dispatch"];
    /** 当前编辑的参数。 */
    editParam: ModelDeps["editParam"];
    /** 会话切片（轨道列表、参考轨道可见集合都在这里）。 */
    s: ModelDeps["s"];
    /** 根轨道与其 id。 */
    rootTrackId: ModelDeps["rootTrackId"];
    rootTrack: ModelDeps["s"]["tracks"][number] | null;
    /** 处理器参数描述符（顺序即药丸顺序）。 */
    orderedProcessorParams: ProcessorParamDescriptor[];
    processorParams: ProcessorParamDescriptor[];
    /** 分离（张力 / 气声）门禁是否开启。 */
    separationEnabled: boolean;
    /** 工具栏当前溢出分级（决定几个药丸让位）。 */
    toolbarTier: number;
    /** 用户"眼睛"开关。 */
    secondaryParamVisible: Partial<Record<string, boolean>>;
    setSecondaryParamVisible: Dispatch<SetStateAction<Partial<Record<string, boolean>>>>;
    /** 参考轨道组上悬停的轨道。 */
    hoveredReferenceRootTrackId: string | null;
    setHoveredReferenceRootTrackId: Dispatch<SetStateAction<string | null>>;
    /** i18n。 */
    t: ModelDeps["t"];
};

/**
 * 门禁与可见性。
 *
 * 调用点必须固定在模型里原来的位置：两个 `useEffect`（把 `editParam` 移开、把
 * 失效的悬停轨道清空）在这里注册，它们的相对顺序是行为的一部分。
 */
export function usePianoRollParamGates({
    dispatch,
    editParam,
    s,
    rootTrackId,
    rootTrack,
    orderedProcessorParams,
    processorParams,
    separationEnabled,
    toolbarTier,
    secondaryParamVisible,
    setSecondaryParamVisible,
    hoveredReferenceRootTrackId,
    setHoveredReferenceRootTrackId,
    t,
}: PianoRollParamGatesInputs) {
    // 被门禁的药丸逐个让位（第 3 / 4 级各让一个），顺序**先张力后气声**：
    // 气声药丸左侧挂着分离开关，先让气声会让开关独自悬空。
    // 组首的药丸让位时开关保留（见下方组首分支）。
    //
    // 【让位就是让位：曲线跟着眼睛走，不再被强行点亮】这里曾额外在"渲染口径"上把
    // 被让位的参数一律视为可见（`effectiveSecondaryParamVisible`），理由是"关闭分离时
    // 曲线保持可见"。但那把两件不同的事混成了一件：**门禁**（开关关掉 → 参数置灰）
    // 与**横向放不下**（药丸被让位）。后者会把用户从未打开过的参数曲线画出来 ——
    // 表现为参数编辑器里凭空多出一条平直的副参数曲线（张力未编辑时默认值恒定，
    // 于是就是一条横贯左右的直线），而用户没有任何开关能关掉它：药丸已经让位了。
    // 现在只有真实的眼睛状态决定是否绘制；门禁那侧的"保持可见"由下面的
    // `paramNeedingVisibilityOnGate` 负责（关掉开关时若正在编辑该参数才点亮它）。
    const gatedPillHideOrder = useMemo(
        () => gatedParamHideOrder(orderedProcessorParams.map((p) => p.id)),
        [orderedProcessorParams],
    );
    const gatedPillsHidden = separationEnabled
        ? 0
        : Math.max(0, Math.min(toolbarTier - 2, gatedPillHideOrder.length));

    // 所属轨道组的 Compose 开关。它与分离开关是**两道独立门禁**，但都决定
    // 轨道级"合成"参数（共振峰/气声/张力）是否生效 —— Compose 关闭时必须
    // 完全静默（需求：听到未经任何改动的原音频）。
    // 取不到根轨时按"开启"处理：宁可不过度置灰，也不要误锁住用户。
    const composeEnabled = rootTrack?.composeEnabled ?? true;

    // 开关刚被关闭时，若当前正在编辑被门禁的参数，把 `editParam` 移开。
    //
    // 【为什么必须回退而不是仅仅置灰】曲线编辑的唯一闸门是 `editParam`
    // （画布绘制路径经 `usePianoRollInteractions` 读同一状态）。若不移开，
    // 用户仍能在画布上继续画该参数的曲线 —— 与"不可编辑"矛盾。
    //
    // 【为什么同时要把"眼睛"打开】曲线只在两种情况下被绘制：它是当前
    // `editParam`，或它的 `secondaryParamVisible` 为真。而 `editParam` 马上
    // 就要被移走，眼睛默认又是关的 —— 不补这一步，用户在编辑张力时关掉开关，
    // 曲线会**直接消失**，与"置灰但保持可见、只是不可编辑"的要求相反。
    //
    // 该 effect 是**收敛**的：回退到 `pitch` 后判据不再成立，不会反复 dispatch；
    // 眼睛置位也做了幂等判断，不会产生新的 state 引用。
    useEffect(() => {
        // 两道门禁任一命中都要把 editParam 移开：Compose 关闭时效果参数同样
        // 不可编辑，若不移开，画布上的绘制仍会落到它身上。
        const blocked =
            findBlockedEditParam(editParam, separationEnabled, "pitch") ??
            findBlockedEditParam(editParam, composeEnabled, "pitch");
        if (!blocked) return;
        const leaving =
            paramNeedingVisibilityOnGate(editParam, separationEnabled, "pitch") ??
            paramNeedingVisibilityOnGate(editParam, composeEnabled, "pitch");
        if (leaving) {
            setSecondaryParamVisible((prev) =>
                prev[leaving] ? prev : { ...prev, [leaving]: true },
            );
        }
        dispatch(setEditParam(blocked as typeof editParam));
    }, [editParam, separationEnabled, composeEnabled, dispatch]);

    // 收集轨道组内所有 trackId（root + 递归所有子轨道）
    const groupTrackIds = useMemo(() => {
        const ids = new Set<string>();
        if (!rootTrackId) return ids;
        ids.add(rootTrackId);
        const frontier = [rootTrackId];
        let idx = 0;
        while (idx < frontier.length) {
            const cur = frontier[idx++];
            const track = s.tracks.find((t) => t.id === cur);
            if (track?.childTrackIds) {
                for (const childId of track.childTrackIds) {
                    if (!ids.has(childId)) {
                        ids.add(childId);
                        frontier.push(childId);
                    }
                }
            }
        }
        return ids;
    }, [rootTrackId, s.tracks]);

    const referenceRootTrackOptions = useMemo(
        () =>
            listReferenceRootTracks({
                tracks: s.tracks,
                currentRootTrackId: rootTrackId,
            }),
        [rootTrackId, s.tracks],
    );

    const visibleReferenceRootTrackIds = useMemo(
        () =>
            cleanupVisibleReferenceRootTrackIds({
                tracks: s.tracks,
                currentRootTrackId: rootTrackId,
                visibleReferenceRootTrackIds: s.visibleReferenceRootTrackIds,
            }),
        [rootTrackId, s.tracks, s.visibleReferenceRootTrackIds],
    );

    useEffect(() => {
        if (sameStringArray(visibleReferenceRootTrackIds, s.visibleReferenceRootTrackIds)) {
            return;
        }
        dispatch(setVisibleReferenceRootTrackIds(visibleReferenceRootTrackIds));
        void dispatch(persistUiSettings());
    }, [dispatch, s.visibleReferenceRootTrackIds, visibleReferenceRootTrackIds]);

    useEffect(() => {
        if (
            hoveredReferenceRootTrackId &&
            !visibleReferenceRootTrackIds.includes(hoveredReferenceRootTrackId)
        ) {
            setHoveredReferenceRootTrackId(null);
        }
    }, [hoveredReferenceRootTrackId, visibleReferenceRootTrackIds]);

    const pitchHardDisableReason = useMemo(() => {
        if (editParam !== "pitch") return null;
        if (!rootTrack) return null;
        if (!rootTrack.composeEnabled) return t("pitch_requires_compose");
        if (rootTrack.pitchAnalysisAlgo === "none") return t("pitch_requires_algo");
        return null;
    }, [editParam, rootTrack, t]);

    const childPitchHardDisableReason = useMemo(() => {
        if (!isChildPitchOffsetParam(editParam)) return null;
        if (!rootTrack) return null;
        if (!rootTrack.composeEnabled) return t("pitch_requires_compose");
        return null;
    }, [editParam, rootTrack, t]);

    const pitchEnabled =
        editParam === "pitch"
            ? pitchHardDisableReason == null
            : isChildPitchOffsetParam(editParam)
              ? childPitchHardDisableReason == null
              : true;

    const visibleSecondaryParamIds = useMemo(() => {
        // 依赖 state 版 processorParams 而非 ref：切换算法后 ref 已更新，但
        // 若 editParam/开关不变，memo 不重算会让副参数列表继续持有旧算法
        // 的参数 id（持续对已不存在的参数发起取数，眼睛开关与叠加层不一致）。
        return getVisibleSecondaryParamIds({
            editParam,
            processorParamIds: processorParams.map((p) => p.id as ParamName),
            // 只认真实眼睛状态：曲线画不画由用户的开关决定，横向空间不足而让位的
            // 药丸不再把自己的曲线点亮（见 `gatedPillHideOrder` 处的说明）。
            secondaryParamVisible,
        });
    }, [editParam, secondaryParamVisible, processorParams]);

    const updateVisibleReferenceRootTrackIds = useCallback(
        (nextTrackIds: string[]) => {
            dispatch(setVisibleReferenceRootTrackIds(nextTrackIds));
            void dispatch(persistUiSettings());
        },
        [dispatch],
    );
    return {
        gatedPillHideOrder,
        gatedPillsHidden,
        composeEnabled,
        groupTrackIds,
        referenceRootTrackOptions,
        visibleReferenceRootTrackIds,
        pitchHardDisableReason,
        childPitchHardDisableReason,
        pitchEnabled,
        visibleSecondaryParamIds,
        updateVisibleReferenceRootTrackIds,
    };
}
