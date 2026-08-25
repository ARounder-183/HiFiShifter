/**
 * 吸附竖线高亮总线测试（node tsx 直接运行）。
 */
import {
    SNAP_HIGHLIGHT_GROUP,
    buildCandidateHighlightEntry,
    buildLoopBoundaryHighlightEntry,
    clearSnapHighlights,
    getSnapHighlightSnapshot,
    publishSnapHighlights,
    subscribeSnapHighlight,
} from "./snapHighlight";

let checks = 0;
function assertEqual(actual: unknown, expected: unknown, label: string): void {
    const a = JSON.stringify(actual);
    const e = JSON.stringify(expected);
    if (a !== e) throw new Error(`${label}: ${a} != ${e}`);
    checks += 1;
}

function assertTrue(value: boolean, label: string): void {
    if (!value) throw new Error(`${label}: expected true`);
    checks += 1;
}

// ── 初始为空 ──
assertEqual(getSnapHighlightSnapshot().entries.length, 0, "initial empty");

// ── 发布与订阅 ──
let notified = 0;
const unsubscribe = subscribeSnapHighlight(() => {
    notified += 1;
});
const entryA = buildCandidateHighlightEntry({
    id: "primary",
    kind: "grid",
    sec: 2,
    sources: [{ trackId: "t1", clipId: "c1" }],
});
publishSnapHighlights(SNAP_HIGHLIGHT_GROUP, [entryA]);
assertTrue(notified >= 1, "listener notified");
assertEqual(getSnapHighlightSnapshot().entries.length, 1, "one entry after publish");

const stored = getSnapHighlightSnapshot().entries[0];
assertEqual(stored.kind, "grid", "entry kind");
assertEqual(stored.markers.length, 2, "target + source markers");
assertEqual(stored.markers[0].role, "target", "first marker is target");
assertEqual(stored.markers[0].trackId, null, "grid target is full-height");
assertEqual(stored.markers[1].role, "source", "second marker is source");
assertEqual(stored.markers[1].sec, 2, "source defaults to snapped sec");
assertEqual(stored.markers[1].trackId, "t1", "source trackId");

// ── 同组发布整组替换 ──
const entryB = buildCandidateHighlightEntry({ id: "primary", kind: "clipEnd", sec: 3 });
publishSnapHighlights(SNAP_HIGHLIGHT_GROUP, [entryB]);
assertEqual(getSnapHighlightSnapshot().entries.length, 1, "group replace keeps single entry");
assertEqual(getSnapHighlightSnapshot().entries[0].kind, "clipEnd", "replaced kind");

// ── 多条目（框选两缘）──
publishSnapHighlights(SNAP_HIGHLIGHT_GROUP, [
    buildCandidateHighlightEntry({ id: "edge-left", kind: "grid", sec: 1 }),
    buildCandidateHighlightEntry({ id: "edge-right", kind: "cursor", sec: 4 }),
]);
assertEqual(getSnapHighlightSnapshot().entries.length, 2, "two edge entries");
assertEqual(
    getSnapHighlightSnapshot().entries.map((e) => e.id),
    [`${SNAP_HIGHLIGHT_GROUP}\u0000edge-left`, `${SNAP_HIGHLIGHT_GROUP}\u0000edge-right`],
    "ids namespaced by group",
);

// ── 按组清除 / 全清 ──
publishSnapHighlights("other-group", [
    buildCandidateHighlightEntry({ id: "x", kind: "selection", sec: 9 }),
]);
assertEqual(getSnapHighlightSnapshot().entries.length, 3, "other group coexists");
clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
assertEqual(getSnapHighlightSnapshot().entries.length, 1, "group-scoped clear");
clearSnapHighlights();
assertEqual(getSnapHighlightSnapshot().entries.length, 0, "clear all");

// 幂等：重复 clear 不触发通知
const before = notified;
clearSnapHighlights();
assertEqual(notified, before, "no notify on redundant clear");

// ── 循环节高亮构建 ──
const loopEntry = buildLoopBoundaryHighlightEntry({
    secs: [5],
    trackId: "t2",
    clipId: "c2",
});
assertEqual(loopEntry.kind, "loopBoundary", "loop kind");
assertEqual(loopEntry.markers.length, 2, "loop single sec duplicated to target+source");

const loopEntry2 = buildLoopBoundaryHighlightEntry({
    secs: [5, 7],
    trackId: "t2",
    clipId: "c2",
});
assertEqual(loopEntry2.markers.map((m) => m.sec), [5, 7], "loop two secs preserved");

unsubscribe();

console.log(`snapHighlight checks passed (${checks})`);
