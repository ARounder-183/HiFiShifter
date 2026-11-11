/**
 * 前端校验端口（./paramBlockValidate）自检。
 *
 * 【本测试守护什么】每条用例对应 Rust `validate.rs` 的一条规则。它们的存在
 * 意义不是"再测一遍后端"，而是钉住**前端预检覆盖了哪些规则** —— 少一条，
 * 那条规则就会退化成"用户看到随机失败"。
 */
import { describe, expect, it } from "vitest";

import {
    MAX_BLOCKS,
    MAX_GROUP_DEPTH,
    validateParamBlockEditSet,
    validateParamBlockView,
} from "./paramBlockValidate";
import type {
    ParamBlockEditSet,
    ParamBlockView,
    TakeSpan,
    UserParamBlock,
} from "./paramBlockTypes";

function span(overrides: Partial<TakeSpan> = {}): TakeSpan {
    return {
        clipId: "clip",
        takeId: "take",
        cycle: 0,
        sourceStartSec: 0,
        sourceEndSec: 1,
        label: "manual",
        ...overrides,
    };
}

function takeBlock(id: string, members: readonly TakeSpan[] = [span()]): UserParamBlock {
    return { id, geometry: { kind: "take", members }, createdSeq: 1 };
}

function edits(blocks: readonly UserParamBlock[]): ParamBlockEditSet {
    return { schemaVersion: 2, revision: 0, blocks };
}

describe("validateParamBlockEditSet", () => {
    it("accepts a well-formed edit set", () => {
        expect(validateParamBlockEditSet(edits([takeBlock("a")]))).toEqual({ ok: true });
        expect(validateParamBlockEditSet(edits([]))).toEqual({ ok: true });
    });

    it("rejects a wrong schema version", () => {
        const bad = { schemaVersion: 1, revision: 0, blocks: [] } as unknown as ParamBlockEditSet;
        expect(validateParamBlockEditSet(bad)).toMatchObject({ ok: false, code: "schema_version" });
    });

    it("rejects empty / oversized / duplicate ids", () => {
        expect(validateParamBlockEditSet(edits([takeBlock("")]))).toMatchObject({
            code: "block_id",
        });
        expect(validateParamBlockEditSet(edits([takeBlock("x".repeat(513))]))).toMatchObject({
            code: "block_id",
        });
        expect(
            validateParamBlockEditSet(edits([takeBlock("dup"), takeBlock("dup")])),
        ).toMatchObject({ code: "block_id" });
    });

    it("rejects empty members and invalid source spans", () => {
        expect(validateParamBlockEditSet(edits([takeBlock("a", [])]))).toMatchObject({
            code: "block_members",
        });
        expect(
            validateParamBlockEditSet(edits([takeBlock("a", [span({ sourceEndSec: 0 })])])),
        ).toMatchObject({ code: "block_source_span" });
        expect(
            validateParamBlockEditSet(
                edits([takeBlock("a", [span({ sourceStartSec: Number.NaN })])]),
            ),
        ).toMatchObject({ code: "block_source_span" });
    });

    it("★ rejects overlapping members but accepts touching ones", () => {
        expect(
            validateParamBlockEditSet(
                edits([
                    takeBlock("a", [
                        span({ sourceStartSec: 0, sourceEndSec: 1 }),
                        span({ sourceStartSec: 0.5, sourceEndSec: 1.5 }),
                    ]),
                ]),
            ),
        ).toMatchObject({ code: "block_member_overlap" });
        expect(
            validateParamBlockEditSet(
                edits([
                    takeBlock("a", [
                        span({ sourceStartSec: 0, sourceEndSec: 1 }),
                        span({ sourceStartSec: 1, sourceEndSec: 2 }),
                    ]),
                ]),
            ),
        ).toEqual({ ok: true });
        // 不同 take 之间重叠是合法的。
        expect(
            validateParamBlockEditSet(
                edits([
                    takeBlock("a", [
                        span({ sourceStartSec: 0, sourceEndSec: 1 }),
                        span({ takeId: "other", sourceStartSec: 0.5, sourceEndSec: 1.5 }),
                    ]),
                ]),
            ),
        ).toEqual({ ok: true });
    });

    it("validates root blocks", () => {
        const root = (overrides: Partial<UserParamBlock> = {}): UserParamBlock => ({
            id: "r",
            geometry: { kind: "root", span: { startSec: 0, endSec: 1 } },
            createdSeq: 1,
            ...overrides,
        });
        expect(validateParamBlockEditSet(edits([root()]))).toEqual({ ok: true });
        expect(
            validateParamBlockEditSet(
                edits([root({ geometry: { kind: "root", span: { startSec: 1, endSec: 1 } } })]),
            ),
        ).toMatchObject({ code: "block_root_span" });
        expect(validateParamBlockEditSet(edits([root({ deleted: true })]))).toMatchObject({
            code: "block_deleted_root",
        });
        expect(
            validateParamBlockEditSet(
                edits([root({ baseCover: { spans: [span()], full: false } })]),
            ),
        ).toMatchObject({ code: "block_root_base_cover" });
    });

    it("validates group blocks", () => {
        const group = (children: readonly UserParamBlock[]): UserParamBlock => ({
            id: "g",
            geometry: { kind: "group", children },
            createdSeq: 1,
        });
        expect(
            validateParamBlockEditSet(edits([group([takeBlock("c1"), takeBlock("c2")])])),
        ).toEqual({ ok: true });
        expect(validateParamBlockEditSet(edits([group([takeBlock("c1")])]))).toMatchObject({
            code: "block_group_members",
        });
        expect(
            validateParamBlockEditSet(
                edits([group([takeBlock("c1"), { ...takeBlock("c2"), deleted: true }])]),
            ),
        ).toMatchObject({ code: "block_deleted_child" });
        // take 与 root 混用
        expect(
            validateParamBlockEditSet(
                edits([
                    group([
                        takeBlock("c1"),
                        {
                            id: "c2",
                            geometry: { kind: "root", span: { startSec: 0, endSec: 1 } },
                            createdSeq: 1,
                        },
                    ]),
                ]),
            ),
        ).toMatchObject({ code: "block_group_mixed" });
        // 组不能有 baseCover
        expect(
            validateParamBlockEditSet(
                edits([
                    {
                        ...group([takeBlock("c1"), takeBlock("c2")]),
                        baseCover: { spans: [span()], full: false },
                    },
                ]),
            ),
        ).toMatchObject({ code: "block_group_base_cover" });
    });

    it("rejects a group deeper than the limit", () => {
        let node: UserParamBlock = takeBlock("leaf");
        for (let depth = 0; depth <= MAX_GROUP_DEPTH; depth += 1) {
            node = {
                id: "g" + depth,
                geometry: { kind: "group", children: [node, takeBlock("sibling" + depth)] },
                createdSeq: 1,
            };
        }
        expect(validateParamBlockEditSet(edits([node]))).toMatchObject({ code: "group_depth" });
    });

    it("rejects a non-finite transition or value", () => {
        expect(
            validateParamBlockEditSet(edits([{ ...takeBlock("a"), transitionOutSec: Number.NaN }])),
        ).toMatchObject({ code: "block_transition" });
        expect(
            validateParamBlockEditSet(edits([{ ...takeBlock("a"), transitionOutSec: -1 }])),
        ).toMatchObject({ code: "block_transition" });
        expect(
            validateParamBlockEditSet(
                edits([{ ...takeBlock("a"), value: Number.POSITIVE_INFINITY }]),
            ),
        ).toMatchObject({ code: "block_value" });
    });

    it("validates base covers", () => {
        expect(
            validateParamBlockEditSet(
                edits([{ ...takeBlock("a"), baseCover: { spans: [], full: false } }]),
            ),
        ).toMatchObject({ code: "block_base_cover" });
        expect(
            validateParamBlockEditSet(
                edits([
                    {
                        ...takeBlock("a"),
                        baseCover: {
                            spans: [
                                span({ sourceStartSec: 0, sourceEndSec: 1 }),
                                span({ sourceStartSec: 0, sourceEndSec: 1 }),
                            ],
                            full: false,
                        },
                    },
                ]),
            ),
        ).toMatchObject({ code: "block_base_cover_duplicate" });
    });

    it("rejects more than MAX_BLOCKS", () => {
        const many = Array.from({ length: MAX_BLOCKS + 1 }, (_, index) => takeBlock("b" + index));
        expect(validateParamBlockEditSet(edits(many))).toMatchObject({ code: "block_count" });
    });
});

describe("validateParamBlockView", () => {
    function view(overrides: Partial<ParamBlockView> = {}): ParamBlockView {
        return {
            schemaVersion: 1,
            rootTrackId: "root",
            editRevision: 1,
            layoutKey: "k",
            framePeriodMs: 5,
            blocks: [],
            orphanCount: 0,
            analysisPending: false,
            ...overrides,
        };
    }

    it("accepts an empty, well-formed view", () => {
        expect(validateParamBlockView(view())).toEqual({ ok: true });
    });

    it("rejects a bad schema or frame period", () => {
        expect(validateParamBlockView(view({ framePeriodMs: 0 }))).toMatchObject({
            code: "view_schema",
        });
    });

    it("requires sorted, non-overlapping, non-empty intervals", () => {
        const block = (id: string, startFrame: number, endFrameExclusive: number) => ({
            id,
            startFrame,
            endFrameExclusive,
            kind: "user" as const,
            members: [],
        });
        expect(
            validateParamBlockView(view({ blocks: [block("a", 0, 5), block("b", 5, 9)] })),
        ).toEqual({ ok: true });
        expect(
            validateParamBlockView(view({ blocks: [block("a", 0, 5), block("b", 4, 9)] })),
        ).toMatchObject({ code: "view_interval" });
        expect(validateParamBlockView(view({ blocks: [block("a", 5, 5)] }))).toMatchObject({
            code: "view_interval",
        });
        expect(validateParamBlockView(view({ blocks: [block("a", 0, 1_000_001)] }))).toMatchObject({
            code: "view_interval",
        });
    });

    it("rejects a transition wider than the block", () => {
        expect(
            validateParamBlockView(
                view({
                    blocks: [
                        {
                            id: "a",
                            startFrame: 0,
                            endFrameExclusive: 10,
                            kind: "user",
                            members: [],
                            transitionFrames: 11,
                        },
                    ],
                }),
            ),
        ).toMatchObject({ code: "view_transition" });
    });
});
