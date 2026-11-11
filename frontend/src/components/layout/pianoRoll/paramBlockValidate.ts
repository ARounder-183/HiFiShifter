/**
 * 参数块编辑集 / 视图的**前端**校验端口（与 Rust `param_blocks/validate.rs` 逐条同构）。
 *
 * 【为什么必须在前端再实现一遍】
 * 后端 `ParamBlockEditSet::replace` 会先 `validate()`，而前端唯一的"校验"是
 * `composeParamBlockView` —— 它用 subtract 静默裁掉重叠。于是"前端看起来正常、
 * 后端 `Invalid:` 拒绝"成为一类**确定性**失败，用户感知为随机。
 * 把规则搬到前端之后：非法编辑集在离开浏览器之前就被挡住，并且能给出具体 code。
 *
 * 【它不做什么】
 * - 不改写编辑集（不裁剪、不补默认值）—— 规范化是 `normalizeParamBlockEditSet` 的事；
 * - 不判断"业务上是否合理"（例如块是否与素材重叠）—— 那属于投影。
 *
 * ⚠ 后端改动本组规则时必须同步本文件（与 `paramRanges::clampParamWriteValue` 同一约定）。
 *
 * 【错误码 ↔ Rust 原文】
 * | code | Rust |
 * | --- | --- |
 * | schema_version | Invalid: parameter block schema version |
 * | block_count | Invalid: parameter block count exceeded |
 * | block_id | Invalid: parameter block id |
 * | block_members | Invalid: parameter block members |
 * | block_source_span | Invalid: parameter block source span |
 * | block_member_overlap | Invalid: parameter block overlapping members |
 * | block_root_span | Invalid: parameter block root span |
 * | block_root_base_cover | Invalid: parameter block root base cover |
 * | block_group_members | Invalid: parameter block group members |
 * | block_group_base_cover | Invalid: parameter block group base cover |
 * | block_group_mixed | Invalid: mixed parameter block group geometry |
 * | block_deleted_group | Invalid: deleted grouped block |
 * | block_deleted_root | Invalid: deleted root block |
 * | block_deleted_child | Invalid: deleted group child |
 * | block_transition | Invalid: parameter block transition |
 * | block_value | Invalid: parameter block value |
 * | block_base_cover | Invalid: parameter block base cover |
 * | block_base_cover_span | Invalid: parameter block base cover span |
 * | block_base_cover_duplicate | Invalid: parameter block base cover duplicate |
 * | group_depth | Invalid: parameter block group depth |
 * | view_schema | Invalid: parameter block view schema |
 * | view_count | Invalid: parameter block view count |
 * | view_interval | Invalid: parameter block view interval |
 * | view_transition | Invalid: parameter block view transition |
 */
import type {
    BlockGeometry,
    ParamBlockEditSet,
    ParamBlockView,
    TakeSpan,
    UserParamBlock,
} from "./paramBlockTypes";

/** 与 Rust 逐值一致。 */
export const MAX_BLOCKS = 20_000;
const MAX_MEMBERS = 20_000;
export const MAX_GROUP_DEPTH = 8;
/** 视图帧区间上界（Rust 的 `1_000_000`）。 */
const MAX_VIEW_END_FRAME = 1_000_000;

export type ParamBlockValidationCode =
    | "schema_version"
    | "block_count"
    | "block_id"
    | "block_members"
    | "block_source_span"
    | "block_member_overlap"
    | "block_root_span"
    | "block_root_base_cover"
    | "block_group_members"
    | "block_group_base_cover"
    | "block_group_mixed"
    | "block_deleted_group"
    | "block_deleted_root"
    | "block_deleted_child"
    | "block_transition"
    | "block_value"
    | "block_base_cover"
    | "block_base_cover_span"
    | "block_base_cover_duplicate"
    | "group_depth"
    | "view_schema"
    | "view_count"
    | "view_interval"
    | "view_transition";

export type ParamBlockValidation =
    | { readonly ok: true }
    | {
          readonly ok: false;
          readonly code: ParamBlockValidationCode;
          /** 出问题的块 id / 下标，仅用于诊断。 */
          readonly detail?: string;
      };

const OK: ParamBlockValidation = { ok: true };

function fail(code: ParamBlockValidationCode, detail?: string): ParamBlockValidation {
    return detail === undefined ? { ok: false, code } : { ok: false, code, detail };
}

function isGroup(block: UserParamBlock): boolean {
    return block.geometry.kind === "group";
}

function isTake(block: UserParamBlock): boolean {
    return block.geometry.kind === "take";
}

function validSpan(span: TakeSpan): boolean {
    return (
        span.clipId.length > 0 &&
        span.takeId.length > 0 &&
        Number.isFinite(span.sourceStartSec) &&
        Number.isFinite(span.sourceEndSec) &&
        span.sourceEndSec > span.sourceStartSec
    );
}

function validateCoverSpans(spans: readonly TakeSpan[]): ParamBlockValidation {
    for (let index = 0; index < spans.length; index += 1) {
        const span = spans[index];
        if (!validSpan(span)) return fail("block_base_cover_span");
        for (let other = 0; other < index; other += 1) {
            const previous = spans[other];
            if (
                span.clipId === previous.clipId &&
                span.takeId === previous.takeId &&
                span.cycle === previous.cycle &&
                Math.abs(span.sourceStartSec - previous.sourceStartSec) <= 1e-9 &&
                Math.abs(span.sourceEndSec - previous.sourceEndSec) <= 1e-9
            ) {
                return fail("block_base_cover_duplicate");
            }
        }
    }
    return OK;
}

/** 组容器的叶子是否混用了 take / root 几何（Rust 的 `flatten_children`）。 */
function flattenLeaves(block: UserParamBlock, out: UserParamBlock[]): void {
    const geometry: BlockGeometry = block.geometry;
    if (geometry.kind === "group") {
        for (const child of geometry.children) flattenLeaves(child, out);
        return;
    }
    out.push(block);
}

interface ValidationContext {
    total: number;
    readonly ids: Set<string>;
}

function validateBlock(
    block: UserParamBlock,
    depth: number,
    ctx: ValidationContext,
): ParamBlockValidation {
    if (depth > MAX_GROUP_DEPTH) return fail("group_depth", block.id);
    ctx.total += 1;
    if (ctx.total > MAX_BLOCKS) return fail("block_count");
    if (block.id.length === 0 || block.id.length > 512 || ctx.ids.has(block.id)) {
        return fail("block_id", block.id);
    }
    ctx.ids.add(block.id);
    if (block.deleted === true && isGroup(block)) return fail("block_deleted_group", block.id);

    const geometry = block.geometry;
    if (geometry.kind === "take") {
        const members = geometry.members;
        if (members.length === 0 || members.length > MAX_MEMBERS) {
            return fail("block_members", block.id);
        }
        for (let index = 0; index < members.length; index += 1) {
            const member = members[index];
            if (!validSpan(member)) return fail("block_source_span", block.id);
            for (let other = 0; other < index; other += 1) {
                const previous = members[other];
                if (
                    member.clipId === previous.clipId &&
                    member.takeId === previous.takeId &&
                    member.cycle === previous.cycle &&
                    member.sourceStartSec < previous.sourceEndSec - 1e-9 &&
                    previous.sourceStartSec < member.sourceEndSec - 1e-9
                ) {
                    return fail("block_member_overlap", block.id);
                }
            }
        }
    } else if (geometry.kind === "root") {
        if (block.deleted === true) return fail("block_deleted_root", block.id);
        const span = geometry.span;
        if (
            !Number.isFinite(span.startSec) ||
            !Number.isFinite(span.endSec) ||
            span.endSec <= span.startSec
        ) {
            return fail("block_root_span", block.id);
        }
        if (block.baseCover !== undefined) return fail("block_root_base_cover", block.id);
    } else {
        if (block.deleted === true) return fail("block_deleted_group", block.id);
        const children = geometry.children;
        if (children.length < 2 || children.length > MAX_MEMBERS) {
            return fail("block_group_members", block.id);
        }
        if (block.baseCover !== undefined) return fail("block_group_base_cover", block.id);
        for (const child of children) {
            if (child.deleted === true) return fail("block_deleted_child", block.id);
            const result = validateBlock(child, depth + 1, ctx);
            if (!result.ok) return result;
        }
        const leaves: UserParamBlock[] = [];
        for (const child of children) flattenLeaves(child, leaves);
        const hasTake = leaves.some(isTake);
        const hasRoot = leaves.some((leaf) => leaf.geometry.kind === "root");
        if (hasTake && hasRoot) return fail("block_group_mixed", block.id);
    }

    const transition = block.transitionOutSec;
    if (transition !== undefined && (!Number.isFinite(transition) || transition < 0)) {
        return fail("block_transition", block.id);
    }
    const value = block.value;
    if (value !== undefined && value !== null && !Number.isFinite(value)) {
        return fail("block_value", block.id);
    }
    const cover = block.baseCover;
    if (cover !== undefined) {
        if (cover.spans.length === 0 || cover.spans.length > MAX_MEMBERS) {
            return fail("block_base_cover", block.id);
        }
        const coverResult = validateCoverSpans(cover.spans);
        if (!coverResult.ok) return coverResult;
    }
    return OK;
}

/** 编辑集校验（对应 Rust `validate_edit_set`）。 */
export function validateParamBlockEditSet(edits: ParamBlockEditSet): ParamBlockValidation {
    if (edits.schemaVersion !== 2) return fail("schema_version");
    const ctx: ValidationContext = { total: 0, ids: new Set<string>() };
    for (const block of edits.blocks) {
        const result = validateBlock(block, 0, ctx);
        if (!result.ok) return result;
    }
    if (ctx.total > MAX_BLOCKS) return fail("block_count");
    return OK;
}

/** 派生视图校验（对应 Rust `validate_view`）。 */
export function validateParamBlockView(view: ParamBlockView): ParamBlockValidation {
    if (
        view.schemaVersion !== 1 ||
        !Number.isFinite(view.framePeriodMs) ||
        view.framePeriodMs <= 0
    ) {
        return fail("view_schema");
    }
    if (view.blocks.length > MAX_BLOCKS) return fail("view_count");
    const ids = new Set<string>();
    let previousEnd = 0;
    for (const block of view.blocks) {
        if (
            block.id.length === 0 ||
            block.id.length > 512 ||
            ids.has(block.id) ||
            block.endFrameExclusive <= block.startFrame ||
            block.startFrame < previousEnd ||
            block.endFrameExclusive > MAX_VIEW_END_FRAME
        ) {
            return fail("view_interval", block.id);
        }
        if (
            block.transitionFrames !== undefined &&
            block.transitionFrames > block.endFrameExclusive - block.startFrame
        ) {
            return fail("view_transition", block.id);
        }
        previousEnd = block.endFrameExclusive;
    }
    return OK;
}
