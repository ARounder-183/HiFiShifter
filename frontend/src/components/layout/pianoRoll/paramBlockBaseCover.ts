/**
 * 参数块 base cover · 源域裁剪
 *
 * 【职责】把一个 baseCover 裁剪到“当前 run / fragment 实际拥有的源域成员”。
 *
 * 【为什么必须裁剪】baseCover 表达“我替代了哪一段基块”。多 run 的 Loop 块在
 * 合成时会把同一份父 baseCover 复制给每个 run；若不裁剪，只编辑第 2 个 run 也会
 * 抑制第 1 / 3 个 run 下的基块，画面上出现无缘无故的空白。
 *
 * 【规则】
 * - 只保留与成员同 clip / take / cycle 且有源域交集的 span；
 * - `full:true` 仅当原 cover 是 full **且**每个 cover span 都被成员完整覆盖；
 *   任何裁剪都会降为 `full:false`，避免“半个基块仍宣称整块被替代”。
 */
import type { BaseCover, TakeSpan } from "./paramBlockTypes";

const EPS = 1e-7;

type CoverMember = Pick<
    TakeSpan,
    "clipId" | "takeId" | "cycle" | "sourceStartSec" | "sourceEndSec"
>;

function sameSource(member: CoverMember, span: TakeSpan): boolean {
    return (
        member.clipId === span.clipId &&
        member.takeId === span.takeId &&
        member.cycle === span.cycle
    );
}

/** 成员区间是否完整覆盖某个 cover span（允许多个相邻成员拼接）。 */
function membersCoverSpan(members: readonly CoverMember[], span: TakeSpan): boolean {
    const intervals = members
        .filter((member) => sameSource(member, span))
        .map(
            (member) =>
                [
                    Math.max(span.sourceStartSec, member.sourceStartSec),
                    Math.min(span.sourceEndSec, member.sourceEndSec),
                ] as [number, number],
        )
        .filter(([start, end]) => end > start + EPS)
        .sort((a, b) => a[0] - b[0]);
    if (intervals.length === 0) return false;
    let cursor = span.sourceStartSec;
    for (const [start, end] of intervals) {
        if (start > cursor + EPS) return false;
        cursor = Math.max(cursor, end);
        if (cursor >= span.sourceEndSec - EPS) return true;
    }
    return cursor >= span.sourceEndSec - EPS;
}

/**
 * 把 cover 裁剪到成员集合。
 *
 * @returns 无任何交集时返回 undefined；否则返回新的 span 数组（不修改输入）。
 */
export function narrowBaseCoverToMembers(
    cover: BaseCover | undefined,
    members: readonly CoverMember[],
): BaseCover | undefined {
    if (cover === undefined || cover.spans.length === 0) return undefined;
    if (members.length === 0) return undefined;
    const spans: TakeSpan[] = [];
    for (const span of cover.spans) {
        for (const member of members) {
            if (!sameSource(member, span)) continue;
            const start = Math.max(span.sourceStartSec, member.sourceStartSec);
            const end = Math.min(span.sourceEndSec, member.sourceEndSec);
            if (end <= start + EPS) continue;
            spans.push({ ...span, sourceStartSec: start, sourceEndSec: end });
        }
    }
    if (spans.length === 0) return undefined;
    const full =
        cover.full === true && cover.spans.every((span) => membersCoverSpan(members, span));
    return { spans, full };
}
