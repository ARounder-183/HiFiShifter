/**
 * 编辑集的**结构指纹**。
 *
 * 【它回答什么问题】「这些块的形状有没有被第三方改过」。视觉预览只应该在
 * **结构**变化时撤下 —— 一次拖动只改 `value`（块中心）与曲线，形状没动。
 *
 * 【为什么不能用 revision】`ParamBlockEditSet.revision` 在前端是**内容代次**
 * （每落一个块 +1，一次切分 +3），在后端是**CAS 代次**（每次提交 +1，与前端传的
 * 值无关）。两者**没有可比性**，于是旧实现里的「自己提交的 revision 归属」
 * （`ownedEditRevision`）永远对不上服务端回包 —— 回包落地的那一刻预览被判成
 * 「第三方结构编辑」撤下，曲线回到旧值，等权威到了再跳到新值。用户看到的就是
 * 「松手后先回弹旧参数、再生效新参数」。
 *
 * 【包含】块 id、删除标记、几何（take 成员的 clip / take / cycle / 源区间 /
 * 标签；root 的秒区间；group 的子块递归）、过渡宽度、baseCover 覆盖。
 * 【不包含】`value`（块中心）、`revision`、`createdSeq`、块在数组里的顺序。
 *
 * 【为什么排除 value】拖动块中心是**值**编辑，不是结构编辑。把它算进结构会让
 * 一次拖动撤销自己的预览 —— 那正是「先回弹」的成因。
 *
 * 【为什么是字符串】它是视觉 epoch 的身份，必须可比较、可打印、可出现在诊断里；
 * 哈希成 32 位十六进制只是为了短。
 *
 * 【违反边界的症状】
 * - 把 value 算进来 ⇒ 拖块松手后自己撤自己的预览（回弹）；
 * - 把 revision 算进来 ⇒ 同上，而且更隐蔽（服务端 revision 不可预测）；
 * - 漏掉某一类几何 ⇒ 第三方分割 / 伸缩 / 换 Clip 后预览不撤，画面上留着幽灵块。
 */
import type { BlockGeometry, ParamBlockEditSet, UserParamBlock } from "./paramBlockTypes";

/** 秒域数值的稳定写法：抹掉浮点噪声，避免同一几何算出两个指纹。 */
function sec(value: number): string {
    if (!Number.isFinite(value)) return "nan";
    return String(Math.round(value * 1e9) / 1e9);
}

function geometryToken(geometry: BlockGeometry): string {
    if (geometry.kind === "root") {
        return `root:${sec(geometry.span.startSec)}-${sec(geometry.span.endSec)}`;
    }
    if (geometry.kind === "group") {
        const children = geometry.children.map(blockToken).sort();
        return `group:[${children.join(";")}]`;
    }
    const members = geometry.members
        .map(
            (member) =>
                `${member.clipId}/${member.takeId}/${member.cycle}/` +
                `${sec(member.sourceStartSec)}-${sec(member.sourceEndSec)}/` +
                `${member.label}${member.syllableId === undefined ? "" : `#${member.syllableId}`}`,
        )
        .sort();
    return `take:[${members.join(";")}]`;
}

function baseCoverToken(block: UserParamBlock): string {
    const cover = block.baseCover;
    if (cover === undefined) return "";
    const spans = cover.spans
        .map(
            (span) =>
                `${span.clipId}/${span.takeId}/${span.cycle}/` +
                `${sec(span.sourceStartSec)}-${sec(span.sourceEndSec)}`,
        )
        .sort();
    return `|cover:${cover.full ? "full" : "part"}[${spans.join(";")}]`;
}

function blockToken(block: UserParamBlock): string {
    return (
        `${block.id}|${block.deleted === true ? "del" : "live"}|#${geometryToken(block.geometry)}` +
        `|t:${block.transitionOutSec === undefined ? "" : sec(block.transitionOutSec)}` +
        baseCoverToken(block)
    );
}

/** FNV-1a（32 位）—— 只需要稳定与短，不需要抗碰撞。 */
function fnv1a(text: string): string {
    let hash = 0x811c9dc5;
    for (let index = 0; index < text.length; index += 1) {
        hash ^= text.charCodeAt(index);
        hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    return hash.toString(16).padStart(8, "0");
}

/**
 * 编辑集的结构指纹。
 *
 * 【空集返回 `structure-empty` 而不是空串】空串与「没有令牌」不可区分，
 * 而 epoch 的判定必须能区分「还没算出来」与「确实没有块」。
 */
export function paramBlockStructureToken(edits: ParamBlockEditSet): string {
    if (edits.blocks.length === 0) return "structure-empty";
    const tokens = edits.blocks.map(blockToken).sort();
    return `structure-${fnv1a(tokens.join("\n"))}`;
}
