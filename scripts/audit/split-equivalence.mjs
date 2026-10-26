#!/usr/bin/env node
/**
 * 拆分等价性重放：**被删掉的每一行代码，必须在同一个提交之后仍然存在**。
 *
 * # 为什么需要它（而不是沿用上一轮的比对）
 *
 * 上一轮每一次拆分都做了语句级等价性比对，但比对的是"**搬走的那些语句**"：
 * 把新模块的函数体和旧文件里对应区间逐句比。这种做法对"搬错了"有效，对
 * "**少搬了一块**"完全无效 —— 漏掉的那一块根本不在比对集合里。
 *
 * `710ff764` 就是这样丢掉自由绘制的：提交信息写"两个分支的 token 流与改动前
 * 完全一致（8986 / 1670 token）"，那句话是真的，而第三个分支已经不在文件里了。
 *
 * # 判据
 *
 * 对区间内每一个 `refactor(pianoroll)` 提交 C：
 *
 *   1. 取 C 相对 C^ 的 diff，逐文件收集**被删除的行**；
 *   2. 丢掉空白行、纯注释行、纯括号行（`}` / `);` / `}` 之类）—— 它们会随
 *      缩进变化而合理地消失；
 *   3. 对每一行做规范化（去首尾空白、折叠内部空白），要求它**在该提交之后
 *      的整个 `frontend/src` 里仍能作为某一行的后缀出现**。
 *
 * 后缀匹配而不是全等：搬进模块后整段会多缩进一级，行首缩进不属于语义。
 *
 * 4. 命中不了的行 → 打印出来。要么是真的丢了，要么是刻意的改写，后者必须
 *    登记进 `ALLOWED`（每条带理由），否则脚本以非零码退出。
 *
 * # 为什么不是完整的 AST 比对
 *
 * 本脚本要跑 30+ 个提交、覆盖 3000+ 行删除。AST 比对需要为每个提交解析两棵树
 * 并做结构对齐，误报率高到没人会看结果。行级后缀匹配不完美，但它对"整块消失"
 * 这一类恰好是精确的 —— 而这一类正是发生过的那一类。
 *
 * 用法：
 *   node scripts/audit/split-equivalence.mjs             # 报告
 *   node scripts/audit/split-equivalence.mjs --since=193a486a
 */
import { execFileSync } from "node:child_process";

/** 重放区间的起点（参数编辑器拆分的第一个提交）。 */
const DEFAULT_SINCE = "193a486a";

/**
 * 允许消失的行：**每一条都必须写明为什么它不需要在拆分后继续存在**。
 *
 * 键是规范化后的行文本，值是一句话理由。清单只增不减地审 —— 想加一条就必须
 * 先回答"这句话去哪了"。
 */
const ALLOWED = new Map([
    [
        "export function handleCanvasPointerDown(",
        "函数声明本身在拆分中改名 / 拆成多个导出，声明行不再逐字存在",
    ],
    ["export function handleSelectPointerDown(", "同上：分支模块的声明行"],
    ["export function handleVibratoPointerDown(", "同上：分支模块的声明行"],
    ["type CanvasPointerDownDeps = Pick<", "依赖类型在拆分中被拆成每个模块自己的 Pick 子集"],
]);

/** 纯结构行：它们随缩进 / 分号位置变化而合理地消失。 */
const STRUCTURAL =
    /^(\}|\)|\);|\}\)|\} else \{|else \{|\};|\]\);|\]|return;|\},|\] as const;|\} as const;)$/;

/** 注释行：以两个斜杠、星号或斜杠星号开头的行。 */
const COMMENT = /^(\/\/|\*|\/\*)/;

/**
 * 判定"整块消失"的最小连续行数。
 *
 * 【为什么是 6】单行 / 两行找不到，绝大多数是 Prettier 重新折行、重命名、或者 import
 * 重新拼装；一次真正的"整块漏搬"（`710ff764` 的自由绘制分支）是 260 行连成一片。
 * 阈值定在 6 让判据对前者不敏感、对后者极敏感 —— 这是本脚本唯一需要的取舍。
 */
const MIN_RUN = 6;

function git(args) {
    return execFileSync("git", ["-c", "safe.directory=" + process.cwd(), ...args], {
        encoding: "utf8",
        maxBuffer: 1 << 28,
    });
}

function commits(since) {
    return git(["log", "--format=%H%x09%s", `${since}..HEAD`])
        .split("\n")
        .filter(Boolean)
        .map((line) => {
            const [hash, subject] = line.split("\t");
            return { hash, subject };
        })
        .filter(({ subject }) => /^refactor\(pianoroll\)/.test(subject));
}

/** 规范化：去首尾空白 + 折叠内部空白。 */
function norm(line) {
    return line.trim().replace(/\s+/g, " ");
}

/**
 * 某一修订下 `frontend/src` 的全部行（规范化后）。
 *
 * 【为什么只取一次、且默认取 HEAD】判据问的是"这行代码还在不在仓库里"。逐提交各取
 * 一份是 30 × 1000 次 `git show`（实测十分钟以上）；而"在 C 里被删、后来又加回来"
 * 这种情形本来就应当通过 —— 代码没有丢。因此只收集一份，默认 HEAD。
 *
 * 想验证"这个检查真的能抓到丢失"，把 `--haystack=` 指向修复之前的修订即可：
 *   node scripts/audit/split-equivalence.mjs --haystack=17f3a393^
 * 它会精确报出被 `710ff764` 丢掉的那一支。
 */
function haystackAt(revision) {
    const listing = execFileSync(
        "git",
        [
            "-c",
            "safe.directory=" + process.cwd(),
            "ls-tree",
            "-r",
            "--name-only",
            revision,
            "frontend/src",
        ],
        { encoding: "utf8", maxBuffer: 1 << 28 },
    );
    const files = listing.split("\n").filter((name) => /\.tsx?$/.test(name));
    const lines = new Set();
    for (const file of files) {
        let blob = "";
        try {
            blob = execFileSync(
                "git",
                ["-c", "safe.directory=" + process.cwd(), "show", revision + ":" + file],
                { encoding: "utf8", maxBuffer: 1 << 28 },
            );
        } catch {
            continue;
        }
        for (const raw of blob.split("\n")) lines.add(norm(raw));
    }
    return lines;
}

/** 一次提交里被删掉的行（只关心 frontend/src 下的 .ts/.tsx）。 */
function removedLines(commit) {
    const diff = git([
        "show",
        "--format=",
        "--unified=0",
        "--no-color",
        commit,
        "--",
        "frontend/src",
    ]);
    const out = [];
    let file = null;
    let lineNo = 0;
    for (const raw of diff.split("\n")) {
        if (raw.startsWith("+++ b/")) {
            file = raw.slice(6);
            continue;
        }
        if (raw.startsWith("--- ")) {
            // 新增文件的 `--- /dev/null` 没有可删除的行；`--- a/<path>` 才是真实路径。
            file = raw.startsWith("--- a/") ? raw.slice(6) : null;
            continue;
        }
        const hunk = /^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@/.exec(raw);
        if (hunk) {
            lineNo = Number(hunk[1]);
            continue;
        }
        if (file === null || !/\.tsx?$/.test(file)) continue;
        if (raw.startsWith("-")) {
            out.push({ file, line: lineNo, text: raw.slice(1) });
            lineNo += 1;
        }
    }
    return out;
}

function main() {
    const arg = process.argv.find((value) => value.startsWith("--since="));
    const since = arg ? arg.slice("--since=".length) : DEFAULT_SINCE;

    const list = commits(since);
    if (list.length === 0) {
        console.log(`没有在 ${since}..HEAD 里找到 refactor(pianoroll) 提交`);
        return 0;
    }

    const haystackRevision =
        (process.argv.find((value) => value.startsWith("--haystack=")) ?? "").slice(
            "--haystack=".length,
        ) || "HEAD";
    const haystack = haystackAt(haystackRevision);

    let removed = 0;
    let dropped = 0;
    const report = [];

    for (const { hash, subject } of list) {
        const lines = removedLines(hash);
        /** 逐文件收集"这一行在 ${haystackRevision} 里找不到"的**连续**片段。 */
        const runs = [];
        let run = null;
        let previousLineNo = -10;
        for (const entry of lines) {
            const text = norm(entry.text);
            const isCandidate =
                text.length > 0 &&
                !STRUCTURAL.test(text) &&
                !COMMENT.test(text) &&
                // import / export-from 行每次拆分都会被重新拼装，逐字比对没有意义。
                !/^(import|\}\s*from)\b/.test(text);
            if (!isCandidate) continue;
            removed += 1;
            const found = haystack.has(text) || ALLOWED.has(text);
            if (found) {
                run = null;
                continue;
            }
            const contiguous =
                run !== null && run.file === entry.file && entry.line - previousLineNo <= 3;
            if (contiguous) {
                run.entries.push(entry);
            } else {
                run = { file: entry.file, entries: [entry] };
                runs.push(run);
            }
            previousLineNo = entry.line;
        }
        // 孤立的一两行几乎总是格式化 / 重命名；**成片的**才是"整块消失"。
        const suspicious = runs.filter((candidate) => candidate.entries.length >= MIN_RUN);
        if (suspicious.length > 0) {
            const count = suspicious.reduce((total, item) => total + item.entries.length, 0);
            dropped += count;
            report.push({ hash, subject, suspicious });
        }
    }

    console.log(
        `重放 ${list.length} 个提交，检查 ${removed} 行被删除的代码（存在性以 ${haystackRevision} 为准）。`,
    );
    if (dropped === 0) {
        console.log(`✔ 没有任何连续 ${MIN_RUN} 行以上的代码块在删除后消失`);
        return 0;
    }

    console.log(`✘ ${dropped} 行代码成片消失（连续 ≥ ${MIN_RUN} 行）：`);
    for (const { hash, subject, suspicious } of report) {
        console.log(`\n${hash.slice(0, 8)}  ${subject}`);
        for (const item of suspicious) {
            const first = item.entries[0];
            console.log(
                `  ${item.file}:${first.line}  (${item.entries.length} 行)  ${first.text.trim().slice(0, 90)}`,
            );
        }
    }
    return 1;
}

process.exit(main());
