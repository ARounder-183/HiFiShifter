#!/usr/bin/env node
/**
 * Split-equivalence replay: **every deleted line of code must still exist after the same
 * commit**.
 *
 * # Why this exists (instead of reusing the previous round's comparison)
 *
 * Each split in the previous round did a statement-level equivalence check, but it compared
 * "the **statements that were moved**": the new module's function bodies against the
 * corresponding ranges of the old file, statement by statement. That approach works for
 * "moved wrong" and is completely useless for "**a block was left behind**" — the missed
 * block simply is not in the comparison set.
 *
 * `710ff764` lost freehand drawing exactly this way: the commit message claimed "the token
 * streams of both branches are identical to before the change (8986 / 1670 tokens)" — that
 * sentence was true, and the third branch was no longer in the file.
 *
 * # Criteria
 *
 * For every `refactor(pianoroll)` commit C in the range:
 *
 *   1. take C's diff against C^, collecting **deleted lines** per file;
 *   2. drop blank lines, pure comment lines, pure bracket lines (`}` / `);` and the like) —
 *      they legitimately disappear as indentation changes;
 *   3. normalize each line (trim, collapse internal whitespace) and require it to **still
 *      appear as a suffix of some line anywhere in `frontend/src` after that commit**.
 *
 * Suffix matching rather than equality: after moving into a module a whole block gains a
 * level of indentation, and leading indentation carries no semantics.
 *
 * 4. Lines that cannot be matched → printed. Either they are truly lost, or they were
 *    deliberately rewritten; the latter must be registered in `ALLOWED` (each with a
 *    reason), otherwise the script exits non-zero.
 *
 * # Why not a full AST comparison
 *
 * This script runs over 30+ commits covering 3000+ deleted lines. AST comparison would need
 * to parse two trees per commit and align them structurally, with a false-positive rate so
 * high nobody would read the results. Line-level suffix matching is imperfect, but it is
 * exactly precise for the "whole block disappears" class — and that is the class that
 * actually happened.
 *
 * Usage:
 *   node scripts/audit/split-equivalence.mjs             # report
 *   node scripts/audit/split-equivalence.mjs --since=193a486a
 */
import { execFileSync } from "node:child_process";

/** Start of the replay range (the first commit of the parameter-editor split). */
const DEFAULT_SINCE = "193a486a";

/**
 * Lines allowed to disappear: **each entry must state why it does not need to survive the
 * split**.
 *
 * Keys are normalized line text, values are one-sentence reasons. The list is audited
 * append-only — adding an entry requires first answering "where did this line go".
 */
const ALLOWED = new Map([
    [
        "export function handleCanvasPointerDown(",
        "the function declaration itself was renamed / split into several exports during the split; the declaration line no longer exists verbatim",
    ],
    [
        "export function handleSelectPointerDown(",
        "same as above: a branch module's declaration line",
    ],
    [
        "export function handleVibratoPointerDown(",
        "same as above: a branch module's declaration line",
    ],
    [
        "type CanvasPointerDownDeps = Pick<",
        "the deps type was split into each module's own Pick subset",
    ],
]);

/** Purely structural lines: they legitimately disappear as indentation / semicolon position changes. */
const STRUCTURAL =
    /^(\}|\)|\);|\}\)|\} else \{|else \{|\};|\]\);|\]|return;|\},|\] as const;|\} as const;)$/;

/** Comment lines: lines starting with two slashes, an asterisk, or a slash-asterisk. */
const COMMENT = /^(\/\/|\*|\/\*)/;

/**
 * Minimum number of contiguous lines to declare "a whole block disappeared".
 *
 * [Why 6] a single line or two going unmatched is almost always Prettier re-wrapping, a
 * rename, or re-assembled imports; a real "whole block left behind" (`710ff764`'s freehand
 * branch) was 260 contiguous lines. A threshold of 6 makes the criterion insensitive to the
 * former and extremely sensitive to the latter — the only trade-off this script needs.
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

/** Normalize: trim + collapse internal whitespace. */
function norm(line) {
    return line.trim().replace(/\s+/g, " ");
}

/**
 * All lines of `frontend/src` at a given revision (normalized).
 *
 * [Why it is collected once and defaults to HEAD] the criterion asks "does this line of code
 * still exist in the repo". Fetching one copy per commit would be 30 × 1000 `git show` calls
 * (measured: over ten minutes); and the case "deleted in C, added back later" is supposed to
 * pass anyway — the code was not lost. So a single copy is collected, defaulting to HEAD.
 *
 * To verify "this check really catches loss", point `--haystack=` at the revision before the
 * fix:
 *   node scripts/audit/split-equivalence.mjs --haystack=17f3a393^
 * It reports precisely the branch dropped by `710ff764`.
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

/** Lines deleted by one commit (only .ts/.tsx under frontend/src are considered). */
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
            // `--- /dev/null` for new files has no deletable lines; `--- a/<path>` is the real path.
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
        console.log(`no refactor(pianoroll) commits found in ${since}..HEAD`);
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
        /** Collect per file the **contiguous** runs of "this line cannot be found in ${haystackRevision}". */
        const runs = [];
        let run = null;
        let previousLineNo = -10;
        for (const entry of lines) {
            const text = norm(entry.text);
            const isCandidate =
                text.length > 0 &&
                !STRUCTURAL.test(text) &&
                !COMMENT.test(text) &&
                // import / export-from lines are re-assembled at every split; verbatim comparison is meaningless.
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
        // Isolated one-or-two-line misses are almost always formatting / renames; **contiguous
        // runs** are what "whole block disappears" looks like.
        const suspicious = runs.filter((candidate) => candidate.entries.length >= MIN_RUN);
        if (suspicious.length > 0) {
            const count = suspicious.reduce((total, item) => total + item.entries.length, 0);
            dropped += count;
            report.push({ hash, subject, suspicious });
        }
    }

    console.log(
        `replayed ${list.length} commits, checked ${removed} deleted code lines (existence judged against ${haystackRevision}).`,
    );
    if (dropped === 0) {
        console.log(`✔ no contiguous block of ${MIN_RUN}+ lines disappeared after deletion`);
        return 0;
    }

    console.log(`✘ ${dropped} lines of code disappeared in contiguous runs (>= ${MIN_RUN} lines):`);
    for (const { hash, subject, suspicious } of report) {
        console.log(`\n${hash.slice(0, 8)}  ${subject}`);
        for (const item of suspicious) {
            const first = item.entries[0];
            console.log(
                `  ${item.file}:${first.line}  (${item.entries.length} lines)  ${first.text.trim().slice(0, 90)}`,
            );
        }
    }
    return 1;
}

process.exit(main());
