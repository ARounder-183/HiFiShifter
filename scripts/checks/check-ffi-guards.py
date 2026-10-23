"""CI gates for the plugin/App panic-containment and locking discipline.

1. FFI guard gate: every `unsafe extern "system" fn` in the plugin crate must
   contain a panic guard (guard_ffi / catch_unwind / boundary) in its body.
   Since Rust 1.71 a panic across `extern "system"` is an abort - every host
   entry is a potential DAW crash otherwise. Discipline: docs/thread-model.md.

2. Baseline gates: `.lock().unwrap()` and `error.contains(` are allowed only at
   the exact occurrences listed in scripts/checks/baseline-*.txt (path -> count).
   New occurrences fail CI. New code must use `lock_or_recover()` (poison
   recovery) and the structured error codes instead of string matching;
   the baselines exist so legacy sites can be paid down without the count
   growing back.

`#[cfg(test)]` modules are skipped (test-only fakes never run in the host).
"""
import os
import re
import sys

GUARD_MARKERS = ('guard_ffi(', 'catch_unwind', 'boundary(')
sig_re = re.compile(r'^[ ]*(?:pub )?unsafe extern "system" fn (\w+)\(', re.M)
# All paths come from this file's own location. Scan roots are relative to
# `backend/`, so the backend root must never be derived from a caller-supplied
# plugin `src/` path: that produced
# `backend/hifishifter-plugin/hifishifter-plugin/src`, `os.walk` found nothing
# and both baseline gates passed vacuously.
CHECKS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CHECKS_DIR, '..', '..'))
BACKEND_ROOT = os.path.join(REPO_ROOT, 'backend')
PLUGIN_SRC = os.path.join(BACKEND_ROOT, 'hifishifter-plugin', 'src')
BASELINE_DIR = CHECKS_DIR


def scan_roots():
    """Scan roots derived from the backend workspace members, so a new crate
    joins the baseline gates automatically instead of when someone remembers
    to extend a hand-maintained list (that gap is how the gate went red for
    months without anyone noticing the crates it never looked at).

    A member without a `src/` directory is a hard failure, never a silent
    skip: an empty walk means "no violations" and the gate would go green
    while checking nothing (exactly the bug the missing-root check below
    exists for).
    """
    with open(os.path.join(BACKEND_ROOT, 'Cargo.toml'), encoding='utf-8') as f:
        cargo = f.read()
    block = re.search(r'^members\s*=\s*\[(.*?)\]', cargo, re.M | re.S)
    if not block:
        raise SystemExit('check-ffi-guards: no `members = [...]` in backend/Cargo.toml')
    members = re.findall(r'"([^"]+)"', block.group(1))
    if not members:
        raise SystemExit('check-ffi-guards: backend/Cargo.toml lists no workspace members')
    roots = [member.replace('\\', '/').rstrip('/') + '/src' for member in members]
    missing = [root for root in roots if not os.path.isdir(os.path.join(BACKEND_ROOT, root))]
    if missing:
        raise SystemExit(f'check-ffi-guards: workspace member(s) without src/: {missing}')
    return roots


def count_braces(src, start):
    i = start
    n = len(src)
    depth = 0
    while i < n:
        c = src[i]
        if c == '/' and i + 1 < n and src[i + 1] == '/':
            j = src.find('\n', i)
            i = n if j == -1 else j + 1
            continue
        if c == '/' and i + 1 < n and src[i + 1] == '*':
            depth_c = 1
            i += 2
            while i < n and depth_c:
                if src.startswith('/*', i):
                    depth_c += 1
                    i += 2
                elif src.startswith('*/', i):
                    depth_c -= 1
                    i += 2
                else:
                    i += 1
            continue
        if c == '"':
            i += 1
            while i < n:
                if src[i] == '\\':
                    i += 2
                    continue
                if src[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if c == "'":
            m = re.match(r"'(\\.|[^\\'])'", src[i:])
            if m:
                i += m.end()
                continue
            i += 1
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError('unbalanced braces')


# 【为什么不是一句 `#[cfg(test)]\s*mod \w+`】那样只认裸 `mod`。测试模块普遍写成
# `#[cfg(test)]\n pub(crate) mod tests {`（`editor/session.rs` 就是），带可见性的那一种
# 匹配不上，整个模块体于是被当成**生产代码**扫描 —— 里面每一句断言都会被算成违规。
# 这个门禁曾经因此对着一个只出现在 `assert!` 里的 `error.contains(` 报红，而基线允许 0。
CFG_TEST_MOD = re.compile(
    r'#\[cfg\(test\)\]'                 # the attribute itself
    r'(?:\s*#\[[^\]]*\])*'              # any further attributes on the item
    r'\s*(?:pub(?:\s*\([^)]*\))?\s+)?'  # optional visibility (pub / pub(crate) / …)
    r'mod\s+\w+\s*\{'                    # the module body
)


def cfg_test_module_ranges(src):
    """Byte ranges of `#[cfg(test)] mod X { ... }` bodies."""
    ranges = []
    for m in CFG_TEST_MOD.finditer(src):
        try:
            close = count_braces(src, m.end() - 1)
        except ValueError:
            continue
        ranges.append((m.start(), close))
    return ranges


def is_test_path(rel_path):
    base = os.path.basename(rel_path)
    return base == 'tests.rs' or base.endswith('_tests.rs')


def production_src(rel_path):
    """The file's source with every `#[cfg(test)] mod …` body removed.

    Returns None for test-only files (`tests.rs` / `*_tests.rs`). The caller
    must scan the returned text: re-reading the file would put the cfg(test)
    bodies back and count test-only fakes as production violations.
    """
    if is_test_path(rel_path):
        return None
    with open(rel_path, encoding='utf-8') as f:
        src = f.read()
    # Strip cfg(test) module bodies before scanning for the patterns.
    for a, b in sorted(cfg_test_module_ranges(src), reverse=True):
        src = src[:a] + src[b:]
    return src


def baseline_gate(backend_root, pattern, baseline_file, label):
    baseline_path = os.path.join(BASELINE_DIR, baseline_file)
    baseline = {}
    if os.path.exists(baseline_path):
        with open(baseline_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                rel, count = line.rsplit(':', 1)
                baseline[rel.replace('\\', '/')] = int(count)
    current = {}
    for scan_root in scan_roots():
        root = os.path.join(backend_root, scan_root)
        if not os.path.isdir(root):
            # A missing scan root must be a hard failure, never a silent pass:
            # an empty walk means "no violations" and the gate would go green
            # while checking nothing (exactly the bug this guard used to have).
            print(f'{label} gate FAILED - scan root does not exist: {root}')
            return 1
        for dirpath, _dirs, files in os.walk(root):
            for fn in sorted(files):
                if not fn.endswith('.rs'):
                    continue
                path = os.path.join(dirpath, fn)
                rel = os.path.relpath(path, backend_root).replace('\\', '/')
                src = production_src(path)
                if src is None:
                    continue
                count = len(re.findall(pattern, src))
                if count:
                    current[rel] = count
    offenders = []
    for rel, count in sorted(current.items()):
        allowed = baseline.get(rel, 0)
        if count > allowed:
            offenders.append(f'  {rel}: {count} (baseline allows {allowed})')
    missing = [rel for rel in baseline if rel not in current]
    if offenders:
        print(f'{label} gate FAILED - new occurrences beyond baseline '
              f'({baseline_file}); use lock_or_recover()/structured codes instead:')
        print('\n'.join(offenders))
        return 1
    if missing:
        print(f'{label} gate: baseline shrank; remove these lines from '
              f'{baseline_file}: {", ".join(missing)}')
    print(f'{label} gate OK')
    return 0


def main():
    rc = ffi_guard_gate(PLUGIN_SRC)
    rc |= baseline_gate(BACKEND_ROOT, r'\.lock\(\)\.unwrap\(\)',
                        'baseline-lock-unwrap.txt', 'lock().unwrap()')
    rc |= baseline_gate(BACKEND_ROOT, r'error\.contains\(',
                        'baseline-error-contains.txt', 'error.contains()')
    return rc


def ffi_guard_gate(root):
    offenders = []
    checked = 0
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith('.rs'):
                continue
            # `X/tests.rs` / `*_tests.rs` are cfg(test)-gated modules declared
            # in their parent (`#[cfg(test)] mod tests;`); their fakes never
            # run inside the host process.
            if fn == 'tests.rs' or fn.endswith('_tests.rs'):
                continue
            path = os.path.join(dirpath, fn)
            with open(path, encoding='utf-8') as f:
                src = f.read()
            test_ranges = cfg_test_module_ranges(src)
            for m in sig_re.finditer(src):
                if any(a <= m.start() <= b for a, b in test_ranges):
                    continue
                name = m.group(1)
                depth = 1
                k = m.end()
                while depth and k < len(src):
                    if src[k] == '(':
                        depth += 1
                    elif src[k] == ')':
                        depth -= 1
                    k += 1
                brace = src.find('{', k)
                if brace == -1:
                    continue
                body_close = count_braces(src, brace)
                body = src[brace:body_close]
                checked += 1
                if not any(marker in body for marker in GUARD_MARKERS):
                    line = src.count('\n', 0, m.start()) + 1
                    offenders.append(f'{path}:{line}: fn {name}')
    if offenders:
        print(
            'FFI guard gate FAILED - these extern "system" entries have no '
            'panic guard (guard_ffi/catch_unwind/boundary); a panic there '
            'aborts the host process. See docs/thread-model.md:'
        )
        for o in offenders:
            print(f'  {o}')
        return 1
    print(f'FFI guard gate OK: {checked} extern "system" entries all guarded')
    return 0

if __name__ == '__main__':
    sys.exit(main())
