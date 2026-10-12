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
BASELINE_DIR = os.path.join(os.path.dirname(__file__))
SCAN_ROOTS = ['hifishifter-plugin/src', 'hifishifter-kernel/src', 'src-tauri/src']


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


def cfg_test_module_ranges(src):
    """Byte ranges of `#[cfg(test)] mod X { ... }` bodies."""
    ranges = []
    for m in re.finditer(r'#\[cfg\(test\)\]\s*\n\s*mod \w+[^{]*\{', src):
        try:
            close = count_braces(src, m.end() - 1)
        except ValueError:
            continue
        ranges.append((m.start(), close))
    return ranges


def is_test_path(rel_path):
    base = os.path.basename(rel_path)
    return base == 'tests.rs' or base.endswith('_tests.rs')


def is_production_src(rel_path):
    """True when the file is production source (not a cfg(test)-gated module)."""
    if is_test_path(rel_path):
        return False
    with open(rel_path, encoding='utf-8') as f:
        src = f.read()
    ranges = cfg_test_module_ranges(src)
    # Strip cfg(test) module bodies before scanning for the patterns.
    for a, b in sorted(ranges, reverse=True):
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
    for scan_root in SCAN_ROOTS:
        root = os.path.join(backend_root, scan_root)
        for dirpath, _dirs, files in os.walk(root):
            for fn in sorted(files):
                if not fn.endswith('.rs'):
                    continue
                path = os.path.join(dirpath, fn)
                rel = os.path.relpath(path, backend_root).replace('\\', '/')
                if not is_production_src(path):
                    continue
                with open(path, encoding='utf-8') as f:
                    src = f.read()
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


def main(root):
    rc = ffi_guard_gate(root)
    backend_root = os.path.dirname(root.rstrip('/\\'))
    rc |= baseline_gate(backend_root, r'\.lock\(\)\.unwrap\(\)',
                        'baseline-lock-unwrap.txt', 'lock().unwrap()')
    rc |= baseline_gate(backend_root, r'error\.contains\(',
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
    plugin_src = os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'hifishifter-plugin', 'src')
    sys.exit(main(plugin_src))
