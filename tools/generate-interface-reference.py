"""Generate a source navigation inventory, not an ABI parser or runtime benchmark."""
from pathlib import Path
from datetime import date
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'juce/src'
HEADERS = {
    'ProjectModel.h': 'ProjectModel',
    'AudioEngine.h': 'AudioEngine',
    'SampleSettings.h': 'SampleSettings',
    'AccompanimentTempo.h': 'AccompanimentTempoTask',
    'backend/AnalysisService.h': 'AnalysisService',
    'backend/RenderService.h': 'RenderService',
    'backend/UtauRenderer.h': 'UtauRenderer',
    'backend/NsfHifiganRenderer.h': 'NsfHifiganRenderer',
    'backend/DiffSingerRenderer.h': 'DiffSingerRenderer',
    'backend/McpServer.h': 'McpServer',
    'backend/LiveMcpBridge.h': 'LiveMcpBridge',
}

def mask_comments(text):
    return re.sub(r'//[^\n]*|/\*[\s\S]*?\*/',
                  lambda m: re.sub(r'[^\n]', ' ', m[0]), text)

def methods(text, cls):
    clean = mask_comments(text)
    start = re.search(r'\bclass\s+' + cls + r'\b[^;{]*\{', clean).end()
    i = begin = start
    depth, parens, public = 1, 0, False
    found = []
    def record(end):
        raw = clean[begin:end].strip()
        if not public or re.match(r'(?:struct|class|enum|using|typedef)\b', raw):
            return
        raw = re.sub(r'\[\[.*?\]\]', '', raw, flags=re.S).strip()
        match = re.search(r'(~?[A-Za-z_]\w*)\s*\(', raw)
        if not match:
            return
        name = match[1]
        if name in {'function', 'decltype', 'sizeof', 'if', 'for'}:
            return
        pos = begin + len(clean[begin:end]) - len(clean[begin:end].lstrip())
        found.append(dict(name=name, line=text.count('\n', 0, pos)+1,
                          signature=' '.join(raw.split())))
    while i < len(clean) and depth:
        if depth == 1 and parens == 0:
            access = re.match(r'(public|private|protected)\s*:', clean[i:])
            if access:
                public = access[1] == 'public'
                i += access.end(); begin = i
                continue
        c = clean[i]
        if depth == 1:
            if c == '(':
                parens += 1
            elif c == ')':
                parens -= 1
            elif c == ';' and parens == 0:
                record(i); begin = i+1
            elif c == '{' and parens == 0:
                record(i)
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 1 and parens == 0:
                begin = i+1
        i += 1
    return found

def link(rel, line):
    return f'[{rel}:{line}](../juce/src/{rel}#L{line})'

def esc(text):
    return text.replace('|', '\\|').replace('\n', ' ')

def generate():
    today = date.today().isoformat()
    out = ['# 实际接口声明清单（自动生成）', '',
           f'核对日期：{today}。配套解读：[功能主入口与架构审计](AI_INTERFACE_ARCHITECTURE.md)。', '',
           '这是源文件导航，不是承诺稳定的公共 SDK。仅枚举下列 11 个边界类的公开方法，'
           '不把 private 函数、UI 回调、纯计算辅助函数或每个第三方函数算作产品接口。'
           '重载各列一行；诊断接口单独标记。源码行号为生成时位置。', '',
           '生成命令：`python tools/generate-interface-reference.py`。词法提取不是 C++ 编译器，'
           '不能用条目数证明所有业务路径正确；新增边界类时需更新脚本 HEADERS。', '']
    inventory = {}
    for rel, cls in HEADERS.items():
        rows = methods((SRC/rel).read_text(encoding='utf-8-sig'), cls)
        inventory[cls] = rows
        out += [f'## {cls}（{len(rows)} 个声明）', '', '| 实际签名 | 类型 | 定义入口 |', '| --- | --- | --- |']
        for row in rows:
            kind = '诊断' if row['name'].startswith('diagnostic') else '生命周期' if row['name'].lstrip('~') == cls else '业务／适配'
            out.append(f"| `{esc(row['signature'])}` | {kind} | {link(rel, row['line'])} |")
        out.append('')
    out += ['## MCP 命令目录', '',
            'Headless 命令来自 `McpServer::diagnosticTools`；Live 复用它并叠加窗口命令，'
            '并非两套完整独立业务实现。Live 写操作另有 expected_revision、会话和草稿提交约束。'
            '字段类型／必填项以链接处 schema 为准；不是所有 GUI 功能都有 MCP 命令。', '',
            '| 命令 | 注册位置 | 源码描述 |', '| --- | --- | --- |']
    commands = {}
    for rel, prefix in [('backend/McpServer.cpp', 'makeTool'), ('MainComponentMcp.cpp', 'add')]:
        text = (SRC/rel).read_text(encoding='utf-8-sig')
        for m in re.finditer(r'\b'+prefix+r'\(\s*"([\w_]+)"\s*,\s*"([^"\n]*)"', text):
            name, description = m[1], m[2]
            line = text.count('\n', 0, m.start())+1
            commands[name] = dict(file=rel, line=line, description=description)
        if prefix == 'add':
            line = next(i for i, s in enumerate(text.splitlines(), 1) if 'for(const auto* name:{"hamood_preview"' in s)
            for name, desc in [('hamood_preview', '只读预览和声方案'), ('hamood_generate', '生成独立和声轨道，一次撤销')]:
                commands[name] = dict(file=rel, line=line, description=desc)
    for name, row in commands.items():
        out.append(f"| `{name}` | {link(row['file'],row['line'])} | {esc(row['description'])} |")
    out += ['', '## 启动参数字面量', '',
            '仅来自 Main.cpp 的 `--…` 字符串去重；包括诊断、启动选项和帮助文字，'
            '不代表全是用户功能，也不等同于测试已执行。参数说明见原有 [命令索引](AI_SOURCE_INDEX.md)。', '',
            '| 参数 | 首次出现位置 |', '| --- | --- |']
    main = (SRC/'Main.cpp').read_text(encoding='utf-8-sig')
    flags = {}
    for m in re.finditer(r'--[a-z][a-z0-9-]*', main):
        flags.setdefault(m[0], main.count('\n',0,m.start())+1)
    for flag, line in flags.items():
        out.append(f'| `{flag}` | {link("Main.cpp",line)} |')
    files = sorted(p for p in SRC.rglob('*') if p.suffix in {'.h','.cpp','.inc'} and 'tests' not in p.relative_to(SRC).parts and 'Smoke' not in p.name)
    metrics = []
    for p in files:
        data = p.read_bytes()
        metrics.append(dict(file=p.relative_to(ROOT).as_posix(), lines=len(data.splitlines()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    out += ['', '## 范围与静态体量', '',
            f'扫描自有 `juce/src` 非测试 `.h/.cpp/.inc`：{len(metrics)} 个文件，'
            f'{sum(r["lines"] for r in metrics):,} 行（包括注释和空行）。不含 juce/third_party、tests 和名称带 Smoke 的包含单元。', '',
            '| 文件 | 行数 |', '| --- | --- |']
    for row in sorted(metrics, key=lambda r:r['lines'], reverse=True)[:15]:
        out.append(f'| [{row["file"]}](../{row["file"]}) | {row["lines"]:,} |')
    out += ['', '完整扫描指纹及接口条目：[interface-audit-inventory.json](interface-audit-inventory.json)。'
            '行数用于导航和识别职责集中，不是性能或质量评分。', '']
    (ROOT/'docs/AI_API_REFERENCE.md').write_text('\n'.join(out), encoding='utf-8')
    data = dict(date=today, scope='source-navigation-not-runtime-benchmark', interfaces=inventory, mcp=commands, main_flags=flags, source_files=metrics)
    (ROOT/'docs/interface-audit-inventory.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(classes=len(inventory), declarations=sum(map(len,inventory.values())), project_model=len(inventory['ProjectModel']), mcp=len(commands), main_flags=len(flags), source_files=len(metrics), source_lines=sum(r['lines'] for r in metrics))))

if __name__ == '__main__':
    generate()
