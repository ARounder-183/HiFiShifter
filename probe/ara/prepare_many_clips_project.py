"""中文一次性只读原工程提取：仅写全新隔离副本，保留母轨/三个子轨与素材路径，剔除第三方FX。"""
import pathlib
import re
import struct
import sys


def chunk_end(lines, begin):
    """按RPP块边界提取，不执行工程里的任何脚本。"""
    depth = 0
    for i in range(begin, len(lines)):
        text = lines[i].strip()
        if text.startswith('<'):
            depth += 1
        elif text == '>':
            depth -= 1
        if depth == 0:
            return i + 1
    raise ValueError('unbalanced RPP')


def without_fx(lines):
    """只从副本去掉FX块，避免加载无关插件/激活弹窗；不改变item、源或几何。"""
    out = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith(('<FXCHAIN', '<MASTERFXLIST', '<TAKEFX')):
            i = chunk_end(lines, i)
        else:
            out.append(lines[i])
            i += 1
    return out


original = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
if destination.exists():
    raise ValueError('never overwrite an existing reproduction project')
lines = original.read_text(encoding='utf-8-sig').splitlines(keepends=True)
starts = [i for i, line in enumerate(lines) if re.match(r'^  <TRACK\s', line)]
tracks = [lines[start:chunk_end(lines, start)] for start in starts]
names = [next((line.strip()[5:].strip('"') for line in track if line.startswith('    NAME ')), '') for track in tracks]
index = names.index(sys.argv[3])
selected = tracks[index:index + 4]
assert len(selected) == 4, 'expected parent and three child tracks'
result = without_fx(lines[:starts[0]]) + sum((without_fx(track) for track in selected), []) + ['>\n']
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(''.join(result), encoding='utf-8')
sources = sorted(set(re.findall(r'^\s*FILE "([^\"]+)"', ''.join(''.join(track) for track in selected), re.MULTILINE)))
print(f'COPIED tracks={names[index:index+4]} clips={sum(line.strip().startswith("<ITEM") for track in selected for line in track)} sources={len(sources)}')
for source in sources:
    path = pathlib.Path(source)
    if not path.exists():
        print('MISSING_BASENAME=' + path.name)
        continue
    with path.open('rb') as f:
        header = f.read(128)
    if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
        p = header.find(b'fmt ')
        if p >= 0 and p + 24 <= len(header):
            fmt, channels, rate = struct.unpack_from('<HHI', header, p + 8)
            print(f'WAVE_BASENAME={path.name} format={fmt} channels={channels} rate={rate}')
