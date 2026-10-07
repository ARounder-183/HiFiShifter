"""中文一次性只读转储检查：解析异常上下文/模块及候选返回槽，不执行目标进程。"""
import pathlib
import struct
import sys
import uuid


def inspect(path: pathlib.Path) -> None:
    """仅检查传入的单个minidump，候选栈地址不是已完成的原生unwind。"""
    data = path.read_bytes()
    def unpack(fmt, offset):
        return struct.unpack_from('<' + fmt, data, offset)
    def string(rva):
        size, = unpack('I', rva)
        return data[rva + 4:rva + 4 + size].decode('utf-16le', errors='replace')
    assert data[:4] == b'MDMP', 'not a minidump'
    count, directory = unpack('II', 8)
    streams = {unpack('III', directory + 12 * i)[0]: unpack('III', directory + 12 * i)[1:] for i in range(count)}
    _, ex = streams[6]
    thread, = unpack('I', ex)
    code, = unpack('I', ex + 8)
    address, = unpack('Q', ex + 24)
    parameters, = unpack('I', ex + 32)
    details = unpack('Q' * parameters, ex + 40)
    ctx_size, ctx = unpack('II', ex + 160)
    registers = dict(zip(['rax', 'rcx', 'rdx', 'rbx', 'rsp', 'rbp', 'rsi', 'rdi', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15', 'rip'], unpack('Q' * 17, ctx + 120)))
    print(f'DUMP={path.name} thread={thread} exception={code:#x} address={address:#x} details={details}')
    print('REGS ' + ' '.join(f'{k}={v:#x}' for k, v in registers.items()))
    def read(address, size):
        if 5 not in streams:
            return None
        _, start = streams[5]
        ranges, = unpack('I', start)
        for i in range(ranges):
            base, length, rva = unpack('QII', start + 4 + i * 16)
            if base <= address and address + size <= base + length:
                return data[rva + address - base:rva + address - base + size]
        return None
    # 已反汇编的故障循环：集合首地址/字节数，不能把这些未知结构域冒称SDK类型。
    head = read(registers['r13'] + 0x190, 16)
    if head:
        pointer, allocated, length = struct.unpack('<QII', head)
        entries = read(pointer, length) if length <= 32768 else None
        if entries:
            slots = struct.unpack('<' + 'Q' * (len(entries) // 8), entries)
            print(f'FAULT_COLLECTION bytes={length} slots={len(slots)} null_indices={[i for i,v in enumerate(slots) if not v]}')
        else:
            print(f'FAULT_COLLECTION bytes={length} contents_not_captured')
    filename = read(registers['r12'], 256)
    if filename:
        text = filename.split(b'\0')[0].decode('utf8', errors='replace')
        print('FAULT_SOURCE_BASENAME=' + text.replace('\\','/').rsplit('/',1)[-1])
    _, modules_stream = streams[4]
    n, = unpack('I', modules_stream)
    modules = []
    for i in range(n):
        p = modules_stream + 4 + 108 * i
        base, size, checksum, stamp, name_rva = unpack('QIIII', p)
        name = string(name_rva)
        cv_size, cv = unpack('II', p + 76)
        identity = ''
        if cv_size >= 24 and data[cv:cv + 4] == b'RSDS':
            age, = unpack('I', cv + 20)
            identity = f'{uuid.UUID(bytes_le=data[cv + 4:cv + 20])} age={age}'
        modules.append((base, base + size, name))
        if 'hifishifter' in name.lower() or name.lower().endswith('reaper.exe'):
            print(f'MODULE {name} base={base:#x} size={size:#x} codeview={identity}')
    _, threads = streams[3]
    n, = unpack('I', threads)
    for i in range(n):
        p = threads + 4 + 48 * i
        tid, = unpack('I', p)
        if tid != thread:
            continue
        stack_base, stack_size, stack_rva = unpack('QII', p + 24)
        first = max(0, registers['rsp'] - stack_base)
        print(f'STACK base={stack_base:#x} size={stack_size:#x}')
        shown = 0
        for offset in range(first, min(stack_size, first + 16384) - 7, 8):
            candidate, = unpack('Q', stack_rva + offset)
            for lo, hi, name in modules:
                if lo <= candidate < hi and ('hifishifter' in name.lower() or name.lower().endswith('reaper.exe')):
                    print(f'SLOT rsp+{offset-first:#x} {name.rsplit(chr(92),1)[-1]} RVA={candidate-lo:#x}')
                    shown += 1
            if shown >= 60:
                break


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        inspect(pathlib.Path(arg))
