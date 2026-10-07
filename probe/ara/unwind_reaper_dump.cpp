// 中文一次性离线转储展开：只读dump/PE与本地符号，绝不启动目标REAPER或设置系统调试器。
#include <windows.h>
#include <dbghelp.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct Image {DWORD64 base;DWORD size;std::wstring path;std::vector<char> bytes;std::vector<RUNTIME_FUNCTION> functions;};
static std::vector<char> dump;
static std::vector<Image> images;
static MINIDUMP_MEMORY_LIST* memory;

// 中文：原文件读取，不LoadLibrary目标模块，不执行其入口。
static std::vector<char> file(const std::wstring& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);if (!f) return {};
    auto size=f.tellg();std::vector<char> bytes(static_cast<size_t>(size));f.seekg(0);f.read(bytes.data(),size);return bytes;
}
static void* stream(ULONG type) {PMINIDUMP_DIRECTORY dir=nullptr;void* p=nullptr;ULONG n=0;return MiniDumpReadDumpStream(dump.data(),type,&dir,&p,&n)?p:nullptr;}
static Image* image(DWORD64 address) {for (auto& m:images) if (address>=m.base&&address-m.base<m.size) return &m;return nullptr;}
static const char* image_bytes(Image& m,DWORD rva,DWORD size) {
    if (m.bytes.size()<sizeof(IMAGE_DOS_HEADER)) return nullptr;
    auto dos=reinterpret_cast<const IMAGE_DOS_HEADER*>(m.bytes.data());
    if (dos->e_lfanew<0||size_t(dos->e_lfanew)+sizeof(IMAGE_NT_HEADERS64)>m.bytes.size()) return nullptr;
    auto nt=reinterpret_cast<const IMAGE_NT_HEADERS64*>(m.bytes.data()+dos->e_lfanew);
    if (rva<nt->OptionalHeader.SizeOfHeaders&&size_t(rva)+size<=m.bytes.size()) return m.bytes.data()+rva;
    auto section=IMAGE_FIRST_SECTION(nt);
    for (unsigned i=0;i<nt->FileHeader.NumberOfSections;i++,section++) {
        if (rva>=section->VirtualAddress&&uint64_t(rva)+size<=uint64_t(section->VirtualAddress)+section->SizeOfRawData) {
            auto offset=uint64_t(section->PointerToRawData)+rva-section->VirtualAddress;
            if (offset+size<=m.bytes.size()) return m.bytes.data()+offset;
        }
    }return nullptr;
}
// 中文：首先使用故障时刻的内存；只有缺失的代码/展开表才从相同模块PE读。
static BOOL CALLBACK read_memory(HANDLE,DWORD64 address,PVOID buffer,DWORD size,LPDWORD actual) {
    if (memory) for (ULONG i=0;i<memory->NumberOfMemoryRanges;i++) {
        auto& range=memory->MemoryRanges[i];
        if (address>=range.StartOfMemoryRange&&address-range.StartOfMemoryRange+size<=range.Memory.DataSize) {
            memcpy(buffer,dump.data()+range.Memory.Rva+address-range.StartOfMemoryRange,size);*actual=size;return TRUE;
        }
    }
    if (auto m=image(address)) if (auto p=image_bytes(*m,DWORD(address-m->base),size)) {memcpy(buffer,p,size);*actual=size;return TRUE;}
    *actual=0;return FALSE;
}
static DWORD64 CALLBACK module_base(HANDLE,DWORD64 address) {auto m=image(address);return m?m->base:0;}
static PVOID CALLBACK function_table(HANDLE,DWORD64 address) {
    auto m=image(address);if (!m) return nullptr;auto rva=address-m->base;
    for (auto& fn:m->functions) if (rva>=fn.BeginAddress&&rva<fn.EndAddress) return &fn;return nullptr;
}
// 中文：仅展开异常线程，每层给模块/RVA；有本地符号时补名字，不网络下载符号。
int wmain(int argc,wchar_t** argv) {
    if (argc!=3) return 2;dump=file(argv[1]);if (dump.empty()) return 3;
    auto ex=static_cast<MINIDUMP_EXCEPTION_STREAM*>(stream(ExceptionStream));
    auto list=static_cast<MINIDUMP_MODULE_LIST*>(stream(ModuleListStream));
    memory=static_cast<MINIDUMP_MEMORY_LIST*>(stream(MemoryListStream));if (!ex||!list) return 4;
    HANDLE session=GetCurrentProcess();SymSetOptions(SYMOPT_DEFERRED_LOADS|SYMOPT_UNDNAME|SYMOPT_FAIL_CRITICAL_ERRORS);
    SymInitializeW(session,argv[2],FALSE);
    for (ULONG i=0;i<list->NumberOfModules;i++) {
        auto& entry=list->Modules[i];auto name=reinterpret_cast<MINIDUMP_STRING*>(dump.data()+entry.ModuleNameRva);
        Image m{entry.BaseOfImage,entry.SizeOfImage,std::wstring(name->Buffer,name->Length/2),{}, {}};m.bytes=file(m.path);
        if (!m.bytes.empty()) {
            auto dos=reinterpret_cast<IMAGE_DOS_HEADER*>(m.bytes.data());auto nt=reinterpret_cast<IMAGE_NT_HEADERS64*>(m.bytes.data()+dos->e_lfanew);
            auto table=nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
            if (auto p=image_bytes(m,table.VirtualAddress,table.Size)) {auto first=reinterpret_cast<const RUNTIME_FUNCTION*>(p);m.functions.assign(first,first+table.Size/sizeof(RUNTIME_FUNCTION));}
            SymLoadModuleExW(session,nullptr,m.path.c_str(),nullptr,m.base,m.size,nullptr,0);
        }images.push_back(std::move(m));
    }
    CONTEXT context{};memcpy(&context,dump.data()+ex->ThreadContext.Rva,(std::min)(size_t(ex->ThreadContext.DataSize),sizeof(context)));
    STACKFRAME64 frame{};frame.AddrPC.Offset=context.Rip;frame.AddrStack.Offset=context.Rsp;frame.AddrFrame.Offset=context.Rbp;
    frame.AddrPC.Mode=frame.AddrStack.Mode=frame.AddrFrame.Mode=AddrModeFlat;
    for (unsigned i=0;i<96;i++) {
        if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64,session,nullptr,&frame,&context,read_memory,function_table,module_base,nullptr)||!frame.AddrPC.Offset) break;
        auto m=image(frame.AddrPC.Offset);if (m) {wprintf(L"FRAME %u %s RVA=0x%llx SP=0x%llx",i,m->path.c_str(),frame.AddrPC.Offset-m->base,frame.AddrStack.Offset);
            char info[sizeof(SYMBOL_INFO)+1024]{};auto symbol=reinterpret_cast<SYMBOL_INFO*>(info);symbol->SizeOfStruct=sizeof(SYMBOL_INFO);symbol->MaxNameLen=1023;DWORD64 delta=0;
            if (SymFromAddr(session,frame.AddrPC.Offset,&delta,symbol)) printf(" %s+0x%llx",symbol->Name,delta);wprintf(L"\n");}
        else wprintf(L"FRAME %u unknown=0x%llx\n",i,frame.AddrPC.Offset);
    }SymCleanup(session);return 0;
}
