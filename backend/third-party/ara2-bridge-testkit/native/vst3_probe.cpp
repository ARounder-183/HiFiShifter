#include "ara_vst3_shim.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>

static void print_id(Ara2Vst3InterfaceKind kind)
{
    Ara2Vst3InterfaceId id {};
    if (ara2_vst3_interface_id(kind, &id) != ARA2_VST3_OK)
        std::abort();
    std::printf("[%u,%u,%u,%u]", id.words[0], id.words[1], id.words[2], id.words[3]);
}

int main()
{
    std::printf("{\"category\":\"%s\",\"iids\":{\"unknown\":", ara2_vst3_main_factory_category());
    print_id(ARA2_VST3_INTERFACE_UNKNOWN);
    std::printf(",\"main_factory\":");
    print_id(ARA2_VST3_INTERFACE_MAIN_FACTORY);
    std::printf(",\"plugin_entry\":");
    print_id(ARA2_VST3_INTERFACE_PLUGIN_ENTRY);
    std::printf(",\"plugin_entry_2\":");
    print_id(ARA2_VST3_INTERFACE_PLUGIN_ENTRY_2);
    std::printf(
        "},\"layouts\":{"
        "\"interface_id\":{\"size\":%zu,\"align\":%zu},"
        "\"main_callbacks\":{\"size\":%zu,\"align\":%zu},"
        "\"entry_callbacks\":{\"size\":%zu,\"align\":%zu}},"
        "\"exception_result\":%d}\n",
        sizeof(Ara2Vst3InterfaceId), alignof(Ara2Vst3InterfaceId),
        sizeof(Ara2Vst3MainFactoryCallbacks), alignof(Ara2Vst3MainFactoryCallbacks),
        sizeof(Ara2Vst3PluginEntryCallbacks), alignof(Ara2Vst3PluginEntryCallbacks),
        ara2_vst3_probe_exception_boundary(1));
    return 0;
}
