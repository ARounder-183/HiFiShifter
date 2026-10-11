#include <stddef.h>
#include <stdint.h>
#if defined(ARA2_CLAP_PROBE_MAIN)
#include <stdio.h>
#endif

#include "ARACLAP.h"

uint32_t ara2_clap_version_major(void) { return CLAP_VERSION_MAJOR; }
uint32_t ara2_clap_version_minor(void) { return CLAP_VERSION_MINOR; }
uint32_t ara2_clap_version_revision(void) { return CLAP_VERSION_REVISION; }

const char *ara2_clap_ara_factory_id(void) { return CLAP_EXT_ARA_FACTORY; }
const char *ara2_clap_ara_factory_compat_id(void) { return CLAP_EXT_ARA_FACTORY_COMPAT; }
const char *ara2_clap_ara_plugin_extension_id(void) { return CLAP_EXT_ARA_PLUGINEXTENSION; }
const char *ara2_clap_ara_plugin_extension_compat_id(void) { return CLAP_EXT_ARA_PLUGINEXTENSION_COMPAT; }
const char *ara2_clap_ara_supported_feature(void) { return CLAP_PLUGIN_FEATURE_ARA_SUPPORTED; }
const char *ara2_clap_ara_required_feature(void) { return CLAP_PLUGIN_FEATURE_ARA_REQUIRED; }

size_t ara2_sizeof_clap_ara_factory(void) { return sizeof(clap_ara_factory_t); }
size_t ara2_alignof_clap_ara_factory(void) { return _Alignof(clap_ara_factory_t); }
size_t ara2_offset_clap_ara_factory_count(void) { return offsetof(clap_ara_factory_t, get_factory_count); }
size_t ara2_offset_clap_ara_factory_factory(void) { return offsetof(clap_ara_factory_t, get_ara_factory); }
size_t ara2_offset_clap_ara_factory_plugin_id(void) { return offsetof(clap_ara_factory_t, get_plugin_id); }

size_t ara2_sizeof_clap_ara_plugin_extension(void) { return sizeof(clap_ara_plugin_extension_t); }
size_t ara2_alignof_clap_ara_plugin_extension(void) { return _Alignof(clap_ara_plugin_extension_t); }
size_t ara2_offset_clap_ara_plugin_extension_factory(void) { return offsetof(clap_ara_plugin_extension_t, get_factory); }
size_t ara2_offset_clap_ara_plugin_extension_bind(void) { return offsetof(clap_ara_plugin_extension_t, bind_to_document_controller); }

size_t ara2_sizeof_clap_plugin_entry(void) { return sizeof(clap_plugin_entry_t); }
size_t ara2_sizeof_clap_plugin(void) { return sizeof(clap_plugin_t); }
size_t ara2_sizeof_clap_plugin_factory(void) { return sizeof(clap_plugin_factory_t); }

#if defined(ARA2_CLAP_PROBE_MAIN)
int main(void) {
    printf("{\"clap_ara_factory\":{\"size\":%zu,\"align\":%zu,\"offsets\":[%zu,%zu,%zu]},"
           "\"clap_ara_plugin_extension\":{\"size\":%zu,\"align\":%zu,\"offsets\":[%zu,%zu]},"
           "\"clap_plugin_entry\":{\"size\":%zu},\"clap_plugin\":{\"size\":%zu},"
           "\"clap_plugin_factory\":{\"size\":%zu}}\n",
           sizeof(clap_ara_factory_t), _Alignof(clap_ara_factory_t),
           offsetof(clap_ara_factory_t, get_factory_count), offsetof(clap_ara_factory_t, get_ara_factory),
           offsetof(clap_ara_factory_t, get_plugin_id), sizeof(clap_ara_plugin_extension_t),
           _Alignof(clap_ara_plugin_extension_t), offsetof(clap_ara_plugin_extension_t, get_factory),
           offsetof(clap_ara_plugin_extension_t, bind_to_document_controller), sizeof(clap_plugin_entry_t),
           sizeof(clap_plugin_t), sizeof(clap_plugin_factory_t));
    return 0;
}
#endif
