#include "ara_au_shim.h"
#include "ARAAudioUnit.h"

#include <cstddef>
#include <cstdio>

int main()
{
    std::printf(
        "{\"constants\":{"
        "\"magic\":%u,\"factory\":%u,\"binding\":%u,\"binding_with_roles\":%u},"
        "\"factory\":{\"size\":%zu,\"align\":%zu,\"offsets\":[%zu,%zu]},"
        "\"binding\":{\"size\":%zu,\"align\":%zu,\"offsets\":[%zu,%zu,%zu,%zu,%zu]}}\n",
        static_cast<unsigned>(ARA::kARAAudioUnitMagic),
        static_cast<unsigned>(ARA::kAudioUnitProperty_ARAFactory),
        static_cast<unsigned>(ARA::kAudioUnitProperty_ARAPlugInExtensionBinding),
        static_cast<unsigned>(ARA::kAudioUnitProperty_ARAPlugInExtensionBindingWithRoles),
        sizeof(ARA::ARAAudioUnitFactory), alignof(ARA::ARAAudioUnitFactory),
        offsetof(ARA::ARAAudioUnitFactory, inOutMagicNumber),
        offsetof(ARA::ARAAudioUnitFactory, outFactory),
        sizeof(ARA::ARAAudioUnitPlugInExtensionBinding),
        alignof(ARA::ARAAudioUnitPlugInExtensionBinding),
        offsetof(ARA::ARAAudioUnitPlugInExtensionBinding, inOutMagicNumber),
        offsetof(ARA::ARAAudioUnitPlugInExtensionBinding, inDocumentControllerRef),
        offsetof(ARA::ARAAudioUnitPlugInExtensionBinding, outPlugInExtension),
        offsetof(ARA::ARAAudioUnitPlugInExtensionBinding, knownRoles),
        offsetof(ARA::ARAAudioUnitPlugInExtensionBinding, assignedRoles));
    return 0;
}
