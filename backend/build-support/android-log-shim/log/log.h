/* Header shim that lets fdk-aac compile against the Android NDK.
 *
 * [Why this file exists] `fdk-aac-sys` 0.5.0 compiles the upstream FDK-AAC sources, and
 * `aac/libSBRdec/src/lpp_tran.cpp` starts like this:
 *
 *     #ifdef __ANDROID__
 *     #include "log/log.h"
 *     #endif
 *
 * `log/log.h` is AOSP's **internal** liblog header (system/core/log/include/log/log.h).
 * It is not part of the NDK -- the NDK ships <android/log.h> only -- and `__ANDROID__` is
 * predefined by clang for the android triple, so the include is always taken when
 * cross-compiling with the NDK and the build dies with
 *
 *     fatal error: 'log/log.h' file not found
 *
 * The only thing FDK takes from that header is `android_errorWriteLog` (lpp_tran.cpp
 * lines 342 and 940), which AOSP uses to report a sanitizer finding for a CVE-relevant
 * out-of-bounds write to the framework. It is not NDK API, no NDK header declares it, and
 * liblog is not in the link line, so there is nothing to link against either.
 *
 * This shim supplies exactly that one symbol as a no-op. The code it silences only
 * *reports*; it does not participate in decoding, so audio output is unaffected.
 *
 * [Why a shim instead of a fork] A `[patch.crates-io]` fork would vendor the whole FDK
 * tree into this repository. The build already solves the same class of problem the same
 * way for SoundTouch's `afxres.h` (see the RC shim in
 * backend/hifishifter-kernel/build.rs), and the include path is added **only for Android
 * targets** (scripts/env/android-env.ps1, scripts/build/android.sh via CPATH), so desktop
 * builds never see this directory.
 */
#ifndef HIFISHIFTER_ANDROID_LOG_LOG_SHIM_H
#define HIFISHIFTER_ANDROID_LOG_LOG_SHIM_H

/* Signature copied from AOSP system/core/log/include/log/log.h. */
static inline int android_errorWriteLog(int tag, const char* msg) {
    (void)tag;
    (void)msg;
    return 0;
}

#endif /* HIFISHIFTER_ANDROID_LOG_LOG_SHIM_H */
