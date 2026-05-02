//////////////////////////////////////////////////////////////////////////////
///
/// SoundTouch C wrapper - provides a C-compatible API for SoundTouch library.
/// This is the macOS/Linux counterpart of the Windows SoundTouchDLL API.
///
/// Based on SoundTouch DLL wrapper header by Olli Parviainen.
/// SoundTouch WWW: http://www.surina.net/soundtouch
///
////////////////////////////////////////////////////////////////////////////////

#ifndef _SoundTouchC_h_
#define _SoundTouchC_h_

#ifdef __cplusplus
extern "C" {
#endif

typedef void * SOUNDTOUCH_HANDLE;

/// Create a new instance of SoundTouch processor.
SOUNDTOUCH_HANDLE soundtouch_createInstance(void);

/// Destroys a SoundTouch processor instance.
void soundtouch_destroyInstance(SOUNDTOUCH_HANDLE h);

/// Get SoundTouch library version string
const char *soundtouch_getVersionString(void);

/// Sets new rate control value. Normal rate = 1.0
void soundtouch_setRate(SOUNDTOUCH_HANDLE h, float newRate);

/// Sets new tempo control value. Normal tempo = 1.0
void soundtouch_setTempo(SOUNDTOUCH_HANDLE h, float newTempo);

/// Sets new pitch control value. Original pitch = 1.0
void soundtouch_setPitch(SOUNDTOUCH_HANDLE h, float newPitch);

/// Sets the number of channels, 1 = mono, 2 = stereo
int soundtouch_setChannels(SOUNDTOUCH_HANDLE h, unsigned int numChannels);

/// Sets sample rate.
int soundtouch_setSampleRate(SOUNDTOUCH_HANDLE h, unsigned int srate);

/// Flushes the last samples from the processing pipeline to the output.
int soundtouch_flush(SOUNDTOUCH_HANDLE h);

/// Adds 'numSamples' pcs of samples into the input of the object.
int soundtouch_putSamples(SOUNDTOUCH_HANDLE h,
        const float *samples,
        unsigned int numSamples);

/// Clears all the samples in the object's output and internal processing buffers.
void soundtouch_clear(SOUNDTOUCH_HANDLE h);

/// Returns number of samples currently available.
unsigned int soundtouch_numSamples(SOUNDTOUCH_HANDLE h);

/// Receive processed samples from the object.
unsigned int soundtouch_receiveSamples(SOUNDTOUCH_HANDLE h,
        float *outBuffer,
        unsigned int maxSamples);

/// Returns nonzero if there aren't any samples available for outputting.
int soundtouch_isEmpty(SOUNDTOUCH_HANDLE h);

#ifdef __cplusplus
}
#endif

#endif  // _SoundTouchC_h_
