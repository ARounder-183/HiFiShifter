//////////////////////////////////////////////////////////////////////////////
///
/// SoundTouch C wrapper implementation.
/// Wraps the SoundTouch C++ class in a C-compatible API that matches
/// the Windows SoundTouchDLL API.
///
/// Based on SoundTouch library by Olli Parviainen.
/// SoundTouch WWW: http://www.surina.net/soundtouch
///
////////////////////////////////////////////////////////////////////////////////

#include "SoundTouch.h"
#include "soundtouch_c.h"

#include <cstdlib>
#include <cstring>

SOUNDTOUCH_HANDLE soundtouch_createInstance(void)
{
    soundtouch::SoundTouch *p = new soundtouch::SoundTouch();
    return static_cast<SOUNDTOUCH_HANDLE>(p);
}

void soundtouch_destroyInstance(SOUNDTOUCH_HANDLE h)
{
    soundtouch::SoundTouch *p = static_cast<soundtouch::SoundTouch *>(h);
    delete p;
}

const char *soundtouch_getVersionString(void)
{
    return soundtouch::SoundTouch::getVersionString();
}

void soundtouch_setRate(SOUNDTOUCH_HANDLE h, float newRate)
{
    soundtouch::SoundTouch *p = static_cast<soundtouch::SoundTouch *>(h);
    p->setRate(static_cast<double>(newRate));
}

void soundtouch_setTempo(SOUNDTOUCH_HANDLE h, float newTempo)
{
    soundtouch::SoundTouch *p = static_cast<soundtouch::SoundTouch *>(h);
    p->setTempo(static_cast<double>(newTempo));
}

void soundtouch_setPitch(SOUNDTOUCH_HANDLE h, float newPitch)
{
    soundtouch::SoundTouch *p = static_cast<soundtouch::SoundTouch *>(h);
    p->setPitch(static_cast<double>(newPitch));
}

int soundtouch_setChannels(SOUNDTOUCH_HANDLE h, unsigned int numChannels)
{
    if (numChannels == 0) return 0;
    soundtouch::SoundTouch *p = static_cast<soundtouch::SoundTouch *>(h);
    p->setChannels(numChannels);
    return 1;
}

int soundtouch_setSampleRate(SOUNDTOUCH_HANDLE h, unsigned int srate)
{
    if (srate == 0) return 0;
    soundtouch::SoundTouch *p = static_cast<soundtouch::SoundTouch *>(h);
    p->setSampleRate(srate);
    return 1;
}

int soundtouch_flush(SOUNDTOUCH_HANDLE h)
{
    soundtouch::SoundTouch *p = static_cast<soundtouch::SoundTouch *>(h);
    p->flush();
    return 1;
}

int soundtouch_putSamples(SOUNDTOUCH_HANDLE h,
        const float *samples,
        unsigned int numSamples)
{
    if (numSamples == 0) return 1;
    soundtouch::SoundTouch *p = static_cast<soundtouch::SoundTouch *>(h);
    p->putSamples(samples, numSamples);
    return 1;
}

void soundtouch_clear(SOUNDTOUCH_HANDLE h)
{
    soundtouch::SoundTouch *p = static_cast<soundtouch::SoundTouch *>(h);
    p->clear();
}

unsigned int soundtouch_numSamples(SOUNDTOUCH_HANDLE h)
{
    soundtouch::SoundTouch *p = static_cast<soundtouch::SoundTouch *>(h);
    return p->numSamples();
}

unsigned int soundtouch_receiveSamples(SOUNDTOUCH_HANDLE h,
        float *outBuffer,
        unsigned int maxSamples)
{
    soundtouch::SoundTouch *p = static_cast<soundtouch::SoundTouch *>(h);
    return p->receiveSamples(
        static_cast<soundtouch::SAMPLETYPE *>(outBuffer),
        maxSamples);
}

int soundtouch_isEmpty(SOUNDTOUCH_HANDLE h)
{
    soundtouch::SoundTouch *p = static_cast<soundtouch::SoundTouch *>(h);
    return p->isEmpty() ? 1 : 0;
}
