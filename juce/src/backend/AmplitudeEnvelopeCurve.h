#pragma once

#include <cmath>
#include <optional>
#include "TailFadeSettings.h"

namespace hachi::backend
{
// Invert monotone Bezier time before evaluating fade progress. Bisection also
// handles vertical tangents and coincident control points without division by zero.
template<class Settings> double tailFadeBezierProgress(double progress,const Settings& s)
{
    const auto cubic=[](double t,double a,double b){const auto r=1-t;return 3*r*r*t*a+3*r*t*t*b+t*t*t;};
    if(progress<=0)return 0;if(progress>=1)return 1;
    double lo=0,hi=1;
    for(int i=0;i<24;++i){const auto t=(lo+hi)*.5;if(cubic(t,s.control1Time,s.control2Time)<progress)lo=t;else hi=t;}
    return cubic((lo+hi)*.5,s.control1Progress,s.control2Progress);
}

// How a loudness envelope runs from one point to the next.
//
// Straight in dB is how every envelope here has always been read, and stays
// the default.  -60 dB is silence, so a ramp to or from it spends most of its
// length all but inaudible and does its real rising or falling at one end: a
// 15 ms rise is below a tenth of full level for its first 10 ms.  Straight in
// amplitude is how UTAU and OpenUtau read an envelope, and what the length of
// a ramp means to anyone who tunes with them.  A point asks for it for the
// stretch that follows it.
//
// One rule for the renderer, the roll and the model, so what is heard, what is
// drawn and where a split note is cut all agree.
[[nodiscard]] inline float envelopeGainFromDb(float gainDb)
{
    return gainDb <= -59.9f ? 0.0f : std::pow(10.0f, gainDb / 20.0f);
}

[[nodiscard]] inline float envelopeDbFromGain(float gain)
{
    return gain <= 0.001f ? -60.0f : 20.0f * std::log10(gain);
}

// The level, in dB, `amount` of the way (0..1) from a point at leftDb to the
// next one at rightDb.
[[nodiscard]] inline float envelopeDbBetween(float leftDb, float rightDb, float amount,
                                             bool straightInAmplitude, const std::optional<EnvelopeBezier>& bezier = {})
{
    if(bezier)amount=static_cast<float>(tailFadeBezierProgress(amount,*bezier));
    if (!straightInAmplitude) return leftDb + (rightDb - leftDb) * amount;
    const auto left = envelopeGainFromDb(leftDb);
    const auto right = envelopeGainFromDb(rightDb);
    return envelopeDbFromGain(left + (right - left) * amount);
}
}
