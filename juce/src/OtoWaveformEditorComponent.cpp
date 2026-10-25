#include "OtoWaveformEditorComponent.h"
#include "MaterialAnnotationEditor.h"
#include "MaterialAnnotationSession.h"
#include "OttoMaterialMigration.h"
#include <juce_cryptography/juce_cryptography.h>
#include <cstdlib>
#include "backend/UtauRenderer.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <vector>

namespace hachi
{
namespace
{
juce::String utf8(const char* text) { return I18n::uiText(text); }

bool lightEditor() { return Palette::background == juce::Colour(0xfffffffb); }
juce::Colour sourcePitchColour()
{
    return lightEditor() ? Palette::pitchLine : juce::Colour(0xff7aeeee);
}
std::array<juce::Colour, 5> timingColours()
{
    if (lightEditor()) return { Palette::heading, Palette::text, Palette::heading,
                                Palette::accent, Palette::textMuted };
    return { juce::Colours::mediumpurple.brighter(0.35f), juce::Colours::orangered,
             juce::Colours::mediumpurple.brighter(0.35f), juce::Colours::red, juce::Colours::limegreen };
}

constexpr std::array<const char*, 8> parameterNames {
    "偏移", "辅音", "终止", "先行声音", "重叠",
    // The three Jie boundaries, named after the region each one closes.
    "声母末", "介音末", "韵腹末"
};

// Onset / glide / nucleus / coda, in the order the regions appear.  These
// are 界's names: four regions that are always the four parts of a Chinese
// syllable, in that order.
constexpr std::array<const char*, 4> jieRegionNames {
    "声母", "介音", "韵腹", "韵尾"
};

// What to call one region.  谋 numbers them: an entry there has two, three or
// four regions and any of them may be a consonant, so naming the last one 韵尾
// would be asserting a structure the entry is free to contradict -- and the
// number is what the tick boxes beside the wave are labelled with, so the two
// can be read against each other.
juce::String regionName(bool mou, int index)
{
    if (mou) return utf8("第") + juce::String(index + 1) + utf8("区");
    return juce::isPositiveAndBelow(index, 4)
        ? utf8(jieRegionNames[static_cast<std::size_t>(index)]) : juce::String{};
}

// A whole recording, as two channels: a mono file is copied to both, or it
// would come out of the left speaker only.  Empty when it cannot be read.
juce::AudioBuffer<float> readRecording(const juce::File& file, double& sampleRate)
{
    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    const std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
    if (reader == nullptr || reader->sampleRate <= 0.0 || reader->lengthInSamples <= 0
        || reader->lengthInSamples > std::numeric_limits<int>::max())
        return {};
    sampleRate = reader->sampleRate;
    const auto count = static_cast<int>(reader->lengthInSamples);
    const auto channels = juce::jlimit(1, 2, static_cast<int>(reader->numChannels));
    juce::AudioBuffer<float> read(channels, count);
    read.clear();
    if (!reader->read(&read, 0, count, 0, true, channels > 1)) return {};
    juce::AudioBuffer<float> recording(2, count);
    for (int channel = 0; channel < 2; ++channel)
        recording.copyFrom(channel, 0, read, std::min(channel, channels - 1), 0, count);
    return recording;
}
}

// A recording, played once from the top at whatever rate the device runs, and
// silent after its end.  The audio thread only reads the samples and moves the
// position; the window reads the position back for the playhead, so the
// position is the one thing the two share.
class OtoWaveformEditorComponent::SourcePreview final : public juce::AudioSource
{
public:
    SourcePreview(juce::AudioBuffer<float> audio, double rate)
        : recording(std::move(audio)), recordingRate(rate) {}

    void prepareToPlay(int, double sampleRate) override
    {
        step.store(sampleRate > 0.0 ? recordingRate / sampleRate : 1.0);
    }
    void releaseResources() override {}
    void getNextAudioBlock(const juce::AudioSourceChannelInfo& info) override
    {
        info.clearActiveBufferRegion();
        render(*info.buffer, info.startSample, info.numSamples);
    }
    // Fills up to count samples from startSample and says how many there were
    // to give.  Past the end nothing is written.
    int render(juce::AudioBuffer<float>& buffer, int startSample, int count)
    {
        const auto length = recording.getNumSamples();
        const auto advance = step.load();
        auto at = position.load();
        auto written = 0;
        for (; written < count && at < static_cast<double>(length); ++written, at += advance)
        {
            const auto left = static_cast<int>(at);
            const auto right = std::min(length - 1, left + 1);
            const auto fraction = static_cast<float>(at - left);
            for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
            {
                const auto* samples = recording.getReadPointer(std::min(channel, 1));
                buffer.setSample(channel, startSample + written,
                    samples[left] + (samples[right] - samples[left]) * fraction);
            }
        }
        position.store(at);
        return written;
    }
    // How far in, in seconds.
    [[nodiscard]] double seconds() const
    {
        return std::min(position.load(), static_cast<double>(recording.getNumSamples()))
            / recordingRate;
    }
    [[nodiscard]] bool finished() const
    {
        return position.load() >= static_cast<double>(recording.getNumSamples());
    }

private:
    const juce::AudioBuffer<float> recording;
    const double recordingRate;
    std::atomic<double> position { 0.0 };
    std::atomic<double> step { 1.0 };
};

OtoWaveformEditorComponent::WaveformView::WaveformView(
    VoicebankOtoEntry& value, bool jieMode, bool mouMode,
    std::function<void()> changedCallback)
    : entry(value), jie(jieMode), mou(mouMode), changed(std::move(changedCallback))
{
    setOpaque(true);
    formats.registerBasicFormats();
    loadWaveform();
    // Open on the whole file.  Framing the entry instead put the rest of the
    // sample off screen, so the waveform looked like it ended where the view
    // did; anyone wanting a closer look can zoom in, and FIT still returns to
    // the entry's own range.
    showWholeFile();
    if (durationMs > 0.0)
    {
        analysisRequest = backend::OtoAudioAnalysis::request(entry.audioFile,
            backend::OtoAudioAnalysis::editorConfig());
        pollAnalysis();
        if (!analysisReady()) startTimer(80);
    }
}

OtoWaveformEditorComponent::WaveformView::~WaveformView()
{
    stopTimer();
    if (analysisRequest) analysisRequest->cancelled = true;
}

void OtoWaveformEditorComponent::WaveformView::pollAnalysis()
{
    if (!analysisRequest) return;
    const auto ready = analysisRequest->snapshot();
    if (ready != analysis) { analysis = ready; repaint(); }
    if (analysisReady()) stopTimer();
}

void OtoWaveformEditorComponent::WaveformView::setAnalysisLayers(bool spectrum, bool f0)
{
    spectrumVisible = spectrum; sourceF0Visible = f0;
    if (!f0 && selectedPitch)
    {
        selectedPitch.reset();
        if (onPitchSelected) onPitchSelected(std::nullopt);
    }
    repaint();
}

juce::Rectangle<int> OtoWaveformEditorComponent::WaveformView::waveBounds() const
{
    auto area = plotBounds();
    if (spectrumVisible || sourceF0Visible) area.setHeight(juce::roundToInt(area.getHeight() * 0.40));
    return area;
}

juce::Rectangle<int> OtoWaveformEditorComponent::WaveformView::analysisBounds() const
{
    if (!spectrumVisible && !sourceF0Visible) return {};
    auto area = plotBounds(); area.setTop(waveBounds().getBottom() + 8); return area;
}

const backend::FcpeFrame* OtoWaveformEditorComponent::WaveformView::pitchFrameNear(
    juce::Point<float> position) const
{
    if (!sourceF0Visible || !analysis || analysis->pitch.empty()) return nullptr;
    const auto area = waveBounds().contains(position.toInt()) ? waveBounds() : analysisBounds();
    if (area.isEmpty() || !area.contains(position.toInt())) return nullptr;
    const double seconds = millisecondsForX(position.x) / 1000.0;
    const auto& pitch = analysis->pitch;
    auto it = std::lower_bound(pitch.begin(), pitch.end(), seconds,
        [](const auto& frame, double time) { return frame.timeSeconds < time; });
    if (it != pitch.begin() && (it == pitch.end()
        || seconds - (it - 1)->timeSeconds < it->timeSeconds - seconds)) --it;
    if (it == pitch.end() || !it->voiced || std::abs(it->timeSeconds - seconds) > 0.025)
        return nullptr;
    const auto hz = 440.0 * std::pow(2.0, (it->midi - 69.0) / 12.0);
    if (!std::isfinite(hz) || hz < analysis->minimumHz || hz > analysis->maximumHz) return nullptr;
    const auto y = area.getBottom() - analysis->fractionForHz(hz) * area.getHeight();
    return std::abs(position.y - y) <= 7.0 ? &*it : nullptr;
}

void OtoWaveformEditorComponent::WaveformView::paintSourceF0(
    juce::Graphics& g, juce::Rectangle<int> area)
{
    if (!sourceF0Visible || !analysis || area.isEmpty()) return;
    const auto yForHz = [&](double hz)
    { return static_cast<float>(area.getBottom() - analysis->fractionForHz(hz) * area.getHeight()); };
    juce::Graphics::ScopedSaveState save(g); g.reduceClipRegion(area);
    juce::Path path;
    const auto& pitch = analysis->pitch;
    auto from = std::lower_bound(pitch.begin(), pitch.end(), viewStartMs() / 1000.0,
        [](const auto& frame, double time) { return frame.timeSeconds < time; });
    if (from != pitch.begin()) --from;
    bool connected = false; double previousTime = -1;
    for (auto it = from; it != pitch.end(); ++it)
    {
        if (it->timeSeconds > (viewStartMs() + visibleSpanMs()) / 1000.0 + 0.02) break;
        const double hz = 440.0 * std::pow(2.0, (it->midi - 69.0) / 12.0);
        if (!it->voiced || !std::isfinite(hz) || hz < analysis->minimumHz || hz > analysis->maximumHz)
        { connected = false; continue; }
        const auto x = xForMilliseconds(it->timeSeconds * 1000.0), y = yForHz(hz);
        if (!connected || it->timeSeconds - previousTime > 0.025) path.startNewSubPath(x, y);
        else path.lineTo(x, y);
        connected = true; previousTime = it->timeSeconds;
    }
    g.setColour(lightEditor() ? Palette::base : juce::Colours::black.withAlpha(0.85f));
    g.strokePath(path, juce::PathStrokeType(3.8f));
    g.setColour(sourcePitchColour()); g.strokePath(path, juce::PathStrokeType(1.8f));
    if (selectedPitch)
    {
        const auto x = xForMilliseconds(selectedPitch->timeSeconds * 1000.0);
        const auto hz = 440.0 * std::pow(2.0, (selectedPitch->midi - 69.0) / 12.0);
        const auto y = yForHz(hz);
        const float dashes[] { 3.0f, 4.0f };
        g.setColour(sourcePitchColour().withAlpha(0.65f));
        g.drawDashedLine(juce::Line<float>(x, static_cast<float>(area.getY()),
            x, static_cast<float>(area.getBottom())), dashes, 2, 1.0f);
        g.setColour(lightEditor() ? Palette::base : juce::Colour(0xff101922)); g.fillEllipse(x - 4, y - 4, 8, 8);
        g.setColour(sourcePitchColour()); g.drawEllipse(x - 4, y - 4, 8, 8, 1.8f);
    }
}

void OtoWaveformEditorComponent::WaveformView::paintAnalysis(juce::Graphics& g)
{
    const auto area = analysisBounds();
    if (area.isEmpty()) return;
    const auto analysisBackground = lightEditor() ? Palette::graphBackground : juce::Colour(0xff101922);
    g.setColour(analysisBackground); g.fillRect(area);
    if (analysis && spectrumVisible && analysis->spectrogram.isValid())
    {
        juce::Graphics::ScopedSaveState save(g); g.reduceClipRegion(area);
        if (lightEditor() && (lightSpectrogramSource != analysis->spectrogram || !lightSpectrogram.isValid()))
        {
            lightSpectrogramSource = analysis->spectrogram;
            lightSpectrogram = juce::Image(juce::Image::RGB, lightSpectrogramSource.getWidth(),
                lightSpectrogramSource.getHeight(), false, juce::SoftwareImageType());
            const juce::Image::BitmapData source(lightSpectrogramSource, juce::Image::BitmapData::readOnly);
            juce::Image::BitmapData destination(lightSpectrogram, juce::Image::BitmapData::writeOnly);
            for (int y = 0; y < source.height; ++y)
                for (int x = 0; x < source.width; ++x)
                {
                    // The backend heat ramp has strictly increasing red values
                    // (16..255). Recolour only its display, never analysis data.
                    const auto strength = juce::jlimit(0.0f, 1.0f, (source.getPixelColour(x, y).getRed() - 16) / 239.0f);
                    destination.setPixelColour(x, y, Palette::graphBackground.interpolatedWith(Palette::heading, strength));
                }
        }
        const auto& image = lightEditor() ? lightSpectrogram : analysis->spectrogram;
        const auto scaleX = static_cast<float>(durationMs / visibleSpanMs() * area.getWidth() / image.getWidth());
        const auto scaleY = static_cast<float>(area.getHeight()) / image.getHeight();
        g.setOpacity(1.0f);
        g.drawImageTransformed(image, juce::AffineTransform::scale(scaleX, scaleY)
            .translated(xForMilliseconds(0), static_cast<float>(area.getY())));
    }
    if (analysis)
    {
        const auto yForHz = [&](double hz)
        { return static_cast<float>(area.getBottom() - analysis->fractionForHz(hz) * area.getHeight()); };
        g.setFont(10.0f);
        float lastLabel = -1000;
        for (double hz : {10000.0, 5000.0, 2000.0, 1000.0, 500.0, 200.0, 100.0, 50.0})
        {
            if (hz < analysis->minimumHz || hz > analysis->maximumHz) continue;
            const auto y = yForHz(hz);
            if (y - lastLabel < 16) continue;
            lastLabel = y;
            g.setColour(lightEditor() ? Palette::beatGrid : juce::Colours::white.withAlpha(0.10f));
            g.drawHorizontalLine(juce::roundToInt(y), static_cast<float>(area.getX()), static_cast<float>(area.getRight()));
            g.setColour(Palette::textMuted);
            const auto label = hz >= 1000 ? juce::String(hz / 1000.0, hz == 10000 ? 0 : 1) + "k" : juce::String(static_cast<int>(hz));
            g.drawText(label, 0, juce::roundToInt(y) - 7, area.getX() - 5, 14, juce::Justification::centredRight);
        }
        g.setColour(Palette::textMuted); g.drawText("Hz", 0, area.getY() - 15, area.getX() - 5, 14, juce::Justification::centredRight);
        paintSourceF0(g, area);
    }
    auto caption = !analysisReady() ? utf8("正在分析原始录音…")
        : analysis && analysis->error.isNotEmpty() ? analysis->error
        : sourceF0Visible ? utf8("原始 F0 · ") + (analysis ? analysis->pitchBackend : juce::String{})
        : utf8("原始录音频谱");
    if (analysisReady() && sourceF0Visible && analysis && std::none_of(analysis->pitch.begin(), analysis->pitch.end(),
        [](const auto& point) { return point.voiced; })) caption += utf8(" · 未检测到有声音高");
    if (analysis && analysis->warning.isNotEmpty()) caption += utf8(" · 内置识别回退");
    const auto badge = area.withHeight(18).reduced(4, 0);
    const auto badgeBackground = lightEditor() ? Palette::panel : juce::Colour(0xff101922);
    g.setColour(badgeBackground); g.fillRect(badge);
    g.setColour(Palette::contrastingText(badgeBackground)); g.setFont(11.0f);
    g.drawText(caption, badge.reduced(4, 0), juce::Justification::centredLeft, true);
}

void OtoWaveformEditorComponent::WaveformView::paint(juce::Graphics& g)
{
    g.fillAll(Palette::graphBackground);
    const auto plot = plotBounds();
    const auto wave = waveBounds();
    g.setColour(Palette::border);
    g.drawRect(plot);
    if (durationMs <= 0.0)
    {
        g.setColour(Palette::textMuted);
        g.drawFittedText(utf8("无法读取音频：") + entry.audioFile.getFullPathName(), plot.reduced(12),
                         juce::Justification::centred, 2);
        return;
    }

    const auto offsetX = xForMilliseconds(entry.offsetMs);
    const auto endX = xForMilliseconds(endMilliseconds());
    const auto consonantX = xForMilliseconds(entry.offsetMs + entry.consonantMs);
    const auto preutteranceX = xForMilliseconds(entry.offsetMs + entry.preutteranceMs);
    const auto overlapX = xForMilliseconds(entry.offsetMs + entry.overlapMs);

    g.setColour((lightEditor() ? Palette::accent : juce::Colours::mediumpurple).withAlpha(0.27f));
    g.fillRect(juce::Rectangle<float>(static_cast<float>(plot.getX()),
        static_cast<float>(plot.getY()), std::max(0.0f, offsetX - plot.getX()),
        static_cast<float>(plot.getHeight())));
    g.fillRect(juce::Rectangle<float>(endX, static_cast<float>(plot.getY()),
        std::max(0.0f, plot.getRight() - endX), static_cast<float>(plot.getHeight())));

    if (!jie)
    {
        g.setColour((lightEditor() ? Palette::text : juce::Colours::red).withAlpha(0.18f));
        g.fillRect(juce::Rectangle<float>(std::min(offsetX, consonantX),
            static_cast<float>(plot.getY()), std::abs(consonantX - offsetX),
            static_cast<float>(plot.getHeight())));
    }

    // In Jie mode the four regions are shaded instead of the single consonant
    // block, so a collapsed region still reads as a visible seam.
    const auto jieOnsetX = xForMilliseconds(entry.offsetMs + entry.jieOnsetMs);
    const auto jieGlideX = xForMilliseconds(entry.offsetMs + entry.jieGlideMs);
    const auto jieNucleusX = xForMilliseconds(entry.offsetMs + entry.jieNucleusMs);
    const std::array<juce::Colour, 4> jieColours = lightEditor()
        ? std::array<juce::Colour, 4>{ Palette::text, Palette::accent, Palette::heading, Palette::textMuted }
        : std::array<juce::Colour, 4>{ juce::Colours::orangered, juce::Colours::gold,
                                     juce::Colours::deepskyblue, juce::Colours::mediumseagreen };
    // The spans this entry really has.  The last one always closes at the
    // end of the entry, so three regions use two inner boundaries and two
    // regions use one; the rest of the array is not drawn at all.
    const std::array<float, 3> inner { jieOnsetX, jieGlideX, jieNucleusX };
    const auto count = juce::jlimit(2, 4, regions);
    std::array<std::pair<float, float>, 4> spans {};
    {
        auto from = offsetX;
        for (int index = 0; index < count; ++index)
        {
            const auto to = index + 1 == count
                ? endX : inner[static_cast<std::size_t>(index)];
            spans[static_cast<std::size_t>(index)] = { from, to };
            from = to;
        }
    }
    if (jie)
    {
        for (std::size_t index = 0; index < static_cast<std::size_t>(count); ++index)
        {
            g.setColour(jieColours[index].withAlpha(0.17f));
            g.fillRect(juce::Rectangle<float>(spans[index].first,
                static_cast<float>(plot.getY()),
                std::max(0.0f, spans[index].second - spans[index].first),
                static_cast<float>(plot.getHeight())));
        }
    }

    if (!waveformPeaks.empty())
    {
        juce::Graphics::ScopedSaveState waveClip(g); g.reduceClipRegion(wave);
        g.setColour(lightEditor() ? Palette::textMuted : Palette::accentLight);
        const auto centreY = static_cast<float>(wave.getCentreY());
        const auto amplitude = static_cast<float>(wave.getHeight()) * 0.43f
            * static_cast<float>(amplitudeZoom);
        const auto columns = std::max(1, plot.getWidth());
        // Each column covers one slice of the visible window, not of the file.
        const auto peakAt = [this](double milliseconds)
        {
            const auto ratio = juce::jlimit(0.0, 1.0, milliseconds / std::max(1.0, durationMs));
            return static_cast<std::size_t>(juce::jlimit<double>(0.0,
                static_cast<double>(waveformPeaks.size() - 1),
                ratio * static_cast<double>(waveformPeaks.size() - 1)));
        };
        const auto spanMs = visibleSpanMs();
        const auto startMs = viewStartMs();
        for (int column = 0; column < columns; ++column)
        {
            const auto first = peakAt(startMs + spanMs * column / columns);
            const auto last = std::max(first + 1,
                peakAt(startMs + spanMs * (column + 1) / columns));
            auto minimum = 1.0f;
            auto maximum = -1.0f;
            for (auto index = first; index < std::min(last, waveformPeaks.size()); ++index)
            {
                minimum = std::min(minimum, waveformPeaks[index].first);
                maximum = std::max(maximum, waveformPeaks[index].second);
            }
            if (minimum > maximum) minimum = maximum = 0.0f;
            const auto x = static_cast<float>(plot.getX() + column);
            g.drawVerticalLine(static_cast<int>(x), centreY - maximum * amplitude,
                               centreY - minimum * amplitude);
        }
    }
    else
    {
        g.setColour(Palette::textMuted);
        g.drawText(utf8("音频中没有可显示的波形"), plot, juce::Justification::centred);
    }

    paintSourceF0(g, wave);
    paintAnalysis(g);

    if (nativeDetail)
    {
        const float dash[]{4, 3};
        for (const auto& segment : nativeSegments)
        {
            const auto left = xForMilliseconds(segment.sourceStartSeconds * 1000);
            const auto right = xForMilliseconds(segment.sourceEndSeconds * 1000);
            const auto colour = lightEditor()
                ? (segment.role == NativeSegmentRole::consonant ? Palette::text : Palette::accent)
                : (segment.role == NativeSegmentRole::consonant ? juce::Colours::orangered
                    : segment.role == NativeSegmentRole::vowel ? juce::Colours::deepskyblue : juce::Colours::gold);
            g.setColour(colour.withAlpha(0.12f));
            g.fillRect(juce::Rectangle<float>(left, static_cast<float>(plot.getY()), std::max(0.0f, right-left), static_cast<float>(plot.getHeight())));
            g.setColour(colour);
            g.drawDashedLine(juce::Line<float>(right, static_cast<float>(plot.getY()), right, static_cast<float>(plot.getBottom())), dash, 2);
            if (lightEditor()) g.setColour(Palette::contrastingText(Palette::graphBackground.overlaidWith(colour.withAlpha(0.12f))));
            const auto caption = segment.role == NativeSegmentRole::vowel
                && segment.alias.trim().isNotEmpty() && segment.alias != "-" ? segment.alias : nativeSegmentRoleName(segment.role);
            const auto captionLeft = std::max(plot.getX(), juce::roundToInt(left)) + 3;
            const auto captionWidth = std::min(plot.getRight(), juce::roundToInt(right)) - captionLeft - 2;
            if (captionWidth >= 24)
                g.drawText(caption, captionLeft, plot.getY()+3, captionWidth, 18, juce::Justification::centredLeft, true);
        }
    }

    const auto drawBoundary = [&g, &plot](float x, juce::Colour colour, float thickness)
    {
        g.setColour(colour);
        g.drawLine(x, static_cast<float>(plot.getY()), x,
                   static_cast<float>(plot.getBottom()), thickness);
    };
    const auto colours = timingColours();
    drawBoundary(offsetX, colours[0], 2.0f);
    drawBoundary(endX, colours[2], 2.0f);
    if (!jie) drawBoundary(consonantX, colours[1], 2.0f);
    drawBoundary(preutteranceX, colours[3], 2.0f);
    drawBoundary(overlapX, colours[4], 2.0f);
    if (preferOuterRangeDrag)
    {
        // Keep the draggable purple limits visible even when C/P/O coincide.
        drawBoundary(offsetX, colours[0], 2.0f);
        drawBoundary(endX, colours[2], 2.0f);
    }

    g.setFont(11.0f);
    for (const auto& badge : handleBadges())
    {
        // Opaque swatches make their WCAG foreground deterministic in both themes.
        g.setColour(badge.colour);
        g.fillRoundedRectangle(badge.bounds, 3.0f);
        g.setColour(Palette::contrastingText(badge.colour));
        g.drawText(badge.text, badge.bounds, juce::Justification::centred);
    }

    if (jie)
    {
        for (int index = 0; index + 1 < count; ++index)
            drawBoundary(inner[static_cast<std::size_t>(index)],
                         jieColours[static_cast<std::size_t>(index) + 1].darker(0.15f),
                         2.0f);

        // Region names sit inside their own span, on a row below the oto
        // badges so the two sets never collide.
        for (std::size_t index = 0; index < static_cast<std::size_t>(count); ++index)
        {
            const auto width = spans[index].second - spans[index].first;
            if (width < 26.0f) continue;
            auto badge = juce::Rectangle<float>(
                spans[index].first + width * 0.5f - 17.0f,
                static_cast<float>(wave.getBottom() - 19), 34.0f, 15.0f);
            g.setColour(jieColours[index]);
            g.fillRoundedRectangle(badge, 3.0f);
            g.setColour(Palette::contrastingText(jieColours[index]));
            g.drawText(regionName(mou, static_cast<int>(index)), badge,
                       juce::Justification::centred);
        }
    }

    g.setColour(Palette::textMuted);
    g.setFont(11.0f);
    for (int division = 0; division <= 10; ++division)
    {
        const auto value = viewStartMs() + visibleSpanMs() * division / 10.0;
        const auto x = xForMilliseconds(value);
        g.drawVerticalLine(static_cast<int>(x), static_cast<float>(plot.getY()),
                           static_cast<float>(plot.getY() + 5));
        g.drawText(juce::String(value / 1000.0, 3), static_cast<int>(x) - 28,
                   plot.getBottom() + 2, 56, 16, juce::Justification::centred);
    }

    // 播放原音: how far the sound has got, over everything else.
    if (playheadMs)
    {
        const auto x = xForMilliseconds(*playheadMs);
        if (x >= static_cast<float>(plot.getX()) && x <= static_cast<float>(plot.getRight()))
        {
            g.setColour(lightEditor() ? Palette::playhead : juce::Colours::white);
            g.drawLine(x, static_cast<float>(plot.getY()), x,
                       static_cast<float>(plot.getBottom()), 1.5f);
        }
    }
}

void OtoWaveformEditorComponent::WaveformView::mouseMove(const juce::MouseEvent& event)
{
    updateCursor(event.position);
    if (analysis && (analysisBounds().contains(event.position.toInt())
        || (sourceF0Visible && waveBounds().contains(event.position.toInt()))))
    {
        const auto seconds = millisecondsForX(event.position.x) / 1000.0;
        auto tip = juce::String(seconds, 3) + " s";
        auto it = std::lower_bound(analysis->pitch.begin(), analysis->pitch.end(), seconds,
            [](const auto& frame, double time) { return frame.timeSeconds < time; });
        if (it != analysis->pitch.begin() && (it == analysis->pitch.end()
            || seconds - (it - 1)->timeSeconds < it->timeSeconds - seconds)) --it;
        if (sourceF0Visible && it != analysis->pitch.end() && it->voiced && std::abs(it->timeSeconds - seconds) <= 0.025)
        {
            const auto hz = 440.0 * std::pow(2.0, (it->midi - 69.0) / 12.0);
            tip += utf8("  原始 F0：") + juce::String(hz, 1) + " Hz  "
                + juce::MidiMessage::getMidiNoteName(juce::roundToInt(it->midi), true, true, 4);
        }
        else if (sourceF0Visible) tip += utf8("  原始 F0：无有声音高");
        if (analysis->warning.isNotEmpty()) tip += "\n" + analysis->warning;
        setTooltip(tip);
    }
    else setTooltip({});
}

void OtoWaveformEditorComponent::WaveformView::mouseExit(const juce::MouseEvent&)
{
    if (dragging == Handle::none) setMouseCursor(juce::MouseCursor::NormalCursor);
}

void OtoWaveformEditorComponent::WaveformView::mouseDown(const juce::MouseEvent& event)
{
    const auto handle = handleNear(event.position);
    const bool outer = preferOuterRangeDrag && (handle == Handle::offset || handle == Handle::cutoff);
    nativeBoundaryDragging = !outer && nativeDetail && plotBounds().contains(event.position.toInt())
        ? nativeBoundaryNear(event.position) : -1;
    dragging = nativeBoundaryDragging >= 0 ? Handle::none : handle;
    dragGrabOffsetMs = dragging == Handle::none ? 0.0
        : millisecondsForX(event.position.x) - millisecondsForHandle(dragging);
    const auto* frame = event.mods.isLeftButtonDown() ? pitchFrameNear(event.position) : nullptr;
    if (frame)
    {
        selectedPitch = *frame;
        if (onPitchSelected) onPitchSelected(selectedPitch);
        repaint();
    }
    // A pitch click reads the frame; boundary drags keep their original priority.
    panning = dragging == Handle::none && nativeBoundaryDragging < 0 && frame == nullptr && viewZoom > 1.0;
    panStartCentreMs = viewCentreMs;
    updateCursor(event.position);
}

void OtoWaveformEditorComponent::WaveformView::mouseDrag(const juce::MouseEvent& event)
{
    if (nativeBoundaryDragging >= 0)
    {
        diagnosticDragNativeBoundary(nativeBoundaryDragging, millisecondsForX(event.position.x) / 1000.0);
        return;
    }
    if (panning && durationMs > 0.0)
    {
        const auto perPixel = visibleSpanMs() / std::max(1, plotBounds().getWidth());
        viewCentreMs = panStartCentreMs - event.getDistanceFromDragStartX() * perPixel;
        clampView();
        repaint();
        if (onViewChanged) onViewChanged();
        return;
    }
    if (dragging == Handle::none || durationMs <= 0.0) return;
    applyHandle(dragging, millisecondsForX(event.position.x) - dragGrabOffsetMs);
}

void OtoWaveformEditorComponent::WaveformView::applyHandle(Handle handle, double value)
{
    if (handle == Handle::none || durationMs <= 0.0) return;
    if (lockOuterRange && (handle == Handle::offset || handle == Handle::cutoff)) return;
    // Overlap and preutterance are signed timing markers, not audio cuts.
    // Keep recording bounds only for handles which delimit real sample data.
    if (handle != Handle::overlap && handle != Handle::preutterance)
        value = juce::jlimit(0.0, durationMs, value);
    switch (handle)
    {
        case Handle::offset:
        {
            const auto maximum = entry.cutoffMs < 0.0
                ? std::max(0.0, durationMs + entry.cutoffMs) : endMilliseconds();
            const auto previous = entry.offsetMs;
            entry.offsetMs = juce::jlimit(0.0, std::max(0.0, maximum - 0.1), value);
            // Every other field is a distance measured from the offset, so
            // moving it on its own carried the whole entry along.  Take the
            // move back out of each one so the boundaries keep the place in
            // the waveform they were put on, and only the offset line moves.
            const auto shift = entry.offsetMs - previous;
            if (std::abs(shift) > 1.0e-9)
            {
                entry.consonantMs = std::max(0.0, entry.consonantMs - shift);
                entry.preutteranceMs -= shift;
                entry.overlapMs -= shift;
                // A negative cutoff is a length from the offset and so moves
                // with it; a positive one counts back from the end of the file
                // and is already independent.
                if (entry.cutoffMs < 0.0) entry.cutoffMs += shift;
                entry.jieOnsetMs = std::max(0.0, entry.jieOnsetMs - shift);
                entry.jieGlideMs = std::max(entry.jieOnsetMs, entry.jieGlideMs - shift);
                entry.jieNucleusMs = std::max(entry.jieGlideMs,
                                              entry.jieNucleusMs - shift);
            }
            break;
        }
        case Handle::consonant:
            entry.consonantMs = std::max(0.0, value - entry.offsetMs);
            break;
        case Handle::cutoff:
            value = juce::jlimit(entry.offsetMs + 0.1, durationMs, value);
            entry.cutoffMs = entry.cutoffMs < 0.0
                ? -(value - entry.offsetMs) : durationMs - value;
            break;
        case Handle::preutterance:
            entry.preutteranceMs = value - entry.offsetMs;
            break;
        case Handle::overlap:
            entry.overlapMs = value - entry.offsetMs;
            break;
        // The Jie boundaries stay ordered and inside [offset, end], so a
        // drag can collapse a region to zero but never turn it inside out.
        case Handle::jieOnset:
            entry.jieOnsetMs = juce::jlimit(0.0, boundaryCeiling(0),
                                            value - entry.offsetMs);
            entry.consonantMs = entry.jieOnsetMs;
            break;
        case Handle::jieGlide:
            entry.jieGlideMs = juce::jlimit(entry.jieOnsetMs, boundaryCeiling(1),
                                            value - entry.offsetMs);
            break;
        case Handle::jieNucleus:
            entry.jieNucleusMs = juce::jlimit(entry.jieGlideMs,
                std::max(entry.jieGlideMs, boundaryCeiling(2)),
                value - entry.offsetMs);
            break;
        case Handle::none: break;
    }
    orderBoundaries();
    if (changed) changed();
    repaint();
}

int OtoWaveformEditorComponent::WaveformView::nativeBoundaryNear(juce::Point<float> position) const
{
    int nearest = -1; float distance = 8;
    for (std::size_t i = 0; i + 1 < nativeSegments.size(); ++i)
    {
        if (nativeSegments[i].role != NativeSegmentRole::vowel || nativeSegments[i + 1].role != NativeSegmentRole::vowel
            || !nativeSegments[i].stretchable || !nativeSegments[i + 1].stretchable) continue;
        const auto delta = std::abs(position.x - xForMilliseconds(nativeSegments[i].sourceEndSeconds * 1000));
        if (delta < distance) { distance = delta; nearest = static_cast<int>(i); }
    }
    return nearest;
}

void OtoWaveformEditorComponent::WaveformView::diagnosticDragNativeBoundary(int index, double seconds)
{
    if (!nativeDetail || index < 0 || static_cast<std::size_t>(index + 1) >= nativeSegments.size()
        || !std::isfinite(seconds)) return;
    auto& left = nativeSegments[static_cast<std::size_t>(index)];
    auto& right = nativeSegments[static_cast<std::size_t>(index + 1)];
    if (left.role != NativeSegmentRole::vowel || right.role != NativeSegmentRole::vowel
        || !left.stretchable || !right.stretchable) return;
    const bool contiguous = std::abs(left.sourceEndSeconds - right.sourceStartSeconds) <= 1.e-9;
    const auto lower = left.sourceStartSeconds + 1.e-6;
    const auto upper = contiguous ? right.sourceEndSeconds - 1.e-6 : right.sourceStartSeconds;
    if (upper <= lower) return;
    left.sourceEndSeconds = juce::jlimit(lower, upper, seconds);
    if (contiguous) right.sourceStartSeconds = left.sourceEndSeconds;
    if (changed) changed();
    repaint();
}

void OtoWaveformEditorComponent::WaveformView::mouseUp(const juce::MouseEvent& event)
{
    nativeBoundaryDragging = -1;
    dragging = Handle::none;
    panning = false;
    updateCursor(event.position);
}

void OtoWaveformEditorComponent::WaveformView::loadWaveform()
{
    waveformPeaks.clear();
    if (!entry.audioFile.existsAsFile()) return;
    auto reader = std::unique_ptr<juce::AudioFormatReader>(
        formats.createReaderFor(entry.audioFile));
    if (reader == nullptr || reader->sampleRate <= 0.0 || reader->lengthInSamples <= 0)
        return;

    durationMs = static_cast<double>(reader->lengthInSamples)
        / reader->sampleRate * 1000.0;
    constexpr int peakCount = 2048;
    constexpr int chunkSize = 8192;
    waveformPeaks.assign(peakCount, { 1.0f, -1.0f });
    const auto channels = juce::jlimit(1, 2, static_cast<int>(reader->numChannels));
    juce::AudioBuffer<float> buffer(channels, chunkSize);
    for (juce::int64 position = 0; position < reader->lengthInSamples; position += chunkSize)
    {
        const auto amount = static_cast<int>(std::min<juce::int64>(
            chunkSize, reader->lengthInSamples - position));
        buffer.clear();
        if (!reader->read(&buffer, 0, amount, position, true, channels > 1))
            continue;
        for (int sample = 0; sample < amount; ++sample)
        {
            const auto absolute = position + sample;
            const auto peakIndex = static_cast<std::size_t>(std::min<juce::int64>(
                peakCount - 1, absolute * peakCount / reader->lengthInSamples));
            auto minimum = 1.0f;
            auto maximum = -1.0f;
            for (int channel = 0; channel < channels; ++channel)
            {
                const auto value = buffer.getSample(channel, sample);
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
            }
            waveformPeaks[peakIndex].first = std::min(waveformPeaks[peakIndex].first, minimum);
            waveformPeaks[peakIndex].second = std::max(waveformPeaks[peakIndex].second, maximum);
        }
    }
    for (auto& peak : waveformPeaks)
        if (peak.first > peak.second) peak = { 0.0f, 0.0f };
}

juce::Rectangle<int> OtoWaveformEditorComponent::WaveformView::plotBounds() const
{
    auto bounds = getLocalBounds().reduced(14, 2);
    if (spectrumVisible || sourceF0Visible) bounds.removeFromLeft(34);
    bounds.removeFromTop(36);
    bounds.removeFromBottom(22);
    return bounds;
}

double OtoWaveformEditorComponent::WaveformView::visibleSpanMs() const
{
    // Below 1 the window is wider than the file, which is how a handle that
    // sits outside it -- a large overlap, a preutterance past the end -- is
    // brought on screen at all.
    return durationMs / std::max(0.01, viewZoom);
}

double OtoWaveformEditorComponent::WaveformView::viewStartMs() const
{
    return viewCentreMs - visibleSpanMs() * 0.5;
}

void OtoWaveformEditorComponent::WaveformView::clampView()
{
    // Half of 1x is two clicks of the zoom-out button, enough margin either
    // side of the file for an oto value that falls outside it.
    viewZoom = juce::jlimit(0.5, 400.0, viewZoom);
    amplitudeZoom = juce::jlimit(0.25, 16.0, amplitudeZoom);
    const auto span = visibleSpanMs();
    if (span >= durationMs)
        // Wider than the file: centre it, so the margin is shared rather than
        // all of it landing on one side.
        viewCentreMs = durationMs * 0.5;
    else
        // Zoomed in, the window may not run past either end of the file.
        viewCentreMs = juce::jlimit(span * 0.5, durationMs - span * 0.5, viewCentreMs);
}

void OtoWaveformEditorComponent::WaveformView::nudgeHorizontalZoom(double factor)
{
    viewZoom *= factor;
    clampView();
    repaint();
    if (onViewChanged) onViewChanged();
}

void OtoWaveformEditorComponent::WaveformView::setVisibleStartMilliseconds(double startMs)
{
    viewCentreMs = startMs + visibleSpanMs() * 0.5;
    clampView();
    repaint();
}

void OtoWaveformEditorComponent::WaveformView::nudgeVerticalZoom(double factor)
{
    amplitudeZoom *= factor;
    clampView();
    repaint();
}

void OtoWaveformEditorComponent::WaveformView::showWholeFile()
{
    viewZoom = 1.0;
    viewCentreMs = durationMs * 0.5;
    clampView();
    repaint();
    if (onViewChanged) onViewChanged();
}

void OtoWaveformEditorComponent::WaveformView::frameEntry()
{
    // Fit the oto region with a little air on both sides.
    const auto span = std::max(1.0, endMilliseconds() - entry.offsetMs);
    const auto padded = span * 1.25;
    viewCentreMs = entry.offsetMs + span * 0.5;
    viewZoom = durationMs > 0.0 ? std::max(1.0, durationMs / padded) : 1.0;
    clampView();
    repaint();
    if (onViewChanged) onViewChanged();
}

float OtoWaveformEditorComponent::WaveformView::xForMilliseconds(double milliseconds) const
{
    const auto plot = plotBounds();
    if (durationMs <= 0.0) return static_cast<float>(plot.getX());
    const auto span = std::max(1.0e-9, visibleSpanMs());
    // Not clamped to the plot: boundary lines outside the window are drawn off
    // to the side and clipped, rather than piling up on the edge.
    return static_cast<float>(plot.getX()) + static_cast<float>(plot.getWidth())
        * static_cast<float>((milliseconds - viewStartMs()) / span);
}

double OtoWaveformEditorComponent::WaveformView::millisecondsForX(float x) const
{
    const auto plot = plotBounds();
    const auto ratio = static_cast<double>(x - plot.getX()) / std::max(1, plot.getWidth());
    return viewStartMs() + ratio * visibleSpanMs();
}

double OtoWaveformEditorComponent::WaveformView::millisecondsForHandle(Handle handle) const
{
    switch (handle)
    {
        case Handle::offset: return entry.offsetMs;
        case Handle::consonant: return entry.offsetMs + entry.consonantMs;
        case Handle::cutoff: return endMilliseconds();
        case Handle::preutterance: return entry.offsetMs + entry.preutteranceMs;
        case Handle::overlap: return entry.offsetMs + entry.overlapMs;
        case Handle::jieOnset: return entry.offsetMs + entry.jieOnsetMs;
        case Handle::jieGlide: return entry.offsetMs + entry.jieGlideMs;
        case Handle::jieNucleus: return entry.offsetMs + entry.jieNucleusMs;
        case Handle::none: return 0.0;
    }
    return 0.0;
}

std::vector<OtoWaveformEditorComponent::WaveformView::HandleBadge>
OtoWaveformEditorComponent::WaveformView::handleBadges() const
{
    const auto plot = plotBounds();
    const std::array<Handle, 5> handles { Handle::offset, Handle::consonant,
        Handle::cutoff, Handle::preutterance, Handle::overlap };
    const std::array<juce::String, 5> labels { utf8("偏移"), utf8("辅音"),
        utf8("终止"), utf8("先行"), utf8("重叠") };
    const auto colours = timingColours();
    std::vector<HandleBadge> badges;
    for (std::size_t i = 0; i < handles.size(); ++i)
    {
        if (jie && handles[i] == Handle::consonant) continue;
        if (lockOuterRange && (handles[i] == Handle::offset || handles[i] == Handle::cutoff)) continue;
        const auto x = juce::jlimit(static_cast<float>(plot.getX()),
            static_cast<float>(std::max(plot.getX(), plot.getRight() - 38)),
            xForMilliseconds(millisecondsForHandle(handles[i])) - 18.0f);
        badges.push_back({ handles[i], labels[i], colours[i],
            { x, static_cast<float>(2 + (i % 2) * 16), 38.0f, 15.0f } });
    }
    return badges;
}

double OtoWaveformEditorComponent::WaveformView::endMilliseconds() const
{
    return entry.cutoffMs < 0.0 ? entry.offsetMs - entry.cutoffMs
                                : durationMs - entry.cutoffMs;
}

double OtoWaveformEditorComponent::WaveformView::boundaryCeiling(int index) const
{
    const auto span = std::max(0.0, endMilliseconds() - entry.offsetMs);
    if (index + 1 >= boundaryCount()) return span;
    return index == 0 ? entry.jieGlideMs : entry.jieNucleusMs;
}

void OtoWaveformEditorComponent::WaveformView::orderBoundaries()
{
    entry.jieGlideMs = std::max(entry.jieGlideMs, entry.jieOnsetMs);
    entry.jieNucleusMs = std::max(entry.jieNucleusMs, entry.jieGlideMs);
}

OtoWaveformEditorComponent::WaveformView::Handle
OtoWaveformEditorComponent::WaveformView::handleNear(const juce::Point<float>& position) const
{
    if (durationMs <= 0.0) return Handle::none;
    // The visible badge selects its own marker even when several lines coincide.
    // Hit-test in reverse paint order, using the exact same rectangles as paint.
    const auto badges = handleBadges();
    for (auto it = badges.rbegin(); it != badges.rend(); ++it)
        if (it->bounds.contains(position)) return it->handle;
    if (preferOuterRangeDrag && !lockOuterRange && plotBounds().expanded(8).contains(position.toInt()))
    {
        const auto startDistance = std::abs(position.x - xForMilliseconds(entry.offsetMs));
        const auto endDistance = std::abs(position.x - xForMilliseconds(endMilliseconds()));
        if (std::min(startDistance, endDistance) < 9.0f)
            return startDistance <= endDistance ? Handle::offset : Handle::cutoff;
    }
    if (!plotBounds().expanded(8, 32).contains(position.toInt())) return Handle::none;
    std::vector<std::pair<Handle, float>> handles {
        { Handle::preutterance, xForMilliseconds(entry.offsetMs + entry.preutteranceMs) },
        { Handle::overlap, xForMilliseconds(entry.offsetMs + entry.overlapMs) },
        { Handle::offset, xForMilliseconds(entry.offsetMs) },
        { Handle::cutoff, xForMilliseconds(endMilliseconds()) }
    };
    // Only classic mode has a consonant of its own; in Jie mode the onset
    // boundary is that same line and there is nothing to disambiguate.
    if (!jie)
        handles.push_back({ Handle::consonant,
            xForMilliseconds(entry.offsetMs + entry.consonantMs) });
    if (jie)
    {
        // Only the boundaries this entry has.  The ones past the count are not
        // drawn, so a press that took hold of one would be pulling on a line
        // nobody can see.
        const auto shown = boundaryCount();
        // Ties go to whichever handle was pushed first.  Where the onset and
        // the glide sit on the same millisecond -- which is how every entry
        // seeded from a classic oto starts out -- only the glide can still
        // move: the onset is clamped by it and would not budge, leaving the
        // pair permanently welded together.
        const auto glideOnOnset =
            std::abs(entry.jieGlideMs - entry.jieOnsetMs) < 1.0e-6;
        if (glideOnOnset && shown > 1)
            handles.push_back({ Handle::jieGlide,
                xForMilliseconds(entry.offsetMs + entry.jieGlideMs) });
        if (shown > 0)
            handles.push_back({ Handle::jieOnset,
                xForMilliseconds(entry.offsetMs + entry.jieOnsetMs) });
        if (!glideOnOnset && shown > 1)
            handles.push_back({ Handle::jieGlide,
                xForMilliseconds(entry.offsetMs + entry.jieGlideMs) });
        if (shown > 2)
            handles.push_back({ Handle::jieNucleus,
                xForMilliseconds(entry.offsetMs + entry.jieNucleusMs) });
    }
    auto nearest = Handle::none;
    auto nearestDistance = 9.0f;
    for (const auto& handle : handles)
    {
        if (lockOuterRange && (handle.first == Handle::offset || handle.first == Handle::cutoff)) continue;
        const auto distance = std::abs(position.x - handle.second);
        if (distance < nearestDistance)
        {
            nearestDistance = distance;
            nearest = handle.first;
        }
    }
    return nearest;
}

juce::String OtoWaveformEditorComponent::WaveformView::diagnosticHandleAtX(float x) const
{
    const auto middle = plotBounds().getCentreY();
    switch (handleNear({ x, static_cast<float>(middle) }))
    {
        case Handle::offset:      return "offset";
        case Handle::consonant:   return "consonant";
        case Handle::cutoff:      return "cutoff";
        case Handle::preutterance:return "preutterance";
        case Handle::overlap:     return "overlap";
        case Handle::jieOnset:    return "b1";
        case Handle::jieGlide:    return "b2";
        case Handle::jieNucleus:  return "b3";
        case Handle::none:        break;
    }
    return "none";
}

void OtoWaveformEditorComponent::WaveformView::updateCursor(
    const juce::Point<float>& position)
{
    if (dragging != Handle::none || handleNear(position) != Handle::none)
        setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
    else if (pitchFrameNear(position) != nullptr)
        setMouseCursor(juce::MouseCursor::PointingHandCursor);
    else
        setMouseCursor(viewZoom > 1.0 ? juce::MouseCursor::DraggingHandCursor
                                      : juce::MouseCursor::NormalCursor);
}

juce::String OtoWaveformEditorComponent::materialText(const char* zh, const char* en) const
{
    juce::ignoreUnused(zh);
    return materialStrings ? materialStrings->text(en) : juce::String::fromUTF8(en);
}

std::unique_ptr<OtoWaveformEditorComponent> OtoWaveformEditorComponent::forMaterial(
    const juce::File& audio, const juce::File& manifest, I18n& strings,
    std::function<void(const std::vector<SampleRegionSetting>&)> onSaved, bool quick, juce::String* error,
    const juce::String& selectedAudioId)
{
    juce::String problem;
    auto draft = MaterialAnnotationSession::open(audio, manifest, problem, selectedAudioId);
    if (!draft) { if (error) *error = problem; return {}; }
    VoicebankOtoEntry initial;
    initial.audioFile = audio; initial.otoFile = manifest; initial.sourceName = audio.getFileName();
    initial.hasJieOto = true; initial.cutoffMs = 0;
    initial.jieOnsetMs = draft->durationSeconds() * 250;
    initial.jieGlideMs = draft->durationSeconds() * 500;
    initial.jieNucleusMs = draft->durationSeconds() * 750;
    initial.mouClasses = "CVVV";
    // Build every existing detailed control once, then switch its display mode.
    auto editor = std::make_unique<OtoWaveformEditorComponent>(initial, true, true, [] {});
    editor->materialSession = std::shared_ptr<MaterialAnnotationSession>(std::move(draft));
    editor->materialStrings = &strings; editor->materialSaved = std::move(onSaved);
    editor->materialModeLabel.setText(editor->materialText("编辑模式", "Editing mode"), juce::dontSendNotification);
    editor->materialMode.addItem(editor->materialText("原生标注（默认）", "Native annotations (default)"), 1);
    editor->materialMode.addItem("UTAU Classic", 2); editor->materialMode.addItem("UTAU Jie", 3); editor->materialMode.addItem("UTAU Mou", 4);
    editor->materialMode.setSelectedId(1, juce::dontSendNotification);
    editor->materialMode.setTooltip(editor->materialText("模式仅改变参数视图；源标注保存到 otto.ini，本音符范围仅应用到工程。旧 OTO 不变。",
        "Modes change the view only: source scope saves otto.ini; this-note scope applies only to the project. Old OTO stays unchanged."));
    editor->materialBackButton.setButtonText(editor->materialText("返回发音总览", "Back to pronunciation overview"));
    for (auto* component : std::initializer_list<juce::Component*>{&editor->materialModeLabel, &editor->materialMode, &editor->materialBackButton})
        editor->addAndMakeVisible(component);
    juce::Component::SafePointer<OtoWaveformEditorComponent> safe(editor.get());
    editor->materialOverview.reset(new MaterialAnnotationEditor(editor->materialSession, strings,
        [safe](const auto& rows)
        {
            if (!safe || safe->editScope() != EditScope::source) return;
            safe->materialSourceSaved(rows);
        }));
    editor->materialOverview->onEditPronunciation = [safe](const juce::String& id)
    { if (safe) safe->selectMaterialPronunciation(id, true); };
    editor->addAndMakeVisible(*editor->materialOverview);
    editor->materialBackButton.onClick = [safe] { if (safe) safe->diagnosticMaterialBack(); };
    editor->materialSingleSyllable.setButtonText(editor->materialText("单音节音素", "Single syllable / phoneme"));
    editor->materialSingleSyllable.setTooltip(editor->materialText(
        "此录音直接使用详细参数，取消勾选返回发音总览。已有发音标注保留；空录音建立一个未保存的发音草稿。",
        "Use detail directly for this recording; uncheck to return to overview. Existing pronunciations are preserved. Unannotated audio gets one unsaved pronunciation draft."));
    editor->addAndMakeVisible(editor->materialSingleSyllable);
    editor->materialSingleSyllable.onClick = [safe]
    {
        if (safe && !safe->setSingleSyllableMaterial(safe->materialSingleSyllable.getToggleState()))
            safe->materialSingleSyllable.setToggleState(safe->singleSyllableMaterial, juce::dontSendNotification);
    };
    editor->materialScope.addItem(editor->materialText("源标注（otto.ini）", "Source annotations (otto.ini)"), 1);
    editor->materialScope.addItem(editor->materialText("仅此音符（工程）", "This note only (project)"), 2);
    editor->materialScope.setSelectedId(1, juce::dontSendNotification);
    editor->addChildComponent(editor->materialScope);
    editor->materialScope.onChange = [safe]
    { if (safe) safe->requestEditScope(safe->materialScope.getSelectedId() == 2 ? EditScope::thisNote : EditScope::source); };
    editor->materialRestoreLocal.setButtonText(editor->materialText("恢复源设置", "Restore source settings"));
    editor->addChildComponent(editor->materialRestoreLocal);
    editor->materialRestoreLocal.onClick = [safe] { if (safe) safe->requestRestoreLocal(); };
    editor->materialMode.onChange = [safe]
    {
        if (!safe || safe->loadingMaterial) return;
        juce::String error;
        if (safe->materialDetail && !safe->commitMaterialDetail(error))
        {
            safe->materialMode.setSelectedId(safe->waveform.nativeDetail ? 1 : !safe->jie ? 2 : safe->mou ? 4 : 3,
                juce::dontSendNotification);
            safe->helpLabel.setText(error, juce::dontSendNotification); return;
        }
        if (safe->materialDetail)
        {
            const auto id = safe->materialRegionId;
            for (const auto& row : safe->materialSession->rows()) if (row.materialRegionId == id) { safe->loadMaterialDetail(row); break; }
        }
        safe->refreshMaterialVisibility();
    };
    editor->materialPronunciationKindChoice.addItem(editor->materialText("普通音节", "Normal syllable"), 1);
    editor->materialPronunciationKindChoice.addItem(editor->materialText("过渡发音", "Transition pronunciation"), 2);
    editor->materialPronunciationKindChoice.addItem(editor->materialText("呼吸", "Breath"), 3);
    editor->materialPronunciationKindChoice.setTooltip(editor->materialText("整个发音的类别，不是内部 C/V 区段。", "Pronunciation identity, not an internal C/V role."));
    editor->materialKindLabel.setText(editor->materialText("发音身份", "Pronunciation identity"), juce::dontSendNotification);
    editor->materialSegmentLabel.setText(editor->materialText("内部元音区段", "Internal vowel segment"), juce::dontSendNotification);
    editor->materialSegmentChoice.setTextWhenNothingSelected(editor->materialText("内部元音区段", "Internal vowel segment"));
    editor->materialSplitSegment.setButtonText(editor->materialText("命名并切分元音", "Split named vowel"));
    editor->materialMergeSegment.setButtonText(editor->materialText("合并下一区段", "Merge next segment"));
    editor->materialSimplifiedConsonant.setButtonText(editor->materialText("简化辅音", "Simplified consonant"));
    editor->materialSimplifiedConsonant.setTooltip(editor->materialText("开启后 C 随 P 保持相对位置；默认关闭。", "When enabled, C follows P by its derived offset. Off by default."));
    editor->materialGapLabel.setText("C - P ms", juce::dontSendNotification);
    editor->materialVowelNameLabel.setText(editor->materialText("名称", "Name"), juce::dontSendNotification);
    editor->materialVowelWeightLabel.setText(editor->materialText("拉伸权重", "Stretch weight"), juce::dontSendNotification);
    editor->materialVowelWeight.setTooltip(editor->materialText("正权重控制各手动元音分配的拉伸时长；不改变 C/P 固定区。", "Positive relative weight distributes elastic vowel time; it never changes the fixed C/P zone."));
    editor->materialSplitLeftLabel.setText(editor->materialText("左元音名", "Left name"), juce::dontSendNotification);
    editor->materialSplitRightLabel.setText(editor->materialText("右元音名", "Right name"), juce::dontSendNotification);
    editor->materialSplitConfirm.setButtonText(editor->materialText("确认命名切分", "Confirm split"));
    editor->materialSplitCancel.setButtonText(editor->materialText("取消", "Cancel"));
    for (auto* component : std::initializer_list<juce::Component*>{&editor->materialKindLabel, &editor->materialSegmentLabel,
        &editor->materialSegmentChoice, &editor->materialPronunciationKindChoice, &editor->materialSplitSegment,
        &editor->materialMergeSegment, &editor->materialSimplifiedConsonant, &editor->materialGapLabel,
        &editor->materialConsonantGap, &editor->materialVowelNameLabel, &editor->materialVowelName,
        &editor->materialVowelWeightLabel, &editor->materialVowelWeight, &editor->materialSplitLeftLabel,
        &editor->materialSplitRightLabel, &editor->materialSplitLeftName, &editor->materialSplitRightName,
        &editor->materialSplitConfirm, &editor->materialSplitCancel}) editor->addAndMakeVisible(component);
    editor->materialSegmentChoice.onChange = [safe]
    {
        if (!safe || safe->refreshingMaterialVowel || safe->loadingMaterial) return;
        if (!safe->commitMaterialVowelFields())
        {
            const auto old = std::find(safe->materialVowelIds.begin(), safe->materialVowelIds.end(), safe->materialVowelEditId);
            if (old != safe->materialVowelIds.end()) safe->materialSegmentChoice.setSelectedId(
                static_cast<int>(old - safe->materialVowelIds.begin()) + 1, juce::dontSendNotification);
            return;
        }
        safe->refreshNativeSegmentControls();
    };
    editor->materialVowelName.onReturnKey = editor->materialVowelWeight.onReturnKey = [safe]
    { if (safe) safe->commitMaterialVowelFields(); };
    editor->materialVowelName.onFocusLost = editor->materialVowelWeight.onFocusLost = editor->materialVowelName.onReturnKey;
    editor->materialSplitConfirm.onClick = [safe] { if (safe) safe->confirmMaterialVowelSplit(); };
    editor->materialSplitCancel.onClick = [safe] { if (safe) safe->cancelMaterialVowelSplit(); };
    editor->materialPronunciationKindChoice.onChange = [safe]
    {
        if (!safe || safe->loadingMaterial || !safe->materialDetail) return;
        safe->materialDetailDraft.pronunciationKind = static_cast<PronunciationKind>(
            juce::jlimit(1, 3, safe->materialPronunciationKindChoice.getSelectedId()) - 1);
    };
    editor->materialSimplifiedConsonant.onClick = [safe]
    {
        if (!safe || safe->loadingMaterial || !safe->materialDetail) return;
        const bool enabled = safe->materialSimplifiedConsonant.getToggleState();
        juce::String error;
        if (!safe->syncMaterialTiming(error) || !setSimplifiedConsonant(safe->materialDetailDraft,
            enabled, -1.0, error))
            safe->helpLabel.setText(error, juce::dontSendNotification);
        else safe->refreshMaterialTimingControls();
    };
    editor->materialConsonantGap.onReturnKey = [safe] { if (safe) safe->commitMaterialConsonantGap(); };
    editor->materialConsonantGap.onFocusLost = editor->materialConsonantGap.onReturnKey;
    editor->materialSplitSegment.onClick = [safe] { if (safe) safe->beginMaterialVowelSplit(); };
    editor->materialMergeSegment.onClick = [safe]
    {
        if (!safe || !safe->waveform.nativeDetail || !safe->commitMaterialVowelFields()) return;
        const auto index = safe->selectedMaterialVowelIndex();
        if (index < 0) return;
        auto row = safe->materialDetailDraft; row.segments = safe->waveform.nativeSegments;
        juce::String error;
        if (!MaterialAnnotationSession::planMergeVowels(row, row.segments[static_cast<std::size_t>(index)].id, error))
        { safe->helpLabel.setText(error, juce::dontSendNotification); return; }
        safe->materialDetailDraft = std::move(row); safe->waveform.nativeSegments = safe->materialDetailDraft.segments;
        safe->cancelMaterialVowelSplit(); safe->refreshNativeSegmentControls(); safe->waveform.repaint();
    };
    editor->materialOverview->refreshing = true;
    editor->materialOverview->list.deselectAllRows();
    editor->materialOverview->refreshing = false;
    editor->materialOverview->activeId.clear(); editor->materialOverview->showFields();
    editor->materialDetail = false;
    editor->refreshMaterialVisibility();
    editor->lookAndFeelChanged();
    editor->setSize(quick ? 1000 : 1100, 780);
    if (error) error->clear();
    return editor;
}

void OtoWaveformEditorComponent::setPublicationValidator(std::function<bool(
    const std::vector<SampleRegionSetting>&, juce::String&)> validator)
{
    if (materialSession) materialSession->validatePublication = std::move(validator);
}

OtoWaveformEditorComponent::EditScope OtoWaveformEditorComponent::editScope() const
{ return materialSession ? materialSession->editScope() : EditScope::source; }

bool OtoWaveformEditorComponent::enableNoteScope(NoteScopeBinding binding, EditScope initial, juce::String& error)
{
    if (!materialSession) { error = "Note scope requires the shared material editor"; return false; }
    if (materialDetail && !commitMaterialDetail(error)) return false;
    if (!materialDetail && materialOverview && !materialOverview->commitFields())
    { error = "Invalid unpublished source fields"; return false; }
    if (!materialSession->enableNoteScope(std::move(binding), error)) return false;
    if (!materialSession->setEditScope(initial, error)) return false;
    refreshScopeView();
    return true;
}

bool OtoWaveformEditorComponent::refreshNoteScopeBinding(juce::String& error)
{
    if (!materialSession) { error = "Not a material editor"; return false; }
    if (editScope() != EditScope::source)
    { error = "Local drafts retain their original note revision; return to clean source scope before refreshing"; return false; }
    if (materialDetail && !commitMaterialDetail(error)) return false;
    if (!materialDetail && materialOverview && !materialOverview->commitFields())
    { error = "Invalid unpublished source fields"; return false; }
    return materialSession->refreshNoteScopeBinding(error); // No mode/draft replacement.
}

void OtoWaveformEditorComponent::materialSourceSaved(const std::vector<SampleRegionSetting>& rows)
{
    if (!materialSession || editScope() != EditScope::source) return;
    const auto session = materialSession;
    const auto callback = materialSaved;
    juce::Component::SafePointer<OtoWaveformEditorComponent> safe(this);
    if (callback) callback(rows); // Host publishes to the model BEFORE a fresh getter.
    if (!safe || safe->materialSession != session || session->editScope() != EditScope::source) return;
    // The detail and outer pronunciation list share a draft, but the ListBox
    // still needs its content/selection refreshed after publication.
    if (safe->materialOverview)
        safe->materialOverview->refresh(safe->materialOverview->selectedIds());
    if (!session->hasNoteScope() || !session->noteScopeBinding().refresh) return;
    juce::String error;
    if (!safe->refreshNoteScopeBinding(error) && safe)
        safe->helpLabel.setText("Source saved; note-context refresh failed: " + error, juce::dontSendNotification);
}

bool OtoWaveformEditorComponent::setEditScope(EditScope scope, juce::String& error)
{
    if (!materialSession) { error = "Not a material editor"; return false; }
    if (materialDetail && !commitMaterialDetail(error)) return false;
    if (!materialDetail && materialOverview && !materialOverview->commitFields())
    { error = "Invalid unpublished source fields"; return false; }
    if (!materialSession->setEditScope(scope, error)) return false;
    refreshScopeView();
    return true;
}

bool OtoWaveformEditorComponent::discardAndSetEditScope(EditScope scope, juce::String& error)
{
    if (!materialSession) { error = "Not a material editor"; return false; }
    if (!materialSession->discardAndSetEditScope(scope, error)) return false;
    cancelMaterialVowelSplit();
    refreshScopeView();
    return true;
}

bool OtoWaveformEditorComponent::restoreLocal(juce::String& error)
{
    if (!materialSession) { error = "Not a material editor"; return false; }
    if (materialDetail && !commitMaterialDetail(error)) return false;
    const auto session = materialSession;
    juce::Component::SafePointer<OtoWaveformEditorComponent> safe(this);
    if (!session->restoreLocal(error)) return false;
    if (safe) refreshScopeView();
    return true;
}

void OtoWaveformEditorComponent::refreshScopeView()
{
    if (!materialSession) return;
    stopPlayback();
    const bool local = editScope() == EditScope::thisNote;
    materialScope.setSelectedId(local ? 2 : 1, juce::dontSendNotification);
    materialDetailError.clear();
    if (local)
    {
        materialDetail = true;
        materialRegionId = materialSession->noteScopeBinding().regionId;
        loadMaterialDetail(materialSession->rows().front());
        materialScope.setTooltip(materialSession->noteScopeBinding().noteId);
        helpLabel.setText(materialSession->isClassicNoteScope()
            ? materialText("仅此音符的经典 OTO 标量时序；不保存音节类型或命名元音，恢复表示继承引擎默认 OTO。", "Classic scalar OTO timing for this note only; kind/named vowels are not stored. Restore inherits engine OTO defaults.")
            : materialText("仅应用到此音符；不修改源录音、OTTO 或旧 OTO。", "Applies to this note only; recording, OTTO and old OTO unchanged."), juce::dontSendNotification);
    }
    else
    {
        materialDetail = false; materialRegionId.clear();
        materialOverview->refresh();
    }
    refreshMaterialVisibility();
}

void OtoWaveformEditorComponent::requestEditScope(EditScope target)
{
    juce::String error;
    if (setEditScope(target, error)) return;
    materialScope.setSelectedId(editScope() == EditScope::thisNote ? 2 : 1, juce::dontSendNotification);
    if (target == editScope()) { helpLabel.setText(error, juce::dontSendNotification); return; }
    juce::Component::SafePointer<OtoWaveformEditorComponent> safe(this);
    juce::AlertWindow::showOkCancelBox(juce::MessageBoxIconType::WarningIcon,
        materialText("切换编辑范围", "Change editing scope"), error + "\n\n"
            + materialText("丢弃未应用的草稿并重新载入目标范围？", "Discard the unpublished draft and reload the destination scope?"),
        materialText("丢弃并切换", "Discard and switch"), materialText("取消", "Cancel"), this,
        juce::ModalCallbackFunction::create([safe, target](int accepted)
        {
            if (!safe || !accepted) return;
            juce::String problem;
            if (!safe->discardAndSetEditScope(target, problem)) safe->helpLabel.setText(problem, juce::dontSendNotification);
        }));
}

void OtoWaveformEditorComponent::requestRestoreLocal()
{
    juce::String error;
    if (restoreLocal(error)) return;
    if (!materialSession || editScope() != EditScope::thisNote || !materialSession->canRestoreLocal() || !materialSession->dirty())
    { helpLabel.setText(error, juce::dontSendNotification); return; }
    juce::Component::SafePointer<OtoWaveformEditorComponent> safe(this);
    juce::AlertWindow::showOkCancelBox(juce::MessageBoxIconType::WarningIcon,
        materialText("恢复源设置", "Restore source settings"), error + "\n\n"
            + materialText("丢弃未应用的草稿并恢复此音符的源设置？", "Discard the unpublished draft and restore this note's source settings?"),
        materialText("丢弃并恢复", "Discard and restore"), materialText("取消", "Cancel"), this,
        juce::ModalCallbackFunction::create([safe](int accepted)
        {
            if (!safe || !accepted) return;
            juce::String problem;
            if (!safe->discardAndSetEditScope(EditScope::thisNote, problem) || !safe->restoreLocal(problem))
                safe->helpLabel.setText(problem, juce::dontSendNotification);
        }));
}

std::vector<SampleRegionSetting> OtoWaveformEditorComponent::diagnosticMaterialRows() const
{ return materialSession ? materialSession->rows() : std::vector<SampleRegionSetting>{}; }

bool OtoWaveformEditorComponent::diagnosticCreateMaterialAlias(double start, double end,
    const juce::String& alias, juce::String& error)
{
    if (!materialSession) { error = "Not a material editor"; return false; }
    if (materialDetail && !commitMaterialDetail(error)) return false;
    if (!materialSession->createPronunciation(start, end, alias, error)) return false;
    materialDetail = false; materialOverview->refresh(); refreshMaterialVisibility(); return true;
}

bool OtoWaveformEditorComponent::diagnosticRemoveMaterialAlias(const juce::String& id,
    const juce::String& alias, juce::String& error)
{
    if (!materialSession) { error = "Not a material editor"; return false; }
    if (materialDetail && !commitMaterialDetail(error)) return false;
    if (!materialSession->removeAlias(id, alias, error)) return false;
    materialDetail = false; materialOverview->refresh(); refreshMaterialVisibility(); return true;
}

bool OtoWaveformEditorComponent::diagnosticOuterRangeLocked() const { return waveform.lockOuterRange; }

std::optional<juce::Range<double>> OtoWaveformEditorComponent::diagnosticMaterialSourceSelection() const
{
    if (materialDetail) return juce::Range<double>(materialDetailDraft.regionStartSeconds, materialDetailDraft.regionEndSeconds);
    if (materialOverview && materialOverview->rangeSelected)
        return juce::Range<double>(materialOverview->selectedRangeStart, materialOverview->selectedRangeEnd);
    if (materialOverview && materialSession)
    {
        const auto ids = materialOverview->selectedIds();
        for (const auto& row : materialSession->rows())
            if (std::find(ids.begin(), ids.end(), row.materialRegionId) != ids.end())
                return juce::Range<double>(row.regionStartSeconds, row.regionEndSeconds);
    }
    return {};
}

void OtoWaveformEditorComponent::bindMaterialViewPreferences(juce::PropertiesFile& preferences)
{
    if (!materialSession || editScope() == EditScope::thisNote) return;
    materialViewPreferences = &preferences;
    auto path = materialSession->sourceAudio().getFullPathName().replaceCharacter('\\', '/');
   #if JUCE_WINDOWS
    path = path.toLowerCase();
   #endif
    materialViewPreferenceKey = "material.singleSyllable." + juce::SHA256(path.toRawUTF8(), path.getNumBytesAsUTF8()).toHexString();
    const bool enabled = preferences.getBoolValue(materialViewPreferenceKey, false);
    if (enabled && !materialDetail && materialRegionId.isEmpty() && materialOverview)
    {
        const auto preferred = preferences.getValue(materialViewPreferenceKey + ".region");
        for (const auto& row : materialSession->rows())
            if (row.materialRegionId == preferred) { materialOverview->refresh({preferred}); break; }
    }
    setSingleSyllableMaterial(enabled);
}

bool OtoWaveformEditorComponent::setSingleSyllableMaterial(bool enabled)
{
    if (!materialSession || editScope() == EditScope::thisNote) return false;
    if (enabled == singleSyllableMaterial) return true;
    juce::String error;
    if (materialDetail && !commitMaterialDetail(error))
    { helpLabel.setText(error, juce::dontSendNotification); return false; }
    if (!materialDetail && materialOverview && !materialOverview->commitFields()) return false;
    auto id = materialRegionId;
    if (!materialDetail && materialOverview && materialOverview->activeRow()) id = materialOverview->activeRow()->materialRegionId;
    if (enabled && materialSession->rows().empty())
    {
        const auto range = diagnosticMaterialSourceSelection().value_or(juce::Range<double>(0.0, materialSession->durationSeconds()));
        if (!materialSession->createPronunciation(range.getStart(), range.getEnd(), "-", error))
        { materialOverview->showError(error); return false; }
        id = materialSession->rows().front().materialRegionId;
    }
    if (enabled && std::none_of(materialSession->rows().begin(), materialSession->rows().end(),
        [&](const auto& row) { return row.materialRegionId == id; })) id = materialSession->rows().front().materialRegionId;
    singleSyllableMaterial = enabled;
    materialSingleSyllable.setToggleState(enabled, juce::dontSendNotification);
    stopPlayback(); materialDetail = enabled; materialRegionId = id;
    if (enabled)
    {
        for (const auto& row : materialSession->rows()) if (row.materialRegionId == id) { loadMaterialDetail(row); break; }
    }
    else
    {
        materialOverview->rangeSelected = false; materialOverview->selectedRangeAnchor.clear();
        materialOverview->refresh({id});
    }
    if (materialViewPreferences)
    {
        materialViewPreferences->setValue(materialViewPreferenceKey, enabled);
        if (enabled) materialViewPreferences->setValue(materialViewPreferenceKey + ".region", id);
        materialViewPreferences->saveIfNeeded();
    }
    refreshMaterialVisibility(); return true;
}

bool OtoWaveformEditorComponent::focusMaterialSourceRange(juce::Range<double> range, bool detail)
{
    if (!materialSession || !std::isfinite(range.getStart()) || !std::isfinite(range.getEnd())) return false;
    range = range.getIntersectionWith({0.0, materialSession->durationSeconds()});
    if (range.getLength() <= 1.e-9) return false;
    juce::String error;
    if (materialDetail && !commitMaterialDetail(error)) { helpLabel.setText(error, juce::dontSendNotification); return false; }
    if (!materialDetail && materialOverview && !materialOverview->commitFields()) return false;
    std::vector<juce::String> exact, containing, overlapping;
    for (const auto& row : materialSession->rows())
    {
        if (std::abs(row.regionStartSeconds-range.getStart()) < 1.e-6 && std::abs(row.regionEndSeconds-range.getEnd()) < 1.e-6)
            exact.push_back(row.materialRegionId);
        if (row.regionStartSeconds <= range.getStart()+1.e-6 && row.regionEndSeconds >= range.getEnd()-1.e-6)
            containing.push_back(row.materialRegionId);
        if (std::min(row.regionEndSeconds,range.getEnd())-std::max(row.regionStartSeconds,range.getStart()) > 1.e-6)
            overlapping.push_back(row.materialRegionId);
    }
    const auto& matches = !exact.empty() ? exact : !containing.empty() ? containing : overlapping;
    if (matches.size() == 1) return selectMaterialPronunciation(matches.front(), detail);
    // Keep the current authored pronunciation when the source range is ambiguous.
    if (singleSyllableMaterial) return materialDetail;
    // No unique authored pronunciation: show the exact physical range without
    // inventing an alias or silently choosing an overlapping alternative.
    stopPlayback(); materialDetail = false; materialRegionId.clear();
    if (materialOverview) materialOverview->selectSourceRange(range);
    refreshMaterialVisibility(); return true;
}

bool OtoWaveformEditorComponent::selectMaterialPronunciation(const juce::String& id, bool detail)
{
    if (!materialSession) return false;
    detail = detail || singleSyllableMaterial;
    juce::String error;
    if (materialDetail && !commitMaterialDetail(error)) { helpLabel.setText(error, juce::dontSendNotification); return false; }
    if (!materialDetail && materialOverview && !materialOverview->commitFields()) return false;
    for (const auto& row : materialSession->rows())
        if (row.materialRegionId == id)
        {
            materialRegionId = id;
            if (singleSyllableMaterial && materialViewPreferences)
            {
                materialViewPreferences->setValue(materialViewPreferenceKey + ".region", id);
                materialViewPreferences->saveIfNeeded();
            }
            if (detail) { materialDetail = true; loadMaterialDetail(row); }
            else if (materialOverview)
            {
                stopPlayback(); materialDetail = false;
                materialOverview->rangeSelected = false; materialOverview->selectedRangeAnchor.clear();
                materialOverview->refresh({id});
            }
            refreshMaterialVisibility(); return true;
        }
    return false;
}

bool OtoWaveformEditorComponent::diagnosticMaterialBack()
{
    if (!materialSession || !materialDetail || singleSyllableMaterial || editScope() == EditScope::thisNote) return false;
    juce::String error;
    if (!commitMaterialDetail(error)) { helpLabel.setText(error, juce::dontSendNotification); return false; }
    stopPlayback(); materialDetail = false;
    materialOverview->refresh({materialRegionId}); refreshMaterialVisibility(); return true;
}

void OtoWaveformEditorComponent::diagnosticMaterialAdjacent(bool enabled)
{
    if (materialOverview) materialOverview->adjacentPronunciation.setToggleState(enabled, juce::dontSendNotification);
}
juce::Component* OtoWaveformEditorComponent::diagnosticMaterialOverviewWaveform()
{ return materialOverview ? &materialOverview->overviewWaveform() : nullptr; }
void OtoWaveformEditorComponent::diagnosticMaterialAliasText(const juce::String& text)
{ if (materialOverview) materialOverview->alias.setText(text, false); }
void OtoWaveformEditorComponent::diagnosticCreateSelectedMaterialAlias()
{ if (materialOverview) materialOverview->createRangeAlias(); }
void OtoWaveformEditorComponent::diagnosticSimplifiedConsonant(bool enabled)
{
    materialSimplifiedConsonant.setToggleState(enabled, juce::dontSendNotification);
    if (materialSimplifiedConsonant.onClick) materialSimplifiedConsonant.onClick();
}
void OtoWaveformEditorComponent::diagnosticConsonantGap(const juce::String& text)
{ materialConsonantGap.setText(text, false); commitMaterialConsonantGap(); }

void OtoWaveformEditorComponent::refreshMaterialTimingControls()
{
    if (!materialSession || !materialDetail) return;
    const auto timing = pronunciationTimingFor(materialDetailDraft);
    edited.offsetMs = timing.sourceStartSeconds * 1000;
    edited.cutoffMs = (materialSession->durationSeconds() - timing.sourceEndSeconds) * 1000;
    edited.consonantMs = (timing.consonantEndSeconds - timing.sourceStartSeconds) * 1000;
    edited.preutteranceMs = (timing.alignmentSeconds - timing.sourceStartSeconds) * 1000;
    edited.overlapMs = timing.overlapSeconds * 1000;
    if (materialSession->isClassicNoteScope())
    {
        backend::UtauOtoOverride scalar; juce::String error;
        if (!ProjectModel::makeClassicNoteMaterialOto(materialDetailDraft, scalar, error))
        { materialDetailError = error; return; }
        // The existing scalar adapter preserves negative-cutoff duration and
        // relative C/P/O. Native/source display remains absolute-end based.
        edited = SampleSettings::entryWithNoteOto(edited, scalar);
    }
    edited.jieOnsetMs = edited.consonantMs;
    materialTimingEntry = edited;
    waveform.nativeSegments = materialDetailDraft.segments;
    materialSimplifiedConsonant.setToggleState(timing.simplifiedConsonant, juce::dontSendNotification);
    materialConsonantGap.setEnabled(timing.simplifiedConsonant);
    materialConsonantGap.setText(formatNumber(consonantAfterAlignmentSeconds(timing) * 1000), false);
    refreshEditors(); refreshNativeSegmentControls();
}

bool OtoWaveformEditorComponent::syncMaterialTiming(juce::String& error)
{
    if (!materialSession || !materialDetail || loadingMaterial) return true;
    auto row = materialDetailDraft;
    row.segments = waveform.nativeSegments;
    const auto requested = edited;
    if (materialSession->isClassicNoteScope())
    {
        if (!classicScalarProjection)
        { error = "No authoritative classic scalar projection is bound"; return false; }
        // Reuse strict OTO conversion on the exact pinned source row, rather
        // than applying native absolute-boundary edits to relative OTO fields.
        if (!classicScalarProjection(requested, row, error)) return false;
        materialDetailDraft = std::move(row);
        refreshMaterialTimingControls();
        return true;
    }
    const bool startChanged = !waveform.lockOuterRange && requested.offsetMs != materialTimingEntry.offsetMs;
    const bool endChanged = !waveform.lockOuterRange && requested.cutoffMs != materialTimingEntry.cutoffMs;
    const bool alignmentChanged = requested.preutteranceMs != materialTimingEntry.preutteranceMs || startChanged;
    const bool consonantChanged = requested.consonantMs != materialTimingEntry.consonantMs || startChanged;
    if (startChanged || endChanged || requested.overlapMs != materialTimingEntry.overlapMs)
    {
        auto timing = pronunciationTimingFor(row);
        if (startChanged) timing.sourceStartSeconds = requested.offsetMs / 1000;
        if (endChanged) timing.sourceEndSeconds = requested.cutoffMs < 0
            ? requested.offsetMs / 1000 - requested.cutoffMs / 1000
            : materialSession->durationSeconds() - requested.cutoffMs / 1000;
        timing.overlapSeconds = requested.overlapMs / 1000;
        if (!applyPronunciationTiming(row, timing, error)) return false;
    }
    if (alignmentChanged || consonantChanged)
    {
        detailBaseline.jieGlideMs = requested.jieGlideMs; detailBaseline.jieNucleusMs = requested.jieNucleusMs;
    }
    if (alignmentChanged && !editPronunciationAlignment(row, (requested.offsetMs + requested.preutteranceMs) / 1000, error)) return false;
    if (consonantChanged && !editPronunciationConsonant(row, (requested.offsetMs + requested.consonantMs) / 1000, error)) return false;
    materialDetailDraft = std::move(row);
    refreshMaterialTimingControls(); return true;
}

void OtoWaveformEditorComponent::commitMaterialConsonantGap()
{
    if (!materialSession || !materialDetail || loadingMaterial || !materialDetailDraft.simplifiedConsonant) return;
    const auto text = materialConsonantGap.getText().trim();
    const auto* begin = text.toRawUTF8(); char* end = nullptr;
    const auto value = std::strtod(begin, &end);
    juce::String error;
    if (text.isEmpty() || end == begin || *end != 0 || !std::isfinite(value) || value < 0)
        error = materialText("C - P 必须为非负有限毫秒数。", "C - P must be finite nonnegative milliseconds.");
    else if (syncMaterialTiming(error) && setSimplifiedConsonant(materialDetailDraft, true, value / 1000, error))
    { refreshMaterialTimingControls(); return; }
    helpLabel.setText(error, juce::dontSendNotification);
}

void OtoWaveformEditorComponent::loadMaterialDetail(const SampleRegionSetting& row)
{
    loadingMaterial = true;
    materialVowelSplitPending = false; materialPendingSplitId.clear(); materialVowelEditId.clear(); materialVowelIds.clear();
    materialRegionId = row.materialRegionId;
    const int mode = materialMode.getSelectedId();
    jie = mode == 3 || mode == 4; mou = mode == 4;
    VoicebankOtoEntry entry;
    entry.audioFile = materialSession->sourceAudio(); entry.otoFile = materialSession->manifestFile();
    entry.sourceName = entry.audioFile.getFileName(); entry.alias = row.name;
    entry.offsetMs = row.regionStartSeconds * 1000;
    entry.cutoffMs = (materialSession->durationSeconds() - row.regionEndSeconds) * 1000;
    entry.consonantMs = row.fixedDurationSeconds * 1000;
    entry.preutteranceMs = (row.alignmentSeconds - row.regionStartSeconds) * 1000;
    entry.overlapMs = row.overlapSeconds * 1000; entry.hasJieOto = true;
    const auto span = (row.regionEndSeconds - row.regionStartSeconds) * 1000;
    double boundaries[]{span * .25, span * .5, span * .75};
    if (row.segments.size() >= 2 && row.segments.size() <= 4)
    {
        for (std::size_t i = 0; i < row.segments.size(); ++i)
        {
            const auto& segment = row.segments[i];
            entry.mouClasses += segment.role == NativeSegmentRole::consonant ? 'C'
                : segment.role == NativeSegmentRole::silence ? 'S' : 'V';
            if (i + 1 < row.segments.size()) boundaries[i] = (segment.sourceEndSeconds - row.regionStartSeconds) * 1000;
        }
    }
    else entry.mouClasses = "CVVV"; // Display-only fallback; never silently replaces native segments.
    entry.jieOnsetMs = boundaries[0]; entry.jieGlideMs = std::max(boundaries[0], boundaries[1]);
    entry.jieNucleusMs = std::max(entry.jieGlideMs, boundaries[2]);
    entry.jieOnsetMs = entry.consonantMs; // The same canonical C marker in every view.
    original = edited = entry; detailBaseline = entry; detailSegmentsBaseline = row.segments;
    materialDetailDraft = row; materialTimingEntry = entry;
    waveform.setTimingMode(jie, mou); waveform.nativeDetail = mode == 1;
    waveform.nativeSegments = row.segments;
    const bool nativeSource = waveform.nativeDetail && editScope() == EditScope::source;
    waveform.preferOuterRangeDrag = nativeSource && singleSyllableMaterial;
    waveform.lockOuterRange = nativeSource ? !singleSyllableMaterial : materialSession->rows().size() > 1;
    classesEditor.setText(entry.mouClasses, false);
    if (mou) syncRegionCount(); else { waveform.regions = 4; waveform.classes.clear(); }
    // Locked outer bounds are inspectable/copyable, not editable. The drag and
    // commit guards still enforce the same lock independently of presentation.
    parameterEditors[0].setEnabled(true); parameterEditors[2].setEnabled(true);
    parameterEditors[0].setReadOnly(waveform.lockOuterRange);
    parameterEditors[2].setReadOnly(waveform.lockOuterRange);
    lookAndFeelChanged();
    fileLabel.setText(entry.sourceName + "  ·  " + row.name, juce::dontSendNotification);
    saveButton.setButtonText(materialText("保存到 otto.ini", "Save to otto.ini"));
    helpLabel.setText(waveform.lockOuterRange ? materialText("多发音：外边界仅在总览编辑；此处只编辑内部参数。",
        "Multiple pronunciations: outer bounds are edited only in overview.") : materialText("原生草稿，原音频与旧 OTO 保持不变。", "Native draft; original audio and old OTO stay unchanged."), juce::dontSendNotification);
    refreshEditors(); waveform.frameEntry(); loadingMaterial = false;
    refreshMaterialTimingControls(); detailBaseline = edited;
}

int OtoWaveformEditorComponent::selectedMaterialVowelIndex() const
{
    const auto ordinal = materialSegmentChoice.getSelectedId();
    if (ordinal <= 0 || ordinal > static_cast<int>(materialVowelIds.size())) return -1;
    const auto& id = materialVowelIds[static_cast<std::size_t>(ordinal - 1)];
    for (std::size_t i = 0; i < waveform.nativeSegments.size(); ++i) if (waveform.nativeSegments[i].id == id) return static_cast<int>(i);
    return -1;
}

bool OtoWaveformEditorComponent::commitMaterialVowelFields()
{
    if (!materialSession || !materialDetail || !waveform.nativeDetail || loadingMaterial || refreshingMaterialVowel
        || materialVowelEditId.isEmpty()) return true;
    const auto text = materialVowelWeight.getText().trim(); const auto* begin = text.toRawUTF8(); char* end = nullptr;
    auto weight = std::strtod(begin, &end); juce::String error;
    auto row = materialDetailDraft; row.segments = waveform.nativeSegments;
    // Navigating away must not treat an untouched inherited '-' as a new name.
    if (text == materialVowelWeightBaseline)
        for (const auto& segment : row.segments)
            if (segment.id == materialVowelEditId && segment.alias == "-" && materialVowelName.getText() == segment.alias) return true;
    if (text == materialVowelWeightBaseline)
        for (const auto& segment : row.segments) if (segment.id == materialVowelEditId) { weight = segment.stretchWeight; break; }
    if (text.isEmpty() || end == begin || *end != 0 || !std::isfinite(weight) || weight <= 0)
        error = materialText("拉伸权重必须为正有限数。", "Stretch weight must be positive and finite.");
    else if (MaterialAnnotationSession::planEditVowel(row, materialVowelEditId, materialVowelName.getText(), weight, error))
    {
        materialDetailDraft = std::move(row); waveform.nativeSegments = materialDetailDraft.segments;
        helpLabel.setText(materialText("已更新元音名称与权重，C/P 保持不变。", "Vowel name and weight updated; C/P unchanged."), juce::dontSendNotification);
        refreshNativeSegmentControls(); waveform.repaint(); return true;
    }
    helpLabel.setText(error, juce::dontSendNotification); return false;
}

void OtoWaveformEditorComponent::beginMaterialVowelSplit()
{
    if (!materialSession || !materialDetail || !waveform.nativeDetail || !commitMaterialVowelFields()) return;
    commitEditors(); if (materialDetailError.isNotEmpty()) return;
    const auto index = selectedMaterialVowelIndex(); if (index < 0) return;
    const auto& segment = waveform.nativeSegments[static_cast<std::size_t>(index)];
    if (segment.role != NativeSegmentRole::vowel || !segment.stretchable) return;
    materialPendingSplitId = segment.id;
    materialPendingSplitCut = (segment.sourceStartSeconds + segment.sourceEndSeconds) * .5;
    materialSplitLeftName.setText(juce::String{}, false); materialSplitRightName.setText(juce::String{}, false);
    materialVowelSplitPending = true; refreshMaterialVisibility();
    if (materialSplitLeftName.isShowing()) materialSplitLeftName.grabKeyboardFocus();
}

bool OtoWaveformEditorComponent::confirmMaterialVowelSplit()
{
    if (!materialVowelSplitPending || !materialSession || !materialDetail) return false;
    auto row = materialDetailDraft; row.segments = waveform.nativeSegments; juce::String error;
    if (!MaterialAnnotationSession::planSplitVowel(row, materialPendingSplitId, materialPendingSplitCut,
        materialSplitLeftName.getText(), materialSplitRightName.getText(), error))
    { helpLabel.setText(error, juce::dontSendNotification); return false; }
    materialDetailDraft = std::move(row); waveform.nativeSegments = materialDetailDraft.segments;
    materialVowelSplitPending = false; materialPendingSplitId.clear();
    helpLabel.setText(materialText("已创建命名元音分段，可选择各段调整权重。", "Named vowel parts created; select a part to adjust its weight."), juce::dontSendNotification);
    refreshNativeSegmentControls(); refreshMaterialVisibility(); waveform.repaint(); return true;
}
void OtoWaveformEditorComponent::cancelMaterialVowelSplit()
{
    materialVowelSplitPending = false; materialPendingSplitId.clear();
    if (materialSession) refreshMaterialVisibility();
}
void OtoWaveformEditorComponent::diagnosticBeginMaterialVowelSplit() { beginMaterialVowelSplit(); }
bool OtoWaveformEditorComponent::diagnosticConfirmMaterialVowelSplit(const juce::String& left, const juce::String& right)
{
    materialSplitLeftName.setText(left, false); materialSplitRightName.setText(right, false);
    return confirmMaterialVowelSplit();
}
void OtoWaveformEditorComponent::diagnosticCancelMaterialVowelSplit() { cancelMaterialVowelSplit(); }
bool OtoWaveformEditorComponent::diagnosticSplitMaterialVowel(const juce::String& left, const juce::String& right)
{ beginMaterialVowelSplit(); return diagnosticConfirmMaterialVowelSplit(left, right); }
bool OtoWaveformEditorComponent::diagnosticEditMaterialVowel(const juce::String& name, const juce::String& weight)
{
    materialVowelName.setText(name, false); materialVowelWeight.setText(weight, false); return commitMaterialVowelFields();
}
juce::StringArray OtoWaveformEditorComponent::diagnosticMaterialVowelItems() const
{
    juce::StringArray items; for (int i = 0; i < materialSegmentChoice.getNumItems(); ++i) items.add(materialSegmentChoice.getItemText(i)); return items;
}

void OtoWaveformEditorComponent::refreshNativeSegmentControls()
{
    if (!materialSession) return;
    refreshingMaterialVowel = true;
    const auto selected = materialSegmentChoice.getSelectedId();
    const auto selectedId = selected > 0 && selected <= static_cast<int>(materialVowelIds.size())
        ? materialVowelIds[static_cast<std::size_t>(selected - 1)] : juce::String{};
    materialSegmentChoice.clear(juce::dontSendNotification); materialVowelIds.clear(); int current = 0;
    for (const auto& segment : waveform.nativeSegments)
    {
        if (segment.role != NativeSegmentRole::vowel || !segment.stretchable) continue;
        materialVowelIds.push_back(segment.id); const auto ordinal = static_cast<int>(materialVowelIds.size());
        materialSegmentChoice.addItem(juce::String(ordinal) + ": " + segment.alias
            + " [" + juce::String(segment.sourceStartSeconds * 1000, 1) + ", "
            + juce::String(segment.sourceEndSeconds * 1000, 1) + "]", ordinal);
        if (segment.id == selectedId) current = ordinal;
    }
    if (current == 0 && !materialVowelIds.empty()) current = 1;
    materialSegmentChoice.setSelectedId(current, juce::dontSendNotification);
    const auto index = selectedMaterialVowelIndex(); const bool any = index >= 0;
    const bool scalar = materialSession->isClassicNoteScope();
    materialVowelEditId = any ? waveform.nativeSegments[static_cast<std::size_t>(index)].id : juce::String{};
    materialVowelName.setText(any ? waveform.nativeSegments[static_cast<std::size_t>(index)].alias : juce::String{}, false);
    const auto weight = any ? waveform.nativeSegments[static_cast<std::size_t>(index)].stretchWeight : 1.0;
    materialVowelWeightBaseline = any ? (weight >= 1.e-6 && weight <= 1.e8
        ? juce::String(weight, 9).trimCharactersAtEnd("0").trimCharactersAtEnd(".") : juce::String(weight, 16, true)) : juce::String{};
    materialVowelWeight.setText(materialVowelWeightBaseline, false);
    materialVowelName.setEnabled(any && !scalar); materialVowelWeight.setEnabled(any && !scalar);
    materialPronunciationKindChoice.setEnabled(materialDetail && !scalar);
    materialSimplifiedConsonant.setEnabled(!scalar);
    materialConsonantGap.setEnabled(!scalar && materialDetailDraft.simplifiedConsonant);
    materialPronunciationKindChoice.setSelectedId(static_cast<int>(materialDetailDraft.pronunciationKind) + 1, juce::dontSendNotification);
    const auto elastic = [&](int i)
    { return juce::isPositiveAndBelow(i, static_cast<int>(waveform.nativeSegments.size()))
        && waveform.nativeSegments[static_cast<std::size_t>(i)].role == NativeSegmentRole::vowel
        && waveform.nativeSegments[static_cast<std::size_t>(i)].stretchable; };
    materialSplitSegment.setEnabled(any && !scalar);
    materialMergeSegment.setEnabled(any && !scalar && elastic(index + 1));
    refreshingMaterialVowel = false;
}

void OtoWaveformEditorComponent::refreshMaterialVisibility()
{
    if (!materialSession) return;
    for (int i = 0; i < getNumChildComponents(); ++i) getChildComponent(i)->setVisible(materialDetail);
    materialMode.setVisible(true); materialModeLabel.setVisible(true);
    const bool local = editScope() == EditScope::thisNote;
    materialBackButton.setVisible(materialDetail && !local && !singleSyllableMaterial);
    const bool nativeView = materialMode.getSelectedId() == 1;
    materialSingleSyllable.setVisible(nativeView);
    materialSingleSyllable.setEnabled(nativeView && !local);
    materialScope.setVisible(materialSession->hasNoteScope());
    materialRestoreLocal.setVisible(local);
    materialRestoreLocal.setEnabled(materialSession->canRestoreLocal());
    const bool scalar = materialSession->isClassicNoteScope();
    materialMode.setEnabled(!scalar);
    materialRestoreLocal.setButtonText(scalar
        ? materialText("继承引擎 OTO 默认值", "Inherit engine OTO defaults")
        : materialText("恢复源设置", "Restore source settings"));
    if (materialSession->hasNoteScope())
        saveButton.setButtonText(local ? materialText("应用到此音符", "Apply to this note") : materialText("保存源标注", "Save source annotations"));
    materialOverview->setVisible(!materialDetail);
    if (materialDetail)
    {
        applyParameterVisibility();
        for (auto* component : std::initializer_list<juce::Component*>{&classesLabel, &classesEditor, &consonantLabel}) component->setVisible(mou);
        classesEditor.setReadOnly(true);
        for (auto& button : countButtons) { button.setVisible(mou); button.setEnabled(false); }
        for (auto& button : consonantButtons) { button.setVisible(mou); button.setEnabled(false); }
        for (auto* component : std::initializer_list<juce::Component*>{&materialSegmentLabel, &materialSegmentChoice,
            &materialSplitSegment, &materialMergeSegment, &materialVowelNameLabel, &materialVowelName,
            &materialVowelWeightLabel, &materialVowelWeight}) component->setVisible(waveform.nativeDetail);
        for (auto* component : std::initializer_list<juce::Component*>{&materialSplitLeftLabel, &materialSplitLeftName,
            &materialSplitRightLabel, &materialSplitRightName, &materialSplitConfirm, &materialSplitCancel})
            component->setVisible(waveform.nativeDetail && materialVowelSplitPending);
        materialPronunciationKindChoice.setVisible(true);
        refreshNativeSegmentControls();
    }
    resized(); repaint();
}

OtoWaveformEditorComponent::OtoWaveformEditorComponent(
    VoicebankOtoEntry entry, bool jieMode, bool mouMode,
    std::function<void()> savedCallback)
    : jie(jieMode), mou(mouMode), original(entry), edited(std::move(entry)),
      onSaved(std::move(savedCallback)),
      waveform(edited, jieMode, mouMode, [this]
      {
          if (materialSession && materialDetail && !loadingMaterial)
          {
              juce::String error;
              if (!syncMaterialTiming(error)) helpLabel.setText(error, juce::dontSendNotification);
          }
          refreshEditors(); if (materialSession) refreshNativeSegmentControls();
      })
{
    setOpaque(true);
    // Opening an entry the voicebank has not been extended for yet: start from
    // the classic layout written as four regions, so the boundaries exist and
    // can be dragged apart rather than appearing stacked at zero.
    if (jie)
    {
        const auto span = std::max(0.0, waveform.entrySpanMilliseconds());
        if (!edited.hasJieOto)
        {
            // Extending a classic entry: the onset is its consonant and the
            // glide starts empty, the way seeding a whole voicebank writes it.
            edited.jieOnsetMs = juce::jlimit(0.0, span, edited.consonantMs);
            edited.jieGlideMs = edited.jieOnsetMs;
            edited.jieNucleusMs = span;
        }
        if (edited.jieOnsetMs <= 1.0e-6 && edited.jieGlideMs <= 1.0e-6)
        {
            // Nothing has been marked: a sample with no consonant to seed
            // from, or one already saved while still in that state.  Every
            // inner boundary sits on zero, stacking the handles on the left
            // edge where none of them can be told apart or grabbed.  Quarters
            // give each region a span wide enough to drag from.  Checked here
            // rather than only while seeding, because saving once makes the
            // entry look extended while leaving it just as unusable.
            edited.jieOnsetMs = span * 0.25;
            edited.jieGlideMs = span * 0.50;
            edited.jieNucleusMs = span * 0.75;
        }
    }
    fileLabel.setText(edited.sourceName + (edited.alias.isNotEmpty()
        ? utf8("  ·  别名：") + edited.alias : juce::String{}), juce::dontSendNotification);
    fileLabel.setFont(fileLabel.getFont().boldened());
    fileLabel.setTooltip(edited.audioFile.getFullPathName());
    addAndMakeVisible(fileLabel);

    helpLabel.setText(utf8("拖动彩色边界或线条调整参数；数值单位均为 ms"),
                      juce::dontSendNotification);
    helpLabel.setColour(juce::Label::textColourId, Palette::textMuted);
    helpLabel.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(helpLabel);

    // Every label, not only the ones on screen now: a two-region entry opens
    // with the last two hidden, and they used to become visible with no text
    // at all the moment the count grew.
    for (std::size_t index = 0; index < parameterNames.size(); ++index)
        parameterLabels[index].setText(
            index >= 5 && mou
                ? utf8("第") + juce::String(static_cast<int>(index) - 4) + utf8("区末")
                : utf8(parameterNames[index]),
            juce::dontSendNotification);
    // Built for every parameter, not only the ones this entry opens with: an
    // entry that opens at two regions used to leave the last boundaries out of
    // the component altogether, so growing it to four showed nothing where
    // they should have been.
    for (std::size_t index = 0; index < parameterEditors.size(); ++index)
    {
        parameterLabels[index].setJustificationType(juce::Justification::centredRight);
        addAndMakeVisible(parameterLabels[index]);
        parameterEditors[index].setInputRestrictions(16, "0123456789.-");
        parameterEditors[index].setSelectAllWhenFocused(true);
        parameterEditors[index].onFocusLost = [this] { commitEditors(); };
        parameterEditors[index].onReturnKey = [this] { commitEditors(); };
        addAndMakeVisible(parameterEditors[index]);
    }
    applyParameterVisibility();
    addAndMakeVisible(waveform);

    // Button captions stay ASCII: the axis letter tells the two pairs apart
    // at 24 px, and the Chinese explanation lives in the tooltip.
    struct ZoomButton { juce::TextButton* button; const char* text; const char* tip; };
    const std::array<ZoomButton, 5> zoomButtons {{
        { &hZoomInButton,   "H+",  "横向放大（时间轴）" },
        { &hZoomOutButton,  "H-",  "横向缩小（时间轴）" },
        { &vZoomInButton,   "V+",  "纵向放大（波形振幅）" },
        { &vZoomOutButton,  "V-",  "纵向缩小（波形振幅）" },
        { &zoomResetButton, "FIT", "回到本条目的取样范围" }
    }};
    for (const auto& item : zoomButtons)
    {
        item.button->setButtonText(item.text);
        item.button->setTooltip(utf8(item.tip));
        addAndMakeVisible(*item.button);
    }
    hZoomInButton.onClick  = [this] { waveform.nudgeHorizontalZoom(1.4); };
    hZoomOutButton.onClick = [this] { waveform.nudgeHorizontalZoom(1.0 / 1.4); };
    vZoomInButton.onClick  = [this] { waveform.nudgeVerticalZoom(1.4); };
    vZoomOutButton.onClick = [this] { waveform.nudgeVerticalZoom(1.0 / 1.4); };
    zoomResetButton.onClick = [this] { waveform.frameEntry(); };

    waveformScroll.setAutoHide(false);
    waveformScroll.addListener(this);
    addAndMakeVisible(waveformScroll);
    // The bar mirrors whatever moved the view, whether that was a zoom button
    // or dragging the waveform itself.
    waveform.onViewChanged = [this] { refreshScrollBar(); };

    saveButton.setButtonText(mou ? utf8("保存谋•OTO（独立）")
                             : jie ? utf8("保存界•OTO（独立）")
                                   : utf8("保存到 oto.ini"));
    if (mou)
    {
        // An entry with nothing written for it opens at CVVV -- which is what
        // an oto4 row has always meant -- rather than at an empty string the
        // field shows as CVVV and the count reads as four.  The two have to
        // agree from the first frame, or picking "4 段分" appears to do
        // nothing because it already thinks it is there.
        if (edited.mouClasses.trim().length() < 2) edited.mouClasses = "CVVV";
        classesLabel.setText(utf8("音素类别"), juce::dontSendNotification);
        classesLabel.setTooltip(utf8("每区一个字母：C 辅音 / V 元音 / S 静音。"
                                     "串长就是区数（2~4），字母说明哪一段是元音。"));
        addAndMakeVisible(classesLabel);
        classesEditor.setText(edited.mouClasses.isNotEmpty() ? edited.mouClasses
                                                             : juce::String("CVVV"),
                              false);
        classesEditor.setTooltip(classesLabel.getTooltip());
        classesEditor.onTextChange = [this]
        {
            // Only ever letters this means something in, upper case, and never
            // longer than there are regions.
            auto text = classesEditor.getText().toUpperCase()
                            .retainCharacters("CVS").substring(0, 4);
            if (text != classesEditor.getText())
                classesEditor.setText(text, false);
            if (text.length() >= 2)
            {
                edited.mouClasses = text;
                syncRegionCount();
                resized();
                refreshEditors();
            }
        };
        addAndMakeVisible(classesEditor);
        consonantLabel.setText(utf8("辅音区"), juce::dontSendNotification);
        consonantLabel.setTooltip(utf8("勾选哪几个区是辅音，可以一个不勾，也可以勾多个。"
                                       "勾上的区不参与主动拉伸，并且走辅音那套合成。"));
        addAndMakeVisible(consonantLabel);
        for (int index = 0; index < 4; ++index)
        {
            auto& tick = consonantButtons[static_cast<std::size_t>(index)];
            tick.setButtonText(utf8("第") + juce::String(index + 1) + utf8("区"));
            tick.setTooltip(consonantLabel.getTooltip());
            tick.onClick = [this, index]
            {
                setRegionIsConsonant(index,
                    consonantButtons[static_cast<std::size_t>(index)].getToggleState());
            };
            addAndMakeVisible(tick);
        }
        for (int index = 0; index < 3; ++index)
        {
            const auto count = index + 2;
            auto& button = countButtons[static_cast<std::size_t>(index)];
            button.setButtonText(juce::String(count) + utf8(" 段分"));
            button.setTooltip(utf8("把这条 oto 切成 ") + juce::String(count)
                              + utf8(" 段；波形与下面的边界数值会立刻跟着变"));
            button.setColour(juce::TextButton::buttonOnColourId, Palette::primaryButton);
            button.setColour(juce::TextButton::textColourOnId, Palette::contrastingText(Palette::primaryButton));
            button.setClickingTogglesState(true);
            button.setRadioGroupId(0x6d6f75);
            button.onClick = [this, count] { setRegionCount(count); };
            addAndMakeVisible(button);
        }
        syncRegionCount();
    }
    // 播放原音 sits at the other end of the row from Save and Cancel: listening
    // is not a way of leaving the window.
    playButton.setButtonText(utf8("播放原音"));
    playButton.setTooltip(utf8("从头播放这条 oto 所在的整个 wav 原始录音，"
                               "没经过任何引擎；再点一次停止"));
    playButton.onClick = [this] { togglePlayback(); };
    addAndMakeVisible(playButton);
    saveButton.onClick = [this] { save(); };
    addAndMakeVisible(saveButton);
    cancelButton.setButtonText(utf8("取消"));
    cancelButton.onClick = [this] { closeWindow(); };
    addAndMakeVisible(cancelButton);

    pitchReadoutTitle.setText(utf8("原始 F0"), juce::dontSendNotification);
    pitchReadoutHz.setText("-- Hz", juce::dontSendNotification);
    pitchReadoutTime.setText(utf8("点击音高线"), juce::dontSendNotification);
    pitchReadoutTitle.setFont(12.0f);
    pitchReadoutHz.setFont(18.0f);
    pitchReadoutTime.setFont(11.0f);
    pitchReadoutHz.setColour(juce::Label::textColourId, Palette::contrastingText(Palette::graphBackground));
    for (auto* label : { &pitchReadoutTitle, &pitchReadoutHz, &pitchReadoutTime })
    {
        label->setJustificationType(juce::Justification::centred);
        label->setInterceptsMouseClicks(false, false);
        addAndMakeVisible(*label);
    }
    waveform.onPitchSelected = [this](std::optional<backend::FcpeFrame> frame)
    {
        pitchReadoutHz.setText(frame ? juce::String(440.0 * std::pow(2.0, (frame->midi - 69.0) / 12.0), 1) + " Hz"
                                    : juce::String("-- Hz"), juce::dontSendNotification);
        pitchReadoutTime.setText(frame ? juce::String(frame->timeSeconds, 3) + " s"
                                      : utf8("点击音高线"), juce::dontSendNotification);
    };

    spectrumButton.setButtonText(utf8("频谱"));
    sourceF0Button.setButtonText(utf8("原始 F0"));
    spectrumButton.setTooltip(utf8("显示原始录音的时频图，与波形共用时间轴"));
    sourceF0Button.setTooltip(utf8("在波形和频谱上同时显示原始基频；点击音高线在左侧查看该帧 Hz，悬停查看音名。清音与静音处断开"));
    for (auto* button : { &spectrumButton, &sourceF0Button })
    {
        button->setToggleState(true, juce::dontSendNotification);
        button->onClick = [this]
        { waveform.setAnalysisLayers(spectrumButton.getToggleState(), sourceF0Button.getToggleState()); };
        addAndMakeVisible(*button);
    }
    setLookAndFeel(&editorLookAndFeel);
    setSize(1040, 620);
    refreshEditors();
}

void OtoWaveformEditorComponent::lookAndFeelChanged()
{
    editorLookAndFeel.refreshColours();
    helpLabel.setColour(juce::Label::textColourId, Palette::contrastingText(Palette::panel, true));
    for (auto* label : { &pitchReadoutTitle, &pitchReadoutHz, &pitchReadoutTime })
        label->setColour(juce::Label::textColourId, Palette::contrastingText(Palette::graphBackground,
            label == &pitchReadoutTime));
    for (auto& button : countButtons)
    {
        button.setColour(juce::TextButton::buttonOnColourId, Palette::primaryButton);
        button.setColour(juce::TextButton::textColourOnId, Palette::contrastingText(Palette::primaryButton));
    }
    // TextEditor stores per-run colours; JUCE's lookAndFeelChanged only resets
    // its caret. Refresh existing text too, without rewriting text or selection.
    const auto refresh = [](auto&& self, juce::Component& component) -> void
    {
        if (auto* editor = dynamic_cast<juce::TextEditor*>(&component))
        {
            const auto background = editor->isReadOnly() ? Palette::panel : Palette::base;
            const auto foreground = Palette::contrastingText(background, editor->isReadOnly() || !editor->isEnabled());
            editor->setColour(juce::TextEditor::backgroundColourId, background);
            editor->setColour(juce::TextEditor::textColourId, foreground);
            editor->applyColourToAllText(foreground, false);
        }
        if (auto* combo = dynamic_cast<juce::ComboBox*>(&component))
        {
            combo->setColour(juce::ComboBox::backgroundColourId, Palette::base);
            combo->setColour(juce::ComboBox::textColourId, Palette::contrastingText(Palette::base));
        }
        for (int i = 0; i < component.getNumChildComponents(); ++i)
            self(self, *component.getChildComponent(i));
    };
    refresh(refresh, *this);
    repaint();
}

void OtoWaveformEditorComponent::paint(juce::Graphics& g)
{
    g.fillAll(Palette::panel);
    if (materialSession && !materialDetail) return;
    g.setColour(Palette::graphBackground);
    g.fillRoundedRectangle(pitchReadoutBounds.toFloat(), 5.0f);
    g.setColour(Palette::border);
    g.drawRoundedRectangle(pitchReadoutBounds.toFloat(), 5.0f, 1.0f);
}

void OtoWaveformEditorComponent::scrollBarMoved(juce::ScrollBar* bar, double newRangeStart)
{
    if (bar != &waveformScroll) return;
    waveform.setVisibleStartMilliseconds(newRangeStart);
}

void OtoWaveformEditorComponent::refreshScrollBar()
{
    const auto total = std::max(1.0, waveform.totalMilliseconds());
    const auto visible = waveform.visibleLengthMilliseconds();
    waveformScroll.setRangeLimits({ 0.0, total }, juce::dontSendNotification);
    waveformScroll.setCurrentRange({ waveform.visibleStartMilliseconds(),
                                     waveform.visibleStartMilliseconds() + visible },
                                   juce::dontSendNotification);
    // Nothing to scroll to while the whole file is on screen.
    waveformScroll.setEnabled(visible < total - 1.0e-6);
}

void OtoWaveformEditorComponent::resized()
{
    auto area = getLocalBounds().reduced(12);
    if (materialSession)
    {
        auto header = area.removeFromTop(32);
        materialModeLabel.setBounds(header.removeFromLeft(100));
        materialMode.setBounds(header.removeFromLeft(230).reduced(0, 2));
        materialSingleSyllable.setBounds(header.removeFromLeft(205).reduced(4, 2));
        materialBackButton.setBounds(header.removeFromRight(240).reduced(0, 2));
        if (materialSession->hasNoteScope())
        {
            auto destination = area.removeFromTop(32);
            materialScope.setBounds(destination.removeFromLeft(280).reduced(0, 2));
            materialRestoreLocal.setBounds(destination.removeFromLeft(240).reduced(8, 2));
        }
        area.removeFromTop(6);
        if (!materialDetail) { if (materialOverview) materialOverview->setBounds(area); return; }
    }
    auto title = area.removeFromTop(28);
    fileLabel.setBounds(title.removeFromLeft(title.getWidth() / 2));
    helpLabel.setBounds(title);
    area.removeFromTop(6);
    auto parameters = area.removeFromTop(34);
    // The class string sits at the left of the parameter row, ahead of the
    // numbers: it says how many of them mean anything.
    if (mou)
    {
        classesLabel.setBounds(parameters.removeFromLeft(64));
        classesEditor.setBounds(parameters.removeFromLeft(72).reduced(3, 2));
        parameters.removeFromLeft(8);
    }
    const auto order = visibleParameters();
    const auto itemWidth = std::max(108,
        parameters.getWidth() / static_cast<int>(order.size()));
    for (std::size_t position = 0; position < order.size(); ++position)
    {
        const auto index = order[position];
        auto item = position + 1 == order.size() ? parameters
                                                 : parameters.removeFromLeft(itemWidth);
        parameterLabels[index].setBounds(item.removeFromLeft(jie ? 56 : 60));
        parameterEditors[index].setBounds(item.reduced(3, 2));
    }
    area.removeFromTop(7);
    auto layers = area.removeFromTop(26);
    spectrumButton.setBounds(layers.removeFromLeft(86));
    sourceF0Button.setBounds(layers.removeFromLeft(120));
    if (materialSession)
    {
        materialSimplifiedConsonant.setBounds(layers.removeFromLeft(176));
        materialGapLabel.setBounds(layers.removeFromLeft(62));
        materialConsonantGap.setBounds(layers.removeFromLeft(82).reduced(2));
        materialKindLabel.setBounds(layers.removeFromLeft(150));
        materialPronunciationKindChoice.setBounds(layers.removeFromLeft(180).reduced(2));
    }
    if (materialSession && waveform.nativeDetail)
    {
        auto native = area.removeFromTop(30);
        materialSegmentLabel.setBounds(native.removeFromLeft(135));
        materialSegmentChoice.setBounds(native.removeFromLeft(std::max(180, native.getWidth() / 3)).reduced(2));
        materialSplitSegment.setBounds(native.removeFromLeft(180).reduced(2));
        materialMergeSegment.setBounds(native.removeFromLeft(170).reduced(2));
        auto fields = area.removeFromTop(30);
        materialVowelNameLabel.setBounds(fields.removeFromLeft(45));
        materialVowelName.setBounds(fields.removeFromLeft(220).reduced(2));
        fields.removeFromLeft(8); materialVowelWeightLabel.setBounds(fields.removeFromLeft(110));
        materialVowelWeight.setBounds(fields.removeFromLeft(100).reduced(2));
        if (materialVowelSplitPending)
        {
            auto names = area.removeFromTop(32);
            materialSplitLeftLabel.setBounds(names.removeFromLeft(70));
            materialSplitLeftName.setBounds(names.removeFromLeft(180).reduced(2));
            materialSplitRightLabel.setBounds(names.removeFromLeft(75));
            materialSplitRightName.setBounds(names.removeFromLeft(180).reduced(2));
            materialSplitConfirm.setBounds(names.removeFromLeft(130).reduced(2));
            materialSplitCancel.setBounds(names.removeFromLeft(90).reduced(2));
        }
    }
    auto buttons = area.removeFromBottom(34);
    playButton.setBounds(buttons.removeFromLeft(112).reduced(0, 2));
    cancelButton.setBounds(buttons.removeFromRight(92).reduced(0, 2));
    buttons.removeFromRight(8);
    saveButton.setBounds(buttons.removeFromRight(142).reduced(0, 2));
    area.removeFromBottom(8);
    // Zoom column down the right-hand side: horizontal pair, vertical pair,
    // then "fit to this entry".
    if (mou)
    {
        auto countColumn = area.removeFromRight(82);
        area.removeFromRight(4);
        countColumn.removeFromTop(2);
        for (auto& button : countButtons)
        {
            button.setBounds(countColumn.removeFromTop(26).reduced(0, 2));
            countColumn.removeFromTop(2);
        }
        // The ticks sit under the counts: how many regions there are, then
        // which of them are consonants.
        countColumn.removeFromTop(12);
        consonantLabel.setBounds(countColumn.removeFromTop(20));
        for (auto& tick : consonantButtons)
            tick.setBounds(countColumn.removeFromTop(24).reduced(2, 1));
    }
    // Wide enough for the captions once JUCE has taken its indents out of
     // both edges: at 26 the three-letter one had ten pixels to live in and
     // came out as an ellipsis.  Checked, not guessed -- see
     // diagnosticZoomCaptionsFit.
    auto zoomColumn = area.removeFromRight(44);
    area.removeFromRight(4);
    auto readoutColumn = area.removeFromLeft(108);
    area.removeFromLeft(4);
    readoutColumn.removeFromTop(38);
    pitchReadoutBounds = readoutColumn.removeFromTop(94);
    auto readoutContent = pitchReadoutBounds.reduced(3, 5);
    pitchReadoutTitle.setBounds(readoutContent.removeFromTop(22));
    pitchReadoutHz.setBounds(readoutContent.removeFromTop(34));
    pitchReadoutTime.setBounds(readoutContent.removeFromTop(24));
    waveformScroll.setBounds(area.removeFromBottom(12));
    waveform.setBounds(area);
    refreshScrollBar();
    zoomColumn.removeFromTop(36);            // line up with the plot, not the badges
    const auto cell = 24;
    hZoomInButton.setBounds(zoomColumn.removeFromTop(cell).reduced(1));
    hZoomOutButton.setBounds(zoomColumn.removeFromTop(cell).reduced(1));
    zoomColumn.removeFromTop(8);
    vZoomInButton.setBounds(zoomColumn.removeFromTop(cell).reduced(1));
    vZoomOutButton.setBounds(zoomColumn.removeFromTop(cell).reduced(1));
    zoomColumn.removeFromTop(8);
    zoomResetButton.setBounds(zoomColumn.removeFromTop(cell).reduced(1));
}

namespace
{
// What LookAndFeel_V2::drawButtonText leaves for the text, worked out the same
// way it does: a font from the button height, then an indent at each edge.
int buttonTextRoom(const juce::TextButton& button)
{
    auto& look = const_cast<juce::TextButton&>(button).getLookAndFeel();
    const auto font = look.getTextButtonFont(const_cast<juce::TextButton&>(button),
                                             button.getHeight());
    const auto fontHeight = juce::roundToInt(font.getHeight() * 0.6f);
    const auto cornerSize = juce::jmin(button.getHeight(), button.getWidth()) / 2;
    const auto left = juce::jmin(fontHeight,
        2 + cornerSize / (button.isConnectedOnLeft() ? 4 : 2));
    const auto right = juce::jmin(fontHeight,
        2 + cornerSize / (button.isConnectedOnRight() ? 4 : 2));
    return button.getWidth() - left - right;
}

int buttonTextWidth(const juce::TextButton& button)
{
    auto& look = const_cast<juce::TextButton&>(button).getLookAndFeel();
    return look.getTextButtonFont(const_cast<juce::TextButton&>(button),
                                  button.getHeight())
        .getStringWidth(button.getButtonText());
}
}  // namespace

bool OtoWaveformEditorComponent::diagnosticZoomCaptionsFit(juce::String* report) const
{
    const std::array<const juce::TextButton*, 5> buttons {
        &hZoomInButton, &hZoomOutButton, &vZoomInButton, &vZoomOutButton,
        &zoomResetButton };
    auto fits = true;
    for (const auto* button : buttons)
    {
        const auto room = buttonTextRoom(*button);
        const auto width = buttonTextWidth(*button);
        fits = fits && width <= room;
        if (report != nullptr)
            *report += " " + button->getButtonText() + ":" + juce::String(width)
                + "/" + juce::String(room);
    }
    return fits;
}

int OtoWaveformEditorComponent::regionCount() const
{
    return mou ? SampleSettings::mouRegionCount(edited.mouClasses) : 4;
}

void OtoWaveformEditorComponent::setRegionCount(int count)
{
    count = juce::jlimit(2, 4, count);
    if (!mou || count == regionCount()) return;
    edited.mouClasses = SampleSettings::mouClassesForCount(edited.mouClasses, count);
    // The boundaries this count does not reach keep their times.  They are no
    // longer drawn, grabbable or read back, so they cost nothing here and are
    // still there if the entry is split the other way again -- which is what
    // the voicebank panel's buttons do too.
    classesEditor.setText(edited.mouClasses, false);
    syncRegionCount();
    resized();
    refreshEditors();
}

bool OtoWaveformEditorComponent::regionIsConsonant(int index) const
{
    const auto classes = edited.mouClasses.trim().toUpperCase();
    return juce::isPositiveAndBelow(index, classes.length())
        && classes[index] == 'C';
}

void OtoWaveformEditorComponent::setRegionIsConsonant(int index, bool consonant)
{
    if (!mou || !juce::isPositiveAndBelow(index, regionCount())) return;
    auto classes = SampleSettings::mouClassesForCount(edited.mouClasses,
                                                      regionCount());
    if (consonant == (classes[index] == 'C')) return;
    // Untick a silence and it stays a silence: the boxes only ever say
    // consonant or not, and throwing an S away would be answering a question
    // they never asked.
    juce::String next;
    for (int position = 0; position < classes.length(); ++position)
        next += position != index ? classes[position]
              : consonant         ? juce::juce_wchar('C')
              : classes[position] == 'S' ? juce::juce_wchar('S')
                                         : juce::juce_wchar('V');
    edited.mouClasses = next;
    classesEditor.setText(edited.mouClasses, false);
    syncRegionCount();
    refreshEditors();
}

juce::String OtoWaveformEditorComponent::diagnosticRegionName(int index) const
{
    return regionName(mou, index);
}

void OtoWaveformEditorComponent::applyParameterVisibility()
{
    const auto shown = visibleParameters();
    for (std::size_t index = 0; index < parameterEditors.size(); ++index)
    {
        const auto visible = std::find(shown.begin(), shown.end(), index) != shown.end();
        parameterLabels[index].setVisible(visible);
        parameterEditors[index].setVisible(visible);
    }
}

void OtoWaveformEditorComponent::syncRegionCount()
{
    waveform.regions = regionCount();
    waveform.classes = edited.mouClasses.trim().toUpperCase();
    for (int index = 0; index < 3; ++index)
        countButtons[static_cast<std::size_t>(index)].setToggleState(
            index + 2 == regionCount(), juce::dontSendNotification);
    // A region the count does not reach cannot be marked anything.
    for (int index = 0; index < 4; ++index)
    {
        auto& tick = consonantButtons[static_cast<std::size_t>(index)];
        tick.setEnabled(index < regionCount());
        tick.setToggleState(index < regionCount() && regionIsConsonant(index),
                            juce::dontSendNotification);
    }
    applyParameterVisibility();
    waveform.repaint();
}

void OtoWaveformEditorComponent::refreshEditors()
{
    refreshingEditors = true;
    const std::array<double, 8> values { edited.offsetMs, edited.consonantMs,
        edited.cutoffMs, edited.preutteranceMs, edited.overlapMs,
        edited.jieOnsetMs, edited.jieGlideMs, edited.jieNucleusMs };
    for (std::size_t index = 0; index < parameterEditors.size(); ++index)
        parameterEditors[index].setText(formatNumber(values[index]), false);
    refreshingEditors = false;
    waveform.repaint();
}

void OtoWaveformEditorComponent::commitEditors()
{
    if (refreshingEditors || loadingMaterial) return;
    if (materialSession)
    {
        materialDetailError.clear();
        if (!materialDetail) return;
        auto candidate = edited;
        double* fields[]{&candidate.offsetMs, &candidate.consonantMs, &candidate.cutoffMs,
            &candidate.preutteranceMs, &candidate.overlapMs, &candidate.jieOnsetMs,
            &candidate.jieGlideMs, &candidate.jieNucleusMs};
        const double previous[]{edited.offsetMs, edited.consonantMs, edited.cutoffMs,
            edited.preutteranceMs, edited.overlapMs, edited.jieOnsetMs, edited.jieGlideMs, edited.jieNucleusMs};
        for (const auto index : visibleParameters())
        {
            if (waveform.lockOuterRange && (index == 0 || index == 2)) continue;
            const auto text = parameterEditors[index].getText().trim();
            if (text == formatNumber(previous[index])) continue; // Keep exact inherited timing bits.
            const auto* start = text.toRawUTF8(); char* end = nullptr;
            const auto number = std::strtod(start, &end);
            if (text.isEmpty() || end == start || *end != 0 || !std::isfinite(number))
            {
                materialDetailError = materialText("参数必须为有限的毫秒数。", "Parameters must be finite numeric milliseconds.");
                helpLabel.setText(materialDetailError, juce::dontSendNotification); return;
            }
            *fields[index] = number;
        }
        if (jie && candidate.jieOnsetMs != edited.jieOnsetMs) candidate.consonantMs = candidate.jieOnsetMs;
        edited = candidate;
        if (!syncMaterialTiming(materialDetailError)) helpLabel.setText(materialDetailError, juce::dontSendNotification);
        refreshEditors(); return;
    }
    edited.offsetMs = parameterEditors[0].getText().getDoubleValue();
    if (!jie) edited.consonantMs = parameterEditors[1].getText().getDoubleValue();
    edited.cutoffMs = parameterEditors[2].getText().getDoubleValue();
    edited.preutteranceMs = parameterEditors[3].getText().getDoubleValue();
    edited.overlapMs = parameterEditors[4].getText().getDoubleValue();
    if (jie)
    {
        // Same ordering rule the drag handles enforce, applied to typed
        // values -- and to the same boundaries.  A hidden editor still holds
        // the text it was last given, so reading all three would let a
        // boundary the entry does not have clamp the ones it does.
        const auto span = std::max(0.0, waveform.entrySpanMilliseconds());
        const auto count = regionCount();
        const std::array<double*, 3> bounds { &edited.jieOnsetMs,
                                              &edited.jieGlideMs,
                                              &edited.jieNucleusMs };
        for (int index = count - 2; index >= 0; --index)
            *bounds[static_cast<std::size_t>(index)] = juce::jlimit(0.0,
                index + 2 < count ? *bounds[static_cast<std::size_t>(index) + 1] : span,
                parameterEditors[static_cast<std::size_t>(5 + index)]
                    .getText().getDoubleValue());
        edited.jieGlideMs = std::max(edited.jieGlideMs, edited.jieOnsetMs);
        edited.jieNucleusMs = std::max(edited.jieNucleusMs, edited.jieGlideMs);
        // The classic row still carries a consonant: the two-region fallback
        // and the host's own timing read it.  It is simply no longer edited
        // separately -- the onset boundary is what it means here.
        edited.consonantMs = edited.jieOnsetMs;
    }
    refreshEditors();
}

bool OtoWaveformEditorComponent::commitMaterialDetail(juce::String& error)
{
    if (!materialSession || !materialDetail) return true;
    if (!commitMaterialVowelFields()) { error = "Invalid manual vowel name or stretch weight"; return false; }
    commitEditors();
    if (materialDetailError.isNotEmpty()) { error = materialDetailError; return false; }
    auto row = materialDetailDraft;
    row.segments = waveform.nativeSegments;
    // UTAU views expose the same C/P authority. Additional displayed boundaries
    // may retime existing elastic vowel pairs, never assign arbitrary C/V kinds.
    if (jie)
    {
        const double now[]{edited.jieGlideMs, edited.jieNucleusMs};
        const double before[]{detailBaseline.jieGlideMs, detailBaseline.jieNucleusMs};
        for (int i = 0; i < 2; ++i)
        {
            if (now[i] == before[i]) continue;
            const auto left = static_cast<std::size_t>(i + 1), right = left + 1;
            if (right >= row.segments.size() || row.segments[left].role != NativeSegmentRole::vowel
                || row.segments[right].role != NativeSegmentRole::vowel
                || !row.segments[left].stretchable || !row.segments[right].stretchable)
            { error = materialText("请在原生视图选择内部弹性元音区段。", "Choose an internal elastic vowel segment in native view."); return false; }
            const auto position = row.regionStartSeconds + now[i] / 1000;
            if (position <= row.segments[left].sourceStartSeconds || position >= row.segments[right].sourceEndSeconds)
            { error = "Internal vowel boundary must remain inside its adjacent vowels"; return false; }
            row.segments[left].sourceEndSeconds = position; row.segments[right].sourceStartSeconds = position;
        }
    }
    if (!materialSession->setRow(materialRegionId, row, error)) return false;
    for (const auto& updated : materialSession->rows()) if (updated.materialRegionId == materialRegionId)
    {
        materialDetailDraft = updated; detailSegmentsBaseline = updated.segments;
        waveform.nativeSegments = updated.segments; break;
    }
    detailBaseline = edited; materialTimingEntry = edited;
    return true;
}

void OtoWaveformEditorComponent::saveMaterial()
{
    if (!materialSession) return;
    if (!materialDetail) { if (materialOverview) materialOverview->save(); return; }
    juce::String error;
    if (!commitMaterialDetail(error)) { helpLabel.setText(error, juce::dontSendNotification); return; }
    const auto session = materialSession; // model callback may close this window
    juce::Component::SafePointer<OtoWaveformEditorComponent> safe(this);
    const bool saved = session->save(error);
    if (!safe) return;
    if (!saved) { helpLabel.setText(error, juce::dontSendNotification); return; }
    if (editScope() == EditScope::thisNote)
    {
        helpLabel.setText(materialText("已应用到此音符；源录音、OTTO 和旧 OTO 未修改。", "Applied to this note; recording, OTTO and old OTO unchanged."), juce::dontSendNotification);
        return; // Never call the SOURCE publisher/cache invalidator for local Save.
    }
    backend::UtauRenderer::invalidateVoicebankCache();
    helpLabel.setText(materialText("标注已保存到 otto.ini；原音频和旧 OTO 未修改。", "Saved otto.ini; original audio and old OTO unchanged."), juce::dontSendNotification);
    const auto rows = materialSession->rows();
    materialSourceSaved(rows); // Host may close this component; no later access.
}

void OtoWaveformEditorComponent::save()
{
    if (materialSession) { saveMaterial(); return; }
    commitEditors();
    if (noteHandoff)
    {
        // 单独OTO编辑: the entry goes to one note and nowhere else.  No oto file
        // is written, and the voicebank's cached entries are left alone,
        // because nothing about the voicebank has changed.  谋's classes are
        // in the entry already: the box writes each valid string it holds
        // straight into it.
        noteHandoff(edited);
        closeWindow();
        return;
    }
    juce::String error;
    // A from-scratch material is stored natively: the override writes the HJM
    // sidecar and the oto files are left untouched.
    if (saveOverride)
    {
        if (!saveOverride(edited, error))
        {
            juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon,
                utf8("无法保存原生标注"), error);
            return;
        }
        backend::UtauRenderer::invalidateVoicebankCache();
        if (onSaved) onSaved();
        closeWindow();
        return;
    }
    const auto savedClassicTiming = jie
        ? SampleSettings::updateJieVoicebankOtoEntry(original, edited, error)
        : SampleSettings::updateVoicebankOtoEntry(original, edited, error);
    if (!savedClassicTiming)
    {
        juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon,
            jie ? utf8("无法保存独立界•OTO") : utf8("无法保存 oto.ini"), error);
        return;
    }
    // The four-region row is keyed by the independent classic offset just
    // saved above.  It never writes the original oto.ini.
    // 谋 writes its own file; 界 writes oto4.ini.  A 谋 track never touches
    // oto4.ini, so annotating a voicebank cannot disturb what 界 renders.
    if (mou)
    {
        edited.mouClasses = classesEditor.getText().trim().toUpperCase();
        if (edited.mouClasses.length() < 2) edited.mouClasses = "CVVV";
        if (!SampleSettings::updateMouOtoEntry(original, edited, error))
        {
            juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon,
                utf8("无法保存谋•OTO（otomou.ini）"), error);
            return;
        }
    }
    else if (jie && !SampleSettings::updateJieOtoEntry(original, edited, error))
    {
        juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon,
            utf8("无法保存界•OTO（oto4.ini）"), error);
        return;
    }
    backend::UtauRenderer::invalidateVoicebankCache();
    if (onSaved) onSaved();
    closeWindow();
}

std::unique_ptr<OtoWaveformEditorComponent> OtoWaveformEditorComponent::forClassicNote(
    ProjectModel& project, const juce::String& noteId, I18n& strings,
    const backend::UtauRenderer::ResolvedSample& sample, std::uint64_t resolvedAtRevision, juce::String& error,
    EditScope initial, const juce::String& selectedAudioId, const juce::String& selectedRegionId,
    std::function<void(const std::vector<SampleRegionSetting>&)> onSourceSaved)
{
    error.clear();
    const auto reject = [&](const juce::String& why)
    { error = why; return std::unique_ptr<OtoWaveformEditorComponent>{}; };
    if (!sample.found || !sample.sourceOto || sample.sourceOto->lineIndex < 0
        || sample.sourceOto->sourceFingerprint.isEmpty() || sample.sourceOto->audioFile != sample.file)
        return reject("Classic note scope requires the exact canonical source OTO ticket; no alias fallback is available");
    if (project.revisionNumber() != resolvedAtRevision) return reject("The project changed while resolving classic note timing");
    const auto data = project.snapshot(); const auto expectedRevision = resolvedAtRevision;
    if (project.revisionNumber() != expectedRevision) return reject("The note changed while opening classic timing");
    const TrackData* owner = nullptr; const NoteData* note = nullptr; int matches = 0;
    for (const auto& track : data.tracks) for (const auto& clip : track.clips) for (const auto& candidate : clip.notes)
        if (candidate.id == noteId) { owner = &track; note = &candidate; ++matches; }
    if (matches != 1 || !owner || !note || !trackUsesVoicebankSynthesis(*owner)
        || trackIsDiffSinger(*owner) || owner->utauMode != UtauMode::classic)
        return reject("A unique classic UTAU note is required for scalar note OTO editing");
    const auto root = owner->voicebankDirectory, manifest = root.getChildFile("otto.ini");
    if (!sample.file.isAChildOf(root)) return reject("The canonical recording is outside the note's voicebank");
    OttoMaterialMigration::OtoSource otoSource;
    if (!OttoMaterialMigration::readOtoSource(*sample.sourceOto, UtauMode::classic, otoSource, error)) return {};
    SampleRegionSetting engineDefault;
    if (!OttoMaterialMigration::convertOtoEntry(*sample.sourceOto, UtauMode::classic,
            sample.fileSeconds, otoSource, engineDefault, error)) return {};
    const auto requestedAudio = selectedAudioId.isNotEmpty() ? selectedAudioId : note->materialAudioId;
    const auto requestedRegion = selectedRegionId.isNotEmpty() ? selectedRegionId
        : note->materialRegionId.isNotEmpty() ? note->materialRegionId : engineDefault.materialRegionId;
    auto editor = forMaterial(sample.file, manifest, strings, std::move(onSourceSaved), true, &error, requestedAudio);
    if (!editor) return {};
    const auto& sourceRows = editor->materialSession->rows();
    const auto source = std::find_if(sourceRows.begin(), sourceRows.end(),
        [&](const auto& row) { return row.materialRegionId == requestedRegion; });
    if (source == sourceRows.end()) return reject("No matching pronunciation identity in the governing source; select an explicit variant/region");
    auto entry = SampleSettings::entryWithNoteOto(*sample.sourceOto, note->utauOto);
    SampleRegionSetting localRow;
    if (!OttoMaterialMigration::convertOtoEntry(entry, UtauMode::classic, sample.fileSeconds, otoSource, localRow, error)) return {};
    // Session editing identities are from the governing source. They are never
    // written to NoteData by the scalar callback or treated as a native binding.
    const auto adoptSessionIdentity = [&](SampleRegionSetting& row)
    {
        row.materialAudioId = source->materialAudioId; row.materialRegionId = source->materialRegionId;
        row.name = source->name; row.additionalAliases = source->additionalAliases;
        row.connectedNextRegionId = source->connectedNextRegionId;
    };
    adoptSessionIdentity(localRow); adoptSessionIdentity(engineDefault);
    editor->classicScalarProjection = [ticket = sample.sourceOto, otoSource, duration = sample.fileSeconds]
        (const VoicebankOtoEntry& requested, SampleRegionSetting& row, juce::String& why)
    {
        auto candidate = *ticket;
        candidate.offsetMs = requested.offsetMs; candidate.cutoffMs = requested.cutoffMs;
        candidate.consonantMs = requested.consonantMs; candidate.preutteranceMs = requested.preutteranceMs;
        candidate.overlapMs = requested.overlapMs;
        SampleRegionSetting converted;
        if (!OttoMaterialMigration::convertOtoEntry(candidate, UtauMode::classic, duration, otoSource, converted, why)) return false;
        // Keep immutable Session/source identity and metadata, replacing only
        // scalar-authoritative geometry from the existing strict converter.
        row.regionStartSeconds = converted.regionStartSeconds; row.regionEndSeconds = converted.regionEndSeconds;
        row.alignmentSeconds = converted.alignmentSeconds; row.fixedDurationSeconds = converted.fixedDurationSeconds;
        row.overlapSeconds = converted.overlapSeconds; row.segments = std::move(converted.segments);
        return true;
    };
    backend::UtauOtoOverride check;
    if (!ProjectModel::makeClassicNoteMaterialOto(localRow, check, error)
        || !ProjectModel::makeClassicNoteMaterialOto(engineDefault, check, error)) return {};
    const auto sourceSize = sample.file.getSize(); const auto sourceTime = sample.file.getLastModificationTime();
    const auto audioFingerprint = juce::SHA256(sample.file).toHexString();
    if (!sample.file.existsAsFile() || sample.file.getSize() != sourceSize || sample.file.getLastModificationTime() != sourceTime)
        return reject("The recording changed while opening classic note timing");
    const auto manifestToken = editor->materialSession->revisionToken();
    const auto ticket = sample.sourceOto;
    const auto guardSource = [ticket, manifest, manifestToken, audioFingerprint](const NoteScopeBinding& binding, juce::String& why)
    {
        if (binding.sourceAudio != ticket->audioFile || binding.sourceFingerprint != audioFingerprint
            || !ticket->audioFile.existsAsFile() || juce::SHA256(ticket->audioFile).toHexString() != audioFingerprint)
        { why = "The classic source recording changed; reopen note timing"; return false; }
        OttoMaterialMigration::OtoSource current;
        if (!OttoMaterialMigration::readOtoSource(*ticket, UtauMode::classic, current, why)) return false;
        juce::MemoryBlock bytes;
        if (manifest.exists() && (!manifest.existsAsFile() || !manifest.loadFileAsData(bytes)))
        { why = "Cannot revalidate the governing native manifest"; return false; }
        const auto token = manifest.exists() ? "v1:" + juce::SHA256(bytes).toHexString() : juce::String("absent");
        if (token != manifestToken) { why = "The governing source annotations changed; reopen note timing"; return false; }
        return true;
    };
    NoteScopeBinding binding;
    binding.noteId = noteId; binding.audioId = source->materialAudioId; binding.regionId = source->materialRegionId;
    binding.sourceAudio = sample.file; binding.sourceFingerprint = audioFingerprint;
    binding.projectRevision = expectedRevision; binding.localRow = localRow;
    binding.capability = MaterialAnnotationSession::NoteScopeCapability::classicScalarOto;
    binding.apply = [&project, noteId, guardSource](const auto& bound, const auto& row, juce::uint64& revision, juce::String& why)
    {
        if (bound.noteId != noteId || !guardSource(bound, why)) return false;
        backend::UtauOtoOverride scalar;
        if (!ProjectModel::makeClassicNoteMaterialOto(row, scalar, why)) return false;
        std::uint64_t committed = 0;
        if (!project.setNoteUtauOtoChecked(noteId, bound.projectRevision, scalar, committed, why)) return false;
        revision = committed; return true;
    };
    binding.restore = [&project, noteId, guardSource, engineDefault](const auto& bound, auto& restored, juce::uint64& revision, juce::String& why)
    {
        if (bound.noteId != noteId || !guardSource(bound, why)) return false;
        backend::UtauOtoOverride scalar;
        if (!ProjectModel::makeClassicNoteMaterialOto(engineDefault, scalar, why)) return false;
        std::uint64_t committed = 0;
        if (!project.setNoteUtauOtoChecked(noteId, bound.projectRevision, backend::UtauOtoOverride{}, committed, why)) return false;
        restored = engineDefault; revision = committed; return true; // Explicit engine inheritance, not native restore.
    };
    if (!guardSource(binding, error) || !editor->enableNoteScope(std::move(binding), initial, error)) return {};
    editor->setPublicationValidator([](const auto&, juce::String& why)
    { why = "Source publication requires the host's source preflight/completion adapter"; return false; });
    return editor;
}

std::unique_ptr<OtoWaveformEditorComponent> OtoWaveformEditorComponent::forNote(
    ProjectModel& project, const juce::String& noteId, juce::String& error)
{
    error.clear();
    const auto expectedRevision = project.revisionNumber();
    const auto data = project.snapshot();
    if (project.revisionNumber() != expectedRevision)
    { error = "The project changed while opening note timing"; return {}; }
    const TrackData* owner = nullptr;
    const NoteData* target = nullptr;
    for (const auto& track : data.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (note.id == noteId) { owner = &track; target = &note; }
    if (owner == nullptr || target == nullptr) return {};
    if (owner->pitchAlgorithm != PitchAlgorithm::utau)
    {
        error = utf8("单独OTO编辑仅用于 UTAU 轨道。");
        return {};
    }
    if (!owner->voicebankDirectory.isDirectory())
    {
        error = utf8("请先为当前轨道选择 UTAU 音源库。");
        return {};
    }
    if (owner->utauMode == UtauMode::classic)
    {
        const auto sample = backend::UtauRenderer::resolveVoiceSample(owner->voicebankDirectory,
            target->label, target->midiNote, 100, false, false);
        auto strings = std::make_unique<I18n>();
        auto editor = forClassicNote(project, noteId, *strings, sample, expectedRevision, error);
        if (editor) editor->ownedMaterialStrings = std::move(strings);
        return editor;
    }
    // Regional legacy capability is intentionally retained: its auxiliary
    // Jie/Mou files are not covered by a classic provenance/scalar adapter.
    // This branch is NOT claimed to be the unified native/classic Session path.
    juce::StringArray warnings;
    const auto mouMode = owner->utauMode == UtauMode::mou;
    const auto regions = utauModeUsesRegions(owner->utauMode);
    const auto entries = SampleSettings::loadVoicebankOto(owner->voicebankDirectory,
                                                          warnings, regions, mouMode);
    const auto index = SampleSettings::findEntryForAlias(entries, target->label);
    if (index < 0)
    {
        error = utf8("音源库里没有这个歌词的 oto 条目。");
        return {};
    }
    auto editor = std::make_unique<OtoWaveformEditorComponent>(
        SampleSettings::entryWithNoteOto(entries[static_cast<std::size_t>(index)],
                                         target->utauOto),
        regions, mouMode, [] {});
    // The project outlives every window opened on it.
    editor->saveToNoteInstead([&project, noteId, regions](const VoicebankOtoEntry& entry)
    {
        project.setNoteUtauOto(noteId, SampleSettings::noteOtoFromEntry(entry, regions));
    });
    return editor;
}

void OtoWaveformEditorComponent::saveToNoteInstead(
    std::function<void(const VoicebankOtoEntry&)> handoff)
{
    noteHandoff = std::move(handoff);
    saveButton.setButtonText(utf8("应用到此音符"));
    saveButton.setTooltip(utf8("只作用于这一个音符，不修改 oto 文件"));
    helpLabel.setText(utf8("单独 OTO：只作用于这一个音符，不修改 oto 文件；数值单位均为 ms"),
                      juce::dontSendNotification);
}

OtoWaveformEditorComponent::~OtoWaveformEditorComponent()
{
    // Closing the window is how most playing ends: the device must not be left
    // calling a player that is about to go.
    stopPlayback();
    setLookAndFeel(nullptr);
}

void OtoWaveformEditorComponent::togglePlayback()
{
    if (preview != nullptr) stopPlayback();
    else startPlayback();
}

void OtoWaveformEditorComponent::startPlayback()
{
    stopPlayback();
    // The whole recording the entry is cut from, not only 偏移 to 终止: what is
    // either side of the entry is how you hear where its edges belong.
    auto rate = 0.0;
    auto recording = readRecording(edited.audioFile, rate);
    if (recording.getNumSamples() <= 0)
    {
        juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon,
            utf8("无法播放原音"),utf8("无法读取音频文件：")+edited.audioFile.getFullPathName());
        return;
    }
    if (playbackHost.beforeStart && !playbackHost.beforeStart()) return;
    juce::AudioDeviceManager* devices = nullptr;
    if (playbackHost.devices)
    {
        devices = playbackHost.devices();
        if (devices == nullptr) return;
    }
    preview = std::make_unique<SourcePreview>(std::move(recording), rate);
    if (devices != nullptr)
    {
        previewPlayer = std::make_unique<juce::AudioSourcePlayer>();
        previewPlayer->setSource(preview.get());
        devices->addAudioCallback(previewPlayer.get());
        previewDevices = devices;
    }
    playButton.setButtonText(utf8("停止播放"));
    waveform.setPlayheadMilliseconds(0.0);
    startTimerHz(30);
}

void OtoWaveformEditorComponent::stopPlayback()
{
    stopTimer();
    if (previewDevices != nullptr)
    {
        // Only while that device is still there to be told: the window can
        // outlive whoever handed it over, and the device along with them.
        if (playbackHost.devices && playbackHost.devices() == previewDevices)
            previewDevices->removeAudioCallback(previewPlayer.get());
        previewDevices = nullptr;
    }
    if (previewPlayer != nullptr) previewPlayer->setSource(nullptr);
    previewPlayer.reset();
    preview.reset();
    waveform.setPlayheadMilliseconds({});
    playButton.setButtonText(utf8("播放原音"));
}

void OtoWaveformEditorComponent::timerCallback()
{
    if (preview == nullptr)
    {
        stopTimer();
        return;
    }
    if (preview->finished())
    {
        stopPlayback();
        return;
    }
    waveform.setPlayheadMilliseconds(preview->seconds() * 1000.0);
}

int OtoWaveformEditorComponent::diagnosticPullPreview(juce::AudioBuffer<float>& out,
                                                      double outputRate, double maxSeconds)
{
    if (preview == nullptr || outputRate <= 0.0) return 0;
    constexpr int block = 512;
    preview->prepareToPlay(block, outputRate);
    const auto wanted = static_cast<int>(std::llround(maxSeconds * outputRate));
    juce::AudioBuffer<float> scratch(2, block);
    auto added = 0;
    while (preview != nullptr && added < wanted)
    {
        const auto count = std::min(block, wanted - added);
        scratch.clear();
        const auto produced = preview->render(scratch, 0, count);
        if (produced > 0)
        {
            const auto base = out.getNumSamples();
            out.setSize(2, base + produced, true, true, false);
            for (int channel = 0; channel < 2; ++channel)
                out.copyFrom(channel, base, scratch, channel, 0, produced);
        }
        added += produced;
        timerCallback();
        if (produced < count) break;
    }
    return added;
}

void OtoWaveformEditorComponent::closeWindow()
{
    if (auto* window = findParentComponentOfClass<juce::DialogWindow>())
        window->closeButtonPressed();
}

juce::String OtoWaveformEditorComponent::formatNumber(double value)
{
    if (std::abs(value) < 0.0005) value = 0.0;
    auto text = juce::String(value, 3);
    while (text.containsChar('.') && text.endsWithChar('0')) text = text.dropLastCharacters(1);
    if (text.endsWithChar('.')) text = text.dropLastCharacters(1);
    return text;
}
}
