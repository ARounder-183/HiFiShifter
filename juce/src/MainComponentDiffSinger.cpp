#include "MainComponent.h"
#include "DiffSingerRequest.h"
#include "backend/DiffSingerRenderer.h"
#include "DiffSingerPhonemeEditor.h"
#include "DiffSingerPronunciationEditor.h"

namespace hachi
{

void MainComponent::showDiffSingerPronunciationEditor(const juce::String& noteId)
{
    if (diffSingerBusy) return;
    const auto data = project.snapshot();
    const auto track = std::find_if(data.tracks.begin(), data.tracks.end(), [&](const auto& item)
    {
        if (noteId.isEmpty()) return item.id == selectedTrackId;
        for (const auto& clip : item.clips) for (const auto& note : clip.notes) if (note.id == noteId) return true;
        return false;
    });
    if (track == data.tracks.end() || !trackIsDiffSinger(*track)) return;
    juce::StringArray ids, shown;
    const auto request = diffSingerRequest(data, *track, ids);
    if (ids.isEmpty()) return;
    const auto selection = pianoRoll.selectedNoteIds();
    for (int i = 0; i < ids.size(); ++i)
        if (std::find(selection.begin(), selection.end(), ids[i]) != selection.end()) shown.add(juce::String(i));
    if (ids.contains(noteId) && !shown.contains(juce::String(ids.indexOf(noteId)))) shown = {juce::String(ids.indexOf(noteId))};
    if ((!noteId.isEmpty() && !ids.contains(noteId)) || (!selection.empty() && shown.isEmpty()))
    { showDiffSingerError(juce::String::fromUTF8("所选音符不在当前可用轨道内，请重新选择。")); return; }
    auto json = backend::DiffSingerRenderer::requestJson(request, "phonemize");
    if (!shown.isEmpty())
    {
        juce::Array<juce::var> selected;
        for (const auto& id : shown) selected.add(id);
        json.getDynamicObject()->setProperty("retake_ids", selected);
    }
    const auto revision = project.revisionNumber();
    const auto trackId = track->id;
    const juce::Component::SafePointer<MainComponent> safe(this);
    auto* editor = new DiffSingerPronunciationEditor(json["notes"], shown, track->diffSingerDictionary,
        [safe, revision, trackId, ids, json](const auto& readings, const auto& dictionary, bool commit, auto done)
        {
            if (safe == nullptr) return;
            if (safe->diffSingerBusy) { done(juce::JSON::parse(R"({"ok":false,"error":"已有 DS 任务进行中，请稍后重试"})")); return; }
            auto requestJson = juce::JSON::parse(juce::JSON::toString(json));
            requestJson.getDynamicObject()->setProperty("dictionary", dictionary);
            for (auto& item : *requestJson["notes"].getArray()) for (const auto& [id, text] : readings)
                if (item["id"].toString() == id) item.getDynamicObject()->setProperty("pronunciation", text);
            safe->diffSingerBusy = true;
            safe->diffSingerCancel = std::make_shared<std::atomic<bool>>(false);
            const auto cancel = safe->diffSingerCancel;
            juce::Thread::launch([safe, cancel, revision, trackId, ids, readings, dictionary, commit, done, requestJson]
            {
                const auto result = backend::DiffSingerRenderer::invoke(requestJson, [cancel] { return cancel->load(); });
                juce::MessageManager::callAsync([safe, cancel, revision, trackId, ids, readings, dictionary, commit, done, result]
                {
                    if (safe == nullptr) return;
                    safe->diffSingerBusy = false;
                    auto response = result;
                    if (cancel->load()) response = juce::JSON::parse(R"({"ok":false,"error":"发音检查已取消"})");
                    if ((bool) response["ok"] && commit)
                    {
                        std::vector<std::pair<juce::String, juce::String>> mapped;
                        for (const auto& [id, text] : readings)
                            if (id.getIntValue() >= 0 && id.getIntValue() < ids.size()) mapped.emplace_back(ids[id.getIntValue()], text);
                        if (!safe->project.applyDiffSingerPronunciation(revision, trackId, mapped, dictionary))
                            response = juce::JSON::parse(R"({"ok":false,"error":"工程已变化，请关闭窗口后重新打开发音编辑器"})");
                        else
                        {
                            safe->diffSingerStatusUntil = juce::Time::getMillisecondCounterHiRes()+5000;
                            safe->statusLabel.setText(juce::String::fromUTF8("DS 发音已应用；现有 pitch 保留，需要时可重生成选中音符。Ctrl+Z 撤销。"), juce::dontSendNotification);
                        }
                    }
                    done(response);
                });
            });
        });
    juce::DialogWindow::LaunchOptions options;
    options.dialogTitle = juce::String::fromUTF8("谋•UTAU — DS 发音转换与读音覆盖");
    options.dialogBackgroundColour = Palette::panel;
    options.escapeKeyTriggersCloseButton = true; options.useNativeTitleBar = true; options.resizable = true;
    options.content.setOwned(editor);
    if (auto* window = options.launchAsync()) window->setResizeLimits(800, 500, 1600, 1000);
    editor->validate(false);
}

void MainComponent::loadDiffSingerVoicebank(const juce::String& trackId, const juce::File& directory)
{
    if (diffSingerBusy)
    {
        statusLabel.setText(juce::String::fromUTF8("DiffSinger 任务进行中，请等待或在编辑菜单取消后再选择音源。"), juce::dontSendNotification);
        return;
    }
    const auto data = project.snapshot();
    const auto found = std::find_if(data.tracks.begin(), data.tracks.end(),
        [&](const auto& track) { return track.id == trackId; });
    if (found == data.tracks.end()) return;
    backend::UtauRenderRequest request;
    request.voicebankDirectory = directory;
    request.diffSingerLanguage = found->diffSingerLanguage;
    request.diffSingerSpeaker = found->diffSingerSpeaker;
    diffSingerBusy = true;
    diffSingerCancel = std::make_shared<std::atomic<bool>>(false);
    const auto cancel = diffSingerCancel;
    const juce::Component::SafePointer<MainComponent> safe(this);
    statusLabel.setText(juce::String::fromUTF8("正在预检 DiffSinger 音源与声码器… 可在编辑菜单取消"), juce::dontSendNotification);
    juce::Thread::launch([safe, cancel, trackId, directory,
                         json = backend::DiffSingerRenderer::requestJson(request, "inspect")]
    {
        const auto result = backend::DiffSingerRenderer::invoke(json, [cancel] { return cancel->load(); });
        juce::MessageManager::callAsync([safe, cancel, trackId, directory, result]
        {
            if (safe == nullptr) return;
            safe->diffSingerBusy = false;
            if (cancel->load())
            {
                safe->statusLabel.setText(juce::String::fromUTF8("已取消音源预检"), juce::dontSendNotification);
                return;
            }
            if (!(bool) result["ok"]) { safe->showError(result["error"].toString()); return; }
            const auto current = safe->project.snapshot();
            if (std::none_of(current.tracks.begin(), current.tracks.end(),
                [&](const auto& track) { return track.id == trackId; })) return;
            safe->diffSingerCapabilities[directory.getFullPathName()] = result["expressions"];
            safe->project.setTrackVoicebankDirectory(trackId, directory);
            if (safe->preferences != nullptr)
            {
                safe->preferences->setValue("algorithm.utauVoicebank", directory.getFullPathName());
                safe->preferences->saveIfNeeded();
            }
            juce::StringArray warnings;
            if (const auto* values = result["warnings"].getArray())
                for (const auto& value : *values) warnings.add(value.toString());
            safe->diffSingerStatusUntil = juce::Time::getMillisecondCounterHiRes() + 5000.0;
            safe->statusLabel.setText(juce::String::fromUTF8("DiffSinger 预检通过；")
                + (warnings.isEmpty() ? juce::String::fromUTF8("模型已就绪，可生成 pitch、试听或导出") : warnings.joinIntoString(" ")),
                juce::dontSendNotification);
            safe->refreshProjectControls();
            safe->refreshSelectedNoteParameter();
        });
    });
}

bool MainComponent::refreshDiffSingerFlagContext()
{
    const auto data = project.snapshot();
    const auto track = std::find_if(data.tracks.begin(), data.tracks.end(),
        [this](const auto& t) { return t.id == selectedTrackId; });
    const auto enabled = track != data.tracks.end() && trackIsDiffSinger(*track);
    flagCurveButton.setButtonText(juce::String::fromUTF8(enabled ? "FLAG" : "线性flag"));
    flagEnvelopeButton.setButtonText(juce::String::fromUTF8(enabled ? "FLAG曲线" : "flag 包络"));
    flagCurveButton.setClickingTogglesState(!enabled);
    flagCurveButton.setTooltip(juce::String::fromUTF8(enabled
        ? "打开预测 FLAG 实参；在下方参数选择旁切换「实参 / 偏移」。启用偏移开关位于偏移通道"
        : "为选中音符开启 WCSNDM 线性 FLAG（普通 UTAU / 界 / 谋模式）；每条参数曲线独立保存，需使用支持此扩展的 WCSNDM 引擎"));
    flagEnvelopeButton.setTooltip(juce::String::fromUTF8(enabled
        ? "点击打开预测实参，下方切换实参 / 偏移；右键可局部重算。实参：橙色为直接值，蓝色虚线为原始预测，绿色为叠加偏移后的值；实参无需开启 FLAG。偏移保留旧单位；PEXP 在生成 pitch 时生效"
        : "在底部通道绘制 FLAG 曲线；先为音符开启线性 FLAG"));
    const auto modeHelp = juce::String::fromUTF8("；右键 FLAG 按钮可切换逐音符控制点 / 跨音符连续绘制。连续模式作用于当前轨道已开启 FLAG 的音符，Esc 取消本笔");
    flagCurveButton.setTooltip(flagCurveButton.getTooltip() + modeHelp);
    flagEnvelopeButton.setTooltip(flagEnvelopeButton.getTooltip() + modeHelp);
    if (!enabled)
    {
        pianoRoll.setDiffSingerFlagContext(false, {});
        const auto hifi = track != data.tracks.end() && trackUsesVoicebankSynthesis(*track)
            && effectiveUtauOutputEngine(*track, audio.currentUtauOutputEngine()) == UtauOutputEngine::pcNsfHifigan;
        pianoRoll.setHifisamplerFlagContext(hifi);
        if (hifi) {
            flagCurveButton.setTooltip(juce::String::fromUTF8("启用 HiFisampler 参数曲线：g / Hb / Hv / Ht / HG / P / t / A；G、He 在 FLAG 文本中输入") + modeHelp);
            flagEnvelopeButton.setTooltip(juce::String::fromUTF8("在底部绘制 HiFisampler FLAG 曲线，曲线覆盖同名文本参数") + modeHelp);
        }
        diffSingerParameterLanePendingBank.clear();
        return false;
    }
    pianoRoll.setHifisamplerFlagContext(false);
    const auto key = track->voicebankDirectory.getFullPathName();
    const auto found = diffSingerCapabilities.find(key);
    pianoRoll.setDiffSingerFlagContext(true,
        found != diffSingerCapabilities.end() ? found->second : juce::var{});
    if (found != diffSingerCapabilities.end() || diffSingerCapabilityPending.count(key)) return true;
    diffSingerCapabilityPending.insert(key);
    backend::UtauRenderRequest request;
    request.voicebankDirectory = track->voicebankDirectory;
    request.diffSingerLanguage = track->diffSingerLanguage;
    const auto cancel = diffSingerCapabilityCancel;
    const juce::Component::SafePointer<MainComponent> safe(this);
    juce::Thread::launch([safe, cancel, key, json = backend::DiffSingerRenderer::requestJson(request, "inspect")]
    {
        const auto result = backend::DiffSingerRenderer::invoke(json, [cancel] { return cancel->load(); });
        juce::MessageManager::callAsync([safe, cancel, key, result]
        {
            if (safe == nullptr || cancel->load()) return;
            safe->diffSingerCapabilityPending.erase(key);
            // Cache failures too: repainting must not repeatedly spawn Python.
            auto capabilities = result["expressions"];
            if (!capabilities.isArray())
            {
                juce::Array<juce::var> fallback;
                for (const auto& kind : diffSingerFlagCurveKinds())
                {
                    auto* item = new juce::DynamicObject();
                    item->setProperty("key",kind.flag);
                    item->setProperty("label",juce::String::fromUTF8(kind.label));
                    item->setProperty("supported",juce::String(kind.flag)=="DS:DYN");
                    item->setProperty("reason",juce::String::fromUTF8("读取失败，请在音源库设置重试"));
                    fallback.add(juce::var(item));
                }
                capabilities=juce::var(fallback);
            }
            safe->diffSingerCapabilities[key] = capabilities;
            safe->refreshSelectedNoteParameter();
            if (safe->diffSingerParameterLanePendingBank == key)
            {
                safe->diffSingerParameterLanePendingBank.clear();
                if ((bool)result["ok"] && safe->envelopeLane == EnvelopeLane::flagCurve)
                    for (const auto& t : safe->project.snapshot().tracks)
                        if (t.id == safe->selectedTrackId && t.voicebankDirectory.getFullPathName() == key)
                        { safe->showDiffSingerParameterLane(); break; }
            }
            if (!(bool) result["ok"])
                safe->statusLabel.setText(juce::String::fromUTF8("DiffSinger 参数读取失败：")
                    + result["error"].toString(), juce::dontSendNotification);
        });
    });
    return true;
}

void MainComponent::showDiffSingerPhonemeEditor(const juce::String& trackId, const juce::String& noteId)
{
    if (diffSingerBusy) return;
    const auto data = project.snapshot();
    const auto wanted = trackId.isEmpty() ? selectedTrackId : trackId;
    const auto track = std::find_if(data.tracks.begin(), data.tracks.end(),
        [&](const auto& item) { return item.id == wanted; });
    if (track == data.tracks.end() || !trackIsDiffSinger(*track)) return;
    backend::UtauRenderRequest request;
    request.voicebankDirectory = track->voicebankDirectory;
    request.diffSingerLanguage = track->diffSingerLanguage;
    request.diffSingerSpeaker = track->diffSingerSpeaker;
    request.diffSingerDictionary = track->diffSingerDictionary;
    juce::StringArray ids;
    for (const auto& clip : track->clips) if (!clip.muted)
        for (const auto& note : clip.notes)
        {
            backend::UtauNoteRenderSpec spec;
            spec.alias = note.label;
            spec.startSeconds = clip.startSeconds + note.startSeconds;
            spec.durationSeconds = note.durationSeconds;
            spec.midiNote = note.midiNote;
            spec.diffSingerTiming = note.diffSingerTiming;
            spec.diffSingerPronunciation = note.diffSingerPronunciation;
            spec.diffSingerContext = clip.id;
            request.targetDurationSeconds = std::max(request.targetDurationSeconds,
                spec.startSeconds + spec.durationSeconds + .2);
            request.notes.push_back(std::move(spec)); ids.add(note.id);
        }
    if (ids.isEmpty()) return;
    const auto selection = pianoRoll.selectedNoteIds();
    juce::StringArray visibleIds, auditionIds;
    for (int i = 0; i < ids.size(); ++i)
        if (std::find(selection.begin(), selection.end(), ids[i]) != selection.end())
        { visibleIds.add(juce::String(i)); auditionIds.add(ids[i]); }
    // A context menu on an unselected note targets that note alone.
    if (ids.contains(noteId) && !auditionIds.contains(noteId))
    {
        visibleIds = {juce::String(ids.indexOf(noteId))}; auditionIds = {noteId};
    }
    if ((!noteId.isEmpty() && !ids.contains(noteId)) || (!selection.empty() && visibleIds.isEmpty()))
    { showDiffSingerError(juce::String::fromUTF8("所选音符不在当前可用轨道内，请重新选择。")); return; }
    if (auditionIds.isEmpty()) auditionIds = ids;
    const auto focusId = ids.contains(noteId) ? juce::String(ids.indexOf(noteId))
        : (visibleIds.isEmpty() ? juce::String{} : visibleIds[0]);
    auto json = backend::DiffSingerRenderer::requestJson(request, "timing");
    // Scope prediction as well as the dialog. Keep adjacent notes in the request
    // for timing context; unrelated overlapping phrases must not block editing.
    if (!visibleIds.isEmpty())
    {
        juce::Array<juce::var> selected;
        for (const auto& id : visibleIds) selected.add(id);
        json.getDynamicObject()->setProperty("retake_ids", selected);
    }
    const auto revision = project.revisionNumber();
    diffSingerBusy = true;
    diffSingerCancel = std::make_shared<std::atomic<bool>>(false);
    const auto cancel = diffSingerCancel;
    const juce::Component::SafePointer<MainComponent> safe(this);
    statusLabel.setText(juce::String::fromUTF8("谋•UTAU：正在预测 DS 音素时长… 可在编辑菜单取消"), juce::dontSendNotification);
    juce::Thread::launch([safe, cancel, revision, ids, visibleIds, auditionIds, focusId, json]
    {
        const auto result = backend::DiffSingerRenderer::invoke(json, [cancel] { return cancel->load(); });
        juce::MessageManager::callAsync([safe, cancel, revision, ids, visibleIds, auditionIds, focusId, json, result]
        {
            if (safe == nullptr) return;
            safe->diffSingerBusy = false;
            if (cancel->load()) return;
            if (!(bool) result["ok"]) { safe->showError(result["error"].toString()); return; }
            if (safe->project.revisionNumber() != revision)
            {
                safe->showError(juce::String::fromUTF8("预测期间工程已变化，请重新打开音素时长编辑器。"));
                return;
            }
            juce::DialogWindow::LaunchOptions options;
            options.dialogTitle = juce::String::fromUTF8("谋•UTAU — DiffSinger 音素时长");
            options.dialogBackgroundColour = Palette::panel;
            options.escapeKeyTriggersCloseButton = true;
            options.useNativeTitleBar = true; options.resizable = true;
            options.content.setOwned(new DiffSingerPhonemeEditor(result, json["notes"],
                [safe, revision, ids](const DiffSingerPhonemeEditor::Timings& edits)
                {
                    if (safe == nullptr) return false;
                    DiffSingerPhonemeEditor::Timings mapped;
                    for (const auto& [index, text] : edits)
                    {
                        const auto i = index.getIntValue();
                        if (i < 0 || i >= ids.size()) return false;
                        mapped.emplace_back(ids[i], text);
                    }
                    if (!safe->project.applyDiffSingerTiming(revision, mapped)) return false;
                    safe->diffSingerStatusUntil = juce::Time::getMillisecondCounterHiRes() + 5000;
                    safe->statusLabel.setText(juce::String::fromUTF8("DS 音素时长已应用；试听和导出使用新时长，Ctrl+Z 可撤销。"), juce::dontSendNotification);
                    return true;
                }, [safe, auditionIds]
                {
                    if (safe == nullptr) return;
                    std::vector<juce::String> selected;
                    for (const auto& id : auditionIds) selected.push_back(id);
                    safe->pianoRoll.setSelectedNoteIds(selected);
                    safe->audio.stop();
                    safe->syncAudio(safe->project.snapshot());
                    safe->togglePlayback();
                }, focusId, visibleIds));
            if (auto* window = options.launchAsync()) window->setResizeLimits(820, 350, 2000, 700);
        });
    });
}

void MainComponent::rememberDiffSingerParameters()
{
    // Derived playback metadata must not invalidate the revision captured by
    // an open transactional editor. Import it after that dialog has closed.
    if (diffSingerBusy || juce::Component::getCurrentlyModalComponent()!=nullptr) return;
    audio.refreshUtauWaveforms();
    const auto waves=audio.utauNoteWaveforms();
    if (!waves || waves==rememberedParameterWaveforms) return;
    rememberedParameterWaveforms=waves;
    std::map<juce::String,std::vector<FlagCurve>> predictions;
    const auto data=project.snapshot();
    for (const auto& track:data.tracks) if (trackIsDiffSinger(track))
    for (const auto& clip:track.clips) for (const auto& note:clip.notes)
        for (const auto& wave:*waves) if (wave.noteId==note.id && !wave.diffSingerParameters.empty()
            && wave.audioHash==AudioEngine::utauNoteAudioHash(note))
            predictions[note.id]=wave.diffSingerParameters;
    project.rememberDiffSingerParameters(predictions);
}

void MainComponent::showDiffSingerParameterLane()
{
    for (const auto& track : project.snapshot().tracks) if (track.id == selectedTrackId && trackIsDiffSinger(track))
    {
        const auto key = track.voicebankDirectory.getFullPathName();
        if (diffSingerCapabilities.find(key) == diffSingerCapabilities.end() && diffSingerCapabilityPending.count(key))
        {
            diffSingerParameterLanePendingBank = key;
            setEnvelopeLane(EnvelopeLane::flagCurve);
            statusLabel.setText(juce::String::fromUTF8("正在读取音源参数能力，完成后打开实参…"),juce::dontSendNotification);
            return;
        }
        if (!pianoRoll.setDiffSingerParameterLayer(true))
        {
            setEnvelopeLane(EnvelopeLane::flagCurve);
            statusLabel.setText(juce::String::fromUTF8("当前音源没有可用的预测实参；可在偏移通道使用它支持的其他控制。"),juce::dontSendNotification);
            return;
        }
        setEnvelopeLane(EnvelopeLane::flagCurve);
        bool exists = false;
        const auto selected=pianoRoll.selectedNoteIds();
        for (const auto& clip : track.clips) if(!clip.muted) for (const auto& note : clip.notes)
            if(selected.empty() || std::find(selected.begin(),selected.end(),note.id)!=selected.end())
                exists = exists || !flagCurvePointsFor(note, pianoRoll.flagLaneFlag()).empty();
        if (!exists) generateDiffSingerParameters();
        return;
    }
}

void MainComponent::generateDiffSingerParameters(bool wholeTrack, const juce::String& noteId)
{
    if(diffSingerBusy) return;
    const auto data=project.snapshot();
    const auto track=std::find_if(data.tracks.begin(),data.tracks.end(),[&](const auto& t){
        if(noteId.isEmpty())return t.id==selectedTrackId;
        for(const auto& clip:t.clips)for(const auto& note:clip.notes)if(note.id==noteId)return true;
        return false;
    });
    if(track==data.tracks.end()||!trackIsDiffSinger(*track)) return;
    juce::StringArray ids;
    auto request=backend::DiffSingerRenderer::requestJson(diffSingerRequest(data,*track,ids,true),"parameters");
    if(ids.isEmpty()) return;
    auto selected=wholeTrack ? std::vector<juce::String>{} : pianoRoll.selectedNoteIds();
    std::erase_if(selected,[&](const auto& id){return !ids.contains(id);});
    if(!noteId.isEmpty()) {
        // A note-menu action must never fall back to regenerating the whole track.
        if(!ids.contains(noteId)) {showDiffSingerError(juce::String::fromUTF8("此音符不在当前可用片段内，请重新选择音符。"));return;}
        if(std::find(selected.begin(),selected.end(),noteId)==selected.end())selected={noteId};
    }
    else if(!wholeTrack && !pianoRoll.selectedNoteIds().empty() && selected.empty()) {
        showDiffSingerError(juce::String::fromUTF8("所选音符不在当前可用轨道内。"));return;
    }
    if(!selected.empty()) {
        juce::Array<juce::var> chosen;
        for(const auto& id:selected) if(ids.contains(id)) chosen.add(juce::String(ids.indexOf(id)));
        if(chosen.isEmpty()) {showDiffSingerError(juce::String::fromUTF8("所选音符不在当前可用轨道内。"));return;}
        request.getDynamicObject()->setProperty("retake_ids",chosen);
    }
    const auto revision=project.revisionNumber();
    const auto trackId=track->id;
    diffSingerBusy=true;diffSingerCancel=std::make_shared<std::atomic<bool>>(false);
    const auto cancel=diffSingerCancel;
    const juce::Component::SafePointer<MainComponent> safe(this);
    statusLabel.setText(juce::String::fromUTF8(selected.empty()
        ? "DS 正在重新生成整轨实参… 可在编辑菜单取消"
        : "DS 正在局部重算实参，锁定未选区域… 可在编辑菜单取消"),juce::dontSendNotification);
    juce::Thread::launch([safe,cancel,revision,ids,request,trackId,selected] {
        const auto result=backend::DiffSingerRenderer::invoke(request,[cancel]{return cancel->load();});
        juce::MessageManager::callAsync([safe,cancel,revision,ids,result,trackId,selected] {
            if(safe==nullptr) return;
            safe->diffSingerBusy=false;
            if(cancel->load()) {safe->finishMcpDiffSinger(false, "Cancelled");safe->statusLabel.setText(juce::String::fromUTF8("已取消实参预测"),juce::dontSendNotification);return;}
            if(!(bool)result["ok"]) {safe->showDiffSingerError(result["error"].toString());return;}
            std::map<juce::String,std::vector<FlagCurve>> edits;
            if(auto* rows=result["parameters"].getArray()) for(const auto& row:*rows) {
                const int index=row["id"].toString().getIntValue();
                if(index<0||index>=ids.size() || (!selected.empty() && std::find(selected.begin(),selected.end(),ids[index])==selected.end())) continue;
                for(const auto& kind:diffSingerParameterKinds()) {
                    const auto code=juce::String(kind.flag).substring(7);
                    if(auto* values=row["curves"][juce::Identifier(code)].getArray()) {
                        FlagCurve curve;curve.flag="DS:REF:"+code;
                        for(const auto& p:*values) if(p.isArray()&&p.size()==2) curve.points.push_back({(double)p[0],(float)p[1]});
                        if(!curve.points.empty()) edits[ids[index]].push_back(std::move(curve));
                    }
                }
            }
            if(!safe->project.applyDiffSingerParameters(revision,edits)) {
                safe->showDiffSingerError(juce::String::fromUTF8("工程在预测期间已修改，或没有有效实参。结果未写入，请重试。"));return;
            }
            safe->finishMcpDiffSinger(true);
            if(safe->selectedTrackId==trackId) {
                safe->pianoRoll.setDiffSingerParameterLayer(true);
                safe->setEnvelopeLane(EnvelopeLane::flagCurve);
            }
            safe->statusLabel.setText(juce::String::fromUTF8(selected.empty()
                ? "整轨实参已重生成，FLAG 偏移保留；Ctrl+Z 撤销。"
                : "局部实参已重算，未选音符和 FLAG 偏移保留；Ctrl+Z 撤销。"),juce::dontSendNotification);
        });
    });
}

void MainComponent::generateDiffSingerPitch(bool wholeTrack, const juce::String& noteId)
{
    if (diffSingerBusy) return;
    auto data = project.snapshot();
    // Retake context must not bake the separate output offset into a new prediction.
    for (auto& t : data.tracks) for (auto& c : t.clips) for (auto& n : c.notes) n.diffSingerPitchOffset.clear();
    const auto track = std::find_if(data.tracks.begin(), data.tracks.end(),
        [this, &noteId](const auto& t)
        {
            if (noteId.isEmpty()) return t.id == selectedTrackId;
            for (const auto& clip : t.clips) for (const auto& n : clip.notes) if (n.id == noteId) return true;
            return false;
        });
    if (track == data.tracks.end() || !trackIsDiffSinger(*track))
    {
        showDiffSingerError(juce::String::fromUTF8("请先在谋•UTAU 模式选择含 dsconfig.yaml 的 DiffSinger 音源目录。"));
        return;
    }
    juce::StringArray noteIds;
    const auto request = diffSingerRequest(data, *track, noteIds, true);
    if (noteIds.isEmpty()) return;
    auto selectedNotes = wholeTrack ? std::vector<juce::String>{} : pianoRoll.selectedNoteIds();
    std::erase_if(selectedNotes, [&](const auto& id) { return !noteIds.contains(id); });
    if (!noteId.isEmpty() && !noteIds.contains(noteId))
    { showDiffSingerError(juce::String::fromUTF8("此音符不在当前可用片段内，请重新选择音符。")); return; }
    if (noteIds.contains(noteId) && std::find(selectedNotes.begin(), selectedNotes.end(), noteId) == selectedNotes.end())
        selectedNotes = {noteId};
    if (!wholeTrack && noteId.isEmpty() && !pianoRoll.selectedNoteIds().empty() && selectedNotes.empty())
    { showDiffSingerError(juce::String::fromUTF8("当前所选音符不在本轨的可用片段内，请重新选择音符。")); return; }
    auto json = backend::DiffSingerRenderer::requestJson(request, "pitch");
    if (!selectedNotes.empty())
    {
        juce::Array<juce::var> retake;
        for (const auto& id : selectedNotes) retake.add(juce::String(noteIds.indexOf(id)));
        json.getDynamicObject()->setProperty("retake_ids", retake);
    }
    const auto revision = project.revisionNumber();
    diffSingerBusy = true;
    diffSingerCancel = std::make_shared<std::atomic<bool>>(false);
    const auto cancel = diffSingerCancel;
    const juce::Component::SafePointer<MainComponent> safe(this);
    statusLabel.setText(juce::String::fromUTF8(selectedNotes.empty()
        ? "DiffSinger 正在生成整轨 pitch… 可在编辑菜单取消"
        : "DS 正在局部重生成 pitch；其余音高作为锁定上下文… 可在编辑菜单取消"), juce::dontSendNotification);
    juce::Thread::launch([safe, cancel, revision, noteIds, selectedNotes, json]
    {
        auto result = backend::DiffSingerRenderer::invoke(json, [cancel] { return cancel->load(); });
        juce::MessageManager::callAsync([safe, cancel, revision, noteIds, selectedNotes, json, result]
        {
            if (safe == nullptr) return;
            safe->diffSingerBusy = false;
            if (cancel->load())
            {
                safe->finishMcpDiffSinger(false, "Cancelled");
                safe->statusLabel.setText(juce::String::fromUTF8("已取消 DiffSinger pitch 生成"), juce::dontSendNotification);
                return;
            }
            if (!(bool) result["ok"]) { safe->showDiffSingerError(result["error"].toString()); return; }
            std::vector<std::pair<juce::String, std::vector<PitchCurveEditPoint>>> curves;
            if (const auto* array = result["curves"].getArray())
                for (const auto& item : *array)
                {
                    const auto index = item["id"].toString().getIntValue();
                    if (index < 0 || index >= noteIds.size()) continue;
                    const auto id = noteIds[index];
                    if (!selectedNotes.empty() && std::find(selectedNotes.begin(), selectedNotes.end(), id) == selectedNotes.end()) continue;
                    std::vector<PitchCurveEditPoint> points;
                    if (const auto* values = item["points"].getArray())
                        for (const auto& p : *values)
                            if (p.isArray() && p.size() == 2) points.push_back({ (double) p[0], (float) p[1] });
                    if (!selectedNotes.empty() && !points.empty())
                    {
                        // Lock seam values to the pitch supplied to the model.
                        // Preserve selected notes' handles outside the retake span.
                        const auto current = safe->project.snapshot();
                        const auto incoming = json["notes"][index];
                        std::vector<PitchCurveEditPoint> existing;
                        if (auto* values = incoming["pitch"].getArray()) for (const auto& p : *values)
                            existing.push_back({(double)p[0], (float)p[1]});
                        const auto end = (double)incoming["duration"];
                        if (!existing.empty())
                        {
                            // A short taper entirely INSIDE the selection avoids
                            // a discontinuity without moving any locked frames.
                            for (auto& p : points)
                            {
                                const auto blend = juce::jlimit(0.0, 1.0, std::min(p.timeSeconds, end-p.timeSeconds)/.02);
                                p.targetMidi = (float)(evaluatePitchCurve(existing, p.timeSeconds)*(1-blend)+p.targetMidi*blend);
                            }
                        }
                        for (const auto& t : current.tracks) for (const auto& c : t.clips) for (const auto& n : c.notes)
                            if (n.id == id) for (const auto& p : n.pitchControlPoints)
                                if (p.timeSeconds < 0 || p.timeSeconds > end) points.push_back(p);
                        std::stable_sort(points.begin(), points.end(), [](const auto& a, const auto& b) { return a.timeSeconds < b.timeSeconds; });
                    }
                    if (!points.empty()) curves.emplace_back(id, std::move(points));
                }
            if (!safe->project.applyDiffSingerPitch(revision, curves, !selectedNotes.empty()))
            {
                safe->showDiffSingerError(juce::String::fromUTF8("生成期间工程已修改，或选中音符没有可生成的歌词。此次结果未写入，请重新生成。"));
                return;
            }
            safe->finishMcpDiffSinger(true);
            safe->statusLabel.setText(juce::String::fromUTF8("DiffSinger pitch 已写入 ")
                + juce::String((int) curves.size()) + juce::String::fromUTF8(" 个音符；可继续编辑，Ctrl+Z 撤销本次生成"), juce::dontSendNotification);
        });
    });
}

void MainComponent::showDiffSingerSettings(const TrackData& track)
{
    if (diffSingerBusy) return;
    diffSingerBusy = true;
    diffSingerCancel = std::make_shared<std::atomic<bool>>(false);
    const auto cancel = diffSingerCancel;
    backend::UtauRenderRequest request;
    request.voicebankDirectory = track.voicebankDirectory;
    request.diffSingerLanguage = track.diffSingerLanguage;
    const juce::Component::SafePointer<MainComponent> safe(this);
    juce::Thread::launch([safe, cancel, track, json = backend::DiffSingerRenderer::requestJson(request, "inspect")]
    {
        const auto result = backend::DiffSingerRenderer::invoke(json, [cancel] { return cancel->load(); });
        juce::MessageManager::callAsync([safe, cancel, track, result]
        {
            if (safe == nullptr) return;
            safe->diffSingerBusy = false;
            if (cancel->load()) return;
            if (!(bool) result["ok"]) { safe->showError(result["error"].toString()); return; }
            safe->diffSingerCapabilities[track.voicebankDirectory.getFullPathName()] = result["expressions"];
            safe->refreshSelectedNoteParameter();
            juce::StringArray languages, speakers;
            if (auto* values = result["languages"].getArray()) for (const auto& v : *values) languages.add(v.toString());
            languages.addIfNotAlreadyThere(track.diffSingerLanguage);
            speakers.add(juce::String::fromUTF8("默认"));
            if (auto* values = result["speakers"].getArray()) for (const auto& v : *values) speakers.add(v.toString());
            auto* window = new juce::AlertWindow("DiffSinger — " + track.voicebankDirectory.getFileName(),
                (bool) result["has_pitch"]
                    ? juce::String::fromUTF8("支持自动 pitch。编辑 → DiffSinger 生成 pitch；未选音符时生成整轨；选中音符时锁定未选音高作为上下文，局部重生成。")
                    : juce::String::fromUTF8("音源没有 pitch 预测模型，可按已有 MIDI / pitch 曲线合成。"), juce::MessageBoxIconType::InfoIcon);
            window->addComboBox("language", languages, juce::String::fromUTF8("歌词词典（zh 为中文拼音；汉字自动转拼音）"));
            window->getComboBoxComponent("language")->setSelectedItemIndex(languages.indexOf(track.diffSingerLanguage));
            window->addComboBox("speaker", speakers, juce::String::fromUTF8("音色 / 说话人"));
            window->getComboBoxComponent("speaker")->setSelectedItemIndex(std::max(0, speakers.indexOf(track.diffSingerSpeaker)));
            window->addTextBlock(juce::String::fromUTF8("谋•UTAU：右键音符 → DS 音素时长（Ctrl+G），可拖动音素边界；编辑菜单可生成 pitch。编辑菜单或音符右键可打开 DS 发音转换，预览中英文读音、设置覆盖和用户词典；多音节分配用 +2、+3，延音用 +，休止用 RR。GPU 与质量档位在设置 → DiffSinger 中调整。点击 FLAG 查看预测实参，在下方切换实参 / 偏移；偏移侧的启用按钮控制所选音符的偏移，可叠加多条参数曲线；灰色项表示音源不支持。PEXP 修改后需重新生成 pitch。传统 UTAU FLAG、OTO 和分区参数不适用于此音源。"));
            window->addTextBlock(juce::String::fromUTF8("音源预检通过。反复试听会复用模型和未修改乐句；闲置两分钟后释放模型。"));
            if (const auto* values = result["warnings"].getArray())
                for (const auto& value : *values) window->addTextBlock(value.toString());
            window->addButton(juce::String::fromUTF8("GPU / 质量设置"), 2);
            window->addButton(juce::String::fromUTF8("确定"), 1, juce::KeyPress(juce::KeyPress::returnKey));
            window->addButton(juce::String::fromUTF8("取消"), 0, juce::KeyPress(juce::KeyPress::escapeKey));
            window->enterModalState(true, juce::ModalCallbackFunction::create([safe, window, track, speakers](int code)
            {
                if (safe == nullptr) return;
                if (code == 2) { safe->showSettings(5); return; }
                if (code != 1) return;
                const auto language = window->getComboBoxComponent("language")->getText();
                const auto index = window->getComboBoxComponent("speaker")->getSelectedItemIndex();
                safe->project.setDiffSingerOptions(track.id, language, index > 0 ? speakers[index] : juce::String{});
            }), true);
        });
    });
}
}
