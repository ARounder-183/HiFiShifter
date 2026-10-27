#include "MainComponent.h"
#include "DiffSingerRequest.h"
#include "Hamood.h"
#include "HamoodProject.h"
#include "HamoodAudio.h"
#include "backend/McpServer.h"
#include "backend/LiveMcpBridge.h"
#include "backend/DiffSingerRenderer.h"
#include <cmath>
#include <iostream>

namespace hachi
{
namespace
{
using Server = backend::McpServer;
juce::var obj() { return new juce::DynamicObject(); }
void put(juce::var& v, const char* key, juce::var value) { v.getDynamicObject()->setProperty(key, std::move(value)); }
juce::var ok(juce::var v) { return Server::result(juce::JSON::toString(v, true)); }
juce::var fail(const juce::String& text) { return Server::result(text, true); }
juce::String str(const juce::var& v, const char* key) { return v[key].toString(); }
std::vector<juce::String> ids(const juce::var& v)
{
    std::vector<juce::String> result;
    if (auto* a = v.getArray()) for (const auto& n : *a) result.push_back(n.toString());
    return result;
}
bool in(const juce::String& name, const char* list)
{ return juce::StringArray::fromTokens(list, " ", "").contains(name); }
bool readOnly(const juce::String& name)
{
    return in(name, "hamood_get_context hamood_alignment_preview hamood_analyse_audio hamood_audio_context hamood_preview editor_selection editor_status editor_query editor_job_status project_snapshot render_status transport_status analysis_status analyse_audio read_file list_directory sample_settings_read ds_capabilities ds_query_phonemes");
}
juce::int64 renderEditFingerprint(const ProjectData& data)
{
    auto copy=data;
    for(auto& t:copy.tracks)for(auto& c:t.clips)for(auto& n:c.notes)
        std::erase_if(n.utauFlagCurves,[](const auto& curve){return curve.flag.startsWith("DS:AUTO:");});
    ProjectModel model;model.replace(std::move(copy));return model.contentFingerprint();
}
juce::var property(const char* type, const char* description)
{
    auto p = obj(); put(p, "type", type); put(p, "description", description); return p;
}
juce::var toolsForWindow()
{
    juce::var tools = juce::Array<juce::var>();
    auto add = [&](const char* name, const char* desc, std::initializer_list<std::pair<const char*,juce::var>> fields) {
        auto t = obj(), schema = obj(), props = obj();
        put(t,"name",name); put(t,"description",juce::String::fromUTF8(desc));
        put(schema,"type","object"); for (const auto& [key,value] : fields) put(props,key,value);
        put(schema,"properties",props); put(schema,"additionalProperties",false); put(t,"inputSchema",schema);
        tools.getArray()->add(t);
    };
    auto notes = property("array","Exact note IDs; editor_select accepts [] to clear selection");
    put(notes,"items",property("string","Stable note ID"));
    add("editor_selection","Read ONLY notes currently selected in the editor / 读取当前选区。Use FIRST for 'these notes', 'this phrase', or 'what did I select'. Returns lyrics, note names, timing and tracks; empty selection returns no notes, NEVER the whole project.",{
        {"include_curves",property("boolean","Include selected notes' pitch/reference/FLAG curves; default false")},
        {"offset",property("integer","Pagination start, default 0")},{"limit",property("integer","Notes per page, 1..256, default 64")}});
    add("editor_status","Read the connected window, current file, revision, selection, playback and DS state",{});
    add("editor_query","Read selected notes by default; range uses absolute project seconds; curves are opt-in",{
        {"scope",property("string","selection (default) or project")}, {"track_id",property("string","Optional track filter")},
        {"from_seconds",property("number","Range start on project timeline")},{"to_seconds",property("number","Range end on project timeline")},
        {"include_curves",property("boolean","Include dense pitch, references and FLAG curves, default false")},
        {"offset",property("integer","Pagination start, default 0")},{"limit",property("integer","Notes per page, 1..256, default 64")}});
    add("editor_select","Change real GUI selection and focus; [] clears it",{{"note_ids",notes},{"track_id",property("string","Focus this track")}});
    add("editor_batch","Apply up to 128 project edit commands atomically with one Undo; imports, file and transport commands are excluded",{
        {"commands",property("array","[{name: tool_name, arguments: {...}}, ...]")}});
    add("editor_job_status","Read a background job's state/result; completed does not imply success, check result.isError",{{"job_id",property("string","ID returned by an asynchronous command")}});
    add("editor_cancel_job","Cancel a pending job; staged edits are discarded, external file writes may already have completed",{{"job_id",property("string","Job ID")}});
    add("ds_generate_pitch","Regenerate DS pitch using GUI logic; selection required unless whole_track=true",{{"whole_track",property("boolean","Explicitly regenerate entire focused track, default false")}});
    add("ds_generate_parameters","Regenerate DS prediction parameters using GUI logic; preserve offsets and unselected notes",{{"whole_track",property("boolean","Explicitly regenerate entire focused track, default false")}});
    add("ds_query_phonemes","Predict phoneme timing in full phrase context, returning only selected notes by default; includes stable note IDs and timing context",{{"whole_track",property("boolean","Return entire focused track instead of selection")}});
    add("ds_capabilities","Inspect the focused DS voicebank and its parameter support asynchronously",{});
    add("ds_set_pitch_offset","Set an independent additive semitone curve; preserves the original prediction reference",{
        {"note_id",property("string","DS note ID")},{"points",property("array","[[note_local_seconds, offset_semitones], ...]; [] resets to zero")}});
    add("ds_set_parameter","Set a DS actual parameter or offset curve; supports only parameters exposed by this voicebank",{
        {"note_id",property("string","DS note ID")},{"flag",property("string","Parameter code, e.g. BREC, ENE, TENC, VOIC")},
        {"layer",property("string","actual or offset")},{"points",property("array","[[note_local_seconds, value], ...]; [] clears curve")}});
    add("ds_set_pronunciation","Set per-note pronunciation override; empty text restores automatic conversion",{
        {"note_id",property("string","DS note ID")},{"text",property("string","Reading override, as in the GUI pronunciation editor")}});
    add("ds_set_timing","Set phoneme onset overrides from an inspected timing context; null restores prediction",{
        {"note_id",property("string","DS note ID")},{"timing",property("object","{context,tokens,starts}; starts in note-local seconds, null entries retain prediction")}});
    add("hamood_analyse_audio","Analyse one accompaniment clip with Beat This! small0 + BTC; background job, cached by source audio/model hash. Read-only project operation. No pitch or tempo edits.",{
        {"clip_id",property("string","Source accompaniment clip ID")},{"force",property("boolean","Recompute instead of using the cache, default false")}});
    add("hamood_alignment_preview","Preview a small beat-phase offset against the existing project tempo grid. Does NOT resolve whole-bar ambiguity or change audio/tempo. Inspect start/middle/end windows before applying move_clip.",{
        {"clip_id",property("string","Analysed accompaniment clip ID")},
        {"from_seconds",property("number","Project range start; default clip start")},{"to_seconds",property("number","Project range end; default clip end")},
        {"grid_divisions_per_quarter",property("integer","1..4; default 2 for an eighth-note grid")}});
    add("hamood_audio_context","Read cached timestamped chords, low-register chroma candidates, beats and downbeats on the CURRENT project timeline. Scores are not calibrated probabilities. Defaults to selected-note time range; explicit range or whole_clip required otherwise.",{
        {"clip_id",property("string","Analysed accompaniment clip ID")},
        {"from_seconds",property("number","Project time range start")},{"to_seconds",property("number","Project time range end")},
        {"whole_clip",property("boolean","Read the complete clip explicitly")},
        {"offset",property("integer","Chord pagination offset, default 0")},{"limit",property("integer","Chords per page, 1..256, default 64")}});
    add("hamood_get_context","Read project-persisted HAMOOD settings, manual key ranges, chord timeline, section labels and confirmation states. Timeline positions use quarter notes.",{});
    add("hamood_set_context","Replace HAMOOD project metadata with one undo step; read hamood_get_context first and preserve unrelated entries. schema=1; settings object; chords, sections, key_predictions arrays. Chords require start_quarter/end_quarter, label, pitch_classes 0..11, score 0..1 and confirmed. Confirmed chords survive audio reanalysis.",{{"context",property("object","Complete HAMOOD context from hamood_get_context, with your edits")}});
    {auto schema=tools.getArray()->getLast()["inputSchema"];put(schema,"required",juce::Array<juce::var>{"context"});}
    for(const auto* name:{"hamood_preview","hamood_generate"})
    {
        auto voiceSteps=property("array","Signed diatonic steps: +/-2 third, +/-3 fourth, +/-4 fifth, +/-5 sixth, +/-7 octave. Default [2]");
        put(voiceSteps,"items",property("integer","Diatonic steps"));
        auto manualSections=property("array","Manual key ranges, inclusive ruler bars; [{start_bar:1,end_bar:8,tonic:0,minor:false},...]. Must cover all target note onsets without overlaps; manual mode only.");
        auto segmentSchema=property("object","One manual key segment"),segmentProps=obj();
        put(segmentProps,"start_bar",property("integer","First ruler bar, 1-based; 0 for pickup"));
        put(segmentProps,"end_bar",property("integer","Last ruler bar, inclusive"));
        put(segmentProps,"confirmed",property("boolean","Whether this manual key range has been reviewed; default true"));
        put(segmentProps,"tonic",property("integer","C=0 ... B=11"));put(segmentProps,"minor",property("boolean","Natural minor, default false"));
        put(segmentSchema,"properties",segmentProps);put(segmentSchema,"required",juce::Array<juce::var>{"start_bar","end_bar","tonic"});put(segmentSchema,"additionalProperties",false);put(manualSections,"items",segmentSchema);
        add(name,juce::String(name)=="hamood_preview"
            ?"HAMOOD: preview key analysis and harmony pitches without changing the project. Defaults to current selection on focused track."
            :"HAMOOD: create independent harmony tracks with one Undo. Preview first; current selection by default, whole_track must be explicit.",{
            {"track_id",property("string","Source melodic/UTAU/DS track; defaults to focused track")},
            {"whole_track",property("boolean","Process entire source track instead of selection, default false")},
            {"key_mode",property("string","auto (single key), sections (detect modulation), manual")},
            {"tonic",property("integer","Manual tonic: C=0, C#=1 ... B=11")},{"minor",property("boolean","Manual natural minor instead of major")},
            {"section_bars",property("integer","Bars per modulation segment, 1..64, default 8")},{"voices",voiceSteps},{"manual_sections",manualSections},
            {"preserve_pitch",property("boolean","Preserve pitch shape and additive DS offsets, default true")},
            {"gain_db",property("number","New track gain relative to source, -36..6 dB, default -6")},
            {"audio_clip_id",property("string","Optional analysed accompaniment clip. Uses cached chords mapped to its current placement, choosing nearby chord tones with phrase voice leading; low-score/uncovered notes retain scale targets.")},
            {"minimum_chord_score",property("number","Uncalibrated BTC score threshold, 0..1, default .35")}});
    }
    const auto legacy = Server::diagnosticTools();
    for (auto t : *legacy.getArray())
    {
        if (str(t,"name")=="project_snapshot")
            put(t,"description","Explicit whole-project dump including dense curves. For current selected notes use editor_selection instead / 整个工程；当前选区请用 editor_selection");
        tools.getArray()->add(t);
    }
    for (auto& t : *tools.getArray())
    {
        const auto name = str(t,"name");
        auto schema = t["inputSchema"], props = schema["properties"];
        put(props,"expected_revision",property("integer","Required for live edits: revision read from editor_status/editor_query"));
        put(props,"session_id",property("string","Optional assertion of the connected window session ID"));
        if (!readOnly(name) && !in(name,"editor_cancel_job transport_stop"))
        {
            auto required = schema["required"];
            if (!required.isArray()) required = juce::Array<juce::var>();
            required.getArray()->add("expected_revision"); put(schema,"required",required);
        }
        if (in(name,"project_new project_open import_melodyne")) put(props,"discard_unsaved",property("boolean","Explicitly allow replacing an unsaved document"));
        if (name.startsWith("ds_set_"))
        {
            auto required=schema["required"];
            required.getArray()->add("note_id");
            if(name=="ds_set_timing"){auto timing=props["timing"];put(timing,"type",juce::Array<juce::var>{"object","null"});}
            if(name=="ds_set_pronunciation")required.getArray()->add("text");
            else if(name=="ds_set_timing")required.getArray()->add("timing");
            else {required.getArray()->add("points");if(name=="ds_set_parameter"){required.getArray()->add("flag");required.getArray()->add("layer");}}
        }
        if (name == "export_wav")
        {
            put(props,"channels",property("integer","1=mono, 2=stereo, default 2"));
            put(props,"sample_rate",property("integer","8000..192000 Hz; 0 follows device"));
            put(props,"bit_depth",property("integer","16/24 PCM or 32 float, default 24"));
        }
    }
    return tools;
}
juce::String validateValue(const juce::var& value,const juce::var& schema,const juce::String& path)
{
    const auto type=schema["type"];
    auto matches=[&](const juce::String& t){
        if(t=="object")return value.getDynamicObject()!=nullptr;if(t=="array")return value.isArray();
        if(t=="string")return value.isString();if(t=="boolean")return value.isBool();if(t=="null")return value.isVoid();
        if(t=="number"||t=="integer")return (value.isInt()||value.isInt64()||value.isDouble())&&std::isfinite((double)value)&&(t!="integer"||std::floor((double)value)==(double)value);
        return true;
    };
    const bool match=type.isArray()?std::any_of(type.getArray()->begin(),type.getArray()->end(),[&](const auto& t){return matches(t.toString());}):matches(type.toString());
    if(!match)return "Invalid type or non-finite value: "+path;
    if(const auto choices=schema["enum"];choices.isArray()&&!choices.getArray()->contains(value))return "Invalid value: "+path;
    if(value.getDynamicObject()!=nullptr)
    {
        if(const auto required=schema["required"];required.isArray())for(const auto& key:*required.getArray())
            if(!value.hasProperty(key.toString()))return "Missing argument: "+path+"."+key.toString();
        for(const auto& p:value.getDynamicObject()->getProperties())
        {
            const auto child=schema["properties"][p.name];
            if(child.isVoid()){if(!(bool)schema.getProperty("additionalProperties",true))return "Unknown argument: "+path+"."+p.name.toString();continue;}
            const auto error=validateValue(p.value,child,path+"."+p.name.toString());if(error.isNotEmpty())return error;
        }
    }
    if(value.isArray()&&schema["items"].isObject())
        for(int i=0;i<value.size();++i){const auto error=validateValue(value[i],schema["items"],path+"["+juce::String(i)+"]");if(error.isNotEmpty())return error;}
    return {};
}
juce::String validate(const juce::String& name,const juce::var& args)
{
    const auto tools=toolsForWindow();
    for(const auto& tool:*tools.getArray())if(str(tool,"name")==name)return validateValue(args,tool["inputSchema"],"arguments");
    return "Unknown tool: "+name;
}
juce::String targetsValid(const ProjectData& data, const juce::var& args)
{
    std::set<juce::String> tracks,clips,notes;
    for (const auto& t:data.tracks) { tracks.insert(t.id); for(const auto& c:t.clips) { clips.insert(c.id); for(const auto& n:c.notes) notes.insert(n.id); } }
    for (const auto* key:{"track_id","clip_id","note_id"})
    {
        const auto v=str(args,key); if(v.isEmpty())continue;
        const auto& set=juce::String(key)=="track_id"?tracks:juce::String(key)=="clip_id"?clips:notes;
        if (!set.contains(v)) return "Unknown "+juce::String(key)+": "+v;
    }
    for(const auto& id:ids(args["note_ids"])) if(!notes.contains(id)) return "Unknown note_id: "+id;
    return {};
}
}

struct MainComponent::McpState
{
    struct Job
    {
        juce::String id=juce::Uuid().toString(), kind, state="running";
        juce::var result;
        std::uint64_t revision=0;
        juce::int64 renderFingerprint=0;
        std::vector<juce::String> selection;
        juce::String track;
        double began=juce::Time::getMillisecondCounterHiRes(), timeout=300;
        juce::var args;
        std::shared_ptr<std::atomic<bool>> cancel=std::make_shared<std::atomic<bool>>(false);
    };
    std::unique_ptr<Server> server;
    std::unique_ptr<backend::LiveMcpBridge> bridge;
    std::vector<std::shared_ptr<Job>> jobs;
    std::shared_ptr<Job> dsJob;
    juce::StringArray roots;
    bool alive=true;
    juce::String document=juce::Uuid().toString();
    std::shared_ptr<Job> add(const juce::String& kind, std::uint64_t revision, juce::var args={})
    {
        auto j=std::make_shared<Job>(); j->kind=kind;j->revision=revision;j->args=args;
        j->timeout=(double)args.getProperty("timeout_seconds",300.0);
        jobs.push_back(j);
        while(jobs.size()>64){auto it=std::find_if(jobs.begin(),jobs.end(),[](const auto& old){return old->state!="running";});if(it==jobs.end())break;jobs.erase(it);}
        return j;
    }
};

void MainComponent::startLiveMcp()
{
    if(mcp)return;
    mcp=std::make_shared<McpState>();
    mcp->server=std::make_unique<Server>(project,&audio);
    mcp->bridge=std::make_unique<backend::LiveMcpBridge>(
        [this](const auto& request,bool& respond){return handleLiveMcp(request,respond);},
        [this]{return liveMcpStatus();});
}
void MainComponent::stopLiveMcp()
{
    if(!mcp)return;
    mcp->alive=false;
    for(auto& j:mcp->jobs) j->cancel->store(true);
    mcp->bridge.reset();mcp.reset();
}
juce::var MainComponent::liveMcpStatus() const
{
    const auto data=project.snapshot();
    auto v=obj();put(v,"mode","live");put(v,"project_name",data.name);
    put(v,"project_path",currentProjectFile.getFullPathName());
    put(v,"revision",(juce::int64)project.revisionNumber());put(v,"unsaved",project.revisionNumber()!=savedProjectRevision);
    put(v,"session_id",mcp&&mcp->bridge?mcp->bridge->sessionId():juce::String());
    put(v,"document_id",mcp?mcp->document:juce::String());
    put(v,"selected_track_id",selectedTrackId);put(v,"selected_clip_id",selectedClipId);
    juce::Array<juce::var> clips, lanes;
    for(const auto& id:timeline.selectedClipIds()) clips.add(id);
    for(const auto& id:timeline.selectedTrackIds()) lanes.add(id);
    put(v,"selected_clip_ids",clips);put(v,"selected_track_ids",lanes);
    juce::Array<juce::var> selected;for(const auto& id:pianoRoll.selectedNoteIds()) selected.add(id);put(v,"selected_note_ids",selected);
    int selectedNotes=0,selectedRegions=0;const auto chosen=pianoRoll.selectedNoteIds();
    for(const auto& track:data.tracks)for(const auto& clip:track.clips){int count=0;for(const auto& note:clip.notes)if(std::find(chosen.begin(),chosen.end(),note.id)!=chosen.end())++count;selectedNotes+=count;if(count)++selectedRegions;}
    put(v,"selected_note_count",selectedNotes);put(v,"selected_region_count",selectedRegions);
    put(v,"playing",audio.isPlaying());put(v,"position_seconds",audio.position());
    const auto renderProgress=audio.renderProgress();put(v,"rendering",renderProgress.has_value());
    put(v,"render_progress",renderProgress.value_or(1.0));put(v,"backend",audio.activeRenderBackends());
    put(v,"render_warning",audio.activeRenderWarnings());put(v,"ds_busy",diffSingerBusy);
    put(v,"ds_inference",backend::DiffSingerRenderer::inferenceOptions());
    put(v,"can_undo",project.canUndo());put(v,"can_redo",project.canRedo());return v;
}
juce::var MainComponent::liveMcpNotes(const juce::var& args, bool selectionOnly) const
{
    const auto scope = selectionOnly ? juce::String("selection") : args.getProperty("scope","selection").toString();
    const auto data = project.snapshot();
    const auto selected = pianoRoll.selectedNoteIds();
    struct Row { const TrackData* track; const ClipData* clip; const NoteData* note; double start; };
    std::vector<Row> rows;
    for (const auto& track : data.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
            {
                if (scope=="selection" && std::find(selected.begin(),selected.end(),note.id)==selected.end()) continue;
                if (args.hasProperty("track_id") && str(args,"track_id")!=track.id) continue;
                const auto start = clip.startSeconds + note.startSeconds;
                if (args.hasProperty("from_seconds") && start+note.durationSeconds<=(double)args["from_seconds"]) continue;
                if (args.hasProperty("to_seconds") && start>=(double)args["to_seconds"]) continue;
                rows.push_back({&track,&clip,&note,start});
            }
    // Equal-time notes retain track/clip/note order, independent of selection click order.
    std::stable_sort(rows.begin(),rows.end(),[](const auto& a,const auto& b){return a.start<b.start;});
    auto result=liveMcpStatus();
    const auto total=static_cast<int>(rows.size());
    const int offset=std::clamp((int)args["offset"],0,total);
    const int limit=juce::jlimit(1,256,(int)args.getProperty("limit",64));
    const bool curves=(bool)args.getProperty("include_curves",false);
    juce::Array<juce::var> out, tracks;
    juce::StringArray seenTracks;
    double end=rows.empty()?0.0:rows.front().start;
    for (const auto& row : rows)
    {
        end=std::max(end,row.start+row.note->durationSeconds);
        if (!seenTracks.contains(row.track->id))
        {
            seenTracks.add(row.track->id);auto t=obj();
            put(t,"id",row.track->id);put(t,"name",row.track->name);tracks.add(t);
        }
    }
    juce::String summary;
    for (int i=offset;i<std::min(total,offset+limit);++i)
    {
        const auto& row=rows[(size_t)i];const auto& note=*row.note;
        // Only serialize rows on this page. Unselected curves never enter the response.
        auto n=selectionOnly&&!curves?obj():Server::noteJson(note,curves);
        put(n,"id",note.id);put(n,"label",note.label);put(n,"lyric",note.label);
        put(n,"midi",note.midiNote);
        const auto pitch=juce::MidiMessage::getMidiNoteName(note.midiNote,true,true,4);
        put(n,"pitch_name",pitch);
        put(n,"start_seconds",note.startSeconds);put(n,"duration_seconds",note.durationSeconds);
        put(n,"track_id",row.track->id);put(n,"track_name",row.track->name);
        put(n,"clip_id",row.clip->id);put(n,"project_start_seconds",row.start);
        put(n,"project_end_seconds",row.start+note.durationSeconds);
        put(n,"diffsinger",trackIsDiffSinger(*row.track));
        put(n,"tail_fade",note.utauTailFadeMode==1?"linear":note.utauTailFadeMode==2?"smooth":"off");
        put(n,"tail_fade_start_percent",note.utauTailFade.startFraction*100.0);
        put(n,"tail_fade_end_percent",note.utauTailFade.endFraction*100.0);
        put(n,"tail_fade_start_gain_percent",note.utauTailFade.startGain*100.0);
        put(n,"tail_fade_end_gain_percent",note.utauTailFade.endGain*100.0);
        put(n,"tail_fade_curve_power",note.utauTailFade.curvePower*1.0);
        put(n,"tail_fade_custom_curve",note.utauTailFade.customCurve);
        put(n,"head_envelope",note.utauTailFade.head.mode==0?"off":note.utauTailFade.head.mode==1?"linear":"curve");
        put(n,"head_envelope_custom_curve",note.utauTailFade.head.customCurve);
        put(n,"head_envelope_start_percent",note.utauTailFade.head.startFraction*100.0);
        put(n,"head_envelope_end_percent",note.utauTailFade.head.endFraction*100.0);
        put(n,"head_envelope_start_gain_percent",note.utauTailFade.head.startGain*100.0);
        put(n,"head_envelope_end_gain_percent",note.utauTailFade.head.endGain*100.0);
        put(n,"head_envelope_curve_power",note.utauTailFade.head.curvePower*1.0);
        put(n,"head_envelope_control1_time_percent",note.utauTailFade.head.control1Time*100.0);
        put(n,"head_envelope_control1_progress_percent",note.utauTailFade.head.control1Progress*100.0);
        put(n,"head_envelope_control2_time_percent",note.utauTailFade.head.control2Time*100.0);
        put(n,"head_envelope_control2_progress_percent",note.utauTailFade.head.control2Progress*100.0);
        put(n,"tail_fade_control1_time_percent",note.utauTailFade.control1Time*100.0);
        put(n,"tail_fade_control1_progress_percent",note.utauTailFade.control1Progress*100.0);
        put(n,"tail_fade_control2_time_percent",note.utauTailFade.control2Time*100.0);
        put(n,"tail_fade_control2_progress_percent",note.utauTailFade.control2Progress*100.0);
        put(n,"diffsinger_pronunciation",note.diffSingerPronunciation);
        out.add(n);
        if(summary.isNotEmpty())summary+="\n";
        summary+=juce::String(i+1)+". "+note.label+" | "+pitch+" (MIDI "+juce::String(note.midiNote)+") | "
            +juce::String(row.start,3)+" - "+juce::String(row.start+note.durationSeconds,3)+" s | "+row.track->name;
    }
    put(result,"scope",scope);put(result,"notes",out);put(result,"total",total);
    put(result,"selection_empty",selected.empty());put(result,"returned_count",out.size());
    put(result,"offset",offset);put(result,"next_offset",offset+out.size()<total?juce::var(offset+out.size()):juce::var());
    put(result,"tracks",tracks);put(result,"include_curves",curves);
    put(result,"time_basis","project_start/end_seconds are absolute; start_seconds and curve times are clip-local and note-local respectively. Pitch names use C4 = MIDI 60.");
    juce::var range;
    if(!rows.empty()) { range=obj();put(range,"start_seconds",rows.front().start);put(range,"end_seconds",end); }
    put(result,"range",range);
    put(result,"summary",summary.isEmpty()?juce::String("No notes on this page / 当前页没有音符"):summary);
    put(result,"message",scope=="selection"&&selected.empty()
        ?juce::String::fromUTF8("当前没有选中音符。请先在钢琴卷帘中点选或框选；不会自动读取整个工程。")
        :juce::String::fromUTF8("仅读取当前范围，未改变选区或工程。"));
    return result;
}

void MainComponent::finishMcpDiffSinger(bool success,const juce::String& message)
{
    if(!mcp||!mcp->dsJob)return;
    auto j=std::exchange(mcp->dsJob,{});j->state=j->cancel->load()?"cancelled":success?"completed":"failed";
    j->result=success?ok(liveMcpStatus()):fail(message);
}
bool MainComponent::reportMcpDiffSingerError(const juce::String& message)
{
    if(!mcp||!mcp->dsJob)return false;
    finishMcpDiffSinger(false,message);return true;
}

void MainComponent::liveMcpDocumentChanged()
{
    // Single-note tools belong to this document; bank tools edit external files.
    noteOtoWindows.clear();
    if(mcp)mcp->document=juce::Uuid().toString();
}
void MainComponent::showDiffSingerError(const juce::String& message)
{
    if(!reportMcpDiffSingerError(message))showError(message);
}

bool MainComponent::mcpExportInProgress() const
{
    return mcp&&std::any_of(mcp->jobs.begin(),mcp->jobs.end(),[](const auto& j){return j->state=="running"&&j->kind=="render:export_wav";});
}
void MainComponent::pollLiveMcp()
{
    if(!mcp)return;
    for(auto& j:mcp->jobs)
    {
        if(j->state!="running"||!(j->kind.startsWith("render:")||j->kind=="transport_play"))continue;
        if(j->cancel->load()){j->state="cancelled";continue;}
        if(project.revisionNumber()!=j->revision && renderEditFingerprint(project.snapshot())!=j->renderFingerprint){j->state="failed";j->result=fail("STALE_REVISION: document changed before rendering completed");continue;}
        if((juce::Time::getMillisecondCounterHiRes()-j->began)*.001>j->timeout){j->state="failed";j->result=fail("Render timed out");continue;}
        if(j->selection!=pianoRoll.selectedNoteIds()||j->track!=selectedTrackId){j->state="failed";j->result=fail("SELECTION_CHANGED: playback/render scope changed; request again");continue;}
        if(audio.renderProgress())continue;
        if(audio.activeRenderWarnings().isNotEmpty()){j->state="failed";j->result=fail(audio.activeRenderWarnings());continue;}
        if(j->kind=="transport_play")
        {
            if(j->args.hasProperty("position_seconds"))audio.setPosition((double)j->args["position_seconds"]);
            if(j->args.hasProperty("play_until_seconds"))audio.setPlayUntil((double)j->args["play_until_seconds"]);
            juce::String deviceError;
            if(!audio.ensureOutputDevice(deviceError)){j->state="failed";j->result=fail(deviceError);continue;}
            startPreparedPlayback();j->result=ok(liveMcpStatus());
        }
        else if(j->kind=="render:export_wav")
        {
            const auto a=j->args;WavExportOptions opts;opts.channels=(int)a.getProperty("channels",2);
            opts.sampleRate=(int)a.getProperty("sample_rate",0);opts.bitDepth=(int)a.getProperty("bit_depth",24);
            juce::String error;const juce::File file(str(a,"path"));
            if(audio.exportWav(file,error,str(a,"track_id"),(double)a.getProperty("from_seconds",0.0),(double)a.getProperty("to_seconds",0.0),opts))
            {auto v=obj();put(v,"path",file.getFullPathName());put(v,"size",file.getSize());j->result=ok(v);mcp->server->allowFile(file);}
            else j->result=fail(error);
        }
        else j->result=ok(liveMcpStatus());
        j->state=(bool)j->result["isError"]?"failed":"completed";
        if (j->kind=="render:export_wav") syncAudio(project.snapshot());
    }
}

juce::var MainComponent::handleLiveMcp(const juce::var& request,bool& respond)
{
    respond=request.hasProperty("id");
    const auto method=str(request,"method");
    auto reply=obj();put(reply,"jsonrpc","2.0");put(reply,"id",request["id"]);
    if(method=="tools/list") {auto v=obj();put(v,"tools",toolsForWindow());put(reply,"result",v);return reply;}
    if(method!="tools/call")
    {
        if(method=="resources/read" && str(request["params"],"uri")=="hachishifter://editor/selection")
        {
            auto content=obj(), result=obj();
            put(content,"uri","hachishifter://editor/selection");put(content,"mimeType","application/json");
            put(content,"text",juce::JSON::toString(liveMcpNotes(obj(),true),true));
            put(result,"contents",juce::Array<juce::var>{content});put(reply,"result",result);return reply;
        }
        auto response=mcp->server->processRequest(request,respond);
        if(method=="resources/list")
        {
            auto resource=obj();put(resource,"uri","hachishifter://editor/selection");
            put(resource,"name",juce::String::fromUTF8("当前选中音符 / Current selected notes"));
            put(resource,"description","Live GUI selection only; first 64 notes. Use editor_selection with next_offset for remaining pages. Empty selection never expands to the project.");
            put(resource,"mimeType","application/json");
            response["result"]["resources"].getArray()->insert(0,resource);
        }
        if(method=="initialize")response["result"].getDynamicObject()->setProperty("instructions",
            "Connected to a live HachiShifter window. Read editor_status before editing and pass expected_revision. "
            "Use editor_selection FIRST to read what the user currently selected, with lyrics, note names and absolute times. "
            "No selection means no notes; never fall back to the whole project. Read every page using next_offset. "
            "Only use project_snapshot or editor_query scope=project when the user asks for the whole project. "
            "editor_query also defaults to the real GUI selection. Long commands return job_id; poll editor_job_status until completed/failed/cancelled, and check result.isError. "
            "Use editor_batch for one atomic Undo. project_new/open require discard_unsaved=true if dirty. "
            "DS actual, offset and prediction-reference curves are distinct; preserve original references.");
        return response;
    }
    const auto params=request["params"],args=params.getProperty("arguments",obj());
    const auto name=str(params,"name");
    auto perform=[&]() -> juce::var {
        const auto trace=[&](const char* stage){if(juce::SystemStats::getEnvironmentVariable("HACHI_MCP_TRACE",{})=="1")std::cerr<<"mcp stage "<<name<<" "<<stage<<std::endl;};
        trace("validate");
        if(const auto e=validate(name,args);e.isNotEmpty())return fail(e);
        trace("validated");
        if(args.hasProperty("session_id")&&str(args,"session_id")!=mcp->bridge->sessionId())return fail("SESSION_CHANGED: reconnect to the intended window");
        if(args.hasProperty("expected_revision")&&(juce::int64)args["expected_revision"]!=(juce::int64)project.revisionNumber())return fail("STALE_REVISION: query the editor again; nothing changed");
        if(!readOnly(name)&&!in(name,"editor_cancel_job transport_stop")&&juce::Component::getCurrentlyModalComponent()!=nullptr)
            return fail("EDITOR_MODAL: finish or close the active dialog first");
        if(const auto e=targetsValid(project.snapshot(),args);e.isNotEmpty())return fail(e);
        if(name=="editor_status"||name=="transport_status"||name=="render_status")return ok(liveMcpStatus());
        if(name=="editor_query" || name=="editor_selection")
        {
            const auto scope=args.getProperty("scope","selection").toString();
            if(scope!="selection"&&scope!="project")return fail("scope must be selection or project");
            if(args.hasProperty("from_seconds")&&args.hasProperty("to_seconds")&&(double)args["from_seconds"]>=(double)args["to_seconds"])
                return fail("from_seconds must be less than to_seconds");
            return ok(liveMcpNotes(args,name=="editor_selection"));
        }
        if(name=="hamood_alignment_preview")
        {
            const auto data=project.snapshot();const auto* clip=hamoodaudio::audioClip(data,str(args,"clip_id"));
            if(!clip)return fail("Choose an existing accompaniment clip");
            const double from=(double)args.getProperty("from_seconds",clip->startSeconds),to=(double)args.getProperty("to_seconds",clip->startSeconds+clip->durationSeconds);
            const int division=(int)args.getProperty("grid_divisions_per_quarter",2);
            if(to<=from||division<1||division>4)return fail("Invalid alignment range or grid division");
            const auto result=hamoodaudio::alignment(data,*clip,hamoodaudio::cached(hamoodaudio::cacheFolder(currentProjectFile),*clip),from,to,division);
            return (bool)result["ok"]?ok(result):fail(result["error"].toString());
        }
        if(name=="hamood_audio_context")
        {
            const auto data=project.snapshot();const auto* clip=hamoodaudio::audioClip(data,str(args,"clip_id"));
            if(!clip)return fail("Choose an existing accompaniment clip");
            double from=clip->startSeconds,to=from+clip->durationSeconds;
            if(!(bool)args.getProperty("whole_clip",false) && !(args.hasProperty("from_seconds")&&args.hasProperty("to_seconds")))
            {
                const auto selection=pianoRoll.selectedNoteIds();bool found=false;
                from=1.e100;to=-1.e100;
                for(const auto& t:data.tracks)for(const auto& c:t.clips)for(const auto& n:c.notes)
                    if(std::find(selection.begin(),selection.end(),n.id)!=selection.end())
                    {found=true;from=std::min(from,c.startSeconds+n.startSeconds);to=std::max(to,c.startSeconds+n.startSeconds+n.durationSeconds);}
                if(!found)return fail("Select notes, provide from_seconds/to_seconds, or explicitly use whole_clip=true");
            }
            if(args.hasProperty("from_seconds"))from=(double)args["from_seconds"];
            if(args.hasProperty("to_seconds"))to=(double)args["to_seconds"];
            const int offset=(int)args.getProperty("offset",0),limit=(int)args.getProperty("limit",64);
            if(to<=from||offset<0||limit<1||limit>256)return fail("Invalid context range or pagination");
            const auto result=hamoodaudio::context(data,*clip,hamoodaudio::cached(hamoodaudio::cacheFolder(currentProjectFile),*clip),from,to,offset,limit);
            return (bool)result["ok"]?ok(result):fail(result["error"].toString());
        }
        if(name=="hamood_get_context")return ok(hamoodstate::read(project.snapshot()));
        if(name=="hamood_set_context")
        {
            juce::String error;if(!project.setHamoodState(hamoodstate::encode(args["context"]),error))return fail(error);
            return ok(liveMcpStatus());
        }
        if(name=="hamood_preview"||name=="hamood_generate")
        {
            hamood::Options options;juce::String error;
            if(!hamood::parseOptions(hamoodstate::defaults(project.snapshot(),args),selectedTrackId,pianoRoll.selectedNoteIds(),options,error))return fail(error);
            auto data=project.snapshot();
            if(str(args,"audio_clip_id").isNotEmpty())
            {
                if(!hamoodaudio::attach(data,str(args,"audio_clip_id"),hamoodaudio::cacheFolder(currentProjectFile),options,error))return fail(error);
                auto evidence=hamoodstate::read(data);hamoodstate::importChords(data,evidence,options.chords,str(args,"audio_clip_id"));
                options.chords.clear();hamoodstate::attach(data,evidence,options);
            }
            if(str(args,"audio_clip_id").isEmpty())hamoodstate::attach(data,hamoodstate::read(data),options);
            const auto plan=hamood::analyse(data,options);
            if(plan.error.isNotEmpty())return fail(plan.error);
            auto result=plan.json();
            if(name=="hamood_generate")
            {
                if(diffSingerBusy)return fail("Wait for the running DS task before generating harmonies");
                const auto tracks=hamood::generate(data,options,plan);
                if(tracks.empty())return fail("No harmony tracks generated");
                data.hamoodState=hamoodstate::remember(data,options,plan);
                if(str(args,"audio_clip_id").isNotEmpty()){auto state=hamoodstate::read(data);hamoodstate::importChords(data,state,options.chords,str(args,"audio_clip_id"));data.hamoodState=hamoodstate::encode(state);}
                project.replace(std::move(data));juce::Array<juce::var> added;
                for(const auto& id:tracks)added.add(id);put(result,"created_track_ids",added);
            }
            put(result,"revision",(juce::int64)project.revisionNumber());return ok(result);
        }
        if(name=="editor_select")
        {
            if(!args.hasProperty("note_ids")&&!args.hasProperty("track_id"))return fail("Provide note_ids or track_id");
            const auto selected=ids(args["note_ids"]);
            if(args.hasProperty("track_id")) {selectedTrackId=str(args,"track_id");trackList.setSelectedTrack(selectedTrackId);pianoRoll.setFocusedTrack(selectedTrackId);}
            trace("select_focus");
            if(!selected.empty())focusNote(selected.front());
            trace("select_notes");
            pianoRoll.setSelectedNoteIds(selected);
            trace("select_done");
            return ok(liveMcpStatus());
        }
        pollLiveMcp();
        if(name=="editor_job_status"||name=="editor_cancel_job")
        {
            for(const auto& j:mcp->jobs)if(j->id==str(args,"job_id"))
            {
                if(name=="editor_cancel_job"&&j->state=="running") {j->cancel->store(true);if(j->kind.startsWith("render:")||j->kind=="transport_play")j->state="cancelled";}
                auto v=obj();put(v,"job_id",j->id);put(v,"state",j->state);put(v,"kind",j->kind);put(v,"result",j->result);put(v,"revision",(juce::int64)project.revisionNumber());return ok(v);
            }
            return fail("Unknown or expired job_id");
        }
        auto accepted=[&](const std::shared_ptr<McpState::Job>& j){auto v=obj();put(v,"job_id",j->id);put(v,"state",j->state);put(v,"poll_tool","editor_job_status");return ok(v);};
        if(name=="ds_generate_pitch"||name=="ds_generate_parameters")
        {
            if(diffSingerBusy)return fail("Another DS task is running");
            const bool whole=(bool)args.getProperty("whole_track",false);
            const auto selected=pianoRoll.selectedNoteIds();
            if(!whole&&selected.empty())return fail("Select notes first, or explicitly set whole_track=true");
            const auto data=project.snapshot();
            const auto t=std::find_if(data.tracks.begin(),data.tracks.end(),[&](const auto& t){return t.id==selectedTrackId&&trackIsDiffSinger(t);});
            if(t==data.tracks.end())return fail("Focused track is not DiffSinger");
            for(const auto& id:selected)if(!whole){bool found=false;for(const auto& c:t->clips)if(!c.muted)for(const auto& n:c.notes)found|=n.id==id;if(!found)return fail("Selection must belong to this track's unmuted clips");}
            auto j=mcp->add(name,project.revisionNumber(),args);mcp->dsJob=j;
            if(name=="ds_generate_pitch")generateDiffSingerPitch(whole);else generateDiffSingerParameters(whole);
            if(diffSingerBusy)j->cancel=diffSingerCancel;else if(mcp->dsJob)finishMcpDiffSinger(false,"No eligible notes or unsupported voicebank");
            return accepted(j);
        }
        if(name=="transport_stop") {audio.stop();for(auto& j:mcp->jobs)if(j->kind=="transport_play"&&j->state=="running"){j->cancel->store(true);j->state="cancelled";}return ok(liveMcpStatus());}
        if(name=="transport_seek") {audio.setPosition((double)args["position_seconds"]);return ok(liveMcpStatus());}
        if(name=="utau_render_selection")
        {
            std::vector<juce::String> chosen; juce::String error;
            if (!backend::McpServer::resolveUtauRenderSelection(project.snapshot(), args,
                pianoRoll.selectedNoteIds(), chosen, error)) return fail(error);
            audio.setUtauRenderNoteSelection(chosen);activeUtauSelectionCount=(int)chosen.size();
            syncAudio(project.snapshot());return ok(liveMcpStatus());
        }
        if(name=="render_prepare"||name=="transport_play"||name=="export_wav")
        {
            if(exportWaitingForRender)return fail("A GUI export is running");
            for(const auto& j:mcp->jobs)if(j->state=="running"&&(j->kind.startsWith("render:")||j->kind=="transport_play"))return fail("Another MCP render/play/export job is running");
            if(name=="export_wav")
            {
                WavExportOptions opts;opts.channels=(int)args.getProperty("channels",2);opts.sampleRate=(int)args.getProperty("sample_rate",0);opts.bitDepth=(int)args.getProperty("bit_depth",24);
                if(!opts.isValid())return fail("Invalid WAV channels, sample_rate or bit_depth");
                activeUtauSelectionCount=audio.selectEveryUtauNote(project.snapshot());
            }
            else if(name=="transport_play")armSelectionPlayback();
            audio.syncProject(project.snapshot(),name=="export_wav");
            auto j=mcp->add(name=="transport_play"?name:"render:"+name,project.revisionNumber(),args);
            j->renderFingerprint=renderEditFingerprint(project.snapshot());j->selection=pianoRoll.selectedNoteIds();j->track=selectedTrackId;
            return accepted(j);
        }
        if(name=="undo"||name=="redo")return mcp->server->executeTool(name,args);
        if(name=="project_snapshot")return mcp->server->executeTool(name,args);
        if(name=="project_save")
        {
            // Use the same prediction collection, path bookkeeping and cache migration as Ctrl+S.
            rememberDiffSingerParameters();
            juce::String error;const juce::File file(str(args,"path"));
            if(!project.save(file,error))return fail(error);
            const bool cached=backend::DiffSingerRenderer::saveProjectCache(file);
            discardProjectRecovery();
            currentProjectFile=file;savedProjectRevision=project.revisionNumber();addRecentProject(file);mcp->server->allowFile(file);
            auto v=liveMcpStatus();put(v,"cache_saved",cached);return ok(v);
        }
        if(name=="set_utau_resampler")return mcp->server->executeTool(name,args);
        const bool replacing=in(name,"project_new project_open import_melodyne");
        if(replacing&&project.revisionNumber()!=savedProjectRevision&&!(bool)args.getProperty("discard_unsaved",false))
            return fail("UNSAVED_DOCUMENT: save first, or explicitly pass discard_unsaved=true");
        // Read/file/import operations may be slow. They run on a captured draft,
        // then compare revision before a single, undoable commit on the UI thread.
        const bool background=in(name,"hamood_analyse_audio project_open import_audio analyse_audio analysis_status import_midi import_ust import_melodyne export_midi export_ust sample_settings_read sample_settings_save oto_import oto_export jie_oto_create voicebank_import read_file list_directory ds_capabilities ds_query_phonemes");
        auto draft=std::make_shared<ProjectModel>();draft->replace(project.snapshot());
        auto roots=mcp->roots;for(const auto& r:Server::projectRoots(project.snapshot()))roots.addIfNotAlreadyThere(r.getFullPathName());
        if(currentProjectFile!=juce::File())roots.addIfNotAlreadyThere(currentProjectFile.getParentDirectory().getFullPathName());
        const auto before=draft->contentFingerprint();const auto revision=project.revisionNumber();
        const auto focusedTrack=selectedTrackId;
        const auto hamoodCache=hamoodaudio::cacheFolder(currentProjectFile);
        const auto capabilities=diffSingerCapabilities;
        const auto selection=pianoRoll.selectedNoteIds();
        const auto cacheSession=backend::DiffSingerRenderer::cacheSession();
        const auto commandCancel=std::make_shared<std::atomic<bool>>(false);
        auto execute=[draft,roots,name,args,focusedTrack,capabilities,selection,cacheSession,commandCancel,hamoodCache]() -> juce::var {
            Server server(*draft,nullptr,roots);
            if(name=="hamood_analyse_audio")
            {
                const auto data=draft->snapshot();const auto* clip=hamoodaudio::audioClip(data,str(args,"clip_id"));
                if(!clip)return fail("Choose an existing accompaniment clip");
                const auto result=hamoodaudio::analyse(data,clip->id,hamoodCache,(bool)args.getProperty("force",false),commandCancel);
                return (bool)result["ok"]?ok(hamoodaudio::summary(result,*clip)):fail(result["error"].toString());
            }
            if(name=="ds_query_phonemes")
            {
                const auto data=draft->snapshot();const bool whole=(bool)args.getProperty("whole_track",false);
                if(!whole&&selection.empty())return fail("Select DS notes first or use whole_track=true");
                for(const auto& t:data.tracks)if(t.id==focusedTrack&&trackIsDiffSinger(t))
                {
                    juce::StringArray noteIds;auto r=diffSingerRequest(data,t,noteIds);r.diffSingerCacheSession=cacheSession;
                    auto request=backend::DiffSingerRenderer::requestJson(r,"timing");
                    if(!whole)
                    {
                        juce::Array<juce::var> chosen;
                        for(int i=0;i<noteIds.size();++i)
                            if(std::find(selection.begin(),selection.end(),noteIds[i])!=selection.end())chosen.add(juce::String(i));
                        if(chosen.isEmpty())return fail("Selected notes are not in the available DS clips");
                        request.getDynamicObject()->setProperty("retake_ids",chosen);
                    }
                    auto result=backend::DiffSingerRenderer::invoke(request,[commandCancel]{return commandCancel->load();});
                    if(!(bool)result["ok"])return fail(result["error"].toString());
                    juce::Array<juce::var> rows;
                    if(const auto ph=result["phonemes"];ph.isArray())for(auto row:*ph.getArray())
                    {
                        const auto index=row["id"].toString().getIntValue();
                        if(row["id"].toString().isEmpty()||index<0||index>=noteIds.size())continue;
                        const auto id=noteIds[index];if(!whole&&std::find(selection.begin(),selection.end(),id)==selection.end())continue;
                        put(row,"note_id",id);rows.add(row);
                    }
                    put(result,"phonemes",rows);return ok(result);
                }
                return fail("Focused track is not DiffSinger");
            }
            if(name=="ds_capabilities")
            {
                for(const auto& t:draft->snapshot().tracks)if(t.id==focusedTrack&&trackIsDiffSinger(t))
                {
                    backend::UtauRenderRequest r;r.voicebankDirectory=t.voicebankDirectory;r.diffSingerLanguage=t.diffSingerLanguage;r.diffSingerCacheSession=cacheSession;
                    const auto result=backend::DiffSingerRenderer::invoke(backend::DiffSingerRenderer::requestJson(r,"inspect"),[commandCancel]{return commandCancel->load();});
                    return (bool)result["ok"]?ok(result):fail(result["error"].toString());
                }
                return fail("Focused track is not DiffSinger");
            }
            auto one=[&](const juce::String& command,const juce::var& a) -> juce::var {
                if(const auto e=targetsValid(draft->snapshot(),a);e.isNotEmpty())return fail(e);
                if(command.startsWith("ds_set_"))
                {
                    const auto data=draft->snapshot();const TrackData* owner=nullptr;for(const auto& t:data.tracks)for(const auto& c:t.clips)for(const auto& n:c.notes)if(n.id==str(a,"note_id"))owner=&t;
                    if(!owner||!trackIsDiffSinger(*owner))return fail("Target is not a DS note");
                    if(command=="ds_set_parameter")
                    {
                        if(str(a,"layer")=="actual"&&!in(str(a,"flag"),"ENE BREC TENC VOIC"))return fail("No actual-prediction layer for this flag");
                        const auto key="DS:"+str(a,"flag");
                        const auto found=capabilities.find(owner->voicebankDirectory.getFullPathName());
                        bool supported=false;
                        if(found!=capabilities.end()&&found->second.isArray())for(const auto& entry:*found->second.getArray())
                            if(entry["key"].toString()==key&&(bool)entry["supported"])supported=true;
                        if(!supported)return fail("Parameter unsupported or capabilities not loaded; focus its DS track and query ds_capabilities first");
                    }
                    bool changed=false;
                    if(command=="ds_set_pronunciation")changed=draft->applyDiffSingerPronunciation(draft->revisionNumber(),owner->id,{{str(a,"note_id"),str(a,"text")}},owner->diffSingerDictionary);
                    else if(command=="ds_set_timing")
                    {
                        changed=draft->applyDiffSingerTiming(draft->revisionNumber(),{{str(a,"note_id"),a["timing"].isVoid()?juce::String():juce::JSON::toString(a["timing"],true)}});
                        if(!changed)return fail("Invalid phoneme timing context");
                    }
                    else
                    {
                        if(!a["points"].isArray())return fail("points must be an array");
                        std::vector<PitchCurveEditPoint> pitch;std::vector<FlagCurvePoint> flags;double last=-1e100;
                        for(const auto& p:*a["points"].getArray())
                        {
                            if(!p.isArray()||p.size()!=2||!(p[0].isInt()||p[0].isInt64()||p[0].isDouble())||!(p[1].isInt()||p[1].isInt64()||p[1].isDouble())||!std::isfinite((double)p[0])||!std::isfinite((double)p[1])||(double)p[0]<last)return fail("Points must contain finite [seconds,value] pairs in time order");
                            last=(double)p[0];pitch.push_back({(double)p[0],(float)p[1]});flags.push_back({(double)p[0],(float)p[1]});
                        }
                        if(command=="ds_set_pitch_offset")changed=draft->setDiffSingerPitchOffset(str(a,"note_id"),std::move(pitch));
                        else if(str(a,"layer")=="actual")changed=draft->setDiffSingerParameterCurve(str(a,"note_id"),"DS:ABS:"+str(a,"flag"),std::move(flags));
                        else if(str(a,"layer")=="offset")changed=draft->setNoteUtauFlagCurve(str(a,"note_id"),"DS:"+str(a,"flag"),std::move(flags));
                        else return fail("layer must be actual or offset");
                    }
                    return changed?Server::result("ok"):Server::result("already_satisfied");
                }
                return server.executeTool(command,a);
            };
            if(name=="editor_batch")
            {
                const auto commands=args["commands"];
                if(!commands.isArray()||commands.size()<1||commands.size()>128)return fail("commands requires 1..128 entries");
                for(const auto& c:*commands.getArray())
                {
                    const auto command=str(c,"name");auto a=c.getProperty("arguments",obj());
                    if(!in(command,"add_track set_track set_clip move_clip resize_clip duplicate_clip transpose_note edit_notes_pitch duplicate_notes resize_note set_note set_pitch_curve add_note remove_note toggle_note_connection remove_clip remove_track set_tempo ds_set_pitch_offset ds_set_parameter ds_set_pronunciation ds_set_timing"))return fail("Command not allowed in atomic batch: "+command);
                    put(a,"expected_revision",(juce::int64)draft->revisionNumber());
                    if(const auto e=validate(command,a);e.isNotEmpty())return fail(e);
                    const auto r=one(command,a);if((bool)r["isError"])return r;
                }
                return Server::result("Batch applied as one Undo");
            }
            return one(name,args);
        };
        auto apply=[this,draft,before,revision,name,args,replacing](juce::var result) -> juce::var {
            if((bool)result["isError"])return result;
            if(name=="hamood_analyse_audio" && project.revisionNumber()!=revision)return fail("STALE_REVISION: audio cached, but project changed; query hamood_audio_context again");
            const bool changed=draft->contentFingerprint()!=before;
            if((changed||replacing)&&project.revisionNumber()!=revision)return fail("STALE_REVISION: document changed during processing; draft discarded");
            if(replacing)project.resetDocument(draft->snapshot());
            else if(changed)project.replace(draft->snapshot());
            if(replacing)
            {
                discardProjectRecovery();
                audio.stop();audio.setPosition(0);audio.setUtauRenderNoteSelection({});pianoRoll.clearNoteSelection();selectedTrackId.clear();selectedClipId.clear();selectedNoteId.clear();
                pianoRoll.setFocusedTrack({});pianoRoll.setFocusedClip({});
                const juce::File file(str(args,"path"));
                currentProjectFile=name=="project_open"&&file.hasFileExtension("hjpx;hspx")?file:juce::File();
                backend::DiffSingerRenderer::openProjectCache(currentProjectFile);mcp->document=juce::Uuid().toString();
                savedProjectRevision=currentProjectFile!=juce::File()||name=="project_new"?project.revisionNumber():0;
                if(currentProjectFile!=juce::File())addRecentProject(currentProjectFile);
            }
            for(const auto* key:{"path","audio_path","oto_path","voicebank_path","voicebank_directory"})if(str(args,key).isNotEmpty())
            {const juce::File f(str(args,key));mcp->server->allowFile(f);mcp->roots.addIfNotAlreadyThere((f.isDirectory()?f:f.getParentDirectory()).getFullPathName());}
            put(result,"revision",(juce::int64)project.revisionNumber());put(result,"changed",changed);return result;
        };
        if(!background)return apply(execute());
        if(std::count_if(mcp->jobs.begin(),mcp->jobs.end(),[](const auto& j){return j->state=="running";})>=4)return fail("Too many pending jobs; wait or cancel first");
        auto j=mcp->add(name,revision,args);j->cancel=commandCancel;const auto state=mcp;const auto document=mcp->document;
        const juce::Component::SafePointer<MainComponent> safe(this);
        juce::Thread::launch([safe,state,j,document,execute,apply] {
            juce::var result;
            try {result=execute();} catch(const std::exception& e){result=fail(e.what());}catch(...){result=fail("Background command failed");}
            juce::MessageManager::callAsync([safe,state,j,document,result,apply] {
                if(safe==nullptr||!state->alive)return;
                if(j->cancel->load()){j->state="cancelled";j->result=fail("Cancelled; draft not applied");return;}
                if(state->document!=document){j->state="failed";j->result=fail("DOCUMENT_CHANGED: draft discarded");return;}
                j->result=apply(result);j->state=(bool)j->result["isError"]?"failed":"completed";
            });
        });return accepted(j);
    };
    put(reply,"result",perform());return reply;
}
}
