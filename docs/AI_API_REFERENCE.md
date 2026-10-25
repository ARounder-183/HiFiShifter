# 实际接口声明清单（自动生成）

核对日期：2026-10-25。配套解读：[功能主入口与架构审计](AI_INTERFACE_ARCHITECTURE.md)。

这是源文件导航，不是承诺稳定的公共 SDK。仅枚举下列 11 个边界类的公开方法，不把 private 函数、UI 回调、纯计算辅助函数或每个第三方函数算作产品接口。重载各列一行；诊断接口单独标记。源码行号为生成时位置。

生成命令：`python tools/generate-interface-reference.py`。词法提取不是 C++ 编译器，不能用条目数证明所有业务路径正确；新增边界类时需更新脚本 HEADERS。

## ProjectModel（193 个声明）

| 实际签名 | 类型 | 定义入口 |
| --- | --- | --- |
| `ProjectModel()` | 生命周期 | [ProjectModel.h:924](../juce/src/ProjectModel.h#L924) |
| `ProjectData snapshot() const` | 业务／适配 | [ProjectModel.h:926](../juce/src/ProjectModel.h#L926) |
| `std::uint64_t revisionNumber() const` | 业务／适配 | [ProjectModel.h:927](../juce/src/ProjectModel.h#L927) |
| `void replace(ProjectData replacement)` | 业务／适配 | [ProjectModel.h:928](../juce/src/ProjectModel.h#L928) |
| `void resetDocument(ProjectData replacement = {})` | 业务／适配 | [ProjectModel.h:930](../juce/src/ProjectModel.h#L930) |
| `void clear()` | 业务／适配 | [ProjectModel.h:931](../juce/src/ProjectModel.h#L931) |
| `void setAssetManagerVisible(bool visible)` | 业务／适配 | [ProjectModel.h:933](../juce/src/ProjectModel.h#L933) |
| `bool setHamoodState(const juce::String& state, juce::String& error)` | 业务／适配 | [ProjectModel.h:934](../juce/src/ProjectModel.h#L934) |
| `bool undo()` | 业务／适配 | [ProjectModel.h:935](../juce/src/ProjectModel.h#L935) |
| `bool redo()` | 业务／适配 | [ProjectModel.h:936](../juce/src/ProjectModel.h#L936) |
| `bool canUndo() const` | 业务／适配 | [ProjectModel.h:937](../juce/src/ProjectModel.h#L937) |
| `bool canRedo() const` | 业务／适配 | [ProjectModel.h:938](../juce/src/ProjectModel.h#L938) |
| `juce::String addAudioFile(const juce::File& file, double durationSeconds, double startSeconds = 0.0, const juce::String& targetTrackId = {}, const VoicebankOtoEntry* oto = nullptr, UtauMode otoMode = UtauMode::classic)` | 业务／适配 | [ProjectModel.h:940](../juce/src/ProjectModel.h#L940) |
| `juce::String addMaterialPronunciations( const juce::File& audio, const std::vector<SampleRegionSetting>& completeRows, const std::vector<juce::String>& selectedRegionIds, double absoluteStart, const juce::String& targetTrackId, juce::String& error, const juce::String& chosenAlias = {})` | 业务／适配 | [ProjectModel.h:952](../juce/src/ProjectModel.h#L952) |
| `juce::String addTypedTrack(TrackType type = TrackType::hachi, const juce::String& name = {})` | 业务／适配 | [ProjectModel.h:957](../juce/src/ProjectModel.h#L957) |
| `juce::String addTrack(const juce::String& name, bool compose = true, bool referenceOnly = false, bool accompaniment = false, TrackType type = TrackType::hachi)` | 业务／适配 | [ProjectModel.h:959](../juce/src/ProjectModel.h#L959) |
| `void setTrackReferenceOnly(const juce::String& trackId, bool referenceOnly)` | 业务／适配 | [ProjectModel.h:962](../juce/src/ProjectModel.h#L962) |
| `void setTrackName(const juce::String& trackId, const juce::String& name)` | 业务／适配 | [ProjectModel.h:963](../juce/src/ProjectModel.h#L963) |
| `bool setClipNotesIfEmpty(const juce::String& clipId, std::vector<NoteData> notes)` | 业务／适配 | [ProjectModel.h:964](../juce/src/ProjectModel.h#L964) |
| `std::size_t markNativeUnpitchedRegions(const juce::String& clipId)` | 业务／适配 | [ProjectModel.h:965](../juce/src/ProjectModel.h#L965) |
| `bool setNativeTrimSourceAnalysis(const juce::File& file, const std::vector<NoteData>& sourceNotes, const juce::String& expectedSourceFingerprint = {})` | 业务／适配 | [ProjectModel.h:966](../juce/src/ProjectModel.h#L966) |
| `bool setClipAudioAnalysis(const juce::String& clipId, std::vector<NoteData> notes, const ClipData& importedClip = ClipData{})` | 业务／适配 | [ProjectModel.h:968](../juce/src/ProjectModel.h#L968) |
| `bool addMidiFile(const juce::File& file, juce::String& error)` | 业务／适配 | [ProjectModel.h:970](../juce/src/ProjectModel.h#L970) |
| `static std::vector<MidiTrackChoice> midiTrackChoices(const juce::File& file, juce::String& error)` | 业务／适配 | [ProjectModel.h:980](../juce/src/ProjectModel.h#L980) |
| `juce::String addMidiTrack(const juce::File& file, int trackIndex, juce::String& error, const juce::String& targetTrackId = {}, double startSeconds = 0.0)` | 业务／适配 | [ProjectModel.h:989](../juce/src/ProjectModel.h#L989) |
| `static bool writeMidiFile(const ProjectData& data, const juce::File& file, juce::String& error, const juce::String& trackId = {}, bool includeMuted = false)` | 业务／适配 | [ProjectModel.h:998](../juce/src/ProjectModel.h#L998) |
| `bool addUstFile(const juce::File& file, juce::String& error, juce::StringArray& warnings, UstImportMode mode = UstImportMode::addTrack, juce::String* importedTrackId = nullptr, int encoding = 0, TrackType destinationType = TrackType::hachi)` | 业务／适配 | [ProjectModel.h:1016](../juce/src/ProjectModel.h#L1016) |
| `bool exportUst(const juce::File& file, const juce::String& trackId, juce::String& error, juce::StringArray& warnings, int encoding = 0) const` | 业务／适配 | [ProjectModel.h:1021](../juce/src/ProjectModel.h#L1021) |
| `void setTempo(double bpm, int numerator, int denominator = 4)` | 业务／适配 | [ProjectModel.h:1023](../juce/src/ProjectModel.h#L1023) |
| `void setTempoChange(double quarterPosition, double bpm, bool synchronizeNotes = true)` | 业务／适配 | [ProjectModel.h:1026](../juce/src/ProjectModel.h#L1026) |
| `juce::int64 contentFingerprint() const` | 业务／适配 | [ProjectModel.h:1030](../juce/src/ProjectModel.h#L1030) |
| `static juce::String registrationMusicSourceFingerprint(const ProjectData&)` | 业务／适配 | [ProjectModel.h:1032](../juce/src/ProjectModel.h#L1032) |
| `bool adoptRegisteredMaterialBindings(std::uint64_t expectedRevision, const juce::String& expectedFingerprint, const std::vector<ProjectMaterialRegistrationOwnerBinding>& owners, const std::vector<ProjectMaterialRegistrationNoteBinding>& notes, juce::String& error)` | 业务／适配 | [ProjectModel.h:1033](../juce/src/ProjectModel.h#L1033) |
| `void setGridDivision(const juce::String& division)` | 业务／适配 | [ProjectModel.h:1038](../juce/src/ProjectModel.h#L1038) |
| `void setNoteEditDivision(int division)` | 业务／适配 | [ProjectModel.h:1039](../juce/src/ProjectModel.h#L1039) |
| `void setBaseScale(const juce::String& scale)` | 业务／适配 | [ProjectModel.h:1040](../juce/src/ProjectModel.h#L1040) |
| `void setTrackCompose(const juce::String& trackId, bool enabled)` | 业务／适配 | [ProjectModel.h:1041](../juce/src/ProjectModel.h#L1041) |
| `void setTrackMuted(const juce::String& trackId, bool muted)` | 业务／适配 | [ProjectModel.h:1042](../juce/src/ProjectModel.h#L1042) |
| `void setTrackSolo(const juce::String& trackId, bool solo)` | 业务／适配 | [ProjectModel.h:1043](../juce/src/ProjectModel.h#L1043) |
| `void setTrackVolume(const juce::String& trackId, float volume)` | 业务／适配 | [ProjectModel.h:1044](../juce/src/ProjectModel.h#L1044) |
| `void setClipGainEnvelope(const juce::String& clipId, std::vector<GainEnvelopePoint> points)` | 业务／适配 | [ProjectModel.h:1045](../juce/src/ProjectModel.h#L1045) |
| `void setTrackPan(const juce::String& trackId, float pan)` | 业务／适配 | [ProjectModel.h:1046](../juce/src/ProjectModel.h#L1046) |
| `void setTrackSmoothOverlaps(const juce::String& trackId, bool enabled)` | 业务／适配 | [ProjectModel.h:1047](../juce/src/ProjectModel.h#L1047) |
| `void setTrackNormalizeVolume(const juce::String& trackId, bool enabled)` | 业务／适配 | [ProjectModel.h:1048](../juce/src/ProjectModel.h#L1048) |
| `void setTrackNsfSmoothPitchTransitions(const juce::String& trackId, bool enabled)` | 业务／适配 | [ProjectModel.h:1049](../juce/src/ProjectModel.h#L1049) |
| `void setTrackNsfNoiseProtection(const juce::String& trackId, bool enabled)` | 业务／适配 | [ProjectModel.h:1050](../juce/src/ProjectModel.h#L1050) |
| `void setTrackVoicebankDirectory(const juce::String& trackId, const juce::File& directory)` | 业务／适配 | [ProjectModel.h:1051](../juce/src/ProjectModel.h#L1051) |
| `void setTrackUtauConsonantVelocity(const juce::String& trackId, int velocity)` | 业务／适配 | [ProjectModel.h:1053](../juce/src/ProjectModel.h#L1053) |
| `void setTrackUtauGlobalFlags(const juce::String& trackId, const juce::String& flags)` | 业务／适配 | [ProjectModel.h:1054](../juce/src/ProjectModel.h#L1054) |
| `void setUtauMode(UtauMode mode)` | 业务／适配 | [ProjectModel.h:1055](../juce/src/ProjectModel.h#L1055) |
| `void setTrackUtauMode(const juce::String& trackId, UtauMode mode)` | 业务／适配 | [ProjectModel.h:1056](../juce/src/ProjectModel.h#L1056) |
| `void setTrackChineseCvvc(const juce::String& trackId, bool enabled)` | 业务／适配 | [ProjectModel.h:1057](../juce/src/ProjectModel.h#L1057) |
| `void setTrackPianoPreview(const juce::String& trackId, bool enabled)` | 业务／适配 | [ProjectModel.h:1058](../juce/src/ProjectModel.h#L1058) |
| `void setTrackCvvcCurveCrossfade(const juce::String& trackId, bool curved)` | 业务／适配 | [ProjectModel.h:1059](../juce/src/ProjectModel.h#L1059) |
| `void setTrackOutputEngine(const juce::String& trackId, UtauOutputEngine engine, const juce::File& resampler = {}, const juce::File& wavtool = {})` | 业务／适配 | [ProjectModel.h:1060](../juce/src/ProjectModel.h#L1060) |
| `void setNotesVibrato(const std::vector<juce::String>& noteIds, const NoteData& parameters, bool enabled, std::optional<bool> realLine = std::nullopt)` | 业务／适配 | [ProjectModel.h:1062](../juce/src/ProjectModel.h#L1062) |
| `void setNotesVibratoRealLine(const std::vector<juce::String>& noteIds, bool enabled)` | 业务／适配 | [ProjectModel.h:1065](../juce/src/ProjectModel.h#L1065) |
| `bool bakeNoteVibratoIntoPitch(const juce::String& noteId)` | 业务／适配 | [ProjectModel.h:1069](../juce/src/ProjectModel.h#L1069) |
| `void setNotesUtauSplice(const std::vector<juce::String>& noteIds, bool enabled)` | 业务／适配 | [ProjectModel.h:1070](../juce/src/ProjectModel.h#L1070) |
| `bool setPointCurves(std::vector<PointCurveEdit> edits, std::uint64_t expectedRevision, juce::String* error = nullptr)` | 业务／适配 | [ProjectModel.h:1097](../juce/src/ProjectModel.h#L1097) |
| `bool previewPointCurves(const std::vector<PointCurveEdit>& edits, std::uint64_t expectedRevision, PointCurvePlanPreview&, juce::String& error) const` | 业务／适配 | [ProjectModel.h:1099](../juce/src/ProjectModel.h#L1099) |
| `bool resetCvvcBoundaries(const std::vector<juce::String>& noteIds, std::uint64_t expectedRevision)` | 业务／适配 | [ProjectModel.h:1102](../juce/src/ProjectModel.h#L1102) |
| `bool setCvvcBoundary(const juce::String& noteId, int rightPart, double position, std::uint64_t expectedRevision)` | 业务／适配 | [ProjectModel.h:1103](../juce/src/ProjectModel.h#L1103) |
| `bool setCvvcEnvelopes(std::vector<CvvcEnvelopeEdit>, std::uint64_t expectedRevision)` | 业务／适配 | [ProjectModel.h:1104](../juce/src/ProjectModel.h#L1104) |
| `bool setCvvcEnvelope(const juce::String& noteId, int part, const juce::String& flag, backend::CvvcEnvelopeCurve points, std::uint64_t expectedRevision)` | 业务／适配 | [ProjectModel.h:1105](../juce/src/ProjectModel.h#L1105) |
| `void setNotesUtauFlagCurveEnabled(const std::vector<juce::String>& noteIds, bool enabled)` | 业务／适配 | [ProjectModel.h:1107](../juce/src/ProjectModel.h#L1107) |
| `bool resetNotesUtauFlagCurves(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1114](../juce/src/ProjectModel.h#L1114) |
| `bool resetNotesUtauFlagCurve(const std::vector<juce::String>& noteIds, const juce::String& flag)` | 业务／适配 | [ProjectModel.h:1116](../juce/src/ProjectModel.h#L1116) |
| `bool setNoteUtauFlagCurves(const juce::String& flag, const std::map<juce::String, std::vector<FlagCurvePoint>>& edits, std::optional<std::uint64_t> expectedRevision = std::nullopt, bool enabledOnly = true)` | 业务／适配 | [ProjectModel.h:1120](../juce/src/ProjectModel.h#L1120) |
| `bool setNoteUtauFlagCurve(const juce::String& noteId, const juce::String& flag, std::vector<FlagCurvePoint> points)` | 业务／适配 | [ProjectModel.h:1123](../juce/src/ProjectModel.h#L1123) |
| `void setNotesRegionFlags(const std::vector<juce::String>& noteIds, bool split, const juce::String& first, const juce::String& second, const juce::String& third, const juce::String& fourth)` | 业务／适配 | [ProjectModel.h:1125](../juce/src/ProjectModel.h#L1125) |
| `void setNotesUtauJieSplitCleared(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1129](../juce/src/ProjectModel.h#L1129) |
| `void setNotesUtauJieSplit(const std::vector<juce::String>& noteIds, double first, double second, double third)` | 业务／适配 | [ProjectModel.h:1130](../juce/src/ProjectModel.h#L1130) |
| `void clearNotesUtauJieSplit(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1132](../juce/src/ProjectModel.h#L1132) |
| `void setTrackPitchAlgorithm(const juce::String& trackId, PitchAlgorithm algorithm)` | 业务／适配 | [ProjectModel.h:1133](../juce/src/ProjectModel.h#L1133) |
| `void setTrackStretchAlgorithm(const juce::String& trackId, StretchAlgorithm algorithm)` | 业务／适配 | [ProjectModel.h:1134](../juce/src/ProjectModel.h#L1134) |
| `void setPitchAlgorithm(PitchAlgorithm algorithm)` | 业务／适配 | [ProjectModel.h:1135](../juce/src/ProjectModel.h#L1135) |
| `void setStretchAlgorithm(StretchAlgorithm algorithm)` | 业务／适配 | [ProjectModel.h:1136](../juce/src/ProjectModel.h#L1136) |
| `void setTrackRenderOrder(const juce::String& trackId, RenderOrder order, std::optional<bool> mergeContiguous = {})` | 业务／适配 | [ProjectModel.h:1137](../juce/src/ProjectModel.h#L1137) |
| `void setRenderOrder(RenderOrder order)` | 业务／适配 | [ProjectModel.h:1139](../juce/src/ProjectModel.h#L1139) |
| `void moveClip(const juce::String& clipId, double startSeconds)` | 业务／适配 | [ProjectModel.h:1140](../juce/src/ProjectModel.h#L1140) |
| `bool canSplitClip(const juce::String& clipId, double absoluteSeconds) const` | 业务／适配 | [ProjectModel.h:1141](../juce/src/ProjectModel.h#L1141) |
| `juce::String splitClip(const juce::String& clipId, double absoluteSeconds)` | 业务／适配 | [ProjectModel.h:1144](../juce/src/ProjectModel.h#L1144) |
| `juce::String duplicateClip(const juce::String& clipId, double startSeconds = -1.0, const juce::String& targetTrackId = {})` | 业务／适配 | [ProjectModel.h:1145](../juce/src/ProjectModel.h#L1145) |
| `void resizeClip(const juce::String& clipId, double startSeconds, double durationSeconds)` | 业务／适配 | [ProjectModel.h:1148](../juce/src/ProjectModel.h#L1148) |
| `bool trimClip(const juce::String& clipId, double startSeconds, double durationSeconds)` | 业务／适配 | [ProjectModel.h:1150](../juce/src/ProjectModel.h#L1150) |
| `bool setClipEditableLength(const juce::String& clipId, double durationSeconds)` | 业务／适配 | [ProjectModel.h:1151](../juce/src/ProjectModel.h#L1151) |
| `void setClipGain(const juce::String& clipId, float gain)` | 业务／适配 | [ProjectModel.h:1152](../juce/src/ProjectModel.h#L1152) |
| `void setClipFades(const juce::String& clipId, double fadeInSeconds, double fadeOutSeconds)` | 业务／适配 | [ProjectModel.h:1153](../juce/src/ProjectModel.h#L1153) |
| `void setClipMuted(const juce::String& clipId, bool muted)` | 业务／适配 | [ProjectModel.h:1155](../juce/src/ProjectModel.h#L1155) |
| `void setClipNoteHints(const juce::String& clipId, bool enabled)` | 业务／适配 | [ProjectModel.h:1156](../juce/src/ProjectModel.h#L1156) |
| `void setClipNormalDisplay(const juce::String& clipId, bool enabled)` | 业务／适配 | [ProjectModel.h:1157](../juce/src/ProjectModel.h#L1157) |
| `void removeClip(const juce::String& clipId)` | 业务／适配 | [ProjectModel.h:1158](../juce/src/ProjectModel.h#L1158) |
| `void removeClips(const std::vector<juce::String>& clipIds)` | 业务／适配 | [ProjectModel.h:1159](../juce/src/ProjectModel.h#L1159) |
| `bool canMergeClips(const std::vector<juce::String>& clipIds) const` | 业务／适配 | [ProjectModel.h:1160](../juce/src/ProjectModel.h#L1160) |
| `juce::String mergeClips(const std::vector<juce::String>& clipIds)` | 业务／适配 | [ProjectModel.h:1161](../juce/src/ProjectModel.h#L1161) |
| `void moveClips(const std::vector<juce::String>& clipIds, double deltaSeconds, int trackDelta = 0)` | 业务／适配 | [ProjectModel.h:1162](../juce/src/ProjectModel.h#L1162) |
| `void removeTrack(const juce::String& trackId)` | 业务／适配 | [ProjectModel.h:1163](../juce/src/ProjectModel.h#L1163) |
| `bool moveTrackBefore(const juce::String& trackId, const juce::String& beforeTrackId)` | 业务／适配 | [ProjectModel.h:1165](../juce/src/ProjectModel.h#L1165) |
| `void transposeNote(const juce::String& noteId, float semitones)` | 业务／适配 | [ProjectModel.h:1166](../juce/src/ProjectModel.h#L1166) |
| `void transposeNotes(const std::vector<juce::String>& noteIds, float semitones)` | 业务／适配 | [ProjectModel.h:1167](../juce/src/ProjectModel.h#L1167) |
| `bool moveUtauNotes(const std::vector<juce::String>& noteIds, double deltaSeconds, float semitones, juce::String* error = nullptr)` | 业务／适配 | [ProjectModel.h:1171](../juce/src/ProjectModel.h#L1171) |
| `bool moveNativeNotes(const std::vector<juce::String>& noteIds, double deltaSeconds, float semitones, juce::String* error = nullptr)` | 业务／适配 | [ProjectModel.h:1175](../juce/src/ProjectModel.h#L1175) |
| `bool trimNativeNoteEdge(const juce::String& noteId, double deltaSeconds, bool leftEdge)` | 业务／适配 | [ProjectModel.h:1177](../juce/src/ProjectModel.h#L1177) |
| `bool resizeNativeNoteEdge(const juce::String& noteId, double deltaSeconds, bool leftEdge)` | 业务／适配 | [ProjectModel.h:1178](../juce/src/ProjectModel.h#L1178) |
| `bool alignNotePreutteranceToGrid(const juce::String& noteId, double quarterStep, juce::String& error)` | 业务／适配 | [ProjectModel.h:1181](../juce/src/ProjectModel.h#L1181) |
| `void setNotesMidi(const std::vector<juce::String>& noteIds, float midiNote)` | 业务／适配 | [ProjectModel.h:1182](../juce/src/ProjectModel.h#L1182) |
| `void averageNotesMidi(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1183](../juce/src/ProjectModel.h#L1183) |
| `void quantizeNotesMidi(const std::vector<juce::String>& noteIds, float stepSemitones = 1.0f)` | 业务／适配 | [ProjectModel.h:1184](../juce/src/ProjectModel.h#L1184) |
| `std::vector<juce::String> notesOverlapping( const juce::String& clipId, double fromSeconds, double toSeconds) const` | 业务／适配 | [ProjectModel.h:1188](../juce/src/ProjectModel.h#L1188) |
| `std::vector<juce::String> insertNotes( const juce::String& clipId, const std::vector<NoteData>& noteTemplates, double absoluteStartSeconds)` | 业务／适配 | [ProjectModel.h:1190](../juce/src/ProjectModel.h#L1190) |
| `std::vector<juce::String> insertNativeAudioClips( const juce::String& trackId, const std::vector<ClipData>& templates, const std::vector<NativeConnection>& connections, double atSeconds)` | 业务／适配 | [ProjectModel.h:1193](../juce/src/ProjectModel.h#L1193) |
| `std::vector<juce::String> duplicateNotes( const std::vector<juce::String>& noteIds, const juce::String& targetClipId, double absoluteStartSeconds)` | 业务／适配 | [ProjectModel.h:1196](../juce/src/ProjectModel.h#L1196) |
| `void resizeNote(const juce::String& noteId, double newStart, double newDuration)` | 业务／适配 | [ProjectModel.h:1199](../juce/src/ProjectModel.h#L1199) |
| `juce::String splitNote(const juce::String& noteId, double localSeconds)` | 业务／适配 | [ProjectModel.h:1200](../juce/src/ProjectModel.h#L1200) |
| `juce::String mergeNotes(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1202](../juce/src/ProjectModel.h#L1202) |
| `bool canJoinNativeNotes(const juce::String&,const juce::String&) const` | 业务／适配 | [ProjectModel.h:1203](../juce/src/ProjectModel.h#L1203) |
| `juce::String joinNativeNotes(const juce::String&,const juce::String&)` | 业务／适配 | [ProjectModel.h:1204](../juce/src/ProjectModel.h#L1204) |
| `void setNotesConnection(const std::vector<juce::String>& noteIds, bool enabled)` | 业务／适配 | [ProjectModel.h:1207](../juce/src/ProjectModel.h#L1207) |
| `bool setNotesPortamento(const std::vector<juce::String>& noteIds, bool enabled, juce::String& error, double durationSeconds = 0.1)` | 业务／适配 | [ProjectModel.h:1210](../juce/src/ProjectModel.h#L1210) |
| `bool setConnectionPitchCurve(const juce::String& connectionId, std::vector<PitchCurveEditPoint> absolutePoints, juce::String& error)` | 业务／适配 | [ProjectModel.h:1216](../juce/src/ProjectModel.h#L1216) |
| `bool setConnectionPitchPoint(const juce::String& connectionId, int pointIndex, PitchCurveEditPoint absolutePoint, juce::String& error)` | 业务／适配 | [ProjectModel.h:1218](../juce/src/ProjectModel.h#L1218) |
| `bool disconnectNativeAudio(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1220](../juce/src/ProjectModel.h#L1220) |
| `bool linkNativeAudio(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1221](../juce/src/ProjectModel.h#L1221) |
| `bool setNativeAudioOverlap(const juce::String& trackId, bool enabled)` | 业务／适配 | [ProjectModel.h:1222](../juce/src/ProjectModel.h#L1222) |
| `juce::String insertPrefixNote(const juce::String& noteId, double targetOverlapSeconds)` | 业务／适配 | [ProjectModel.h:1227](../juce/src/ProjectModel.h#L1227) |
| `void setNoteModulation(const juce::String& noteId, float modulation)` | 业务／适配 | [ProjectModel.h:1229](../juce/src/ProjectModel.h#L1229) |
| `void setNoteDrift(const juce::String& noteId, float drift)` | 业务／适配 | [ProjectModel.h:1230](../juce/src/ProjectModel.h#L1230) |
| `void setNoteTension(const juce::String& noteId, float tension)` | 业务／适配 | [ProjectModel.h:1231](../juce/src/ProjectModel.h#L1231) |
| `void setNoteBreath(const juce::String& noteId, float breath)` | 业务／适配 | [ProjectModel.h:1232](../juce/src/ProjectModel.h#L1232) |
| `void setNoteFormant(const juce::String& noteId, float semitones)` | 业务／适配 | [ProjectModel.h:1233](../juce/src/ProjectModel.h#L1233) |
| `void setNoteGain(const juce::String& noteId, float gain)` | 业务／适配 | [ProjectModel.h:1234](../juce/src/ProjectModel.h#L1234) |
| `bool setNotesTailFade(const std::vector<juce::String>& noteIds, int mode)` | 业务／适配 | [ProjectModel.h:1235](../juce/src/ProjectModel.h#L1235) |
| `bool setNotesNativeEnvelope(const std::vector<juce::String>& noteIds, const backend::NativeEnvelopeSettings& settings)` | 业务／适配 | [ProjectModel.h:1236](../juce/src/ProjectModel.h#L1236) |
| `bool setNotesTailFade(const std::vector<juce::String>& noteIds, int mode, const backend::TailFadeSettings& settings)` | 业务／适配 | [ProjectModel.h:1237](../juce/src/ProjectModel.h#L1237) |
| `void setNotesAmplitudeEnvelopeBase(const std::vector<juce::String>& noteIds, float basePercent)` | 业务／适配 | [ProjectModel.h:1239](../juce/src/ProjectModel.h#L1239) |
| `void setNoteAttack(const juce::String& noteId, double consonantSeconds, float attackSpeed)` | 业务／适配 | [ProjectModel.h:1241](../juce/src/ProjectModel.h#L1241) |
| `void setNoteAttackSpeed(const juce::String& noteId, float attackSpeed)` | 业务／适配 | [ProjectModel.h:1242](../juce/src/ProjectModel.h#L1242) |
| `void setNoteLabel(const juce::String& noteId, const juce::String& label)` | 业务／适配 | [ProjectModel.h:1243](../juce/src/ProjectModel.h#L1243) |
| `int convertTrackLyricsToPinyin(const juce::String& trackId)` | 业务／适配 | [ProjectModel.h:1247](../juce/src/ProjectModel.h#L1247) |
| `void setNoteLabels(const std::vector<std::pair<juce::String, juce::String>>& labels)` | 业务／适配 | [ProjectModel.h:1251](../juce/src/ProjectModel.h#L1251) |
| `void setNoteUtauFlags(const juce::String& noteId, const juce::String& flags)` | 业务／适配 | [ProjectModel.h:1255](../juce/src/ProjectModel.h#L1255) |
| `void setNotesUtauFlags(const std::vector<juce::String>& noteIds, const juce::String& flags)` | 业务／适配 | [ProjectModel.h:1256](../juce/src/ProjectModel.h#L1256) |
| `void setNotesUtauConsonantVelocity(const std::vector<juce::String>& noteIds, int velocity)` | 业务／适配 | [ProjectModel.h:1258](../juce/src/ProjectModel.h#L1258) |
| `void setNoteUtauOto(const juce::String& noteId, const backend::UtauOtoOverride& oto)` | 业务／适配 | [ProjectModel.h:1285](../juce/src/ProjectModel.h#L1285) |
| `bool setNoteUtauOtoChecked(const juce::String& noteId, std::uint64_t expectedRevision, const backend::UtauOtoOverride& value, std::uint64_t& resultingRevision, juce::String& error)` | 业务／适配 | [ProjectModel.h:1288](../juce/src/ProjectModel.h#L1288) |
| `void clearNotesUtauOto(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1293](../juce/src/ProjectModel.h#L1293) |
| `void setNotesUtauStp(const std::vector<juce::String>& noteIds, double seconds)` | 业务／适配 | [ProjectModel.h:1295](../juce/src/ProjectModel.h#L1295) |
| `void setNoteUtauTimingOverrides(const juce::String& noteId, bool enabled, double preutteranceSeconds, double overlapSeconds)` | 业务／适配 | [ProjectModel.h:1296](../juce/src/ProjectModel.h#L1296) |
| `void setNoteRobustPitchCurve(const juce::String& noteId, bool enabled)` | 业务／适配 | [ProjectModel.h:1299](../juce/src/ProjectModel.h#L1299) |
| `bool rememberDiffSingerParameters(const std::map<juce::String, std::vector<FlagCurve>>& curves)` | 业务／适配 | [ProjectModel.h:1300](../juce/src/ProjectModel.h#L1300) |
| `bool applyDiffSingerParameters(std::uint64_t expectedRevision, const std::map<juce::String, std::vector<FlagCurve>>& curves)` | 业务／适配 | [ProjectModel.h:1301](../juce/src/ProjectModel.h#L1301) |
| `bool setDiffSingerParameterCurve(const juce::String& id, const juce::String& key, std::vector<FlagCurvePoint> points)` | 业务／适配 | [ProjectModel.h:1303](../juce/src/ProjectModel.h#L1303) |
| `bool setDiffSingerPitchOffset(const juce::String& noteId, std::vector<PitchCurveEditPoint> points)` | 业务／适配 | [ProjectModel.h:1305](../juce/src/ProjectModel.h#L1305) |
| `bool setNotePitchCurve(const juce::String& noteId, std::vector<PitchCurveEditPoint> points, bool storeControlPoints = false, bool authorPitchSupport = false)` | 业务／适配 | [ProjectModel.h:1307](../juce/src/ProjectModel.h#L1307) |
| `bool applyDiffSingerTiming(std::uint64_t expectedRevision, const std::vector<std::pair<juce::String, juce::String>>& timings)` | 业务／适配 | [ProjectModel.h:1310](../juce/src/ProjectModel.h#L1310) |
| `bool applyDiffSingerPronunciation(std::uint64_t expectedRevision, const juce::String& trackId, const std::vector<std::pair<juce::String, juce::String>>& readings, const juce::String& dictionary)` | 业务／适配 | [ProjectModel.h:1312](../juce/src/ProjectModel.h#L1312) |
| `bool restoreDiffSingerPitchPoint(std::uint64_t expectedRevision, const juce::String& noteId, const std::vector<PitchCurveEditPoint>& displayedPoints, int index)` | 业务／适配 | [ProjectModel.h:1314](../juce/src/ProjectModel.h#L1314) |
| `bool applyDiffSingerPitch(std::uint64_t expectedRevision, const std::vector<std::pair<juce::String, std::vector<PitchCurveEditPoint>>>& curves, bool preserveOutsideNoteHandles = false)` | 业务／适配 | [ProjectModel.h:1316](../juce/src/ProjectModel.h#L1316) |
| `void setDiffSingerOptions(const juce::String& trackId, const juce::String& language, const juce::String& speaker)` | 业务／适配 | [ProjectModel.h:1319](../juce/src/ProjectModel.h#L1319) |
| `bool setNoteAmplitudeEnvelope(const juce::String& noteId, std::vector<AmplitudeEnvelopePoint> points)` | 业务／适配 | [ProjectModel.h:1321](../juce/src/ProjectModel.h#L1321) |
| `int crossfadeNotes(std::vector<CrossfadeSpan> spans, std::uint64_t expectedRevision, bool curved = false)` | 业务／适配 | [ProjectModel.h:1333](../juce/src/ProjectModel.h#L1333) |
| `bool rememberCvvcPhonemes(const std::vector<std::pair<juce::String,juce::String>>& caches, std::uint64_t expectedRevision)` | 业务／适配 | [ProjectModel.h:1334](../juce/src/ProjectModel.h#L1334) |
| `bool setNotesAmplitudeEnvelopes( std::vector<std::pair<juce::String, std::vector<AmplitudeEnvelopePoint>>> envelopes)` | 业务／适配 | [ProjectModel.h:1336](../juce/src/ProjectModel.h#L1336) |
| `static PlannedNote plannedNoteFor(double startSeconds, double requestedSeconds, double clipSeconds, const std::vector<NoteSpan>& occupied)` | 业务／适配 | [ProjectModel.h:1362](../juce/src/ProjectModel.h#L1362) |
| `juce::String addClip(const juce::String& trackId, double startSeconds, double durationSeconds)` | 业务／适配 | [ProjectModel.h:1370](../juce/src/ProjectModel.h#L1370) |
| `bool canAddEmptyTuningClip(const juce::String& trackId, double startSeconds) const` | 业务／适配 | [ProjectModel.h:1372](../juce/src/ProjectModel.h#L1372) |
| `juce::String addEmptyTuningClip(const juce::String& trackId, double startSeconds, double durationSeconds = 10.0)` | 业务／适配 | [ProjectModel.h:1374](../juce/src/ProjectModel.h#L1374) |
| `static std::vector<std::size_t> regionsToImport( const std::vector<RegionSpan>& regions)` | 业务／适配 | [ProjectModel.h:1391](../juce/src/ProjectModel.h#L1391) |
| `void flattenNotePitch(const std::vector<juce::String>& noteIds, bool snapNativeToSemitone = false)` | 业务／适配 | [ProjectModel.h:1407](../juce/src/ProjectModel.h#L1407) |
| `bool restoreNativeSourcePitch(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1408](../juce/src/ProjectModel.h#L1408) |
| `static double firstFreeStartFrom(double startSeconds, const std::vector<NoteSpan>& occupied)` | 业务／适配 | [ProjectModel.h:1409](../juce/src/ProjectModel.h#L1409) |
| `juce::String addNote(const juce::String& preferredClipId, double absoluteStart, double duration, float midiNote)` | 业务／适配 | [ProjectModel.h:1412](../juce/src/ProjectModel.h#L1412) |
| `juce::String addNoteFrom(const juce::String& preferredClipId, double absoluteFromSeconds, double maximumDuration, float midiNote)` | 业务／适配 | [ProjectModel.h:1420](../juce/src/ProjectModel.h#L1420) |
| `void removeNote(const juce::String& noteId)` | 业务／适配 | [ProjectModel.h:1423](../juce/src/ProjectModel.h#L1423) |
| `void removeNotes(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1424](../juce/src/ProjectModel.h#L1424) |
| `void removeNotesRippling(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1428](../juce/src/ProjectModel.h#L1428) |
| `void insertGapBeforeNote(const juce::String& noteId, double seconds)` | 业务／适配 | [ProjectModel.h:1432](../juce/src/ProjectModel.h#L1432) |
| `void closeGapBeforeNote(const juce::String& noteId, double seconds)` | 业务／适配 | [ProjectModel.h:1435](../juce/src/ProjectModel.h#L1435) |
| `void toggleNoteConnection(const juce::String& noteId)` | 业务／适配 | [ProjectModel.h:1436](../juce/src/ProjectModel.h#L1436) |
| `MaterialUpdateReport applySourceSettingsWithReport( const juce::File& source, const std::vector<SampleRegionSetting>& rows, bool apply = true)` | 业务／适配 | [ProjectModel.h:1457](../juce/src/ProjectModel.h#L1457) |
| `void applySourceSettings(const juce::File& source, const std::vector<SampleRegionSetting>& rows)` | 业务／适配 | [ProjectModel.h:1460](../juce/src/ProjectModel.h#L1460) |
| `void setNoteMaterialLocalOverride(const juce::String& noteId, bool enabled)` | 业务／适配 | [ProjectModel.h:1462](../juce/src/ProjectModel.h#L1462) |
| `bool getNoteMaterialLocalBinding(const juce::String& noteId, NoteMaterialLocalBinding&, SampleRegionSetting& row, juce::String& error) const` | 业务／适配 | [ProjectModel.h:1474](../juce/src/ProjectModel.h#L1474) |
| `bool applyNoteMaterialLocalRow(const NoteMaterialLocalBinding&, const SampleRegionSetting&, std::uint64_t& resultingRevision, juce::String& error)` | 业务／适配 | [ProjectModel.h:1476](../juce/src/ProjectModel.h#L1476) |
| `bool restoreNoteMaterialLocalRow(const NoteMaterialLocalBinding&, SampleRegionSetting& restored, std::uint64_t& resultingRevision, juce::String& error)` | 业务／适配 | [ProjectModel.h:1478](../juce/src/ProjectModel.h#L1478) |
| `static bool makeClassicNoteMaterialOto(const SampleRegionSetting&, backend::UtauOtoOverride&, juce::String& error)` | 业务／适配 | [ProjectModel.h:1481](../juce/src/ProjectModel.h#L1481) |
| `MaterialSourceReplacementReport replaceNoteMaterialSource( const juce::String& noteId, const juce::File& audio, const std::vector<SampleRegionSetting>& completeRows, const juce::String& audioId, const juce::String& regionId, const juce::String& chosenAlias, juce::String& error, bool apply = true)` | 业务／适配 | [ProjectModel.h:1501](../juce/src/ProjectModel.h#L1501) |
| `std::vector<MaterialSourceBindingTicket> pendingMaterialSourceAnalyses() const` | 业务／适配 | [ProjectModel.h:1506](../juce/src/ProjectModel.h#L1506) |
| `bool completeMaterialSourceAnalysis(const MaterialSourceBindingTicket& ticket, const std::vector<NoteData>& fullFileSourceNotes, juce::String& error)` | 业务／适配 | [ProjectModel.h:1511](../juce/src/ProjectModel.h#L1511) |
| `bool save(const juce::File& file, juce::String& error) const` | 业务／适配 | [ProjectModel.h:1514](../juce/src/ProjectModel.h#L1514) |
| `bool load(const juce::File& file, juce::String& error)` | 业务／适配 | [ProjectModel.h:1515](../juce/src/ProjectModel.h#L1515) |
| `static bool saveRecoverySnapshot(ProjectData data, const juce::File& recoveryFile, const juce::File& originalFile, juce::String& error)` | 业务／适配 | [ProjectModel.h:1516](../juce/src/ProjectModel.h#L1516) |

## AudioEngine（73 个声明）

| 实际签名 | 类型 | 定义入口 |
| --- | --- | --- |
| `explicit AudioEngine(bool initialiseOutputDevice = true)` | 生命周期 | [AudioEngine.h:59](../juce/src/AudioEngine.h#L59) |
| `~AudioEngine() override` | 生命周期 | [AudioEngine.h:60](../juce/src/AudioEngine.h#L60) |
| `void prepareToPlay(int samplesPerBlockExpected, double sampleRate) override` | 业务／适配 | [AudioEngine.h:62](../juce/src/AudioEngine.h#L62) |
| `void releaseResources() override` | 业务／适配 | [AudioEngine.h:63](../juce/src/AudioEngine.h#L63) |
| `void getNextAudioBlock(const juce::AudioSourceChannelInfo& bufferToFill) override` | 业务／适配 | [AudioEngine.h:64](../juce/src/AudioEngine.h#L64) |
| `void syncProject(const ProjectData& project, bool forExport = false, WavExportComponent component = WavExportComponent::full)` | 业务／适配 | [AudioEngine.h:67](../juce/src/AudioEngine.h#L67) |
| `static juce::String componentExportIssue(const ProjectData& project, const juce::File& engine, const juce::String& trackId = {}, const std::vector<juce::String>& noteIds = {}, juce::Range<double> range = {}, const juce::String& auditionTrack = {}, UtauOutputEngine defaultEngine = UtauOutputEngine::resampler)` | 业务／适配 | [AudioEngine.h:69](../juce/src/AudioEngine.h#L69) |
| `void setPlayUntil(double seconds)` | 业务／适配 | [AudioEngine.h:76](../juce/src/AudioEngine.h#L76) |
| `void setUtauRenderNoteSelection(const std::vector<juce::String>& noteIds)` | 业务／适配 | [AudioEngine.h:77](../juce/src/AudioEngine.h#L77) |
| `void setAuditionTrack(const juce::String& trackId)` | 业务／适配 | [AudioEngine.h:81](../juce/src/AudioEngine.h#L81) |
| `static bool trackIsAudible(bool muted, bool solo, bool anySolo, bool referenceOnly, bool beingWorkedOn)` | 业务／适配 | [AudioEngine.h:84](../juce/src/AudioEngine.h#L84) |
| `int selectEveryUtauNote(const ProjectData& project)` | 业务／适配 | [AudioEngine.h:90](../juce/src/AudioEngine.h#L90) |
| `bool selectAllRenderedUtauNotes(const ProjectData& project)` | 业务／适配 | [AudioEngine.h:91](../juce/src/AudioEngine.h#L91) |
| `void setHifiganModelDirectory(const juce::File& directory)` | 业务／适配 | [AudioEngine.h:92](../juce/src/AudioEngine.h#L92) |
| `void setUtauResamplerFile(const juce::File& executable)` | 业务／适配 | [AudioEngine.h:97](../juce/src/AudioEngine.h#L97) |
| `void setUtauOutputDefaults(UtauOutputEngine engine, const juce::File& wavtool)` | 业务／适配 | [AudioEngine.h:98](../juce/src/AudioEngine.h#L98) |
| `UtauOutputEngine currentUtauOutputEngine() const` | 业务／适配 | [AudioEngine.h:99](../juce/src/AudioEngine.h#L99) |
| `juce::File currentUtauWavtoolFile() const` | 业务／适配 | [AudioEngine.h:100](../juce/src/AudioEngine.h#L100) |
| `juce::String outputEngineDisplayName(const TrackData& track) const` | 业务／适配 | [AudioEngine.h:101](../juce/src/AudioEngine.h#L101) |
| `static juce::File bundledUtauResampler( const juce::File& executableDirectory)` | 业务／适配 | [AudioEngine.h:106](../juce/src/AudioEngine.h#L106) |
| `static juce::File resolveUtauResampler( const juce::String& configured, const juce::File& executableDirectory)` | 业务／适配 | [AudioEngine.h:109](../juce/src/AudioEngine.h#L109) |
| `juce::File diagnosticUtauResamplerFile() const` | 诊断 | [AudioEngine.h:116](../juce/src/AudioEngine.h#L116) |
| `juce::File currentUtauResamplerFile() const` | 业务／适配 | [AudioEngine.h:120](../juce/src/AudioEngine.h#L120) |
| `void setInferenceConfiguration(backend::InferenceBackend inference, int deviceIndex)` | 业务／适配 | [AudioEngine.h:124](../juce/src/AudioEngine.h#L124) |
| `std::optional<double> probeDuration(const juce::File& file)` | 业务／适配 | [AudioEngine.h:125](../juce/src/AudioEngine.h#L125) |
| `bool setAuditionFile(const juce::File& file)` | 业务／适配 | [AudioEngine.h:126](../juce/src/AudioEngine.h#L126) |
| `void clearAuditionFile()` | 业务／适配 | [AudioEngine.h:127](../juce/src/AudioEngine.h#L127) |
| `void play()` | 业务／适配 | [AudioEngine.h:129](../juce/src/AudioEngine.h#L129) |
| `void stop()` | 业务／适配 | [AudioEngine.h:130](../juce/src/AudioEngine.h#L130) |
| `void setPosition(double seconds)` | 业务／适配 | [AudioEngine.h:131](../juce/src/AudioEngine.h#L131) |
| `double position() const` | 业务／适配 | [AudioEngine.h:132](../juce/src/AudioEngine.h#L132) |
| `float trackPeak(const juce::String& trackId) const` | 业务／适配 | [AudioEngine.h:133](../juce/src/AudioEngine.h#L133) |
| `std::optional<double> renderProgress() const` | 业务／适配 | [AudioEngine.h:134](../juce/src/AudioEngine.h#L134) |
| `bool playbackNeedsRender(double lookAheadSeconds = 0.75) const` | 业务／适配 | [AudioEngine.h:137](../juce/src/AudioEngine.h#L137) |
| `bool hasPlayableRenderedAudio() const` | 业务／适配 | [AudioEngine.h:138](../juce/src/AudioEngine.h#L138) |
| `bool hasCurrentRenderedAudio() const` | 业务／适配 | [AudioEngine.h:139](../juce/src/AudioEngine.h#L139) |
| `bool rewindToFirstPlayableRenderedAudio(double leadInSeconds = 0.03)` | 业务／适配 | [AudioEngine.h:140](../juce/src/AudioEngine.h#L140) |
| `juce::String activeRenderBackends() const` | 业务／适配 | [AudioEngine.h:141](../juce/src/AudioEngine.h#L141) |
| `juce::String activeRenderWarnings() const` | 业务／适配 | [AudioEngine.h:142](../juce/src/AudioEngine.h#L142) |
| `static juce::StringArray renderCapabilityWarnings( const ProjectData& project, UtauOutputEngine defaultEngine = UtauOutputEngine::resampler)` | 业务／适配 | [AudioEngine.h:148](../juce/src/AudioEngine.h#L148) |
| `static bool clipsJoinAt(const ClipData& clip, const ClipData& neighbour, bool asNext)` | 业务／适配 | [AudioEngine.h:160](../juce/src/AudioEngine.h#L160) |
| `static std::vector<std::vector<const ClipData*>> glideChains( const std::vector<const ClipData*>& orderedClips)` | 业务／适配 | [AudioEngine.h:168](../juce/src/AudioEngine.h#L168) |
| `static backend::Mld5FileRenderRequest mergedRequestFor( const std::vector<const ClipData*>& group, const TrackData& track, const juce::File& hifiganModelDirectory, const backend::OrtExecutionConfig& inference, std::optional<bool> legacyUvCarrier = {}, bool skipSharedPitch = false, const SharedPitchLines* sharedLines = nullptr)` | 业务／适配 | [AudioEngine.h:174](../juce/src/AudioEngine.h#L174) |
| `void setDiagnosticNsfRendering(bool incrementalRender, bool legacyUvCarrier = false, bool skipSharedPitch = false)` | 业务／适配 | [AudioEngine.h:184](../juce/src/AudioEngine.h#L184) |
| `void setDiagnosticUtauTimelinePitch(bool enabled)` | 业务／适配 | [AudioEngine.h:194](../juce/src/AudioEngine.h#L194) |
| `void setDiagnosticUtauOwnPitch(bool enabled)` | 业务／适配 | [AudioEngine.h:195](../juce/src/AudioEngine.h#L195) |
| `void setDiagnosticUtauNominalBridge(bool enabled)` | 业务／适配 | [AudioEngine.h:196](../juce/src/AudioEngine.h#L196) |
| `int diagnosticMergedPhraseCount() const` | 诊断 | [AudioEngine.h:199](../juce/src/AudioEngine.h#L199) |
| `std::uint64_t diagnosticScheduledNsfUtauRenders() const` | 诊断 | [AudioEngine.h:200](../juce/src/AudioEngine.h#L200) |
| `double diagnosticRenderPriorityPosition() const` | 诊断 | [AudioEngine.h:201](../juce/src/AudioEngine.h#L201) |
| `static std::vector<ProjectedNotePitch> projectTrackNotePitch( const TrackData&, const std::vector<NativeConnection>&)` | 业务／适配 | [AudioEngine.h:217](../juce/src/AudioEngine.h#L217) |
| `static std::vector<float> diagnosticNativeTargetMidi( const ProjectData& project, int trackIndex, int clipIndex)` | 诊断 | [AudioEngine.h:223](../juce/src/AudioEngine.h#L223) |
| `static backend::Mld5FileRenderRequest diagnosticNativeRequest(const ClipData& clip,const TrackData& track)` | 诊断 | [AudioEngine.h:225](../juce/src/AudioEngine.h#L225) |
| `std::shared_ptr<const std::vector<UtauNoteWaveform>> utauNoteWaveforms() const` | 业务／适配 | [AudioEngine.h:231](../juce/src/AudioEngine.h#L231) |
| `void refreshUtauWaveforms()` | 业务／适配 | [AudioEngine.h:237](../juce/src/AudioEngine.h#L237) |
| `void refreshNativeWaveforms()` | 业务／适配 | [AudioEngine.h:238](../juce/src/AudioEngine.h#L238) |
| `std::shared_ptr<const std::vector<NativeRenderedWaveform>> nativeClipWaveforms() const` | 业务／适配 | [AudioEngine.h:239](../juce/src/AudioEngine.h#L239) |
| `static std::uint64_t nativeClipWaveformHash(const ClipData&, const TrackData&)` | 业务／适配 | [AudioEngine.h:240](../juce/src/AudioEngine.h#L240) |
| `static std::uint64_t utauNoteRenderHash(const NoteData& note)` | 业务／适配 | [AudioEngine.h:246](../juce/src/AudioEngine.h#L246) |
| `static std::uint64_t utauNoteAudioHash(const NoteData& note)` | 业务／适配 | [AudioEngine.h:249](../juce/src/AudioEngine.h#L249) |
| `static std::uint64_t utauNotePhonemeHash(const NoteData& note)` | 业务／适配 | [AudioEngine.h:250](../juce/src/AudioEngine.h#L250) |
| `static std::uint64_t utauPhonemeMixContextHash(const TrackData& track)` | 业务／适配 | [AudioEngine.h:251](../juce/src/AudioEngine.h#L251) |
| `static std::vector<backend::UtauNoteRenderSpec> diagnosticUtauRequestNotes( const ProjectData& project, const juce::String& clipId, const std::vector<juce::String>& selection = {})` | 诊断 | [AudioEngine.h:255](../juce/src/AudioEngine.h#L255) |
| `static std::vector<backend::UtauNoteRenderSpec> utauRequestNotesForClip( const ProjectData& project, const juce::String& clipId)` | 业务／适配 | [AudioEngine.h:259](../juce/src/AudioEngine.h#L259) |
| `static backend::RequestedVoicebankFamily diagnosticUtauRequestedFamily( const ProjectData& project, const juce::String& clipId)` | 诊断 | [AudioEngine.h:261](../juce/src/AudioEngine.h#L261) |
| `static std::string diagnosticUtauRenderKey( const ProjectData& project, const juce::String& clipId, UtauOutputEngine defaultEngine = UtauOutputEngine::resampler, const juce::File& resampler = {}, const juce::File& wavtool = {})` | 诊断 | [AudioEngine.h:263](../juce/src/AudioEngine.h#L263) |
| `bool exportWav(const juce::File& file, juce::String& error, const juce::String& trackId = {}, double fromSeconds = 0.0, double toSeconds = 0.0, WavExportOptions options = {})` | 业务／适配 | [AudioEngine.h:268](../juce/src/AudioEngine.h#L268) |
| `double playbackSampleRate() const` | 业务／适配 | [AudioEngine.h:272](../juce/src/AudioEngine.h#L272) |
| `bool isPlaying() const` | 业务／适配 | [AudioEngine.h:273](../juce/src/AudioEngine.h#L273) |
| `juce::AudioDeviceManager& devices()` | 业务／适配 | [AudioEngine.h:274](../juce/src/AudioEngine.h#L274) |
| `bool ensureOutputDevice(juce::String& error)` | 业务／适配 | [AudioEngine.h:275](../juce/src/AudioEngine.h#L275) |
| `void restoreDeviceState(juce::PropertiesFile& properties)` | 业务／适配 | [AudioEngine.h:276](../juce/src/AudioEngine.h#L276) |
| `void saveDeviceState(juce::PropertiesFile& properties) const` | 业务／适配 | [AudioEngine.h:277](../juce/src/AudioEngine.h#L277) |

## SampleSettings（36 个声明）

| 实际签名 | 类型 | 定义入口 |
| --- | --- | --- |
| `static juce::File sidecarFor(const juce::File& audio)` | 业务／适配 | [SampleSettings.h:93](../juce/src/SampleSettings.h#L93) |
| `static juce::File annotationFor(const juce::File& audio)` | 业务／适配 | [SampleSettings.h:95](../juce/src/SampleSettings.h#L95) |
| `static void setMaterialRoots(const juce::StringArray& roots)` | 业务／适配 | [SampleSettings.h:96](../juce/src/SampleSettings.h#L96) |
| `static juce::StringArray materialRoots()` | 业务／适配 | [SampleSettings.h:97](../juce/src/SampleSettings.h#L97) |
| `static bool hasNativeAnnotation(const juce::File& audio, juce::String* error = nullptr)` | 业务／适配 | [SampleSettings.h:98](../juce/src/SampleSettings.h#L98) |
| `static bool copyAnnotations(const juce::File& source, const juce::File& destination, juce::String& error)` | 业务／适配 | [SampleSettings.h:100](../juce/src/SampleSettings.h#L100) |
| `static std::vector<NativeSegment> nativeSegmentsFor( const SampleRegionSetting& row)` | 业务／适配 | [SampleSettings.h:105](../juce/src/SampleSettings.h#L105) |
| `static std::vector<SampleRegionSetting> loadOrDerive(const juce::File& audio, const ProjectData& project, juce::String* error = nullptr, bool explicitLegacyMigration = false, const juce::String& audioId = {})` | 业务／适配 | [SampleSettings.h:109](../juce/src/SampleSettings.h#L109) |
| `static bool save(const juce::File& audio, const std::vector<SampleRegionSetting>& rows, juce::String& error, const juce::String& audioId = {})` | 业务／适配 | [SampleSettings.h:114](../juce/src/SampleSettings.h#L114) |
| `static bool convertMelodyneProject(ProjectData& project, juce::StringArray& warnings)` | 业务／适配 | [SampleSettings.h:119](../juce/src/SampleSettings.h#L119) |
| `static bool importOto(const juce::File& oto, const juce::File& audio, double audioDuration, std::vector<SampleRegionSetting>& rows, juce::String& error)` | 业务／适配 | [SampleSettings.h:120](../juce/src/SampleSettings.h#L120) |
| `static bool exportOto(const juce::File& oto, const juce::File& audio, const std::vector<SampleRegionSetting>& rows, double audioDuration, juce::String& error)` | 业务／适配 | [SampleSettings.h:123](../juce/src/SampleSettings.h#L123) |
| `static bool importVoicebank(const juce::File& root, juce::StringArray& audioFiles, int& sidecarsWritten, int& regionsWritten, juce::StringArray& warnings)` | 业务／适配 | [SampleSettings.h:126](../juce/src/SampleSettings.h#L126) |
| `static backend::UtauOtoOverride noteOtoFromEntry(const VoicebankOtoEntry& entry, bool withRegions)` | 业务／适配 | [SampleSettings.h:134](../juce/src/SampleSettings.h#L134) |
| `static VoicebankOtoEntry entryWithNoteOto(VoicebankOtoEntry entry, const backend::UtauOtoOverride& oto)` | 业务／适配 | [SampleSettings.h:151](../juce/src/SampleSettings.h#L151) |
| `static std::vector<VoicebankOtoEntry> loadVoicebankOto( const juce::File& root, juce::StringArray& warnings, bool jieMode = false, bool mouMode = false, const juce::File& onlyOtoFile = {})` | 业务／适配 | [SampleSettings.h:170](../juce/src/SampleSettings.h#L170) |
| `static std::vector<VoicebankOtoEntry> loadAudioOto(const juce::File& audio, const juce::File& root, UtauMode mode, juce::StringArray& warnings)` | 业务／适配 | [SampleSettings.h:175](../juce/src/SampleSettings.h#L175) |
| `static std::optional<ClipData> nativeAudioFromOto(const VoicebankOtoEntry& entry, UtauMode mode, double sourceDurationSeconds)` | 业务／适配 | [SampleSettings.h:178](../juce/src/SampleSettings.h#L178) |
| `static std::uint64_t voicebankFilesRevision()` | 业务／适配 | [SampleSettings.h:184](../juce/src/SampleSettings.h#L184) |
| `static bool updateVoicebankOtoEntry(const VoicebankOtoEntry& original, const VoicebankOtoEntry& updated, juce::String& error)` | 业务／适配 | [SampleSettings.h:185](../juce/src/SampleSettings.h#L185) |
| `static juce::File jieClassicOtoFileFor(const juce::File& otoFile)` | 业务／适配 | [SampleSettings.h:191](../juce/src/SampleSettings.h#L191) |
| `static bool updateJieVoicebankOtoEntry(const VoicebankOtoEntry& original, const VoicebankOtoEntry& updated, juce::String& error)` | 业务／适配 | [SampleSettings.h:192](../juce/src/SampleSettings.h#L192) |
| `static bool duplicateVoicebankOtoEntry(const VoicebankOtoEntry& source, const juce::String& newAlias, bool jieMode, juce::String& error)` | 业务／适配 | [SampleSettings.h:197](../juce/src/SampleSettings.h#L197) |
| `static juce::String entryLookupName(const VoicebankOtoEntry& entry)` | 业务／适配 | [SampleSettings.h:205](../juce/src/SampleSettings.h#L205) |
| `static int findEntryForAlias(const std::vector<VoicebankOtoEntry>& entries, const juce::String& alias)` | 业务／适配 | [SampleSettings.h:208](../juce/src/SampleSettings.h#L208) |
| `static juce::File jieOtoFileFor(const juce::File& otoFile)` | 业务／适配 | [SampleSettings.h:210](../juce/src/SampleSettings.h#L210) |
| `static void mergeJieOto(std::vector<VoicebankOtoEntry>& entries)` | 业务／适配 | [SampleSettings.h:212](../juce/src/SampleSettings.h#L212) |
| `static void mergeMouOto(std::vector<VoicebankOtoEntry>& entries)` | 业务／适配 | [SampleSettings.h:215](../juce/src/SampleSettings.h#L215) |
| `static int mouRegionCount(const juce::String& classes)` | 业务／适配 | [SampleSettings.h:218](../juce/src/SampleSettings.h#L218) |
| `static juce::String mouClassesForCount(const juce::String& classes, int count)` | 业务／适配 | [SampleSettings.h:222](../juce/src/SampleSettings.h#L222) |
| `static bool updateMouOtoEntry(const VoicebankOtoEntry& original, const VoicebankOtoEntry& updated, juce::String& error)` | 业务／适配 | [SampleSettings.h:226](../juce/src/SampleSettings.h#L226) |
| `static bool createMouOto(const juce::File& root, int& written, int& kept, int& merged, juce::String& error)` | 业务／适配 | [SampleSettings.h:238](../juce/src/SampleSettings.h#L238) |
| `static juce::File mouOtoFileFor(const juce::File& otoFile)` | 业务／适配 | [SampleSettings.h:240](../juce/src/SampleSettings.h#L240) |
| `static bool createJieOto(const juce::File& root, int& written, int& kept, juce::String& error)` | 业务／适配 | [SampleSettings.h:243](../juce/src/SampleSettings.h#L243) |
| `static bool createJieOtoFromMou(const juce::File& root, JieFromMouResult& result, juce::String& error)` | 业务／适配 | [SampleSettings.h:248](../juce/src/SampleSettings.h#L248) |
| `static bool updateJieOtoEntry(const VoicebankOtoEntry& original, const VoicebankOtoEntry& updated, juce::String& error)` | 业务／适配 | [SampleSettings.h:250](../juce/src/SampleSettings.h#L250) |

## AccompanimentTempoTask（8 个声明）

| 实际签名 | 类型 | 定义入口 |
| --- | --- | --- |
| `explicit AccompanimentTempoTask(ClipData region, juce::File runtime = {})` | 生命周期 | [AccompanimentTempo.h:12](../juce/src/AccompanimentTempo.h#L12) |
| `~AccompanimentTempoTask() override` | 生命周期 | [AccompanimentTempo.h:13](../juce/src/AccompanimentTempo.h#L13) |
| `void cancel()` | 业务／适配 | [AccompanimentTempo.h:14](../juce/src/AccompanimentTempo.h#L14) |
| `bool finished() const` | 业务／适配 | [AccompanimentTempo.h:15](../juce/src/AccompanimentTempo.h#L15) |
| `int stage() const` | 业务／适配 | [AccompanimentTempo.h:16](../juce/src/AccompanimentTempo.h#L16) |
| `juce::var result() const` | 业务／适配 | [AccompanimentTempo.h:17](../juce/src/AccompanimentTempo.h#L17) |
| `static bool canAnalyse(const ClipData&)` | 业务／适配 | [AccompanimentTempo.h:18](../juce/src/AccompanimentTempo.h#L18) |
| `static juce::File runtimeConfig()` | 业务／适配 | [AccompanimentTempo.h:19](../juce/src/AccompanimentTempo.h#L19) |

## AnalysisService（7 个声明）

| 实际签名 | 类型 | 定义入口 |
| --- | --- | --- |
| `static AnalysisConfig configFromProperties(const juce::PropertiesFile* properties)` | 业务／适配 | [backend/AnalysisService.h:49](../juce/src/backend/AnalysisService.h#L49) |
| `static AnalysisConfig configFromEnvironment()` | 业务／适配 | [backend/AnalysisService.h:50](../juce/src/backend/AnalysisService.h#L50) |
| `static AnalysisStatus status(const AnalysisConfig& config)` | 业务／适配 | [backend/AnalysisService.h:51](../juce/src/backend/AnalysisService.h#L51) |
| `static AnalysisResult analyse(const juce::File& file, const AnalysisConfig& config, juce::String& error, Progress progress = {})` | 业务／适配 | [backend/AnalysisService.h:52](../juce/src/backend/AnalysisService.h#L52) |
| `static std::size_t applySourcePitch(ClipData& clip, const std::vector<NoteData>& sourceNotes)` | 业务／适配 | [backend/AnalysisService.h:56](../juce/src/backend/AnalysisService.h#L56) |
| `static bool reanalyseProjectSourcePitch(ProjectData& project, const AnalysisConfig& config, juce::String& error, Progress progress = {}, AnalysisStatus* usedStatus = nullptr)` | 业务／适配 | [backend/AnalysisService.h:58](../juce/src/backend/AnalysisService.h#L58) |
| `static juce::String backendText(const AnalysisStatus& status)` | 业务／适配 | [backend/AnalysisService.h:63](../juce/src/backend/AnalysisService.h#L63) |

## RenderService（12 个声明）

| 实际签名 | 类型 | 定义入口 |
| --- | --- | --- |
| `RenderService()` | 生命周期 | [backend/RenderService.h:90](../juce/src/backend/RenderService.h#L90) |
| `~RenderService()` | 生命周期 | [backend/RenderService.h:91](../juce/src/backend/RenderService.h#L91) |
| `void renderMld5(Mld5RenderRequest request, Completion completion)` | 业务／适配 | [backend/RenderService.h:92](../juce/src/backend/RenderService.h#L92) |
| `void renderMld5File(Mld5FileRenderRequest request, FileCompletion completion, RenderSchedule schedule = {})` | 业务／适配 | [backend/RenderService.h:93](../juce/src/backend/RenderService.h#L93) |
| `void renderUtau(UtauRenderRequest request, FileCompletion completion, RenderSchedule schedule = {})` | 业务／适配 | [backend/RenderService.h:95](../juce/src/backend/RenderService.h#L95) |
| `void renderNsfUtau(UtauRenderRequest request, juce::File modelDirectory, OrtExecutionConfig execution, FileCompletion completion, RenderSchedule schedule = {})` | 业务／适配 | [backend/RenderService.h:101](../juce/src/backend/RenderService.h#L101) |
| `void cancelAll()` | 业务／适配 | [backend/RenderService.h:104](../juce/src/backend/RenderService.h#L104) |
| `bool hasActiveJobs()` | 业务／适配 | [backend/RenderService.h:105](../juce/src/backend/RenderService.h#L105) |
| `void setPlaybackPosition(double seconds) noexcept` | 业务／适配 | [backend/RenderService.h:106](../juce/src/backend/RenderService.h#L106) |
| `void beginUpdate()` | 业务／适配 | [backend/RenderService.h:107](../juce/src/backend/RenderService.h#L107) |
| `void endUpdate(const std::unordered_set<std::string>& keys)` | 业务／适配 | [backend/RenderService.h:108](../juce/src/backend/RenderService.h#L108) |
| `double playbackPriorityPosition() const` | 业务／适配 | [backend/RenderService.h:109](../juce/src/backend/RenderService.h#L109) |

## UtauRenderer（29 个声明）

| 实际签名 | 类型 | 定义入口 |
| --- | --- | --- |
| `static std::optional<UtauSampleTiming> mappedAliasTiming( const juce::File& directory, const juce::String& alias, float midiNote)` | 业务／适配 | [backend/UtauRenderer.h:253](../juce/src/backend/UtauRenderer.h#L253) |
| `static bool supportsComponentExport(const juce::File& engine, const juce::String& effectiveFlags)` | 业务／适配 | [backend/UtauRenderer.h:257](../juce/src/backend/UtauRenderer.h#L257) |
| `static juce::String diagnosticPitchbend( const UtauNoteRenderSpec& note, double bpm, double preutterance, double outputSeconds)` | 诊断 | [backend/UtauRenderer.h:259](../juce/src/backend/UtauRenderer.h#L259) |
| `static std::vector<UtauAmplitudePoint> fitAmplitudeEnvelope( const std::vector<UtauAmplitudePoint>& drawn, double preutteranceSeconds, double soundingEndSeconds)` | 业务／适配 | [backend/UtauRenderer.h:263](../juce/src/backend/UtauRenderer.h#L263) |
| `static void invalidateVoicebankCache()` | 业务／适配 | [backend/UtauRenderer.h:266](../juce/src/backend/UtauRenderer.h#L266) |
| `static void prewarmHfDaemon(const juce::File& resamplerExecutable)` | 业务／适配 | [backend/UtauRenderer.h:272](../juce/src/backend/UtauRenderer.h#L272) |
| `static bool startHfDaemonIfNeeded(const juce::File& resamplerExecutable, int port = 51765)` | 业务／适配 | [backend/UtauRenderer.h:275](../juce/src/backend/UtauRenderer.h#L275) |
| `static void shutdownHfDaemon()` | 业务／适配 | [backend/UtauRenderer.h:280](../juce/src/backend/UtauRenderer.h#L280) |
| `static juce::String hfDaemonInterpreter(const juce::File& engineDirectory)` | 业务／适配 | [backend/UtauRenderer.h:285](../juce/src/backend/UtauRenderer.h#L285) |
| `static bool voicebankIndexReady(const juce::File& voicebankDirectory, bool fourRegion, bool consonantClasses, std::function<void()> whenReady)` | 业务／适配 | [backend/UtauRenderer.h:291](../juce/src/backend/UtauRenderer.h#L291) |
| `static int diagnosticIndexBuilds()` | 诊断 | [backend/UtauRenderer.h:299](../juce/src/backend/UtauRenderer.h#L299) |
| `static int diagnosticMessageThreadWaits()` | 诊断 | [backend/UtauRenderer.h:302](../juce/src/backend/UtauRenderer.h#L302) |
| `static void diagnosticRecheckVoicebankFiles()` | 诊断 | [backend/UtauRenderer.h:303](../juce/src/backend/UtauRenderer.h#L303) |
| `static void recheckVoicebankFiles(const juce::File& root)` | 业务／适配 | [backend/UtauRenderer.h:305](../juce/src/backend/UtauRenderer.h#L305) |
| `static juce::String diagnosticResolve(const juce::File& voicebankDirectory, const juce::String& alias, float midiNote, bool fourRegion, bool consonantClasses, bool byScan)` | 诊断 | [backend/UtauRenderer.h:306](../juce/src/backend/UtauRenderer.h#L306) |
| `static double crossfadeEnd(double soundStart, double overlap, double nextNoteEnd)` | 业务／适配 | [backend/UtauRenderer.h:318](../juce/src/backend/UtauRenderer.h#L318) |
| `static bool crossfadesInto(double soundStart, double noteEnd)` | 业务／适配 | [backend/UtauRenderer.h:327](../juce/src/backend/UtauRenderer.h#L327) |
| `static bool sendsFlagCurves(bool noteFlagCurve)` | 业务／适配 | [backend/UtauRenderer.h:338](../juce/src/backend/UtauRenderer.h#L338) |
| `static bool readsOnlyFirstTwoRegions(bool fourRegion, bool consonantClasses, double durationSeconds)` | 业务／适配 | [backend/UtauRenderer.h:339](../juce/src/backend/UtauRenderer.h#L339) |
| `static std::array<double, 4> firstTwoRegions( const std::array<double, 4>& regionSeconds)` | 业务／适配 | [backend/UtauRenderer.h:342](../juce/src/backend/UtauRenderer.h#L342) |
| `static std::optional<UtauSampleTiming> sampleTiming( const juce::File& voicebankDirectory, const juce::String& alias, float midiNote, int consonantVelocity = 100, bool fourRegion = false, bool consonantClasses = false)` | 业务／适配 | [backend/UtauRenderer.h:344](../juce/src/backend/UtauRenderer.h#L344) |
| `static std::optional<UtauSampleTiming> sampleTiming( const juce::File& voicebankDirectory, const juce::String& alias, float midiNote, int consonantVelocity, bool fourRegion, bool consonantClasses, const UtauOtoOverride* noteOto)` | 业务／适配 | [backend/UtauRenderer.h:351](../juce/src/backend/UtauRenderer.h#L351) |
| `static ResolvedSample resolveVoiceSample( const juce::File& voicebankDirectory, const juce::String& alias, float midiNote, int consonantVelocity = 100, bool fourRegion = false, bool consonantClasses = false, double stpSeconds = 0.0, bool preutteranceOverrideEnabled = false, double preutteranceSeconds = 0.0, bool overlapOverrideEnabled = false, double overlapSeconds = 0.0, const UtauOtoOverride* noteOto = nullptr, bool aliasIsResolved = false)` | 业务／适配 | [backend/UtauRenderer.h:383](../juce/src/backend/UtauRenderer.h#L383) |
| `static UtauRenderResult render(const UtauRenderRequest& request)` | 业务／适配 | [backend/UtauRenderer.h:390](../juce/src/backend/UtauRenderer.h#L390) |
| `static juce::AudioBuffer<float> pianoPreview(const UtauNoteRenderSpec& note, double sampleRate = 44100.0)` | 业务／适配 | [backend/UtauRenderer.h:391](../juce/src/backend/UtauRenderer.h#L391) |
| `static float amplitudeGain(const std::vector<UtauAmplitudePoint>& envelope, double localSeconds)` | 业务／适配 | [backend/UtauRenderer.h:392](../juce/src/backend/UtauRenderer.h#L392) |
| `static std::optional<UtauTailFadeSpan> tailFadeSpan( const UtauSampleTiming& timing,double soundStart,double soundEnd, double nominalDuration,int velocity,bool fourRegion,bool mou, const std::array<double,3>* manual=nullptr)` | 业务／适配 | [backend/UtauRenderer.h:394](../juce/src/backend/UtauRenderer.h#L394) |
| `static std::optional<UtauTailFadeSpan> headEnvelopeSpan( const UtauSampleTiming& timing,double soundStart,double soundEnd, double nominalDuration,int velocity,bool fourRegion,bool mou, const std::array<double,3>* manual=nullptr)` | 业务／适配 | [backend/UtauRenderer.h:398](../juce/src/backend/UtauRenderer.h#L398) |
| `static UtauRegionSplit regionSplit( const std::array<double, 4>& sourceSeconds, double outputSeconds, int consonantVelocity, double leadInSeconds, const std::array<double, 3>* manualFractions, const juce::String* classes = nullptr)` | 业务／适配 | [backend/UtauRenderer.h:409](../juce/src/backend/UtauRenderer.h#L409) |

## NsfHifiganRenderer（3 个声明）

| 实际签名 | 类型 | 定义入口 |
| --- | --- | --- |
| `static void shutdown()` | 业务／适配 | [backend/NsfHifiganRenderer.h:66](../juce/src/backend/NsfHifiganRenderer.h#L66) |
| `static bool modelAvailable(const juce::File& configuredModelDirectory)` | 业务／适配 | [backend/NsfHifiganRenderer.h:67](../juce/src/backend/NsfHifiganRenderer.h#L67) |
| `static NsfHifiganRenderResult render( const juce::AudioBuffer<float>& source, double sampleRate, int targetSamples, double framePeriodMs, const std::vector<float>& targetMidi, const std::vector<float>& formantSemitones, const std::vector<NsfHifiganTimeMapPoint>& timeMap, const juce::File& configuredModelDirectory, const OrtExecutionConfig& execution, NsfHifiganStretchOrder stretchOrder, bool normalizeVolume = false, const NsfHifiganEdgeGuard& edgeGuard = NsfHifiganEdgeGuard{}, const std::function<bool()>& cancelled = {}, bool protectNativeNoise = false, bool smoothPitchTransitions = true, bool incrementalNativeRender = false, const std::vector<float>& sourceMidi = {})` | 业务／适配 | [backend/NsfHifiganRenderer.h:68](../juce/src/backend/NsfHifiganRenderer.h#L68) |

## DiffSingerRenderer（12 个声明）

| 实际签名 | 类型 | 定义入口 |
| --- | --- | --- |
| `static void openProjectCache(const juce::File& project = {})` | 业务／适配 | [backend/DiffSingerRenderer.h:10](../juce/src/backend/DiffSingerRenderer.h#L10) |
| `static bool saveProjectCache(const juce::File& project)` | 业务／适配 | [backend/DiffSingerRenderer.h:11](../juce/src/backend/DiffSingerRenderer.h#L11) |
| `static juce::String cacheSession()` | 业务／适配 | [backend/DiffSingerRenderer.h:12](../juce/src/backend/DiffSingerRenderer.h#L12) |
| `static void configure(const DiffSingerOptions&)` | 业务／适配 | [backend/DiffSingerRenderer.h:13](../juce/src/backend/DiffSingerRenderer.h#L13) |
| `static juce::var inferenceOptions(bool exporting = false)` | 业务／适配 | [backend/DiffSingerRenderer.h:14](../juce/src/backend/DiffSingerRenderer.h#L14) |
| `static juce::String inferenceStatus()` | 业务／适配 | [backend/DiffSingerRenderer.h:15](../juce/src/backend/DiffSingerRenderer.h#L15) |
| `static juce::var inferenceReport()` | 业务／适配 | [backend/DiffSingerRenderer.h:16](../juce/src/backend/DiffSingerRenderer.h#L16) |
| `static void shutdown()` | 业务／适配 | [backend/DiffSingerRenderer.h:17](../juce/src/backend/DiffSingerRenderer.h#L17) |
| `static bool isVoicebank(const juce::File& directory)` | 业务／适配 | [backend/DiffSingerRenderer.h:18](../juce/src/backend/DiffSingerRenderer.h#L18) |
| `static juce::var invoke(juce::var request, const std::function<bool()>& cancelled = {}, juce::AudioBuffer<float>* audio = nullptr, double* rate = nullptr)` | 业务／适配 | [backend/DiffSingerRenderer.h:19](../juce/src/backend/DiffSingerRenderer.h#L19) |
| `static juce::var requestJson(const UtauRenderRequest&, const juce::String& operation)` | 业务／适配 | [backend/DiffSingerRenderer.h:21](../juce/src/backend/DiffSingerRenderer.h#L21) |
| `static UtauRenderResult render(const UtauRenderRequest&)` | 业务／适配 | [backend/DiffSingerRenderer.h:22](../juce/src/backend/DiffSingerRenderer.h#L22) |

## McpServer（12 个声明）

| 实际签名 | 类型 | 定义入口 |
| --- | --- | --- |
| `explicit McpServer(const juce::StringArray& extraRoots = {})` | 生命周期 | [backend/McpServer.h:18](../juce/src/backend/McpServer.h#L18) |
| `McpServer(ProjectModel& model, AudioEngine* engine, const juce::StringArray& extraRoots = {})` | 生命周期 | [backend/McpServer.h:20](../juce/src/backend/McpServer.h#L20) |
| `juce::var processRequest(const juce::var& request, bool& shouldRespond)` | 业务／适配 | [backend/McpServer.h:22](../juce/src/backend/McpServer.h#L22) |
| `juce::var executeTool(const juce::String& name, const juce::var& args)` | 业务／适配 | [backend/McpServer.h:23](../juce/src/backend/McpServer.h#L23) |
| `juce::var snapshotJson() const` | 业务／适配 | [backend/McpServer.h:24](../juce/src/backend/McpServer.h#L24) |
| `static juce::var noteJson(const NoteData& note, bool includeCurves = true)` | 业务／适配 | [backend/McpServer.h:25](../juce/src/backend/McpServer.h#L25) |
| `void allowFile(const juce::File& file)` | 业务／适配 | [backend/McpServer.h:26](../juce/src/backend/McpServer.h#L26) |
| `static juce::var result(const juce::String& text, bool failed = false)` | 业务／适配 | [backend/McpServer.h:27](../juce/src/backend/McpServer.h#L27) |
| `int run()` | 业务／适配 | [backend/McpServer.h:28](../juce/src/backend/McpServer.h#L28) |
| `static bool pathWithinRoots(const std::vector<juce::File>& roots, const juce::File& target)` | 业务／适配 | [backend/McpServer.h:32](../juce/src/backend/McpServer.h#L32) |
| `static std::vector<juce::File> projectRoots(const ProjectData& data)` | 业务／适配 | [backend/McpServer.h:36](../juce/src/backend/McpServer.h#L36) |
| `static juce::var diagnosticTools()` | 诊断 | [backend/McpServer.h:38](../juce/src/backend/McpServer.h#L38) |

## LiveMcpBridge（5 个声明）

| 实际签名 | 类型 | 定义入口 |
| --- | --- | --- |
| `LiveMcpBridge(Handler handler, std::function<juce::var()> describe)` | 生命周期 | [backend/LiveMcpBridge.h:13](../juce/src/backend/LiveMcpBridge.h#L13) |
| `~LiveMcpBridge() override` | 生命周期 | [backend/LiveMcpBridge.h:14](../juce/src/backend/LiveMcpBridge.h#L14) |
| `const juce::String& sessionId() const` | 业务／适配 | [backend/LiveMcpBridge.h:15](../juce/src/backend/LiveMcpBridge.h#L15) |
| `static int runProxy(const juce::String& session = {}, bool listOnly = false)` | 业务／适配 | [backend/LiveMcpBridge.h:16](../juce/src/backend/LiveMcpBridge.h#L16) |
| `static juce::File sessionsDirectory()` | 业务／适配 | [backend/LiveMcpBridge.h:17](../juce/src/backend/LiveMcpBridge.h#L17) |

## MCP 命令目录

Headless 命令来自 `McpServer::diagnosticTools`；Live 复用它并叠加窗口命令，并非两套完整独立业务实现。Live 写操作另有 expected_revision、会话和草稿提交约束。字段类型／必填项以链接处 schema 为准；不是所有 GUI 功能都有 MCP 命令。

| 命令 | 注册位置 | 源码描述 |
| --- | --- | --- |
| `project_new` | [backend/McpServer.cpp:461](../juce/src/backend/McpServer.cpp#L461) | Create an empty HachiShifter project / 新建工程 |
| `project_open` | [backend/McpServer.cpp:462](../juce/src/backend/McpServer.cpp#L462) | Open HJPX, legacy HSPX, MPD, MIDI, or audio from path / 打开工程或素材 |
| `project_save` | [backend/McpServer.cpp:469](../juce/src/backend/McpServer.cpp#L469) | Save current project as HJPX / 保存工程 |
| `project_snapshot` | [backend/McpServer.cpp:473](../juce/src/backend/McpServer.cpp#L473) | Read every current track, clip, note, pitch and marker / 读取全部工程内容 |
| `import_audio` | [backend/McpServer.cpp:474](../juce/src/backend/McpServer.cpp#L474) | Import an audio file at start_seconds / 导入音频 |
| `analyse_audio` | [backend/McpServer.cpp:483](../juce/src/backend/McpServer.cpp#L483) | Run configured GAME+FCPE analysis with native fallback and return actual backend / 执行 GAME+FCPE 分析并报告实际后端 |
| `analysis_status` | [backend/McpServer.cpp:489](../juce/src/backend/McpServer.cpp#L489) | Inspect GAME large/small, FCPE and inference availability / 查看 GAME、FCPE 与推理状态 |
| `import_midi` | [backend/McpServer.cpp:491](../juce/src/backend/McpServer.cpp#L491) | Import MIDI notes and tempo / 导入 MIDI |
| `export_midi` | [backend/McpServer.cpp:495](../juce/src/backend/McpServer.cpp#L495) | Write all tracks or one track as MIDI, retaining conductor tempo/meter and lyrics / 导出 MIDI |
| `export_ust` | [backend/McpServer.cpp:503](../juce/src/backend/McpServer.cpp#L503) | Export one monophonic track as UST, retaining imported opaque data / 导出单轨 UST |
| `import_ust` | [backend/McpServer.cpp:508](../juce/src/backend/McpServer.cpp#L508) | Import a UTAU project as a plain UTAU track / 导入 UST |
| `import_melodyne` | [backend/McpServer.cpp:513](../juce/src/backend/McpServer.cpp#L513) | Import Melodyne MPD edits; recursive_media, preserve_edits and source_pitch control import / 导入 Melodyne 工程并控制素材搜索、工程编辑与原始 F0 |
| `set_tempo` | [backend/McpServer.cpp:527](../juce/src/backend/McpServer.cpp#L527) | Set BPM and time signature / 设置速度与拍号 |
| `add_track` | [backend/McpServer.cpp:535](../juce/src/backend/McpServer.cpp#L535) | Create a melodic, audio or accompaniment track / 新建旋律、普通或伴奏轨道 |
| `set_track` | [backend/McpServer.cpp:543](../juce/src/backend/McpServer.cpp#L543) | Set compose, mute, solo, gain, pan and render algorithms / 设置轨道及算法 |
| `set_clip` | [backend/McpServer.cpp:587](../juce/src/backend/McpServer.cpp#L587) | Set clip gain, fades and mute state / 设置采样增益、淡入淡出和静音 |
| `move_clip` | [backend/McpServer.cpp:598](../juce/src/backend/McpServer.cpp#L598) | Move a clip on the timeline / 移动采样 |
| `resize_clip` | [backend/McpServer.cpp:604](../juce/src/backend/McpServer.cpp#L604) | Stretch a whole clip while preserving its source media / 整体拉伸采样并保留原始素材 |
| `duplicate_clip` | [backend/McpServer.cpp:612](../juce/src/backend/McpServer.cpp#L612) | Deep-copy a clip and its notes to a timeline position / 深度复制采样及其音符到指定位置 |
| `transpose_note` | [backend/McpServer.cpp:620](../juce/src/backend/McpServer.cpp#L620) | Move a note and its whole contour / 整体移动音高线 |
| `edit_notes_pitch` | [backend/McpServer.cpp:626](../juce/src/backend/McpServer.cpp#L626) | Batch transpose, set, average or quantize note pitches / 批量移调、设置、平均或量化音符 |
| `duplicate_notes` | [backend/McpServer.cpp:642](../juce/src/backend/McpServer.cpp#L642) | Deep-copy selected notes into a target clip / 深度复制所选音符到目标采样 |
| `resize_note` | [backend/McpServer.cpp:652](../juce/src/backend/McpServer.cpp#L652) | Change note time bounds / 修改音符时间 |
| `set_note` | [backend/McpServer.cpp:660](../juce/src/backend/McpServer.cpp#L660) | Set pitch, Robust Pitch Curve, tension, breath, formant, gain, Attack and consonant parameters / 设置稳健音高线及全部音符参数 |
| `set_pitch_curve` | [backend/McpServer.cpp:735](../juce/src/backend/McpServer.cpp#L735) | Draw an absolute target-pitch curve without replacing source F0 / 绘制目标音高线并保留原始 F0 |
| `add_note` | [backend/McpServer.cpp:742](../juce/src/backend/McpServer.cpp#L742) | Create a note in a clip / 在采样中创建音符 |
| `remove_note` | [backend/McpServer.cpp:752](../juce/src/backend/McpServer.cpp#L752) | Delete a note / 删除音符 |
| `toggle_note_connection` | [backend/McpServer.cpp:756](../juce/src/backend/McpServer.cpp#L756) | Connect or separate adjacent notes / 连接或分离相邻音符 |
| `set_notes_portamento` | [backend/McpServer.cpp:761](../juce/src/backend/McpServer.cpp#L761) | Set real pitch-only glide bridges; never join source audio or change envelopes / 设置真实滑音连接，不拼接素材 |
| `remove_clip` | [backend/McpServer.cpp:767](../juce/src/backend/McpServer.cpp#L767) | Delete a clip / 删除采样 |
| `remove_track` | [backend/McpServer.cpp:771](../juce/src/backend/McpServer.cpp#L771) | Delete a track / 删除轨道 |
| `undo` | [backend/McpServer.cpp:775](../juce/src/backend/McpServer.cpp#L775) | Undo the last project edit / 撤销工程编辑 |
| `redo` | [backend/McpServer.cpp:776](../juce/src/backend/McpServer.cpp#L776) | Redo the last project edit / 重做工程编辑 |
| `utau_render_selection` | [backend/McpServer.cpp:777](../juce/src/backend/McpServer.cpp#L777) | Choose which UTAU notes render and play; empty selects every note / 选择参与 UTAU 渲染与试听的音符 |
| `set_utau_resampler` | [backend/McpServer.cpp:784](../juce/src/backend/McpServer.cpp#L784) | Point UTAU rendering at a resampler executable / 指定 UTAU 重采样器 |
| `render_prepare` | [backend/McpServer.cpp:789](../juce/src/backend/McpServer.cpp#L789) | Pre-render the current project with its selected algorithms / 按当前所选算法预渲染工程 |
| `render_status` | [backend/McpServer.cpp:795](../juce/src/backend/McpServer.cpp#L795) | Read pre-render progress and active backends / 读取预渲染进度与实际后端 |
| `export_wav` | [backend/McpServer.cpp:796](../juce/src/backend/McpServer.cpp#L796) | Render every note and export to WAV; track_id exports one track alone, from_seconds/to_seconds one stretch / 渲染全曲并导出 WAV，track_id 可单独导出一个轨道 |
| `transport_play` | [backend/McpServer.cpp:811](../juce/src/backend/McpServer.cpp#L811) | Render if needed and start playback; play_until_seconds stops where a selection ends / 必要时预渲染并开始播放 |
| `transport_stop` | [backend/McpServer.cpp:819](../juce/src/backend/McpServer.cpp#L819) | Stop transport playback / 停止播放 |
| `transport_seek` | [backend/McpServer.cpp:820](../juce/src/backend/McpServer.cpp#L820) | Seek transport to position_seconds / 跳转播放位置 |
| `transport_status` | [backend/McpServer.cpp:824](../juce/src/backend/McpServer.cpp#L824) | Read playback position and render state / 读取播放位置与渲染状态 |
| `sample_settings_read` | [backend/McpServer.cpp:825](../juce/src/backend/McpServer.cpp#L825) | Read or derive audio otto.ini regions / 读取或生成音频 otto.ini 分段 |
| `sample_settings_save` | [backend/McpServer.cpp:830](../juce/src/backend/McpServer.cpp#L830) | Save audio regions to otto.ini / 保存音频分段到 otto.ini |
| `oto_import` | [backend/McpServer.cpp:837](../juce/src/backend/McpServer.cpp#L837) | Import one audio file's regions from UTAU oto.ini / 从 UTAU oto.ini 导入单个音频分段 |
| `oto_export` | [backend/McpServer.cpp:846](../juce/src/backend/McpServer.cpp#L846) | Export one audio file's regions to UTAU oto.ini / 将单个音频分段导出为 UTAU oto.ini |
| `jie_oto_create` | [backend/McpServer.cpp:852](../juce/src/backend/McpServer.cpp#L852) | Seed a voicebank's four-region oto4.ini from its oto.ini / 按 oto.ini 生成界•OTO |
| `voicebank_import` | [backend/McpServer.cpp:856](../juce/src/backend/McpServer.cpp#L856) | Inspect a UTAU or native material folder without writing annotations / 只读检查 UTAU 或原生素材目录，不写入标注 |
| `read_file` | [backend/McpServer.cpp:860](../juce/src/backend/McpServer.cpp#L860) | Read a byte range as base64 / 读取任意文件内容 |
| `list_directory` | [backend/McpServer.cpp:868](../juce/src/backend/McpServer.cpp#L868) | List a directory with type and size / 列出目录内容 |
| `editor_selection` | [MainComponentMcp.cpp:57](../juce/src/MainComponentMcp.cpp#L57) | Read ONLY notes currently selected in the editor / 读取当前选区。Use FIRST for 'these notes', 'this phrase', or 'what did I select'. Returns lyrics, note names, timing and tracks; empty selection returns no notes, NEVER the whole project. |
| `editor_status` | [MainComponentMcp.cpp:60](../juce/src/MainComponentMcp.cpp#L60) | Read the connected window, current file, revision, selection, playback and DS state |
| `editor_query` | [MainComponentMcp.cpp:61](../juce/src/MainComponentMcp.cpp#L61) | Read selected notes by default; range uses absolute project seconds; curves are opt-in |
| `editor_select` | [MainComponentMcp.cpp:66](../juce/src/MainComponentMcp.cpp#L66) | Change real GUI selection and focus; [] clears it |
| `editor_batch` | [MainComponentMcp.cpp:67](../juce/src/MainComponentMcp.cpp#L67) | Apply up to 128 project edit commands atomically with one Undo; imports, file and transport commands are excluded |
| `editor_job_status` | [MainComponentMcp.cpp:69](../juce/src/MainComponentMcp.cpp#L69) | Read a background job's state/result; completed does not imply success, check result.isError |
| `editor_cancel_job` | [MainComponentMcp.cpp:70](../juce/src/MainComponentMcp.cpp#L70) | Cancel a pending job; staged edits are discarded, external file writes may already have completed |
| `ds_generate_pitch` | [MainComponentMcp.cpp:71](../juce/src/MainComponentMcp.cpp#L71) | Regenerate DS pitch using GUI logic; selection required unless whole_track=true |
| `ds_generate_parameters` | [MainComponentMcp.cpp:72](../juce/src/MainComponentMcp.cpp#L72) | Regenerate DS prediction parameters using GUI logic; preserve offsets and unselected notes |
| `ds_query_phonemes` | [MainComponentMcp.cpp:73](../juce/src/MainComponentMcp.cpp#L73) | Predict phoneme timing in full phrase context, returning only selected notes by default; includes stable note IDs and timing context |
| `ds_capabilities` | [MainComponentMcp.cpp:74](../juce/src/MainComponentMcp.cpp#L74) | Inspect the focused DS voicebank and its parameter support asynchronously |
| `ds_set_pitch_offset` | [MainComponentMcp.cpp:75](../juce/src/MainComponentMcp.cpp#L75) | Set an independent additive semitone curve; preserves the original prediction reference |
| `ds_set_parameter` | [MainComponentMcp.cpp:77](../juce/src/MainComponentMcp.cpp#L77) | Set a DS actual parameter or offset curve; supports only parameters exposed by this voicebank |
| `ds_set_pronunciation` | [MainComponentMcp.cpp:80](../juce/src/MainComponentMcp.cpp#L80) | Set per-note pronunciation override; empty text restores automatic conversion |
| `ds_set_timing` | [MainComponentMcp.cpp:82](../juce/src/MainComponentMcp.cpp#L82) | Set phoneme onset overrides from an inspected timing context; null restores prediction |
| `hamood_analyse_audio` | [MainComponentMcp.cpp:84](../juce/src/MainComponentMcp.cpp#L84) | Analyse one accompaniment clip with Beat This! small0 + BTC; background job, cached by source audio/model hash. Read-only project operation. No pitch or tempo edits. |
| `hamood_alignment_preview` | [MainComponentMcp.cpp:86](../juce/src/MainComponentMcp.cpp#L86) | Preview a small beat-phase offset against the existing project tempo grid. Does NOT resolve whole-bar ambiguity or change audio/tempo. Inspect start/middle/end windows before applying move_clip. |
| `hamood_audio_context` | [MainComponentMcp.cpp:90](../juce/src/MainComponentMcp.cpp#L90) | Read cached timestamped chords, low-register chroma candidates, beats and downbeats on the CURRENT project timeline. Scores are not calibrated probabilities. Defaults to selected-note time range; explicit range or whole_clip required otherwise. |
| `hamood_get_context` | [MainComponentMcp.cpp:95](../juce/src/MainComponentMcp.cpp#L95) | Read project-persisted HAMOOD settings, manual key ranges, chord timeline, section labels and confirmation states. Timeline positions use quarter notes. |
| `hamood_set_context` | [MainComponentMcp.cpp:96](../juce/src/MainComponentMcp.cpp#L96) | Replace HAMOOD project metadata with one undo step; read hamood_get_context first and preserve unrelated entries. schema=1; settings object; chords, sections, key_predictions arrays. Chords require start_quarter/end_quarter, label, pitch_classes 0..11, score 0..1 and confirmed. Confirmed chords survive audio reanalysis. |
| `hamood_preview` | [MainComponentMcp.cpp:98](../juce/src/MainComponentMcp.cpp#L98) | 只读预览和声方案 |
| `hamood_generate` | [MainComponentMcp.cpp:98](../juce/src/MainComponentMcp.cpp#L98) | 生成独立和声轨道，一次撤销 |

## 启动参数字面量

仅来自 Main.cpp 的 `--…` 字符串去重；包括诊断、启动选项和帮助文字，不代表全是用户功能，也不等同于测试已执行。参数说明见原有 [命令索引](AI_SOURCE_INDEX.md)。

| 参数 | 首次出现位置 |
| --- | --- |
| `--migrate-otto-folder` | [Main.cpp:192](../juce/src/Main.cpp#L192) |
| `--export-otto-folder` | [Main.cpp:193](../juce/src/Main.cpp#L193) |
| `--include-bare` | [Main.cpp:203](../juce/src/Main.cpp#L203) |
| `--replace` | [Main.cpp:207](../juce/src/Main.cpp#L207) |
| `--smoke-main-project-registration-ui` | [Main.cpp:217](../juce/src/Main.cpp#L217) |
| `--smoke-classic-note-scope` | [Main.cpp:223](../juce/src/Main.cpp#L223) |
| `--smoke-oto-provenance` | [Main.cpp:228](../juce/src/Main.cpp#L228) |
| `--smoke-community-dataset-menu` | [Main.cpp:233](../juce/src/Main.cpp#L233) |
| `--smoke-material-root-registration` | [Main.cpp:238](../juce/src/Main.cpp#L238) |
| `--smoke-variant-asset-ui` | [Main.cpp:243](../juce/src/Main.cpp#L243) |
| `--smoke-voicebank-family` | [Main.cpp:248](../juce/src/Main.cpp#L248) |
| `--smoke-empty-lyric-piano` | [Main.cpp:253](../juce/src/Main.cpp#L253) |
| `--smoke-asset-context` | [Main.cpp:258](../juce/src/Main.cpp#L258) |
| `--inspect-piano-project` | [Main.cpp:263](../juce/src/Main.cpp#L263) |
| `--smoke-track-types` | [Main.cpp:268](../juce/src/Main.cpp#L268) |
| `--smoke-clip-length` | [Main.cpp:273](../juce/src/Main.cpp#L273) |
| `--smoke-engine-view-capabilities` | [Main.cpp:280](../juce/src/Main.cpp#L280) |
| `--smoke-track-editing-interfaces-ui` | [Main.cpp:287](../juce/src/Main.cpp#L287) |
| `--smoke-material-variants` | [Main.cpp:294](../juce/src/Main.cpp#L294) |
| `--smoke-community-dataset` | [Main.cpp:299](../juce/src/Main.cpp#L299) |
| `--smoke-ustx-importer` | [Main.cpp:305](../juce/src/Main.cpp#L305) |
| `--smoke-community-import-gui` | [Main.cpp:310](../juce/src/Main.cpp#L310) |
| `--smoke-registered-project-audio` | [Main.cpp:316](../juce/src/Main.cpp#L316) |
| `--smoke-registration-facts` | [Main.cpp:322](../juce/src/Main.cpp#L322) |
| `--smoke-main-material-variants` | [Main.cpp:327](../juce/src/Main.cpp#L327) |
| `--audit-project-material-registration` | [Main.cpp:333](../juce/src/Main.cpp#L333) |
| `--smoke-project-material-registration-model` | [Main.cpp:342](../juce/src/Main.cpp#L342) |
| `--smoke-project-material-registration` | [Main.cpp:347](../juce/src/Main.cpp#L347) |
| `--smoke-note-material-local-override` | [Main.cpp:352](../juce/src/Main.cpp#L352) |
| `--smoke-local-material-scope` | [Main.cpp:357](../juce/src/Main.cpp#L357) |
| `--smoke-connection-pitch-ui` | [Main.cpp:362](../juce/src/Main.cpp#L362) |
| `--smoke-host-pitch-projection` | [Main.cpp:367](../juce/src/Main.cpp#L367) |
| `--smoke-material-pronunciation-catalog` | [Main.cpp:372](../juce/src/Main.cpp#L372) |
| `--smoke-graph-pitch-edit-batch` | [Main.cpp:377](../juce/src/Main.cpp#L377) |
| `--smoke-overlap-selector` | [Main.cpp:382](../juce/src/Main.cpp#L382) |
| `--smoke-forced-portamento-audio` | [Main.cpp:387](../juce/src/Main.cpp#L387) |
| `--smoke-track-pitch-scope` | [Main.cpp:393](../juce/src/Main.cpp#L393) |
| `--smoke-forced-portamento` | [Main.cpp:398](../juce/src/Main.cpp#L398) |
| `--smoke-named-vowel` | [Main.cpp:403](../juce/src/Main.cpp#L403) |
| `--smoke-vowel-timing-audio` | [Main.cpp:408](../juce/src/Main.cpp#L408) |
| `--smoke-pronunciation-timing` | [Main.cpp:414](../juce/src/Main.cpp#L414) |
| `--smoke-adjacent-pronunciation` | [Main.cpp:419](../juce/src/Main.cpp#L419) |
| `--smoke-vowel-stretch-alignment` | [Main.cpp:424](../juce/src/Main.cpp#L424) |
| `--smoke-material-source-replacement` | [Main.cpp:429](../juce/src/Main.cpp#L429) |
| `--smoke-native-nsf-compatibility` | [Main.cpp:434](../juce/src/Main.cpp#L434) |
| `--smoke-native-workflow-e2e` | [Main.cpp:440](../juce/src/Main.cpp#L440) |
| `--smoke-material-topology-audio` | [Main.cpp:449](../juce/src/Main.cpp#L449) |
| `--smoke-midi-export-scope` | [Main.cpp:455](../juce/src/Main.cpp#L455) |
| `--smoke-unified-draw-action` | [Main.cpp:461](../juce/src/Main.cpp#L461) |
| `--smoke-main-source-replacement-ui` | [Main.cpp:467](../juce/src/Main.cpp#L467) |
| `--smoke-single-entry-ui` | [Main.cpp:475](../juce/src/Main.cpp#L475) |
| `--smoke-nsf-native-expression` | [Main.cpp:483](../juce/src/Main.cpp#L483) |
| `--smoke-material-topology` | [Main.cpp:491](../juce/src/Main.cpp#L491) |
| `--smoke-native-connection-graph` | [Main.cpp:497](../juce/src/Main.cpp#L497) |
| `--smoke-material-format-auto` | [Main.cpp:503](../juce/src/Main.cpp#L503) |
| `--smoke-unified-material-editor` | [Main.cpp:509](../juce/src/Main.cpp#L509) |
| `--smoke-material-insertion` | [Main.cpp:515](../juce/src/Main.cpp#L515) |
| `--smoke-material-asset-tree` | [Main.cpp:521](../juce/src/Main.cpp#L521) |
| `--smoke-material-annotation` | [Main.cpp:527](../juce/src/Main.cpp#L527) |
| `--smoke-material-binding` | [Main.cpp:533](../juce/src/Main.cpp#L533) |
| `--smoke-otto-material` | [Main.cpp:539](../juce/src/Main.cpp#L539) |
| `--audit-hamood-keys` | [Main.cpp:548](../juce/src/Main.cpp#L548) |
| `--smoke-hamood-utau` | [Main.cpp:567](../juce/src/Main.cpp#L567) |
| `--smoke-hamood` | [Main.cpp:573](../juce/src/Main.cpp#L573) |
| `--smoke-diffsinger-pitch-offset` | [Main.cpp:580](../juce/src/Main.cpp#L580) |
| `--smoke-accompaniment-tempo` | [Main.cpp:587](../juce/src/Main.cpp#L587) |
| `--smoke-accompaniment` | [Main.cpp:603](../juce/src/Main.cpp#L603) |
| `--smoke-wav-export-options` | [Main.cpp:610](../juce/src/Main.cpp#L610) |
| `--smoke-diffsinger-pitch-persistence` | [Main.cpp:620](../juce/src/Main.cpp#L620) |
| `--reopen` | [Main.cpp:624](../juce/src/Main.cpp#L624) |
| `--smoke-diffsinger-pitch-handles` | [Main.cpp:628](../juce/src/Main.cpp#L628) |
| `--smoke-diffsinger-pitch-restore` | [Main.cpp:634](../juce/src/Main.cpp#L634) |
| `--smoke-diffsinger-pitch-reference` | [Main.cpp:641](../juce/src/Main.cpp#L641) |
| `--smoke-continuous-flag` | [Main.cpp:647](../juce/src/Main.cpp#L647) |
| `--smoke-classic-flag-ui` | [Main.cpp:654](../juce/src/Main.cpp#L654) |
| `--smoke-diffsinger-inference` | [Main.cpp:663](../juce/src/Main.cpp#L663) |
| `--smoke-diffsinger-parameter-retake` | [Main.cpp:669](../juce/src/Main.cpp#L669) |
| `--smoke-diffsinger-cache` | [Main.cpp:676](../juce/src/Main.cpp#L676) |
| `--smoke-diffsinger-parameters` | [Main.cpp:681](../juce/src/Main.cpp#L681) |
| `--smoke-diffsinger-expressions` | [Main.cpp:686](../juce/src/Main.cpp#L686) |
| `--smoke-diffsinger-consonants` | [Main.cpp:693](../juce/src/Main.cpp#L693) |
| `--smoke-diffsinger-timing` | [Main.cpp:699](../juce/src/Main.cpp#L699) |
| `--smoke-diffsinger-ui` | [Main.cpp:706](../juce/src/Main.cpp#L706) |
| `--smoke-diffsinger` | [Main.cpp:715](../juce/src/Main.cpp#L715) |
| `--smoke-chinese-cvvc` | [Main.cpp:722](../juce/src/Main.cpp#L722) |
| `--smoke-output-engine` | [Main.cpp:730](../juce/src/Main.cpp#L730) |
| `--smoke-native-vibrato` | [Main.cpp:739](../juce/src/Main.cpp#L739) |
| `--smoke-native-vibrato-render` | [Main.cpp:750](../juce/src/Main.cpp#L750) |
| `--smoke-native-noise-option` | [Main.cpp:756](../juce/src/Main.cpp#L756) |
| `--smoke-native-incremental` | [Main.cpp:764](../juce/src/Main.cpp#L764) |
| `--smoke-native-shutdown` | [Main.cpp:772](../juce/src/Main.cpp#L772) |
| `--smoke-native-noise` | [Main.cpp:780](../juce/src/Main.cpp#L780) |
| `--smoke-wcs-nonloop` | [Main.cpp:787](../juce/src/Main.cpp#L787) |
| `--smoke-hn-cache` | [Main.cpp:794](../juce/src/Main.cpp#L794) |
| `--force` | [Main.cpp:797](../juce/src/Main.cpp#L797) |
| `--smoke-hifisampler` | [Main.cpp:801](../juce/src/Main.cpp#L801) |
| `--smoke-hifishifter-mel` | [Main.cpp:801](../juce/src/Main.cpp#L801) |
| `--smoke-nsf-regions` | [Main.cpp:810](../juce/src/Main.cpp#L810) |
| `--smoke-nsf-pitch-transitions` | [Main.cpp:818](../juce/src/Main.cpp#L818) |
| `--smoke-voicebank-import` | [Main.cpp:824](../juce/src/Main.cpp#L824) |
| `--smoke-voicebank-selection` | [Main.cpp:830](../juce/src/Main.cpp#L830) |
| `--smoke-nsf-picker` | [Main.cpp:839](../juce/src/Main.cpp#L839) |
| `--smoke-cross-region-editing` | [Main.cpp:848](../juce/src/Main.cpp#L848) |
| `--smoke-hamood-persistence` | [Main.cpp:848](../juce/src/Main.cpp#L848) |
| `--smoke-mou-to-jie` | [Main.cpp:855](../juce/src/Main.cpp#L855) |
| `--smoke-oto-overlap` | [Main.cpp:861](../juce/src/Main.cpp#L861) |
| `--smoke-modeless-oto` | [Main.cpp:867](../juce/src/Main.cpp#L867) |
| `--smoke-piano-transport` | [Main.cpp:872](../juce/src/Main.cpp#L872) |
| `--smoke-track-reorder` | [Main.cpp:879](../juce/src/Main.cpp#L879) |
| `--smoke-decibel-input` | [Main.cpp:885](../juce/src/Main.cpp#L885) |
| `--smoke-oto-continuity` | [Main.cpp:891](../juce/src/Main.cpp#L891) |
| `--smoke-native-envelope` | [Main.cpp:896](../juce/src/Main.cpp#L896) |
| `--smoke-cvvc-layout-cache` | [Main.cpp:901](../juce/src/Main.cpp#L901) |
| `--smoke-auto-crossfade` | [Main.cpp:906](../juce/src/Main.cpp#L906) |
| `--smoke-pitch-point-selection` | [Main.cpp:911](../juce/src/Main.cpp#L911) |
| `--smoke-point-group` | [Main.cpp:916](../juce/src/Main.cpp#L916) |
| `--smoke-envelope-point-selection` | [Main.cpp:921](../juce/src/Main.cpp#L921) |
| `--smoke-native-linked-envelope` | [Main.cpp:926](../juce/src/Main.cpp#L926) |
| `--smoke-advanced-envelope` | [Main.cpp:931](../juce/src/Main.cpp#L931) |
| `--smoke-advanced-envelope-panel` | [Main.cpp:931](../juce/src/Main.cpp#L931) |
| `--smoke-playback-priority` | [Main.cpp:936](../juce/src/Main.cpp#L936) |
| `--smoke-ust-fidelity` | [Main.cpp:937](../juce/src/Main.cpp#L937) |
| `--smoke-project-safety` | [Main.cpp:938](../juce/src/Main.cpp#L938) |
| `--smoke-normal-display` | [Main.cpp:946](../juce/src/Main.cpp#L946) |
| `--smoke-note-hints` | [Main.cpp:954](../juce/src/Main.cpp#L954) |
| `--smoke-clip-editor-scope` | [Main.cpp:962](../juce/src/Main.cpp#L962) |
| `--smoke-empty-tuning-clip` | [Main.cpp:970](../juce/src/Main.cpp#L970) |
| `--smoke-track-gain-envelope` | [Main.cpp:978](../juce/src/Main.cpp#L978) |
| `--smoke-native-rendered-waveform` | [Main.cpp:986](../juce/src/Main.cpp#L986) |
| `--smoke-native-note-move` | [Main.cpp:993](../juce/src/Main.cpp#L993) |
| `--smoke-native-source-pitch-restore` | [Main.cpp:999](../juce/src/Main.cpp#L999) |
| `--smoke-native-audio-overlap-focus` | [Main.cpp:1005](../juce/src/Main.cpp#L1005) |
| `--smoke-native-audio-overlap` | [Main.cpp:1011](../juce/src/Main.cpp#L1011) |
| `--smoke-native-linked-pitch` | [Main.cpp:1017](../juce/src/Main.cpp#L1017) |
| `--smoke-source-breathiness` | [Main.cpp:1024](../juce/src/Main.cpp#L1024) |
| `--smoke-native-unpitched-regions` | [Main.cpp:1031](../juce/src/Main.cpp#L1031) |
| `--smoke-native-consonant-display` | [Main.cpp:1039](../juce/src/Main.cpp#L1039) |
| `--smoke-native-pitch-visibility` | [Main.cpp:1046](../juce/src/Main.cpp#L1046) |
| `--smoke-native-audio-link` | [Main.cpp:1052](../juce/src/Main.cpp#L1052) |
| `--smoke-native-note-join` | [Main.cpp:1058](../juce/src/Main.cpp#L1058) |
| `--smoke-native-trim-expand` | [Main.cpp:1063](../juce/src/Main.cpp#L1063) |
| `--smoke-native-audio-trim` | [Main.cpp:1068](../juce/src/Main.cpp#L1068) |
| `--smoke-native-audio-disconnect` | [Main.cpp:1074](../juce/src/Main.cpp#L1074) |
| `--smoke-native-note-copy-paste` | [Main.cpp:1080](../juce/src/Main.cpp#L1080) |
| `--smoke-native-import-performance` | [Main.cpp:1087](../juce/src/Main.cpp#L1087) |
| `--smoke-native-source-pitch` | [Main.cpp:1094](../juce/src/Main.cpp#L1094) |
| `--smoke-native-waveform-preview` | [Main.cpp:1102](../juce/src/Main.cpp#L1102) |
| `--smoke-timeline-notes` | [Main.cpp:1110](../juce/src/Main.cpp#L1110) |
| `--smoke-clip-gain-knob` | [Main.cpp:1118](../juce/src/Main.cpp#L1118) |
| `--smoke-clip-merge` | [Main.cpp:1126](../juce/src/Main.cpp#L1126) |
| `--smoke-timeline-selection` | [Main.cpp:1134](../juce/src/Main.cpp#L1134) |
| `--smoke-clip-trim` | [Main.cpp:1142](../juce/src/Main.cpp#L1142) |
| `--smoke-clip-split` | [Main.cpp:1151](../juce/src/Main.cpp#L1151) |
| `--smoke-independent-zoom` | [Main.cpp:1160](../juce/src/Main.cpp#L1160) |
| `--smoke-scroll-zoom` | [Main.cpp:1169](../juce/src/Main.cpp#L1169) |
| `--preview-note-dance` | [Main.cpp:1178](../juce/src/Main.cpp#L1178) |
| `--smoke-tempo-change` | [Main.cpp:1186](../juce/src/Main.cpp#L1186) |
| `--smoke-timeline-pit` | [Main.cpp:1195](../juce/src/Main.cpp#L1195) |
| `--mcp-live` | [Main.cpp:1201](../juce/src/Main.cpp#L1201) |
| `--mcp-list-sessions` | [Main.cpp:1201](../juce/src/Main.cpp#L1201) |
| `--session` | [Main.cpp:1204](../juce/src/Main.cpp#L1204) |
| `--smoke-hifisampler-envelope` | [Main.cpp:1209](../juce/src/Main.cpp#L1209) |
| `--smoke-nsf-project-note` | [Main.cpp:1215](../juce/src/Main.cpp#L1215) |
| `--mcp` | [Main.cpp:1221](../juce/src/Main.cpp#L1221) |
| `--roots` | [Main.cpp:1223](../juce/src/Main.cpp#L1223) |
| `--smoke-oto-playback` | [Main.cpp:1235](../juce/src/Main.cpp#L1235) |
| `--smoke-oto-editor` | [Main.cpp:1435](../juce/src/Main.cpp#L1435) |
| `--click` | [Main.cpp:1546](../juce/src/Main.cpp#L1546) |
| `--smoke-splice-envelope` | [Main.cpp:1558](../juce/src/Main.cpp#L1558) |
| `--smoke-dialog-enter` | [Main.cpp:1668](../juce/src/Main.cpp#L1668) |
| `--smoke-source-edit-view` | [Main.cpp:1726](../juce/src/Main.cpp#L1726) |
| `--smoke-consonant-hold` | [Main.cpp:1788](../juce/src/Main.cpp#L1788) |
| `--smoke-mou-panel-counts` | [Main.cpp:2008](../juce/src/Main.cpp#L2008) |
| `--smoke-mou-consonant-ticks` | [Main.cpp:2099](../juce/src/Main.cpp#L2099) |
| `--smoke-mou-boundary-drag` | [Main.cpp:2252](../juce/src/Main.cpp#L2252) |
| `--smoke-prefix-map` | [Main.cpp:2341](../juce/src/Main.cpp#L2341) |
| `--repair-mou-oto` | [Main.cpp:2382](../juce/src/Main.cpp#L2382) |
| `--smoke-mou-oto-file` | [Main.cpp:2403](../juce/src/Main.cpp#L2403) |
| `--smoke-utau-mode` | [Main.cpp:2602](../juce/src/Main.cpp#L2602) |
| `--smoke-lane-marquee` | [Main.cpp:2742](../juce/src/Main.cpp#L2742) |
| `--smoke-flag-lane` | [Main.cpp:2843](../juce/src/Main.cpp#L2843) |
| `--smoke-keyboard-labels` | [Main.cpp:3704](../juce/src/Main.cpp#L3704) |
| `--smoke-lane-labels` | [Main.cpp:3758](../juce/src/Main.cpp#L3758) |
| `--smoke-export-targets` | [Main.cpp:3786](../juce/src/Main.cpp#L3786) |
| `--smoke-audition-position` | [Main.cpp:3892](../juce/src/Main.cpp#L3892) |
| `--smoke-lyric-envelope` | [Main.cpp:3934](../juce/src/Main.cpp#L3934) |
| `--smoke-lyric-timing` | [Main.cpp:4006](../juce/src/Main.cpp#L4006) |
| `--smoke-vibrato-bake` | [Main.cpp:4105](../juce/src/Main.cpp#L4105) |
| `--smoke-mou-note-regions` | [Main.cpp:4218](../juce/src/Main.cpp#L4218) |
| `--smoke-region-guides` | [Main.cpp:4504](../juce/src/Main.cpp#L4504) |
| `--smoke-rest-playback` | [Main.cpp:4578](../juce/src/Main.cpp#L4578) |
| `--smoke-rest-lyric` | [Main.cpp:4674](../juce/src/Main.cpp#L4674) |
| `--smoke-missing-alias-piano` | [Main.cpp:4837](../juce/src/Main.cpp#L4837) |
| `--smoke-flag-curve-mode` | [Main.cpp:4964](../juce/src/Main.cpp#L4964) |
| `--smoke-flag-point-value` | [Main.cpp:5149](../juce/src/Main.cpp#L5149) |
| `--smoke-flag-overlap` | [Main.cpp:5239](../juce/src/Main.cpp#L5239) |
| `--smoke-load-timing` | [Main.cpp:5354](../juce/src/Main.cpp#L5354) |
| `--smoke-flatten-pitch-line` | [Main.cpp:5467](../juce/src/Main.cpp#L5467) |
| `--smoke-note-across-tracks` | [Main.cpp:5627](../juce/src/Main.cpp#L5627) |
| `--smoke-clip-across-tracks` | [Main.cpp:5783](../juce/src/Main.cpp#L5783) |
| `--smoke-reference-track` | [Main.cpp:5932](../juce/src/Main.cpp#L5932) |
| `--smoke-native-pitch-seam` | [Main.cpp:6080](../juce/src/Main.cpp#L6080) |
| `--smoke-native-pitch-points` | [Main.cpp:6152](../juce/src/Main.cpp#L6152) |
| `--smoke-paste-at-pointer` | [Main.cpp:6376](../juce/src/Main.cpp#L6376) |
| `--smoke-note-menu-split` | [Main.cpp:6519](../juce/src/Main.cpp#L6519) |
| `--smoke-lyric-gate` | [Main.cpp:6701](../juce/src/Main.cpp#L6701) |
| `--smoke-lyric-tab` | [Main.cpp:6798](../juce/src/Main.cpp#L6798) |
| `--smoke-syllable-cuts` | [Main.cpp:6998](../juce/src/Main.cpp#L6998) |
| `--smoke-single-syllable` | [Main.cpp:7180](../juce/src/Main.cpp#L7180) |
| `--smoke-deleted-notes-silent` | [Main.cpp:7287](../juce/src/Main.cpp#L7287) |
| `--smoke-delete-clip` | [Main.cpp:7434](../juce/src/Main.cpp#L7434) |
| `--smoke-draw-unit` | [Main.cpp:7531](../juce/src/Main.cpp#L7531) |
| `--smoke-new-track-draw` | [Main.cpp:7625](../juce/src/Main.cpp#L7625) |
| `--smoke-draw-drag` | [Main.cpp:7734](../juce/src/Main.cpp#L7734) |
| `--smoke-draw-overlap` | [Main.cpp:7906](../juce/src/Main.cpp#L7906) |
| `--smoke-dropdown-arrow` | [Main.cpp:8092](../juce/src/Main.cpp#L8092) |
| `--smoke-pitch-line` | [Main.cpp:8183](../juce/src/Main.cpp#L8183) |
| `--smoke-view-menu` | [Main.cpp:8474](../juce/src/Main.cpp#L8474) |
| `--inspect-melodyne-tracks` | [Main.cpp:8587](../juce/src/Main.cpp#L8587) |
| `--smoke-melodyne-provider` | [Main.cpp:8598](../juce/src/Main.cpp#L8598) |
| `--probe-melodyne-vst3-instance` | [Main.cpp:8623](../juce/src/Main.cpp#L8623) |
| `--smoke-native-timing` | [Main.cpp:8635](../juce/src/Main.cpp#L8635) |
| `--smoke-integrated-ui` | [Main.cpp:8692](../juce/src/Main.cpp#L8692) |
| `--smoke-integrated` | [Main.cpp:8701](../juce/src/Main.cpp#L8701) |
| `--smoke-native-hjm` | [Main.cpp:8707](../juce/src/Main.cpp#L8707) |
| `--smoke-batch-lyrics` | [Main.cpp:8774](../juce/src/Main.cpp#L8774) |
| `--smoke-envelope-lanes` | [Main.cpp:8866](../juce/src/Main.cpp#L8866) |
| `--smoke-wheel` | [Main.cpp:8925](../juce/src/Main.cpp#L8925) |
| `--smoke-waveform-toggle` | [Main.cpp:8966](../juce/src/Main.cpp#L8966) |
| `--smoke-envelope-base` | [Main.cpp:9076](../juce/src/Main.cpp#L9076) |
| `--smoke-amplitude-waveform` | [Main.cpp:9397](../juce/src/Main.cpp#L9397) |
| `--smoke-vibrato-end` | [Main.cpp:9855](../juce/src/Main.cpp#L9855) |
| `--smoke-hanzi-pinyin` | [Main.cpp:10115](../juce/src/Main.cpp#L10115) |
| `--smoke-shared-pitch-line` | [Main.cpp:10254](../juce/src/Main.cpp#L10254) |
| `--smoke-waveform-alignment` | [Main.cpp:10912](../juce/src/Main.cpp#L10912) |
| `--smoke-note-waveform` | [Main.cpp:11111](../juce/src/Main.cpp#L11111) |
| `--smoke-utau-waveform` | [Main.cpp:11417](../juce/src/Main.cpp#L11417) |
| `--smoke-voicebank-index` | [Main.cpp:11591](../juce/src/Main.cpp#L11591) |
| `--smoke-ust-pitch` | [Main.cpp:12057](../juce/src/Main.cpp#L12057) |
| `--low` | [Main.cpp:12362](../juce/src/Main.cpp#L12362) |
| `--smoke-hf-daemon-owner` | [Main.cpp:12481](../juce/src/Main.cpp#L12481) |
| `--quit` | [Main.cpp:12489](../juce/src/Main.cpp#L12489) |
| `--smoke-hf-prewarm` | [Main.cpp:12494](../juce/src/Main.cpp#L12494) |
| `--smoke-ust-vibrato` | [Main.cpp:12627](../juce/src/Main.cpp#L12627) |
| `--smoke-mcp-roots` | [Main.cpp:12872](../juce/src/Main.cpp#L12872) |
| `--smoke-mcp-schema` | [Main.cpp:12979](../juce/src/Main.cpp#L12979) |
| `--smoke-dragged-transition` | [Main.cpp:13144](../juce/src/Main.cpp#L13144) |
| `--smoke-forward-bend` | [Main.cpp:13367](../juce/src/Main.cpp#L13367) |
| `--smoke-midi-track-import` | [Main.cpp:13539](../juce/src/Main.cpp#L13539) |
| `--smoke-midi-export` | [Main.cpp:13954](../juce/src/Main.cpp#L13954) |
| `--sounding` | [Main.cpp:14079](../juce/src/Main.cpp#L14079) |
| `--still` | [Main.cpp:14211](../juce/src/Main.cpp#L14211) |
| `--smoke-ust-import` | [Main.cpp:14247](../juce/src/Main.cpp#L14247) |
| `--smoke-ust-encoding` | [Main.cpp:14388](../juce/src/Main.cpp#L14388) |
| `--smoke-ust` | [Main.cpp:14540](../juce/src/Main.cpp#L14540) |
| `--smoke-active-resampler` | [Main.cpp:14745](../juce/src/Main.cpp#L14745) |
| `--smoke-bundled-resampler` | [Main.cpp:14790](../juce/src/Main.cpp#L14790) |
| `--smoke-anchor-frequency` | [Main.cpp:14848](../juce/src/Main.cpp#L14848) |
| `--smoke-note-oto` | [Main.cpp:15034](../juce/src/Main.cpp#L15034) |
| `--smoke-prefix-jie` | [Main.cpp:15464](../juce/src/Main.cpp#L15464) |
| `--smoke-prefix-note` | [Main.cpp:15618](../juce/src/Main.cpp#L15618) |
| `--smoke-utau-overlap` | [Main.cpp:15975](../juce/src/Main.cpp#L15975) |
| `--smoke-note-stp` | [Main.cpp:16290](../juce/src/Main.cpp#L16290) |
| `--smoke-track-toggle-tips` | [Main.cpp:16529](../juce/src/Main.cpp#L16529) |
| `--smoke-track-area-menu` | [Main.cpp:16776](../juce/src/Main.cpp#L16776) |
| `--smoke-stretch-items` | [Main.cpp:16877](../juce/src/Main.cpp#L16877) |
| `--smoke-render-order` | [Main.cpp:16929](../juce/src/Main.cpp#L16929) |
| `--smoke-llsm2-length` | [Main.cpp:17061](../juce/src/Main.cpp#L17061) |
| `--smoke-glide-seam` | [Main.cpp:17113](../juce/src/Main.cpp#L17113) |
| `--smoke-native-envelope-base` | [Main.cpp:17177](../juce/src/Main.cpp#L17177) |
| `--smoke-nsf-utau-phrase` | [Main.cpp:17261](../juce/src/Main.cpp#L17261) |
| `--smoke-nsf-utau-synth` | [Main.cpp:17293](../juce/src/Main.cpp#L17293) |
| `--smoke-nsf-utau-f0` | [Main.cpp:17341](../juce/src/Main.cpp#L17341) |
| `--smoke-nsf-utau-mix` | [Main.cpp:17386](../juce/src/Main.cpp#L17386) |
| `--smoke-nsf-utau-plan` | [Main.cpp:17451](../juce/src/Main.cpp#L17451) |
| `--smoke-render-capability` | [Main.cpp:17508](../juce/src/Main.cpp#L17508) |
| `--smoke-oto-hjm-discipline` | [Main.cpp:17559](../juce/src/Main.cpp#L17559) |
| `--smoke-asset-register-dir` | [Main.cpp:17629](../juce/src/Main.cpp#L17629) |
| `--smoke-oto-audio` | [Main.cpp:17673](../juce/src/Main.cpp#L17673) |
| `--smoke-asset-audio` | [Main.cpp:17686](../juce/src/Main.cpp#L17686) |
| `--smoke-project-voicebank-assets` | [Main.cpp:17700](../juce/src/Main.cpp#L17700) |
| `--smoke-asset-search` | [Main.cpp:17706](../juce/src/Main.cpp#L17706) |
| `--smoke-asset-manager` | [Main.cpp:17715](../juce/src/Main.cpp#L17715) |
| `--smoke-gap-menu` | [Main.cpp:17840](../juce/src/Main.cpp#L17840) |
| `--smoke-insert-gap` | [Main.cpp:17959](../juce/src/Main.cpp#L17959) |
| `--smoke-ripple-delete` | [Main.cpp:18075](../juce/src/Main.cpp#L18075) |
| `--smoke-paste-notes` | [Main.cpp:18164](../juce/src/Main.cpp#L18164) |
| `--smoke-consonant-edge` | [Main.cpp:18340](../juce/src/Main.cpp#L18340) |
| `--smoke-consonant-reset` | [Main.cpp:18496](../juce/src/Main.cpp#L18496) |
| `--smoke-region-split` | [Main.cpp:18603](../juce/src/Main.cpp#L18603) |
| `--smoke-vibrato-presets` | [Main.cpp:18695](../juce/src/Main.cpp#L18695) |
| `--smoke-follow-rule` | [Main.cpp:18765](../juce/src/Main.cpp#L18765) |
| `--smoke-play-until` | [Main.cpp:18837](../juce/src/Main.cpp#L18837) |
| `--smoke-flags-field` | [Main.cpp:18896](../juce/src/Main.cpp#L18896) |
| `--smoke-vertical-drag` | [Main.cpp:18961](../juce/src/Main.cpp#L18961) |
| `--smoke-scroll-sync` | [Main.cpp:19054](../juce/src/Main.cpp#L19054) |
| `--smoke-edit-follow` | [Main.cpp:19135](../juce/src/Main.cpp#L19135) |
| `--smoke-note-drag-threshold` | [Main.cpp:19235](../juce/src/Main.cpp#L19235) |
| `--smoke-roll-drag` | [Main.cpp:19310](../juce/src/Main.cpp#L19310) |
| `--smoke-playhead-band` | [Main.cpp:19441](../juce/src/Main.cpp#L19441) |
| `--smoke-roll-cull` | [Main.cpp:19494](../juce/src/Main.cpp#L19494) |
| `--smoke-roll-timing` | [Main.cpp:19567](../juce/src/Main.cpp#L19567) |
| `--smoke-envelope-shapes` | [Main.cpp:19613](../juce/src/Main.cpp#L19613) |
| `--smoke-envelope-presets` | [Main.cpp:19930](../juce/src/Main.cpp#L19930) |
| `--smoke-piano-roll` | [Main.cpp:19946](../juce/src/Main.cpp#L19946) |
| `--render-project` | [Main.cpp:19994](../juce/src/Main.cpp#L19994) |
| `--inspect-project` | [Main.cpp:20123](../juce/src/Main.cpp#L20123) |
| `--inspect-settings` | [Main.cpp:20159](../juce/src/Main.cpp#L20159) |
| `--smoke-oto-spectrum` | [Main.cpp:20186](../juce/src/Main.cpp#L20186) |
| `--smoke-game-defaults` | [Main.cpp:20193](../juce/src/Main.cpp#L20193) |
| `--inspect-analysis` | [Main.cpp:20199](../juce/src/Main.cpp#L20199) |
| `--inspect-fcpe` | [Main.cpp:20221](../juce/src/Main.cpp#L20221) |
| `--smoke-export` | [Main.cpp:20247](../juce/src/Main.cpp#L20247) |
| `--smoke-mld5` | [Main.cpp:20277](../juce/src/Main.cpp#L20277) |
| `--smoke-utau-selection` | [Main.cpp:20370](../juce/src/Main.cpp#L20370) |
| `--smoke-utau-voicebank` | [Main.cpp:20484](../juce/src/Main.cpp#L20484) |
| `--smoke-utau` | [Main.cpp:20567](../juce/src/Main.cpp#L20567) |
| `--inspect-midi` | [Main.cpp:20653](../juce/src/Main.cpp#L20653) |
| `--inspect-audio` | [Main.cpp:20676](../juce/src/Main.cpp#L20676) |
| `--inspect-mpd` | [Main.cpp:20711](../juce/src/Main.cpp#L20711) |

## 范围与静态体量

扫描自有 `juce/src` 非测试 `.h/.cpp/.inc`：182 个文件，111,066 行（包括注释和空行）。不含 juce/third_party、tests 和名称带 Smoke 的包含单元。

| 文件 | 行数 |
| --- | --- |
| [juce/src/Main.cpp](../juce/src/Main.cpp) | 20,879 |
| [juce/src/PianoRollComponent.cpp](../juce/src/PianoRollComponent.cpp) | 11,623 |
| [juce/src/ProjectModel.cpp](../juce/src/ProjectModel.cpp) | 11,260 |
| [juce/src/MainComponent.cpp](../juce/src/MainComponent.cpp) | 6,790 |
| [juce/src/AudioEngine.cpp](../juce/src/AudioEngine.cpp) | 3,832 |
| [juce/src/OtoWaveformEditorComponent.cpp](../juce/src/OtoWaveformEditorComponent.cpp) | 2,891 |
| [juce/src/backend/UtauRenderer.cpp](../juce/src/backend/UtauRenderer.cpp) | 2,844 |
| [juce/src/PinyinTable.inc](../juce/src/PinyinTable.inc) | 2,709 |
| [juce/src/backend/McpServer.cpp](../juce/src/backend/McpServer.cpp) | 2,123 |
| [juce/src/backend/NsfHifiganRenderer.cpp](../juce/src/backend/NsfHifiganRenderer.cpp) | 2,102 |
| [juce/src/SampleSettings.cpp](../juce/src/SampleSettings.cpp) | 2,022 |
| [juce/src/AssetManagerComponent.cpp](../juce/src/AssetManagerComponent.cpp) | 1,875 |
| [juce/src/PianoRollComponent.h](../juce/src/PianoRollComponent.h) | 1,581 |
| [juce/src/ProjectModel.h](../juce/src/ProjectModel.h) | 1,576 |
| [juce/src/TimelineComponent.cpp](../juce/src/TimelineComponent.cpp) | 1,191 |

完整扫描指纹及接口条目：[interface-audit-inventory.json](interface-audit-inventory.json)。行数用于导航和识别职责集中，不是性能或质量评分。
