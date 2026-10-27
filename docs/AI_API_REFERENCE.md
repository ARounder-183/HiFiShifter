# 实际接口声明清单（自动生成）

核对日期：2026-10-27。配套解读：[功能主入口与架构审计](AI_INTERFACE_ARCHITECTURE.md)。

这是源文件导航，不是承诺稳定的公共 SDK。仅枚举下列 11 个边界类的公开方法，不把 private 函数、UI 回调、纯计算辅助函数或每个第三方函数算作产品接口。重载各列一行；诊断接口单独标记。源码行号为生成时位置。

生成命令：`python tools/generate-interface-reference.py`。词法提取不是 C++ 编译器，不能用条目数证明所有业务路径正确；新增边界类时需更新脚本 HEADERS。

## ProjectModel（198 个声明）

| 实际签名 | 类型 | 定义入口 |
| --- | --- | --- |
| `ProjectModel()` | 生命周期 | [ProjectModel.h:931](../juce/src/ProjectModel.h#L931) |
| `ProjectData snapshot() const` | 业务／适配 | [ProjectModel.h:933](../juce/src/ProjectModel.h#L933) |
| `std::uint64_t revisionNumber() const` | 业务／适配 | [ProjectModel.h:934](../juce/src/ProjectModel.h#L934) |
| `void replace(ProjectData replacement)` | 业务／适配 | [ProjectModel.h:935](../juce/src/ProjectModel.h#L935) |
| `void resetDocument(ProjectData replacement = {})` | 业务／适配 | [ProjectModel.h:937](../juce/src/ProjectModel.h#L937) |
| `void clear()` | 业务／适配 | [ProjectModel.h:938](../juce/src/ProjectModel.h#L938) |
| `void setAssetManagerVisible(bool visible)` | 业务／适配 | [ProjectModel.h:940](../juce/src/ProjectModel.h#L940) |
| `void setMaterialBrowserState(const juce::String& state)` | 业务／适配 | [ProjectModel.h:941](../juce/src/ProjectModel.h#L941) |
| `bool setHamoodState(const juce::String& state, juce::String& error)` | 业务／适配 | [ProjectModel.h:942](../juce/src/ProjectModel.h#L942) |
| `bool undo()` | 业务／适配 | [ProjectModel.h:943](../juce/src/ProjectModel.h#L943) |
| `bool redo()` | 业务／适配 | [ProjectModel.h:944](../juce/src/ProjectModel.h#L944) |
| `bool canUndo() const` | 业务／适配 | [ProjectModel.h:945](../juce/src/ProjectModel.h#L945) |
| `bool canRedo() const` | 业务／适配 | [ProjectModel.h:946](../juce/src/ProjectModel.h#L946) |
| `juce::String addAudioFile(const juce::File& file, double durationSeconds, double startSeconds = 0.0, const juce::String& targetTrackId = {}, const VoicebankOtoEntry* oto = nullptr, UtauMode otoMode = UtauMode::classic)` | 业务／适配 | [ProjectModel.h:948](../juce/src/ProjectModel.h#L948) |
| `juce::String addMaterialPronunciations( const juce::File& audio, const std::vector<SampleRegionSetting>& completeRows, const std::vector<juce::String>& selectedRegionIds, double absoluteStart, const juce::String& targetTrackId, juce::String& error, const juce::String& chosenAlias = {})` | 业务／适配 | [ProjectModel.h:960](../juce/src/ProjectModel.h#L960) |
| `juce::String addTypedTrack(TrackType type = TrackType::hachi, const juce::String& name = {})` | 业务／适配 | [ProjectModel.h:965](../juce/src/ProjectModel.h#L965) |
| `juce::String addTrack(const juce::String& name, bool compose = true, bool referenceOnly = false, bool accompaniment = false, TrackType type = TrackType::hachi)` | 业务／适配 | [ProjectModel.h:967](../juce/src/ProjectModel.h#L967) |
| `void setTrackReferenceOnly(const juce::String& trackId, bool referenceOnly)` | 业务／适配 | [ProjectModel.h:970](../juce/src/ProjectModel.h#L970) |
| `void setTrackName(const juce::String& trackId, const juce::String& name)` | 业务／适配 | [ProjectModel.h:971](../juce/src/ProjectModel.h#L971) |
| `bool setClipNotesIfEmpty(const juce::String& clipId, std::vector<NoteData> notes)` | 业务／适配 | [ProjectModel.h:972](../juce/src/ProjectModel.h#L972) |
| `std::size_t markNativeUnpitchedRegions(const juce::String& clipId)` | 业务／适配 | [ProjectModel.h:973](../juce/src/ProjectModel.h#L973) |
| `bool setNativeTrimSourceAnalysis(const juce::File& file, const std::vector<NoteData>& sourceNotes, const juce::String& expectedSourceFingerprint = {})` | 业务／适配 | [ProjectModel.h:974](../juce/src/ProjectModel.h#L974) |
| `bool setClipAudioAnalysis(const juce::String& clipId, std::vector<NoteData> notes, const ClipData& importedClip = ClipData{})` | 业务／适配 | [ProjectModel.h:976](../juce/src/ProjectModel.h#L976) |
| `bool addMidiFile(const juce::File& file, juce::String& error)` | 业务／适配 | [ProjectModel.h:978](../juce/src/ProjectModel.h#L978) |
| `static std::vector<MidiTrackChoice> midiTrackChoices(const juce::File& file, juce::String& error)` | 业务／适配 | [ProjectModel.h:988](../juce/src/ProjectModel.h#L988) |
| `juce::String addMidiTrack(const juce::File& file, int trackIndex, juce::String& error, const juce::String& targetTrackId = {}, double startSeconds = 0.0)` | 业务／适配 | [ProjectModel.h:997](../juce/src/ProjectModel.h#L997) |
| `static bool writeMidiFile(const ProjectData& data, const juce::File& file, juce::String& error, const juce::String& trackId = {}, bool includeMuted = false)` | 业务／适配 | [ProjectModel.h:1006](../juce/src/ProjectModel.h#L1006) |
| `bool addUstFile(const juce::File& file, juce::String& error, juce::StringArray& warnings, UstImportMode mode = UstImportMode::addTrack, juce::String* importedTrackId = nullptr, int encoding = 0, TrackType destinationType = TrackType::hachi)` | 业务／适配 | [ProjectModel.h:1024](../juce/src/ProjectModel.h#L1024) |
| `bool exportUst(const juce::File& file, const juce::String& trackId, juce::String& error, juce::StringArray& warnings, int encoding = 0) const` | 业务／适配 | [ProjectModel.h:1029](../juce/src/ProjectModel.h#L1029) |
| `void setTempo(double bpm, int numerator, int denominator = 4)` | 业务／适配 | [ProjectModel.h:1031](../juce/src/ProjectModel.h#L1031) |
| `void setTempoChange(double quarterPosition, double bpm, bool synchronizeNotes = true)` | 业务／适配 | [ProjectModel.h:1034](../juce/src/ProjectModel.h#L1034) |
| `juce::int64 contentFingerprint() const` | 业务／适配 | [ProjectModel.h:1038](../juce/src/ProjectModel.h#L1038) |
| `static juce::String registrationMusicSourceFingerprint(const ProjectData&)` | 业务／适配 | [ProjectModel.h:1040](../juce/src/ProjectModel.h#L1040) |
| `bool adoptRegisteredMaterialBindings(std::uint64_t expectedRevision, const juce::String& expectedFingerprint, const std::vector<ProjectMaterialRegistrationOwnerBinding>& owners, const std::vector<ProjectMaterialRegistrationNoteBinding>& notes, juce::String& error)` | 业务／适配 | [ProjectModel.h:1041](../juce/src/ProjectModel.h#L1041) |
| `void setGridDivision(const juce::String& division)` | 业务／适配 | [ProjectModel.h:1046](../juce/src/ProjectModel.h#L1046) |
| `void setNoteEditDivision(int division)` | 业务／适配 | [ProjectModel.h:1047](../juce/src/ProjectModel.h#L1047) |
| `void setBaseScale(const juce::String& scale)` | 业务／适配 | [ProjectModel.h:1048](../juce/src/ProjectModel.h#L1048) |
| `void setTrackCompose(const juce::String& trackId, bool enabled)` | 业务／适配 | [ProjectModel.h:1049](../juce/src/ProjectModel.h#L1049) |
| `void setTrackMuted(const juce::String& trackId, bool muted)` | 业务／适配 | [ProjectModel.h:1050](../juce/src/ProjectModel.h#L1050) |
| `void setTrackSolo(const juce::String& trackId, bool solo)` | 业务／适配 | [ProjectModel.h:1051](../juce/src/ProjectModel.h#L1051) |
| `void setTrackVolume(const juce::String& trackId, float volume)` | 业务／适配 | [ProjectModel.h:1052](../juce/src/ProjectModel.h#L1052) |
| `void setClipGainEnvelope(const juce::String& clipId, std::vector<GainEnvelopePoint> points)` | 业务／适配 | [ProjectModel.h:1053](../juce/src/ProjectModel.h#L1053) |
| `void setTrackPan(const juce::String& trackId, float pan)` | 业务／适配 | [ProjectModel.h:1054](../juce/src/ProjectModel.h#L1054) |
| `void setTrackSmoothOverlaps(const juce::String& trackId, bool enabled)` | 业务／适配 | [ProjectModel.h:1055](../juce/src/ProjectModel.h#L1055) |
| `void setTrackNormalizeVolume(const juce::String& trackId, bool enabled)` | 业务／适配 | [ProjectModel.h:1056](../juce/src/ProjectModel.h#L1056) |
| `void setTrackNsfSmoothPitchTransitions(const juce::String& trackId, bool enabled)` | 业务／适配 | [ProjectModel.h:1057](../juce/src/ProjectModel.h#L1057) |
| `void setTrackNsfNoiseProtection(const juce::String& trackId, bool enabled)` | 业务／适配 | [ProjectModel.h:1058](../juce/src/ProjectModel.h#L1058) |
| `void setTrackVoicebankDirectory(const juce::String& trackId, const juce::File& directory)` | 业务／适配 | [ProjectModel.h:1059](../juce/src/ProjectModel.h#L1059) |
| `void setTrackUtauConsonantVelocity(const juce::String& trackId, int velocity)` | 业务／适配 | [ProjectModel.h:1061](../juce/src/ProjectModel.h#L1061) |
| `void setTrackUtauGlobalFlags(const juce::String& trackId, const juce::String& flags)` | 业务／适配 | [ProjectModel.h:1062](../juce/src/ProjectModel.h#L1062) |
| `void setUtauMode(UtauMode mode)` | 业务／适配 | [ProjectModel.h:1063](../juce/src/ProjectModel.h#L1063) |
| `void setTrackUtauMode(const juce::String& trackId, UtauMode mode)` | 业务／适配 | [ProjectModel.h:1064](../juce/src/ProjectModel.h#L1064) |
| `void setTrackChineseCvvc(const juce::String& trackId, bool enabled)` | 业务／适配 | [ProjectModel.h:1065](../juce/src/ProjectModel.h#L1065) |
| `void setTrackPianoPreview(const juce::String& trackId, bool enabled)` | 业务／适配 | [ProjectModel.h:1066](../juce/src/ProjectModel.h#L1066) |
| `void setTrackCvvcCurveCrossfade(const juce::String& trackId, bool curved)` | 业务／适配 | [ProjectModel.h:1067](../juce/src/ProjectModel.h#L1067) |
| `void setTrackOutputEngine(const juce::String& trackId, UtauOutputEngine engine, const juce::File& resampler = {}, const juce::File& wavtool = {})` | 业务／适配 | [ProjectModel.h:1068](../juce/src/ProjectModel.h#L1068) |
| `void setNotesVibrato(const std::vector<juce::String>& noteIds, const NoteData& parameters, bool enabled, std::optional<bool> realLine = std::nullopt)` | 业务／适配 | [ProjectModel.h:1070](../juce/src/ProjectModel.h#L1070) |
| `void setNotesVibratoRealLine(const std::vector<juce::String>& noteIds, bool enabled)` | 业务／适配 | [ProjectModel.h:1073](../juce/src/ProjectModel.h#L1073) |
| `bool bakeNoteVibratoIntoPitch(const juce::String& noteId)` | 业务／适配 | [ProjectModel.h:1077](../juce/src/ProjectModel.h#L1077) |
| `void setNotesUtauSplice(const std::vector<juce::String>& noteIds, bool enabled)` | 业务／适配 | [ProjectModel.h:1078](../juce/src/ProjectModel.h#L1078) |
| `bool setPointCurves(std::vector<PointCurveEdit> edits, std::uint64_t expectedRevision, juce::String* error = nullptr)` | 业务／适配 | [ProjectModel.h:1105](../juce/src/ProjectModel.h#L1105) |
| `bool previewPointCurves(const std::vector<PointCurveEdit>& edits, std::uint64_t expectedRevision, PointCurvePlanPreview&, juce::String& error) const` | 业务／适配 | [ProjectModel.h:1107](../juce/src/ProjectModel.h#L1107) |
| `bool resetCvvcBoundaries(const std::vector<juce::String>& noteIds, std::uint64_t expectedRevision)` | 业务／适配 | [ProjectModel.h:1110](../juce/src/ProjectModel.h#L1110) |
| `bool setCvvcBoundary(const juce::String& noteId, int rightPart, double position, std::uint64_t expectedRevision)` | 业务／适配 | [ProjectModel.h:1111](../juce/src/ProjectModel.h#L1111) |
| `bool setCvvcEnvelopes(std::vector<CvvcEnvelopeEdit>, std::uint64_t expectedRevision)` | 业务／适配 | [ProjectModel.h:1112](../juce/src/ProjectModel.h#L1112) |
| `bool setCvvcEnvelope(const juce::String& noteId, int part, const juce::String& flag, backend::CvvcEnvelopeCurve points, std::uint64_t expectedRevision)` | 业务／适配 | [ProjectModel.h:1113](../juce/src/ProjectModel.h#L1113) |
| `void setNotesUtauFlagCurveEnabled(const std::vector<juce::String>& noteIds, bool enabled)` | 业务／适配 | [ProjectModel.h:1115](../juce/src/ProjectModel.h#L1115) |
| `bool resetNotesUtauFlagCurves(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1122](../juce/src/ProjectModel.h#L1122) |
| `bool resetNotesUtauFlagCurve(const std::vector<juce::String>& noteIds, const juce::String& flag)` | 业务／适配 | [ProjectModel.h:1124](../juce/src/ProjectModel.h#L1124) |
| `bool setNoteUtauFlagCurves(const juce::String& flag, const std::map<juce::String, std::vector<FlagCurvePoint>>& edits, std::optional<std::uint64_t> expectedRevision = std::nullopt, bool enabledOnly = true)` | 业务／适配 | [ProjectModel.h:1128](../juce/src/ProjectModel.h#L1128) |
| `bool setNoteUtauFlagCurve(const juce::String& noteId, const juce::String& flag, std::vector<FlagCurvePoint> points)` | 业务／适配 | [ProjectModel.h:1131](../juce/src/ProjectModel.h#L1131) |
| `void setNotesRegionFlags(const std::vector<juce::String>& noteIds, bool split, const juce::String& first, const juce::String& second, const juce::String& third, const juce::String& fourth)` | 业务／适配 | [ProjectModel.h:1133](../juce/src/ProjectModel.h#L1133) |
| `void setNotesUtauJieSplitCleared(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1137](../juce/src/ProjectModel.h#L1137) |
| `void setNotesUtauJieSplit(const std::vector<juce::String>& noteIds, double first, double second, double third)` | 业务／适配 | [ProjectModel.h:1138](../juce/src/ProjectModel.h#L1138) |
| `void clearNotesUtauJieSplit(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1140](../juce/src/ProjectModel.h#L1140) |
| `void setTrackEditingType(const juce::String& trackId, TrackType type, std::optional<PitchAlgorithm> algorithm = {}, std::optional<UtauMode> mode = {})` | 业务／适配 | [ProjectModel.h:1141](../juce/src/ProjectModel.h#L1141) |
| `void setTrackPitchAlgorithm(const juce::String& trackId, PitchAlgorithm algorithm)` | 业务／适配 | [ProjectModel.h:1143](../juce/src/ProjectModel.h#L1143) |
| `void setTrackStretchAlgorithm(const juce::String& trackId, StretchAlgorithm algorithm)` | 业务／适配 | [ProjectModel.h:1144](../juce/src/ProjectModel.h#L1144) |
| `void setPitchAlgorithm(PitchAlgorithm algorithm)` | 业务／适配 | [ProjectModel.h:1145](../juce/src/ProjectModel.h#L1145) |
| `void setStretchAlgorithm(StretchAlgorithm algorithm)` | 业务／适配 | [ProjectModel.h:1146](../juce/src/ProjectModel.h#L1146) |
| `void setTrackRenderOrder(const juce::String& trackId, RenderOrder order, std::optional<bool> mergeContiguous = {})` | 业务／适配 | [ProjectModel.h:1147](../juce/src/ProjectModel.h#L1147) |
| `void setRenderOrder(RenderOrder order)` | 业务／适配 | [ProjectModel.h:1149](../juce/src/ProjectModel.h#L1149) |
| `void moveClip(const juce::String& clipId, double startSeconds)` | 业务／适配 | [ProjectModel.h:1150](../juce/src/ProjectModel.h#L1150) |
| `bool canSplitClip(const juce::String& clipId, double absoluteSeconds) const` | 业务／适配 | [ProjectModel.h:1151](../juce/src/ProjectModel.h#L1151) |
| `juce::String splitClip(const juce::String& clipId, double absoluteSeconds)` | 业务／适配 | [ProjectModel.h:1154](../juce/src/ProjectModel.h#L1154) |
| `juce::String duplicateClip(const juce::String& clipId, double startSeconds = -1.0, const juce::String& targetTrackId = {})` | 业务／适配 | [ProjectModel.h:1155](../juce/src/ProjectModel.h#L1155) |
| `void resizeClip(const juce::String& clipId, double startSeconds, double durationSeconds)` | 业务／适配 | [ProjectModel.h:1158](../juce/src/ProjectModel.h#L1158) |
| `bool trimClip(const juce::String& clipId, double startSeconds, double durationSeconds)` | 业务／适配 | [ProjectModel.h:1160](../juce/src/ProjectModel.h#L1160) |
| `bool setClipEditableLength(const juce::String& clipId, double durationSeconds)` | 业务／适配 | [ProjectModel.h:1161](../juce/src/ProjectModel.h#L1161) |
| `void setClipGain(const juce::String& clipId, float gain)` | 业务／适配 | [ProjectModel.h:1162](../juce/src/ProjectModel.h#L1162) |
| `void setClipFades(const juce::String& clipId, double fadeInSeconds, double fadeOutSeconds)` | 业务／适配 | [ProjectModel.h:1163](../juce/src/ProjectModel.h#L1163) |
| `void setClipMuted(const juce::String& clipId, bool muted)` | 业务／适配 | [ProjectModel.h:1165](../juce/src/ProjectModel.h#L1165) |
| `void setClipNoteHints(const juce::String& clipId, bool enabled)` | 业务／适配 | [ProjectModel.h:1166](../juce/src/ProjectModel.h#L1166) |
| `void setClipNormalDisplay(const juce::String& clipId, bool enabled)` | 业务／适配 | [ProjectModel.h:1167](../juce/src/ProjectModel.h#L1167) |
| `void removeClip(const juce::String& clipId)` | 业务／适配 | [ProjectModel.h:1168](../juce/src/ProjectModel.h#L1168) |
| `void removeClips(const std::vector<juce::String>& clipIds)` | 业务／适配 | [ProjectModel.h:1169](../juce/src/ProjectModel.h#L1169) |
| `bool canMergeClips(const std::vector<juce::String>& clipIds) const` | 业务／适配 | [ProjectModel.h:1170](../juce/src/ProjectModel.h#L1170) |
| `juce::String mergeClips(const std::vector<juce::String>& clipIds)` | 业务／适配 | [ProjectModel.h:1171](../juce/src/ProjectModel.h#L1171) |
| `void moveClips(const std::vector<juce::String>& clipIds, double deltaSeconds, int trackDelta = 0)` | 业务／适配 | [ProjectModel.h:1172](../juce/src/ProjectModel.h#L1172) |
| `void removeTrack(const juce::String& trackId)` | 业务／适配 | [ProjectModel.h:1173](../juce/src/ProjectModel.h#L1173) |
| `bool moveTrackBefore(const juce::String& trackId, const juce::String& beforeTrackId)` | 业务／适配 | [ProjectModel.h:1175](../juce/src/ProjectModel.h#L1175) |
| `void transposeNote(const juce::String& noteId, float semitones)` | 业务／适配 | [ProjectModel.h:1176](../juce/src/ProjectModel.h#L1176) |
| `void transposeNotes(const std::vector<juce::String>& noteIds, float semitones)` | 业务／适配 | [ProjectModel.h:1177](../juce/src/ProjectModel.h#L1177) |
| `bool moveUtauNotes(const std::vector<juce::String>& noteIds, double deltaSeconds, float semitones, juce::String* error = nullptr)` | 业务／适配 | [ProjectModel.h:1181](../juce/src/ProjectModel.h#L1181) |
| `bool moveNativeNotes(const std::vector<juce::String>& noteIds, double deltaSeconds, float semitones, juce::String* error = nullptr)` | 业务／适配 | [ProjectModel.h:1185](../juce/src/ProjectModel.h#L1185) |
| `bool trimNativeNoteEdge(const juce::String& noteId, double deltaSeconds, bool leftEdge)` | 业务／适配 | [ProjectModel.h:1187](../juce/src/ProjectModel.h#L1187) |
| `bool resizeNativeNoteEdge(const juce::String& noteId, double deltaSeconds, bool leftEdge)` | 业务／适配 | [ProjectModel.h:1188](../juce/src/ProjectModel.h#L1188) |
| `bool stretchNativeNoteBoundary(const juce::String& leftId, const juce::String& rightId, double deltaSeconds, std::uint64_t expectedRevision)` | 业务／适配 | [ProjectModel.h:1189](../juce/src/ProjectModel.h#L1189) |
| `bool alignNotePreutteranceToGrid(const juce::String& noteId, double quarterStep, juce::String& error)` | 业务／适配 | [ProjectModel.h:1193](../juce/src/ProjectModel.h#L1193) |
| `void setNotesMidi(const std::vector<juce::String>& noteIds, float midiNote)` | 业务／适配 | [ProjectModel.h:1194](../juce/src/ProjectModel.h#L1194) |
| `void averageNotesMidi(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1195](../juce/src/ProjectModel.h#L1195) |
| `void quantizeNotesMidi(const std::vector<juce::String>& noteIds, float stepSemitones = 1.0f)` | 业务／适配 | [ProjectModel.h:1196](../juce/src/ProjectModel.h#L1196) |
| `std::vector<juce::String> notesOverlapping( const juce::String& clipId, double fromSeconds, double toSeconds) const` | 业务／适配 | [ProjectModel.h:1200](../juce/src/ProjectModel.h#L1200) |
| `std::vector<juce::String> insertNotes( const juce::String& clipId, const std::vector<NoteData>& noteTemplates, double absoluteStartSeconds)` | 业务／适配 | [ProjectModel.h:1202](../juce/src/ProjectModel.h#L1202) |
| `std::vector<juce::String> insertNativeAudioClips( const juce::String& trackId, const std::vector<ClipData>& templates, const std::vector<NativeConnection>& connections, double atSeconds)` | 业务／适配 | [ProjectModel.h:1205](../juce/src/ProjectModel.h#L1205) |
| `std::vector<juce::String> duplicateNotes( const std::vector<juce::String>& noteIds, const juce::String& targetClipId, double absoluteStartSeconds)` | 业务／适配 | [ProjectModel.h:1208](../juce/src/ProjectModel.h#L1208) |
| `void resizeNote(const juce::String& noteId, double newStart, double newDuration)` | 业务／适配 | [ProjectModel.h:1211](../juce/src/ProjectModel.h#L1211) |
| `juce::String splitNote(const juce::String& noteId, double localSeconds)` | 业务／适配 | [ProjectModel.h:1212](../juce/src/ProjectModel.h#L1212) |
| `juce::String mergeNotes(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1214](../juce/src/ProjectModel.h#L1214) |
| `bool canJoinNativeNotes(const juce::String&,const juce::String&) const` | 业务／适配 | [ProjectModel.h:1215](../juce/src/ProjectModel.h#L1215) |
| `juce::String joinNativeNotes(const juce::String&,const juce::String&)` | 业务／适配 | [ProjectModel.h:1216](../juce/src/ProjectModel.h#L1216) |
| `void setNotesConnection(const std::vector<juce::String>& noteIds, bool enabled)` | 业务／适配 | [ProjectModel.h:1219](../juce/src/ProjectModel.h#L1219) |
| `bool setNotesPortamento(const std::vector<juce::String>& noteIds, bool enabled, juce::String& error, double durationSeconds = 0.1)` | 业务／适配 | [ProjectModel.h:1222](../juce/src/ProjectModel.h#L1222) |
| `bool setConnectionPitchCurve(const juce::String& connectionId, std::vector<PitchCurveEditPoint> absolutePoints, juce::String& error)` | 业务／适配 | [ProjectModel.h:1228](../juce/src/ProjectModel.h#L1228) |
| `bool setConnectionPitchPoint(const juce::String& connectionId, int pointIndex, PitchCurveEditPoint absolutePoint, juce::String& error)` | 业务／适配 | [ProjectModel.h:1230](../juce/src/ProjectModel.h#L1230) |
| `bool disconnectNativeAudio(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1232](../juce/src/ProjectModel.h#L1232) |
| `bool linkNativeAudio(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1233](../juce/src/ProjectModel.h#L1233) |
| `bool setNativeAudioOverlap(const juce::String& trackId, bool enabled)` | 业务／适配 | [ProjectModel.h:1234](../juce/src/ProjectModel.h#L1234) |
| `juce::String insertPrefixNote(const juce::String& noteId, double targetOverlapSeconds)` | 业务／适配 | [ProjectModel.h:1239](../juce/src/ProjectModel.h#L1239) |
| `void setNoteModulation(const juce::String& noteId, float modulation)` | 业务／适配 | [ProjectModel.h:1241](../juce/src/ProjectModel.h#L1241) |
| `void setNoteDrift(const juce::String& noteId, float drift)` | 业务／适配 | [ProjectModel.h:1242](../juce/src/ProjectModel.h#L1242) |
| `void setNoteTension(const juce::String& noteId, float tension)` | 业务／适配 | [ProjectModel.h:1243](../juce/src/ProjectModel.h#L1243) |
| `void setNoteBreath(const juce::String& noteId, float breath)` | 业务／适配 | [ProjectModel.h:1244](../juce/src/ProjectModel.h#L1244) |
| `void setNoteFormant(const juce::String& noteId, float semitones)` | 业务／适配 | [ProjectModel.h:1245](../juce/src/ProjectModel.h#L1245) |
| `void setNoteGain(const juce::String& noteId, float gain)` | 业务／适配 | [ProjectModel.h:1246](../juce/src/ProjectModel.h#L1246) |
| `bool setNoteGains(const std::map<juce::String, float>& gains, std::optional<std::uint64_t> expectedRevision = {})` | 业务／适配 | [ProjectModel.h:1247](../juce/src/ProjectModel.h#L1247) |
| `bool setNotesTailFade(const std::vector<juce::String>& noteIds, int mode)` | 业务／适配 | [ProjectModel.h:1249](../juce/src/ProjectModel.h#L1249) |
| `bool setNotesNativeEnvelope(const std::vector<juce::String>& noteIds, const backend::NativeEnvelopeSettings& settings)` | 业务／适配 | [ProjectModel.h:1250](../juce/src/ProjectModel.h#L1250) |
| `bool setNotesTailFade(const std::vector<juce::String>& noteIds, int mode, const backend::TailFadeSettings& settings)` | 业务／适配 | [ProjectModel.h:1251](../juce/src/ProjectModel.h#L1251) |
| `void setNotesAmplitudeEnvelopeBase(const std::vector<juce::String>& noteIds, float basePercent)` | 业务／适配 | [ProjectModel.h:1253](../juce/src/ProjectModel.h#L1253) |
| `void setNoteAttack(const juce::String& noteId, double consonantSeconds, float attackSpeed)` | 业务／适配 | [ProjectModel.h:1255](../juce/src/ProjectModel.h#L1255) |
| `void setNoteAttackSpeed(const juce::String& noteId, float attackSpeed)` | 业务／适配 | [ProjectModel.h:1256](../juce/src/ProjectModel.h#L1256) |
| `void setNoteLabel(const juce::String& noteId, const juce::String& label)` | 业务／适配 | [ProjectModel.h:1257](../juce/src/ProjectModel.h#L1257) |
| `int convertTrackLyricsToPinyin(const juce::String& trackId)` | 业务／适配 | [ProjectModel.h:1261](../juce/src/ProjectModel.h#L1261) |
| `void setNoteLabels(const std::vector<std::pair<juce::String, juce::String>>& labels)` | 业务／适配 | [ProjectModel.h:1265](../juce/src/ProjectModel.h#L1265) |
| `void setNoteUtauFlags(const juce::String& noteId, const juce::String& flags)` | 业务／适配 | [ProjectModel.h:1269](../juce/src/ProjectModel.h#L1269) |
| `void setNotesUtauFlags(const std::vector<juce::String>& noteIds, const juce::String& flags)` | 业务／适配 | [ProjectModel.h:1270](../juce/src/ProjectModel.h#L1270) |
| `void setNotesUtauConsonantVelocity(const std::vector<juce::String>& noteIds, int velocity)` | 业务／适配 | [ProjectModel.h:1272](../juce/src/ProjectModel.h#L1272) |
| `void setNoteUtauOto(const juce::String& noteId, const backend::UtauOtoOverride& oto)` | 业务／适配 | [ProjectModel.h:1299](../juce/src/ProjectModel.h#L1299) |
| `bool setNoteUtauOtoChecked(const juce::String& noteId, std::uint64_t expectedRevision, const backend::UtauOtoOverride& value, std::uint64_t& resultingRevision, juce::String& error)` | 业务／适配 | [ProjectModel.h:1302](../juce/src/ProjectModel.h#L1302) |
| `void clearNotesUtauOto(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1307](../juce/src/ProjectModel.h#L1307) |
| `void setNotesUtauStp(const std::vector<juce::String>& noteIds, double seconds)` | 业务／适配 | [ProjectModel.h:1309](../juce/src/ProjectModel.h#L1309) |
| `void setNoteUtauTimingOverrides(const juce::String& noteId, bool enabled, double preutteranceSeconds, double overlapSeconds)` | 业务／适配 | [ProjectModel.h:1310](../juce/src/ProjectModel.h#L1310) |
| `void setNoteRobustPitchCurve(const juce::String& noteId, bool enabled)` | 业务／适配 | [ProjectModel.h:1313](../juce/src/ProjectModel.h#L1313) |
| `bool rememberDiffSingerParameters(const std::map<juce::String, std::vector<FlagCurve>>& curves)` | 业务／适配 | [ProjectModel.h:1314](../juce/src/ProjectModel.h#L1314) |
| `bool applyDiffSingerParameters(std::uint64_t expectedRevision, const std::map<juce::String, std::vector<FlagCurve>>& curves)` | 业务／适配 | [ProjectModel.h:1315](../juce/src/ProjectModel.h#L1315) |
| `bool setDiffSingerParameterCurve(const juce::String& id, const juce::String& key, std::vector<FlagCurvePoint> points)` | 业务／适配 | [ProjectModel.h:1317](../juce/src/ProjectModel.h#L1317) |
| `bool setDiffSingerPitchOffset(const juce::String& noteId, std::vector<PitchCurveEditPoint> points)` | 业务／适配 | [ProjectModel.h:1319](../juce/src/ProjectModel.h#L1319) |
| `bool setNotePitchCurve(const juce::String& noteId, std::vector<PitchCurveEditPoint> points, bool storeControlPoints = false, bool authorPitchSupport = false)` | 业务／适配 | [ProjectModel.h:1321](../juce/src/ProjectModel.h#L1321) |
| `bool applyDiffSingerTiming(std::uint64_t expectedRevision, const std::vector<std::pair<juce::String, juce::String>>& timings)` | 业务／适配 | [ProjectModel.h:1324](../juce/src/ProjectModel.h#L1324) |
| `bool applyDiffSingerPronunciation(std::uint64_t expectedRevision, const juce::String& trackId, const std::vector<std::pair<juce::String, juce::String>>& readings, const juce::String& dictionary)` | 业务／适配 | [ProjectModel.h:1326](../juce/src/ProjectModel.h#L1326) |
| `bool restoreDiffSingerPitchPoint(std::uint64_t expectedRevision, const juce::String& noteId, const std::vector<PitchCurveEditPoint>& displayedPoints, int index)` | 业务／适配 | [ProjectModel.h:1328](../juce/src/ProjectModel.h#L1328) |
| `bool applyDiffSingerPitch(std::uint64_t expectedRevision, const std::vector<std::pair<juce::String, std::vector<PitchCurveEditPoint>>>& curves, bool preserveOutsideNoteHandles = false)` | 业务／适配 | [ProjectModel.h:1330](../juce/src/ProjectModel.h#L1330) |
| `void setDiffSingerOptions(const juce::String& trackId, const juce::String& language, const juce::String& speaker)` | 业务／适配 | [ProjectModel.h:1333](../juce/src/ProjectModel.h#L1333) |
| `bool setNoteAmplitudeEnvelope(const juce::String& noteId, std::vector<AmplitudeEnvelopePoint> points)` | 业务／适配 | [ProjectModel.h:1335](../juce/src/ProjectModel.h#L1335) |
| `int crossfadeNotes(std::vector<CrossfadeSpan> spans, std::uint64_t expectedRevision, bool curved = false)` | 业务／适配 | [ProjectModel.h:1347](../juce/src/ProjectModel.h#L1347) |
| `bool rememberCvvcPhonemes(const std::vector<std::pair<juce::String,juce::String>>& caches, std::uint64_t expectedRevision)` | 业务／适配 | [ProjectModel.h:1348](../juce/src/ProjectModel.h#L1348) |
| `bool setNotesAmplitudeEnvelopes( std::vector<std::pair<juce::String, std::vector<AmplitudeEnvelopePoint>>> envelopes)` | 业务／适配 | [ProjectModel.h:1350](../juce/src/ProjectModel.h#L1350) |
| `static PlannedNote plannedNoteFor(double startSeconds, double requestedSeconds, double clipSeconds, const std::vector<NoteSpan>& occupied)` | 业务／适配 | [ProjectModel.h:1376](../juce/src/ProjectModel.h#L1376) |
| `juce::String addClip(const juce::String& trackId, double startSeconds, double durationSeconds)` | 业务／适配 | [ProjectModel.h:1384](../juce/src/ProjectModel.h#L1384) |
| `bool canAddEmptyTuningClip(const juce::String& trackId, double startSeconds) const` | 业务／适配 | [ProjectModel.h:1386](../juce/src/ProjectModel.h#L1386) |
| `juce::String addEmptyTuningClip(const juce::String& trackId, double startSeconds, double durationSeconds = 10.0)` | 业务／适配 | [ProjectModel.h:1388](../juce/src/ProjectModel.h#L1388) |
| `static std::vector<std::size_t> regionsToImport( const std::vector<RegionSpan>& regions)` | 业务／适配 | [ProjectModel.h:1405](../juce/src/ProjectModel.h#L1405) |
| `bool initializeNativeNotePitch(const juce::String& noteId)` | 业务／适配 | [ProjectModel.h:1422](../juce/src/ProjectModel.h#L1422) |
| `void flattenNotePitch(const std::vector<juce::String>& noteIds, bool snapNativeToSemitone = false)` | 业务／适配 | [ProjectModel.h:1423](../juce/src/ProjectModel.h#L1423) |
| `bool restoreNativeSourcePitch(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1424](../juce/src/ProjectModel.h#L1424) |
| `static double firstFreeStartFrom(double startSeconds, const std::vector<NoteSpan>& occupied)` | 业务／适配 | [ProjectModel.h:1425](../juce/src/ProjectModel.h#L1425) |
| `juce::String addNote(const juce::String& preferredClipId, double absoluteStart, double duration, float midiNote)` | 业务／适配 | [ProjectModel.h:1428](../juce/src/ProjectModel.h#L1428) |
| `juce::String addNoteFrom(const juce::String& preferredClipId, double absoluteFromSeconds, double maximumDuration, float midiNote)` | 业务／适配 | [ProjectModel.h:1436](../juce/src/ProjectModel.h#L1436) |
| `void removeNote(const juce::String& noteId)` | 业务／适配 | [ProjectModel.h:1439](../juce/src/ProjectModel.h#L1439) |
| `void removeNotes(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1440](../juce/src/ProjectModel.h#L1440) |
| `void removeNotesRippling(const std::vector<juce::String>& noteIds)` | 业务／适配 | [ProjectModel.h:1444](../juce/src/ProjectModel.h#L1444) |
| `void insertGapBeforeNote(const juce::String& noteId, double seconds)` | 业务／适配 | [ProjectModel.h:1448](../juce/src/ProjectModel.h#L1448) |
| `void closeGapBeforeNote(const juce::String& noteId, double seconds)` | 业务／适配 | [ProjectModel.h:1451](../juce/src/ProjectModel.h#L1451) |
| `void toggleNoteConnection(const juce::String& noteId)` | 业务／适配 | [ProjectModel.h:1452](../juce/src/ProjectModel.h#L1452) |
| `MaterialUpdateReport applySourceSettingsWithReport( const juce::File& source, const std::vector<SampleRegionSetting>& rows, bool apply = true)` | 业务／适配 | [ProjectModel.h:1473](../juce/src/ProjectModel.h#L1473) |
| `void applySourceSettings(const juce::File& source, const std::vector<SampleRegionSetting>& rows)` | 业务／适配 | [ProjectModel.h:1476](../juce/src/ProjectModel.h#L1476) |
| `void setNoteMaterialLocalOverride(const juce::String& noteId, bool enabled)` | 业务／适配 | [ProjectModel.h:1478](../juce/src/ProjectModel.h#L1478) |
| `bool getNoteMaterialLocalBinding(const juce::String& noteId, NoteMaterialLocalBinding&, SampleRegionSetting& row, juce::String& error) const` | 业务／适配 | [ProjectModel.h:1490](../juce/src/ProjectModel.h#L1490) |
| `bool applyNoteMaterialLocalRow(const NoteMaterialLocalBinding&, const SampleRegionSetting&, std::uint64_t& resultingRevision, juce::String& error)` | 业务／适配 | [ProjectModel.h:1492](../juce/src/ProjectModel.h#L1492) |
| `bool restoreNoteMaterialLocalRow(const NoteMaterialLocalBinding&, SampleRegionSetting& restored, std::uint64_t& resultingRevision, juce::String& error)` | 业务／适配 | [ProjectModel.h:1494](../juce/src/ProjectModel.h#L1494) |
| `static bool makeClassicNoteMaterialOto(const SampleRegionSetting&, backend::UtauOtoOverride&, juce::String& error)` | 业务／适配 | [ProjectModel.h:1497](../juce/src/ProjectModel.h#L1497) |
| `MaterialSourceReplacementReport replaceNoteMaterialSource( const juce::String& noteId, const juce::File& audio, const std::vector<SampleRegionSetting>& completeRows, const juce::String& audioId, const juce::String& regionId, const juce::String& chosenAlias, juce::String& error, bool apply = true)` | 业务／适配 | [ProjectModel.h:1517](../juce/src/ProjectModel.h#L1517) |
| `std::vector<MaterialSourceBindingTicket> pendingMaterialSourceAnalyses() const` | 业务／适配 | [ProjectModel.h:1522](../juce/src/ProjectModel.h#L1522) |
| `bool completeMaterialSourceAnalysis(const MaterialSourceBindingTicket& ticket, const std::vector<NoteData>& fullFileSourceNotes, juce::String& error)` | 业务／适配 | [ProjectModel.h:1527](../juce/src/ProjectModel.h#L1527) |
| `bool save(const juce::File& file, juce::String& error) const` | 业务／适配 | [ProjectModel.h:1530](../juce/src/ProjectModel.h#L1530) |
| `bool load(const juce::File& file, juce::String& error)` | 业务／适配 | [ProjectModel.h:1531](../juce/src/ProjectModel.h#L1531) |
| `static bool saveRecoverySnapshot(ProjectData data, const juce::File& recoveryFile, const juce::File& originalFile, juce::String& error)` | 业务／适配 | [ProjectModel.h:1532](../juce/src/ProjectModel.h#L1532) |

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

## McpServer（13 个声明）

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
| `static bool resolveUtauRenderSelection(const ProjectData&, const juce::var& arguments, const std::vector<juce::String>& currentSelection, std::vector<juce::String>& result, juce::String& error)` | 业务／适配 | [backend/McpServer.h:40](../juce/src/backend/McpServer.h#L40) |

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
| `utau_render_selection` | [backend/McpServer.cpp:777](../juce/src/backend/McpServer.cpp#L777) | Choose exact UTAU/DS render scope; [] clears it, omitted IDs use GUI selection (headless: empty) / 仅渲染所选音符 |
| `set_utau_resampler` | [backend/McpServer.cpp:786](../juce/src/backend/McpServer.cpp#L786) | Point UTAU rendering at a resampler executable / 指定 UTAU 重采样器 |
| `render_prepare` | [backend/McpServer.cpp:791](../juce/src/backend/McpServer.cpp#L791) | Pre-render the current project with its selected algorithms / 按当前所选算法预渲染工程 |
| `render_status` | [backend/McpServer.cpp:797](../juce/src/backend/McpServer.cpp#L797) | Read pre-render progress and active backends / 读取预渲染进度与实际后端 |
| `export_wav` | [backend/McpServer.cpp:798](../juce/src/backend/McpServer.cpp#L798) | Render every note and export to WAV; track_id exports one track alone, from_seconds/to_seconds one stretch / 渲染全曲并导出 WAV，track_id 可单独导出一个轨道 |
| `transport_play` | [backend/McpServer.cpp:813](../juce/src/backend/McpServer.cpp#L813) | Render if needed and start playback; play_until_seconds stops where a selection ends / 必要时预渲染并开始播放 |
| `transport_stop` | [backend/McpServer.cpp:821](../juce/src/backend/McpServer.cpp#L821) | Stop transport playback / 停止播放 |
| `transport_seek` | [backend/McpServer.cpp:822](../juce/src/backend/McpServer.cpp#L822) | Seek transport to position_seconds / 跳转播放位置 |
| `transport_status` | [backend/McpServer.cpp:826](../juce/src/backend/McpServer.cpp#L826) | Read playback position and render state / 读取播放位置与渲染状态 |
| `sample_settings_read` | [backend/McpServer.cpp:827](../juce/src/backend/McpServer.cpp#L827) | Read or derive audio otto.ini regions / 读取或生成音频 otto.ini 分段 |
| `sample_settings_save` | [backend/McpServer.cpp:832](../juce/src/backend/McpServer.cpp#L832) | Save audio regions to otto.ini / 保存音频分段到 otto.ini |
| `oto_import` | [backend/McpServer.cpp:839](../juce/src/backend/McpServer.cpp#L839) | Import one audio file's regions from UTAU oto.ini / 从 UTAU oto.ini 导入单个音频分段 |
| `oto_export` | [backend/McpServer.cpp:848](../juce/src/backend/McpServer.cpp#L848) | Export one audio file's regions to UTAU oto.ini / 将单个音频分段导出为 UTAU oto.ini |
| `jie_oto_create` | [backend/McpServer.cpp:854](../juce/src/backend/McpServer.cpp#L854) | Seed a voicebank's four-region oto4.ini from its oto.ini / 按 oto.ini 生成界•OTO |
| `voicebank_import` | [backend/McpServer.cpp:858](../juce/src/backend/McpServer.cpp#L858) | Inspect a UTAU or native material folder without writing annotations / 只读检查 UTAU 或原生素材目录，不写入标注 |
| `read_file` | [backend/McpServer.cpp:862](../juce/src/backend/McpServer.cpp#L862) | Read a byte range as base64 / 读取任意文件内容 |
| `list_directory` | [backend/McpServer.cpp:870](../juce/src/backend/McpServer.cpp#L870) | List a directory with type and size / 列出目录内容 |
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
| `--migrate-otto-folder` | [Main.cpp:205](../juce/src/Main.cpp#L205) |
| `--export-otto-folder` | [Main.cpp:206](../juce/src/Main.cpp#L206) |
| `--include-bare` | [Main.cpp:216](../juce/src/Main.cpp#L216) |
| `--replace` | [Main.cpp:220](../juce/src/Main.cpp#L220) |
| `--smoke-main-project-registration-ui` | [Main.cpp:230](../juce/src/Main.cpp#L230) |
| `--smoke-classic-note-scope` | [Main.cpp:236](../juce/src/Main.cpp#L236) |
| `--smoke-oto-provenance` | [Main.cpp:241](../juce/src/Main.cpp#L241) |
| `--smoke-community-dataset-menu` | [Main.cpp:246](../juce/src/Main.cpp#L246) |
| `--smoke-material-root-registration` | [Main.cpp:251](../juce/src/Main.cpp#L251) |
| `--smoke-variant-asset-ui` | [Main.cpp:256](../juce/src/Main.cpp#L256) |
| `--smoke-voicebank-family` | [Main.cpp:261](../juce/src/Main.cpp#L261) |
| `--smoke-empty-lyric-piano` | [Main.cpp:266](../juce/src/Main.cpp#L266) |
| `--smoke-project-material-memory` | [Main.cpp:271](../juce/src/Main.cpp#L271) |
| `--smoke-asset-context` | [Main.cpp:277](../juce/src/Main.cpp#L277) |
| `--inspect-piano-project` | [Main.cpp:282](../juce/src/Main.cpp#L282) |
| `--smoke-track-types` | [Main.cpp:287](../juce/src/Main.cpp#L287) |
| `--smoke-clip-length` | [Main.cpp:292](../juce/src/Main.cpp#L292) |
| `--smoke-engine-view-capabilities` | [Main.cpp:299](../juce/src/Main.cpp#L299) |
| `--smoke-utau-selection-scope` | [Main.cpp:306](../juce/src/Main.cpp#L306) |
| `--smoke-track-editing-interfaces-ui` | [Main.cpp:313](../juce/src/Main.cpp#L313) |
| `--smoke-material-variants` | [Main.cpp:320](../juce/src/Main.cpp#L320) |
| `--smoke-community-dataset` | [Main.cpp:325](../juce/src/Main.cpp#L325) |
| `--smoke-ustx-importer` | [Main.cpp:331](../juce/src/Main.cpp#L331) |
| `--smoke-community-import-gui` | [Main.cpp:336](../juce/src/Main.cpp#L336) |
| `--smoke-registered-project-audio` | [Main.cpp:342](../juce/src/Main.cpp#L342) |
| `--smoke-registration-facts` | [Main.cpp:348](../juce/src/Main.cpp#L348) |
| `--smoke-main-material-variants` | [Main.cpp:353](../juce/src/Main.cpp#L353) |
| `--audit-project-material-registration` | [Main.cpp:359](../juce/src/Main.cpp#L359) |
| `--smoke-project-material-registration-model` | [Main.cpp:368](../juce/src/Main.cpp#L368) |
| `--smoke-project-material-registration` | [Main.cpp:373](../juce/src/Main.cpp#L373) |
| `--smoke-note-material-local-override` | [Main.cpp:378](../juce/src/Main.cpp#L378) |
| `--smoke-local-material-scope` | [Main.cpp:383](../juce/src/Main.cpp#L383) |
| `--smoke-connection-pitch-ui` | [Main.cpp:388](../juce/src/Main.cpp#L388) |
| `--smoke-host-pitch-projection` | [Main.cpp:393](../juce/src/Main.cpp#L393) |
| `--smoke-material-pronunciation-catalog` | [Main.cpp:398](../juce/src/Main.cpp#L398) |
| `--smoke-graph-pitch-edit-batch` | [Main.cpp:403](../juce/src/Main.cpp#L403) |
| `--smoke-overlap-selector` | [Main.cpp:408](../juce/src/Main.cpp#L408) |
| `--smoke-forced-portamento-audio` | [Main.cpp:413](../juce/src/Main.cpp#L413) |
| `--smoke-track-pitch-scope` | [Main.cpp:419](../juce/src/Main.cpp#L419) |
| `--smoke-forced-portamento` | [Main.cpp:424](../juce/src/Main.cpp#L424) |
| `--smoke-named-vowel` | [Main.cpp:429](../juce/src/Main.cpp#L429) |
| `--smoke-vowel-timing-audio` | [Main.cpp:434](../juce/src/Main.cpp#L434) |
| `--smoke-pronunciation-timing` | [Main.cpp:440](../juce/src/Main.cpp#L440) |
| `--smoke-adjacent-pronunciation` | [Main.cpp:445](../juce/src/Main.cpp#L445) |
| `--smoke-vowel-stretch-alignment` | [Main.cpp:450](../juce/src/Main.cpp#L450) |
| `--smoke-material-source-replacement` | [Main.cpp:455](../juce/src/Main.cpp#L455) |
| `--smoke-native-nsf-compatibility` | [Main.cpp:460](../juce/src/Main.cpp#L460) |
| `--smoke-native-workflow-e2e` | [Main.cpp:466](../juce/src/Main.cpp#L466) |
| `--smoke-material-topology-audio` | [Main.cpp:475](../juce/src/Main.cpp#L475) |
| `--smoke-midi-export-scope` | [Main.cpp:481](../juce/src/Main.cpp#L481) |
| `--smoke-unified-draw-action` | [Main.cpp:487](../juce/src/Main.cpp#L487) |
| `--smoke-main-source-replacement-ui` | [Main.cpp:493](../juce/src/Main.cpp#L493) |
| `--smoke-single-entry-ui` | [Main.cpp:501](../juce/src/Main.cpp#L501) |
| `--smoke-nsf-native-expression` | [Main.cpp:509](../juce/src/Main.cpp#L509) |
| `--smoke-material-topology` | [Main.cpp:517](../juce/src/Main.cpp#L517) |
| `--smoke-native-connection-graph` | [Main.cpp:523](../juce/src/Main.cpp#L523) |
| `--smoke-material-format-auto` | [Main.cpp:529](../juce/src/Main.cpp#L529) |
| `--smoke-unified-material-editor` | [Main.cpp:535](../juce/src/Main.cpp#L535) |
| `--smoke-material-insertion` | [Main.cpp:541](../juce/src/Main.cpp#L541) |
| `--smoke-material-asset-tree` | [Main.cpp:547](../juce/src/Main.cpp#L547) |
| `--smoke-material-annotation` | [Main.cpp:553](../juce/src/Main.cpp#L553) |
| `--smoke-material-binding` | [Main.cpp:559](../juce/src/Main.cpp#L559) |
| `--smoke-otto-material` | [Main.cpp:565](../juce/src/Main.cpp#L565) |
| `--audit-hamood-keys` | [Main.cpp:574](../juce/src/Main.cpp#L574) |
| `--smoke-hamood-utau` | [Main.cpp:593](../juce/src/Main.cpp#L593) |
| `--smoke-hamood` | [Main.cpp:599](../juce/src/Main.cpp#L599) |
| `--smoke-diffsinger-pitch-offset` | [Main.cpp:606](../juce/src/Main.cpp#L606) |
| `--smoke-accompaniment-tempo` | [Main.cpp:613](../juce/src/Main.cpp#L613) |
| `--smoke-accompaniment` | [Main.cpp:629](../juce/src/Main.cpp#L629) |
| `--smoke-wav-export-options` | [Main.cpp:636](../juce/src/Main.cpp#L636) |
| `--smoke-diffsinger-pitch-persistence` | [Main.cpp:646](../juce/src/Main.cpp#L646) |
| `--reopen` | [Main.cpp:650](../juce/src/Main.cpp#L650) |
| `--smoke-diffsinger-pitch-handles` | [Main.cpp:654](../juce/src/Main.cpp#L654) |
| `--smoke-diffsinger-pitch-restore` | [Main.cpp:660](../juce/src/Main.cpp#L660) |
| `--smoke-diffsinger-pitch-reference` | [Main.cpp:667](../juce/src/Main.cpp#L667) |
| `--smoke-continuous-flag` | [Main.cpp:673](../juce/src/Main.cpp#L673) |
| `--smoke-classic-flag-ui` | [Main.cpp:680](../juce/src/Main.cpp#L680) |
| `--smoke-diffsinger-inference` | [Main.cpp:689](../juce/src/Main.cpp#L689) |
| `--smoke-diffsinger-parameter-retake` | [Main.cpp:695](../juce/src/Main.cpp#L695) |
| `--smoke-diffsinger-cache` | [Main.cpp:702](../juce/src/Main.cpp#L702) |
| `--smoke-diffsinger-parameters` | [Main.cpp:707](../juce/src/Main.cpp#L707) |
| `--smoke-diffsinger-expressions` | [Main.cpp:712](../juce/src/Main.cpp#L712) |
| `--smoke-diffsinger-consonants` | [Main.cpp:719](../juce/src/Main.cpp#L719) |
| `--smoke-diffsinger-history-playback` | [Main.cpp:725](../juce/src/Main.cpp#L725) |
| `--smoke-diffsinger-timing` | [Main.cpp:732](../juce/src/Main.cpp#L732) |
| `--smoke-diffsinger-ui` | [Main.cpp:739](../juce/src/Main.cpp#L739) |
| `--smoke-diffsinger-timing-selection` | [Main.cpp:740](../juce/src/Main.cpp#L740) |
| `--smoke-diffsinger` | [Main.cpp:749](../juce/src/Main.cpp#L749) |
| `--smoke-chinese-cvvc` | [Main.cpp:756](../juce/src/Main.cpp#L756) |
| `--smoke-output-engine` | [Main.cpp:764](../juce/src/Main.cpp#L764) |
| `--smoke-native-vibrato` | [Main.cpp:773](../juce/src/Main.cpp#L773) |
| `--smoke-native-vibrato-render` | [Main.cpp:784](../juce/src/Main.cpp#L784) |
| `--smoke-native-noise-option` | [Main.cpp:790](../juce/src/Main.cpp#L790) |
| `--smoke-native-incremental` | [Main.cpp:798](../juce/src/Main.cpp#L798) |
| `--smoke-native-shutdown` | [Main.cpp:806](../juce/src/Main.cpp#L806) |
| `--smoke-native-noise` | [Main.cpp:814](../juce/src/Main.cpp#L814) |
| `--smoke-wcs-nonloop` | [Main.cpp:821](../juce/src/Main.cpp#L821) |
| `--smoke-hn-cache` | [Main.cpp:828](../juce/src/Main.cpp#L828) |
| `--force` | [Main.cpp:831](../juce/src/Main.cpp#L831) |
| `--smoke-hifisampler` | [Main.cpp:835](../juce/src/Main.cpp#L835) |
| `--smoke-hifishifter-mel` | [Main.cpp:835](../juce/src/Main.cpp#L835) |
| `--smoke-nsf-regions` | [Main.cpp:844](../juce/src/Main.cpp#L844) |
| `--smoke-nsf-pitch-transitions` | [Main.cpp:852](../juce/src/Main.cpp#L852) |
| `--smoke-voicebank-import` | [Main.cpp:858](../juce/src/Main.cpp#L858) |
| `--smoke-voicebank-selection` | [Main.cpp:864](../juce/src/Main.cpp#L864) |
| `--smoke-nsf-picker` | [Main.cpp:873](../juce/src/Main.cpp#L873) |
| `--smoke-cross-region-editing` | [Main.cpp:882](../juce/src/Main.cpp#L882) |
| `--smoke-hamood-persistence` | [Main.cpp:882](../juce/src/Main.cpp#L882) |
| `--smoke-mou-to-jie` | [Main.cpp:889](../juce/src/Main.cpp#L889) |
| `--smoke-oto-overlap` | [Main.cpp:895](../juce/src/Main.cpp#L895) |
| `--smoke-modeless-oto` | [Main.cpp:901](../juce/src/Main.cpp#L901) |
| `--smoke-piano-transport` | [Main.cpp:906](../juce/src/Main.cpp#L906) |
| `--smoke-track-reorder` | [Main.cpp:913](../juce/src/Main.cpp#L913) |
| `--smoke-decibel-input` | [Main.cpp:919](../juce/src/Main.cpp#L919) |
| `--smoke-oto-continuity` | [Main.cpp:925](../juce/src/Main.cpp#L925) |
| `--smoke-native-envelope` | [Main.cpp:930](../juce/src/Main.cpp#L930) |
| `--smoke-cvvc-layout-cache` | [Main.cpp:935](../juce/src/Main.cpp#L935) |
| `--smoke-auto-crossfade` | [Main.cpp:940](../juce/src/Main.cpp#L940) |
| `--smoke-pitch-point-selection` | [Main.cpp:945](../juce/src/Main.cpp#L945) |
| `--smoke-point-group` | [Main.cpp:950](../juce/src/Main.cpp#L950) |
| `--smoke-envelope-point-selection` | [Main.cpp:955](../juce/src/Main.cpp#L955) |
| `--smoke-native-linked-envelope` | [Main.cpp:960](../juce/src/Main.cpp#L960) |
| `--smoke-advanced-envelope` | [Main.cpp:965](../juce/src/Main.cpp#L965) |
| `--smoke-advanced-envelope-panel` | [Main.cpp:965](../juce/src/Main.cpp#L965) |
| `--smoke-playback-priority` | [Main.cpp:970](../juce/src/Main.cpp#L970) |
| `--smoke-ust-fidelity` | [Main.cpp:971](../juce/src/Main.cpp#L971) |
| `--smoke-project-safety` | [Main.cpp:972](../juce/src/Main.cpp#L972) |
| `--smoke-normal-display` | [Main.cpp:980](../juce/src/Main.cpp#L980) |
| `--smoke-note-hints` | [Main.cpp:988](../juce/src/Main.cpp#L988) |
| `--smoke-clip-editor-scope` | [Main.cpp:996](../juce/src/Main.cpp#L996) |
| `--smoke-empty-tuning-clip` | [Main.cpp:1004](../juce/src/Main.cpp#L1004) |
| `--smoke-track-gain-envelope` | [Main.cpp:1012](../juce/src/Main.cpp#L1012) |
| `--smoke-native-rendered-waveform` | [Main.cpp:1020](../juce/src/Main.cpp#L1020) |
| `--smoke-native-note-move` | [Main.cpp:1027](../juce/src/Main.cpp#L1027) |
| `--smoke-native-source-pitch-restore` | [Main.cpp:1033](../juce/src/Main.cpp#L1033) |
| `--smoke-native-audio-overlap-focus` | [Main.cpp:1039](../juce/src/Main.cpp#L1039) |
| `--smoke-native-audio-overlap` | [Main.cpp:1045](../juce/src/Main.cpp#L1045) |
| `--smoke-native-linked-pitch` | [Main.cpp:1051](../juce/src/Main.cpp#L1051) |
| `--smoke-source-breathiness` | [Main.cpp:1058](../juce/src/Main.cpp#L1058) |
| `--smoke-native-sparse-pitch` | [Main.cpp:1065](../juce/src/Main.cpp#L1065) |
| `--smoke-native-point-first` | [Main.cpp:1070](../juce/src/Main.cpp#L1070) |
| `--smoke-native-unpitched-regions` | [Main.cpp:1076](../juce/src/Main.cpp#L1076) |
| `--smoke-native-consonant-display` | [Main.cpp:1084](../juce/src/Main.cpp#L1084) |
| `--smoke-native-pitch-visibility` | [Main.cpp:1091](../juce/src/Main.cpp#L1091) |
| `--smoke-native-audio-link` | [Main.cpp:1097](../juce/src/Main.cpp#L1097) |
| `--smoke-native-note-join` | [Main.cpp:1103](../juce/src/Main.cpp#L1103) |
| `--smoke-native-trim-expand` | [Main.cpp:1108](../juce/src/Main.cpp#L1108) |
| `--smoke-native-audio-trim` | [Main.cpp:1113](../juce/src/Main.cpp#L1113) |
| `--smoke-native-audio-disconnect` | [Main.cpp:1119](../juce/src/Main.cpp#L1119) |
| `--smoke-native-note-copy-paste` | [Main.cpp:1125](../juce/src/Main.cpp#L1125) |
| `--smoke-native-import-performance` | [Main.cpp:1132](../juce/src/Main.cpp#L1132) |
| `--smoke-native-source-pitch` | [Main.cpp:1139](../juce/src/Main.cpp#L1139) |
| `--smoke-native-waveform-preview` | [Main.cpp:1147](../juce/src/Main.cpp#L1147) |
| `--smoke-timeline-notes` | [Main.cpp:1155](../juce/src/Main.cpp#L1155) |
| `--smoke-clip-gain-knob` | [Main.cpp:1163](../juce/src/Main.cpp#L1163) |
| `--smoke-clip-merge` | [Main.cpp:1171](../juce/src/Main.cpp#L1171) |
| `--smoke-native-boundary-stretch` | [Main.cpp:1179](../juce/src/Main.cpp#L1179) |
| `--smoke-native-gap-stretch` | [Main.cpp:1184](../juce/src/Main.cpp#L1184) |
| `--smoke-native-initial-pitch` | [Main.cpp:1190](../juce/src/Main.cpp#L1190) |
| `--smoke-native-loudness` | [Main.cpp:1195](../juce/src/Main.cpp#L1195) |
| `--smoke-timeline-selection` | [Main.cpp:1201](../juce/src/Main.cpp#L1201) |
| `--smoke-clip-trim` | [Main.cpp:1209](../juce/src/Main.cpp#L1209) |
| `--smoke-clip-split` | [Main.cpp:1218](../juce/src/Main.cpp#L1218) |
| `--smoke-independent-zoom` | [Main.cpp:1227](../juce/src/Main.cpp#L1227) |
| `--smoke-scroll-zoom` | [Main.cpp:1236](../juce/src/Main.cpp#L1236) |
| `--preview-note-dance` | [Main.cpp:1245](../juce/src/Main.cpp#L1245) |
| `--smoke-tempo-change` | [Main.cpp:1253](../juce/src/Main.cpp#L1253) |
| `--smoke-timeline-pit` | [Main.cpp:1262](../juce/src/Main.cpp#L1262) |
| `--mcp-live` | [Main.cpp:1268](../juce/src/Main.cpp#L1268) |
| `--mcp-list-sessions` | [Main.cpp:1268](../juce/src/Main.cpp#L1268) |
| `--session` | [Main.cpp:1271](../juce/src/Main.cpp#L1271) |
| `--smoke-point-first-pitch` | [Main.cpp:1276](../juce/src/Main.cpp#L1276) |
| `--smoke-utau-overlap-pitch` | [Main.cpp:1281](../juce/src/Main.cpp#L1281) |
| `--smoke-cvvc-pitch-project` | [Main.cpp:1287](../juce/src/Main.cpp#L1287) |
| `--smoke-cvvc-linear-project` | [Main.cpp:1293](../juce/src/Main.cpp#L1293) |
| `--smoke-hifisampler-envelope` | [Main.cpp:1299](../juce/src/Main.cpp#L1299) |
| `--smoke-nsf-project-note` | [Main.cpp:1305](../juce/src/Main.cpp#L1305) |
| `--mcp` | [Main.cpp:1311](../juce/src/Main.cpp#L1311) |
| `--roots` | [Main.cpp:1313](../juce/src/Main.cpp#L1313) |
| `--smoke-oto-playback` | [Main.cpp:1325](../juce/src/Main.cpp#L1325) |
| `--smoke-oto-editor` | [Main.cpp:1525](../juce/src/Main.cpp#L1525) |
| `--click` | [Main.cpp:1636](../juce/src/Main.cpp#L1636) |
| `--smoke-splice-envelope` | [Main.cpp:1648](../juce/src/Main.cpp#L1648) |
| `--smoke-dialog-enter` | [Main.cpp:1758](../juce/src/Main.cpp#L1758) |
| `--smoke-source-edit-view` | [Main.cpp:1816](../juce/src/Main.cpp#L1816) |
| `--smoke-consonant-hold` | [Main.cpp:1878](../juce/src/Main.cpp#L1878) |
| `--smoke-mou-panel-counts` | [Main.cpp:2098](../juce/src/Main.cpp#L2098) |
| `--smoke-mou-consonant-ticks` | [Main.cpp:2189](../juce/src/Main.cpp#L2189) |
| `--smoke-mou-boundary-drag` | [Main.cpp:2342](../juce/src/Main.cpp#L2342) |
| `--smoke-prefix-map` | [Main.cpp:2431](../juce/src/Main.cpp#L2431) |
| `--repair-mou-oto` | [Main.cpp:2472](../juce/src/Main.cpp#L2472) |
| `--smoke-mou-oto-file` | [Main.cpp:2493](../juce/src/Main.cpp#L2493) |
| `--smoke-utau-mode` | [Main.cpp:2692](../juce/src/Main.cpp#L2692) |
| `--smoke-lane-marquee` | [Main.cpp:2832](../juce/src/Main.cpp#L2832) |
| `--smoke-flag-lane` | [Main.cpp:2933](../juce/src/Main.cpp#L2933) |
| `--smoke-keyboard-labels` | [Main.cpp:3794](../juce/src/Main.cpp#L3794) |
| `--smoke-lane-labels` | [Main.cpp:3848](../juce/src/Main.cpp#L3848) |
| `--smoke-export-targets` | [Main.cpp:3876](../juce/src/Main.cpp#L3876) |
| `--smoke-audition-position` | [Main.cpp:3982](../juce/src/Main.cpp#L3982) |
| `--smoke-lyric-envelope` | [Main.cpp:4024](../juce/src/Main.cpp#L4024) |
| `--smoke-lyric-timing` | [Main.cpp:4096](../juce/src/Main.cpp#L4096) |
| `--smoke-vibrato-bake` | [Main.cpp:4195](../juce/src/Main.cpp#L4195) |
| `--smoke-mou-note-regions` | [Main.cpp:4308](../juce/src/Main.cpp#L4308) |
| `--smoke-region-guides` | [Main.cpp:4594](../juce/src/Main.cpp#L4594) |
| `--smoke-rest-playback` | [Main.cpp:4668](../juce/src/Main.cpp#L4668) |
| `--smoke-rest-lyric` | [Main.cpp:4764](../juce/src/Main.cpp#L4764) |
| `--smoke-missing-alias-piano` | [Main.cpp:4927](../juce/src/Main.cpp#L4927) |
| `--smoke-flag-curve-mode` | [Main.cpp:5054](../juce/src/Main.cpp#L5054) |
| `--smoke-flag-point-value` | [Main.cpp:5239](../juce/src/Main.cpp#L5239) |
| `--smoke-flag-overlap` | [Main.cpp:5329](../juce/src/Main.cpp#L5329) |
| `--smoke-load-timing` | [Main.cpp:5444](../juce/src/Main.cpp#L5444) |
| `--smoke-flatten-pitch-line` | [Main.cpp:5557](../juce/src/Main.cpp#L5557) |
| `--smoke-note-across-tracks` | [Main.cpp:5717](../juce/src/Main.cpp#L5717) |
| `--smoke-clip-across-tracks` | [Main.cpp:5873](../juce/src/Main.cpp#L5873) |
| `--smoke-reference-track` | [Main.cpp:6022](../juce/src/Main.cpp#L6022) |
| `--smoke-native-pitch-seam` | [Main.cpp:6170](../juce/src/Main.cpp#L6170) |
| `--smoke-native-pitch-points` | [Main.cpp:6242](../juce/src/Main.cpp#L6242) |
| `--smoke-paste-at-pointer` | [Main.cpp:6466](../juce/src/Main.cpp#L6466) |
| `--smoke-note-menu-split` | [Main.cpp:6609](../juce/src/Main.cpp#L6609) |
| `--smoke-lyric-gate` | [Main.cpp:6791](../juce/src/Main.cpp#L6791) |
| `--smoke-lyric-tab` | [Main.cpp:6888](../juce/src/Main.cpp#L6888) |
| `--smoke-syllable-cuts` | [Main.cpp:7088](../juce/src/Main.cpp#L7088) |
| `--smoke-single-syllable` | [Main.cpp:7270](../juce/src/Main.cpp#L7270) |
| `--smoke-deleted-notes-silent` | [Main.cpp:7377](../juce/src/Main.cpp#L7377) |
| `--smoke-delete-clip` | [Main.cpp:7524](../juce/src/Main.cpp#L7524) |
| `--smoke-draw-unit` | [Main.cpp:7621](../juce/src/Main.cpp#L7621) |
| `--smoke-new-track-draw` | [Main.cpp:7715](../juce/src/Main.cpp#L7715) |
| `--smoke-draw-drag` | [Main.cpp:7824](../juce/src/Main.cpp#L7824) |
| `--smoke-draw-overlap` | [Main.cpp:7996](../juce/src/Main.cpp#L7996) |
| `--smoke-dropdown-arrow` | [Main.cpp:8182](../juce/src/Main.cpp#L8182) |
| `--smoke-pitch-line` | [Main.cpp:8273](../juce/src/Main.cpp#L8273) |
| `--smoke-view-menu` | [Main.cpp:8564](../juce/src/Main.cpp#L8564) |
| `--inspect-melodyne-tracks` | [Main.cpp:8677](../juce/src/Main.cpp#L8677) |
| `--smoke-melodyne-provider` | [Main.cpp:8688](../juce/src/Main.cpp#L8688) |
| `--probe-melodyne-vst3-instance` | [Main.cpp:8713](../juce/src/Main.cpp#L8713) |
| `--smoke-native-timing` | [Main.cpp:8725](../juce/src/Main.cpp#L8725) |
| `--smoke-integrated-ui` | [Main.cpp:8782](../juce/src/Main.cpp#L8782) |
| `--smoke-integrated` | [Main.cpp:8791](../juce/src/Main.cpp#L8791) |
| `--smoke-native-hjm` | [Main.cpp:8797](../juce/src/Main.cpp#L8797) |
| `--smoke-batch-lyrics` | [Main.cpp:8864](../juce/src/Main.cpp#L8864) |
| `--smoke-envelope-lanes` | [Main.cpp:8956](../juce/src/Main.cpp#L8956) |
| `--smoke-wheel` | [Main.cpp:9015](../juce/src/Main.cpp#L9015) |
| `--smoke-waveform-toggle` | [Main.cpp:9056](../juce/src/Main.cpp#L9056) |
| `--smoke-envelope-base` | [Main.cpp:9166](../juce/src/Main.cpp#L9166) |
| `--smoke-amplitude-waveform` | [Main.cpp:9487](../juce/src/Main.cpp#L9487) |
| `--smoke-vibrato-end` | [Main.cpp:9945](../juce/src/Main.cpp#L9945) |
| `--smoke-hanzi-pinyin` | [Main.cpp:10205](../juce/src/Main.cpp#L10205) |
| `--smoke-shared-pitch-line` | [Main.cpp:10344](../juce/src/Main.cpp#L10344) |
| `--smoke-waveform-alignment` | [Main.cpp:11002](../juce/src/Main.cpp#L11002) |
| `--smoke-note-waveform` | [Main.cpp:11201](../juce/src/Main.cpp#L11201) |
| `--smoke-utau-waveform` | [Main.cpp:11507](../juce/src/Main.cpp#L11507) |
| `--smoke-voicebank-index` | [Main.cpp:11681](../juce/src/Main.cpp#L11681) |
| `--smoke-ust-pitch` | [Main.cpp:12147](../juce/src/Main.cpp#L12147) |
| `--low` | [Main.cpp:12452](../juce/src/Main.cpp#L12452) |
| `--smoke-hf-daemon-owner` | [Main.cpp:12571](../juce/src/Main.cpp#L12571) |
| `--quit` | [Main.cpp:12579](../juce/src/Main.cpp#L12579) |
| `--smoke-hf-prewarm` | [Main.cpp:12584](../juce/src/Main.cpp#L12584) |
| `--smoke-ust-vibrato` | [Main.cpp:12717](../juce/src/Main.cpp#L12717) |
| `--smoke-mcp-roots` | [Main.cpp:12962](../juce/src/Main.cpp#L12962) |
| `--smoke-mcp-schema` | [Main.cpp:13069](../juce/src/Main.cpp#L13069) |
| `--smoke-dragged-transition` | [Main.cpp:13234](../juce/src/Main.cpp#L13234) |
| `--smoke-forward-bend` | [Main.cpp:13457](../juce/src/Main.cpp#L13457) |
| `--smoke-midi-track-import` | [Main.cpp:13629](../juce/src/Main.cpp#L13629) |
| `--smoke-midi-export` | [Main.cpp:14044](../juce/src/Main.cpp#L14044) |
| `--sounding` | [Main.cpp:14169](../juce/src/Main.cpp#L14169) |
| `--still` | [Main.cpp:14301](../juce/src/Main.cpp#L14301) |
| `--smoke-ust-import` | [Main.cpp:14337](../juce/src/Main.cpp#L14337) |
| `--smoke-ust-encoding` | [Main.cpp:14478](../juce/src/Main.cpp#L14478) |
| `--smoke-ust` | [Main.cpp:14630](../juce/src/Main.cpp#L14630) |
| `--smoke-active-resampler` | [Main.cpp:14835](../juce/src/Main.cpp#L14835) |
| `--smoke-bundled-resampler` | [Main.cpp:14880](../juce/src/Main.cpp#L14880) |
| `--smoke-anchor-frequency` | [Main.cpp:14938](../juce/src/Main.cpp#L14938) |
| `--smoke-note-oto` | [Main.cpp:15124](../juce/src/Main.cpp#L15124) |
| `--smoke-prefix-jie` | [Main.cpp:15554](../juce/src/Main.cpp#L15554) |
| `--smoke-prefix-note` | [Main.cpp:15708](../juce/src/Main.cpp#L15708) |
| `--smoke-utau-overlap` | [Main.cpp:16065](../juce/src/Main.cpp#L16065) |
| `--smoke-note-stp` | [Main.cpp:16380](../juce/src/Main.cpp#L16380) |
| `--smoke-track-toggle-tips` | [Main.cpp:16619](../juce/src/Main.cpp#L16619) |
| `--smoke-track-area-menu` | [Main.cpp:16866](../juce/src/Main.cpp#L16866) |
| `--smoke-stretch-items` | [Main.cpp:16967](../juce/src/Main.cpp#L16967) |
| `--smoke-render-order` | [Main.cpp:17019](../juce/src/Main.cpp#L17019) |
| `--smoke-llsm2-length` | [Main.cpp:17151](../juce/src/Main.cpp#L17151) |
| `--smoke-glide-seam` | [Main.cpp:17203](../juce/src/Main.cpp#L17203) |
| `--smoke-native-envelope-base` | [Main.cpp:17267](../juce/src/Main.cpp#L17267) |
| `--smoke-nsf-utau-phrase` | [Main.cpp:17351](../juce/src/Main.cpp#L17351) |
| `--smoke-nsf-utau-synth` | [Main.cpp:17383](../juce/src/Main.cpp#L17383) |
| `--smoke-nsf-utau-f0` | [Main.cpp:17431](../juce/src/Main.cpp#L17431) |
| `--smoke-nsf-utau-mix` | [Main.cpp:17476](../juce/src/Main.cpp#L17476) |
| `--smoke-nsf-utau-plan` | [Main.cpp:17541](../juce/src/Main.cpp#L17541) |
| `--smoke-render-capability` | [Main.cpp:17598](../juce/src/Main.cpp#L17598) |
| `--smoke-oto-hjm-discipline` | [Main.cpp:17649](../juce/src/Main.cpp#L17649) |
| `--smoke-asset-register-dir` | [Main.cpp:17719](../juce/src/Main.cpp#L17719) |
| `--smoke-oto-audio` | [Main.cpp:17763](../juce/src/Main.cpp#L17763) |
| `--smoke-asset-audio` | [Main.cpp:17776](../juce/src/Main.cpp#L17776) |
| `--smoke-project-voicebank-assets` | [Main.cpp:17790](../juce/src/Main.cpp#L17790) |
| `--smoke-asset-search` | [Main.cpp:17796](../juce/src/Main.cpp#L17796) |
| `--smoke-asset-manager` | [Main.cpp:17805](../juce/src/Main.cpp#L17805) |
| `--smoke-gap-menu` | [Main.cpp:17930](../juce/src/Main.cpp#L17930) |
| `--smoke-insert-gap` | [Main.cpp:18049](../juce/src/Main.cpp#L18049) |
| `--smoke-ripple-delete` | [Main.cpp:18165](../juce/src/Main.cpp#L18165) |
| `--smoke-paste-notes` | [Main.cpp:18254](../juce/src/Main.cpp#L18254) |
| `--smoke-consonant-edge` | [Main.cpp:18430](../juce/src/Main.cpp#L18430) |
| `--smoke-consonant-reset` | [Main.cpp:18586](../juce/src/Main.cpp#L18586) |
| `--smoke-region-split` | [Main.cpp:18693](../juce/src/Main.cpp#L18693) |
| `--smoke-vibrato-presets` | [Main.cpp:18785](../juce/src/Main.cpp#L18785) |
| `--smoke-follow-rule` | [Main.cpp:18855](../juce/src/Main.cpp#L18855) |
| `--smoke-play-until` | [Main.cpp:18927](../juce/src/Main.cpp#L18927) |
| `--smoke-flags-field` | [Main.cpp:18986](../juce/src/Main.cpp#L18986) |
| `--smoke-vertical-drag` | [Main.cpp:19051](../juce/src/Main.cpp#L19051) |
| `--smoke-scroll-sync` | [Main.cpp:19144](../juce/src/Main.cpp#L19144) |
| `--smoke-edit-follow` | [Main.cpp:19225](../juce/src/Main.cpp#L19225) |
| `--smoke-note-drag-threshold` | [Main.cpp:19325](../juce/src/Main.cpp#L19325) |
| `--smoke-roll-drag` | [Main.cpp:19400](../juce/src/Main.cpp#L19400) |
| `--smoke-playhead-band` | [Main.cpp:19531](../juce/src/Main.cpp#L19531) |
| `--smoke-roll-cull` | [Main.cpp:19584](../juce/src/Main.cpp#L19584) |
| `--smoke-roll-timing` | [Main.cpp:19657](../juce/src/Main.cpp#L19657) |
| `--smoke-envelope-shapes` | [Main.cpp:19703](../juce/src/Main.cpp#L19703) |
| `--smoke-envelope-presets` | [Main.cpp:20020](../juce/src/Main.cpp#L20020) |
| `--smoke-piano-roll` | [Main.cpp:20036](../juce/src/Main.cpp#L20036) |
| `--render-project` | [Main.cpp:20084](../juce/src/Main.cpp#L20084) |
| `--inspect-project` | [Main.cpp:20213](../juce/src/Main.cpp#L20213) |
| `--inspect-settings` | [Main.cpp:20249](../juce/src/Main.cpp#L20249) |
| `--smoke-oto-spectrum` | [Main.cpp:20276](../juce/src/Main.cpp#L20276) |
| `--smoke-game-defaults` | [Main.cpp:20283](../juce/src/Main.cpp#L20283) |
| `--inspect-analysis` | [Main.cpp:20289](../juce/src/Main.cpp#L20289) |
| `--inspect-fcpe` | [Main.cpp:20311](../juce/src/Main.cpp#L20311) |
| `--smoke-export` | [Main.cpp:20337](../juce/src/Main.cpp#L20337) |
| `--smoke-mld5` | [Main.cpp:20367](../juce/src/Main.cpp#L20367) |
| `--smoke-utau-selection` | [Main.cpp:20460](../juce/src/Main.cpp#L20460) |
| `--smoke-utau-voicebank` | [Main.cpp:20574](../juce/src/Main.cpp#L20574) |
| `--smoke-utau` | [Main.cpp:20657](../juce/src/Main.cpp#L20657) |
| `--inspect-midi` | [Main.cpp:20743](../juce/src/Main.cpp#L20743) |
| `--inspect-audio` | [Main.cpp:20766](../juce/src/Main.cpp#L20766) |
| `--inspect-mpd` | [Main.cpp:20801](../juce/src/Main.cpp#L20801) |

## 范围与静态体量

扫描自有 `juce/src` 非测试 `.h/.cpp/.inc`：186 个文件，112,693 行（包括注释和空行）。不含 juce/third_party、tests 和名称带 Smoke 的包含单元。

| 文件 | 行数 |
| --- | --- |
| [juce/src/Main.cpp](../juce/src/Main.cpp) | 20,969 |
| [juce/src/PianoRollComponent.cpp](../juce/src/PianoRollComponent.cpp) | 11,849 |
| [juce/src/ProjectModel.cpp](../juce/src/ProjectModel.cpp) | 11,651 |
| [juce/src/MainComponent.cpp](../juce/src/MainComponent.cpp) | 6,826 |
| [juce/src/AudioEngine.cpp](../juce/src/AudioEngine.cpp) | 4,025 |
| [juce/src/OtoWaveformEditorComponent.cpp](../juce/src/OtoWaveformEditorComponent.cpp) | 2,925 |
| [juce/src/backend/UtauRenderer.cpp](../juce/src/backend/UtauRenderer.cpp) | 2,845 |
| [juce/src/PinyinTable.inc](../juce/src/PinyinTable.inc) | 2,709 |
| [juce/src/backend/McpServer.cpp](../juce/src/backend/McpServer.cpp) | 2,149 |
| [juce/src/backend/NsfHifiganRenderer.cpp](../juce/src/backend/NsfHifiganRenderer.cpp) | 2,102 |
| [juce/src/SampleSettings.cpp](../juce/src/SampleSettings.cpp) | 2,022 |
| [juce/src/AssetManagerComponent.cpp](../juce/src/AssetManagerComponent.cpp) | 1,871 |
| [juce/src/PianoRollComponent.h](../juce/src/PianoRollComponent.h) | 1,602 |
| [juce/src/ProjectModel.h](../juce/src/ProjectModel.h) | 1,592 |
| [juce/src/TimelineComponent.cpp](../juce/src/TimelineComponent.cpp) | 1,224 |

完整扫描指纹及接口条目：[interface-audit-inventory.json](interface-audit-inventory.json)。行数用于导航和识别职责集中，不是性能或质量评分。
