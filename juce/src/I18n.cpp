#include "I18n.h"
#include <array>
#include <unordered_map>
#include <atomic>
#include <algorithm>
#include <vector>

namespace hachi
{
namespace
{
using Row = std::array<const char*, 5>;
const std::unordered_map<std::string, Row> strings {
    { "tempo.recognise", { "识别曲速", "辨識曲速", "テンポを解析", "템포 분석", "Detect tempo" } },
    { "tempo.close", { "关闭", "關閉", "閉じる", "닫기", "Close" } },
    { "tempo.mergedAudio", { "合并伴奏音频", "合併伴奏音訊", "結合された伴奏", "병합된 반주", "Merged accompaniment" } },
    { "tempo.preparing", { "正在读取音频…", "正在讀取音訊…", "音声を読み込み中…", "오디오 읽는 중…", "Reading audio…" } },
    { "tempo.running", { "正在识别曲速…", "正在辨識曲速…", "テンポを解析中…", "템포 분석 중…", "Detecting tempo…" } },
    { "tempo.readOnly", { "仅显示分析结果，不修改工程曲速、音频位置或长度。", "僅顯示分析結果，不修改工程曲速、音訊位置或長度。", "解析結果の表示のみ。プロジェクトのテンポ、音声の位置と長さは変更しません。", "분석 결과만 표시하며 프로젝트 템포, 오디오 위치와 길이는 변경하지 않습니다.", "Displays results only. Project tempo, audio positions and lengths stay unchanged." } },
    { "tempo.estimateHint", { "Beat This! small0 · 估计曲速，可能存在半速或倍速歧义。", "Beat This! small0 · 估計曲速，可能存在半速或倍速歧義。", "Beat This! small0 · 推定テンポ。半分または倍のテンポの可能性があります。", "Beat This! small0 · 추정 템포이며 절반 또는 두 배 템포일 수 있습니다.", "Beat This! small0 · Estimated tempo; half/double tempo ambiguity is possible." } },
    { "tempo.variable", { "拍点间隔存在明显变化，此处显示整体估计曲速。", "拍點間隔存在明顯變化，此處顯示整體估計曲速。", "拍間隔に変動があります。全体の推定テンポを表示しています。", "박자 간격에 변동이 있습니다. 전체 추정 템포를 표시합니다.", "Beat intervals vary. This is an overall tempo estimate." } },
    { "tempo.failed", { "未能识别曲速", "未能辨識曲速", "テンポを解析できませんでした", "템포를 분석하지 못했습니다", "Could not detect tempo" } },
    { "tempo.noBeats", { "音频过短或缺少足够的有效拍点，无法可靠估计曲速。", "音訊過短或缺少足夠的有效拍點，無法可靠估計曲速。", "音声が短すぎるか、有効な拍が不足しています。", "오디오가 너무 짧거나 유효한 박자가 부족합니다.", "The audio is too short or has too few usable beats to estimate tempo." } },
    { "tempo.missingModel", { "未找到已有的 Beat This! small0 模型，请检查伴奏分析资源是否完整。", "未找到既有的 Beat This! small0 模型，請檢查伴奏分析資源是否完整。", "既存の Beat This! small0 モデルが見つかりません。伴奏解析リソースを確認してください。", "기존 Beat This! small0 모델이 없습니다. 반주 분석 리소스를 확인하세요.", "The existing Beat This! small0 model is missing. Check the accompaniment analysis resources." } },
    { "tempo.missingRuntime", { "伴奏分析运行环境未找到，请检查 engines/hamood/runtime.json 与 Python 路径。", "伴奏分析執行環境未找到，請檢查 engines/hamood/runtime.json 與 Python 路徑。", "伴奏解析環境がありません。engines/hamood/runtime.json と Python パスを確認してください。", "반주 분석 실행 환경이 없습니다. engines/hamood/runtime.json 및 Python 경로를 확인하세요.", "Analysis runtime missing. Check engines/hamood/runtime.json and the Python path." } },
    { "tempo.workerError", { "曲速分析未完成，请检查音频和伴奏分析运行环境。", "曲速分析未完成，請檢查音訊和伴奏分析執行環境。", "解析に失敗しました。音声と伴奏解析環境を確認してください。", "템포 분석에 실패했습니다. 오디오와 분석 실행 환경을 확인하세요.", "Tempo analysis failed. Check the audio and accompaniment analysis runtime." } },
    { "tempo.cancelled", { "曲速分析已取消。", "曲速分析已取消。", "テンポ解析をキャンセルしました。", "템포 분석이 취소되었습니다.", "Tempo analysis cancelled." } },
    { "settings.softwareRendering", { "回退到软件绘制（关闭 GPU 界面加速）", "退回軟體繪製（關閉 GPU 介面加速）", "ソフトウェア描画に切替（GPU描画を無効化）", "소프트웨어 렌더링 사용 (GPU UI 가속 끄기)", "Use software rendering (disable GPU UI acceleration)" } },
    { "note.unpitched", { "无音高", "無音高", "無音高", "무음정", "Unpitched" } },
    { "clip.markUnpitched", { "标记无音高片段", "標記無音高片段", "無音高区間をマーク", "무음정 구간 표시", "Mark unpitched regions" } },
    { "app.title",       { "HachiShifter Next", "HachiShifter Next", "HachiShifter Next", "HachiShifter Next", "HachiShifter Next" } },
    { "file.recovery", { "恢复自动保存的工程…", "復原自動儲存的工程…", "自動保存から復元…", "자동 저장 프로젝트 복구…", "Recover Autosaved Project…" } },
    { "recovery.title", { "发现可恢复的工程", "發現可復原的工程", "復元可能なプロジェクト", "복구 가능한 프로젝트", "Recoverable projects" } },
    { "recovery.message", { "自动保存可恢复未保存的修改。恢复后作为未保存工程打开，保存时请选择位置。", "自動儲存可復原未儲存的修改。復原後請另選儲存位置。", "未保存の変更を復元できます。復元後に保存先を選んでください。", "저장하지 않은 변경 사항을 복구합니다. 복구 후 저장 위치를 선택하세요.", "Recover unsaved changes. The recovered project opens as unsaved; choose a location when saving." } },
    { "recovery.restore", { "恢复", "復原", "復元", "복구", "Recover" } },
    { "recovery.later", { "稍后处理", "稍後處理", "後で", "나중에", "Later" } },
    { "recovery.openFolder", { "打开恢复文件夹", "開啟復原資料夾", "復元フォルダーを開く", "복구 폴더 열기", "Open recovery folder" } },
    { "recovery.none", { "没有可恢复的自动保存。", "沒有可復原的自動儲存。", "復元可能な自動保存はありません。", "복구할 자동 저장이 없습니다.", "No autosaved projects are available." } },
    { "recovery.failed", { "自动保存失败，原工程未被覆盖：", "自動儲存失敗，原工程未被覆寫：", "自動保存に失敗しました。元のファイルは変更されていません：", "자동 저장 실패. 원본 프로젝트는 변경되지 않았습니다:", "Autosave failed; the original project was not overwritten:" } },
    { "recovery.restored", { "已恢复自动保存；请另存工程。", "已復原自動儲存；請另存工程。", "自動保存を復元しました。名前を付けて保存してください。", "자동 저장이 복구되었습니다. 다른 이름으로 저장하세요.", "Autosave recovered; please save the project to a new location." } },
    { "recovery.busy", { "此恢复文件正被另一编辑器使用。", "此復原檔正由另一編輯器使用。", "別のエディターがこの復元ファイルを使用中です。", "다른 편집기에서 이 복구 파일을 사용 중입니다.", "Another editor is using this recovery file." } },
    { "file.open",       { "打开工程", "開啟工程", "プロジェクトを開く", "프로젝트 열기", "Open Project" } },
    { "file.save",       { "保存工程", "儲存工程", "プロジェクトを保存", "프로젝트 저장", "Save Project" } },
    { "file.saveAs",     { "工程另存为…", "工程另存新檔…", "プロジェクトを別名で保存…", "프로젝트 다른 이름으로 저장…", "Save Project As…" } },
    { "native.range", { "范围", "範圍", "範囲", "범위", "Range" } },
    { "native.envelope", { "包络", "包絡", "エンベロープ", "엔벨로프", "Envelope" } },
    { "native.renderedWave", { "合成波形", "合成波形", "合成波形", "합성 파형", "Rendered waveform" } },
    { "native.sourceWave", { "原始波形预览", "原始波形預覽", "元波形プレビュー", "원본 파형 미리보기", "Source waveform preview" } },
    { "native.actualWave", { "渲染后真实波形", "渲染後真實波形", "レンダリング後の実波形", "렌더링된 실제 파형", "Actual rendered waveform" } },
    { "native.pitchLine", { "音高线", "音高線", "ピッチカーブ", "피치 곡선", "Pitch curve" } },
    { "native.display", { "显示", "顯示", "表示", "표시", "Display" } },
    { "native.vibrato", { "颤音", "顫音", "ビブラート", "비브라토", "Vibrato" } },
    { "native.vibratoHelp", { "选中音频片段后设置颤音；支持多选、预设和关闭颤音。", "選取音訊片段後設定顫音；支援多選、預設和關閉顫音。", "選択した音声ノートのビブラートを設定します。複数選択、プリセット、解除に対応。", "선택한 오디오 음표의 비브라토를 설정합니다. 다중 선택, 프리셋, 해제를 지원합니다.", "Set vibrato on selected audio notes; supports multiple selection, presets and disabling vibrato." } },
    { "native.velocityHelp", { "辅音速度：100 为基准，增大会缩短起音；保持音符总时长和源选区。", "子音速度：100 為基準，增大会縮短起音；保留音符總時長和來源範圍。", "子音速度：100が基準。値を上げるとアタックが短くなります。ノート全長と素材範囲は維持されます。", "자음 속도: 100이 기준입니다. 높이면 어택이 짧아지며 음표 길이와 원본 범위는 유지됩니다.", "Consonant velocity: 100 is neutral; higher values shorten the onset while retaining note duration and source range." } },
    { "file.recent",     { "最近工程", "最近工程", "最近使ったプロジェクト", "최근 프로젝트", "Recent Projects" } },
    { "file.recentEmpty", { "没有最近工程", "沒有最近工程", "最近のプロジェクトはありません", "최근 프로젝트 없음", "No Recent Projects" } },
    { "file.export",     { "导出 WAV", "匯出 WAV", "WAVを書き出す", "WAV 내보내기", "Export WAV" } },
    { "file.exportLastRender", { "导出上次渲染音频", "匯出上次算繪音訊", "前回レンダリングした音声を書き出す", "마지막 렌더링 오디오 내보내기", "Export Last Render" } },
    { "export.chooseDestination", { "下一步：保存位置…", "下一步：儲存位置…", "次へ：保存先…", "다음: 저장 위치…", "Next: save location…" } },
    { "export.advanced", { "高级", "進階", "詳細設定", "고급", "Advanced" } },
    { "export.breathOnly", { "仅导出气声", "僅匯出氣聲", "息成分のみ", "기식 성분만 내보내기", "Export breath only" } },
    { "export.nonBreathOnly", { "仅导出非气声", "僅匯出非氣聲", "非息成分のみ", "비기식 성분만 내보내기", "Export non-breath only" } },
    { "export.componentHelp", { "两项互斥，均不选则导出完整人声。气声为引擎的噪声分量，可能包含清辅音；仅对本次导出生效。", "兩項互斥，均不選則匯出完整人聲。氣聲為引擎的噪聲分量，可能包含清子音；僅對本次匯出生效。", "未選択で全音声を書き出します。息成分には無声子音も含まれます。今回の書き出しのみに適用。", "둘 다 해제하면 전체 음성을 내보냅니다. 기식에는 무성 자음도 포함됩니다. 이번 내보내기에만 적용됩니다.", "Choose one or leave both off for the complete voice. Breath is the noise component and may include unvoiced consonants. Applies to this export only." } },
    { "export.wavSettings", { "WAV 导出设置", "WAV 匯出設定", "WAV 書き出し設定", "WAV 내보내기 설정", "WAV export settings" } },
    { "export.wavHelp", { "单声道会混合左右声道；32 bit 使用浮点格式。设置会保留，并用于下次导出。", "單聲道會混合左右聲道；32 bit 使用浮點格式。設定會保留供下次匯出。", "モノラルは左右をミックスします。32 bit は浮動小数点です。設定は次回も保持されます。", "모노는 좌우 채널을 합칩니다. 32 bit는 부동소수점 형식입니다. 설정은 다음에도 유지됩니다.", "Mono mixes left and right channels. 32 bit uses floating point. Settings are remembered for the next export." } },
    { "export.channels", { "声道", "聲道", "チャンネル", "채널", "Channels" } },
    { "export.mono", { "单声道", "單聲道", "モノラル", "모노", "Mono" } },
    { "export.stereo", { "双声道（立体声）", "雙聲道（立體聲）", "ステレオ", "스테레오", "Stereo" } },
    { "export.bitDepth", { "位深", "位元深度", "ビット深度", "비트 깊이", "Bit depth" } },
    { "export.float32", { "32 bit 浮点", "32 bit 浮點", "32 bit 浮動小数点", "32 bit 부동소수점", "32 bit float" } },
    { "export.sampleRate", { "采样率", "取樣率", "サンプルレート", "샘플 레이트", "Sample rate" } },
    { "export.followDevice", { "跟随播放设备（默认）", "跟隨播放裝置（預設）", "再生デバイスに合わせる（既定）", "재생 장치에 맞춤 (기본값)", "Follow playback device (default)" } },
    { "export.lastRenderSuffix", { "上次渲染", "上次算繪", "前回のレンダリング", "마지막 렌더링", "last render" } },
    { "error.exportNoRender", { "尚未播放所选音符，没有可导出的渲染音频。", "尚未播放所選音符，沒有可匯出的算繪音訊。", "選択したノートをまだ再生していないため、書き出せるレンダリングがありません", "선택한 음표를 재생한 적이 없어 내보낼 렌더링이 없습니다", "No marquee has been played yet, so there is no render to export" } },
    { "export.allTracks", { "全部轨道（每轨一个文件）", "全部音軌（每軌一個檔案）", "全トラック（トラックごとに1ファイル）", "모든 트랙 (트랙당 파일 1개)", "All tracks (one file each)" } },
    { "export.oneTrack", { "单个轨道", "單一音軌", "単一トラック", "단일 트랙", "A single track" } },
    { "export.untitledTrack", { "未命名轨道", "未命名音軌", "名称未設定トラック", "이름 없는 트랙", "Untitled track" } },
    { "status.exportRendering", { "正在渲染全曲音符…", "正在算繪全曲音符…", "曲全体のノートをレンダリング中…", "곡 전체 노트 렌더링 중…", "Rendering every note…" } },
    { "status.exportDone", { "已导出", "已匯出", "書き出しました", "내보냈습니다", "Exported" } },
    { "status.exportFiles", { "个文件", "個檔案", "ファイル", "개 파일", "file(s)" } },
    { "error.exportSilent", { "该轨道被静音或未独奏，导出会是静音", "該音軌被靜音或未獨奏，匯出會是靜音", "このトラックはミュート/非ソロのため無音になります", "이 트랙은 음소거 상태여서 무음으로 내보내집니다", "That track is muted or not soloed, so it would export silence" } },
    { "error.exportEmpty", { "没有可导出的轨道", "沒有可匯出的音軌", "書き出せるトラックがありません", "내보낼 트랙이 없습니다", "There is no track to export" } },
    { "file.audio",      { "导入音频", "匯入音訊", "オーディオを読み込む", "오디오 가져오기", "Import Audio" } },
    { "file.melodyne",   { "导入 Melodyne", "匯入 Melodyne", "Melodyneを読み込む", "Melodyne 가져오기", "Import Melodyne" } },
    { "file.new",        { "新建工程", "新增工程", "新規プロジェクト", "새 프로젝트", "New Project" } },
    { "file.midi",       { "导入 MIDI", "匯入 MIDI", "MIDIを読み込む", "MIDI 가져오기", "Import MIDI" } },
    { "file.ust",        { "导入 UST", "匯入 UST", "USTを読み込む", "UST 가져오기", "Import UST" } },
    { "file.exportMidi", { "导出 MIDI", "匯出 MIDI", "MIDIを書き出す", "MIDI 내보내기", "Export MIDI" } },
    { "error.exportMidi", { "无法导出 MIDI", "無法匯出 MIDI", "MIDIを書き出せません", "MIDI를 내보낼 수 없습니다", "Could not export the MIDI" } },
    { "status.midiExported", { "MIDI 已导出", "MIDI 已匯出", "MIDIを書き出しました", "MIDI를 내보냈습니다", "MIDI exported" } },
    { "error.ust",       { "无法读取 UST 工程", "無法讀取 UST 專案", "USTを読み込めません", "UST를 읽을 수 없습니다", "Could not read the UST" } },
    { "status.ustLoaded", { "UST 已导入为 UTAU 轨道", "UST 已匯入為 UTAU 軌道", "USTをUTAUトラックとして読み込みました", "UST를 UTAU 트랙으로 가져왔습니다", "UST imported as a UTAU track" } },
    { "file.exit",       { "退出", "結束", "終了", "종료", "Exit" } },
    { "file.settings",   { "设置…", "設定…", "設定…", "설정…", "Settings…" } },
    { "file.assets",     { "素材管理器…", "素材管理器…", "素材マネージャー…", "소재 관리자…", "Asset Manager…" } },
    { "material.pronunciationInput", { "发音输入…（Ctrl+Enter）", "發音輸入…（Ctrl+Enter）", "発音を入力…（Ctrl+Enter）", "발음 입력… (Ctrl+Enter)", "Pronunciation Input… (Ctrl+Enter)" } },
    { "draw.action.pitch", { "绘制音高线（右键切换）", "繪製音高線（右鍵切換）", "音高線を描く（右クリックで切替）", "음높이 곡선 그리기 (우클릭 전환)", "Draw pitch (right-click to switch)" } },
    { "draw.action.notes", { "创建音符（右键切换）", "建立音符（右鍵切換）", "ノートを作成（右クリックで切替）", "음표 만들기 (우클릭 전환)", "Create notes (right-click to switch)" } },
    { "draw.noteLength", { "输入音符的最小长度", "輸入音符的最小長度", "入力ノートの最小長", "입력 음표 최소 길이", "Minimum input note length" } },
    { "menu.file",       { "文件", "檔案", "ファイル", "파일", "File" } },
    { "menu.edit",       { "编辑", "編輯", "編集", "편집", "Edit" } },
    { "menu.track",      { "轨道", "軌道", "トラック", "트랙", "Track" } },
    { "menu.view",       { "视图", "檢視", "表示", "보기", "View" } },
    { "menu.help",       { "帮助", "說明", "ヘルプ", "도움말", "Help" } },
    { "edit.undo",       { "撤销", "復原", "元に戻す", "실행 취소", "Undo" } },
    { "edit.redo",       { "重做", "重做", "やり直す", "다시 실행", "Redo" } },
    { "edit.selectAll",  { "全选音符", "全選音符", "全ノートを選択", "모든 노트 선택", "Select All Notes" } },
    { "edit.deselect", { "取消选择", "取消選取", "選択解除", "선택 해제", "Deselect" } },
    { "edit.copyNotes", { "复制所选音符", "複製所選音符", "選択ノートをコピー", "선택 음표 복사", "Copy Selected Notes" } },
    { "edit.cutNotes", { "剪切所选音符", "剪下所選音符", "選択ノートを切り取り", "선택 음표 잘라내기", "Cut Selected Notes" } },
    { "edit.pasteNotes", { "在播放位置粘贴音符", "在播放位置貼上音符", "再生位置にノートを貼り付け", "재생 위치에 음표 붙여넣기", "Paste Notes at Playhead" } },
    { "edit.pasteNotesAtOrigin", { "粘贴到原时间位置", "貼上到原時間位置", "元の時間位置に貼り付け", "원래 시간 위치에 붙여넣기", "Paste Notes at Original Time" } },
    { "edit.transposeCents", { "按音分移调…", "按音分移調…", "セントで移調…", "센트 단위 조옮김…", "Transpose by Cents…" } },
    { "edit.setPitch", { "设置音高…", "設定音高…", "ピッチを設定…", "피치 설정…", "Set Pitch…" } },
    { "edit.averagePitch", { "平均所选音高", "平均所選音高", "選択ピッチを平均化", "선택 피치 평균", "Average Selected Pitch" } },
    { "edit.quantizePitch", { "量化到半音", "量化到半音", "半音にクオンタイズ", "반음으로 퀀타이즈", "Quantize to Semitone" } },
    { "edit.hanziToPinyin", { "汉字转拼音", "漢字轉拼音", "漢字をピンインに変換", "한자를 병음으로 변환", "Convert Chinese Lyrics to Pinyin" } },
    { "edit.cents", { "音分", "音分", "セント", "센트", "Cents" } },
    { "edit.midiNote", { "MIDI 音高", "MIDI 音高", "MIDI ノート", "MIDI 음높이", "MIDI Note" } },
    { "edit.copyClip",   { "复制所选片段", "複製所選片段", "選択クリップをコピー", "선택 클립 복사", "Copy Selected Clip" } },
    { "edit.pasteClip",  { "在播放位置粘贴片段", "在播放位置貼上片段", "再生位置にクリップを貼り付け", "재생 위치에 클립 붙여넣기", "Paste Clip at Playhead" } },
    { "edit.duplicateClip", { "在所选片段后创建副本", "在所選片段後建立副本", "選択クリップを直後に複製", "선택 클립 바로 뒤에 복제", "Duplicate Selected Clip" } },
    { "view.zoomIn",     { "放大", "放大", "拡大", "확대", "Zoom In" } },
    { "view.zoomOut",    { "缩小", "縮小", "縮小", "축소", "Zoom Out" } },
    { "view.zoomFit",    { "缩放至整个工程", "縮放至整個工程", "プロジェクト全体", "프로젝트 맞춤", "Fit Project" } },
    { "view.showWaveforms", { "显示波形", "顯示波形", "波形を表示", "파형 표시", "Show Waveforms" } },
    { "view.showLyrics", { "显示歌词", "顯示歌詞", "歌詞を表示", "가사 표시", "Show Lyrics" } },
    { "view.phonemeInfo", { "拼字信息", "拼字資訊", "音素情報", "음소 정보", "Phoneme information" } },
    { "native.nsfPitchTransitions", { "音高突变平滑", "音高突變平滑", "音高ジャンプを平滑化", "급격한 음높이 변화 완화", "Smooth pitch jumps" } },
    { "native.nsfPitchTransitionsHelp", { "当前轨道的 NSF 渲染设置。开启时平滑音高突变以抑制爆音；关闭时保留编辑曲线的突变。清辅音保护由谐波/噪声分离开关独立控制。", "目前軌道的 NSF 渲染設定。開啟時平滑音高突變以抑制爆音；關閉時保留編輯曲線的突變。清輔音保護由諧波/雜訊分離開關獨立控制。", "急な音高変化を平滑化します。オフでは編集した音高ジャンプを保持します。無声子音の保護は調波・ノイズ分離の設定で独立に切り替えます。", "급격한 음높이 변화를 완화합니다. 끄면 편집한 변화를 유지합니다. 무성 자음 보호는 배음/잡음 분리 설정으로 따로 제어합니다.", "On smooths pitch jumps to reduce clicks; off retains authored jumps. Unvoiced protection is controlled independently by Harmonic/noise." } },
    { "native.nsfNoiseProtection", { "谐波/噪声分离", "諧波/雜訊分離", "調波・ノイズ分離", "배음/잡음 분리", "Harmonic/noise" } },
    { "native.nsfNoiseProtectionHelp", { "当前非 U NSF 轨道，默认开启：谐波经过变调，源噪声按拉伸和时间映射混回，明确清音恢复原始波形。关闭后对完整音频变调。缺少分离模型时仅保留明确清音保护；局部渲染不受此开关影响。", "目前非 U NSF 軌道，預設開啟：諧波經過變調，原雜訊依拉伸及時間映射混回，明確清音恢復原始波形。關閉後對完整音訊變調。缺少分離模型時僅保留明確清音保護；局部渲染不受影響。", "非 U NSF トラック：調波のみを変調し、元のノイズを時間マップに合わせて戻し、明確な無声音を元の波形で復元します。オフでは音声全体を変調します。分離モデルがない場合は無声音保護のみ。部分レンダリングは独立です。", "비 U NSF 트랙: 배음을 변조하고 시간 매핑에 맞춰 원본 잡음을 섞으며 명확한 무성음은 원본 파형으로 복원합니다. 끄면 전체 오디오를 변조합니다. 분리 모델이 없으면 무성음만 보호합니다. 부분 렌더링은 별도입니다.", "Native NSF, on by default: transpose harmonics, mix original noise through the time map and restore clear unvoiced PCM. Off transposes the full audio. Without the separation model only clear-unvoiced protection is available. Incremental rendering remains enabled." } },
    { "native.allowOverlap", { "允许音频重叠", "允許音訊重疊", "オーディオの重なりを許可", "오디오 겹침 허용", "Allow audio overlap" } },
    { "native.overlapHelp", { "当前轨道的音频重叠开关。默认禁止重叠；关闭时前段不变，裁掉后段重叠的开头并保留其结束位置。可一次撤销。", "目前軌道的音訊重疊開關。關閉時前段不變，裁掉後段重疊的開頭並保留結束位置。可復原。", "現在のトラックの重なり設定。オフにすると前の素材を保持し、後の素材の重なる先頭を切り取ります。終了位置は維持します。", "현재 트랙의 겹침 설정. 끄면 앞 소재는 유지하고 뒤 소재의 겹치는 시작 부분을 잘라냅니다. 끝 위치는 유지됩니다.", "Current track: off prevents overlap. Turning off trims the later source's overlapping head, keeps its end and leaves the earlier source unchanged. One undo restores the edit." } },
    { "native.overlap", { "重叠", "重疊", "重なり", "겹침", "Overlap" } },
    { "native.overlapMix", { "同时发声", "同時發聲", "同時再生", "동시 재생", "Simultaneous audio" } },
    { "native.overlapEarlier", { "前段", "前段", "先の素材", "앞 소재", "Earlier" } },
    { "native.overlapLater", { "后段", "後段", "後の素材", "뒤 소재", "Later" } },
    { "native.linkAudio", { "粘连音频", "黏連音訊", "オーディオを連結する", "오디오 연결", "Link Audio" } },
    { "native.disconnectAudio", { "断开音频", "斷開音訊", "オーディオを切り離す", "오디오 연결 해제", "Disconnect Audio" } },
    { "native.restoreOriginalPitch", { "还原为原始素材音高", "還原為原始素材音高", "元の素材のピッチに戻す", "원본 소재 피치로 복원", "Restore Original Source Pitch" } },
    { "native.originalPitchLine", { "显示原始音高线（虚线）", "顯示原始音高線（虛線）", "元ピッチを表示（破線）", "원본 피치 표시 (점선)", "Show Original Pitch (Dashed)" } },
    { "native.consonantPitchDashed", { "气声虚线", "氣聲虛線", "息成分を破線表示", "숨소리 점선 표시", "Dashed breathiness" } },
    { "view.vZoomIn", { "纵向放大", "縱向放大", "縦方向に拡大", "세로 확대", "Zoom In Vertically" } },
    { "view.vZoomOut", { "纵向缩小", "縱向縮小", "縦方向に縮小", "세로 축소", "Zoom Out Vertically" } },
    { "help.about",      { "关于 HachiShifter", "關於 HachiShifter", "HachiShifterについて", "HachiShifter 정보", "About HachiShifter" } },
    { "help.aboutText",  { "HachiShifter Next · JUCE/C++ 原生重构版\n界面图标：Tabler Icons（MIT 许可证）", "HachiShifter Next · JUCE/C++ 原生重構版\n介面圖示：Tabler Icons（MIT 授權）", "HachiShifter Next · JUCE/C++ ネイティブ版\nUI アイコン：Tabler Icons（MIT ライセンス）", "HachiShifter Next · JUCE/C++ 네이티브 버전\nUI 아이콘: Tabler Icons (MIT 라이선스)", "HachiShifter Next · Native JUCE/C++ edition\nUI icons: Tabler Icons (MIT license)" } },
    { "transport.play",  { "播放", "播放", "再生", "재생", "Play" } },
    { "transport.pause", { "暂停", "暫停", "一時停止", "일시 정지", "Pause" } },
    { "transport.stop",  { "停止", "停止", "停止", "정지", "Stop" } },
    { "tool.main",       { "音符编辑", "音符編輯", "ノート編集", "노트 편집", "Note Edit" } },
    { "tool.wrench",     { "采样精修", "取樣精修", "サンプル編集", "샘플 정밀 편집", "Sample Edit" } },
    { "tool.draw",       { "自由绘制", "自由繪製", "フリーハンド", "자유 그리기", "Free Draw" } },
    { "tool.line",       { "直线工具", "直線工具", "直線ツール", "직선 도구", "Line Tool" } },
    { "tool.points",     { "标点音高工具", "標點音高工具", "ピッチポイント", "피치 포인트", "Pitch Points" } },
    { "tool.nativeLoudness", { "响度", "響度", "音量", "음량", "Loudness" } },
    { "tool.nativeLoudnessHint", { "响度：在空白处框选多个音符，再按住选中音符上下拖动，统一增减响度；波形同步预览，松开应用，Esc 取消。", "響度：框選多個音符後上下拖動選中音符，統一增減響度；鬆開套用，Esc 取消。", "ノートを上下にドラッグして音量を調整。Esc で取消。", "음표를 위아래로 드래그하여 음량 조절. Esc 취소.", "Box-select notes on empty space, then drag a selected note up or down to adjust their loudness together. Release to apply, Esc to cancel." } },
    { "tool.trim", { "裁剪音频（仅非 U 模式）：拖动两端裁剪，不拉伸、不吸附网格；再次点击退出", "裁剪音訊（僅非 U 模式）：拖動兩端裁剪，不伸縮、不吸附網格；再次點擊退出", "オーディオの端をトリム（非UTAU、スナップなし）", "오디오 끝 자르기 (비 UTAU, 스냅 없음)", "Trim audio edges (non-UTAU): no stretching or grid snapping; click again to exit" } },
    { "stretch.unit", { "最小拉伸", "最小拉伸", "最小ストレッチ", "최소 늘이기", "Min Stretch" } },
    { "stretch.unitHelp", { "拉伸和移动音符时的最小单位，为一拍的 1/N。默认 1/64。", "拉伸和移動音符時的最小單位，為一拍的 1/N。預設 1/64。", "ノートの伸縮と移動の最小単位（1拍の 1/N）。既定は 1/64。", "노트 늘이기와 이동의 최소 단위(한 박의 1/N). 기본값 1/64.", "Smallest unit for stretching and moving notes, as 1/N of a beat. Default 1/64." } },
    { "pitchCurve.incomingSegment", { "上一点 → 当前点", "上一點 → 目前點", "前の点 → 現在の点", "이전 점 → 현재 점", "Previous Point → This Point" } },
    { "pitchCurve.noPrevious", { "音头点没有上一段", "音頭點沒有上一段", "先頭点には前の区間がありません", "첫 점에는 이전 구간이 없습니다", "The first point has no incoming segment" } },
    { "pitchCurve.natural", { "自然连续（默认）", "自然連續（預設）", "自然連続（既定）", "자연스러운 연속 (기본값)", "Natural Continuous (Default)" } },
    { "pitchCurve.linear", { "直线", "直線", "直線", "직선", "Linear" } },
    { "pitchCurve.smooth", { "S 型平滑", "S 型平滑", "S字スムーズ", "S자 부드럽게", "Smooth S" } },
    { "pitchCurve.easeIn", { "J形曲线（前慢后快）", "J形曲線（前慢後快）", "J字曲線（前半ゆっくり・後半速く）", "J자 곡선 (앞은 느리게, 뒤는 빠르게)", "J Curve (Slow Then Fast)" } },
    { "pitchCurve.easeOut", { "R形曲线（前快后慢）", "R形曲線（前快後慢）", "R字曲線（前半速く・後半ゆっくり）", "R자 곡선 (앞은 빠르게, 뒤는 느리게)", "R Curve (Fast Then Slow)" } },
    { "pitchCurve.customBezier", { "自定义贝塞尔…", "自訂貝茲曲線…", "カスタムベジェ…", "사용자 베지어…", "Custom Bezier…" } },
    { "pitchCurve.snapToNote", { "调整到标准音高", "調整到標準音高", "標準ピッチに合わせる", "표준 음높이로 맞추기", "Snap to Note Pitch" } },
    { "pitchCurve.setFrequency", { "输入音高…", "輸入音高…", "ピッチを入力…",
                                   "음높이 입력…", "Set Frequency…" } },
    { "pitchCurve.frequencyTitle", { "输入音高", "輸入音高", "ピッチを入力",
                                     "음높이 입력", "Set Point Frequency" } },
    { "pitchCurve.frequencyHelp", {
        "直接输入这个标点的基频 F0，单位赫兹（Hz）。440 即标准音 A4。\n只改这一个标点，相邻标点和音符本身都不动。",
        "直接輸入這個標點的基頻 F0，單位赫茲（Hz）。440 即標準音 A4。\n只改這一個標點，相鄰標點和音符本身都不動。",
        "このポイントの基本周波数 F0 をヘルツ（Hz）で直接入力します。440 が基準音 A4 です。\nこのポイントだけが動き、隣のポイントもノート自体も変わりません。",
        "이 점의 기본 주파수 F0을 헤르츠(Hz)로 직접 입력합니다. 440이 표준음 A4입니다.\n이 점만 바뀌고 이웃한 점과 노트 자체는 그대로입니다.",
        "Type this point's fundamental frequency in hertz; 440 is concert A4.\nOnly this point moves -- its neighbours and the note itself stay put." } },
    { "pitchCurve.frequencyField", { "F0（Hz）", "F0（Hz）", "F0（Hz）",
                                     "F0 (Hz)", "F0 (Hz)" } },
    { "pitchCurve.deletePoint", { "删除此标点", "刪除此標點", "このポイントを削除", "이 점 삭제", "Delete This Point" } },
    { "pitchCurve.bezierTitle", { "自定义贝塞尔曲线", "自訂貝茲曲線", "カスタムベジェ曲線", "사용자 베지어 곡선", "Custom Bezier Curve" } },
    { "pitchCurve.bezierHelp", { "设置归一化控制点。X 控制变化发生的时间，Y 控制音高推进量；Y 超出 0–1 可产生转音过冲。", "設定正規化控制點。X 控制變化時間，Y 控制音高推進量；Y 超出 0–1 可產生轉音過衝。", "正規化した制御点を設定します。X は変化のタイミング、Y はピッチの進行量です。0～1 外の Y でオーバーシュートを作れます。", "정규화된 제어점을 설정합니다. X는 변화 시점, Y는 피치 진행량이며 0–1 밖의 Y로 오버슈트를 만들 수 있습니다.", "Set normalised control points. X controls timing and Y controls pitch progress; Y outside 0–1 creates overshoot." } },
    { "pitchCurve.bezierX1", { "控制点 1 X", "控制點 1 X", "制御点 1 X", "제어점 1 X", "Control 1 X" } },
    { "pitchCurve.bezierY1", { "控制点 1 Y", "控制點 1 Y", "制御点 1 Y", "제어점 1 Y", "Control 1 Y" } },
    { "pitchCurve.bezierX2", { "控制点 2 X", "控制點 2 X", "制御点 2 X", "제어점 2 X", "Control 2 X" } },
    { "pitchCurve.bezierY2", { "控制点 2 Y", "控制點 2 Y", "制御点 2 Y", "제어점 2 Y", "Control 2 Y" } },
    { "tool.connect",    { "连接/分离音符", "連接/分離音符", "ノート接続/分離", "노트 연결/분리", "Connect/Separate Notes" } },
    { "editor.parameters", { "参数编辑器", "參數編輯器", "パラメータ", "매개변수 편집기", "Parameters" } },
    { "editor.smooth",   { "平滑", "平滑", "平滑化", "평활", "Smooth" } },
    { "editor.drift",    { "漂移修正", "漂移修正", "ドリフト補正", "드리프트 보정", "Drift Correction" } },
    { "editor.attackSpeed", { "辅音速度", "子音速度", "アタックスピード", "어택 속도", "Attack Speed" } },
    { "param.pitch",     { "音高", "音高", "ピッチ", "피치", "Pitch" } },
    { "param.drift",     { "漂移", "漂移", "ドリフト", "드리프트", "Drift" } },
    { "param.attack",    { "起音", "起音", "アタック", "어택", "Attack" } },
    { "param.breath",    { "呼吸", "呼吸", "ブレス", "브레스", "Breath" } },
    { "param.tension",   { "张力", "張力", "テンション", "텐션", "Tension" } },
    { "param.formant",   { "共振峰", "共振峰", "フォルマント", "포먼트", "Formant" } },
    { "param.volume",    { "音量", "音量", "音量", "음량", "Volume" } },
    { "param.robustPitchCurveShort", { "稳健线", "穩健線", "ロバスト", "강건선", "Robust" } },
    { "param.robustPitchCurve", { "稳健音高线（仅当前音符）", "穩健音高線（僅目前音符）", "ロバストピッチカーブ（現在のノートのみ）", "강건한 피치 곡선 (현재 음표만)", "Robust Pitch Curve (current note only)" } },
    { "beats.bar",       { "每小节", "每小節", "拍子", "마디 박자", "Beats" } },
    { "grid",            { "网格", "網格", "グリッド", "그리드", "Grid" } },
    { "base.scale",      { "基准调", "基準調", "基準キー", "기준 키", "Key" } },
    { "algo.pitch",      { "变调算法", "變調演算法", "ピッチアルゴリズム", "피치 알고리즘", "Pitch Algorithm" } },
    { "algo.stretch",    { "拉伸算法", "拉伸演算法", "タイムアルゴリズム", "타임 알고리즘", "Stretch Algorithm" } },
    { "algo.stretch.wcsNonLoopHelp", { "沿原素材时间向前插值，不循环重复尾部，不自动固定辅音时长。", "沿原素材時間向前插值，不循環重複尾部，不自動固定子音時長。", "元音声の時間を前向きに補間します。末尾をループせず、子音の長さも自動固定しません。", "원본 시간을 순방향으로 보간합니다. 꼬리를 반복하거나 자음 길이를 자동으로 고정하지 않습니다.", "Interpolate forward through the source without looping its tail or automatically fixing consonant duration." } },
    { "algo.stretch.wcsNonLoop", { "WCSNDM 非循环拉伸", "WCSNDM 非循環拉伸", "WCSNDM 非ループ伸縮", "WCSNDM 비루프 스트레치", "WCSNDM non-loop stretch" } },
    { "algo.stretch.hifiShifterMel", { "HiFiShifter Mel 线性拉伸", "HiFiShifter Mel 線性拉伸", "HiFiShifter Mel 線形伸縮", "HiFiShifter Mel 선형 스트레치", "HiFiShifter linear Mel stretch" } },
    { "algo.stretch.melodyneHybrid", { "算法原生拉伸", "演算法原生拉伸", "エンジン標準伸縮", "엔진 기본 타임 스트레치", "Engine-native stretch" } },
    { "algo.stretch.nsfVariableMel", { "NSF 可变 Hop Mel 先拼接后合成", "NSF 可變 Hop Mel 先拼接後合成", "NSF 可変 Hop Mel 結合後合成", "NSF 가변 Hop Mel 연결 후 합성", "NSF Variable-Hop Mel: Splice then Synthesize" } },
    { "algo.stretch.nsfShiftThenSplice", { "NSF 先变调后拼接", "NSF 先變調後拼接", "NSF ピッチ後結合", "NSF 피치 먼저 연결", "NSF Shift then Splice" } },
    { "algo.stretch.loop", { "循环拉伸", "循環拉伸", "ループストレッチ", "루프 스트레치", "Loop Stretch" } },
    { "algo.stretch.soundTouch", { "SoundTouch 拉伸", "SoundTouch 拉伸", "SoundTouch ストレッチ", "SoundTouch 스트레치", "SoundTouch Stretch" } },
    { "algo.order",      { "处理顺序", "處理順序", "処理順", "처리 순서", "Render Order" } },
    { "algo.order.processThenSplice", { "先合成后拼接", "先合成後拼接", "合成後結合", "합성 후 연결", "Process then Splice" } },
    { "algo.order.stretchSpliceThenPitch", { "先拼接后合成", "先拼接後合成", "結合後合成", "연결 후 합성", "Splice then Pitch" } },
    { "algo.order.processMergeContiguous", { "先合成后拼接（同源连续组）", "先合成後拼接（同源連續組）", "合成後結合（同一音源を連続処理）", "합성 후 연결 (동일 소스 연속)", "Process then Splice (source continuity)" } },
    { "track.compose",   { "旋律", "旋律", "メロディック", "멜로디", "Compose" } },
    { "track.toggleCompose", { "切换旋律/普通音轨", "切換旋律/一般音軌", "メロディック/通常を切替", "멜로디/일반 전환", "Toggle Compose/Audio" } },
    { "track.addCompose", { "添加旋律轨道", "新增旋律軌道", "メロディックトラックを追加", "멜로디 트랙 추가", "Add Melodic Track" } },
    { "track.addAudio", { "添加普通音轨", "新增一般音軌", "通常トラックを追加", "일반 오디오 트랙 추가", "Add Audio Track" } },
    { "track.accompaniment", { "伴奏轨道", "伴奏軌道", "伴奏トラック", "반주 트랙", "Accompaniment" } },
    { "track.newAccompaniment", { "新建伴奏轨道", "新建伴奏軌道", "伴奏トラックを新規作成", "새 반주 트랙", "New Accompaniment Track" } },
    { "track.originalAudio", { "原音播放", "原音播放", "原音再生", "원음 재생", "Original audio" } },
    { "track.noProcessing", { "不启用", "不啟用", "無効", "사용 안 함", "Disabled" } },
    { "track.accompanimentHelp", { "伴奏轨道直接播放原音，不进行音高分析、变调或人声合成；参与正常播放与导出。", "伴奏軌道直接播放原音，不進行音高分析、變調或人聲合成；參與正常播放與匯出。", "伴奏は原音を再生します。ピッチ解析・変換・歌声合成を使用せず、通常のミックスに含まれます。", "반주는 음높이 분석이나 보컬 합성 없이 원음을 재생하며 믹스에 포함됩니다.", "Plays the original audio without pitch analysis, tuning or vocal synthesis. Included in playback and export." } },
    { "track.newHere", { "新建轨道", "新建軌道", "トラックを新規作成", "새 트랙", "New Track" } },
    { "track.newReference", { "新建素材轨道", "新建素材軌道", "素材トラックを新規作成",
                              "소재 트랙 새로 만들기", "New Reference Track" } },
    { "track.importMidi", { "导入 MIDI 轨道", "匯入 MIDI 軌道", "MIDIトラックを読み込む",
                            "MIDI 트랙 가져오기", "Import MIDI Track" } },
    { "utau.chooseVoicebank", { "选择音源库…", "選擇音源庫…", "音源を選択…", "음원 선택…", "Choose voicebank…" } },
    { "utau.noVoicebankSelected", { "未选择音源", "未選擇音源", "音源未選択", "선택한 음원 없음", "No voicebank selected" } },
    { "utau.editVoicebankHint", { "双击打开当前轨道模式的 OTO 编辑器", "按兩下開啟目前音軌模式的 OTO 編輯器", "ダブルクリックで現在のトラックの OTO を編集", "두 번 클릭하여 현재 트랙의 OTO 편집", "Double-click to edit the OTO for the current track mode" } },
    { "utau.chooseVoicebankHint", { "选择当前音轨使用的音源文件夹。", "選擇目前音軌使用的音源資料夾。", "現在のトラックで使う音源フォルダーを選択します。", "현재 트랙에 사용할 음원 폴더를 선택합니다.", "Choose the voicebank folder for the current track." } },
    { "track.importMidiHere", { "导入 MIDI…", "匯入 MIDI…", "MIDIを読み込む…", "MIDI 가져오기…", "Import MIDI…" } },
    { "dialog.midiTrackHereMessage", { "选择要导入的 MIDI 轨道，在当前音轨的红线处新增区域。保留 MIDI 开头空白，使用工程速度。", "選擇要匯入的 MIDI 軌道，在目前音軌的紅線處新增區域。保留 MIDI 開頭空白，使用工程速度。", "MIDIトラックを選択し、再生ヘッド位置に新しいリージョンとして読み込みます。先頭の休符を保持し、プロジェクトのテンポを使用します。", "MIDI 트랙을 선택하여 재생 헤드 위치에 새 영역으로 가져옵니다. 시작 쉼표를 유지하고 프로젝트 템포를 사용합니다.", "Choose a MIDI track to add as a region at the playhead. Preserve leading rests and use the project tempo." } },
    { "status.midiTrackImported", { "已导入 MIDI 轨道：", "已匯入 MIDI 軌道：", "MIDIトラックを読み込みました：",
                                    "MIDI 트랙을 가져왔습니다: ", "MIDI track imported: " } },
    { "track.rename", { "重命名所选轨道…", "重新命名所選軌道…", "選択トラック名を変更…", "선택 트랙 이름 바꾸기…", "Rename Selected Track…" } },
    { "track.name", { "轨道名称", "軌道名稱", "トラック名", "트랙 이름", "Track Name" } },
    { "track.delete",    { "删除所选轨道", "刪除所選軌道", "選択トラックを削除", "선택 트랙 삭제", "Delete Selected Track" } },
    { "clip.delete",     { "删除所选片段", "刪除所選片段", "選択クリップを削除", "선택 클립 삭제", "Delete Selected Clip" } },
    { "clip.split",      { "拆分", "拆分", "分割", "분할", "Split" } },
    { "clip.editLength", { "编辑…", "編輯…", "編集…", "편집…", "Edit…" } },
    { "clip.outputEngine", { "输出引擎（整条音轨）", "輸出引擎（整條音軌）", "出力エンジン（トラック全体）", "출력 엔진 (트랙 전체)", "Output engine (whole track)" } },
    { "clip.configureResampler", { "配置重采样器…", "設定重取樣器…", "リサンプラーを設定…", "리샘플러 설정…", "Configure resampler…" } },
    { "clip.lengthSeconds", { "长度（秒）", "長度（秒）", "長さ（秒）", "길이 (초)", "Length (seconds)" } },
    { "clip.lengthHint", { "仅调整可编辑区域的末端，音符位置和时长保持不变。最短 0.01 秒。", "僅調整可編輯區域的末端，音符位置和時長保持不變。最短 0.01 秒。", "編集範囲の終端のみ変更します。ノートの位置と長さは維持します。最小 0.01 秒。", "편집 영역의 끝만 변경합니다. 음표 위치와 길이는 유지됩니다. 최소 0.01초.", "Change only the editable region end. Note positions and durations stay unchanged. Minimum: 0.01 seconds." } },
    { "clip.lengthInvalid", { "无法应用此长度。请输入至少 0.01 秒的有效数字。", "無法套用此長度。請輸入至少 0.01 秒的有效數字。", "長さを適用できません。0.01 秒以上の数値を入力してください。", "길이를 적용할 수 없습니다. 0.01초 이상의 값을 입력하세요.", "Cannot apply this length. Enter a valid number of at least 0.01 seconds." } },
    { "clip.addEmptyTuning", { "添加空白调音片段（10 秒）", "新增空白調音片段（10 秒）", "空の調声クリップを追加（10 秒）", "빈 튜닝 클립 추가 (10초)", "Add Empty Tuning Clip (10 s)" } },
    { "clip.startNormalDisplay", { "开启正常显示", "開啟正常顯示", "通常表示を開始", "일반 표시 시작", "Show Normally" } },
    { "clip.stopNormalDisplay", { "关闭正常显示", "關閉正常顯示", "通常表示を終了", "일반 표시 끄기", "Stop Showing Normally" } },
    { "clip.startNoteHints", { "开启提示显示", "開啟提示顯示", "ノートガイドを表示", "음표 가이드 표시", "Show Note Hints" } },
    { "clip.stopNoteHints", { "关闭提示显示", "關閉提示顯示", "ノートガイドを非表示", "음표 가이드 숨기기", "Hide Note Hints" } },
    { "clip.merge",      { "合并", "合併", "結合", "병합", "Merge" } },
    { "clip.muteToggle", { "静音", "靜音", "ミュート", "음소거", "Mute" } },
    { "clip.mute",       { "静音所选片段", "靜音所選片段", "選択クリップをミュート", "선택 클립 음소거", "Mute Selected Clip" } },
    { "clip.unmute",     { "取消片段静音", "取消片段靜音", "クリップのミュート解除", "클립 음소거 해제", "Unmute Clip" } },
    { "clip.gain",       { "设置片段增益…", "設定片段增益…", "クリップゲインを設定…", "클립 게인 설정…", "Set Clip Gain…" } },
    { "clip.gainDb",     { "增益（dB）", "增益（dB）", "ゲイン（dB）", "게인 (dB)", "Gain (dB)" } },
    { "gain.inputHint", { "输入 dB 数值，支持小数；-inf 表示静音。允许范围：", "輸入 dB 數值，支援小數；-inf 表示靜音。允許範圍：", "dB 値を入力。-inf は無音。範囲：", "dB 값을 입력하세요. -inf는 무음입니다. 범위:", "Enter dB, including decimals; -inf means silence. Range:" } },
    { "track.pan", { "声像", "聲像", "パン", "팬", "Pan" } },
    { "pan.inputHint", { "输入 -1 到 1 的数值，支持小数。-1 最左，0 居中，1 最右。", "輸入 -1 到 1 的數值，支援小數。-1 最左，0 居中，1 最右。", "-1～1 の値を入力。-1 は左、0 は中央、1 は右。", "-1~1 값을 입력하세요. -1 왼쪽, 0 중앙, 1 오른쪽.", "Enter a value from -1 to 1, including decimals. -1 left, 0 centre, 1 right." } },
    { "clip.gainKnobHint", { "片段响度：上下拖动，Shift 精细调节，双击恢复 0 dB。",
                            "片段響度：上下拖動，Shift 精細調節，雙擊恢復 0 dB。",
                            "クリップ音量：上下にドラッグ、Shift で微調整、ダブルクリックで 0 dB。",
                            "클립 음량: 위아래로 드래그, Shift로 미세 조정, 더블 클릭으로 0 dB 복원.",
                            "Clip gain: drag up/down, Shift for fine adjustment, double-click to reset to 0 dB." } },
    { "track.envelopeHint", { "区域响度包络：双击线添加点；拖动点调整时间和响度，Shift 精调；右键点可删除或复位。",
                              "區域響度包絡：雙擊線新增點；拖動點調整時間和響度，Shift 微調；右鍵點可刪除或復位。",
                              "リージョン音量：線をダブルクリックして点を追加。点をドラッグして調整、Shift で微調整。右クリックで削除・リセット。",
                              "영역 음량: 선을 더블 클릭하여 점 추가, 점을 드래그하여 조정, Shift로 미세 조정, 우클릭으로 삭제 또는 초기화.",
                              "Region gain envelope: double-click the line to add a point; drag points to adjust time and gain, Shift for fine adjustment; right-click to delete or reset." } },
    { "track.envelopeDelete", { "删除响度包络点", "刪除響度包絡點", "音量ポイントを削除", "음량 포인트 삭제", "Delete gain envelope point" } },
    { "track.envelopeResetPoint", { "包络点恢复 0 dB", "包絡點恢復 0 dB", "ポイントを 0 dB に戻す", "포인트를 0 dB로 초기화", "Reset point to 0 dB" } },
    { "track.envelopeReset", { "区域包络恢复 0 dB", "區域包絡恢復 0 dB", "リージョン音量を 0 dB に戻す", "영역 음량을 0 dB로 초기화", "Reset region envelope to 0 dB" } },
    { "track.audio",     { "普通音轨", "一般音軌", "通常トラック", "일반 트랙", "Audio Track" } },
    { "track.mute",      { "静音", "靜音", "ミュート", "음소거", "Mute" } },
    { "track.tip.compose", { "旋律轨道：开启后按音符音高渲染，并在钢琴卷帘中显示；关闭后作为普通音轨播放原始素材。",
                             "旋律軌道：開啟後依音符音高算繪，並在鋼琴捲簾中顯示；關閉後作為一般音軌播放原始素材。",
                             "メロディックトラック：オンのときだけノートが音高どおりにレンダリングされ、ピアノロールにも表示されます。オフなら通常トラックとして素材をそのまま再生します",
                             "멜로디 트랙: 켜면 노트가 음높이대로 렌더링되고 피아노 롤에도 표시됩니다. 끄면 일반 트랙으로 소재를 그대로 재생합니다",
                             "Compose track: only then are its notes rendered at their pitch and shown in the piano roll. Off, it is an audio track and the material plays as it is" } },
    { "track.tip.mute",  { "静音：这条轨道不发声，其余轨道照常",
                           "靜音：這條軌道不發聲，其餘軌道照常",
                           "ミュート：このトラックだけ音を出しません",
                           "음소거: 이 트랙만 소리가 나지 않습니다",
                           "Mute: this track is silent, the others play as usual" } },
    { "track.tip.solo",  { "独奏：只播放带独奏标记的轨道，其余全部静音",
                           "獨奏：只播放帶獨奏標記的軌道，其餘全部靜音",
                           "ソロ：ソロが付いたトラックだけを再生し、ほかはすべて無音になります",
                           "솔로: 솔로가 켜진 트랙만 재생하고 나머지는 모두 음소거됩니다",
                           "Solo: only soloed tracks play, everything else is silenced" } },
    { "track.tip.smooth", { "重叠淡化：相邻素材重叠处自动交叉淡化并把叠加音量归一，紧挨的接缝再补一段极短淡入，避免爆音",
                            "重疊淡化：相鄰素材重疊處自動交叉淡化並把疊加音量歸一，緊鄰的接縫再補一段極短淡入，避免爆音",
                            "重なりのフェード：素材が重なる部分を自動でクロスフェードし、合計音量をそろえます。隙間なく続く継ぎ目にはごく短いフェードを足してノイズを防ぎます",
                            "겹침 페이드: 소재가 겹치는 구간을 자동으로 크로스페이드하고 합쳐진 음량을 고르게 맞춥니다. 딱 붙은 이음매에는 아주 짧은 페이드를 넣어 잡음을 막습니다",
                            "Fade overlaps: overlapping material is crossfaded and its summed level evened out, and a very short fade is added at butt joins so they do not click" } },
    { "track.tip.normalize", { "音量归一：渲染后把音量匹配回原素材（NSF-HiFiGAN 按参考 Mel，其余后端按有声段 RMS）",
                               "音量歸一：算繪後把音量匹配回原素材（NSF-HiFiGAN 依參考 Mel，其餘後端依有聲段 RMS）",
                               "音量をそろえる：レンダリング後の音量を元の素材に合わせます（NSF-HiFiGANは参照メル、ほかのバックエンドは有声区間のRMS）",
                               "음량 정규화: 렌더링 후 음량을 원본 소재에 맞춥니다 (NSF-HiFiGAN은 참조 멜, 나머지 백엔드는 유성 구간 RMS)",
                               "Match level: after rendering, the level is matched back to the source (NSF-HiFiGAN against a reference mel, other backends by voiced RMS)" } },
    { "track.volume",    { "音量", "音量", "音量", "음량", "Vol" } },
    { "status.ready",    { "就绪", "就緒", "準備完了", "준비됨", "Ready" } },
    { "status.loading",  { "正在加载…", "正在載入…", "読み込み中…", "불러오는 중…", "Loading…" } },
    { "status.rendering", { "正在预渲染…", "正在預先算繪…", "プリレンダリング中…", "사전 렌더링 중…", "Pre-rendering…" } },
    { "status.analyzing", { "正在分析原始音高…", "正在分析原始音高…", "元ピッチを解析中…", "원본 피치 분석 중…", "Analysing source pitch…" } },
    { "status.analysisComplete", { "音高与音符分析完成", "音高與音符分析完成", "ピッチとノートの解析が完了しました", "피치 및 노트 분석 완료", "Pitch and note analysis complete" } },
    { "status.analysisSkipped", { "已保留现有音符数据", "已保留現有音符資料", "既存のノートデータを保持しました", "기존 노트 데이터를 유지했습니다", "Existing note data preserved" } },
    { "status.exporting", { "正在导出 WAV…", "正在匯出 WAV…", "WAVを書き出しています…", "WAV 내보내는 중…", "Exporting WAV…" } },
    { "status.clipCopied", { "已复制片段", "已複製片段", "クリップをコピーしました", "클립을 복사했습니다", "Clip copied" } },
    { "status.clipPasted", { "已粘贴片段", "已貼上片段", "クリップを貼り付けました", "클립을 붙여넣었습니다", "Clip pasted" } },
    { "status.notesCopied", { "已复制音符", "已複製音符", "ノートをコピーしました", "음표를 복사했습니다", "Notes copied" } },
    { "status.notesCut", { "已剪切音符", "已剪下音符", "ノートを切り取りました", "음표를 잘라냈습니다", "Notes cut" } },
    { "status.notesPasted", { "已粘贴音符", "已貼上音符", "ノートを貼り付けました", "음표를 붙여넣었습니다", "Notes pasted" } },
    { "status.projectOpened", { "已打开工程", "已開啟工程", "プロジェクトを開きました", "프로젝트를 열었습니다", "Project opened" } },
    { "status.projectSaved", { "已保存工程", "已儲存工程", "プロジェクトを保存しました", "프로젝트를 저장했습니다", "Project saved" } },
    { "status.midiPending", { "MIDI 导入器将在下一阶段接入", "MIDI 匯入器將於下一階段接入", "MIDIインポーターは次段階で接続します", "MIDI 가져오기는 다음 단계에서 연결됩니다", "MIDI importer will be connected in the next stage" } },
    { "status.noTracks", { "请导入音频或打开工程以开始编辑", "請匯入音訊或開啟工程以開始編輯", "音声またはプロジェクトを読み込んでください", "오디오 또는 프로젝트를 가져오세요", "Import audio or a project to begin" } },
    { "edit.source",     { "原始采样编辑：此模式不允许拉伸", "原始取樣編輯：此模式不允許拉伸", "元サンプル編集：このモードではストレッチできません", "원본 샘플 편집: 이 모드에서는 늘이기를 사용할 수 없습니다", "Original sample edit: stretching is disabled" } },
    { "error.audio",     { "音频文件读取失败", "音訊檔案讀取失敗", "オーディオを読み込めません", "오디오 파일을 읽지 못했습니다", "Could not read audio file" } },
    { "error.midi",      { "MIDI 导入失败", "MIDI 匯入失敗", "MIDIの読み込みに失敗しました", "MIDI 가져오기에 실패했습니다", "MIDI import failed" } },
    { "error.mpd",       { "Melodyne 工程读取失败", "Melodyne 工程讀取失敗", "Melodyneプロジェクトを読み込めません", "Melodyne 프로젝트를 읽지 못했습니다", "Could not read Melodyne project" } },
    { "error.export",    { "音频导出失败", "音訊匯出失敗", "オーディオの書き出しに失敗しました", "오디오 내보내기 실패", "Audio export failed" } },
    { "warning.missingMedia", { "以下素材未找到", "找不到以下素材", "次の素材が見つかりません", "다음 미디어를 찾지 못했습니다", "The following media files were not found" } },
    { "mpd.stage.open", { "打开工程", "開啟工程", "プロジェクトを開く", "프로젝트 열기", "Opening project" } },
    { "mpd.stage.scan_container", { "扫描工程容器", "掃描工程容器", "コンテナを走査", "프로젝트 컨테이너 검사", "Scanning container" } },
    { "mpd.stage.decompress_graph", { "解压工程数据", "解壓工程資料", "データを展開", "프로젝트 데이터 압축 해제", "Decompressing graph" } },
    { "mpd.stage.read_tracks", { "读取轨道和 BPM", "讀取軌道與 BPM", "トラックとBPMを読込", "트랙 및 BPM 읽기", "Reading tracks and BPM" } },
    { "mpd.stage.create_tracks", { "恢复音符和编辑", "還原音符與編輯", "ノートと編集を復元", "노트 및 편집 복원", "Restoring notes and edits" } },
    { "mpd.stage.reanalyse_pitch", { "重新分析原始 F0", "重新分析原始 F0", "元のF0を再解析", "원본 F0 재분석", "Reanalysing source F0" } },
    { "mpd.stage.complete", { "完成", "完成", "完了", "완료", "Complete" } },
    { "mpd.compose.title", { "选择旋律轨道", "選擇旋律軌道", "Composeトラックを選択", "Compose 트랙 선택", "Choose Compose Tracks" } },
    { "mpd.compose.description", { "勾选需要恢复 Melodyne 音符和修音的旋律轨道；其余轨道按普通音频播放。", "勾選需要還原 Melodyne 音符與修音的旋律軌道；其餘軌道作為一般音訊播放。", "Melodyneのノート編集を復元する旋律トラックを選択します。その他は通常の音声トラックとして扱います。", "Melodyne 노트 편집을 복원할 멜로디 트랙을 선택하세요. 나머지는 일반 오디오 트랙으로 처리됩니다.", "Select melodic tracks whose Melodyne note edits should be restored. Other tracks remain regular audio tracks." } },
    { "dialog.import", { "导入", "匯入", "読み込む", "가져오기", "Import" } },
    { "dialog.cancel", { "取消", "取消", "キャンセル", "취소", "Cancel" } }
    ,{ "dialog.delete", { "删除", "刪除", "削除", "삭제", "Delete" } }
    ,{ "dialog.destructiveMessage", { "此操作会删除所选内容，是否继续？", "此操作會刪除所選內容，是否繼續？", "選択した内容を削除します。続行しますか？", "선택한 내용을 삭제합니다. 계속할까요?", "The selected content will be deleted. Continue?" } }
    ,{ "dialog.apply", { "应用", "套用", "適用", "적용", "Apply" } }
    ,{ "dialog.save", { "保存", "儲存", "保存", "저장", "Save" } }
    ,{ "dialog.discard", { "放弃更改", "放棄變更", "変更を破棄", "변경 내용 버리기", "Discard Changes" } }
    ,{ "dialog.ustImportTitle", { "导入 UST", "匯入 UST", "USTの読み込み", "UST 가져오기", "Import UST" } }
    ,{ "dialog.ustImportMessage", { "当前工程已有内容。要用这个 UST 新建工程，还是把它添加为一条新音轨？", "目前工程已有內容。要用這個 UST 新建工程，還是把它加為一條新軌道？", "プロジェクトには既に内容があります。このUSTで新しく開き直しますか、それとも新しいトラックとして追加しますか？", "프로젝트에 이미 내용이 있습니다. 이 UST로 새로 열까요, 아니면 새 트랙으로 추가할까요?", "This project already has content. Open this UST as the project, or add it as a new track?" } }
    ,{ "dialog.ustReplace", { "作为新工程打开", "作為新工程開啟", "新しいプロジェクトとして開く", "새 프로젝트로 열기", "Open as the Project" } }
    ,{ "dialog.ustAddTrack", { "添加为新音轨", "加為新軌道", "新しいトラックとして追加", "새 트랙으로 추가", "Add as a New Track" } }
    ,{ "dialog.midiTrackTitle", { "导入 MIDI 轨道", "匯入 MIDI 軌道", "MIDIトラックの読み込み", "MIDI 트랙 가져오기", "Import MIDI Track" } }
    ,{ "dialog.midiTrackMessage", { "这个 MIDI 文件有多个带音符的轨道，选择要导入哪一个。", "這個 MIDI 檔案有多個帶音符的軌道，選擇要匯入哪一個。", "このMIDIファイルには音符のあるトラックが複数あります。読み込むトラックを選んでください。", "이 MIDI 파일에는 음표가 있는 트랙이 여러 개 있습니다. 가져올 트랙을 선택하세요.", "This MIDI file has more than one track with notes. Choose the one to import." } }
    ,{ "dialog.midiTrackLabel", { "轨道", "軌道", "トラック", "트랙", "Track" } }
    ,{ "dialog.midiTrackNotes", { "个音符", "個音符", "音符", "개 음표", "notes" } }
    ,{ "dialog.unsavedTitle", { "工程尚未保存", "工程尚未儲存", "プロジェクトは未保存です", "프로젝트가 저장되지 않음", "Unsaved Project" } }
    ,{ "dialog.unsavedMessage", { "是否先保存当前工程的更改？", "是否先儲存目前工程的變更？", "現在のプロジェクトの変更を保存しますか？", "현재 프로젝트 변경 내용을 저장할까요?", "Save changes to the current project first?" } }
    ,{ "settings.title", { "设置", "設定", "設定", "설정", "Settings" } }
    ,{ "settings.interface", { "界面", "介面", "インターフェース", "인터페이스", "Interface" } }
    ,{ "settings.audio", { "音频", "音訊", "オーディオ", "오디오", "Audio" } }
    ,{ "settings.audioDevice", { "当前音频设备", "目前音訊裝置", "現在のオーディオデバイス", "현재 오디오 장치", "Current Audio Device" } }
    ,{ "settings.sampleRate", { "采样率", "取樣率", "サンプルレート", "샘플 레이트", "Sample Rate" } }
    ,{ "settings.bufferSize", { "缓冲区大小", "緩衝區大小", "バッファサイズ", "버퍼 크기", "Buffer Size" } }
    ,{ "settings.advancedAudio", { "选择输入、输出和驱动…", "選擇輸入、輸出與驅動…", "入出力とドライバーを選択…", "입출력 및 드라이버 선택…", "Choose Inputs, Outputs and Driver…" } }
    ,{ "settings.noAudioDevice", { "未选择音频设备", "尚未選擇音訊裝置", "オーディオデバイス未選択", "오디오 장치가 선택되지 않음", "No audio device selected" } }
    ,{ "settings.algorithm", { "算法", "演算法", "アルゴリズム", "알고리즘", "Algorithms" } }
    ,{ "settings.operation", { "操作", "操作", "操作", "조작", "Operations" } }
    ,{ "settings.import", { "文件导入", "檔案匯入", "ファイル読込", "파일 가져오기", "File Import" } }
    ,{ "settings.language", { "语言", "語言", "言語", "언어", "Language" } }
    ,{ "settings.theme", { "颜色主题", "色彩主題", "カラーテーマ", "색상 테마", "Colour Theme" } }
    ,{ "settings.themeDark", { "深色", "深色", "ダーク", "다크", "Dark" } }
    ,{ "settings.themeLight", { "浅色", "淺色", "ライト", "라이트", "Light" } }
    ,{ "settings.accent", { "主色（Hex）", "主色（Hex）", "アクセント（Hex）", "강조색 (Hex)", "Accent (Hex)" } }
    ,{ "settings.accentLight", { "浅主色（Hex）", "淺主色（Hex）", "明るい主色（Hex）", "밝은 강조색 (Hex)", "Light Accent (Hex)" } }
    ,{ "settings.noteColour", { "音符色（Hex）", "音符色（Hex）", "ノート色（Hex）", "노트 색상 (Hex)", "Note Colour (Hex)" } }
    ,{ "settings.showNoteLabels", { "显示已标注的发音/别名", "顯示已標註的發音/別名", "注釈済みの発音・別名を表示", "표시된 발음/별칭 표시", "Show annotated pronunciation/alias" } }
    ,{ "settings.uiScale", { "界面缩放", "介面縮放", "UIスケール", "UI 배율", "UI Scale" } }
    ,{ "settings.gamePath", { "GAME 模型目录", "GAME 模型目錄", "GAMEモデルフォルダー", "GAME 모델 폴더", "GAME Model Directory" } }
    ,{ "settings.gameModel", { "GAME 默认模型", "GAME 預設模型", "GAME既定モデル", "GAME 기본 모델", "Default GAME Model" } }
    ,{ "settings.fcpePath", { "FCPE 模型或目录", "FCPE 模型或目錄", "FCPEモデルまたはフォルダー", "FCPE 모델 또는 폴더", "FCPE Model or Directory" } }
    ,{ "settings.hifiganPath", { "HiFi-GAN 模型目录", "HiFi-GAN 模型目錄", "HiFi-GANモデルフォルダー", "HiFi-GAN 모델 폴더", "HiFi-GAN Model Directory" } }
    ,{ "settings.inference", { "推理方式", "推理方式", "推論バックエンド", "추론 백엔드", "Inference Backend" } }
    ,{ "settings.device", { "推理设备", "推理裝置", "推論デバイス", "추론 장치", "Inference Device" } }
    ,{ "settings.auto", { "自动", "自動", "自動", "자동", "Auto" } }
    ,{ "settings.utauResampler", { "UTAU 重采样器（可选，留空则内置变调）", "UTAU 重取樣器（可選，留空則內建變調）", "UTAUリサンプラー（任意、空欄は内蔵処理）", "UTAU 리샘플러 (선택, 비우면 내장 처리)", "UTAU Resampler (optional; built-in fallback)" } }
    ,{ "settings.shortcuts", { "快捷键方案", "快速鍵配置", "ショートカット方式", "단축키 방식", "Shortcut Scheme" } }
    ,{ "settings.wheel", { "鼠标滚轮", "滑鼠滾輪", "マウスホイール", "마우스 휠", "Mouse Wheel" } }
    ,{ "settings.wheelZoom", { "缩放", "縮放", "ズーム", "확대/축소", "Zoom" } }
    ,{ "settings.wheelScroll", { "滚动", "捲動", "スクロール", "스크롤", "Scroll" } }
    ,{ "settings.spacePlayback", { "空格键播放/暂停", "空白鍵播放/暫停", "スペースで再生/一時停止", "스페이스바 재생/일시정지", "Space toggles playback" } }
    ,{ "settings.confirmDestructive", { "删除前确认", "刪除前確認", "削除前に確認", "삭제 전 확인", "Confirm before delete" } }
    ,{ "settings.melodyneCompose", { "Melodyne 旋律轨道导入方式", "Melodyne 旋律軌道匯入方式", "Melodyne Composeの既定値", "Melodyne Compose 기본값", "Default Melodyne Compose" } }
    ,{ "settings.composeAsk", { "每次询问", "每次詢問", "毎回確認", "매번 확인", "Ask Every Time" } }
    ,{ "settings.composeMelodic", { "旋律轨道", "旋律軌道", "メロディックトラック", "멜로디 트랙", "Melodic Tracks" } }
    ,{ "settings.composeAll", { "全部轨道", "全部軌道", "すべてのトラック", "모든 트랙", "All Tracks" } }
    ,{ "settings.composeAudio", { "仅普通音轨", "僅一般音軌", "通常音声のみ", "일반 오디오만", "Audio Only" } }
    ,{ "settings.melodynePitch", { "Melodyne 原音高来源", "Melodyne 原音高來源", "Melodyne元ピッチの取得元", "Melodyne 원본 피치 소스", "Melodyne Source Pitch" } }
    ,{ "settings.pitchProject", { "工程记录", "工程記錄", "プロジェクトデータ", "프로젝트 데이터", "Project Data" } }
    ,{ "settings.pitchReanalyse", { "GAME + FCPE（无模型时使用原生分析）", "GAME + FCPE（無模型時使用原生分析）", "GAME + FCPE（モデルなしはネイティブ解析）", "GAME + FCPE (모델 미포함 시 네이티브 분석)", "GAME + FCPE (native fallback without models)" } }
    ,{ "settings.importAlgorithm", { "导入工程默认变调算法", "匯入工程預設變調演算法", "読込時の既定ピッチアルゴリズム", "가져오기 기본 피치 알고리즘", "Default Import Pitch Algorithm" } }
    ,{ "settings.importStretchAlgorithm", { "导入工程默认拉伸算法", "匯入工程預設拉伸演算法", "読込時の既定タイムストレッチ", "가져오기 기본 타임 스트레치", "Default Import Stretch Algorithm" } }
    ,{ "settings.preserveEdits", { "保留工程中的修音、Attack、音量和音色编辑", "保留工程中的修音、Attack、音量與音色編輯", "ピッチ補正・Attack・音量・音色の編集を保持", "피치 보정·Attack·음량·음색 편집 유지", "Preserve tuning, Attack, level and timbre edits" } }
    ,{ "settings.recursiveMedia", { "递归查找缺失素材", "遞迴尋找遺失素材", "不足素材を再帰検索", "누락 미디어 재귀 검색", "Search recursively for missing media" } }
    ,{ "sample.alias", { "别名", "別名", "エイリアス", "별칭", "Alias" } }
    ,{ "sample.start", { "起点", "起點", "開始", "시작", "Start" } }
    ,{ "sample.end", { "终点", "終點", "終了", "끝", "End" } }
    ,{ "sample.alignment", { "对齐", "對齊", "整列", "정렬", "Align" } }
    ,{ "sample.fixed", { "辅音", "子音", "子音", "자음", "Fixed" } }
    ,{ "sample.save", { "保存设定", "儲存設定", "設定を保存", "설정 저장", "Save Settings" } }
    ,{ "sample.saved", { "音频设定已保存", "音訊設定已儲存", "音声設定を保存しました", "오디오 설정 저장됨", "Audio settings saved" } }
    ,{ "sample.importOto", { "读取 oto", "讀取 oto", "oto読込", "oto 읽기", "Import oto" } }
    ,{ "sample.exportOto", { "导出 oto", "匯出 oto", "oto書出", "oto 내보내기", "Export oto" } }
    ,{ "asset.title", { "素材管理器", "素材管理器", "素材マネージャー", "소재 관리자", "Asset Manager" } }
    ,{ "asset.register", { "注册音频素材", "註冊音訊素材", "音声素材を登録", "오디오 소재 등록", "Register Audio Assets" } }
    ,{ "asset.utau", { "导入 UTAU 音源库", "匯入 UTAU 音源庫", "UTAU 音源を読み込む", "UTAU 음원 가져오기", "Import UTAU Voicebank" } }
    ,{ "asset.utauDone", { "已导入音源文件夹，共 {files} 个音频。OTO 参数将在编辑或合成时读取。", "已匯入音源資料夾，共 {files} 個音訊。OTO 參數將在編輯或合成時讀取。", "音源フォルダーを登録しました（音声 {files} 件）。OTO は編集・合成時に読み込みます。", "음원 폴더를 등록했습니다(오디오 {files}개). OTO는 편집 또는 합성할 때 읽습니다.", "Registered voicebank folder with {files} audio files. OTO parameters are read when editing or synthesizing." } }
    ,{ "asset.remove", { "移除", "移除", "削除", "제거", "Remove" } }
    ,{ "asset.projectVoicebank", { "工程音源", "專案音源", "使用中の音源", "프로젝트 음원", "Project bank" } }
    ,{ "asset.folderMissing", { "找不到音源文件夹，请在轨道中重新选择音源。", "找不到音源資料夾，請在軌道中重新選擇音源。", "音源フォルダーが見つかりません。トラックで音源を選び直してください。", "음원 폴더를 찾을 수 없습니다. 트랙에서 음원을 다시 선택하세요.", "Voicebank folder not found. Select its location again on the track." } }
    ,{ "asset.empty", { "把素材文件夹或音频拖入此处；音源库和音频文件夹都会成为可复用的素材文件夹。", "將素材資料夾或音訊拖入此處；音源庫和音訊資料夾都會成為可重複使用的素材資料夾。", "素材フォルダーまたは音声をここへドロップします。音源も音声フォルダーも再利用できる素材フォルダーになります。", "소재 폴더나 오디오를 여기에 놓으세요. 음원과 오디오 폴더 모두 재사용 가능한 소재 폴더가 됩니다.", "Drop a material folder or audio here; voicebanks and audio folders both become reusable material folders." } }
    ,{ "asset.newFolder", { "新建素材夹", "新建素材夾", "素材フォルダー作成", "소재 폴더 생성", "New Folder" } }
    ,{ "asset.importFolder", { "导入文件夹", "匯入資料夾", "フォルダー読込", "폴더 가져오기", "Import Folder" } }
    ,{ "asset.addAudio", { "添加音频", "新增音訊", "音声を追加", "오디오 추가", "Add Audio" } }
    ,{ "asset.assemble", { "活字印刷", "活字印刷", "活字印刷", "활자 인쇄", "Movable Type" } }
    ,{ "asset.folderEmpty", { "这个素材文件夹还没有音频。用「添加音频」或拖入音频，系统会自动粗略识别参数。", "這個素材資料夾還沒有音訊。用「新增音訊」或拖入音訊，系統會自動粗略辨識參數。", "この素材フォルダーにはまだ音声がありません。「音声を追加」またはドロップすると自動で概略パラメーターを検出します。", "이 소재 폴더에는 아직 오디오가 없습니다. '오디오 추가' 또는 드롭하면 자동으로 파라미터를 대략 감지합니다.", "This folder has no audio yet. Add or drop audio and parameters are roughly detected automatically." } }
    ,{ "asset.paramsReady", { "参数已识别，可编辑并复用", "參數已辨識，可編輯並重複使用", "パラメーター検出済み・編集/再利用可", "파라미터 감지됨 · 편집/재사용 가능", "Parameters detected — editable and reusable" } }
    ,{ "asset.paramsNone", { "尚无参数", "尚無參數", "パラメーター未検出", "파라미터 없음", "No parameters yet" } }
    ,{ "asset.pickFolderFirst", { "请先在左侧选择一个素材文件夹。", "請先在左側選擇一個素材資料夾。", "先に左側で素材フォルダーを選択してください。", "먼저 왼쪽에서 소재 폴더를 선택하세요.", "Select a material folder on the left first." } }
    ,{ "asset.assembleHint", { "输入歌词，按行/字转拼音并从所选素材库匹配，生成有序素材序列。", "輸入歌詞，按行/字轉拼音並從所選素材庫比對，產生有序素材序列。", "歌詞を入力すると各文字を拼音に変換し、選択した素材から順序付き素材列を生成します。", "가사를 입력하면 각 글자를 병음으로 변환해 선택한 소재에서 순서 있는 소재열을 생성합니다.", "Enter lyrics; each character is converted to pinyin and matched from the chosen material into an ordered sequence." } }
    ,{ "asset.assembleName", { "新素材夹名称", "新素材夾名稱", "新しい素材フォルダー名", "새 소재 폴더 이름", "New folder name" } }
    ,{ "asset.assembleLyrics", { "歌词（每字一个素材）", "歌詞（每字一個素材）", "歌詞（1文字ごとに1素材）", "가사 (글자마다 소재 하나)", "Lyrics (one material per character)" } }
    ,{ "asset.assembleDone", { "已生成 {matched} 个素材，缺失 {missing} 个。", "已產生 {matched} 個素材，缺失 {missing} 個。", "{matched} 件を生成し、{missing} 件が不足しました。", "{matched}개를 생성했고 {missing}개가 누락되었습니다.", "Assembled {matched} materials; {missing} missing." } }
    ,{ "asset.docked", { "素材管理器已在下方打开；再次点击菜单项可收起。", "素材管理器已在下方開啟；再次點選選單項可收起。", "素材マネージャーを下部に表示しました。メニュー項目を再度選ぶと閉じます。", "소재 관리자를 아래에 열었습니다. 메뉴 항목을 다시 선택하면 닫힙니다.", "Material manager docked below; choose the menu item again to hide it." } }
    ,{ "asset.otoNoneShort", { "不读取 OTO", "不讀取 OTO", "OTO なし", "OTO 없음", "No OTO" } }
    ,{ "asset.search", { "搜索声音文件…", "搜尋聲音檔案…", "音声ファイルを検索…", "오디오 파일 검색…", "Search audio files…" } }
    ,{ "asset.clearSearch", { "清空搜索", "清空搜尋", "検索をクリア", "검색 지우기", "Clear search" } }
    ,{ "asset.searching", { "正在搜索声音文件…", "正在搜尋聲音檔案…", "音声ファイルを検索中…", "오디오 파일 검색 중…", "Searching audio files…" } }
    ,{ "asset.noSearchResults", { "没有匹配的声音文件", "沒有符合的聲音檔案", "一致する音声ファイルがありません", "일치하는 오디오 파일 없음", "No matching audio files" } }
    ,{ "asset.otoClassicShort", { "普通 OTO", "普通 OTO", "標準 OTO", "표준 OTO", "Classic OTO" } }
    ,{ "asset.otoJieShort", { "界 OTO", "界 OTO", "Jie OTO", "Jie OTO", "Jie OTO" } }
    ,{ "asset.otoMouShort", { "谋 OTO", "謀 OTO", "Mou OTO", "Mou OTO", "Mou OTO" } }
    ,{ "asset.otoNone", { "OTO 类型：不读取（完整录音）", "OTO 類型：不讀取（完整錄音）", "OTO: 使用しない（全音声）", "OTO: 사용 안 함 (전체 녹음)", "OTO: none (full recording)" } }
    ,{ "asset.otoClassic", { "OTO 类型：普通 OTO（二区）", "OTO 類型：普通 OTO（兩區）", "OTO: 標準（2 区間）", "OTO: 표준 (2구간)", "OTO: classic (2 regions)" } }
    ,{ "asset.otoJie", { "OTO 类型：界 OTO（四区）", "OTO 類型：界 OTO（四區）", "OTO: Jie（4 区間）", "OTO: Jie (4구간)", "OTO: Jie (4 regions)" } }
    ,{ "asset.otoMou", { "OTO 类型：谋 OTO（二至四区）", "OTO 類型：謀 OTO（二至四區）", "OTO: Mou（2～4 区間）", "OTO: Mou (2~4구간)", "OTO: Mou (2–4 regions)" } }
    ,{ "asset.otoMissing", { "未找到所选类型的有效 OTO 条目或分区。请选择匹配的 OTO 类型；要导入完整录音，请选择“不读取”。", "未找到所選類型的有效 OTO 條目或分區。請選擇匹配的 OTO 類型；要匯入完整錄音，請選擇「不讀取」。", "選択した形式の有効な OTO がありません。形式を変更するか、全音声には「使用しない」を選択してください。", "선택한 형식의 유효한 OTO가 없습니다. 형식을 바꾸거나 전체 녹음은 사용 안 함을 선택하세요.", "No valid entry or regions for the selected OTO type. Choose the matching type, or none to import the full recording." } }
    ,{ "asset.otoAlias", { "选择要导入的 OTO 别名", "選擇要匯入的 OTO 別名", "読み込む OTO エイリアス", "불러올 OTO 별칭 선택", "Choose an OTO alias to import" } }
    ,{ "asset.loadAudio", { "加载为音频素材", "載入為音訊素材", "音声素材として読み込む", "오디오 소재로 불러오기", "Load as audio material" } }
    ,{ "asset.browseHint", { "展开文件夹，右键录音选择“加载为音频素材”，或选中后按 Enter。", "展開資料夾，右鍵錄音選擇「載入為音訊素材」，或選取後按 Enter。", "フォルダーを展開し、音声を右クリックして読み込むか、選択後 Enter を押します。", "폴더를 펼치고 녹음을 우클릭하여 불러오거나 선택 후 Enter를 누르세요.", "Expand a folder, then right-click a recording to load it as audio, or select it and press Enter." } }
    ,{ "asset.editTitle", { "素材参数编辑器", "素材參數編輯器", "素材パラメーター編集", "소재 파라미터 편집", "Material Parameter Editor" } }
    ,{ "asset.editNoOto", { "该素材在 oto.ini 中没有对应条目，无法编辑。", "該素材在 oto.ini 中沒有對應條目，無法編輯。", "この素材は oto.ini に対応する行がないため編集できません。", "이 소재는 oto.ini에 해당 항목이 없어 편집할 수 없습니다.", "This material has no oto.ini entry to edit." } }
    ,{ "asset.exportOto", { "另存为 OTO", "另存為 OTO", "OTO として書き出し", "OTO로 내보내기", "Export as OTO" } }
    ,{ "asset.exportOtoDone", { "已导出 {rows} 行到 oto.ini（原生标注仍保留）。", "已匯出 {rows} 行到 oto.ini（原生標註仍保留）。", "{rows} 行を oto.ini に書き出しました（ネイティブ注釈は保持）。", "{rows}행을 oto.ini로 내보냈습니다(네이티브 주석 유지).", "Exported {rows} rows to oto.ini (native annotations kept)." } }
    ,{ "asset.melodyneFolderTitle", { "创建素材文件夹", "建立素材資料夾", "素材フォルダーを作成", "소재 폴더 생성", "Create Material Folder" } }
    ,{ "asset.melodyneFolderPrompt", { "此 Melodyne 工程引用了 {count} 个源音频文件夹。是否登记为可复用的素材文件夹？", "此 Melodyne 工程引用了 {count} 個來源音訊資料夾。是否登記為可重複使用的素材資料夾？", "この Melodyne プロジェクトは {count} 個の音源フォルダーを参照しています。再利用可能な素材フォルダーとして登録しますか？", "이 Melodyne 프로젝트는 소스 오디오 폴더 {count}개를 참조합니다. 재사용 가능한 소재 폴더로 등록할까요?", "This Melodyne project references {count} source audio folder(s). Register them as reusable material folders?" } }
    ,{ "asset.melodyneFolderCreate", { "登记素材夹", "登記素材夾", "登録する", "등록", "Register" } }
    ,{ "asset.melodyneRegisterInPlace", { "登记（默认）", "登記（預設）", "登録（既定）", "등록(기본)", "Register (default)" } }
    ,{ "asset.melodyneRegisterCopy", { "登记并复制到文件夹…", "登記並複製到資料夾…", "登録してフォルダーへコピー…", "등록 후 폴더로 복사…", "Register and copy to folder…" } }
    ,{ "asset.melodyneNoRegister", { "不登记", "不登記", "登録しない", "등록 안 함", "Don't register" } }
    ,{ "asset.melodyneFolderDone", { "已登记 {count} 个素材文件夹，可在素材管理器中复用。", "已登記 {count} 個素材資料夾，可在素材管理器中重複使用。", "{count} 個の素材フォルダーを登録しました。素材マネージャーで再利用できます。", "소재 폴더 {count}개를 등록했습니다. 소재 관리자에서 재사용할 수 있습니다.", "Registered {count} material folder(s), reusable in the Asset Manager." } }
    ,{ "settings.browse", { "浏览…", "瀏覽…", "参照…", "찾아보기…", "Browse…" } }
    ,{ "settings.utauOutputEngine", { "UTAU 默认输出引擎", "UTAU 預設輸出引擎", "UTAU 既定出力エンジン", "UTAU 기본 출력 엔진", "Default UTAU Output Engine" } }
    ,{ "phonemizer.choose", { "发音器", "發音器", "音素変換器", "발음 변환기", "Phonemizer" } }
    ,{ "phonemizer.manualShort", { "手动别名", "手動別名", "手動", "수동", "Manual" } }
    ,{ "utau.pianoPreview", { "钢琴试听", "鋼琴試聽", "ピアノ試聴", "피아노 미리 듣기", "Piano preview" } }
    ,{ "utau.pianoPreviewHelp", { "当前轨道空歌词或找不到样本时播放钢琴；导出时空歌词仍静音", "目前軌道空歌詞或找不到樣本時播放鋼琴；匯出時空歌詞仍靜音", "歌詞が空またはサンプルがない音符をピアノで試聴。書き出し時の空歌詞は無音", "현재 트랙의 빈 가사나 없는 샘플을 피아노로 미리 듣습니다. 빈 가사는 내보낼 때 무음입니다", "Preview blank lyrics or missing samples as piano on this track. Blank lyrics remain silent on export." } }
    ,{ "phonemizer.toolbarHelp", { "选择当前轨道的发音器；中文 CVVC 按音源 presamp.ini 自动拼接发音", "選擇目前軌道的發音器；中文 CVVC 依音源 presamp.ini 自動拼接發音", "トラックの音素変換器を選択。中国語 CVVC は音源の presamp.ini を使用", "트랙 발음 변환기 선택. 중국어 CVVC는 음원 presamp.ini 사용", "Choose this track's phonemizer. Chinese CVVC uses the voicebank's presamp.ini." } }
    ,{ "phonemizer.manual", { "手动别名（默认）", "手動別名（預設）", "手動エイリアス（既定）", "수동 별칭 (기본)", "Manual aliases (default)" } }
    ,{ "phonemizer.zhCvvc", { "中文 CVVC", "中文 CVVC", "中国語 CVVC", "중국어 CVVC", "Chinese CVVC" } }
    ,{ "phonemizer.missingPresamp", { "中文 CVVC 需要音源目录中的 presamp.ini（包含 VOWEL 与 CONSONANT 配置）。请先选择对应音源并补齐配置。", "中文 CVVC 需要音源目錄中的 presamp.ini（包含 VOWEL 與 CONSONANT 配置）。", "中国語 CVVC には音源フォルダーの presamp.ini（VOWEL と CONSONANT）が必要です。", "중국어 CVVC에는 음원 폴더의 presamp.ini (VOWEL, CONSONANT)가 필요합니다.", "Chinese CVVC requires presamp.ini with VOWEL and CONSONANT sections in the voicebank folder." } }
    ,{ "output.choose", { "选择输出引擎", "選擇輸出引擎", "出力エンジンを選択", "출력 엔진 선택", "Choose Output Engine" } }
    ,{ "output.inherit", { "跟随设置", "跟隨設定", "設定に従う", "설정 따르기", "Follow Settings" } }
    ,{ "output.resampler", { "重采样器…", "重取樣器…", "リサンプラー…", "리샘플러…", "Resampler…" } }
    ,{ "output.toolsHelp", { "路径留空时使用设置中的默认值；默认 wavtool 留空时使用内置合成。", "路徑留空時使用設定中的預設值；預設 wavtool 留空時使用內建合成。", "空欄は既定設定を使用。既定 wavtool が空欄の場合は内蔵合成。", "빈 경로는 기본 설정 사용. 기본 wavtool이 비어 있으면 내장 합성 사용.", "Empty paths follow Settings; an empty default wavtool uses internal assembly." } }
    ,{ "output.invalidTool", { "指定的工具文件不存在，请重新选择。", "指定的工具檔案不存在，請重新選擇。", "指定したツールが見つかりません。", "선택한 도구 파일이 없습니다.", "The selected tool file does not exist." } }
    ,{ "settings.utauVoicebank", { "音源选择初始文件夹", "音源選擇起始資料夾", "音源選択の初期フォルダー", "음원 선택 시작 폴더", "Voicebank browse folder" } }
    ,{ "settings.utauVoicebankHint", { "仅用于音源选择窗口的初始位置，不会自动绑定音轨。", "僅用於音源選擇視窗的起始位置，不會自動綁定音軌。", "音源選択画面の初期位置です。トラックへの自動設定は行いません。", "음원 선택 창의 시작 위치입니다. 트랙에 자동 연결하지 않습니다.", "Starting location for the voicebank chooser only. Does not automatically bind any track." } }
    ,{ "settings.utauWavtool", { "UTAU 合成器 / wavtool（可选）", "UTAU 合成器 / wavtool（可選）", "UTAU 合成ツール / wavtool（任意）", "UTAU 합성기 / wavtool (선택)", "UTAU Synthesis Tool / wavtool (optional)" } }
    ,{ "settings.chooseUtauVoicebank", { "选择 UTAU 音源文件夹", "選擇 UTAU 音源資料夾", "UTAU 音源フォルダーを選択", "UTAU 음원 폴더 선택", "Choose UTAU Voicebank Folder" } }
    ,{ "settings.chooseUtauWavtool", { "选择 UTAU 合成器 / wavtool", "選擇 UTAU 合成器 / wavtool", "UTAU 合成ツール / wavtool を選択", "UTAU 합성기 / wavtool 선택", "Choose UTAU Synthesis Tool / wavtool" } }
    ,{ "settings.chooseUtauResampler", { "选择 UTAU 重采样器", "選擇 UTAU 重取樣器", "UTAU リサンプラーを選択", "UTAU 리샘플러 선택", "Choose UTAU Resampler" } }
    ,{ "Editing mode", { "编辑模式", "編輯模式", "編集モード", "편집 모드", "Editing mode" } }
    ,{ "Single syllable / phoneme", { "单音节音素", "單音節音素", "単音節・音素", "단음절 / 음소", "Single syllable / phoneme" } }
    ,{ "Use detail directly for this recording; uncheck to return to overview. Existing pronunciations are preserved. Unannotated audio gets one unsaved pronunciation draft.", { "此录音直接使用详细参数，取消勾选返回发音总览。已有发音标注保留；未标注录音建立一个未保存的发音草稿。", "此錄音直接使用詳細參數，取消勾選返回發音總覽。保留已有發音標註；未標註錄音建立一個未儲存的發音草稿。", "この録音は詳細画面で編集します。チェックを外すと一覧に戻ります。既存の発音は保持し、未注釈の録音には未保存の発音を作成します。", "이 녹음은 상세 화면에서 편집합니다. 선택을 해제하면 개요로 돌아갑니다. 기존 발음은 유지되며 주석이 없는 녹음에는 저장되지 않은 발음 초안이 생성됩니다.", "Use detail directly for this recording; uncheck to return to overview. Existing pronunciations are preserved. Unannotated audio gets one unsaved pronunciation draft." } }
    ,{ "Native annotations (default)", { "原生标注（默认）", "原生標註（預設）", "ネイティブ注釈（既定）", "네이티브 주석 (기본)", "Native annotations (default)" } }
    ,{ "Modes change the view only: source scope saves otto.ini; this-note scope applies only to the project. Old OTO stays unchanged.", { "模式仅改变参数视图；源标注保存到 otto.ini，本音符范围仅应用到工程。旧 OTO 不变。", "模式僅改變檢視；來源範圍儲存 otto.ini，單音符範圍僅套用至工程，舊 OTO 不變。", "モードは表示のみ変更します。ソース範囲は otto.ini に保存し、このノートの範囲はプロジェクトのみに適用します。旧 OTO は変更しません。", "모드는 보기만 변경합니다. 원본 범위는 otto.ini에 저장하고 이 음표 범위는 프로젝트에만 적용합니다. 기존 OTO는 변경하지 않습니다.", "Modes change the view only: source scope saves otto.ini; this-note scope applies only to the project. Old OTO stays unchanged." } }
    ,{ "Back to pronunciation overview", { "返回发音总览", "返回發音總覽", "発音一覧へ戻る", "발음 개요로 돌아가기", "Back to pronunciation overview" } }
    ,{ "Source annotations (otto.ini)", { "源标注（otto.ini）", "來源標註（otto.ini）", "ソース注釈（otto.ini）", "원본 주석 (otto.ini)", "Source annotations (otto.ini)" } }
    ,{ "This note only (project)", { "仅此音符（工程）", "僅此音符（工程）", "このノートのみ（プロジェクト）", "이 음표만 (프로젝트)", "This note only (project)" } }
    ,{ "Restore source settings", { "恢复源设置", "還原來源設定", "ソース設定を復元", "원본 설정 복원", "Restore source settings" } }
    ,{ "Normal syllable", { "普通音节", "普通音節", "通常の音節", "일반 음절", "Normal syllable" } }
    ,{ "Transition pronunciation", { "过渡发音", "過渡發音", "遷移発音", "전이 발음", "Transition pronunciation" } }
    ,{ "Breath", { "呼吸", "呼吸", "ブレス", "호흡", "Breath" } }
    ,{ "Pronunciation identity, not an internal C/V role.", { "整个发音的类别，不是内部 C/V 区段。", "整個發音的類別，不是內部 C/V 區段。", "発音全体の種類です。内部 C/V 区間ではありません。", "전체 발음 유형이며 내부 C/V 구간이 아닙니다.", "Pronunciation identity, not an internal C/V role." } }
    ,{ "Pronunciation identity", { "发音身份", "發音身分", "発音の種類", "발음 유형", "Pronunciation identity" } }
    ,{ "Internal vowel segment", { "内部元音区段", "內部母音區段", "内部母音区間", "내부 모음 구간", "Internal vowel segment" } }
    ,{ "Split named vowel", { "命名并切分元音", "命名並切分母音", "母音に名前を付けて分割", "이름을 지정하여 모음 분할", "Split named vowel" } }
    ,{ "Merge next segment", { "合并下一区段", "合併下一區段", "次の区間と結合", "다음 구간 병합", "Merge next segment" } }
    ,{ "Simplified consonant", { "简化辅音", "簡化子音", "簡易子音", "간소화 자음", "Simplified consonant" } }
    ,{ "When enabled, C follows P by its derived offset. Off by default.", { "开启后 C 随 P 保持相对位置；默认关闭。", "開啟後 C 隨 P 保持相對位置；預設關閉。", "有効時は C が P との相対位置を保ちます。既定は無効です。", "활성화하면 C가 P와의 상대 위치를 유지합니다. 기본값은 꺼짐입니다.", "When enabled, C follows P by its derived offset. Off by default." } }
    ,{ "Name", { "名称", "名稱", "名前", "이름", "Name" } }
    ,{ "Stretch weight", { "拉伸权重", "伸縮權重", "伸縮の重み", "늘이기 가중치", "Stretch weight" } }
    ,{ "Positive relative weight distributes elastic vowel time; it never changes the fixed C/P zone.", { "正权重控制各手动元音分配的拉伸时长；不改变 C/P 固定区。", "正權重分配可伸縮母音時長，不改變固定 C/P 區域。", "正の相対重みで母音の伸縮時間を配分します。固定 C/P 区間は変更しません。", "양수 상대 가중치로 모음 늘이기 시간을 배분하며 고정 C/P 구간은 변경하지 않습니다.", "Positive relative weight distributes elastic vowel time; it never changes the fixed C/P zone." } }
    ,{ "Left name", { "左元音名", "左母音名", "左の母音名", "왼쪽 모음 이름", "Left name" } }
    ,{ "Right name", { "右元音名", "右母音名", "右の母音名", "오른쪽 모음 이름", "Right name" } }
    ,{ "Confirm split", { "确认命名切分", "確認分割", "分割を確定", "분할 확인", "Confirm split" } }
    ,{ "Cancel", { "取消", "取消", "キャンセル", "취소", "Cancel" } }
    ,{ "Classic scalar OTO timing for this note only; kind/named vowels are not stored. Restore inherits engine OTO defaults.", { "仅此音符的经典 OTO 标量时序；不保存音节类型或命名元音，恢复表示继承引擎默认 OTO。", "僅此音符的經典 OTO 純量時序；不儲存類型／命名母音，還原後繼承引擎預設 OTO。", "このノートのみの従来 OTO 時刻です。種類や名前付き母音は保存しません。復元するとエンジン既定 OTO を継承します。", "이 음표의 클래식 OTO 시간만 적용합니다. 유형이나 명명 모음은 저장하지 않으며 복원 시 엔진 기본 OTO를 상속합니다.", "Classic scalar OTO timing for this note only; kind/named vowels are not stored. Restore inherits engine OTO defaults." } }
    ,{ "Applies to this note only; recording, OTTO and old OTO unchanged.", { "仅应用到此音符；不修改源录音、OTTO 或旧 OTO。", "僅套用此音符；錄音、OTTO 與舊 OTO 不變。", "このノートのみに適用し、録音・OTTO・旧 OTO は変更しません。", "이 음표에만 적용하며 녹음, OTTO 및 기존 OTO는 변경하지 않습니다.", "Applies to this note only; recording, OTTO and old OTO unchanged." } }
    ,{ "Change editing scope", { "切换编辑范围", "切換編輯範圍", "編集範囲を変更", "편집 범위 변경", "Change editing scope" } }
    ,{ "Discard the unpublished draft and reload the destination scope?", { "丢弃未应用的草稿并重新载入目标范围？", "捨棄未套用草稿並重新載入目標範圍？", "未適用の下書きを破棄し、変更先の範囲を再読込しますか？", "미적용 초안을 버리고 대상 범위를 다시 불러올까요?", "Discard the unpublished draft and reload the destination scope?" } }
    ,{ "Discard and switch", { "丢弃并切换", "捨棄並切換", "破棄して切替", "버리고 전환", "Discard and switch" } }
    ,{ "Discard the unpublished draft and restore this note's source settings?", { "丢弃未应用的草稿并恢复此音符的源设置？", "捨棄未套用草稿並還原此音符的來源設定？", "未適用の下書きを破棄し、このノートのソース設定を復元しますか？", "미적용 초안을 버리고 이 음표의 원본 설정을 복원할까요?", "Discard the unpublished draft and restore this note's source settings?" } }
    ,{ "Discard and restore", { "丢弃并恢复", "捨棄並還原", "破棄して復元", "버리고 복원", "Discard and restore" } }
    ,{ "C - P must be finite nonnegative milliseconds.", { "C - P 必须为非负有限毫秒数。", "C - P 必須是有限的非負毫秒數。", "C - P は有限の非負ミリ秒で指定してください。", "C - P는 유한한 음이 아닌 밀리초 값이어야 합니다.", "C - P must be finite nonnegative milliseconds." } }
    ,{ "Save to otto.ini", { "保存到 otto.ini", "儲存至 otto.ini", "otto.ini に保存", "otto.ini에 저장", "Save to otto.ini" } }
    ,{ "Multiple pronunciations: outer bounds are edited only in overview.", { "多发音：外边界仅在总览编辑；此处只编辑内部参数。", "多發音：外邊界僅在總覽編輯。", "複数発音：外側の境界は一覧でのみ編集できます。", "다중 발음: 외부 경계는 개요에서만 편집합니다.", "Multiple pronunciations: outer bounds are edited only in overview." } }
    ,{ "Native draft; original audio and old OTO stay unchanged.", { "原生草稿，原音频与旧 OTO 保持不变。", "原生草稿；原音訊與舊 OTO 不變。", "ネイティブ下書きです。元の音声と旧 OTO は変更しません。", "네이티브 초안이며 원본 오디오와 기존 OTO는 변경하지 않습니다.", "Native draft; original audio and old OTO stay unchanged." } }
    ,{ "Stretch weight must be positive and finite.", { "拉伸权重必须为正有限数。", "伸縮權重必須為有限正數。", "伸縮の重みは有限の正数で指定してください。", "늘이기 가중치는 유한한 양수여야 합니다.", "Stretch weight must be positive and finite." } }
    ,{ "Vowel name and weight updated; C/P unchanged.", { "已更新元音名称与权重，C/P 保持不变。", "已更新母音名稱與權重，C/P 不變。", "母音名と重みを更新しました。C/P は変更しません。", "모음 이름과 가중치를 갱신했습니다. C/P는 변경되지 않습니다.", "Vowel name and weight updated; C/P unchanged." } }
    ,{ "Named vowel parts created; select a part to adjust its weight.", { "已创建命名元音分段，可选择各段调整权重。", "已建立命名母音分段；選取區段調整權重。", "名前付き母音区間を作成しました。区間を選択して重みを調整できます。", "명명 모음 구간을 만들었습니다. 구간을 선택하여 가중치를 조절하세요.", "Named vowel parts created; select a part to adjust its weight." } }
    ,{ "Inherit engine OTO defaults", { "继承引擎 OTO 默认值", "繼承引擎 OTO 預設值", "エンジン既定 OTO を継承", "엔진 기본 OTO 상속", "Inherit engine OTO defaults" } }
    ,{ "Apply to this note", { "应用到此音符", "套用至此音符", "このノートに適用", "이 음표에 적용", "Apply to this note" } }
    ,{ "Save source annotations", { "保存源标注", "儲存來源標註", "ソース注釈を保存", "원본 주석 저장", "Save source annotations" } }
    ,{ "Parameters must be finite numeric milliseconds.", { "参数必须为有限的毫秒数。", "參數必須為有限的毫秒數。", "パラメーターは有限のミリ秒値で指定してください。", "매개변수는 유한한 밀리초 값이어야 합니다.", "Parameters must be finite numeric milliseconds." } }
    ,{ "Choose an internal elastic vowel segment in native view.", { "请在原生视图选择内部弹性元音区段。", "請在原生檢視選取內部可伸縮母音區段。", "ネイティブ表示で内部の伸縮可能な母音区間を選択してください。", "네이티브 보기에서 내부 가변 모음 구간을 선택하세요.", "Choose an internal elastic vowel segment in native view." } }
    ,{ "Applied to this note; recording, OTTO and old OTO unchanged.", { "已应用到此音符；源录音、OTTO 和旧 OTO 未修改。", "已套用此音符；錄音、OTTO 與舊 OTO 不變。", "このノートに適用しました。録音・OTTO・旧 OTO は変更しません。", "이 음표에 적용했습니다. 녹음, OTTO 및 기존 OTO는 변경하지 않습니다.", "Applied to this note; recording, OTTO and old OTO unchanged." } }
    ,{ "Saved otto.ini; original audio and old OTO unchanged.", { "标注已保存到 otto.ini；原音频和旧 OTO 未修改。", "已儲存 otto.ini；原音訊與舊 OTO 不變。", "otto.ini を保存しました。元の音声と旧 OTO は変更しません。", "otto.ini를 저장했습니다. 원본 오디오와 기존 OTO는 변경하지 않습니다.", "Saved otto.ini; original audio and old OTO unchanged." } }
    ,{ "Select an existing pronunciation anchor first.", { "请先选择相邻发音的锚点。", "請先選取已有發音作為錨點。", "先に基準となる既存の発音を選択してください。", "먼저 기준이 될 기존 발음을 선택하세요.", "Select an existing pronunciation anchor first." } }
    ,{ "Type an alias, then choose Alias selected range.", { "输入别名，然后点击“选区创建别名”。", "輸入別名後，按「選區建立別名」。", "エイリアスを入力し「選択範囲にエイリアス」を選んでください。", "별칭 입력 후 선택 범위 별칭 만들기를 선택하세요.", "Type an alias, then choose Alias selected range." } }
    ,{ "Primary alias", { "主别名", "主要別名", "主エイリアス", "기본 별칭", "Primary alias" } }
    ,{ "Additional aliases (semicolon)", { "附加别名（分号分隔）", "附加別名（分號分隔）", "追加エイリアス（セミコロン区切り）", "추가 별칭 (세미콜론 구분)", "Additional aliases (semicolon)" } }
    ,{ "Split positions ms (comma/space)", { "切分位置 ms（逗号或空格）", "分割位置 ms（逗號／空格）", "分割位置 ms（コンマ／空白）", "분할 위치 ms (쉼표/공백)", "Split positions ms (comma/space)" } }
    ,{ "Name all pronunciations in order (whitespace)", { "依次命名所有发音（空白分隔）", "依序命名所有發音（空白分隔）", "全発音を順に命名（空白区切り）", "모든 발음을 순서대로 명명 (공백 구분)", "Name all pronunciations in order (whitespace)" } }
    ,{ "Segments (role:absolute end ms, e.g. C:100,V:350,V:600)", { "内部区段（角色:绝对终点 ms，如 C:100,V:350,V:600）", "區段（角色:絕對終點 ms，如 C:100,V:350,V:600）", "区間（役割:絶対終端 ms、例 C:100,V:350,V:600）", "구간 (역할:절대 끝 ms, 예 C:100,V:350,V:600)", "Segments (role:absolute end ms, e.g. C:100,V:350,V:600)" } }
    ,{ "U unknown, C consonant, V vowel, T transition, S silence, B breath, N noise, E ending. Edited form makes contiguous segments ending at pronunciation end. Untouched form preserves all metadata and gaps.", { "U未知 C辅音 V元音 T过渡 S静音 B呼吸 N噪音 E收尾。编辑后为连续区段，最后终点须等于发音终点；不编辑则完整保留原区段及间隙。", "U未知 C子音 V母音 T過渡 S靜音 B呼吸 N噪音 E結尾。編輯後為連續區段且終點須等於發音終點；未編輯則保留中繼資料及間隙。", "U不明 C子音 V母音 T遷移 S無音 Bブレス N雑音 E語尾。編集後は発音終端までの連続区間となります。未編集なら情報と間隔を保ちます。", "U미상 C자음 V모음 T전이 S무음 B호흡 N잡음 E끝. 편집하면 발음 끝까지 연속 구간이 됩니다. 편집하지 않으면 정보와 간격을 보존합니다.", "U unknown, C consonant, V vowel, T transition, S silence, B breath, N noise, E ending. Edited form makes contiguous segments ending at pronunciation end. Untouched form preserves all metadata and gaps." } }
    ,{ "Pronunciation identity, not an internal C/V segment role.", { "整个发音的类别，不是内部 C/V 区段。", "整個發音的類別，不是內部 C/V 區段角色。", "発音全体の種類です。内部 C/V 区間の役割ではありません。", "전체 발음 유형이며 내부 C/V 구간 역할이 아닙니다.", "Pronunciation identity, not an internal C/V segment role." } }
    ,{ "Adjacent pronunciation", { "相邻发音模式", "相鄰發音模式", "隣接発音モード", "인접 발음 모드", "Adjacent pronunciation" } }
    ,{ "Select an existing pronunciation, then drag a neighboring range; its join stays exact.", { "先选择已有发音，向左或右拖选紧邻范围；接点固定。", "先選取已有發音，再拖選相鄰範圍；接點保持精確。", "既存の発音を選び、隣の範囲をドラッグします。接点は正確に維持します。", "기존 발음을 선택한 다음 인접 범위를 드래그하세요. 연결점은 정확히 유지됩니다.", "Select an existing pronunciation, then drag a neighboring range; its join stays exact." } }
    ,{ "Apply fields", { "应用参数", "套用參數", "入力値を適用", "입력값 적용", "Apply fields" } }
    ,{ "Split", { "切分", "分割", "分割", "분할", "Split" } }
    ,{ "Batch name", { "依次命名", "依序命名", "一括命名", "일괄 명명", "Batch name" } }
    ,{ "Merge selected", { "合并所选", "合併所選", "選択区間を結合", "선택 구간 병합", "Merge selected" } }
    ,{ "Delete selected", { "删除所选", "刪除所選", "選択を削除", "선택 항목 삭제", "Delete selected" } }
    ,{ "Undo", { "撤销", "復原", "元に戻す", "실행 취소", "Undo" } }
    ,{ "Redo", { "重做", "重做", "やり直す", "다시 실행", "Redo" } }
    ,{ "Save annotations", { "保存标注", "儲存標註", "注釈を保存", "주석 저장", "Save annotations" } }
    ,{ "Alias selected range", { "选区创建别名", "選區建立別名", "選択範囲にエイリアス", "선택 범위 별칭 만들기", "Alias selected range" } }
    ,{ "Remove current alias", { "取消当前别名", "移除目前別名", "現在のエイリアスを削除", "현재 별칭 제거", "Remove current alias" } }
    ,{ "Cancel a pending range, or remove the existing alias typed in Primary alias. Source audio stays unchanged.", { "取消未创建的范围，或移除主别名框中输入的现有别名；原音频保持不变。", "取消待建範圍，或移除主別名框中的既有別名；原音訊不變。", "未作成の範囲を取り消すか、主エイリアス欄の既存名を削除します。音声は変更しません。", "대기 범위를 취소하거나 기본 별칭에 입력한 기존 별칭을 제거합니다. 원본 오디오는 변경하지 않습니다.", "Cancel a pending range, or remove the existing alias typed in Primary alias. Source audio stays unchanged." } }
    ,{ "Detail parameters", { "详细参数", "詳細參數", "詳細パラメーター", "상세 매개변수", "Detail parameters" } }
    ,{ "Drag a source range to name it. Double-click a pronunciation for waveform detail. Removing aliases never changes audio.", { "拖选范围创建别名；双击发音进入详细波形参数；取消别名不改原音频。", "拖選來源範圍命名；雙擊發音開啟波形詳細參數。移除別名不改音訊。", "ソース範囲をドラッグして命名します。発音をダブルクリックすると波形詳細を表示します。エイリアス削除で音声は変わりません。", "원본 범위를 드래그하여 명명하세요. 발음을 두 번 클릭하면 파형 상세를 엽니다. 별칭을 제거해도 오디오는 변경되지 않습니다.", "Drag a source range to name it. Double-click a pronunciation for waveform detail. Removing aliases never changes audio." } }
    ,{ "Cannot open annotation draft.", { "无法打开标注草稿。", "無法開啟標註草稿。", "注釈の下書きを開けません。", "주석 초안을 열 수 없습니다.", "Cannot open annotation draft." } }
    ,{ "First drag a range on the source waveform.", { "请先拖选原音频范围。", "請先拖選原音訊範圍。", "先にソース波形の範囲をドラッグしてください。", "먼저 원본 파형 범위를 드래그하세요.", "First drag a range on the source waveform." } }
    ,{ "Uncreated range alias cancelled.", { "已取消尚未创建的选区别名。", "已取消尚未建立的選區別名。", "未作成の範囲エイリアスを取り消しました。", "아직 만들지 않은 범위 별칭을 취소했습니다.", "Uncreated range alias cancelled." } }
    ,{ "Enter finite numeric milliseconds.", { "请输入有限的毫秒数。", "請輸入有限的毫秒數。", "有限のミリ秒値を入力してください。", "유한한 밀리초 값을 입력하세요.", "Enter finite numeric milliseconds." } }
    ,{ "Enter internal segments.", { "请输入内部区段。", "請輸入內部區段。", "内部区間を入力してください。", "내부 구간을 입력하세요.", "Enter internal segments." } }
    ,{ "Invalid segment syntax; use C:100,V:350.", { "区段格式无效，请使用 C:100,V:350 等。", "區段格式無效，請使用 C:100,V:350。", "区間形式が無効です。C:100,V:350 の形式で指定してください。", "잘못된 구간 형식입니다. C:100,V:350 형식을 사용하세요.", "Invalid segment syntax; use C:100,V:350." } }
    ,{ "Segment ends must increase within the pronunciation; the last must equal its end.", { "区段须为正长度、递增且位于发音内，最后终点须等于发音终点。", "區段終點須在發音內遞增，最後須等於發音終點。", "区間終端は発音内で昇順とし、最後は発音終端に一致させてください。", "구간 끝은 발음 안에서 증가해야 하며 마지막은 발음 끝과 같아야 합니다.", "Segment ends must increase within the pronunciation; the last must equal its end." } }
    ,{ "Invalid split position.", { "切分位置无效。", "分割位置無效。", "分割位置が無効です。", "잘못된 분할 위치입니다.", "Invalid split position." } }
    ,{ "Enter split positions.", { "请输入切分位置。", "請輸入分割位置。", "分割位置を入力してください。", "분할 위치를 입력하세요.", "Enter split positions." } }
    ,{ "Annotations saved; source audio unchanged.", { "标注已保存；原音频未修改。", "標註已儲存，原音訊不變。", "注釈を保存しました。元の音声は変更しません。", "주석을 저장했습니다. 원본 오디오는 변경되지 않습니다.", "Annotations saved; source audio unchanged." } }
    ,{ "Start ms", { "起点 ms", "起點 ms", "開始 ms", "시작 ms", "Start ms" } }
    ,{ "End ms", { "终点 ms", "終點 ms", "終了 ms", "끝 ms", "End ms" } }
    ,{ "Fixed ms", { "固定时长 ms", "固定時長 ms", "固定長 ms", "고정 길이 ms", "Fixed ms" } }
    ,{ "Preutterance ms", { "先行时长 ms", "先行時長 ms", "先行発声 ms", "선행 발성 ms", "Preutterance ms" } }
    ,{ "Overlap ms", { "重叠 ms", "重疊 ms", "オーバーラップ ms", "오버랩 ms", "Overlap ms" } }
    ,{ "New HachiShifter track", { "新建 HachiShifter 轨道", "新增 HachiShifter 軌道", "HachiShifter トラックを追加", "HachiShifter 트랙 추가", "New HachiShifter track" } }
    ,{ "New UTAU track", { "新建 UTAU 轨道", "新增 UTAU 軌道", "UTAU トラックを追加", "UTAU 트랙 추가", "New UTAU track" } }
    ,{ "New DiffSinger track", { "新建 DiffSinger 轨道", "新增 DiffSinger 軌道", "DiffSinger トラックを追加", "DiffSinger 트랙 추가", "New DiffSinger track" } }
    ,{ "Show note overlap lines", { "显示音符重叠线", "顯示音符重疊線", "ノートの重なり線を表示", "음표 겹침선 표시", "Show note overlap lines" } }
    ,{ "Whole current track pitch edit", { "当前整轨音高编辑（跨素材）", "目前整軌音高編輯（跨素材）", "現在のトラック全体のピッチ編集", "현재 트랙 전체 피치 편집", "Whole current track pitch edit" } }
    ,{ "Track pitch", { "整轨音高", "整軌音高", "トラックのピッチ", "트랙 피치", "Track pitch" } }
    ,{ "Whole-track pitch editing across all materials and clips; gaps are not connected automatically.", { "当前整轨音高编辑：跨所有素材和片段；不会自动连接空隙。", "目前整軌音高編輯：跨所有素材和片段；不會自動連接間隙。", "全素材・クリップをまたいで編集します。隙間は自動接続しません。", "모든 소재와 클립에 걸쳐 피치를 편집합니다. 간격은 자동 연결하지 않습니다.", "Whole-track pitch editing across all materials and clips; gaps are not connected automatically." } }
    ,{ "Clip-focused pitch editing; switch to whole-track scope in View.", { "片段聚焦音高编辑；可在显示菜单切换整轨范围。", "片段聚焦音高編輯；可在顯示選單切換整軌範圍。", "クリップ単位のピッチ編集です。表示メニューで全トラック範囲に切替できます。", "클립 중심 피치 편집입니다. 보기 메뉴에서 트랙 전체 범위로 전환할 수 있습니다.", "Clip-focused pitch editing; switch to whole-track scope in View." } }
    ,{ "Material selector", { "素材选择器", "素材選擇器", "素材セレクター", "소재 선택기", "Material selector" } }
    ,{ "Replace material...", { "替换素材…", "替換素材…", "素材を置換…", "소재 바꾸기…", "Replace material..." } }
    ,{ "Lyrics", { "歌词", "歌詞", "歌詞", "가사", "Lyrics" } }
    ,{ "OTO alias", { "发音／别名", "發音／別名", "発音／エイリアス", "발음/별칭", "OTO alias" } }
    ,{ "Choose model", { "选择模型", "選擇模型", "モデルを選択", "모델 선택", "Choose model" } }
    ,{ "Phoneme timing", { "音素时长", "音素時長", "音素の長さ", "음소 길이", "Phoneme timing" } }
    ,{ "Pronunciation", { "发音／读音", "發音／讀音", "発音／読み", "발음/읽기", "Pronunciation" } }
    ,{ "Generate pitch", { "生成音高", "產生音高", "ピッチを生成", "피치 생성", "Generate pitch" } }
    ,{ "Variance parameters", { "实参／偏移", "實參／偏移", "表現パラメーター／オフセット", "표현 매개변수/오프셋", "Variance parameters" } }
    ,{ "Language / speaker", { "语言／音色设置", "語言／音色設定", "言語／話者の設定", "언어/화자 설정", "Language / speaker" } }
    ,{ "Language: ", { "语言：", "語言：", "言語：", "언어: ", "Language: " } }
    ,{ "  Speaker: ", { "  音色：", "  音色：", "  話者：", "  화자: ", "  Speaker: " } }
    ,{ "Register project source materials", { "登记工程源素材", "登記工程來源素材", "プロジェクトのソース素材を登録", "프로젝트 원본 소재 등록", "Register project source materials" } }
    ,{ "Register project source materials...", { "登记工程源素材…", "登記工程來源素材…", "プロジェクトのソース素材を登録…", "프로젝트 원본 소재 등록…", "Register project source materials..." } }
    ,{ "Retry material folder registration", { "重试登记已保留的素材文件夹", "重試登記已保留的素材資料夾", "保持済み素材フォルダーの登録を再試行", "보존된 소재 폴더 등록 다시 시도", "Retry material folder registration" } }
    ,{ "Register copies...", { "复制并登记…", "複製並登記…", "コピーして登録…", "복사 후 등록…", "Register copies..." } }
    ,{ "Skip", { "跳过", "略過", "スキップ", "건너뛰기", "Skip" } }
    ,{ "Choose parent folder for NEW project materials", { "选择新素材子文件夹的父目录", "選擇新素材子資料夾的父目錄", "新規素材サブフォルダーの親を選択", "새 소재 하위 폴더의 상위 폴더 선택", "Choose parent folder for NEW project materials" } }
    ,{ "Optionally copy the existing unregistered recordings into a NEW material folder and keep each carried clip/part annotation as an independent variant named '-'. This does not recover words or syllables. Original C/P, source clocks, pitch, envelopes and local edits remain exact. Skip keeps the loaded project unchanged; project files are saved only when you choose Save.", { "可把未登记录音复制到新素材文件夹，并将各片段已有标注保留为独立的 '-' 变体。这不是词语或音节识别。原 C/P、源时钟、音高、包络及局部编辑保持不变。跳过保持工程不变，只有主动保存才写工程文件。", "可將未登記錄音複製至新素材資料夾，並將各片段已有標註保留為獨立的 '-' 變體。這不是詞語或音節辨識。原 C/P、來源時鐘、音高、包絡及局部編輯保持不變。略過保持工程不變，僅主動儲存才寫入工程檔。", "未登録の録音を新規素材フォルダーへコピーし、各クリップの既存注釈を '-' という独立バリアントとして保持できます。単語・音節の認識ではありません。C/P、ソース時刻、ピッチ、エンベロープ、局所編集は維持します。スキップすると変更せず、保存を選ぶまでプロジェクトは書き込みません。", "미등록 녹음을 새 소재 폴더에 복사하고 각 클립의 기존 주석을 '-' 이름의 독립 변형으로 보존할 수 있습니다. 단어나 음절 인식이 아닙니다. C/P, 원본 시간, 피치, 엔벌로프 및 로컬 편집을 유지합니다. 건너뛰면 프로젝트는 바뀌지 않으며 저장을 선택할 때만 기록합니다.", "Optionally copy the existing unregistered recordings into a NEW material folder and keep each carried clip/part annotation as an independent variant named '-'. This does not recover words or syllables. Original C/P, source clocks, pitch, envelopes and local edits remain exact. Skip keeps the loaded project unchanged; project files are saved only when you choose Save." } }
    ,{ "No current track; export defaults to the whole project.", { "未选择轨道，默认导出整个工程。", "未選擇軌道，預設匯出整個工程。", "トラック未選択のためプロジェクト全体を出力します。", "현재 트랙이 없어 프로젝트 전체를 내보냅니다.", "No current track; export defaults to the whole project." } }
    ,{ "Current track", { "当前轨道", "目前軌道", "現在のトラック", "현재 트랙", "Current track" } }
    ,{ "Whole project", { "整个工程", "整個工程", "プロジェクト全体", "프로젝트 전체", "Whole project" } }
    ,{ "Export scope", { "导出范围", "匯出範圍", "出力範囲", "내보내기 범위", "Export scope" } }
    ,{ "Include muted tracks and clips", { "包含静音轨道和片段", "包含靜音軌道和片段", "ミュートしたトラック・クリップを含む", "음소거 트랙 및 클립 포함", "Include muted tracks and clips" } }
    ,{ "Choose save location", { "选择保存位置", "選擇儲存位置", "保存先を選択", "저장 위치 선택", "Choose save location" } }
    ,{ "Automatic (GPU preferred, CPU fallback)", { "自动（优先 GPU，失败时使用 CPU）", "自動（優先 GPU，失敗時使用 CPU）", "自動（GPU 優先、失敗時 CPU）", "자동 (GPU 우선, 실패 시 CPU)", "Automatic (GPU preferred, CPU fallback)" } }
    ,{ "Fast", { "快速", "快速", "高速", "빠름", "Fast" } }
    ,{ "Standard", { "标准", "標準", "標準", "표준", "Standard" } }
    ,{ "Fine", { "精细", "精細", "高精度", "정밀", "Fine" } }
    ,{ "Custom", { "自定义", "自訂", "カスタム", "사용자 지정", "Custom" } }
    ,{ "DS inference device", { "DS 推理设备", "DS 推論裝置", "DS 推論デバイス", "DS 추론 장치", "DS inference device" } }
    ,{ "GPU index", { "显卡编号", "顯示卡編號", "GPU 番号", "GPU 번호", "GPU index" } }
    ,{ "Preview / pitch prediction quality", { "试听 / pitch 预测质量", "試聽 / pitch 預測品質", "試聴／ピッチ予測品質", "미리 듣기/피치 예측 품질", "Preview / pitch prediction quality" } }
    ,{ "Export quality", { "导出质量", "匯出品質", "出力品質", "내보내기 품질", "Export quality" } }
    ,{ "Custom: acoustic steps", { "自定义：声学步数", "自訂：聲學步數", "カスタム：音響ステップ数", "사용자 지정: 음향 단계 수", "Custom: acoustic steps" } }
    ,{ "Custom: pitch steps", { "自定义：pitch 步数", "自訂：pitch 步數", "カスタム：ピッチステップ数", "사용자 지정: 피치 단계 수", "Custom: pitch steps" } }
    ,{ "Custom: variance steps", { "自定义：表现参数步数", "自訂：表現參數步數", "カスタム：表現ステップ数", "사용자 지정: 표현 단계 수", "Custom: variance steps" } }
    ,{ "Custom: acoustic depth", { "自定义：声学深度", "自訂：聲學深度", "カスタム：音響深度", "사용자 지정: 음향 깊이", "Custom: acoustic depth" } }
    ,{ "Steps (acoustic / pitch / variance): Fast 8 / 8 / 8, Standard 20 / 20 / 20, Fine 50 / 40 / 40. Export uses existing pitch without changing it. Depth is limited by the voicebank.", { "步数（声学 / pitch / 表现参数）：快速 8 / 8 / 8，标准 20 / 20 / 20，精细 50 / 40 / 40。导出使用现有 pitch，不改音高；深度按音源上限限制。", "步數（聲學 / pitch / 表現參數）：快速 8 / 8 / 8，標準 20 / 20 / 20，精細 50 / 40 / 40。匯出使用現有 pitch，不改音高；深度依音源上限限制。", "ステップ数（音響／ピッチ／表現）：高速 8/8/8、標準 20/20/20、高精度 50/40/40。出力は既存のピッチを変更せず使います。深度は音源の上限内に制限します。", "단계 수 (음향/피치/표현): 빠름 8/8/8, 표준 20/20/20, 정밀 50/40/40. 내보내기는 기존 피치를 변경하지 않고 사용합니다. 깊이는 음원 한도로 제한됩니다.", "Steps (acoustic / pitch / variance): Fast 8 / 8 / 8, Standard 20 / 20 / 20, Fine 50 / 40 / 40. Export uses existing pitch without changing it. Depth is limited by the voicebank." } }
    ,{ "DS has not run yet; check the device during voicebank preflight, prediction or preview after applying.", { "尚未运行 DS；应用后在音源预检、预测或试听时检查设备。", "尚未執行 DS；套用後於音源預檢、預測或試聽時檢查裝置。", "DS は未実行です。適用後、音源検査・予測・試聴でデバイスを確認してください。", "DS가 아직 실행되지 않았습니다. 적용 후 음원 검사, 예측 또는 미리 듣기에서 장치를 확인하세요.", "DS has not run yet; check the device during voicebank preflight, prediction or preview after applying." } }
    ,{ "Last run: ", { "最近运行：", "最近執行：", "前回の実行：", "최근 실행: ", "Last run: " } }
    ,{ "The project changed before registration. Reopen the registration action for the current project.", { "工程在登记前已变化。请为当前工程重新打开登记操作。", "工程在登記前已變更。請為目前工程重新開啟登記操作。", "登録前にプロジェクトが変更されました。現在のプロジェクトで登録を開き直してください。", "등록 전에 프로젝트가 변경되었습니다. 현재 프로젝트에서 등록을 다시 여세요.", "The project changed before registration. Reopen the registration action for the current project." } }
    ,{ "The project changed; registration was cancelled without changing references.", { "工程已变化；登记已取消，未改变引用。", "工程已變更；登記已取消，未改變引用。", "プロジェクトが変更されたため、参照を変更せず登録を取り消しました。", "프로젝트가 변경되어 참조를 바꾸지 않고 등록을 취소했습니다.", "The project changed; registration was cancelled without changing references." } }
    ,{ "The project/source snapshot changed. Only the owned unadopted folder was rolled back.", { "工程或来源快照已变化。仅回滚尚未采用的本次自有文件夹。", "工程或來源快照已變更。僅復原尚未採用的本次自有資料夾。", "プロジェクト／ソースが変更されました。未採用の今回作成フォルダーのみを戻しました。", "프로젝트/원본 스냅샷이 변경되었습니다. 아직 채택하지 않은 이번 소유 폴더만 되돌렸습니다.", "The project/source snapshot changed. Only the owned unadopted folder was rolled back." } }
    ,{ "Project references and the copied audio/OTTO are safely retained at:\n", { "工程引用及复制的音频／OTTO 已安全保留于：\n", "工程引用及複製的音訊／OTTO 已安全保留於：\n", "プロジェクト参照とコピーした音声／OTTO の保存先：\n", "프로젝트 참조와 복사한 오디오/OTTO가 안전하게 보존된 위치:\n", "Project references and the copied audio/OTTO are safely retained at:\n" } }
    ,{ "\nLibrary preference registration failed: ", { "\n素材库偏好登记失败：", "\n素材庫偏好登記失敗：", "\nライブラリ設定の登録に失敗しました：", "\n라이브러리 환경 설정 등록 실패: ", "\nLibrary preference registration failed: " } }
    ,{ "\nUse File > Retry material folder registration. No project file was automatically saved.", { "\n请用“文件 > 重试登记已保留的素材文件夹”。工程文件没有自动保存。", "\n請用「檔案 > 重試登記已保留的素材資料夾」。工程檔沒有自動儲存。", "\n「ファイル > 素材フォルダーの登録を再試行」を使用してください。プロジェクトは自動保存していません。", "\n파일 > 보존된 소재 폴더 등록 다시 시도를 사용하세요. 프로젝트 파일은 자동 저장되지 않았습니다.", "\nUse File > Retry material folder registration. No project file was automatically saved." } }
    ,{ "Project source materials registered. Musical edits were preserved; choose Save to persist new references.", { "工程源素材已登记，调音编辑保持不变；请保存工程以保留新引用。", "工程來源素材已登記，調音編輯保持不變；請儲存工程以保留新引用。", "ソース素材を登録し、音楽編集を維持しました。新しい参照を永続化するには保存してください。", "프로젝트 원본 소재를 등록하고 음악 편집을 보존했습니다. 새 참조를 유지하려면 저장하세요.", "Project source materials registered. Musical edits were preserved; choose Save to persist new references." } }
    ,{ "Material manager is unavailable", { "素材管理器不可用", "素材管理員無法使用", "素材マネージャーを利用できません", "소재 관리자를 사용할 수 없습니다", "Material manager is unavailable" } }
    ,{ "Material folders ▾", { "素材文件夹 ▾", "素材資料夾 ▾", "素材フォルダー ▾", "소재 폴더 ▾", "Material folders ▾" } }
    ,{ "Reading OTO annotations...", { "正在读取 OTO 标注…", "正在讀取 OTO 標註…", "OTO 注釈を読込中…", "OTO 주석 읽는 중…", "Reading OTO annotations..." } }
    ,{ "No pronunciations; right-click to annotate", { "尚无发音标注；右键编辑", "尚無發音標註；按右鍵編輯", "発音注釈なし。右クリックで編集", "발음 주석 없음. 오른쪽 클릭으로 편집", "No pronunciations; right-click to annotate" } }
    ,{ "Migrate to native OTTO annotations...", { "迁移为 OTTO 原生标注…", "遷移為 OTTO 原生標註…", "ネイティブ OTTO 注釈に移行…", "네이티브 OTTO 주석으로 이전…", "Migrate to native OTTO annotations..." } }
    ,{ "Use for current track", { "用于当前轨道", "用於目前軌道", "現在のトラックに使用", "현재 트랙에 사용", "Use for current track" } }
    ,{ "Edit material annotations...", { "编辑素材标注…", "編輯素材標註…", "素材注釈を編集…", "소재 주석 편집…", "Edit material annotations..." } }
    ,{ "Migrate this folder to one otto.ini. Bare audio becomes one pronunciation; existing OTTO entries are kept. Original audio and legacy annotations remain unchanged.", { "将此文件夹旧标注迁移至一个 otto.ini。未标注录音建立整段发音；保留已有 OTTO。原录音与旧标注不变。", "將此資料夾舊標註遷移至一個 otto.ini。未標註錄音建立整段發音；保留已有 OTTO。原錄音與舊標註不變。", "フォルダーを一つの otto.ini に移行します。未注釈音声は全体を一発音にし、既存 OTTO は保持します。元音声と旧注釈は変更しません。", "폴더를 하나의 otto.ini로 이전합니다. 미주석 오디오는 전체를 한 발음으로 만들고 기존 OTTO는 보존합니다. 원본 오디오와 이전 주석은 변경하지 않습니다.", "Migrate this folder to one otto.ini. Bare audio becomes one pronunciation; existing OTTO entries are kept. Original audio and legacy annotations remain unchanged." } }
    ,{ "Migrate this folder to native OTTO annotations first (folder context menu).", { "请先在文件夹右键菜单迁移为 OTTO 原生标注。", "請先在資料夾右鍵選單遷移為 OTTO 原生標註。", "先にフォルダーの右クリックメニューから OTTO 注釈に移行してください。", "먼저 폴더 오른쪽 클릭 메뉴에서 네이티브 OTTO 주석으로 이전하세요.", "Migrate this folder to native OTTO annotations first (folder context menu)." } }
    ,{ "Standard UTAU OTO loses native connections, multi-vowel segmentation, transition flags and expression data. otto.ini is unchanged; existing targets merge by recording path. Continue?", { "标准 UTAU OTO 会丢失原生连接、多元音分段、过渡标记及表达数据。otto.ini 不变；已有目标按录音路径合并。继续？", "標準 UTAU OTO 會遺失原生連接、多母音分段、過渡標記及表現資料。otto.ini 不變；已有目標依錄音路徑合併。繼續？", "標準 UTAU OTO では接続・母音の複数区間・遷移・表現データが失われます。otto.ini は変更せず、既存出力は録音パスで統合します。続行しますか？", "표준 UTAU OTO는 연결, 다중 모음 구간, 전이 및 표현 데이터를 잃습니다. otto.ini는 그대로이며 기존 대상은 녹음 경로별로 병합합니다. 계속할까요?", "Standard UTAU OTO loses native connections, multi-vowel segmentation, transition flags and expression data. otto.ini is unchanged; existing targets merge by recording path. Continue?" } }
    ,{ "Audio: ", { "录音：", "錄音：", "録音：", "녹음: ", "Audio: " } }
    ,{ " / Pronunciations: ", { " / 发音：", " / 發音：", " / 発音：", " / 발음: ", " / Pronunciations: " } }
    ,{ "Note", { "音符", "音符", "ノート", "음표", "Note" } }
    ,{ "Loudness envelope", { "响度包络", "響度包絡", "音量エンベロープ", "음량 엔벌로프", "Loudness envelope" } }
    ,{ "Source-note boundary split/join, not forced pitch portamento", { "源音符边界拆分／连接，不是强制音高滑音", "來源音符邊界分割／連接，不是強制音高滑音", "ソースノート境界の分割／結合。強制ピッチポルタメントではありません", "원본 음표 경계 분할/연결이며 강제 피치 포르타멘토가 아닙니다", "Source-note boundary split/join, not forced pitch portamento" } }
    ,{ "No DiffSinger track selected", { "未选择 DiffSinger 轨道", "未選取 DiffSinger 軌道", "DiffSinger トラック未選択", "DiffSinger 트랙이 선택되지 않음", "No DiffSinger track selected" } }
    ,{ "No model configured", { "尚未配置模型", "尚未設定模型", "モデル未設定", "모델이 설정되지 않음", "No model configured" } }
    ,{ "高级包络", { "高级包络", "進階包絡", "高度なエンベロープ", "고급 엔벌로프", "Advanced envelope" } }
    ,{ "自动交叉淡化", { "自动交叉淡化", "自動交叉淡化", "自動クロスフェード", "자동 크로스페이드", "Automatic crossfade" } }
    ,{ "强制连接", { "强制连接", "強制連接", "強制接続", "강제 연결", "Force connection" } }
    ,{ "线性flag", { "线性flag", "線性 FLAG", "連続 FLAG", "연속 FLAG", "Continuous FLAG" } }
    ,{ "flag 包络", { "flag 包络", "FLAG 包絡", "FLAG エンベロープ", "FLAG 엔벌로프", "FLAG envelope" } }
    ,{ "包络预设", { "包络预设", "包絡預設", "エンベローププリセット", "엔벌로프 프리셋", "Envelope preset" } }
    ,{ "选择预设，应用到所选音符或单独选中的音素", { "选择预设，应用到所选音符或单独选中的音素", "選擇預設，套用至所選音符或單獨所選音素", "プリセットを選択し、選択ノートまたは音素に適用", "선택한 음표 또는 개별 음소에 프리셋 적용", "Choose a preset for selected notes or an individually selected phoneme" } }
    ,{ "包络预设：", { "包络预设：", "包絡預設：", "エンベロープ：", "엔벌로프 프리셋: ", "Envelope preset: " } }
    ,{ " 个音符", { " 个音符", " 個音符", " ノート", "개 음표", " notes" } }
    ,{ "先选中要套用预设的音符。", { "先选中要套用预设的音符。", "請先選取要套用預設的音符。", "先に適用するノートを選択してください。", "먼저 프리셋을 적용할 음표를 선택하세요.", "Select notes before applying a preset." } }
    ,{ "已生成交叉淡化，可在响度包络中编辑或撤销。", { "已生成交叉淡化，可在响度包络中编辑或撤销。", "已產生交叉淡化，可編輯響度包絡或復原。", "クロスフェードを作成しました。音量エンベロープで編集・取消できます。", "크로스페이드를 만들었습니다. 음량 엔벌로프에서 편집하거나 취소할 수 있습니다.", "Crossfade created; edit it in the loudness envelope or undo." } }
    ,{ "请选择同轨的两个前后重叠音符；无重叠、完全包含或音源尚未解析时无法交叉淡化。", { "请选择同轨的两个前后重叠音符；无重叠、完全包含或音源尚未解析时无法交叉淡化。", "請選取同軌前後重疊的兩個音符；無重疊、完全包含或音源未解析時無法交叉淡化。", "同じトラックの前後に重なる二つのノートを選択してください。重なりなし・完全包含・音源未解決の場合は適用できません。", "같은 트랙의 앞뒤로 겹치는 두 음표를 선택하세요. 겹침 없음, 완전 포함 또는 미해결 음원에는 적용할 수 없습니다.", "Select two overlapping notes on one track. No overlap, containment or unresolved sources cannot be crossfaded." } }
    ,{ "普通标点编辑模式", { "普通标点编辑模式", "普通控制點編輯模式", "通常のポイント編集モード", "일반 제어점 편집 모드", "Normal point editing mode" } }
    ,{ "标点编辑模式", { "标点编辑模式", "控制點編輯模式", "ポイント編集モード", "제어점 편집 모드", "Point editing mode" } }
    ,{ "普通标点编辑", { "普通标点编辑", "普通控制點編輯", "通常のポイント編集", "일반 제어점 편집", "Normal point editing" } }
    ,{ "还原至初始音高线（DS）", { "还原至初始音高线（DS）", "還原至初始音高線（DS）", "元のピッチに復元（DS）", "원래 피치로 복원 (DS)", "Restore original pitch (DS)" } }
    ,{ "绘制偏移（DS）", { "绘制偏移（DS）", "繪製偏移（DS）", "オフセット描画（DS）", "오프셋 그리기 (DS)", "Draw offset (DS)" } }
    ,{ "需在 DS 轨道先生成 pitch，保存原始虚线参考", { "需在 DS 轨道先生成 pitch，保存原始虚线参考", "須先產生 DS 音高以保留原始虛線參考", "先に DS ピッチを生成して元の点線を保存してください", "먼저 DS 피치를 생성하여 원래 점선 참조를 저장하세요", "Generate DS pitch first to retain the original dotted reference" } }
    ,{ "；右键可选择 DS 音高还原或绘制偏移", { "；右键可选择 DS 音高还原或绘制偏移", "；右鍵可選 DS 音高還原或繪製偏移", "；右クリックで DS ピッチ復元／オフセット描画", "; 오른쪽 클릭으로 DS 피치 복원/오프셋 그리기", "; right-click for DS pitch restore or offset drawing" } }
    ,{ "DS 还原模式：点击标点恢复原始音高；Esc 退出", { "DS 还原模式：点击标点恢复原始音高；Esc 退出", "DS 還原模式：點選控制點還原原始音高；Esc 離開", "DS 復元モード：ポイントをクリックして元のピッチを復元。Esc で終了", "DS 복원 모드: 제어점을 클릭하여 원래 피치 복원; Esc 종료", "DS restore: click points to restore original pitch; Esc exits" } }
    ,{ "DS 还原模式：点击标点恢复初始音高线；Esc 退出", { "DS 还原模式：点击标点恢复初始音高线；Esc 退出", "DS 還原模式：點選控制點還原初始音高線；Esc 離開", "DS 復元：ポイントをクリックして初期ピッチを復元。Esc で終了", "DS 복원: 제어점 클릭으로 초기 피치 복원; Esc 종료", "DS restore: click points to restore initial pitch; Esc exits" } }
    ,{ "编辑音量；高级形状使用高级包络按钮", { "编辑音量；高级形状使用高级包络按钮", "編輯音量；進階形狀使用進階包絡按鈕", "音量を編集します。複雑な形状は高度なエンベロープを使用", "음량을 편집합니다. 고급 모양은 고급 엔벌로프 사용", "Edit volume; use Advanced envelope for advanced shapes" } }
    ,{ "打开尾段淡出窗口：调整位置、起止响度和曲线，预览后应用或试听。", { "打开尾段淡出窗口：调整位置、起止响度和曲线，预览后应用或试听。", "開啟尾段淡出視窗：調整位置、起止響度及曲線，預覽後套用或試聽。", "語尾フェード画面で位置・開始終了音量・曲線を調整し、確認後に適用または試聴します。", "끝 페이드 창에서 위치, 시작/끝 음량 및 곡선을 조절하고 미리 본 후 적용하거나 들으세요.", "Open the tail fade window to adjust position, endpoint levels and curve; preview, apply or audition." } }
    ,{ "编辑当前音符的高级响度包络", { "编辑当前音符的高级响度包络", "編輯目前音符的進階響度包絡", "現在のノートの高度な音量エンベロープを編集", "현재 음표의 고급 음량 엔벌로프 편집", "Edit this note's advanced loudness envelope" } }
    ,{ "音符发音/别名", { "音符发音/别名", "音符發音／別名", "ノートの発音／エイリアス", "음표 발음/별칭", "Note pronunciation / alias" } }
    ,{ "线性 flag：", { "线性 flag：", "線性 FLAG：", "連続 FLAG：", "연속 FLAG: ", "Continuous FLAG: " } }
    ,{ "取消线性 flag：", { "取消线性 flag：", "取消線性 FLAG：", "連続 FLAG 解除：", "연속 FLAG 해제: ", "Disable continuous FLAG: " } }
    ,{ "DS 参数层", { "DS 参数层", "DS 參數層", "DS パラメーター層", "DS 매개변수 계층", "DS parameter layer" } }
    ,{ "实参曲线（预测值 / 直接编辑）", { "实参曲线（预测值 / 直接编辑）", "實參曲線（預測值／直接編輯）", "実パラメーター曲線（予測値／直接編集）", "실제 매개변수 곡선 (예측값/직접 편집)", "Actual parameter curves (predicted / direct editing)" } }
    ,{ "FLAG 偏移 / 其他控制", { "FLAG 偏移 / 其他控制", "FLAG 偏移／其他控制", "FLAG オフセット／その他の制御", "FLAG 오프셋/기타 제어", "FLAG offsets / other controls" } }
    ,{ "局部重算实参（选中音符；未选则整轨）", { "局部重算实参（选中音符；未选则整轨）", "局部重算實參（所選音符；未選則整軌）", "表現パラメーターを再計算（選択ノート／未選択時は全体）", "표현 매개변수 재계산 (선택 음표/미선택 시 전체 트랙)", "Recalculate parameters (selected notes, otherwise whole track)" } }
    ,{ "重新生成整轨实参", { "重新生成整轨实参", "重新產生整軌實參", "トラック全体の表現パラメーターを再生成", "트랙 전체 표현 매개변수 다시 생성", "Regenerate whole-track parameters" } }
    ,{ "清除实参，恢复自动预测（选中音符；未选则整轨）", { "清除实参，恢复自动预测（选中音符；未选则整轨）", "清除實參並還原自動預測（所選音符；未選則整軌）", "パラメーターを消去して自動予測に戻す（選択／全体）", "매개변수 삭제 후 자동 예측 복원 (선택/전체)", "Clear parameters and restore automatic prediction (selection / whole track)" } }
    ,{ "FLAG 编辑模式", { "FLAG 编辑模式", "FLAG 編輯模式", "FLAG 編集モード", "FLAG 편집 모드", "FLAG editing mode" } }
    ,{ "逐音符（控制点）", { "逐音符（控制点）", "逐音符（控制點）", "ノート単位（制御点）", "음표별 (제어점)", "Per note (control points)" } }
    ,{ "跨音符（连续绘制）", { "跨音符（连续绘制）", "跨音符（連續繪製）", "ノート間（連続描画）", "음표 간 (연속 그리기)", "Across notes (continuous drawing)" } }
    ,{ "连续绘制：当前轨道已开启 FLAG 的音符；Esc 取消本笔", { "连续绘制：当前轨道已开启 FLAG 的音符；Esc 取消本笔", "連續繪製：目前軌道已開啟 FLAG 的音符；Esc 取消本筆", "現在のトラックで FLAG 有効のノートに連続描画。Esc で取り消し", "현재 트랙의 FLAG 활성 음표에 연속 그리기; Esc로 획 취소", "Continuous drawing on FLAG-enabled notes of this track; Esc cancels the stroke" } }
    ,{ "直线", { "直线", "直線", "直線", "직선", "Linear" } }
    ,{ "曲线", { "曲线", "曲線", "曲線", "곡선", "Curve" } }
    ,{ " 小节", { " 小节", " 小節", " 小節", " 마디", " bars" } }
    ,{ "默认交叉方式", { "默认交叉方式", "預設交叉方式", "既定クロスフェード", "기본 크로스페이드 방식", "Default crossfade" } }
    ,{ "选择 UTAU / DiffSinger 音源库目录", { "选择 UTAU / DiffSinger 音源库目录", "選擇 UTAU / DiffSinger 音源庫目錄", "UTAU / DiffSinger 音源フォルダーを選択", "UTAU / DiffSinger 음원 폴더 선택", "Choose UTAU / DiffSinger voicebank folder" } }
    ,{ "UTAU 音源库：", { "UTAU 音源库：", "UTAU 音源庫：", "UTAU 音源：", "UTAU 음원: ", "UTAU voicebank: " } }
    ,{ "请先为当前轨道选择 UTAU 音源库。", { "请先为当前轨道选择 UTAU 音源库。", "請先為目前軌道選取 UTAU 音源庫。", "先に現在のトラックの UTAU 音源を選択してください。", "먼저 현재 트랙의 UTAU 음원을 선택하세요.", "Choose a UTAU voicebank for this track first." } }
    ,{ "音源库设置 — ", { "音源库设置 — ", "音源庫設定 — ", "音源設定 — ", "음원 설정 — ", "Voicebank settings — " } }
    ,{ "单独OTO编辑（只作用于此音符） — ", { "单独OTO编辑（只作用于此音符） — ", "單獨 OTO 編輯（僅此音符） — ", "個別 OTO 編集（このノートのみ） — ", "개별 OTO 편집 (이 음표만) — ", "Individual OTO editing (this note only) — " } }
    ,{ "请在素材管理器中选择素材文件夹并用于当前轨道。", { "请在素材管理器中选择素材文件夹并用于当前轨道。", "請在素材管理員選取資料夾並用於目前軌道。", "素材マネージャーでフォルダーを選び現在のトラックに使用してください。", "소재 관리자에서 폴더를 선택하여 현재 트랙에 사용하세요.", "Choose a material folder in the Asset Manager and use it for this track." } }
    ,{ "编辑器已关闭。", { "编辑器已关闭。", "編輯器已關閉。", "エディターは閉じられました。", "편집기가 닫혔습니다.", "The editor was closed." } }
    ,{ "高级包络 — ", { "高级包络 — ", "進階包絡 — ", "高度なエンベロープ — ", "고급 엔벌로프 — ", "Advanced envelope — " } }
    ,{ "请选择原生音频片段，或具有有效 OTO 区域的 UTAU 音符。", { "请选择原生音频片段，或具有有效 OTO 区域的 UTAU 音符。", "請選取原生音訊片段或具有有效 OTO 區域的 UTAU 音符。", "ネイティブ音声クリップか有効な OTO 区間を持つ UTAU ノートを選択してください。", "네이티브 오디오 클립 또는 유효한 OTO 구간의 UTAU 음표를 선택하세요.", "Select a native audio clip or a UTAU note with a valid OTO region." } }
    ,{ "导出当前轨道 UST…", { "导出当前轨道 UST…", "匯出目前軌道 UST…", "現在のトラックを UST 出力…", "현재 트랙 UST 내보내기…", "Export current track as UST…" } }
    ,{ "HAMOOD 自动和声…", { "HAMOOD 自动和声…", "HAMOOD 自動和聲…", "HAMOOD 自動ハーモニー…", "HAMOOD 자동 화음…", "HAMOOD automatic harmony…" } }
    ,{ "DS 重新生成整轨 pitch", { "DS 重新生成整轨 pitch", "DS 重新產生整軌 pitch", "DS トラック全体のピッチを再生成", "DS 트랙 전체 피치 다시 생성", "DS regenerate whole-track pitch" } }
    ,{ "DS 发音转换 / 读音覆盖…", { "DS 发音转换 / 读音覆盖…", "DS 發音轉換／讀音覆寫…", "DS 発音変換／読み上書き…", "DS 발음 변환/읽기 덮어쓰기…", "DS pronunciation conversion / reading overrides…" } }
    ,{ "DS 音素时长…", { "DS 音素时长…", "DS 音素時長…", "DS 音素の長さ…", "DS 음소 길이…", "DS phoneme timing…" } }
    ,{ "取消 DiffSinger 任务", { "取消 DiffSinger 任务", "取消 DiffSinger 工作", "DiffSinger タスクを中止", "DiffSinger 작업 취소", "Cancel DiffSinger task" } }
    ,{ "快速素材标注 — ", { "快速素材标注 — ", "快速素材標註 — ", "素材クイック注釈 — ", "빠른 소재 주석 — ", "Quick material annotation — " } }
    ,{ "神经音源没有可标注的源录音；请在素材管理器选择音频。", { "神经音源没有可标注的源录音；请在素材管理器选择音频。", "神經音源沒有可標註的來源錄音；請於素材管理員選取音訊。", "ニューラル音源には注釈用録音がありません。素材マネージャーで音声を選択してください。", "신경망 음원에는 주석용 원본 녹음이 없습니다. 소재 관리자에서 오디오를 선택하세요.", "Neural voicebanks have no source recording to annotate; choose audio in the Asset Manager." } }
    ,{ "请选择一个有明确源录音的音符。", { "请选择一个有明确源录音的音符。", "請選取有明確來源錄音的音符。", "ソース録音が明確なノートを選択してください。", "원본 녹음이 명확한 음표를 선택하세요.", "Select a note with an explicit source recording." } }
    ,{ "请选择要编辑素材的音符。", { "请选择要编辑素材的音符。", "請選取要編輯素材的音符。", "素材を編集するノートを選択してください。", "소재를 편집할 음표를 선택하세요.", "Select the note whose material you want to edit." } }
    ,{ "没有找到此发音的源录音。", { "没有找到此发音的源录音。", "找不到此發音的來源錄音。", "この発音のソース録音が見つかりません。", "이 발음의 원본 녹음을 찾지 못했습니다.", "Source recording for this pronunciation was not found." } }
    ,{ "请先选择需要标注的音频或音符。", { "请先选择需要标注的音频或音符。", "請先選取要標註的音訊或音符。", "注釈する音声またはノートを選択してください。", "먼저 주석을 달 오디오나 음표를 선택하세요.", "Select audio or a note to annotate first." } }
    ,{ "请选择一个音符轨道导出 UST。", { "请选择一个音符轨道导出 UST。", "請選取音符軌道以匯出 UST。", "UST 出力するノートトラックを選択してください。", "UST로 내보낼 음표 트랙을 선택하세요.", "Select a note track to export as UST." } }
    ,{ "导出当前轨道 UST", { "导出当前轨道 UST", "匯出目前軌道 UST", "現在のトラックを UST 出力", "현재 트랙 UST 내보내기", "Export current track as UST" } }
    ,{ "保持原编码（新轨道为 UTF-8）", { "保持原编码（新轨道为 UTF-8）", "保留原編碼（新軌道為 UTF-8）", "元の文字コードを保持（新規は UTF-8）", "원래 인코딩 유지 (새 트랙은 UTF-8)", "Keep original encoding (UTF-8 for new tracks)" } }
    ,{ "文件编码", { "文件编码", "檔案編碼", "文字コード", "파일 인코딩", "File encoding" } }
    ,{ "选择保存位置", { "选择保存位置", "選擇儲存位置", "保存先を選択", "저장 위치 선택", "Choose save location" } }
    ,{ "导出 UST", { "导出 UST", "匯出 UST", "UST を出力", "UST 내보내기", "Export UST" } }
    ,{ "已导出 UST：", { "已导出 UST：", "已匯出 UST：", "UST 出力済み：", "UST 내보냄: ", "UST exported: " } }
    ,{ "UST 导出说明", { "UST 导出说明", "UST 匯出說明", "UST 出力の注意事項", "UST 내보내기 안내", "UST export notes" } }
    ,{ "导入目标编辑界面", { "导入目标编辑界面", "匯入目標編輯介面", "読込先の編集画面", "가져올 대상 편집 화면", "Destination editing interface" } }
    ,{ "自动识别", { "自动识别", "自動辨識", "自動判定", "자동 감지", "Automatic detection" } }
    ,{ "UST 编码（USTX 固定 UTF-8）", { "UST 编码（USTX 固定 UTF-8）", "UST 編碼（USTX 固定 UTF-8）", "UST 文字コード（USTX は UTF-8 固定）", "UST 인코딩 (USTX는 UTF-8 고정)", "UST encoding (USTX is always UTF-8)" } }
    ,{ "导入", { "导入", "匯入", "読み込む", "가져오기", "Import" } }
    ,{ "响度包络", { "响度包络", "響度包絡", "音量エンベロープ", "음량 엔벌로프", "Loudness envelope" } }
    ,{ "响度包络（线性百分比）  纵向 0–", { "响度包络（线性百分比）  纵向 0–", "響度包絡（線性百分比） 縱向 0–", "音量エンベロープ（線形 %）縦軸 0–", "음량 엔벌로프 (선형 %) 세로 0–", "Loudness envelope (linear percent), vertical 0–" } }
    ,{ "0 静音", { "0 静音", "0 靜音", "0 無音", "0 무음", "0 silent" } }
    ,{ "响度 0（静音）", { "响度 0（静音）", "響度 0（靜音）", "音量 0（無音）", "음량 0 (무음)", "Loudness 0 (silent)" } }
    ,{ "原始 F0", { "原始 F0", "原始 F0", "元の F0", "원본 F0", "Original F0" } }
    ,{ "点击音高线", { "点击音高线", "點選音高線", "ピッチ線をクリック", "피치 선 클릭", "Click the pitch line" } }
    ,{ "频谱", { "频谱", "頻譜", "スペクトログラム", "스펙트로그램", "Spectrogram" } }
    ,{ "播放原音", { "播放原音", "播放原音", "原音を再生", "원음 재생", "Play original audio" } }
    ,{ "停止播放", { "停止播放", "停止播放", "再生を停止", "재생 중지", "Stop playback" } }
    ,{ "取消", { "取消", "取消", "キャンセル", "취소", "Cancel" } }
    ,{ "偏移", { "偏移", "偏移", "オフセット", "오프셋", "Offset" } }
    ,{ "辅音", { "辅音", "子音", "子音", "자음", "Consonant" } }
    ,{ "终止", { "终止", "終止", "終端", "끝", "Cutoff" } }
    ,{ "先行", { "先行", "先行", "先行発声", "선행 발성", "Preutterance" } }
    ,{ "重叠", { "重叠", "重疊", "オーバーラップ", "오버랩", "Overlap" } }
    ,{ "音素类别", { "音素类别", "音素類別", "音素の種類", "음소 유형", "Phoneme classes" } }
    ,{ "辅音区", { "辅音区", "子音區", "子音区間", "자음 구간", "Consonant regions" } }
    ,{ "保存谋•OTO（独立）", { "保存谋•OTO（独立）", "儲存謀•OTO（獨立）", "謀 OTO を別途保存", "Mou OTO 별도 저장", "Save Mou OTO (separate)" } }
    ,{ "保存界•OTO（独立）", { "保存界•OTO（独立）", "儲存界•OTO（獨立）", "界 OTO を別途保存", "Jie OTO 별도 저장", "Save Jie OTO (separate)" } }
    ,{ "保存到 oto.ini", { "保存到 oto.ini", "儲存至 oto.ini", "oto.ini に保存", "oto.ini에 저장", "Save to oto.ini" } }
    ,{ "横向放大（时间轴）", { "横向放大（时间轴）", "橫向放大（時間軸）", "横に拡大（時間軸）", "가로 확대 (시간축)", "Zoom in horizontally (time)" } }
    ,{ "横向缩小（时间轴）", { "横向缩小（时间轴）", "橫向縮小（時間軸）", "横に縮小（時間軸）", "가로 축소 (시간축)", "Zoom out horizontally (time)" } }
    ,{ "纵向放大（波形振幅）", { "纵向放大（波形振幅）", "縱向放大（波形振幅）", "縦に拡大（波形振幅）", "세로 확대 (파형 진폭)", "Zoom in vertically (amplitude)" } }
    ,{ "纵向缩小（波形振幅）", { "纵向缩小（波形振幅）", "縱向縮小（波形振幅）", "縦に縮小（波形振幅）", "세로 축소 (파형 진폭)", "Zoom out vertically (amplitude)" } }
    ,{ "回到本条目的取样范围", { "回到本条目的取样范围", "返回此條目的取樣範圍", "このエントリーの範囲に戻る", "이 항목의 샘플 범위로 돌아가기", "Fit this entry's sample range" } }
    ,{ "正在分析原始录音…", { "正在分析原始录音…", "正在分析原始錄音…", "元の録音を解析中…", "원본 녹음 분석 중…", "Analysing original recording…" } }
    ,{ "原始录音频谱", { "原始录音频谱", "原始錄音頻譜", "元の録音のスペクトログラム", "원본 녹음 스펙트로그램", "Original recording spectrogram" } }
    ,{ "原始 F0 · ", { "原始 F0 · ", "原始 F0 · ", "元の F0 · ", "원본 F0 · ", "Original F0 · " } }
    ,{ " · 未检测到有声音高", { " · 未检测到有声音高", " · 未偵測到有聲音高", " · 有声音のピッチなし", " · 유성 피치 감지 안 됨", " · No voiced pitch detected" } }
    ,{ " · 内置识别回退", { " · 内置识别回退", " · 內建辨識回退", " · 内蔵解析にフォールバック", " · 내장 분석으로 대체", " · Internal analysis fallback" } }
    ,{ "无法读取音频：", { "无法读取音频：", "無法讀取音訊：", "音声を読み込めません：", "오디오를 읽을 수 없음: ", "Cannot read audio: " } }
    ,{ "音频中没有可显示的波形", { "音频中没有可显示的波形", "音訊中沒有可顯示波形", "表示できる波形がありません", "표시할 파형이 없습니다", "No waveform to display" } }
    ,{ "拖动彩色边界或线条调整参数；数值单位均为 ms", { "拖动彩色边界或线条调整参数；数值单位均为 ms", "拖曳彩色邊界或線條調整參數；單位均為 ms", "色付き境界や線をドラッグして調整します。単位は ms です", "색 경계나 선을 드래그하여 조절합니다. 단위는 ms입니다", "Drag coloured boundaries or lines to adjust parameters; all values are in ms" } }
    ,{ "  ·  别名：", { "  ·  别名：", "  ·  別名：", "  ·  エイリアス：", "  ·  별칭: ", "  ·  Alias: " } }
    ,{ "第", { "第", "第", "#", "#", "#" } }
    ,{ "区", { "区", "區", "区間", "구간", " region" } }
    ,{ "区末", { "区末", "區末", "終端", "끝", " end" } }
    ,{ " 段分", { " 段分", " 段分", " 分割", "분할", " parts" } }
    ,{ "把这条 oto 切成 ", { "把这条 oto 切成 ", "將此 OTO 分割為 ", "この OTO を分割：", "이 OTO를 분할: ", "Split this OTO into " } }
    ,{ " 段；波形与下面的边界数值会立刻跟着变", { " 段；波形与下面的边界数值会立刻跟着变", " 段；波形與下方邊界值會即時更新", " 区間。波形と下の境界値が直ちに更新されます", "구간. 파형과 아래 경계 값이 즉시 갱신됩니다", " parts; waveform and boundary values update immediately" } }
    ,{ "显示原始录音的时频图，与波形共用时间轴", { "显示原始录音的时频图，与波形共用时间轴", "顯示原始錄音時頻圖，與波形共用時間軸", "波形と共通の時間軸で元録音の時間周波数を表示", "파형과 같은 시간축에서 원본 녹음의 시간-주파수 표시", "Show the original spectrogram on the waveform's time axis" } }
    ,{ "在波形和频谱上同时显示原始基频；点击音高线在左侧查看该帧 Hz，悬停查看音名。清音与静音处断开", { "在波形和频谱上同时显示原始基频；点击音高线在左侧查看该帧 Hz，悬停查看音名。清音与静音处断开", "在波形與頻譜上顯示原始基頻；點選查看 Hz，懸停查看音名。清音與靜音處中斷", "波形とスペクトルに元の基本周波数を表示。クリックで Hz、ホバーで音名。有声音以外は途切れます", "파형과 스펙트럼에 원본 기본 주파수 표시. 클릭하면 Hz, 가리키면 음이름 표시. 무성/무음은 끊깁니다", "Show original F0 over waveform and spectrum; click for Hz, hover for note names. Unvoiced and silent regions are gaps" } }
    ,{ "应用到此音符", { "应用到此音符", "套用至此音符", "このノートに適用", "이 음표에 적용", "Apply to this note" } }
    ,{ "只作用于这一个音符，不修改 oto 文件", { "只作用于这一个音符，不修改 oto 文件", "僅作用此音符，不修改 OTO 檔", "このノートのみに適用し OTO ファイルは変更しません", "이 음표에만 적용하며 OTO 파일은 변경하지 않습니다", "Affects this note only; OTO files stay unchanged" } }
    ,{ "DS 绘制偏移：拖动线整体平移，双击加点，拖动点调整，Ctrl 拖动自由绘制；右键可增加标点、输入半音或归零，Esc 取消/退出", { "DS 绘制偏移：拖动线整体平移，双击加点，拖动点调整，Ctrl 拖动自由绘制；右键可增加标点、输入半音或归零，Esc 取消/退出", "DS 繪製偏移：拖線平移，雙擊加點，拖點調整，Ctrl 拖曳自由繪製；右鍵加點、輸入半音或歸零，Esc 取消／離開", "DS オフセット：線をドラッグして移動、ダブルクリックで追加、Ctrl ドラッグで自由描画。右クリックで点追加・半音入力・リセット、Esc で取消／終了", "DS 오프셋: 선 드래그로 이동, 두 번 클릭으로 점 추가, Ctrl 드래그로 자유 그리기. 오른쪽 클릭으로 점 추가/반음 입력/초기화, Esc 취소/종료", "DS offset: drag the line to move, double-click to add points, Ctrl-drag to draw; right-click to add, enter semitones or reset; Esc cancels/exits" } }
    ,{ "把选中的音符连接为一条可渲染的原生边界；保留每个音符和素材数据，支持跨片段连接", { "把选中的音符连接为一条可渲染的原生边界；保留每个音符和素材数据，支持跨片段连接", "連接所選音符為可渲染的原生邊界，保留各音符與素材資料，支援跨片段", "選択ノートをレンダー可能な境界で接続します。ノートと素材を保持しクリップ間接続に対応", "선택 음표를 렌더링 가능한 경계로 연결합니다. 음표와 소재를 유지하며 클립 간 연결을 지원합니다", "Connect selected notes through a renderable native boundary; preserve notes/materials and allow cross-clip connections" } }
    ,{ "DS 使用音素时长编辑（Ctrl+G）；速度参数请使用 VELC 曲线", { "DS 使用音素时长编辑（Ctrl+G）；速度参数请使用 VELC 曲线", "DS 使用音素時長編輯（Ctrl+G）；速度請用 VELC 曲線", "DS は音素長編集（Ctrl+G）を使用。速度は VELC 曲線で調整", "DS는 음소 길이 편집 (Ctrl+G)을 사용합니다. 속도는 VELC 곡선 사용", "DS uses phoneme timing (Ctrl+G); use VELC curves for velocity" } }
    ,{ "点击 FLAG 编辑实参，在曲线通道切换实参 / 偏移", { "点击 FLAG 编辑实参，在曲线通道切换实参 / 偏移", "點選 FLAG 編輯實參，於曲線通道切換實參／偏移", "FLAG をクリックして実パラメーターを編集。レーンで実値／オフセットを切替", "FLAG를 클릭하여 실제 매개변수 편집. 곡선 레인에서 실제 값/오프셋 전환", "Click FLAG to edit actual parameters; switch actual/offset layers in the curve lane" } }
    ,{ "已断开所选音符的滑音连接；素材与包络保持不变。", { "已断开所选音符的滑音连接；素材与包络保持不变。", "已斷開所選音符的滑音連接；素材與包絡不變。", "選択ノートのポルタメントを解除しました。素材とエンベロープは保持します。", "선택 음표의 포르타멘토 연결을 해제했습니다. 소재와 엔벌로프는 유지됩니다.", "Selected portamento disconnected; material and envelopes unchanged." } }
    ,{ "已建立滑音连接；可编辑连接音高曲线，素材与包络保持不变。", { "已建立滑音连接；可编辑连接音高曲线，素材与包络保持不变。", "已建立滑音連接；可編輯連接音高曲線，素材與包絡不變。", "ポルタメントを接続しました。接続ピッチ曲線を編集できます。素材とエンベロープは保持します。", "포르타멘토를 연결했습니다. 연결 피치 곡선을 편집할 수 있으며 소재와 엔벌로프는 유지됩니다.", "Portamento connected; edit its pitch curve without changing material or envelopes." } }
    ,{ "工程已发生其他修改。请重新打开窗口更新音素范围。", { "工程已发生其他修改。请重新打开窗口更新音素范围。", "工程已有其他變更。請重開視窗以更新音素範圍。", "プロジェクトが変更されました。画面を開き直して音素範囲を更新してください。", "프로젝트가 변경되었습니다. 창을 다시 열어 음소 범위를 갱신하세요.", "Project changed. Reopen the window to update phoneme ranges." } }
    ,{ "已应用到音素 ", { "已应用到音素 ", "已套用至音素 ", "適用した音素：", "적용한 음소: ", "Applied to phoneme " } }
    ,{ "；支持撤销。", { "；支持撤销。", "；支援復原。", "。元に戻せます。", ". 실행 취소할 수 있습니다.", "; undo is available." } }
    ,{ "工程已发生其他修改。请关闭并重新打开此窗口，以更新预览和选区。", { "工程已发生其他修改。请关闭并重新打开此窗口，以更新预览和选区。", "工程已有其他變更。請關閉並重開視窗以更新預覽與選區。", "プロジェクトが変更されました。画面を開き直してプレビューと選択を更新してください。", "프로젝트가 변경되었습니다. 창을 닫고 다시 열어 미리 보기와 선택을 갱신하세요.", "Project changed. Close and reopen this window to update preview and selection." } }
    ,{ "已应用到 ", { "已应用到 ", "已套用至 ", "適用：", "적용: ", "Applied to " } }
    ,{ " 个音符；支持撤销。", { " 个音符；支持撤销。", " 個音符；支援復原。", " ノート。元に戻せます。", "개 음표. 실행 취소할 수 있습니다.", " notes; undo is available." } }
    ,{ "HiFisampler（PC-NSF-HiFiGAN）模型未找到，请在设置中指定模型目录。", { "HiFisampler（PC-NSF-HiFiGAN）模型未找到，请在设置中指定模型目录。", "找不到 HiFisampler（PC-NSF-HiFiGAN）模型，請於設定指定模型目錄。", "HiFisampler（PC-NSF-HiFiGAN）モデルがありません。設定でモデルフォルダーを指定してください。", "HiFisampler (PC-NSF-HiFiGAN) 모델이 없습니다. 설정에서 모델 폴더를 지정하세요.", "HiFisampler (PC-NSF-HiFiGAN) model missing. Choose its directory in Settings." } }
    ,{ "显示全部 UTAU 音符的振幅包络；拖动标点编辑，双击曲线加点，右键内部标点删除", { "显示全部 UTAU 音符的振幅包络；拖动标点编辑，双击曲线加点，右键内部标点删除", "顯示全部 UTAU 音符振幅包絡；拖點編輯、雙擊加點、右鍵刪除內部點", "全 UTAU ノートの振幅エンベロープを表示。ドラッグ編集、ダブルクリック追加、内点を右クリックで削除", "전체 UTAU 음표 진폭 엔벌로프 표시. 점 드래그 편집, 두 번 클릭 추가, 내부 점 오른쪽 클릭 삭제", "Show all UTAU amplitude envelopes; drag points, double-click to add, right-click interior points to delete" } }
    ,{ "选中 ", { "选中 ", "選取 ", "選択 ", "선택 ", "Selected " } }
    ,{ " 音符 · ", { " 音符 · ", " 音符 · ", " ノート · ", " 음표 · ", " notes · " } }
    ,{ " 区域", { " 区域", " 區域", " 区間", " 구간", " regions" } }
    ,{ "UTAU 当前选区没有生成可播放音频", { "UTAU 当前选区没有生成可播放音频", "UTAU 目前選區未產生可播放音訊", "現在の UTAU 選択範囲に再生可能な音声がありません", "현재 UTAU 선택 범위에 재생 가능한 오디오가 없습니다", "No playable audio generated for the current UTAU selection" } }
    ,{ "这里放不下：录音素材到此为止，音符只能落在它的范围内。", { "这里放不下：录音素材到此为止，音符只能落在它的范围内。", "此處放不下：錄音素材到此為止，音符須位於其範圍內。", "ここには配置できません。ノートは録音素材の範囲内に収めてください。", "여기에 놓을 수 없습니다. 음표는 녹음 소재 범위 안에 있어야 합니다.", "Cannot place here: notes must remain within the source recording." } }
    ,{ "原生音频片段需粘贴到非 UTAU 旋律轨道，且原素材文件须可读取。", { "原生音频片段需粘贴到非 UTAU 旋律轨道，且原素材文件须可读取。", "原生音訊須貼至非 UTAU 旋律軌道，且原素材檔須可讀。", "ネイティブ音声は UTAU 以外の旋律トラックに貼り付け、元素材が読める必要があります。", "네이티브 오디오는 UTAU 외 멜로디 트랙에 붙여넣어야 하며 원본 파일을 읽을 수 있어야 합니다.", "Paste native audio onto a non-UTAU melodic track with readable source files." } }
    ,{ "这条轨道放不下音符：请选择一条旋律轨道。", { "这条轨道放不下音符：请选择一条旋律轨道。", "此軌道不能放音符：請選取旋律軌道。", "このトラックにノートを置けません。旋律トラックを選択してください。", "이 트랙에는 음표를 놓을 수 없습니다. 멜로디 트랙을 선택하세요.", "Cannot place notes on this track; choose a melodic track." } }
    ,{ "粘贴音符", { "粘贴音符", "貼上音符", "ノートを貼り付け", "음표 붙여넣기", "Paste notes" } }
    ,{ "这一段和轨道上已有的 ", { "这一段和轨道上已有的 ", "此區段與軌道已有的 ", "この範囲と既存の ", "이 범위는 기존 ", "This range overlaps " } }
    ,{ " 个音符重叠。\n\n继续将删除这些音符并粘贴覆盖；其余音符的位置不受影响。", { " 个音符重叠。\n\n继续将删除这些音符并粘贴覆盖；其余音符的位置不受影响。", " 個音符重疊。\n\n繼續會刪除並覆貼這些音符，其他音符位置不變。", " ノートが重なっています。\n\n続行するとこれらを削除して上書きします。他のノート位置は変わりません。", "개 음표와 겹칩니다.\n\n계속하면 해당 음표를 삭제하고 덮어씁니다. 다른 음표 위치는 유지됩니다.", " existing notes.\n\nContinue to delete and overwrite them; other note positions stay unchanged." } }
    ,{ "覆盖粘贴", { "覆盖粘贴", "覆蓋貼上", "上書きして貼付", "덮어써 붙여넣기", "Paste and overwrite" } }
    ,{ "；旧工程的虚线参考来自载入时音高，随工程保存", { "；旧工程的虚线参考来自载入时音高，随工程保存", "；舊工程虛線參考取自載入時音高，隨工程儲存", "；旧プロジェクトの点線参照は読込時のピッチで保存されます", "; 이전 프로젝트 점선 참조는 불러올 때의 피치이며 프로젝트와 함께 저장됩니다", "; old-project dotted reference uses loaded pitch and is saved with the project" } }
    ,{ "；DS 缓存未能迁移，工程已保存", { "；DS 缓存未能迁移，工程已保存", "；DS 快取未能遷移，工程已儲存", "；DS キャッシュを移行できませんでしたが保存は完了しました", "; DS 캐시를 이전하지 못했으나 프로젝트는 저장되었습니다", "; DS cache migration failed, but the project was saved" } }
    ,{ "\n保持原编码时，未修改的导入字段按原文写回。", { "\n保持原编码时，未修改的导入字段按原文写回。", "\n保留原編碼時，未修改的匯入欄位按原文寫回。", "\n元の文字コードを保持すると、未編集フィールドは原文のまま保存します。", "\n원래 인코딩 유지 시 편집하지 않은 필드는 원문으로 저장합니다.", "\nKeeping original encoding preserves unchanged imported fields verbatim." } }
    ,{ "选择 UST 读取编码；乱码时可指定原文件编码后重新导入。", { "选择 UST 读取编码；乱码时可指定原文件编码后重新导入。", "選取 UST 讀取編碼；亂碼時可指定原檔編碼重匯。", "UST の文字コードを選択してください。文字化けする場合は元の文字コードを指定して再読込します。", "UST 인코딩을 선택하세요. 문자가 깨지면 원본 인코딩으로 다시 가져오세요.", "Choose UST encoding; if text is garbled, select the original encoding and reimport." } }
    ,{ "  原始 F0：", { "  原始 F0：", "  原始 F0：", "  元の F0：", "  원본 F0: ", "  Original F0: " } }
    ,{ "  原始 F0：无有声音高", { "  原始 F0：无有声音高", "  原始 F0：無有聲音高", "  元の F0：有声音なし", "  원본 F0: 유성 피치 없음", "  Original F0: no voiced pitch" } }
    ,{ "每区一个字母：C 辅音 / V 元音 / S 静音。串长就是区数（2~4），字母说明哪一段是元音。", { "每区一个字母：C 辅音 / V 元音 / S 静音。串长就是区数（2~4），字母说明哪一段是元音。", "每區一字母：C子音／V母音／S靜音。長度即區數（2~4），字母標示母音。", "各区間を C子音／V母音／S無音の一文字で指定します。文字数が区間数（2～4）です。", "구간마다 C자음/V모음/S무음 한 글자를 지정합니다. 문자열 길이가 구간 수 (2~4)입니다.", "One letter per region: C consonant / V vowel / S silence. Length is region count (2–4); letters identify vowel regions." } }
    ,{ "勾选哪几个区是辅音，可以一个不勾，也可以勾多个。勾上的区不参与主动拉伸，并且走辅音那套合成。", { "勾选哪几个区是辅音，可以一个不勾，也可以勾多个。勾上的区不参与主动拉伸，并且走辅音那套合成。", "勾選子音區，可不選或多選。已勾區不參與主動伸縮，使用子音合成。", "子音区間を選択します。未選択・複数選択も可能です。選択区間は伸縮せず子音として合成します。", "자음 구간을 선택합니다. 미선택 또는 다중 선택 가능합니다. 선택 구간은 늘이지 않고 자음으로 합성합니다.", "Mark any consonant regions, including none or several. Marked regions are protected from active stretching and use consonant synthesis." } }
    ,{ "从头播放这条 oto 所在的整个 wav 原始录音，没经过任何引擎；再点一次停止", { "从头播放这条 oto 所在的整个 wav 原始录音，没经过任何引擎；再点一次停止", "從頭播放此 OTO 所在完整 WAV 原音，不經引擎；再點停止", "エンジンを通さず元 WAV 全体を先頭から再生。再クリックで停止", "엔진을 거치지 않은 원본 WAV 전체를 처음부터 재생. 다시 클릭하면 중지", "Play the entire original WAV from the start without an engine; click again to stop" } }
    ,{ "无法保存原生标注", { "无法保存原生标注", "無法儲存原生標註", "ネイティブ注釈を保存できません", "네이티브 주석을 저장할 수 없음", "Cannot save native annotations" } }
    ,{ "无法保存独立界•OTO", { "无法保存独立界•OTO", "無法儲存獨立界 OTO", "個別の界 OTO を保存できません", "별도 Jie OTO를 저장할 수 없음", "Cannot save separate Jie OTO" } }
    ,{ "无法保存 oto.ini", { "无法保存 oto.ini", "無法儲存 oto.ini", "oto.ini を保存できません", "oto.ini를 저장할 수 없음", "Cannot save oto.ini" } }
    ,{ "无法保存谋•OTO（otomou.ini）", { "无法保存谋•OTO（otomou.ini）", "無法儲存謀 OTO（otomou.ini）", "謀 OTO（otomou.ini）を保存できません", "Mou OTO (otomou.ini)를 저장할 수 없음", "Cannot save Mou OTO (otomou.ini)" } }
    ,{ "无法保存界•OTO（oto4.ini）", { "无法保存界•OTO（oto4.ini）", "無法儲存界 OTO（oto4.ini）", "界 OTO（oto4.ini）を保存できません", "Jie OTO (oto4.ini)를 저장할 수 없음", "Cannot save Jie OTO (oto4.ini)" } }
    ,{ "单独OTO编辑仅用于 UTAU 轨道。", { "单独OTO编辑仅用于 UTAU 轨道。", "單獨 OTO 編輯僅限 UTAU 軌道。", "個別 OTO 編集は UTAU トラック専用です。", "개별 OTO 편집은 UTAU 트랙 전용입니다.", "Individual OTO editing is only available for UTAU tracks." } }
    ,{ "音源库里没有这个歌词的 oto 条目。", { "音源库里没有这个歌词的 oto 条目。", "音源庫沒有此歌詞的 OTO 條目。", "この歌詞の OTO エントリーが音源にありません。", "음원에 이 가사의 OTO 항목이 없습니다.", "No OTO entry for this lyric in the voicebank." } }
    ,{ "单独 OTO：只作用于这一个音符，不修改 oto 文件；数值单位均为 ms", { "单独 OTO：只作用于这一个音符，不修改 oto 文件；数值单位均为 ms", "單獨 OTO：僅此音符，不修改 OTO 檔；單位均為 ms", "個別 OTO：このノートのみ。OTO ファイルは変更せず、単位は ms です", "개별 OTO: 이 음표에만 적용. OTO 파일은 변경하지 않으며 단위는 ms입니다", "Individual OTO: this note only; OTO files unchanged; all values in ms" } }
    ,{ "无法播放原音", { "无法播放原音", "無法播放原音", "原音を再生できません", "원음을 재생할 수 없음", "Cannot play original audio" } }
    ,{ "无法读取音频文件：", { "无法读取音频文件：", "無法讀取音訊檔：", "音声ファイルを読込めません：", "오디오 파일을 읽을 수 없음: ", "Cannot read audio file: " } }
    ,{ "选择预设，或在此输入名称后保存", { "选择预设，或在此输入名称后保存", "選擇預設，或輸入名稱後儲存", "プリセットを選ぶか、名前を入力して保存", "프리셋을 선택하거나 이름을 입력하여 저장", "Choose a preset, or enter a name to save" } }
    ,{ "存为预设", { "存为预设", "存為預設", "プリセットとして保存", "프리셋으로 저장", "Save preset" } }
    ,{ "删除选中的自定义预设", { "删除选中的自定义预设", "刪除所選自訂預設", "選択したカスタムプリセットを削除", "선택한 사용자 프리셋 삭제", "Delete selected custom preset" } }
    ,{ "柔起", { "柔起", "柔起", "ソフトアタック", "부드러운 시작", "Soft attack" } }
    ,{ "缓起", { "缓起", "緩起", "スローアタック", "느린 시작", "Slow attack" } }
    ,{ "渐弱", { "渐弱", "漸弱", "ディミヌエンド", "점점 여리게", "Diminuendo" } }
    ,{ "句尾", { "句尾", "句尾", "フレーズ終端", "프레이즈 끝", "Phrase ending" } }
    ,{ "短收", { "短收", "短收", "短いリリース", "짧은 끝", "Short release" } }
    ,{ "起音 5 ms，释放 35 ms（UTAU 默认）", { "起音 5 ms，释放 35 ms（UTAU 默认）", "起音 5 ms，釋放 35 ms（UTAU 預設）", "アタック 5 ms、リリース 35 ms（UTAU 既定）", "시작 5 ms, 릴리스 35 ms (UTAU 기본)", "Attack 5 ms, release 35 ms (UTAU default)" } }
    ,{ "起音 15 ms，释放 35 ms", { "起音 15 ms，释放 35 ms", "起音 15 ms，釋放 35 ms", "アタック 15 ms、リリース 35 ms", "시작 15 ms, 릴리스 35 ms", "Attack 15 ms, release 35 ms" } }
    ,{ "辅音以 60% 进入，拍点后 50 ms 升到满音量；释放 35 ms（按音量线性）", { "辅音以 60% 进入，拍点后 50 ms 升到满音量；释放 35 ms（按音量线性）", "子音以 60% 進入，拍點後 50 ms 升至全音量；釋放 35 ms（音量線性）", "子音を 60% で開始、拍から 50 ms で最大音量。リリース 35 ms（振幅線形）", "자음 60% 시작, 박자 후 50 ms에 최대 음량. 릴리스 35 ms (선형 진폭)", "Consonant starts at 60%, reaches full level 50 ms after the beat; 35 ms linear-amplitude release" } }
    ,{ "起音 5 ms，平台末尾降到 80%，释放 35 ms（按音量线性）", { "起音 5 ms，平台末尾降到 80%，释放 35 ms（按音量线性）", "起音 5 ms，平台末降至 80%，釋放 35 ms（音量線性）", "アタック 5 ms、維持終端を 80% に低下、リリース 35 ms（振幅線形）", "시작 5 ms, 유지 끝 80%, 릴리스 35 ms (선형 진폭)", "Attack 5 ms, sustain ends at 80%, release 35 ms (linear amplitude)" } }
    ,{ "起音 5 ms；最后 150 ms 先降到 50%，再用 60 ms 淡出（按音量线性）", { "起音 5 ms；最后 150 ms 先降到 50%，再用 60 ms 淡出（按音量线性）", "起音 5 ms；末尾 150 ms 先降至 50%，再以 60 ms 淡出（音量線性）", "アタック 5 ms。最後の 150 ms で 50% に下げ、60 ms でフェード（振幅線形）", "시작 5 ms. 마지막 150 ms에 50%로 낮추고 60 ms로 페이드 (선형 진폭)", "Attack 5 ms; final 150 ms drops to 50%, then fades over 60 ms (linear amplitude)" } }
    ,{ "起音 5 ms，释放 10 ms（按音量线性，不爆音）", { "起音 5 ms，释放 10 ms（按音量线性，不爆音）", "起音 5 ms，釋放 10 ms（音量線性，防爆音）", "アタック 5 ms、リリース 10 ms（振幅線形、クリック防止）", "시작 5 ms, 릴리스 10 ms (선형 진폭, 클릭 방지)", "Attack 5 ms, release 10 ms (linear amplitude, no clicks)" } }
    ,{ "FLAG 实参（直接值）", { "FLAG 实参（直接值）", "FLAG 實參（直接值）", "FLAG 実パラメーター（直接値）", "FLAG 실제 매개변수 (직접 값)", "FLAG actual parameters (direct values)" } }
    ,{ "正在读取音源能力；音源库设置可重新读取", { "正在读取音源能力；音源库设置可重新读取", "正在讀取音源能力；音源設定可重新讀取", "音源機能を読込中。音源設定から再読込できます", "음원 기능 읽는 중. 음원 설정에서 다시 읽을 수 있습니다", "Reading voicebank capabilities; reload from voicebank settings" } }
    ,{ "当前显示轨道：清除不支持的已有曲线（可撤销）", { "当前显示轨道：清除不支持的已有曲线（可撤销）", "目前軌道：清除不支援的已有曲線（可復原）", "表示トラック：未対応の既存曲線を消去（取消可能）", "표시 트랙: 지원하지 않는 기존 곡선 삭제 (실행 취소 가능)", "Displayed track: clear unsupported existing curves (undoable)" } }
    ,{ "清除 ", { "清除 ", "清除 ", "消去 ", "지우기 ", "Clear " } }
    ,{ "删除标点", { "删除标点", "刪除控制點", "ポイントを削除", "제어점 삭제", "Delete point" } }
    ,{ "恢复此点的原始预测", { "恢复此点的原始预测", "還原此點的原始預測", "この点の元の予測を復元", "이 점의 원래 예측 복원", "Restore this point's original prediction" } }
    ,{ "重置为默认值（", { "重置为默认值（", "重設為預設值（", "既定値にリセット（", "기본값으로 초기화 (", "Reset to default (" } }
    ,{ "输入 flag 值…", { "输入 flag 值…", "輸入 FLAG 值…", "FLAG 値を入力…", "FLAG 값 입력…", "Enter FLAG value…" } }
    ,{ "重置该音的 ", { "重置该音的 ", "重設此音的 ", "このノートをリセット：", "이 음표 초기화: ", "Reset this note's " } }
    ,{ "线条形状（首个标点没有前段）", { "线条形状（首个标点没有前段）", "線條形狀（首點無前段）", "線形状（先頭点は前区間なし）", "선 모양 (첫 점은 이전 구간 없음)", "Line shape (first point has no preceding segment)" } }
    ,{ "缓入", { "缓入", "緩入", "イーズイン", "서서히 시작", "Ease in" } }
    ,{ "缓出", { "缓出", "緩出", "イーズアウト", "서서히 끝", "Ease out" } }
    ,{ "线条形状", { "线条形状", "線條形狀", "線の形状", "선 모양", "Line shape" } }
    ,{ "，范围 ", { "，范围 ", "，範圍 ", "、範囲 ", ", 범위 ", ", range " } }
    ,{ "当前：第 ", { "当前：第 ", "目前：第 ", "現在：ポイント ", "현재: 점 ", "Current: point " } }
    ,{ " 个标点（共 ", { " 个标点（共 ", " 個控制點（共 ", "（合計 ", " (총 ", " (of " } }
    ,{ " 个），位于音符内 ", { " 个），位于音符内 ", " 個），位於音符內 ", "）、ノート内時刻 ", "개), 음표 내 위치 ", "), within note at " } }
    ,{ " 秒，现值 ", { " 秒，现值 ", " 秒，目前值 ", " 秒、現在値 ", "초, 현재 값 ", " s, value " } }
    ,{ "标点数值", { "标点数值", "控制點數值", "ポイントの値", "제어점 값", "Point value" } }
    ,{ " 值：", { " 值：", " 值：", " 値：", " 값: ", " value: " } }
    ,{ "归一化控制点。X 是段内的时间位置（0–1），Y 是这一段走完的比例；Y 超出 0–1 会先冲过头再回来。", { "归一化控制点。X 是段内的时间位置（0–1），Y 是这一段走完的比例；Y 超出 0–1 会先冲过头再回来。", "正規化控制點：X 是段內時間（0–1），Y 是完成比例；Y 超出 0–1 會超越後返回。", "正規化制御点。X は区間内時刻（0～1）、Y は進行率です。Y が 0～1 を超えるとオーバーシュートします。", "정규화 제어점. X는 구간 내 시간 (0–1), Y는 진행 비율입니다. Y가 0–1을 벗어나면 초과 후 돌아옵니다.", "Normalized controls: X is segment time (0–1), Y is progress; Y outside 0–1 overshoots and returns." } }
    ,{ "）  纵向 ", { "）  纵向 ", "）  縱向 ", "） 縦軸 ", ") 세로 ", ") vertical " } }
    ,{ "   仅声母区", { "   仅声母区", "   僅聲母區", "   子音区間のみ", "   자음 구간만", "   Onset region only" } }
    ,{ "实参", { "实参", "實參", "実パラメーター", "실제 매개변수", "Actual parameters" } }
    ,{ "音高调整", { "音高调整", "音高調整", "ピッチ調整", "피치 조정", "Pitch adjustment" } }
    ,{ "上移（半音）：", { "上移（半音）：", "上移（半音）：", "上げる（半音）：", "올리기 (반음): ", "Raise (semitones): " } }
    ,{ "下移（半音）：", { "下移（半音）：", "下移（半音）：", "下げる（半音）：", "내리기 (반음): ", "Lower (semitones): " } }
    ,{ "分割音符", { "分割音符", "分割音符", "ノートを分割", "음표 분할", "Split note" } }
    ,{ "合并音符", { "合并音符", "合併音符", "ノートを結合", "음표 병합", "Merge notes" } }
    ,{ "强制连接（保留音符数据）", { "强制连接（保留音符数据）", "強制連接（保留音符資料）", "強制接続（ノート情報を保持）", "강제 연결 (음표 데이터 보존)", "Force connection (preserve note data)" } }
    ,{ "添加拼字音符", { "添加拼字音符", "新增拼字音符", "綴りノートを追加", "발음 연결 음표 추가", "Add spelling note" } }
    ,{ "时序…", { "时序…", "時序…", "タイミング…", "타이밍…", "Timing…" } }
    ,{ "DS 局部重生成 pitch", { "DS 局部重生成 pitch", "DS 局部重新產生 pitch", "DS 選択範囲のピッチを再生成", "DS 선택 범위 피치 다시 생성", "DS regenerate selected pitch" } }
    ,{ "DS 局部重新生成 FLAG（预测实参）", { "DS 局部重新生成 FLAG（预测实参）", "DS 局部重新產生 FLAG（預測實參）", "DS 選択範囲の FLAG（予測値）を再生成", "DS 선택 범위 FLAG (예측 매개변수) 다시 생성", "DS regenerate selected FLAG (predicted parameters)" } }
    ,{ "修改 STP…", { "修改 STP…", "修改 STP…", "STP を変更…", "STP 수정…", "Edit STP…" } }
    ,{ "单独OTO编辑…", { "单独OTO编辑…", "單獨 OTO 編輯…", "個別 OTO 編集…", "개별 OTO 편집…", "Individual OTO editing…" } }
    ,{ "恢复为音源OTO", { "恢复为音源OTO", "還原音源 OTO", "音源 OTO に復元", "음원 OTO로 복원", "Restore voicebank OTO" } }
    ,{ "flag 拆分…", { "flag 拆分…", "FLAG 分割…", "FLAG 分割…", "FLAG 분할…", "FLAG split…" } }
    ,{ "颤音…", { "颤音…", "顫音…", "ビブラート…", "비브라토…", "Vibrato…" } }
    ,{ "退回为标准颤音模式", { "退回为标准颤音模式", "返回標準顫音模式", "標準ビブラートモードに戻る", "표준 비브라토 모드로 복귀", "Return to standard vibrato mode" } }
    ,{ "改为真实颤音线", { "改为真实颤音线", "改為真實顫音線", "実ビブラート線に変換", "실제 비브라토 선으로 변경", "Use actual vibrato line" } }
    ,{ "修改颤音为音高标点…", { "修改颤音为音高标点…", "將顫音轉為音高控制點…", "ビブラートをピッチポイントへ変換…", "비브라토를 피치 제어점으로 변환…", "Convert vibrato to pitch points…" } }
    ,{ "重置线性flag", { "重置线性flag", "重設線性 FLAG", "連続 FLAG をリセット", "연속 FLAG 초기화", "Reset continuous FLAG" } }
    ,{ "包络基础值…", { "包络基础值…", "包絡基礎值…", "エンベロープ基準値…", "엔벌로프 기준값…", "Envelope base…" } }
    ,{ "时序删除", { "时序删除", "時序刪除", "時間を詰めて削除", "시간 삭제", "Ripple delete" } }
    ,{ "添加空格", { "添加空格", "新增空白", "空白を挿入", "공백 추가", "Insert gap" } }
    ,{ "添加指定长度空格…", { "添加指定长度空格…", "新增指定長度空白…", "長さ指定で空白を挿入…", "길이 지정 공백 추가…", "Insert gap with length…" } }
    ,{ "批量输入歌词…", { "批量输入歌词…", "批次輸入歌詞…", "歌詞を一括入力…", "가사 일괄 입력…", "Enter lyrics in bulk…" } }
    ,{ "初始化音高线", { "初始化音高线", "初始化音高線", "ピッチ線を初期化", "피치 선 초기화", "Initialize pitch line" } }
    ,{ "音高调整…", { "音高调整…", "音高調整…", "ピッチ調整…", "피치 조정…", "Pitch adjustment…" } }
    ,{ "删除音符  Delete", { "删除音符  Delete", "刪除音符  Delete", "ノートを削除  Delete", "음표 삭제  Delete", "Delete notes  Delete" } }
    ,{ "删除空格", { "删除空格", "刪除空白", "空白を削除", "공백 삭제", "Delete gap" } }
    ,{ "插入四分之一小节音符", { "插入四分之一小节音符", "插入四分之一小節音符", "四分の一小節ノートを挿入", "4분의 1마디 음표 삽입", "Insert quarter-bar note" } }
    ,{ "批量输入歌词", { "批量输入歌词", "批次輸入歌詞", "歌詞を一括入力", "가사 일괄 입력", "Enter lyrics in bulk" } }
    ,{ "填入", { "填入", "填入", "入力", "입력", "Fill" } }
    ,{ "添加指定长度空格", { "添加指定长度空格", "新增指定長度空白", "長さ指定で空白を挿入", "길이 지정 공백 추가", "Insert gap with length" } }
    ,{ "在该音符前插入一段空白，该轨道之后的音符整体后移。\n单位：1 = 四分音符的 1/128；四分之一小节 = ", { "在该音符前插入一段空白，该轨道之后的音符整体后移。\n单位：1 = 四分音符的 1/128；四分之一小节 = ", "在此音符前插入空白，後續音符整體後移。\n單位：1 = 四分音符的 1/128；四分之一小節 = ", "このノートの前に空白を挿入し、後続ノートを移動します。\n単位：1 = 四分音符の 1/128、四分の一小節 = ", "이 음표 앞에 공백을 삽입하고 뒤 음표를 이동합니다.\n단위: 1 = 4분음표의 1/128; 4분의 1마디 = ", "Insert a gap before this note and move later notes.\nUnit: 1 = 1/128 of a quarter note; quarter bar = " } }
    ,{ "长度（单位）", { "长度（单位）", "長度（單位）", "長さ（単位）", "길이 (단위)", "Length (units)" } }
    ,{ "添加", { "添加", "新增", "追加", "추가", "Add" } }
    ,{ "修改 STP", { "修改 STP", "修改 STP", "STP を変更", "STP 수정", "Edit STP" } }
    ,{ "包络基础值", { "包络基础值", "包絡基礎值", "エンベロープ基準値", "엔벌로프 기준값", "Envelope base" } }
    ,{ "基础值，% （0–200）：", { "基础值，% （0–200）：", "基礎值，%（0–200）：", "基準値 %（0～200）：", "기준값 % (0–200): ", "Base value, % (0–200): " } }
    ,{ "对选中的 ", { "对选中的 ", "對所選的 ", "選択した ", "선택한 ", "For selected " } }
    ,{ " 个音频片段添加颤音，叠加在各自原有的音高线上。长度从终止位置往前算；不会修改原始音高参考。", { " 个音频片段添加颤音，叠加在各自原有的音高线上。长度从终止位置往前算；不会修改原始音高参考。", " 個音訊片段新增顫音並疊加原音高線。長度自終點往前算，不修改原始音高參考。", " 音声クリップの既存ピッチにビブラートを重ねます。長さは終端から逆算し、元の参照は保持します。", "개 오디오 클립의 기존 피치에 비브라토를 더합니다. 길이는 끝부터 역산하며 원본 피치 참조는 유지합니다.", " audio clips, overlay vibrato on existing pitch. Length counts back from the end; original pitch reference is unchanged." } }
    ,{ "以该音的音高线为基准上下颤动；长度从颤音终止处往前算。", { "以该音的音高线为基准上下颤动；长度从颤音终止处往前算。", "以此音的音高線為基準顫動；長度自顫音終點往前算。", "このノートのピッチを中心に振動します。長さはビブラート終端から逆算します。", "이 음표 피치를 중심으로 진동합니다. 길이는 비브라토 끝부터 역산합니다.", "Oscillate around this note's pitch; length counts backward from the vibrato endpoint." } }
    ,{ "终止位置 (% 于音符，100 = 末尾)：", { "终止位置 (% 于音符，100 = 末尾)：", "終止位置（音符 %，100 = 末尾）：", "終了位置（ノート %、100 = 終端）：", "끝 위치 (음표 %, 100 = 끝): ", "End position (% of note, 100 = end): " } }
    ,{ "在音高线上显示实际颤音", { "在音高线上显示实际颤音", "在音高線上顯示實際顫音", "ピッチ線に実際のビブラートを表示", "피치 선에 실제 비브라토 표시", "Show actual vibrato on the pitch line" } }
    ,{ "普通编辑时显示叠加后的音高；标点模式保留基础曲线与颤音参考，便于分别编辑。", { "普通编辑时显示叠加后的音高；标点模式保留基础曲线与颤音参考，便于分别编辑。", "一般編輯顯示疊加音高；控制點模式保留基礎曲線及顫音參考以便分開編輯。", "通常編集では合成ピッチを表示し、ポイントモードでは基礎線とビブラート参照を分けて保持します。", "일반 편집에서는 합산 피치를 표시하며 제어점 모드에서는 기본 곡선과 비브라토 참조를 나누어 유지합니다.", "Normal editing shows combined pitch; point mode retains separate base curve and vibrato reference." } }
    ,{ "关闭颤音", { "关闭颤音", "關閉顫音", "ビブラートを無効化", "비브라토 끄기", "Disable vibrato" } }
    ,{ "时序", { "时序", "時序", "タイミング", "타이밍", "Timing" } }
    ,{ "先行声音 (Pre)，ms：", { "先行声音 (Pre)，ms：", "先行發聲 (Pre)，ms：", "先行発声 (Pre)、ms：", "선행 발성 (Pre), ms: ", "Preutterance (Pre), ms: " } }
    ,{ "重叠 (Overlap)，ms：", { "重叠 (Overlap)，ms：", "重疊 (Overlap)，ms：", "オーバーラップ、ms：", "오버랩 (Overlap), ms: ", "Overlap, ms: " } }
    ,{ "填回音源值", { "填回音源值", "填回音源值", "音源の値を入力", "음원 값 채우기", "Fill voicebank values" } }
    ,{ "连接音高标点 / Connection pitch", { "连接音高标点 / Connection pitch", "連接音高控制點", "接続ピッチポイント", "연결 피치 제어점", "Connection pitch points" } }
    ,{ "删除内部标点 / Delete interior point", { "删除内部标点 / Delete interior point", "刪除內部控制點", "内部ポイントを削除", "내부 제어점 삭제", "Delete interior point" } }
    ,{ "直接编辑预测实参；蓝色虚线保留原始预测", { "直接编辑预测实参；蓝色虚线保留原始预测", "直接編輯預測實參；藍色虛線保留原始預測", "予測パラメーターを直接編集。青い点線は元の予測を保持", "예측 매개변수 직접 편집. 파란 점선은 원래 예측 유지", "Edit predicted parameters directly; blue dotted lines retain original predictions" } }
    ,{ "编辑叠加在实参上的 FLAG 偏移，及 DYN 等控制", { "编辑叠加在实参上的 FLAG 偏移，及 DYN 等控制", "編輯疊加實參的 FLAG 偏移及 DYN 等控制", "実パラメーター上の FLAG オフセットや DYN などを編集", "실제 매개변수에 더하는 FLAG 오프셋 및 DYN 등 편집", "Edit FLAG offsets added to actual parameters, and controls such as DYN" } }
    ,{ "为所选音符开启或关闭偏移；实参与已画曲线保留", { "为所选音符开启或关闭偏移；实参与已画曲线保留", "為所選音符開關偏移；保留實參與已有曲線", "選択ノートのオフセットを切替。実値と描画曲線は保持", "선택 음표 오프셋 전환. 실제 값과 그린 곡선은 유지", "Toggle offsets for selected notes; actual values and drawn curves are retained" } }
    ,{ " · 点击单独编辑此音素的响度／FLAG 包络；点击音符主体返回整音符", { " · 点击单独编辑此音素的响度／FLAG 包络；点击音符主体返回整音符", " · 點選以單獨編輯此音素響度／FLAG 包絡；點音符主體返回整音符", " · クリックでこの音素の音量／FLAG を編集。ノート本体で全体へ戻る", " · 클릭하여 이 음소 음량/FLAG 편집. 음표 본체 클릭으로 전체 복귀", " · Click to edit this phoneme's loudness/FLAG; click the note body to return to whole-note editing" } }
    ,{ "拖动音素分界：调整相邻音素时长，总长度不变（Esc 取消）", { "拖动音素分界：调整相邻音素时长，总长度不变（Esc 取消）", "拖曳音素分界：調整相鄰音素時長，總長不變（Esc 取消）", "音素境界をドラッグ：全長を保ち隣接音素の長さを調整（Esc 取消）", "음소 경계 드래그: 총길이를 유지하며 인접 음소 길이 조정 (Esc 취소)", "Drag phoneme boundary to adjust adjacent durations without changing total length (Esc cancels)" } }
    ,{ "NSF-HiFiGAN 模型未找到，未更改当前算法。\n请将 pc_nsf_hifigan.onnx 和 config.json 放入软件目录下的\nmodels/nsf_hifigan 文件夹，或在算法设置中填写模型目录。", { "NSF-HiFiGAN 模型未找到，未更改当前算法。\n请将 pc_nsf_hifigan.onnx 和 config.json 放入软件目录下的\nmodels/nsf_hifigan 文件夹，或在算法设置中填写模型目录。", "找不到 NSF-HiFiGAN 模型，未變更目前演算法。\n請將 pc_nsf_hifigan.onnx 與 config.json 放入軟體的\nmodels/nsf_hifigan，或在演算法設定指定模型目錄。", "NSF-HiFiGAN モデルがありません。アルゴリズムは変更していません。\npc_nsf_hifigan.onnx と config.json を models/nsf_hifigan に配置するか、\n設定でモデルフォルダーを指定してください。", "NSF-HiFiGAN 모델이 없어 알고리즘을 변경하지 않았습니다.\npc_nsf_hifigan.onnx와 config.json을 models/nsf_hifigan에 넣거나\n알고리즘 설정에서 모델 폴더를 지정하세요.", "NSF-HiFiGAN model missing; the algorithm was not changed.\nPlace pc_nsf_hifigan.onnx and config.json in models/nsf_hifigan,\nor choose the model directory in algorithm settings." } }
    ,{ "让选中音符的 g 随时间连续变化，改由 flag 包络里的曲线决定；Flags 里写的 g 在开启期间不再起作用，其它 flag 照常。支持普通 UTAU、界•UTAU 与谋•UTAU；需使用支持曲线的 WCSNDM 引擎", { "让选中音符的 g 随时间连续变化，改由 flag 包络里的曲线决定；Flags 里写的 g 在开启期间不再起作用，其它 flag 照常。支持普通 UTAU、界•UTAU 与谋•UTAU；需使用支持曲线的 WCSNDM 引擎", "讓所選音符的 g 由 FLAG 曲線隨時間控制；啟用期間忽略 Flags 中的 g，其他不變。支援 UTAU、界•UTAU、謀•UTAU，須用支援曲線的 WCSNDM。", "選択ノートの g を FLAG 曲線で連続制御します。有効中は Flags の g を無視し、他は維持します。UTAU・界・謀に対応。曲線対応 WCSNDM が必要です。", "선택 음표의 g를 FLAG 곡선으로 연속 제어합니다. 활성 중 Flags의 g는 무시하고 나머지는 유지합니다. UTAU/Jie/Mou를 지원하며 곡선 지원 WCSNDM이 필요합니다.", "Control selected notes' g continuously via FLAG curves. Ignore static g while enabled; other flags are unchanged. Supports UTAU/Jie/Mou with a curve-capable WCSNDM engine." } }
    ,{ "在底部通道里画 g 的渐变曲线；先对该音符开启线性 flag（普通 UTAU、界•UTAU 与谋•UTAU 模式）", { "在底部通道里画 g 的渐变曲线；先对该音符开启线性 flag（普通 UTAU、界•UTAU 与谋•UTAU 模式）", "於底部通道繪製 g 漸變；先為音符啟用線性 FLAG（UTAU、界、謀模式）", "下部レーンで g 曲線を描画。先にノートの連続 FLAG を有効にしてください（UTAU・界・謀）。", "하단 레인에서 g 곡선을 그립니다. 먼저 음표의 연속 FLAG를 활성화하세요 (UTAU/Jie/Mou).", "Draw g curves in the bottom lane; first enable continuous FLAG for the note (UTAU/Jie/Mou)." } }
    ,{ "先行声音", { "先行声音", "先行發聲", "先行発声", "선행 발성", "Preutterance" } }
    ,{ "声母末", { "声母末", "聲母末", "子音終端", "자음 끝", "Onset end" } }
    ,{ "介音末", { "介音末", "介音末", "介音終端", "매개음 끝", "Glide end" } }
    ,{ "韵腹末", { "韵腹末", "韻腹末", "核母音終端", "모음핵 끝", "Nucleus end" } }
    ,{ "声母", { "声母", "聲母", "子音", "자음", "Onset" } }
    ,{ "介音", { "介音", "介音", "介音", "매개음", "Glide" } }
    ,{ "韵腹", { "韵腹", "韻腹", "核母音", "모음핵", "Nucleus" } }
    ,{ "韵尾", { "韵尾", "韻尾", "韻尾", "받침", "Coda" } }
    ,{ "把上面的数值存成预设；名称取自左边的输入框", { "把上面的数值存成预设；名称取自左边的输入框", "將上方數值存為預設，名稱取自左側輸入框", "上の値をプリセット保存。名前は左の入力欄を使用", "위 값을 프리셋으로 저장. 이름은 왼쪽 입력란 사용", "Save the values above as a preset using the name in the left box" } }
    ,{ "正值下移共振峰（听感偏成熟/厚），负值上移（偏年轻/薄）。\n", { "正值下移共振峰（听感偏成熟/厚），负值上移（偏年轻/薄）。\n", "正值降低共振峰（較成熟／厚），負值提高（較年輕／薄）。\n", "正の値はフォルマントを下げ（大人／厚い）、負の値は上げます（若い／薄い）。\n", "양수는 포먼트를 낮추고 (성숙/두꺼움), 음수는 높입니다 (젊음/얇음).\n", "Positive values lower formants (mature/thick), negative values raise them (young/thin).\n" } }
    ,{ "按半音平移选中的音符（12 平均律）。\n只改音高，不改时间；音高标点跟着一起移动。\n两个框都填时按两者之差平移。", { "按半音平移选中的音符（12 平均律）。\n只改音高，不改时间；音高标点跟着一起移动。\n两个框都填时按两者之差平移。", "以半音平移所選音符（十二平均律）。\n僅改音高，時序不變；控制點同步移動。\n兩欄皆填時以差值平移。", "選択ノートを半音で移調（十二平均律）。\n時刻は変えずピッチポイントも移動します。\n両欄入力時は差分で移調します。", "선택 음표를 반음 이동합니다 (12평균율).\n시간은 유지하고 피치 제어점도 함께 이동합니다.\n두 칸 입력 시 차이만큼 이동합니다.", "Transpose selected notes by semitones (12-TET).\nTime is unchanged; pitch points move with notes.\nIf both fields are filled, transpose by their difference." } }
    ,{ "把这个音符所用 oto 条目中的全部分段信息整体移动：offset、重叠、先行声音、辅音和 cutoff，以及四区域的划分，一起移动同样的距离。\n正数为后移（读取录音中更靠后的部分），负数为前移；单位是毫秒。\n音符在曲子里的位置不变，变的只是它读到录音的哪一段。", { "把这个音符所用 oto 条目中的全部分段信息整体移动：offset、重叠、先行声音、辅音和 cutoff，以及四区域的划分，一起移动同样的距离。\n正数为后移（读取录音中更靠后的部分），负数为前移；单位是毫秒。\n音符在曲子里的位置不变，变的只是它读到录音的哪一段。", "整體移動此音符 OTO 的 offset、重疊、先行發聲、子音、cutoff 及四區邊界。\n正值往錄音後方，負值往前，單位毫秒。\n曲中位置不變，僅改讀取的錄音範圍。", "このノートの OTO 全境界（offset・重なり・先行発声・子音・cutoff・四区間）を同じ距離移動します。\n正は録音の後方、負は前方、単位 ms。\n曲内の位置は変えず読取範囲のみ変更します。", "이 음표 OTO의 offset, 겹침, 선행 발성, 자음, cutoff 및 네 구간 경계를 함께 이동합니다.\n양수는 녹음 뒤쪽, 음수는 앞쪽이며 단위는 ms입니다.\n곡 안의 위치는 유지하고 읽는 녹음 범위만 변경합니다.", "Move all OTO boundaries together: offset, overlap, preutterance, consonant, cutoff and four-region divisions.\nPositive reads later audio, negative earlier; units are ms.\nThe note's score position stays unchanged; only its recording range changes." } }
    ,{ "先行声音在辅音速度计算后应用；重叠为正时交叉衔接，为负时形成静音间隔。\n填回音源值后直接应用，音符重新跟随音源；手动改过的数值则固定不变。", { "先行声音在辅音速度计算后应用；重叠为正时交叉衔接，为负时形成静音间隔。\n填回音源值后直接应用，音符重新跟随音源；手动改过的数值则固定不变。", "先行發聲在子音速度計算後套用；正重疊為交叉連接，負值為靜音間隔。\n填回音源值直接套用即重新繼承；手動值保持固定。", "先行発声は子音速度の計算後に適用。重なりが正ならクロス接続、負なら無音です。\n音源値を入力して適用すると再継承し、手動値は固定されます。", "선행 발성은 자음 속도 계산 후 적용됩니다. 겹침 양수는 교차 연결, 음수는 무음 간격입니다.\n음원 값을 채워 적용하면 다시 상속하며 수동 값은 고정됩니다.", "Preutterance applies after consonant velocity. Positive overlap crossfades, negative overlap creates a silent gap.\nFill voicebank values and apply to inherit them again; manually edited values stay fixed." } }
    ,{ "\n音源库中没有这个歌词，所以下方的音源值是空的；「填回音源值」仍可用来清掉上一个歌词留下的固定值。", { "\n音源库中没有这个歌词，所以下方的音源值是空的；「填回音源值」仍可用来清掉上一个歌词留下的固定值。", "\n音源庫無此歌詞，下方來源值為空；「填回音源值」仍可清除前一歌詞的固定值。", "\nこの歌詞が音源にないため値は空です。「音源値を入力」で前の歌詞の固定値を消去できます。", "\n음원에 이 가사가 없어 아래 값은 비어 있습니다. 음원 값 채우기로 이전 가사의 고정 값을 지울 수 있습니다.", "\nThis lyric is absent from the voicebank, so values below are empty. Fill voicebank values can still clear fixed values left by the previous lyric." } }
    ,{ "当前轨道没有可显示的响度包络", { "当前轨道没有可显示的响度包络", "目前軌道無可顯示的響度包絡", "現在のトラックに表示可能な音量エンベロープがありません", "현재 트랙에 표시할 음량 엔벌로프가 없습니다", "No loudness envelope to display for this track" } }
    ,{ "读音留空时自动转换；修改读音不改变歌词。中文可填拼音；英文可用 en/hello。\n音素覆盖示例：[zh/zh zh/a en/ng]。多音节词后续音符填 +2、+3；普通延音用 +。", { "读音留空时自动转换；修改读音不改变歌词。中文可填拼音；英文可用 en/hello。\n音素覆盖示例：[zh/zh zh/a en/ng]。多音节词后续音符填 +2、+3；普通延音用 +。", "讀音留空時自動轉換；改讀音不改歌詞。中文可填拼音，英文可用 en/hello。\n音素覆寫例：[zh/zh zh/a en/ng]。多音節後續填 +2、+3；延音用 +。", "読みが空欄なら自動変換し、歌詞は変更しません。中国語はピンイン、英語は en/hello。\n音素上書き例：[zh/zh zh/a en/ng]。複数音節の後続は +2、+3、伸ばす場合は +。", "읽기를 비우면 자동 변환하며 가사는 바꾸지 않습니다. 중국어는 병음, 영어는 en/hello를 사용할 수 있습니다.\n음소 덮어쓰기 예: [zh/zh zh/a en/ng]. 다음 음절은 +2, +3, 일반 연장은 +.", "Leave reading empty for automatic conversion; reading edits do not change lyrics. Chinese accepts pinyin; English accepts en/hello.\nPhoneme override: [zh/zh zh/a en/ng]. Later syllables use +2, +3; ordinary extension uses +." } }
    ,{ "自动读音", { "自动读音", "自動讀音", "自動読み", "자동 읽기", "Automatic reading" } }
    ,{ "轨道词典：一行“歌词 = 读音”，例如 en/read = [r eh d]。修改词典会清除本轨手动音素时长。", { "轨道词典：一行“歌词 = 读音”，例如 en/read = [r eh d]。修改词典会清除本轨手动音素时长。", "軌道詞典：每行「歌詞 = 讀音」，如 en/read = [r eh d]。修改會清除此軌手動音素時長。", "トラック辞書：一行に「歌詞 = 読み」、例 en/read = [r eh d]。変更すると手動音素長を消去します。", "트랙 사전: 한 줄에 가사 = 읽기, 예: en/read = [r eh d]. 변경하면 이 트랙의 수동 음소 길이가 지워집니다.", "Track dictionary: one lyric = reading per line, e.g. en/read = [r eh d]. Changes clear this track's manual phoneme timings." } }
    ,{ "预览发音", { "预览发音", "預覽發音", "発音をプレビュー", "발음 미리 보기", "Preview pronunciation" } }
    ,{ "检查并应用", { "检查并应用", "檢查並套用", "確認して適用", "검사 후 적용", "Check and apply" } }
    ,{ "清除当前读音覆盖", { "清除当前读音覆盖", "清除目前讀音覆寫", "現在の読み上書きを消去", "현재 읽기 덮어쓰기 지우기", "Clear current reading overrides" } }
    ,{ "正在转换并检查发音…", { "正在转换并检查发音…", "正在轉換並檢查發音…", "発音を変換・確認中…", "발음 변환 및 검사 중…", "Converting and checking pronunciation…" } }
    ,{ "发音检查通过；英文生词预测及跨语言音素组合请试听确认。", { "发音检查通过；英文生词预测及跨语言音素组合请试听确认。", "發音檢查通過；英文生詞預測及跨語言音素組合請試聽確認。", "発音確認完了。英語未知語の予測と多言語音素の組合せは試聴で確認してください。", "발음 검사를 통과했습니다. 영어 미등록 단어 예측 및 언어 간 음소 조합은 직접 들어 확인하세요.", "Pronunciation checks passed; audition predicted English words and cross-language phoneme combinations." } }
    ,{ "读音已修改，点击预览检查；应用会再次检查并清除失效的音素时长。", { "读音已修改，点击预览检查；应用会再次检查并清除失效的音素时长。", "讀音已修改，請預覽檢查；套用會再次檢查並清除失效音素時長。", "読みを変更しました。プレビューで確認してください。適用時に再確認し無効な音素長を消去します。", "읽기가 변경되었습니다. 미리 보기로 검사하세요. 적용 시 다시 검사하고 유효하지 않은 음소 길이를 지웁니다.", "Reading changed. Preview to check; applying rechecks and clears invalid phoneme timings." } }
    ,{ "锁定元音起点，保留 MIDI 拍点；关闭后也可拖动元音起点", { "锁定元音起点，保留 MIDI 拍点；关闭后也可拖动元音起点", "鎖定母音起點以保留 MIDI 拍點；關閉後可拖曳母音起點", "母音開始を固定して MIDI 拍を維持。無効にすると母音開始もドラッグ可能", "모음 시작을 잠가 MIDI 박자를 유지합니다. 끄면 모음 시작도 드래그할 수 있습니다", "Lock vowel onsets to preserve MIDI beats; disable to drag vowel onsets too" } }
    ,{ "拖动分界线改变相邻音素时长；双击恢复预测。数字为相对音符起点的毫秒数，允许负值。悬停音素查看完整标识。", { "拖动分界线改变相邻音素时长；双击恢复预测。数字为相对音符起点的毫秒数，允许负值。悬停音素查看完整标识。", "拖曳邊界改相鄰音素時長，雙擊還原預測。數值為相對音符起點毫秒，可負值；懸停顯示完整標識。", "境界をドラッグして隣接音素長を変更、ダブルクリックで予測に復元。数値はノート開始からの ms（負数可）。ホバーで全識別子を表示。", "경계 드래그로 인접 음소 길이를 변경하고 두 번 클릭으로 예측을 복원합니다. 값은 음표 시작 기준 ms (음수 허용)이며 가리키면 전체 식별자를 표시합니다.", "Drag boundaries to change adjacent durations; double-click restores prediction. Values are ms relative to note start and may be negative. Hover for full identifiers." } }
    ,{ "音源、歌词或音符长度已变化：失配的旧编辑已忽略，越界边界已限制；应用后保存当前结果。", { "音源、歌词或音符长度已变化：失配的旧编辑已忽略，越界边界已限制；应用后保存当前结果。", "音源、歌詞或長度已變：忽略失配舊編輯並限制越界邊界，套用後儲存目前結果。", "音源・歌詞・長さが変更され、合わない編集を無視し境界を制限しました。適用で現在の結果を保存します。", "음원, 가사 또는 길이가 바뀌어 맞지 않는 기존 편집은 무시하고 경계를 제한했습니다. 적용하여 현재 결과를 저장하세요.", "Voicebank, lyrics or lengths changed: mismatched edits were ignored and boundaries clamped. Apply to save current results." } }
    ,{ "恢复当前预测", { "恢复当前预测", "還原目前預測", "現在の予測を復元", "현재 예측 복원", "Restore current predictions" } }
    ,{ "时长 ", { "时长 ", "時長 ", "長さ ", "길이 ", "Duration " } }
    ,{ " ms；起点相对音符（ms）", { " ms；起点相对音符（ms）", " ms；起點相對音符（ms）", " ms；開始はノート相対（ms）", " ms; 시작은 음표 기준 (ms)", " ms; onset relative to note (ms)" } }
    ,{ " · 已修改", { " · 已修改", " · 已修改", " · 編集済み", " · 편집됨", " · Edited" } }
    ,{ " · 预测", { " · 预测", " · 預測", " · 予測", " · 예측", " · Predicted" } }
    ,{ "C 辅音 · V 元音 · S 休止；黄色圆点为手动边界，灰线为预测位置", { "C 辅音 · V 元音 · S 休止；黄色圆点为手动边界，灰线为预测位置", "C子音 · V母音 · S休止；黃點是手動邊界，灰線是預測", "C子音 · V母音 · S休符。黄点は手動境界、灰線は予測位置", "C자음 · V모음 · S쉼. 노란 점은 수동 경계, 회색 선은 예측 위치", "C consonant · V vowel · S rest; yellow dots are manual boundaries, grey lines are predictions" } }
    ,{ "工程已变化，未写入此次结果。请关闭后重新打开音素编辑器。", { "工程已变化，未写入此次结果。请关闭后重新打开音素编辑器。", "工程已變，未寫入結果。請重開音素編輯器。", "プロジェクトが変更されたため結果は保存していません。音素エディターを開き直してください。", "프로젝트가 바뀌어 결과를 쓰지 않았습니다. 음소 편집기를 닫고 다시 여세요.", "Project changed; results were not written. Close and reopen the phoneme editor." } }
    ,{ " · 延音", { " · 延音", " · 延音", " · 伸長", " · 연장", " · Extension" } }
    ,{ " · 延音，边界在原歌词音符上编辑", { " · 延音，边界在原歌词音符上编辑", " · 延音，邊界於原歌詞音符編輯", " · 伸長、境界は元の歌詞ノートで編集", " · 연장, 경계는 원래 가사 음표에서 편집", " · Extension; edit boundaries on the original lyric note" } }
    ,{ "保持元音拍点", { "保持元音拍点", "保持母音拍點", "母音の拍を保持", "모음 박자 유지", "Preserve vowel beats" } }
    ,{ "恢复选中预测", { "恢复选中预测", "還原所選預測", "選択の予測を復元", "선택 예측 복원", "Restore selected predictions" } }
    ,{ "恢复全部预测", { "恢复全部预测", "還原全部預測", "全予測を復元", "전체 예측 복원", "Restore all predictions" } }
    ,{ "应用并试听", { "应用并试听", "套用並試聽", "適用して試聴", "적용 후 미리 듣기", "Apply and audition" } }
    ,{ "HAMOOD · 和弦与段落资料", { "HAMOOD · 和弦与段落资料", "HAMOOD · 和弦與段落資料", "HAMOOD · コードとセクション", "HAMOOD · 코드 및 구간 정보", "HAMOOD · Chords and sections" } }
    ,{ "以工程秒数编辑，按拍点保存并跟随曲速。添加、更新和删除立即保存到工程；已确认和弦不会被分析覆盖。", { "以工程秒数编辑，按拍点保存并跟随曲速。添加、更新和删除立即保存到工程；已确认和弦不会被分析覆盖。", "以工程秒數編輯，依拍點儲存並跟隨曲速。增改刪立即記入工程；已確認和弦不被分析覆寫。", "秒単位で編集し、拍として保存してテンポに追従します。追加・更新・削除は直ちに工程へ反映。確定コードは解析で上書きしません。", "프로젝트 초 단위로 편집하고 박자로 저장하여 템포를 따릅니다. 추가/수정/삭제는 즉시 프로젝트에 반영하며 확인한 코드는 분석으로 덮어쓰지 않습니다.", "Edit in project seconds; save by beats and follow tempo. Add/update/delete immediately updates the project; confirmed chords are not overwritten by analysis." } }
    ,{ "和弦", { "和弦", "和弦", "コード", "코드", "Chord" } }
    ,{ "段落", { "段落", "段落", "セクション", "구간", "Section" } }
    ,{ "读取 / 分析伴奏", { "读取 / 分析伴奏", "讀取／分析伴奏", "伴奏を読込／解析", "반주 읽기/분석", "Read / analyse accompaniment" } }
    ,{ "开始（秒）", { "开始（秒）", "開始（秒）", "開始（秒）", "시작 (초)", "Start (seconds)" } }
    ,{ "结束（秒）", { "结束（秒）", "結束（秒）", "終了（秒）", "끝 (초)", "End (seconds)" } }
    ,{ "和弦 / 段落名称", { "和弦 / 段落名称", "和弦／段落名稱", "コード／セクション名", "코드/구간 이름", "Chord / section name" } }
    ,{ "已人工确认", { "已人工确认", "已人工確認", "手動確認済み", "사용자 확인됨", "Manually confirmed" } }
    ,{ "更新所选", { "更新所选", "更新所選", "選択を更新", "선택 항목 갱신", "Update selected" } }
    ,{ "保存失败：工程已变化，请重新打开窗口。", { "保存失败：工程已变化，请重新打开窗口。", "儲存失敗：工程已變更，請重開視窗。", "保存失敗：プロジェクトが変更されました。画面を開き直してください。", "저장 실패: 프로젝트가 변경되었습니다. 창을 다시 여세요.", "Save failed: project changed. Reopen the window." } }
    ,{ "资料已记入工程；保存工程后重开仍可使用。", { "资料已记入工程；保存工程后重开仍可使用。", "資料已記入工程；儲存工程後重開仍可使用。", "情報を工程へ反映しました。保存後に再度開いても使用できます。", "정보를 프로젝트에 기록했습니다. 프로젝트 저장 후 다시 열어도 사용할 수 있습니다.", "Data recorded in the project; save the project to retain it after reopening." } }
    ,{ "请输入有效起止秒数和名称。", { "请输入有效起止秒数和名称。", "請輸入有效起止秒數及名稱。", "有効な開始・終了秒と名前を入力してください。", "유효한 시작/끝 초와 이름을 입력하세요.", "Enter valid start/end seconds and a name." } }
    ,{ "无法识别和弦，请用 C、Am、C7、Bbmaj7、Dm7b5 等写法。", { "无法识别和弦，请用 C、Am、C7、Bbmaj7、Dm7b5 等写法。", "無法辨識和弦，請用 C、Am、C7、Bbmaj7、Dm7b5 等寫法。", "コードを認識できません。C、Am、C7、Bbmaj7、Dm7b5 などを使用してください。", "코드를 인식하지 못했습니다. C, Am, C7, Bbmaj7, Dm7b5 등의 표기를 사용하세요.", "Unknown chord; use notation such as C, Am, C7, Bbmaj7 or Dm7b5." } }
    ,{ "正在读取或分析伴奏，已确认的资料将保留…", { "正在读取或分析伴奏，已确认的资料将保留…", "正在讀取或分析伴奏，保留已確認資料…", "伴奏を読込・解析中。確認済みの情報は保持します…", "반주 읽기/분석 중. 확인한 정보는 유지됩니다…", "Reading or analysing accompaniment; confirmed data is retained…" } }
    ,{ "偏移 ", { "偏移 ", "偏移 ", "オフセット ", "오프셋 ", "Offset " } }
    ,{ "DS 音高偏移（半音）", { "DS 音高偏移（半音）", "DS 音高偏移（半音）", "DS ピッチオフセット（半音）", "DS 피치 오프셋 (반음)", "DS pitch offset (semitones)" } }
    ,{ "增加偏移标点", { "增加偏移标点", "新增偏移控制點", "オフセット点を追加", "오프셋 제어점 추가", "Add offset point" } }
    ,{ "整音偏移数值…", { "整音偏移数值…", "整音偏移數值…", "ノート全体のオフセット値…", "전체 음표 오프셋 값…", "Whole-note offset value…" } }
    ,{ "此音偏移归零", { "此音偏移归零", "此音偏移歸零", "このノートのオフセットをゼロに", "이 음표 오프셋을 0으로", "Zero this note's offset" } }
    ,{ "删除此偏移点", { "删除此偏移点", "刪除此偏移點", "このオフセット点を削除", "이 오프셋 점 삭제", "Delete this offset point" } }
    ,{ "整音偏移", { "整音偏移", "整音偏移", "ノート全体のオフセット", "전체 음표 오프셋", "Whole-note offset" } }
    ,{ "以半音为单位，负数向下。此操作将偏移线设为水平线；原始预测虚线不变。", { "以半音为单位，负数向下。此操作将偏移线设为水平线；原始预测虚线不变。", "以半音為單位，負值向下。偏移線會設為水平；原始預測虛線不變。", "単位は半音、負は下方向。オフセットを水平線にします。元の予測点線は変えません。", "단위는 반음, 음수는 아래입니다. 오프셋을 수평선으로 설정하며 원래 예측 점선은 유지합니다.", "Semitones; negative lowers pitch. Set the offset to a horizontal line; original predicted dotted line is unchanged." } }
    ,{ "偏移（半音）", { "偏移（半音）", "偏移（半音）", "オフセット（半音）", "오프셋 (반음)", "Offset (semitones)" } }
    ,{ "初始化选中标点（Ctrl+R）", { "初始化选中标点（Ctrl+R）", "初始化所選控制點（Ctrl+R）", "選択ポイントを初期化（Ctrl+R）", "선택 제어점 초기화 (Ctrl+R)", "Initialize selected points (Ctrl+R)" } }
    ,{ "删除选中标点（Del）", { "删除选中标点（Del）", "刪除所選控制點（Del）", "選択ポイントを削除（Del）", "선택 제어점 삭제 (Del)", "Delete selected points (Del)" } }
    ,{ "交叉", { "交叉", "交叉", "クロス", "교차", "Crossfade" } }
};
std::atomic<I18n::Language> uiLanguage{I18n::Language::zhCN};
const auto& literalRows()
{
    static const auto index=[]
    {
        std::unordered_map<std::string,const Row*> result;
        std::vector<std::string> keys;keys.reserve(strings.size());
        for(const auto& item:strings)keys.push_back(item.first);
        std::sort(keys.begin(),keys.end());
        // Explicit keys win; source collisions have deterministic key order.
        for(const auto& key:keys)result.emplace(key,&strings.at(key));
        for(const auto& key:keys){const auto& row=strings.at(key);result.emplace(row[0],&row);result.emplace(row[4],&row);}
        return result;
    }();
    return index;
}
}

I18n::I18n()
{
    const auto locale = juce::SystemStats::getUserLanguage().toLowerCase();
    if (locale.startsWith("ja")) language = Language::jaJP;
    else if (locale.startsWith("ko")) language = Language::koKR;
    else if (locale.contains("tw") || locale.contains("hk") || locale.contains("hant")) language = Language::zhTW;
    else if (!locale.startsWith("zh")) language = Language::enUS;
    uiLanguage.store(language,std::memory_order_relaxed);
}

void I18n::setLanguage(Language languageToUse)
{
    language=languageToUse;uiLanguage.store(language,std::memory_order_relaxed);
}

juce::String I18n::uiText(const char* sourceLiteral)
{
    const auto& rows=literalRows();const auto found=rows.find(sourceLiteral);
    return found==rows.end()?juce::String::fromUTF8(sourceLiteral)
        :juce::String::fromUTF8((*found->second)[static_cast<std::size_t>(uiLanguage.load(std::memory_order_relaxed))]);
}

juce::String I18n::text(const juce::String& key) const
{
    const auto found = strings.find(key.toStdString());
    if (found == strings.end()) return key;
    return juce::String::fromUTF8(found->second[static_cast<std::size_t>(language)]);
}
}
