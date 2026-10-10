--[[
  build_reverse_probe.lua —— F-4：倒放 take 的 ARA region 时间坐标系探针。

  问题（见 docs/plans/2026-10-09-ara-plugin-fade-rewrite-reverse-auth-and-stability.md
  的 Task 3.0 与 3.2）：倒放 take 的 ARA region 报的
  `start_in_modification_time` / `duration_in_modification_time`，用的是**正向坐标**
  还是**已镜像坐标**？这决定把 `reversed` 送进渲染时要不要再翻一次源窗口 ——
  判错就是二次镜像 = 又渲染成正放。

  已知（不复测）：ARA 交给插件的 PCM 是**正向**的（`captures/phase3a-FINDINGS.md`：
  倒放输出 vs 正放源差 5.96e-8，vs 倒放源差 0.50）。所以这里只问**时间坐标**。

  做法：在同一个不对称源上放两个**非对称裁切**的 item —— 源内区间都是 [0.25, 1.25]
  （`D_STARTOFFS=0.25`、`D_LENGTH=1.0`），一个正放、一个倒放（官方 action 41051，
  并用 `PCM_Source_GetSectionInfo` 核实 `reversed=true`）。两者几何全等，唯一差别是
  方向。于是插件日志里两个 region 的 `startMod` 是否相等就直接回答 F-4：

    * 相等（都 = 0.25）                → **正向坐标**（渲染层按既有内核倒放数学翻 PCM 即可）
    * 差 `source_len - offset - length`（= 0.75）→ **已镜像坐标**（不得再翻源窗口）

  为什么必须**非对称裁切**：`task11` / `phase3a` 的倒放 item 都是**整段源**（offset 0、
  length = 源长），正向与镜像坐标都落在 [0, 2] —— 两个假设给出同一个数，区分不了。

  产物：
    captures/reverse-plugin.log  —— 插件日志（region 坐标，`HIFISHIFTER_ARA_LOG` 指定）
    captures/reverse-script.log  —— 本脚本的夹具与核实记录
  之后跑 `verify_reverse_capture.ps1` 生成分类结论（forward / mirrored）。

  本脚本是一次性探针产物，不被 backend/ 或 frontend/ 引用。
]]

local sep = package.config:sub(1, 1)
local scriptPath = debug.getinfo(1, 'S').source:sub(2)
local scriptDir = scriptPath:match('^(.*)' .. sep .. '[^' .. sep .. ']*$') or '.'
local fixturesDir = scriptDir .. sep .. 'fixtures'
local capturesDir = scriptDir .. sep .. 'captures'
local logPath = capturesDir .. sep .. 'reverse-script.log'
local pluginLogPath = os.getenv('HIFISHIFTER_ARA_LOG')
local outPath = capturesDir .. sep .. 'reverse-probe.json'
local wavPath = fixturesDir .. sep .. 'phase3a-asymmetric.wav'

-- 非对称裁切窗口（秒）。源长 2.0s；镜像后的源内起点 = 2.0 - 0.25 - 1.0 = 0.75。
local SOURCE_OFFSET = 0.25
local ITEM_LENGTH = 1.0
local FORWARD_POSITION = 0.0
local REVERSE_POSITION = 3.0

local function log(message)
  local line = tostring(message)
  reaper.ShowConsoleMsg('[reverse] ' .. line .. '\n')
  local file = io.open(logPath, 'a')
  if file then file:write(line .. '\n'); file:close() end
end

local function exists(path)
  local file = io.open(path, 'rb')
  if file then file:close(); return true end
  return false
end

local function readFile(path)
  if not path or path == '' then return '' end
  local file = io.open(path, 'r')
  if not file then return '' end
  local content = file:read('a') or ''
  file:close()
  return content
end

local truncate = io.open(logPath, 'w')
if truncate then truncate:close() end
if pluginLogPath then
  local pluginLog = io.open(pluginLogPath, 'w')
  if pluginLog then pluginLog:close() end
end

--- 完成信号：`run_probe_headless.ps1` 用这个文件的存在判定"探针跑完了"。
--- 只在真正完成/放弃时写，不在开头写 —— 否则会在 ARA 还没推模型时就把 REAPER 杀掉。
local function finish(status, detail)
  local file = io.open(outPath, 'w')
  if file then
    file:write(string.format('{"probe":"reverse-coordinates","status":"%s","detail":"%s"}\n', status, detail))
    file:close()
  end
  log('probe status=' .. status .. ' detail=' .. detail)
end

log('=== reverse probe run ' .. os.date('%Y-%m-%d %H:%M:%S') .. ' ===')
if not exists(wavPath) then finish('error', 'missing fixture'); log('MISSING fixture: ' .. wavPath); return end

local trackIndex = reaper.CountTracks(0)
reaper.InsertTrackAtIndex(trackIndex, true)
local track = reaper.GetTrack(0, trackIndex)
reaper.GetSetMediaTrackInfo_String(track, 'P_NAME', 'reverse-coordinates', true)
reaper.SetOnlyTrackSelected(track)

--- 插入一个非对称裁切的 item：源内 [offset, offset+length]，放在 position 处。
--- 与 `build_phase3a_probe.lua` 同一手法，且清掉四个淡变长度 —— 内容淡化会额外
--- 触发委托路径，与本探针问的"时间坐标"无关。
local function insertItem(position)
  reaper.SetEditCurPos(position, false, false)
  reaper.InsertMedia(wavPath, 0)
  local item = reaper.GetTrackMediaItem(track, reaper.CountTrackMediaItems(track) - 1)
  reaper.SetMediaItemInfo_Value(item, 'D_LENGTH', ITEM_LENGTH)
  reaper.SetMediaItemInfo_Value(item, 'B_LOOPSRC', 0)
  reaper.SetMediaItemInfo_Value(item, 'D_FADEINLEN', 0)
  reaper.SetMediaItemInfo_Value(item, 'D_FADEOUTLEN', 0)
  reaper.SetMediaItemInfo_Value(item, 'D_FADEINLEN_AUTO', 0)
  reaper.SetMediaItemInfo_Value(item, 'D_FADEOUTLEN_AUTO', 0)
  reaper.SetMediaItemTakeInfo_Value(reaper.GetActiveTake(item), 'D_STARTOFFS', SOURCE_OFFSET)
  return item
end

local forward = insertItem(FORWARD_POSITION)
local reverse = insertItem(REVERSE_POSITION)

--- 记录 take 自己的源内窗口读数。F-4 的判读依赖"倒放后 item 覆盖的源内区间是否
--- 变了"：`PCM_Source_GetSectionInfo` 对**普通裁切 take** 返回 false（见本次运行日志），
--- 所以窗口要靠 `D_STARTOFFS` / `D_LENGTH` 这两个直接读数。
local function takeFacts(label, item)
  local take = reaper.GetActiveTake(item)
  local startoffs = reaper.GetMediaItemTakeInfo_Value(take, 'D_STARTOFFS')
  local playrate = reaper.GetMediaItemTakeInfo_Value(take, 'D_PLAYRATE')
  local length = reaper.GetMediaItemInfo_Value(item, 'D_LENGTH')
  return string.format('take facts: %s startoffs=%.6f playrate=%.6f length=%.6f',
    label, startoffs, playrate, length)
end

--- 用宿主 section reader 核实方向位。`B_REVERSED` setter 是无效证据
--- （见 `EXECUTION-LEDGER.md:639-640`），必须读 `PCM_Source_GetSectionInfo`。
local function sectionLine(label, item)
  local source = reaper.GetMediaItemTake_Source(reaper.GetActiveTake(item))
  local ok, offset, length, reversed = reaper.PCM_Source_GetSectionInfo(source)
  return string.format(
    'reverse verified: %s section=%s reversed=%s offset=%.6f length=%.6f',
    label, tostring(ok), tostring(reversed), offset or -1, length or -1)
end

log(takeFacts('forward', forward))
log(sectionLine('forward', forward))

-- 官方 action 41051 = "Item properties: Toggle take reverse"。用 kbd_enumerateActions
-- 按名字查，不硬编码命令号（README 的纪律）。
local reverseAction
do
  local section = reaper.SectionFromUniqueID(0)
  local index = 0
  while true do
    local command, name = reaper.kbd_enumerateActions(section, index)
    if command == 0 then break end
    if name == 'Item properties: Toggle take reverse' then reverseAction = command end
    index = index + 1
  end
end
if not reverseAction then finish('error', 'reverse action not found'); log('FATAL: official take reverse action not found'); return end

reaper.SelectAllMediaItems(0, false)
reaper.SetMediaItemSelected(reverse, true)
reaper.Main_OnCommand(reverseAction, 0)

local reverseLine = sectionLine('reversed', reverse)
log(takeFacts('reversed', reverse))
log(reverseLine)
local ok, _, _, reversed = reaper.PCM_Source_GetSectionInfo(
  reaper.GetMediaItemTake_Source(reaper.GetActiveTake(reverse)))
if not (ok and reversed) then finish('error', 'reverse not verified'); log('FATAL: take source must actually be reversed'); return end

reaper.UpdateArrange()
local fxIndex = reaper.TrackFX_AddByName(track, 'HiFiShifter', false, -1)
log('TrackFX_AddByName -> ' .. tostring(fxIndex))
if fxIndex < 0 then finish('error', 'HiFiShifter FX not found'); log('FATAL: HiFiShifter FX not found on this track'); return end

-- 等插件为两个 item 各创建一条 region。region 的坐标由插件日志给出。
local checks = 0
local deadline = reaper.time_precise() + 20.0
local function tick()
  checks = checks + 1
  local pluginLog = readFile(pluginLogPath)
  if pluginLog:find('playback_region #0', 1, true) and pluginLog:find('playback_region #1', 1, true) then
    finish('ok', 'both regions observed at check ' .. checks)
    return
  end
  if reaper.time_precise() >= deadline then
    finish('timeout', 'gave up after ' .. checks .. ' checks')
    return
  end
  reaper.defer(tick)
end

reaper.defer(tick)
