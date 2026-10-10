--[[
  build_reverse_render_probe.lua —— Task 3.4 的端到端验收：插件渲染的倒放片段。

  背景：F-4（见 `REVERSE-FINDINGS.md`）判明——倒放 take 的 ARA region 报的是**已镜像
  坐标**（源内 `[0.25, 1.25]` 的裁切被报成 `[0.75, 1.75]`），而 take 实际播的仍是正向
  `[0.25, 1.25]`。内核的倒放消费窗口要的是**正向**窗口终点，所以播种层必须先把镜像窗口
  翻回正向，否则内核二次镜像 = 又是正放。

  本探针挂**真** HiFiShifter FX，把一个倒放 item 的播放渲染成 WAV —— 问的是**插件**输出
  了哪一段，而不是 REAPER 自己播了哪一段（后者由 `build_reverse_window_probe.lua` 回答）。

  正确结果 = `reverse(源[0.25, 1.25])`。错误结果有两类：
    * `reverse(源[0.75, 1.75])` —— 窗口没翻正（二次镜像前的旧行为）；
    * `源[0.25, 1.25]` / `源[0.75, 1.75]` —— 压根没倒放。

  产物：captures/reverse-render.wav（+ captures/reverse-render-probe.json 完成信号）。
  之后跑 verify_reverse_render.py 判定。
]]

local sep = package.config:sub(1, 1)
local scriptPath = debug.getinfo(1, 'S').source:sub(2)
local scriptDir = scriptPath:match('^(.*)' .. sep .. '[^' .. sep .. ']*$') or '.'
local fixturesDir = scriptDir .. sep .. 'fixtures'
local capturesDir = scriptDir .. sep .. 'captures'
local logPath = capturesDir .. sep .. 'reverse-render-script.log'
local outPath = capturesDir .. sep .. 'reverse-render-probe.json'
local wavPath = fixturesDir .. sep .. 'phase3a-asymmetric.wav'

local RATE = 44100
local SOURCE_OFFSET = 0.25
local ITEM_LENGTH = 1.0
-- 插件要先认领 region、映射文档、物化 PCM 才可能渲染；给足时间再触发渲染。
local SETTLE_SECONDS = 12.0

local function log(message)
  local line = tostring(message)
  reaper.ShowConsoleMsg('[reverse-render] ' .. line .. '\n')
  local file = io.open(logPath, 'a')
  if file then file:write(line .. '\n'); file:close() end
end

local function finish(status, detail)
  local file = io.open(outPath, 'w')
  if file then
    file:write(string.format('{"probe":"reverse-render","status":"%s","detail":"%s"}\n', status, detail))
    file:close()
  end
  log('probe status=' .. status .. ' detail=' .. detail)
end

local truncate = io.open(logPath, 'w')
if truncate then truncate:close() end
log('=== reverse render probe ' .. os.date('%Y-%m-%d %H:%M:%S') .. ' ===')

-- 夹具：与 F-4 探针同一个不对称源（0–1s 173Hz、1–2s 311Hz，另有脉冲）。源长 2.0s。
local function writeFixture()
  local samples = {}
  for n = 0, RATE * 2 - 1 do
    local envelope = n < RATE and 0.2 or 0.07
    local value = envelope * math.sin(2 * math.pi * (n < RATE and 173 or 311) * n / RATE)
    if n >= 7000 and n < 7100 then value = 0.45 end
    samples[#samples + 1] = string.pack('<f', value)
  end
  local pcm = table.concat(samples)
  local file = assert(io.open(wavPath, 'wb'))
  file:write('RIFF', string.pack('<I4', 36 + #pcm), 'WAVEfmt ',
    string.pack('<I4I2I2I4I4I2I2', 16, 3, 1, RATE, RATE * 4, 4, 32), 'data',
    string.pack('<I4', #pcm), pcm)
  file:close()
end
writeFixture()

reaper.InsertTrackAtIndex(0, true)
local track = reaper.GetTrack(0, 0)
reaper.GetSetMediaTrackInfo_String(track, 'P_NAME', 'reverse-render', true)
reaper.SetOnlyTrackSelected(track)

reaper.SetEditCurPos(0.0, false, false)
reaper.InsertMedia(wavPath, 0)
local item = reaper.GetTrackMediaItem(track, reaper.CountTrackMediaItems(track) - 1)
reaper.SetMediaItemInfo_Value(item, 'D_LENGTH', ITEM_LENGTH)
reaper.SetMediaItemInfo_Value(item, 'B_LOOPSRC', 0)
reaper.SetMediaItemInfo_Value(item, 'D_FADEINLEN', 0)
reaper.SetMediaItemInfo_Value(item, 'D_FADEOUTLEN', 0)
reaper.SetMediaItemInfo_Value(item, 'D_FADEINLEN_AUTO', 0)
reaper.SetMediaItemInfo_Value(item, 'D_FADEOUTLEN_AUTO', 0)
reaper.SetMediaItemTakeInfo_Value(reaper.GetActiveTake(item), 'D_STARTOFFS', SOURCE_OFFSET)

-- 官方 action 41051 = "Item properties: Toggle take reverse"。
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
if not reverseAction then finish('error', 'reverse action not found'); return end
reaper.SelectAllMediaItems(0, false)
reaper.SetMediaItemSelected(item, true)
reaper.Main_OnCommand(reverseAction, 0)

local take = reaper.GetActiveTake(item)
local startoffs = reaper.GetMediaItemTakeInfo_Value(take, 'D_STARTOFFS')
local ok, _, _, reversed = reaper.PCM_Source_GetSectionInfo(reaper.GetMediaItemTake_Source(take))
log(string.format('reversed item: startoffs=%.6f section=%s reversed=%s', startoffs, tostring(ok), tostring(reversed)))
if not (ok and reversed) then finish('error', 'reverse not verified'); return end

reaper.UpdateArrange()
local fx = reaper.TrackFX_AddByName(track, 'HiFiShifter', false, -1)
log('TrackFX_AddByName -> ' .. tostring(fx))
if fx < 0 then finish('error', 'HiFiShifter FX not found'); return end

reaper.GetSetProjectInfo(0, 'RENDER_SETTINGS', 0, true)
reaper.GetSetProjectInfo(0, 'RENDER_BOUNDSFLAG', 0, true)
reaper.GetSetProjectInfo(0, 'RENDER_STARTPOS', 0, true)
reaper.GetSetProjectInfo(0, 'RENDER_ENDPOS', ITEM_LENGTH, true)
reaper.GetSetProjectInfo(0, 'RENDER_SRATE', RATE, true)
reaper.GetSetProjectInfo(0, 'RENDER_CHANNELS', 1, true)
reaper.GetSetProjectInfo(0, 'RENDER_TAILFLAG', 0, true)
reaper.GetSetProjectInfo(0, 'RENDER_NORMALIZE', 0, true)
reaper.GetSetProjectInfo_String(0, 'RENDER_FILE', capturesDir, true)
reaper.GetSetProjectInfo_String(0, 'RENDER_PATTERN', 'reverse-render', true)
-- ZXZhdxgAAA== = 24-bit WAV（与 phase3a 同格式，便于逐样本比对）。
reaper.GetSetProjectInfo_String(0, 'RENDER_FORMAT', 'ZXZhdxgAAA==', true)

local renderCommand
do
  local section = reaper.SectionFromUniqueID(0)
  local index = 0
  while true do
    local command, name = reaper.kbd_enumerateActions(section, index)
    if command == 0 then break end
    if name and name:find('Render project, using the most recent render settings', 1, true) then
      if name:find('auto%-close') then renderCommand = command end
    end
    index = index + 1
  end
end
if not renderCommand then finish('error', 'render action not found'); return end

reaper.UpdateArrange()
local deadline = reaper.time_precise() + SETTLE_SECONDS
local function render()
  if reaper.time_precise() < deadline then reaper.defer(render); return end
  reaper.Main_OnCommand(renderCommand, 0)
  finish('ok', 'render command dispatched')
end

reaper.defer(render)
