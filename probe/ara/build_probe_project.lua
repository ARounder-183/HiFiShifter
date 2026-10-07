--[[
  build_probe_project.lua —— HiFiShifter ARA 探针：搭建采集用工程

  作用：用 ReaScript 以确定性的方式把"挂载 ARA 插件 + 摆放素材"这件事做完，
        替代在 REAPER 图形界面里手工点击。跑完后 ARA 插件应已激活，
        并按其插桩逻辑把模型图 dump 到 ARA_PROBE_OUT。

  用法：
    reaper.exe -new build_probe_project.lua

  特殊说明：
    - 素材路径取自脚本同目录下的 fixtures\。路径里的反斜杠与中文无关，纯 ASCII。
    - 插件名 "ARATestPlugIn" 必须与 VST3 扫描后 REAPER 显示的名字一致。
    - 本脚本是一次性探针产物。
]]

local sep = package.config:sub(1, 1)
local scriptPath = debug.getinfo(1, 'S').source:sub(2)
local scriptDir = scriptPath:match('^(.*)' .. sep .. '[^' .. sep .. ']*$') or '.'
local fixturesDir = scriptDir .. sep .. 'fixtures'
local capturesDir = scriptDir .. sep .. 'captures'

local function log(msg)
  reaper.ShowConsoleMsg('[probe] ' .. tostring(msg) .. '\n')
end

log('script dir = ' .. scriptDir)

-- 清空当前工程（-new 已给空工程，这里是幂等保护）
reaper.Main_OnCommand(40023, 0) -- File: New project

local function addToneWithPlugin(wavName, pluginName, label)
  local wavPath = fixturesDir .. sep .. wavName
  local f = io.open(wavPath, 'rb')
  if not f then
    log('MISSING fixture: ' .. wavPath)
    return nil
  end
  f:close()

  local trackIndex = reaper.CountTracks(0)
  reaper.InsertTrackAtIndex(trackIndex, true)
  local track = reaper.GetTrack(0, trackIndex)
  reaper.GetSetMediaTrackInfo_String(track, 'P_NAME', label, true)

  -- 摆放素材：在 0 秒处插入 2 秒
  reaper.SetOnlyTrackSelected(track)
  reaper.InsertMedia(wavPath, 0)

  -- 挂载 ARA 插件
  local fxIndex = reaper.TrackFX_AddByName(track, pluginName, false, -1)
  log(label .. ': wav=' .. wavName .. ' fxIndex=' .. tostring(fxIndex))

  if fxIndex < 0 then
    log('  !! TrackFX_AddByName failed for "' .. pluginName .. '" — 插件未安装或未被扫描到')
    return track
  end

  local _, fxName = reaper.TrackFX_GetFXName(track, fxIndex, '')
  log('  fx[0] name = ' .. tostring(fxName))

  -- 报告该 FX 是否以 ARA 方式激活（不是所有 REAPER 版本都暴露这一点，
  -- 拿不到就只记录 FX 是否成功插入）
  return track
end

addToneWithPlugin('tone44100.wav', 'ARATestPlugIn', 'probe-44k')
addToneWithPlugin('tone48000.wav', 'ARATestPlugIn', 'probe-48k')

reaper.UpdateArrange()

-- 触发一次保存，让宿主把文档状态走完（部分宿主在保存时才完成 ARA 文档同步）
reaper.Main_OnCommand(40026, 0) -- File: Save project

local dumpPath = os.getenv('ARA_PROBE_OUT')
log('ARA_PROBE_OUT = ' .. tostring(dumpPath))
if dumpPath then
  local d = io.open(dumpPath, 'rb')
  if d then
    local size = d:seek('end')
    d:close()
    log('DUMP EXISTS, size = ' .. tostring(size) .. ' bytes')
  else
    log('DUMP NOT WRITTEN (plugin may not have been activated as ARA)')
  end
end

log('done')
