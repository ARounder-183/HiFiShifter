--[[
  query_tracks.lua —— 只读诊断脚本

  作用：把当前 REAPER 实例的轨道清单与 FX 名写到 captures\tracks.log，
        用于判断某个实例里到底有什么。**不做任何修改。**
]]

local sep = package.config:sub(1, 1)
local scriptPath = debug.getinfo(1, 'S').source:sub(2)
local scriptDir = scriptPath:match('^(.*)' .. sep .. '[^' .. sep .. ']*$') or '.'
local logPath = scriptDir .. sep .. 'captures' .. sep .. 'tracks.log'

local lines = {}
local function log(m) lines[#lines + 1] = tostring(m) end

log('=== tracks @ ' .. os.date('%Y-%m-%d %H:%M:%S') .. ' ===')
log('project filename: ' .. tostring(reaper.GetProjectName(0, '')))
local _, projPath = reaper.EnumProjects(-1, '')
log('project path: ' .. tostring(projPath))
log('is dirty: ' .. tostring(reaper.IsProjectDirty(0)))
log('track count = ' .. reaper.CountTracks(0))

for i = 0, reaper.CountTracks(0) - 1 do
  local tr = reaper.GetTrack(0, i)
  local _, nm = reaper.GetSetMediaTrackInfo_String(tr, 'P_NAME', '', false)
  local fxCount = reaper.TrackFX_GetCount(tr)
  local itemCount = reaper.CountTrackMediaItems(tr)
  log(string.format('  [%d] name=%-14s items=%d fx=%d', i, tostring(nm), itemCount, fxCount))
  for f = 0, fxCount - 1 do
    local _, fxName = reaper.TrackFX_GetFXName(tr, f, '')
    log(string.format('        fx[%d] = %s', f, tostring(fxName)))
  end
end

local fh = io.open(logPath, 'w')
if fh then fh:write(table.concat(lines, '\n') .. '\n'); fh:close() end
