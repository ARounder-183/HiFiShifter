--[[ 中文一次性REAPER验收：仅从隔离启动入口运行；GUI参数编辑必须由Computer Use执行。 ]]
local scratch=assert(os.getenv('HIFISHIFTER_ARA_PROBE_DIR'),'isolated scratch required')
local _,project=reaper.EnumProjects(-1,'')
assert(project:lower()==(scratch..'\\embedded-editor.RPP'):lower(),'wrong project: never modify a user project')
local function log(text)
  local file=assert(io.open(scratch..'\\acceptance-script.log','a'));file:write(text..'\n');file:close()
end
-- 中文：只读实际宿主状态，时间格式为秒，不写源/参数/片段几何。
local function observe()
  local file=assert(io.open(scratch..'\\host-state.txt','w'))
  file:write(string.format('playing=%d\nposition_sec=%.9f\nbpm=%.9f\ntracks=%d\nproject=%s\n',
    reaper.GetPlayState(),reaper.GetPlayPosition(),reaper.Master_GetTempo(),reaper.CountTracks(0),project))
  for index=0,reaper.CountTracks(0)-1 do
    local track=reaper.GetTrack(0,index);local _,name=reaper.GetTrackName(track)
    file:write(string.format('track%d=%s solo=%.0f items=%d fx=%d\n',index+1,name,
      reaper.GetMediaTrackInfo_Value(track,'I_SOLO'),reaper.CountTrackMediaItems(track),reaper.TrackFX_GetCount(track)))
    for item_index=0,reaper.CountTrackMediaItems(track)-1 do
      local item=reaper.GetTrackMediaItem(track,item_index);local take=reaper.GetActiveTake(item)
      file:write(string.format('item%d_%d start=%.9f duration=%.9f rate=%.9f fade_in=%.9f fade_out=%.9f\n',index+1,item_index+1,
        reaper.GetMediaItemInfo_Value(item,'D_POSITION'),reaper.GetMediaItemInfo_Value(item,'D_LENGTH'),
        take and reaper.GetMediaItemTakeInfo_Value(take,'D_PLAYRATE') or 0,
        reaper.GetMediaItemInfo_Value(item,'D_FADEINLEN'),reaper.GetMediaItemInfo_Value(item,'D_FADEOUTLEN')))
    end
  end
  file:close()
end
local render_action
for index=0,10000 do
  local id,name=reaper.kbd_enumerateActions(reaper.SectionFromUniqueID(0),index)
  if id==0 then break end
  if name and name:find('Render project, using the most recent render settings',1,true) and name:find('auto%-close') then render_action=id end
end
assert(render_action)
-- 中文：只导出当前真实DAW音频，不脚本设置HiFiShifter参数或更改宿主轨道。
local function render(label)
  reaper.GetSetProjectInfo(0,'RENDER_SETTINGS',0,true);reaper.GetSetProjectInfo(0,'RENDER_BOUNDSFLAG',0,true)
  reaper.GetSetProjectInfo(0,'RENDER_STARTPOS',0,true);reaper.GetSetProjectInfo(0,'RENDER_ENDPOS',6,true)
  reaper.GetSetProjectInfo(0,'RENDER_SRATE',44100,true);reaper.GetSetProjectInfo(0,'RENDER_CHANNELS',2,true)
  reaper.GetSetProjectInfo(0,'RENDER_TAILFLAG',0,true);reaper.GetSetProjectInfo(0,'RENDER_NORMALIZE',0,true)
  reaper.GetSetProjectInfo_String(0,'RENDER_FILE',scratch,true)
  reaper.GetSetProjectInfo_String(0,'RENDER_FORMAT','ZXZhdxgAAA==',true)
  reaper.GetSetProjectInfo_String(0,'RENDER_PATTERN','acceptance-'..label,true)
  reaper.Main_OnCommand(render_action,0);log('render returned '..label)
end
local shown=os.getenv('HIFISHIFTER_ARA_ACCEPTANCE_NO_UI')=='1'
local show_at=reaper.time_precise()+3;local next_observe=0;local previous=''
local function tick()
  local now=reaper.time_precise()
  if not shown and now>show_at then
    shown=true;local track=reaper.GetTrack(0,0);local fx=track and reaper.TrackFX_GetByName(track,'HiFiShifter',false) or -1
    if fx>=0 then reaper.TrackFX_Show(track,fx,3);log('first native FX shown') end
  end
  if now>next_observe then next_observe=now+0.25;observe() end
  local file=io.open(scratch..'\\command.txt','r');local command=file and file:read('*l') or '';if file then file:close() end
  if command~='' and command~=previous then
    previous=command;local label=command:match('^render:([a-z0-9-]+)$')
    if label then render(label) else log('unknown command ignored '..command) end
  end
  reaper.defer(tick)
end
log('isolated project opened '..project);reaper.defer(tick)
