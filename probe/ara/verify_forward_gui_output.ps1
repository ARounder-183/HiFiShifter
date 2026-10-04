# 一次性正向GUI音频验收：检查真实音高差异、间隙及保存重开一致性，不将音量变化视作修音。
param(
    [string]$BaselinePath = "$PSScriptRoot\captures\forward-gui-baseline.wav",
    [string]$EditedPath = "$PSScriptRoot\captures\forward-gui-edited.wav",
    [string]$ReopenedPath = '',
    [string]$SourcePath = "$PSScriptRoot\fixtures\forward-gui-voice.wav",
    [string]$ReportPath = "$PSScriptRoot\captures\forward-gui-output.json",
    [switch]$LibraryOnly
)
$ErrorActionPreference = 'Stop'
if (-not ('AraForwardGuiOutput' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.IO;
public static class AraForwardGuiOutput {
    public sealed class Wave { public int Rate, Channels; public float[] Samples; }
    public sealed class Result {
        public double EditedRms, MeanAbsoluteDifference, GapMax, ReopenMaxDifference;
        public int PitchChangedWindows;
        public double[] BaselineHz, EditedHz;
    }
    // 独立RIFF解析，接受REAPER的24bit PCM与32bit float及extensible子类型。
    public static Wave Read(string path) {
        using(var r = new BinaryReader(File.OpenRead(path))) {
            if(new string(r.ReadChars(4)) != "RIFF") throw new Exception("not RIFF");
            r.ReadUInt32();
            if(new string(r.ReadChars(4)) != "WAVE") throw new Exception("not WAVE");
            int rate=0, channels=0, format=0, bits=0; byte[] data=null;
            while(r.BaseStream.Position+8 <= r.BaseStream.Length) {
                string id=new string(r.ReadChars(4)); int size=checked((int)r.ReadUInt32());
                long end=checked(r.BaseStream.Position+size);
                if(end>r.BaseStream.Length) throw new Exception("truncated chunk");
                if(id=="fmt ") {
                    if(size<16) throw new Exception("short format");
                    format=r.ReadUInt16(); channels=r.ReadUInt16(); rate=r.ReadInt32();
                    r.ReadUInt32(); r.ReadUInt16(); bits=r.ReadUInt16();
                    if(format==65534 && size>=40) { r.ReadUInt16(); r.ReadUInt16(); r.ReadUInt32(); format=r.ReadUInt16(); }
                } else if(id=="data") { data=r.ReadBytes(size); }
                r.BaseStream.Position=end+(size&1);
            }
            if(data==null || rate<=0 || channels<=0 || !(format==3&&bits==32 || format==1&&bits==24)) throw new Exception("unsupported WAV format");
            int width=bits/8;
            if(data.Length%(width*channels)!=0) throw new Exception("partial sample frame");
            var samples=new float[data.Length/width];
            for(int i=0;i<samples.Length;i++) {
                if(format==3) samples[i]=BitConverter.ToSingle(data,i*4);
                else { int v=data[i*3]|data[i*3+1]<<8|data[i*3+2]<<16; if((v&0x800000)!=0)v|=unchecked((int)0xff000000); samples[i]=v/8388608f; }
            }
            return new Wave { Rate=rate, Channels=channels, Samples=samples };
        }
    }
    // 针对此220Hz周期夹具的独立归一化自相关oracle；不使用HiFiShifter音高分析结果。
    private static double Frequency(Wave wave, double seconds) {
        int start=(int)(seconds*wave.Rate), count=(int)(0.08*wave.Rate);
        int low=wave.Rate/1000, high=wave.Rate/60;
        var scores=new double[high+1];
        for(int lag=low;lag<=high;lag++) {
            double xy=0, xx=0, yy=0;
            for(int n=0;n<count;n++) {
                double x=wave.Samples[(start+n)*2], y=wave.Samples[(start+n+lag)*2];
                xy+=x*y; xx+=x*x; yy+=y*y;
            }
            scores[lag]=xx*yy>1e-14 ? xy/Math.Sqrt(xx*yy) : 0;
        }
        for(int lag=low+1;lag<high;lag++) {
            if(scores[lag]>=0.8 && scores[lag]>scores[lag-1] && scores[lag]>=scores[lag+1]) return (double)wave.Rate/lag;
        }
        throw new Exception("no reliable pitch oracle in voiced fixture window");
    }
    private static void Validate(Wave wave) {
        if(wave==null || wave.Rate!=44100 || wave.Channels!=2 || wave.Samples==null || wave.Samples.Length!=44100*5*2) throw new Exception("expected 5 second 44.1k stereo capture");
        foreach(float sample in wave.Samples) if(float.IsNaN(sample)||float.IsInfinity(sample)) throw new Exception("nonfinite PCM");
    }
    // 先确认基线确实是同源普通/裁切布局，避免把旧文件或错位基线当本轮证据。
    public static double CheckBaseline(Wave source, Wave baseline) {
        Validate(baseline);
        if(source==null || source.Rate!=44100 || source.Channels!=1 || source.Samples.Length!=88200) throw new Exception("unexpected source fixture");
        double max=0;
        for(int frame=0;frame<44100*5;frame++) {
            float expected=frame<88200 ? source.Samples[frame] : frame>=132300&&frame<176400 ? source.Samples[frame-132300+11025] : 0;
            if(float.IsNaN(expected)||float.IsInfinity(expected)) throw new Exception("nonfinite source");
            for(int ch=0;ch<2;ch++) max=Math.Max(max,Math.Abs(baseline.Samples[frame*2+ch]-expected));
        }
        if(max>1e-6) throw new Exception("baseline does not match normal/crop/gap source oracle");
        return max;
    }
    // 将未变、仅gain、漏音或重开变化明确拒绝，而非单凭非零/不同哈希通过。
    public static Result Evaluate(Wave baseline, Wave edited, Wave reopened) {
        Validate(baseline); Validate(edited); if(reopened!=null) Validate(reopened);
        var result=new Result { BaselineHz=new double[4], EditedHz=new double[4] };
        double power=0, difference=0; int count=0;
        for(int frame=0;frame<44100*5;frame++) for(int ch=0;ch<2;ch++) {
            int i=frame*2+ch; double value=edited.Samples[i];
            bool active=frame<88200 || frame>=132300&&frame<176400;
            if(active) { power+=value*value; difference+=Math.Abs(value-baseline.Samples[i]); count++; }
            else result.GapMax=Math.Max(result.GapMax,Math.Abs(value));
            if(reopened!=null) result.ReopenMaxDifference=Math.Max(result.ReopenMaxDifference,Math.Abs(value-reopened.Samples[i]));
        }
        result.EditedRms=Math.Sqrt(power/count); result.MeanAbsoluteDifference=difference/count;
        if(result.EditedRms<=0.01 || result.MeanAbsoluteDifference<=0.01) throw new Exception("silent or unchanged edit output");
        if(result.GapMax>1e-6) throw new Exception("nonzero gap output");
        if(result.ReopenMaxDifference>1e-6) throw new Exception("reopened output does not preserve edit");
        double[] times={0.2,0.5,1.1,1.4};
        for(int n=0;n<times.Length;n++) {
            result.BaselineHz[n]=Frequency(baseline,times[n]); result.EditedHz[n]=Frequency(edited,times[n]);
            if(Math.Abs(12*Math.Log(result.EditedHz[n]/result.BaselineHz[n],2))>=0.5) result.PitchChangedWindows++;
        }
        if(result.PitchChangedWindows==0) throw new Exception("waveform differs but pitch unchanged; gain-only is not pitch acceptance");
        return result;
    }
}
'@
}
if ($LibraryOnly) { return }
$baseline = [AraForwardGuiOutput]::Read($BaselinePath)
$baselineError = [AraForwardGuiOutput]::CheckBaseline([AraForwardGuiOutput]::Read($SourcePath), $baseline)
$edited = [AraForwardGuiOutput]::Read($EditedPath)
$reopened = if ($ReopenedPath) { [AraForwardGuiOutput]::Read($ReopenedPath) } else { $null }
$result = [AraForwardGuiOutput]::Evaluate($baseline, $edited, $reopened)
$report = [ordered]@{
    baseline_plain_max_abs_difference = $baselineError
    edited_rms = $result.EditedRms
    mean_absolute_difference = $result.MeanAbsoluteDifference
    gap_max_abs = $result.GapMax
    baseline_pitch_hz = $result.BaselineHz
    edited_pitch_hz = $result.EditedHz
    pitch_changed_windows = $result.PitchChangedWindows
    reopen_checked = [bool]$ReopenedPath
    reopen_max_abs_difference = $result.ReopenMaxDifference
    baseline_sha256 = (Get-FileHash -LiteralPath $BaselinePath -Algorithm SHA256).Hash
    source_sha256 = (Get-FileHash -LiteralPath $SourcePath -Algorithm SHA256).Hash
    edited_sha256 = (Get-FileHash -LiteralPath $EditedPath -Algorithm SHA256).Hash
}
if ($ReopenedPath) { $report.reopened_sha256 = (Get-FileHash -LiteralPath $ReopenedPath -Algorithm SHA256).Hash }
$report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ReportPath -Encoding utf8
$report
