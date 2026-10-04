# 一次性 WAV 输出验收：解析 RIFF chunks 后比较普通/裁切/真倒放；不能凭非零声称通过。
param(
    [string]$SourcePath = "$PSScriptRoot\fixtures\phase3a-asymmetric.wav",
    [string]$OutputPath = "$PSScriptRoot\captures\phase3a-output.wav",
    [string]$ReportPath = "$PSScriptRoot\captures\phase3a-output.json",
    [string]$ScriptLogPath = "$PSScriptRoot\captures\phase3a-script.log"
)
$ErrorActionPreference='Stop'
if (-not ('AraOutputProbe' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Collections.Generic;
public static class AraOutputProbe {
    public sealed class Wave { public int Rate, Channels; public float[] Samples; }
    public static Wave Read(string path) {
        using(var reader = new BinaryReader(File.OpenRead(path))) {
            if (new string(reader.ReadChars(4)) != "RIFF") throw new Exception("not RIFF");
            reader.ReadUInt32();
            if (new string(reader.ReadChars(4)) != "WAVE") throw new Exception("not WAVE");
            int channels=0, rate=0, format=0, bits=0; byte[] data=null;
            while(reader.BaseStream.Position+8 <= reader.BaseStream.Length) {
                string id=new string(reader.ReadChars(4)); int size=checked((int)reader.ReadUInt32());
                long stop=reader.BaseStream.Position+size;
                if(stop>reader.BaseStream.Length) throw new Exception("truncated chunk");
                if(id=="fmt ") { format=reader.ReadUInt16(); channels=reader.ReadUInt16(); rate=reader.ReadInt32(); reader.ReadUInt32(); reader.ReadUInt16(); bits=reader.ReadUInt16();
                    if(format==65534 && size>=40) { reader.ReadUInt16(); reader.ReadUInt16(); reader.ReadUInt32(); format=reader.ReadUInt16(); }
                } else if(id=="data") { data=reader.ReadBytes(size); }
                reader.BaseStream.Position=stop+(size&1);
            }
            if(data==null || rate<=0 || channels<=0 || !(format==3 && bits==32 || format==1 && bits==24)) throw new Exception("unsupported WAV format");
            int width=bits/8; if(data.Length%(width*channels)!=0) throw new Exception("partial frame");
            float[] pcm=new float[data.Length/width];
            for(int i=0;i<pcm.Length;i++) {
                if(format==3) pcm[i]=BitConverter.ToSingle(data,i*width);
                else { int v=data[i*3]|data[i*3+1]<<8|data[i*3+2]<<16; if((v&0x800000)!=0)v|=unchecked((int)0xff000000); pcm[i]=v/8388608f; }
                if(float.IsNaN(pcm[i])||float.IsInfinity(pcm[i]))throw new Exception("nonfinite PCM");
            }
            return new Wave { Rate=rate, Channels=channels, Samples=pcm };
        }
    }
    public static double[] Compare(Wave source, Wave output) {
        if(source.Rate!=44100 || source.Channels!=1 || source.Samples.Length!=88200 || output.Rate!=source.Rate || output.Channels!=2 || output.Samples.Length!=44100*8*2) throw new Exception("unexpected capture geometry");
        double[] error=new double[5];
        for(int frame=0;frame<44100*8;frame++) {
            int category=4; float expected=0, forward=0;
            if(frame<88200) { category=0; expected=source.Samples[frame]; }
            else if(frame>=132300 && frame<176400) { category=1; expected=source.Samples[frame-132300+11025]; }
            else if(frame>=264600) { category=2; forward=source.Samples[frame-264600]; expected=source.Samples[88199-(frame-264600)]; }
            for(int channel=0;channel<2;channel++) {
                double actual=output.Samples[frame*2+channel];
                error[category]=Math.Max(error[category], Math.Abs(actual-expected));
                if(category==2) error[3]=Math.Max(error[3], Math.Abs(actual-forward));
            }
        }
        return error;
    }
    public static void Write(string path, Wave wave) {
        using(var writer=new BinaryWriter(File.Create(path))) {
            writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF")); writer.Write(36+wave.Samples.Length*4);
            writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVEfmt ")); writer.Write(16); writer.Write((short)3); writer.Write((short)wave.Channels);
            writer.Write(wave.Rate); writer.Write(wave.Rate*wave.Channels*4); writer.Write((short)(wave.Channels*4)); writer.Write((short)32);
            writer.Write(System.Text.Encoding.ASCII.GetBytes("data")); writer.Write(wave.Samples.Length*4);
            foreach(float sample in wave.Samples) writer.Write(sample);
        }
    }
    public static void Mutate(Wave source, Wave output, string kind) {
        for(int i=0;i<88200;i++) {
            output.Samples[(264600+i)*2]=source.Samples[88199-i];
            output.Samples[(264600+i)*2+1]=source.Samples[88199-i];
        }
        if(kind=="duplicate-gain") for(int i=0;i<output.Samples.Length;i++) output.Samples[i]*=2;
        if(kind=="silent") Array.Clear(output.Samples,0,output.Samples.Length);
        if(kind=="forward-reverse") for(int i=0;i<88200;i++) output.Samples[(264600+i)*2]=output.Samples[(264600+i)*2+1]=source.Samples[i];
        if(kind=="wrong-seek") output.Samples[132300*2]=0.7f;
        if(kind=="truncated") Array.Resize(ref output.Samples,output.Samples.Length-2);
    }
}
'@
}
$scriptLog=Get-Content -Raw -LiteralPath $ScriptLogPath
if (-not $scriptLog.Contains('reverse verified: section=true reversed=true')) { throw 'Official reverse action was not verified by host section reader' }
$source=[AraOutputProbe]::Read($SourcePath)
$output=[AraOutputProbe]::Read($OutputPath)
$errors=[AraOutputProbe]::Compare($source,$output)
$report=[ordered]@{
    normal_max_abs_error=$errors[0]; crop_max_abs_error=$errors[1]
    reverse_vs_reversed_max_abs_error=$errors[2]; reverse_vs_forward_max_abs_error=$errors[3]
    gap_max_abs_error=$errors[4]; normal_pass=($errors[0] -le 1e-6)
    crop_pass=($errors[1] -le 1e-6); reverse_pass=($errors[2] -le 1e-6)
    reverse_output_direction=$(if($errors[2] -le 1e-6){'reversed'}elseif($errors[3] -le 1e-6){'forward'}else{'neither'})
    output_sha256=(Get-FileHash -LiteralPath $OutputPath -Algorithm SHA256).Hash
    source_sha256=(Get-FileHash -LiteralPath $SourcePath -Algorithm SHA256).Hash
    reversed_verified=$true
}
$report | ConvertTo-Json | Set-Content -LiteralPath $ReportPath -Encoding utf8
$report
if (-not $report.normal_pass -or -not $report.crop_pass -or $errors[4] -gt 1e-6) { throw 'Normal/crop/gap output did not match oracle' }
if (-not $report.reverse_pass) { throw "Reverse output was $($report.reverse_output_direction), not reversed" }
