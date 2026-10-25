#pragma once
#include <algorithm>
#include <vector>
#include <limits>

namespace hachi
{
// Coordinates are in the editor's displayed units (seconds and linear gain,
// MIDI or FLAG value). Intersect limits before moving, preserving group shape.
struct PointGroupCoordinate { double time=0,value=0; };
struct PointGroupDelta
{
    double minTime=-std::numeric_limits<double>::infinity(),maxTime=std::numeric_limits<double>::infinity();
    double minValue=-std::numeric_limits<double>::infinity(),maxValue=std::numeric_limits<double>::infinity();
    void constrain(const std::vector<PointGroupCoordinate>& points,const std::vector<int>& selected,
                   double first,double last,double low,double high,double separation,bool fixedEndpoints=false)
    {
        const auto contains=[&](int i){return std::find(selected.begin(),selected.end(),i)!=selected.end();};
        for(const int i:selected)
        {
            if(i<0||i>=static_cast<int>(points.size()))continue;
            const auto p=points[static_cast<size_t>(i)];
            auto left=first,right=last;
            if(i>0&&!contains(i-1))left=std::max(left,points[static_cast<size_t>(i-1)].time+separation);
            if(i+1<static_cast<int>(points.size())&&!contains(i+1))right=std::min(right,points[static_cast<size_t>(i+1)].time-separation);
            if(fixedEndpoints&&(i==0||i+1==static_cast<int>(points.size())))left=right=p.time;
            minTime=std::max(minTime,std::min(0.0,left-p.time));
            maxTime=std::min(maxTime,std::max(0.0,right-p.time));
            minValue=std::max(minValue,std::min(0.0,low-p.value));
            maxValue=std::min(maxValue,std::max(0.0,high-p.value));
        }
    }
    PointGroupCoordinate clamp(double time,double value) const
    {return {std::clamp(time,minTime,maxTime),std::clamp(value,minValue,maxValue)};}
};
}
