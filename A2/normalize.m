function [data, mins, ranges] = normalize(data, mins, ranges)
    if nargin < 3
       mins = min(data, [], 1);
       ranges = max(data, [], 1) - mins;
    end
    
    data = (data - repmat(mins,size(data,1),1)) ./ repmat(ranges, size(data, 1), 1);
end