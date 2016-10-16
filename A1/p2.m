function [img] = p2(img, num_bins)

    % In case the image only has one colour channel (gif)
    if size(img,3) == 1
        img = cat(3, img, img, img);
    end

    % Create the histogram
    % TODO: Test --- histcounts replaced histogram
    red_bins = histcounts(img(:,:,1), num_bins);
    %red_bins = red_hist.Values;
    
    green_bins = histcounts(img(:,:,2), num_bins);
    %green_bins = green_hist.Values;
    
    blue_bins = histcounts(img(:,:,3), num_bins);
    %blue_bins = blue_hist.Values;

    % Compile all the data into matrices for knn
    img = transpose([red_bins green_bins blue_bins]);    
end
