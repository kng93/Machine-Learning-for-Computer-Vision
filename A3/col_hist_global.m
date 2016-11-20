%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem A1 q2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [img] = col_hist_global(img, num_bins)

    % In case the image only has one colour channel (gif)
    if size(img,3) == 1
        img = cat(3, img, img, img);
    end

    % Create the histogram
    red_bins = histcounts(img(:,:,1), num_bins);
    green_bins = histcounts(img(:,:,2), num_bins);
    blue_bins = histcounts(img(:,:,3), num_bins);

    % Compile all the data into matrices for knn
    img = transpose([red_bins green_bins blue_bins]);    
end
