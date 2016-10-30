%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem A1 q3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [img] = col_hist_local(img, num_bins, num_grid)

    % In case the image only has one colour channel (gif)
    if size(img,3) == 1
        img = cat(3, img, img, img);
    end
    
    % Function to be applied is to get the histogram counts
    fun = @(block_struct) histcounts(block_struct.data, num_bins);

    % Apply the histogram on each patch (as opposed to the whole image)
    red_hist = blockproc(img(:,:,1), ...
                [ceil(size(img,1)/num_grid), ceil(size(img,2)/num_grid)], ...
                fun);
    green_hist = blockproc(img(:,:,2), ...
                [ceil(size(img,1)/num_grid), ceil(size(img,2)/num_grid)], ...
                fun);
    blue_hist = blockproc(img(:,:,3), ...
                [ceil(size(img,1)/num_grid), ceil(size(img,2)/num_grid)], ...
                fun);

    % Compile all the data into matrices for knn
    img = [red_hist(:); green_hist(:); blue_hist(:)];   
end
