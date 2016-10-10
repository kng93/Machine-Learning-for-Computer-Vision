function [img] = p2(img, num_bins)
    if size(img,3) == 3
        red_hist = histogram(img(:,:,1), num_bins);
        red_bins = red_hist.Values;
        green_hist = histogram(img(:,:,2), num_bins);
        green_bins = green_hist.Values;
        blue_hist = histogram(img(:,:,3), num_bins);
        blue_bins = blue_hist.Values;
    else
        h = histogram(img(:), num_bins);
        red_bins = h.Values;
        green_bins = h.Values;
        blue_bins =  h.Values;
    end

    % Compile all the data into matrices for knn
    img = transpose([red_bins green_bins blue_bins]);    
end
