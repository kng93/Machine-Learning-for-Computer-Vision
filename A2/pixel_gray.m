%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem A1 1a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [img] = pixel_gray(img, feature_size)
    % Image size is the square of feature size (defined in run_a1.m)
    img_size = floor(sqrt(feature_size));

    % Resize image
    img = imresize(img, [img_size img_size]);
    
    % Turn the image grey if in colour
    if size(img,3) == 3
        img = rgb2gray(img);
    end
    
    % Flatten image into a long vector
    img = img(:);
end