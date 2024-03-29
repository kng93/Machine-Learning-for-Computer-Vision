%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem 1b
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [img] = p1b(img, feature_size)
    % Image size is the square of feature size (defined in run_a1.m)
    img_size = floor(sqrt(feature_size));

    % Resize image
    img = imresize(img, [img_size img_size]);
    
    if size(img,3) == 1
        img = cat(3, img, img, img);
    end
    
    % Flatten image into a long vector
    img = img(:);
end
