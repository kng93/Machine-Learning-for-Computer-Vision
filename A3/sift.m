%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIFT features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [img] = sift(img, scale_size)

    % In case the image only has one colour channel (gif)
    if size(img,3) == 1
        img = cat(3, img, img, img);
    end

    % Resize image
    img = imresize(img, [scale_size scale_size]);
    img = single(rgb2gray(img));
    [f,d] = vl_dsift(img) ;
    
    % Compile all the data into matrices for knn
    img = d(:);   
end