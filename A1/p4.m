function [img] = p4(img, scale_size, num_cell, orient_num)

    % In case the image only has one colour channel (gif)
    if size(img,3) == 1
        img = cat(3, img, img, img);
    end

    % Resize image
    img = imresize(img, [scale_size scale_size]);
    hog = vl_hog(im2single(img), num_cell, 'numOrientations', orient_num);

    % Compile all the data into matrices for knn
    img = hog(:);   
end