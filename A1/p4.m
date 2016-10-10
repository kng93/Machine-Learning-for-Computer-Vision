function [img] = p4(img, scale_size, num_cell, orient_num)
    % Resize image
    img = imresize(img, [scale_size scale_size]);
    hog = vl_hog(im2single(img), num_cell, 'numOrientations', orient_num);

    % Compile all the data into matrices for knn
    img = hog(:);   
end