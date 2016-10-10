function [img] = p5(img, codewords, scale_size, patch_size, vocab_size)

    % get_patches code
    img = imresize(img, [scale_size scale_size]);

    % Turn the image grey if in colour
    if size(img,3) == 3
        img = rgb2gray(img);
    end
    
    % Set the image up to put into covdet
    img = im2single(img);

    % patches - (size*size, num_patches) -- it is overlapping!
    [frames, patches] = vl_covdet(img, 'descriptor', 'patch', 'PatchResolution', patch_size) ;

    % Normalize the patches (mean 0, std 1)
    % patches = patches - mean(patches(:,:));
    % patches = patches ./ repmat(std(patches(:,:)), size(patches,1), 1);
    
    cluster_idxs = vl_ikmeanspush(uint8(patches * 255), int32(codewords * 255));
    img = transpose(histcounts(cluster_idxs, vocab_size));
end