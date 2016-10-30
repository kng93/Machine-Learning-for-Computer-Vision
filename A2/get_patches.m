% For p5 - get the patches from the training data
function [patches] = get_patches(img, img_size, patch_size)
    img = imread(fullfile(img.folder, img.name));
    img = imresize(img, [img_size img_size]);

    % Turn the image grey if in colour
    if size(img,3) == 3
        img = rgb2gray(img);
    end
    
    % Set the image up to put into covdet
    img = im2single(img);

    % patches - (size*size, num_patches) -- it is overlapping!
    [frames, patches] = vl_covdet(img, 'descriptor', 'patch', 'PatchResolution', patch_size) ;

    % Normalize the patches (mean 0, std 1) then, bring them to [0, 1]
    % range so that they can be converted to uint8
    patches = patches - mean(patches(:,:));
    patches = patches ./ repmat(std(patches(:,:)), size(patches,1), 1);
    patches = (patches - min(patches)) ./ repmat(max(patches) - min(patches), size(patches,1), 1);
end

