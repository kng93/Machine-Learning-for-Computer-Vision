function [codeword_centers] = create_vocab(partition_array, cat_subdirs, img_size, patch_size, vocab_size)
    
    % Get number of images used to get the patches (10% training)
    num_patches = sum(ceil(cellfun(@sum, partition_array)*0.1));
    all_patches = cell(num_patches, 1);
    count_allsample = 1;

    % Loop over the categories
    for cat_idx = 1:numel(cat_subdirs)
        category = cat_subdirs(cat_idx);
        num_subsample_train = ceil(sum([partition_array{cat_idx}]) * 0.1);
        count_subsample = 0;
        
        % Loop over the images
        imgs = dir(fullfile(category.folder, category.name, '*.jpg'));
        for img_idx = 1:numel(imgs)
            % If part of training dataset
            if partition_array{cat_idx}(img_idx)
                patches = get_patches(imgs(img_idx), img_size, patch_size);
                all_patches(count_allsample) = {transpose(patches)};
                
                count_subsample = count_subsample + 1;
                count_allsample = count_allsample + 1;
                % Only use as 10% images
                if count_subsample >= num_subsample_train
                    break;
                end
            end
        end
    end
    
    % Put all the samples together in a matrix
    all_patches = transpose(cell2mat(all_patches));
    [codeword_centers, assignments] = vl_kmeans(all_patches, vocab_size);
end