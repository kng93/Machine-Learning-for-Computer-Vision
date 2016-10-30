% Test the model (on the test data set)
function [] = test_model(test_data, mdl)
    % Split up the result from the data
    test_res = test_data(:, end);
    test_vals = test_data(:, 1:(end-1));
    
    % Go over every test value
    num_correct = 0;
    for test_idx = 1:size(test_vals,1)
       img_class = predict(mdl, test_vals(test_idx,:));
       
       % Count the number of correct values
       if img_class == test_res(test_idx)
          num_correct = num_correct + 1; 
       end
    end
    
    % Print out the correct percentage
    fprintf('correct_test = %f\n', (num_correct/size(test_vals,1)));
end


% For p5 - create the vocabulary
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
