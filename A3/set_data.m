% Set the data in the train and test arrays
function [codewords, train_data, test_data] = set_data(prob_num, ...
                                            cat_subdirs, ...
                                            partition_array, ...
                                            colour, ...
                                            total_train, ...
                                            total_test, ...
                                            bin_val, ...
                                            part_size, ...
                                            feature_val)
                                        
    codewords = []; % Default is empty because only p5 uses this
    
    % Initialization of data and problem-specific information
    if (prob_num == 4)
        % Get a hog result to get the hog size...
        % feature_val == number of orientations
        imgs = dir(fullfile(cat_subdirs(1).folder, cat_subdirs(1).name, '*.jpg'));
        img = imread(fullfile(imgs(1).folder,imgs(1).name));
        img = hog(img, bin_val, part_size, feature_val);
        data_length = size(img ,1);
    elseif (prob_num == 5)
        % feature_val == vocab_size
        data_length = feature_val;
        codewords = create_vocab(partition_array, cat_subdirs, bin_val, part_size, feature_val);
    else
        data_length = bin_val*part_size*part_size*colour;
    end
    
    % Add one for the category designation
    train_data = zeros(total_train, data_length + 1);
    test_data = zeros(total_test, data_length + 1);
    % Keep track of where in the train/test matrix we are
    train_count = 1;
    test_count = 1;
    
    % Loop over the categories
    for category_idx = 1:numel(cat_subdirs)
        category = cat_subdirs(category_idx);
        
        % Loop over the images
        imgs = dir(fullfile(category.folder, category.name, '*.jpg'));
        for img_idx = 1:numel(imgs)
            img = imgs(img_idx);
            img = imread(fullfile(img.folder, img.name));
            
            % Process the image differently depending on the problem
            switch prob_num
                case 1
                    img = pixel_col(img, bin_val);
                case 2
                    img = col_hist_global(img, bin_val);
                case 3
                    img = col_hist_local(img, bin_val, part_size);
                case 4
                    img = hog(img, bin_val, part_size, feature_val);
                case 5
                    img = patch_based(img, codewords, bin_val, part_size, feature_val);
            end
            
            % If part of training dataset
            if partition_array{category_idx}(img_idx)
                train_data(train_count, :) = [img; category_idx];
                train_count = train_count + 1;
            % If part of testing dataset
            else
                test_data(test_count, :) = [img; category_idx];
                test_count = test_count + 1;
            end
        end
    end
end
