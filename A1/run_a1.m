run '/Users/karinng/Documents/SoftwareFiles/Matlab/vlfeat-0.9.20/toolbox/vl_setup'
DB_DIR = '/Users/karinng/Documents/1UniHW/cs9840/Assignments/A1/TenCategories';
TRAIN_PARTITION = 0.8;
PROBLEMS = [1.1, 1.2, 2, 3, 4, 5];
GREYSCALE = 1;
COLOUR = 3;
RANDOMIZE_DATA = true(1);

[cat_subdirs, partition_array] = get_dataset_info(DB_DIR, TRAIN_PARTITION, RANDOMIZE_DATA);

for prob = PROBLEMS
    switch prob
        case 1.1
        % bin_vals: every pixel (resize img to 50x50)
        % partition_val: no partitions
        % feature_val: no extra feature values
        % colour: greyscale
            IMG_SIZE = 50;
            k_vals = [2, 4, 8, 16, 32];

            run_problem(prob, ...
                        cat_subdirs, ...
                        partition_array, ...
                        [IMG_SIZE * IMG_SIZE], ...
                        [1], ...
                        [1], ...
                        k_vals, ...
                        GREYSCALE);
        case 1.2
        % bin_vals: every pixel (resize img to 50x50)
        % partition_val: no partitions
        % feature_val: no extra feature values
        % colour: rgb
            IMG_SIZE = 50;
            k_vals = [2, 4, 8, 16, 32];

            run_problem(prob, ...
                        cat_subdirs, ...
                        partition_array, ...
                        [IMG_SIZE * IMG_SIZE], ...
                        [1], ...
                        [1], ...
                        k_vals, ...
                        COLOUR);
        case 2
        % bin_vals: 50
        % partition_val: no partitions
        % feature_val: no extra feature values
        % colour: rgb
            NUM_BINS = 50;
            k_vals = [2, 4, 8, 16, 32];

            run_problem(prob, ...
                        cat_subdirs, ...
                        partition_array, ...
                        [NUM_BINS], ...
                        [1], ...
                        [1], ...
                        k_vals, ...
                        COLOUR);
        case 3
        % bin_vals: changing bins for local histogram
        % partition_val: grid_vals -- grid of 1x1, 2x2, 3x3, 4x4, 5x5
        % feature_val: no extra feature values
        % colour: rgb
            bin_vals = [10, 20, 30, 40];
            grid_vals = [1, 2, 3, 4, 5];
            k_vals = [2, 4, 8, 16, 32];

            run_problem(prob, ...
                        cat_subdirs, ...
                        partition_array, ...
                        bin_vals, ...
                        grid_vals, ...
                        [1], ...
                        k_vals, ...
                        COLOUR);
        case 4
        % bin_vals: imgsize_vals -- image resize values
        % partition_val: cell_vals -- cells to split by
        % feature_val: num_orients -- number of orientations
        % colour: rgb (not used)
            imgsize_vals = [50, 150, 250];
            cell_vals = [10, 30, 40];
            num_orients = [4, 6, 8, 10, 12, 14];
            k_vals = [2, 4, 8, 16, 32];

            run_problem(prob, ...
                        cat_subdirs, ...
                        partition_array, ...
                        imgsize_vals, ...
                        cell_vals, ...
                        num_orients, ...
                        k_vals, ...
                        COLOUR);
        case 5
        % bin_vals: imgsize_vals -- image resize values
        % partition_val: patch_vals -- size of the patch
        % feature_val: vocab_size -- size of the vocabulary
        % colour: rgb (not used)
            imgsize_vals = [250];
            patch_vals = [10, 20, 30, 40];
            vocab_size = [50, 100, 150, 200];
            k_vals = [2, 4, 8, 16, 32];

            run_problem(prob, ...
                        cat_subdirs, ...
                        partition_array, ...
                        imgsize_vals, ...
                        patch_vals, ...
                        vocab_size, ...
                        k_vals, ...
                        COLOUR);
    end
end


% Get information about each category and get train/test indices
function [cat_subdirs, partition_array] = get_dataset_info(DB_DIR, train_part, random_parts)
    % Get the category subdirectories (ignoring this and previous dir)
    cat_subdirs = dir(DB_DIR);
    cat_subdirs = cat_subdirs(~ismember({cat_subdirs.name}, {'.', '..'}));

    % Set up partition array (how each category will be partitioned)
    partition_array = cell(numel(cat_subdirs), 1);

    for cat_idx = 1:numel(cat_subdirs)
        % Counting images to be used later
        category = cat_subdirs(cat_idx);
        imgs = dir(fullfile(category.folder, category.name, '*.jpg'));
        num_imgs = numel(imgs);
        % total_img_num = total_img_num + num_imgs;

        % Partitioning the categories into train/test (80% train, 20% test)
        num_train = ceil(num_imgs * train_part);
        partition_array(cat_idx) = {[ones(1,num_train) zeros(1, (num_imgs-num_train))]};

        % If want to have random test values
        if random_parts
            shuff_idx = randperm(num_imgs);
            partition_array(cat_idx) = {partition_array{cat_idx}(shuff_idx)};
        end
    end
end


% Control function to run the problems
function [] = run_problem(prob_num, ...
                            cat_subdirs, ...
                            partition_array, ...
                            bin_vals, ...
                            part_vals, ...
                            feature_vals, ...
                            k_vals, ...
                            colour)
    % Set up the train and test arrays
    total_train = sum([partition_array{:}]);
    total_test = sum([partition_array{:}] == 0);
    % 1=loss, 2=mdl, 3=test_data, 4=k, 5=bin, 6=partition, 7=feature, 
    % 8=codewords
    min_vals = cell(8, 1);
    min_vals{1} = 1; % loss initialized for comparison

    % Looping over features to test various values
    for feature_val = feature_vals
        for bin_val = bin_vals
            for part_val = part_vals
                
                [codewords, train_data, test_data] = set_data(prob_num, ...
                                                    cat_subdirs, ...
                                                    partition_array, ...
                                                    colour, ...
                                                    total_train, ...
                                                    total_test, ...
                                                    bin_val, ...
                                                    part_val, ...
                                                    feature_val);

                [mdl, loss, k]= train_model(prob_num, k_vals, train_data, test_data);
                
                % Keep track of the lowest loss and the model associated with it
                if min_vals{1} > loss
                    min_vals = {loss; mdl; test_data; k; bin_val; ...
                                part_val; feature_val; codewords};
                end

                % Print out intermediate test results
                fprintf('CHECK TEST!! k: %d, bin_size: %d, part_size: %d, feature_val: %d, ', ...
                        k, bin_val, part_val, feature_val);
                test_model(test_data, mdl);
            end
        end
    end
    
    % Final test run
    fprintf('FINAL RESULT (k = %d, bin_val = %d, part_size = %d, feature_val = %d):\n', ...
            min_vals{4}, min_vals{5}, min_vals{6}, min_vals{7}); 
    test_model(min_vals{3}, min_vals{2});
    
    % Print out codewords
    if size(min_vals{8}, 1) > 0
        patch_size = sqrt(size(min_vals{8},1));
        vl_imarraysc(reshape(min_vals{8}(:,:), patch_size, patch_size, [])) ;
    end
            
    fprintf('Done part %0.1f\n', prob_num);
end


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
        img = p4(img, bin_val, part_size, feature_val);
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
                case 1.1
                    img = p1a(img, bin_val);
                case 1.2
                    img = p1b(img, bin_val);
                case 2
                    img = p2(img, bin_val);
                case 3
                    img = p3(img, bin_val, part_size);
                case 4
                    img = p4(img, bin_val, part_size, feature_val);
                case 5
                    img = p5(img, codewords, bin_val, part_size, feature_val);
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


% Train the model - changes depending on the problem
function [min_mdl, min_loss, min_k] = train_model(prob_num, k_vals, train_data, test_data)
    % Separate the data and the labels
    train_res = train_data(:, end);
    train_vals = train_data(:, 1:(end-1));
    min_loss = 1;
    
    % Run the model over a number of k values
    for k = k_vals
        mdl = fitcknn(train_vals, train_res, 'NumNeighbors', k);
        rloss = resubLoss(mdl);
        fprintf('k = %d, correct_train = %f ', k, (1-rloss));
       
        % Cross-Validation for problems 3, 4, 5
        if (prob_num >= 3)
            rng(10);
        	cvMdl = crossval(mdl);
        	kloss = kfoldLoss(cvMdl);
            rloss = kloss; % Set the loss value to be the kloss for min comparison
            fprintf('correct_train_cv = %f\n', (1-kloss));
        end
       
        % Keep track of the lowest loss and the model associated with it
        if min_loss > rloss
            min_loss = rloss;
            min_mdl = mdl;
            min_k = k;
        end
       
        % Report test value for every k for problems 1 + 2
        if (prob_num <= 2)
            test_model(test_data, mdl);
        end
    end
end


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
