% SETUP
run '/Users/karinng/Documents/SoftwareFiles/Matlab/vlfeat-0.9.20/toolbox/vl_setup'
addpath('/Users/karinng/Documents/1UniHW/cs9840/Assignments/A2/libsvm-3.21/matlab');
DB_DIR = '/Users/karinng/Documents/1UniHW/cs9840/Assignments/A1/TenCategories';
TRAIN_PARTITION = 0.8;
PROBLEMS = [3]; %[1, 2, 3, 4, 5];
SUB_PROBLEMS = [1, 2, 3, 4, 5];
GREYSCALE = 1;
COLOUR = 3;
RANDOMIZE_DATA = false(1);

[cat_subdirs, partition_array] = get_dataset_info(DB_DIR, TRAIN_PARTITION, RANDOMIZE_DATA);

for prob = PROBLEMS
    switch prob
        % Run A1 w/ discrimant analysis classifier (cdiscr)
        case 1
            disp('Greyscale Pixelwise Q1');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(1)], ...
                    cat_subdirs, partition_array, [50*50], [1], ...
                    [1], GREYSCALE);
            [mdl, loss]= train_model(prob, train_data);
            test_model(prob, test_data, mdl);
            
            disp('Global Histogram Q1');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(2)], ...
                    cat_subdirs, partition_array, [50], [1], ...
                    [1], COLOUR);
            [mdl, loss]= train_model(prob, train_data);
            test_model(prob, test_data, mdl);
            
            disp('Local Histogram Q1');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(3)], ...
                    cat_subdirs, partition_array, [10], [3], ...
                    [1], COLOUR);
            [mdl, loss]= train_model(prob, train_data);
            test_model(prob, test_data, mdl);
            
            disp('HOG Q1');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(4)], ...
                    cat_subdirs, partition_array, [50], [30], ...
                    [10], COLOUR);
            [mdl, loss]= train_model(prob, train_data);
            test_model(prob, test_data, mdl);
            
            disp('Patch Q1');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(5)], ...
                    cat_subdirs, partition_array, [250], [40], ...
                    [100], COLOUR);
            [mdl, loss]= train_model(prob, train_data);
            test_model(prob, test_data, mdl);
              
            disp('Combined Q1');         
            IMG_SIZE = 50;
            % Q# = [2, 4, 5], Bin_Vals = [50, 50, 250], Part_Vals = [1, 30,
            % 40], Feature_Vals = [1, 10, 100] 
            [train_data, test_data] = combine_features([2, 4, 5], ...
                    cat_subdirs, partition_array, [50, 50, 250], ...
                    [1, 30, 40], [1, 10, 100], COLOUR);
             
            [mdl, loss]= train_model(prob, train_data);
            test_model(prob, test_data, mdl);
            
        case 2
            pca_vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130];
            disp('Greyscale Pixelwise Q2');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(1)], ...
                    cat_subdirs, partition_array, [50*50], [1], ...
                    [1], GREYSCALE);
            run_problem(prob, 'Greyscale Pixelwise', train_data, test_data, pca_vals);
            
            disp('Global Histogram Q2');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(2)], ...
                    cat_subdirs, partition_array, [50], [1], ...
                    [1], COLOUR);
            run_problem(prob, 'Global Histogram', train_data, test_data, pca_vals);
            
            disp('Local Histogram Q2');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(3)], ...
                    cat_subdirs, partition_array, [10], [3], ...
                    [1], COLOUR);
            run_problem(prob, 'Local Histogram', train_data, test_data, pca_vals);
            
            disp('HOG Q2');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(4)], ...
                    cat_subdirs, partition_array, [50], [30], ...
                    [10], COLOUR);
            run_problem(prob, 'HOG', train_data, test_data, pca_vals);
            
            disp('Patch Q2');
            if (exist(fullfile(cd, 'patch_data.mat'), 'file'))
                load('patch_data', 'train_data', 'test_data');
            else
                [train_data, test_data] = combine_features([SUB_PROBLEMS(5)], ...
                        cat_subdirs, partition_array, [250], [40], ...
                        [100], COLOUR);
                save('patch_data', 'train_data', 'test_data');
            end
            run_problem(prob, 'Patch', train_data, test_data, pca_vals);
            
            disp('Combined Q2');
            if (exist(fullfile(cd, 'combined_data.mat'), 'file'))
                load('combined_data', 'train_data', 'test_data');
            else
                [train_data, test_data] = combine_features([2, 4, 5], ...
                        cat_subdirs, partition_array, [50, 50, 250], ...
                        [1, 30, 40], [1, 10, 100], COLOUR);
                
                save('combined_data', 'train_data', 'test_data');
            end
             
            run_problem(prob, 'Combined', train_data, test_data, pca_vals);
        case 3
            pca_vals = [10, 20, 30, 40, 50, 60, 70];
            disp('Greyscale Pixelwise Q3');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(1)], ...
                    cat_subdirs, partition_array, [50*50], [1], ...
                    [1], GREYSCALE);
            run_problem(prob, 'Greyscale Pixelwise', train_data, test_data, pca_vals);
            
            disp('Global Histogram Q3');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(2)], ...
                    cat_subdirs, partition_array, [50], [1], ...
                    [1], COLOUR);
            run_problem(prob, 'Global Histogram', train_data, test_data, pca_vals);
            
            disp('Local Histogram Q3');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(3)], ...
                    cat_subdirs, partition_array, [10], [3], ...
                    [1], COLOUR);
            run_problem(prob, 'Local Histogram', train_data, test_data, pca_vals);
            
            disp('HOG Q3');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(4)], ...
                    cat_subdirs, partition_array, [50], [30], ...
                    [10], COLOUR);
            run_problem(prob, 'HOG', train_data, test_data, pca_vals);
            
            disp('Patch Q3');
            if (exist(fullfile(cd, 'patch_data.mat'), 'file'))
                load('patch_data', 'train_data', 'test_data');
            else
                [train_data, test_data] = combine_features([SUB_PROBLEMS(5)], ...
                        cat_subdirs, partition_array, [250], [40], ...
                        [100], COLOUR);
                save('patch_data', 'train_data', 'test_data');
            end
            run_problem(prob, 'Patch', train_data, test_data, pca_vals);
            
            disp('Combined Q3');
            if (exist(fullfile(cd, 'combined_data.mat'), 'file'))
                load('combined_data', 'train_data', 'test_data');
            else
                [train_data, test_data] = combine_features([2, 4, 5], ...
                        cat_subdirs, partition_array, [50, 50, 250], ...
                        [1, 30, 40], [1, 10, 100], COLOUR);
                
                save('combined_data', 'train_data', 'test_data');
            end
             
            run_problem(prob, 'Combined', train_data, test_data, pca_vals);
        case 4
            beta_vals = [0.0625, 0.25, 1, 4, 16];
            pca_vals = [10, 20, 30, 40, 50, 60, 70];
            disp('Greyscale Pixelwise Q4');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(1)], ...
                    cat_subdirs, partition_array, [50*50], [1], ...
                    [1], GREYSCALE);
            run_problem(prob, 'Greyscale Pixelwise', train_data, test_data, pca_vals, beta_vals);
            
            
            disp('Global Histogram Q4');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(2)], ...
                    cat_subdirs, partition_array, [50], [1], ...
                    [1], COLOUR);
            run_problem(prob, 'Global Histogram', train_data, test_data, pca_vals, beta_vals);
            
            disp('Local Histogram Q4');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(3)], ...
                    cat_subdirs, partition_array, [10], [3], ...
                    [1], COLOUR);
            run_problem(prob, 'Local Histogram', train_data, test_data, pca_vals, beta_vals);
            
            disp('HOG Q4');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(4)], ...
                    cat_subdirs, partition_array, [50], [30], ...
                    [10], COLOUR);
            run_problem(prob, 'HOG', train_data, test_data, pca_vals, beta_vals);
            
            disp('Patch Q4');
            if (exist(fullfile(cd, 'patch_data.mat'), 'file'))
                load('patch_data', 'train_data', 'test_data');
            else
                [train_data, test_data] = combine_features([SUB_PROBLEMS(5)], ...
                        cat_subdirs, partition_array, [250], [40], ...
                        [100], COLOUR);
                save('patch_data', 'train_data', 'test_data');
            end
            run_problem(prob, 'Patch', train_data, test_data, pca_vals, beta_vals);
            
            disp('Combined Q4');
            if (exist(fullfile(cd, 'combined_data.mat'), 'file'))
                load('combined_data', 'train_data', 'test_data');
            else
                [train_data, test_data] = combine_features([2, 4, 5], ...
                        cat_subdirs, partition_array, [50, 50, 250], ...
                        [1, 30, 40], [1, 10, 100], COLOUR);
                
                save('combined_data', 'train_data', 'test_data');
            end
             
            run_problem(prob, 'Combined', train_data, test_data, pca_vals, beta_vals);
        case 5
            beta_vals = [0.0625, 0.25, 1, 4, 16];
            pca_vals = [10, 20, 30, 40, 50, 60, 70];
            gamma_vals = [0.0625, 0.25, 1, 4, 16];
            
            disp('Greyscale Pixelwise Q5');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(1)], ...
                    cat_subdirs, partition_array, [50*50], [1], ...
                    [1], GREYSCALE);
            run_problem(prob, 'Greyscale Pixelwise', train_data, test_data, pca_vals, beta_vals, gamma_vals);
            
            
            disp('Global Histogram Q5');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(2)], ...
                    cat_subdirs, partition_array, [50], [1], ...
                    [1], COLOUR);
            run_problem(prob, 'Global Histogram', train_data, test_data, pca_vals, beta_vals, gamma_vals);
            
            disp('Local Histogram Q5');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(3)], ...
                    cat_subdirs, partition_array, [10], [3], ...
                    [1], COLOUR);
            run_problem(prob, 'Local Histogram', train_data, test_data, pca_vals, beta_vals, gamma_vals);
            
            disp('HOG Q5');
            [train_data, test_data] = combine_features([SUB_PROBLEMS(4)], ...
                    cat_subdirs, partition_array, [50], [30], ...
                    [10], COLOUR);
            run_problem(prob, 'HOG', train_data, test_data, pca_vals, beta_vals, gamma_vals);
            
            disp('Patch Q5');
            if (exist(fullfile(cd, 'patch_data.mat'), 'file'))
                load('patch_data', 'train_data', 'test_data');
            else
                [train_data, test_data] = combine_features([SUB_PROBLEMS(5)], ...
                        cat_subdirs, partition_array, [250], [40], ...
                        [100], COLOUR);
                save('patch_data', 'train_data', 'test_data');
            end
            run_problem(prob, 'Patch', train_data, test_data, pca_vals, beta_vals, gamma_vals);
            
            disp('Combined Q5');
            if (exist(fullfile(cd, 'combined_data.mat'), 'file'))
                load('combined_data', 'train_data', 'test_data');
            else
                [train_data, test_data] = combine_features([2, 4, 5], ...
                        cat_subdirs, partition_array, [50, 50, 250], ...
                        [1, 30, 40], [1, 10, 100], COLOUR);
                
                save('combined_data', 'train_data', 'test_data');
            end
             
            run_problem(prob, 'Combined', train_data, test_data, pca_vals, beta_vals, gamma_vals);
    end
end


% Combine the features
function [full_train_data, full_test_data] = combine_features(ft_nums, ...
        cat_subdirs, partition_array, bin_vals, part_vals, feature_vals, colour)

    collection_train = {};
    collection_test = {};
    
    % Set up the train and test arrays 
    for ft_idx = 1:numel(ft_nums)   
        total_train = sum([partition_array{:}]);
        total_test = sum([partition_array{:}] == 0);
        [codewords, train_data, test_data] = set_data(ft_nums(ft_idx), ...
                                                cat_subdirs, ...
                                                partition_array, ...
                                                colour, ...
                                                total_train, ...
                                                total_test, ...
                                                bin_vals(ft_idx), ...
                                                part_vals(ft_idx), ...
                                                feature_vals(ft_idx));
        if (ft_idx == numel(ft_nums))
            collection_train{ft_idx} = train_data;
            collection_test{ft_idx} = test_data;
        else
            collection_train{ft_idx} = train_data(:, 1:(end-1));
            collection_test{ft_idx} = test_data(:, 1:(end-1));
        end
        fprintf('Set data for subq %d. Number of dimensions: %d\n', ft_nums(ft_idx), size(train_data,2)); 
    end
    full_train_data = cat(2, collection_train{:});
    full_test_data = cat(2, collection_test{:});
end

