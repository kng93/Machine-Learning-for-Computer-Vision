DB_DIR = '/Users/karinng/Documents/1UniHW/cs9840/Assignments/A1/TenCategories';
TRAIN_PARTITION = 0.8;
PROBLEMS = [1, 2, 3, 4];
GREYSCALE = 1;
COLOUR = 3;
RANDOMIZE_DATA = false(1);

[cat_subdirs, partition_array] = get_dataset_info(DB_DIR, TRAIN_PARTITION, RANDOMIZE_DATA);

for prob = PROBLEMS
    switch prob
        % Run A1 w/ discrimant analysis classifier (cdiscr)
        case 1
            disp('Question 1');
            if (exist(fullfile(cd, 'combined_data.mat'), 'file'))
                load('combined_data', 'train_data', 'test_data');
            else
                qNums = [1, 2]; %[1, 2, 3, 4, 5];
                bins = [50*50, 50]; %[50*50, 50, 10, 50, 250];
                parts = [1, 1]; %[1, 1, 3, 30, 40];
                features = [1, 1]; %[1, 1, 1, 10, 100];
                [train_data, test_data] = combine_features(qNums, ...
                        cat_subdirs, partition_array, bins, ...
                        parts, features, COLOUR);
                
                save('combined_data', 'train_data', 'test_data');
            end
            disp('Finished setting up features');
              
            max_num_splits = [1, 5, 10, 20];
            max_num_trees = 50;
            run_problem(prob, train_data, test_data, max_num_splits, max_num_trees);
            
        case 2
            disp('Implement 2');
        case 3
            disp('Implement 3');
        case 4
            disp('Implement 4');
    end
end