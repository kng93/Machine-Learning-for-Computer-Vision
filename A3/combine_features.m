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