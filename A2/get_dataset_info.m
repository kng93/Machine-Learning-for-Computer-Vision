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