function [img] = p3(img, num_bins, num_grid)
    fun = @(block_struct) histcounts(block_struct.data, num_bins);

    if size(img,3) == 3
        red_hist = blockproc(img(:,:,1), ...
                    [ceil(size(img,1)/num_grid), ceil(size(img,2)/num_grid)], ...
                    fun);
        green_hist = blockproc(img(:,:,2), ...
                    [ceil(size(img,1)/num_grid), ceil(size(img,2)/num_grid)], ...
                    fun);
        blue_hist = blockproc(img(:,:,3), ...
                    [ceil(size(img,1)/num_grid), ceil(size(img,2)/num_grid)], ...
                    fun);
    else
        red_hist = blockproc(img, ...
                    [ceil(size(img,1)/num_grid), ceil(size(img,2)/num_grid)], ...
                    fun);
        green_hist = blockproc(img, ...
                    [ceil(size(img,1)/num_grid), ceil(size(img,2)/num_grid)], ...
                    fun);
        blue_hist = blockproc(img, ...
                    [ceil(size(img,1)/num_grid), ceil(size(img,2)/num_grid)], ...
                    fun);
    end

    % Compile all the data into matrices for knn
    img = [red_hist(:); green_hist(:); blue_hist(:)];   
end


% function [] = p3(total_img_num, start_path, cat_subdirs)
% 
%     bin_vals = [10, 20, 30, 40];
%     grid_vals = [1, 2, 3, 4, 5];
%     
%     for bin_num = bin_vals
%         for grid_num = grid_vals
%             % Make the full matrix so we are not extending the size of the matrix
%             data_matrix = zeros(total_img_num, bin_num*grid_num*grid_num*3); % bin_num * (row partitions) * (col partitions) * (channels)
%             category_matrix = zeros(total_img_num, 1);
%             total_idx = 0;
% 
%             % Go over all the images
%             for category_idx = 1:numel(cat_subdirs)
%                 category = cat_subdirs(category_idx).name;
% 
%                 imgs = dir(fullfile(start_path, category, '*.jpg'));
%                 for img_idx = 1:numel(imgs)
%                     total_idx = total_idx + 1; % Keep track of the index for all imgs
% 
%                     % Get image
%                     img = imread(fullfile(imgs(img_idx).folder, imgs(img_idx).name));
%                     fun = @(block_struct) histcounts(block_struct.data, bin_num);
% 
%                     if size(img,3) == 3
%                         red_hist = blockproc(img(:,:,1), ...
%                                     [ceil(size(img,1)/grid_num), ceil(size(img,2)/grid_num)], ...
%                                     fun);
%                         green_hist = blockproc(img(:,:,2), ...
%                                     [ceil(size(img,1)/grid_num), ceil(size(img,2)/grid_num)], ...
%                                     fun);
%                         blue_hist = blockproc(img(:,:,3), ...
%                                     [ceil(size(img,1)/grid_num), ceil(size(img,2)/grid_num)], ...
%                                     fun);
%                     else
%                         red_hist = blockproc(img, ...
%                                     [ceil(size(img,1)/grid_num), ceil(size(img,2)/grid_num)], ...
%                                     fun);
%                         green_hist = blockproc(img, ...
%                                     [ceil(size(img,1)/grid_num), ceil(size(img,2)/grid_num)], ...
%                                     fun);
%                         blue_hist = blockproc(img, ...
%                                     [ceil(size(img,1)/grid_num), ceil(size(img,2)/grid_num)], ...
%                                     fun);
%                     end
% 
%                     % Compile all the data into matrices for knn
%                     data_matrix(total_idx, :) = [red_hist(:); green_hist(:); blue_hist(:)];
%                     category_matrix(total_idx) = category_idx;
%                 end
%             end
% 
%             % Combine data and labels for partitioning
%             full_matrix = [data_matrix, category_matrix];
%             [train_data, val_data, test_data] = dividerand(transpose(full_matrix), 0.8, 0.0, 0.2);
% 
%             % Separate dat and labels again
%             train_cat = transpose(train_data(end,:));
%             test_cat = transpose(test_data(end,:));
%             train_data = transpose(train_data(1:(end-1), :));
%             test_data = transpose(test_data(1:(end-1), :));
% 
%             % Train the model with knn
%             k_vals = [2,4,8,16,32];
%             for k_idx = 1:numel(k_vals)
%                 k = k_vals(k_idx);
%                 mdl = fitcknn(train_data, train_cat, 'NumNeighbors', k);
%                 rloss = resubLoss(mdl);
% 
%                 % Cross-Validation
%                 rng(10);
%                 cvMdl = crossval(mdl);
%                 kloss = kfoldLoss(cvMdl);
% 
% 
%                 fprintf('bin = %d, K = %d, grid = %d, LOSS: %f, CVLOSS: %f\n', bin_num, k, grid_num, rloss, kloss);
%             end
%         end
%     end
% 
% %     % Classify the test images
% %     correct_class = 0;
% %     for test_idx = 1:size(test_data,1)
% %        img_class = predict(mdl, test_data(test_idx,:));
% %        if img_class == test_cat(test_idx)
% %            correct_class = correct_class + 1;
% %        end
% %     end
% % 
% %     fprintf('K = %d, CORRECT TEST CLASS: %d\n\n',k, correct_class);
%     fprintf('Done part 3!');
% end