function [img] = p2(img, num_bins)
    if size(img,3) == 3
        red_hist = histogram(img(:,:,1), num_bins);
        red_bins = red_hist.Values;
        green_hist = histogram(img(:,:,2), num_bins);
        green_bins = green_hist.Values;
        blue_hist = histogram(img(:,:,3), num_bins);
        blue_bins = blue_hist.Values;
    else
        h = histogram(img(:), num_bins);
        red_bins = h.Values;
        green_bins = h.Values;
        blue_bins =  h.Values;
    end

    % Compile all the data into matrices for knn
    img = transpose([red_bins green_bins blue_bins]);    
end




% % TODO: Use histcounts instead??
% function [] = p2(total_img_num, num_bins, start_path, cat_subdirs)
%     % Make the full matrix so we are not extending the size of the matrix
%     data_matrix = zeros(total_img_num, num_bins*3);
%     category_matrix = zeros(total_img_num, 1);
%     total_idx = 0;
%     
%     % Go over all the images
%     for category_idx = 1:numel(cat_subdirs)
%         category = cat_subdirs(category_idx).name;
% 
%         imgs = dir(fullfile(start_path, category, '*.jpg'));
%         for img_idx = 1:numel(imgs)
%             total_idx = total_idx + 1; % Keep track of the index for all imgs
% 
%             % Get image
%             img = imread(fullfile(imgs(img_idx).folder, imgs(img_idx).name));
% 
%             if size(img,3) == 3
%                 red_hist = histogram(img(:,:,1), 50);
%                 red_bins = red_hist.Values;
%                 green_hist = histogram(img(:,:,2), 50);
%                 green_bins = green_hist.Values;
%                 blue_hist = histogram(img(:,:,3), 50);
%                 blue_bins = blue_hist.Values;
%             else
%                 h = histogram(img(:), 50);
%                 red_bins = h.Values;
%                 green_bins = h.Values;
%                 blue_bins =  h.Values;
%             end
%             
%             % Compile all the data into matrices for knn
%             data_matrix(total_idx, :) = [red_bins green_bins blue_bins];
%             category_matrix(total_idx) = category_idx;
%         end
%     end
%     
%     % Combine data and labels for partitioning
%     full_matrix = [data_matrix, category_matrix];
%     [train_data, val_data, test_data] = dividerand(transpose(full_matrix), 0.8, 0.0, 0.2);
% 
%     % Separate dat and labels again
%     train_cat = transpose(train_data(end,:));
%     test_cat = transpose(test_data(end,:));
%     train_data = transpose(train_data(1:(end-1), :));
%     test_data = transpose(test_data(1:(end-1), :));
% 
%     % Train the model with knn
%     k_vals = [2,4,8,16,32];
%     for k_idx = 1:numel(k_vals)
%         k = k_vals(k_idx);
%         mdl = fitcknn(train_data, train_cat, 'NumNeighbors', k);
%         rloss = resubLoss(mdl);
% 
%         % Classify the test images
%         correct_class = 0;
%         for test_idx = 1:size(test_data,1)
%            img_class = predict(mdl, test_data(test_idx,:));
%            if img_class == test_cat(test_idx)
%                correct_class = correct_class + 1;
%            end
%         end
% 
%         fprintf('K = %d, LOSS: %f\n',k, rloss);
%         fprintf('K = %d, CORRECT TEST CLASS: %d\n\n',k, correct_class);
%     end
%     fprintf('Done part 2!');
% end