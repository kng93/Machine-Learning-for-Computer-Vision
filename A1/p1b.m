%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem 1b
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [img] = p1b(img, feature_size)
    % Image size is the square of feature size (defined in run_a1.m)
    img_size = floor(sqrt(feature_size));

    % Resize image
    img = imresize(img, [img_size img_size]);
    
    % Flatten image into a long vector
    if size(img,3) == 3
        img = img(:);
    else
        img = [img(:); img(:); img(:)];
    end
    
    % 
    img = img(:);
end


% function [] = p1b(total_img_num, img_size, resized_dir, cat_subdirs)
%     % Make the full matrix so we are not extending the size of the matrix
%     data_matrix = zeros(total_img_num, img_size*img_size*3);
%     category_matrix = zeros(total_img_num, 1);
%     total_idx = 0;
% 
%     % Go over all the resized images
%     for category_idx = 1:numel(cat_subdirs)
%         category = cat_subdirs(category_idx).name;
% 
%         imgs = dir(fullfile(resized_dir, category, '*.jpg'));
%         for img_idx = 1:numel(imgs)
%             total_idx = total_idx + 1; % Keep track of the index for all imgs
% 
%             % Get image
%             img = imread(fullfile(imgs(img_idx).folder, imgs(img_idx).name));
% 
%             if size(img,3) == 3
%                 flatten_img = img(:);
%             else
%                 flatten_img = [img(:); img(:); img(:)];
%             end
% 
%             % Compile all the data into matrices for knn
%             data_matrix(total_idx, :) = flatten_img;
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
%     fprintf('Done part 1b!');
% end