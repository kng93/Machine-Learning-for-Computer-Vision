% Train the model - changes depending on the problem
function [min_mdl, min_loss, min_trainloss, min_fullloss, ...
    min_coeff, min_mins, min_ranges] = train_model(prob, full_train_data, num_pca, beta)
    
    % Separate the data and the labels
    full_train_res = full_train_data(:, end);
    full_train_vals = full_train_data(:, 1:(end-1));
    min_loss = 1;
    
    % Cross-Validation for problems 2, 3, 4, 5
    if (prob >= 2)
        % Cross validation 
        indices = crossvalind('Kfold',full_train_res,10);
        train_avg_err = 0;
        val_avg_err = 0;
        for i = 1:10
            % Partitioning
            val_mask = (indices == i); 
            train_mask = ~val_mask;
            
            % Setting up the data
            train_coeff = pca(full_train_vals(train_mask,:));
            train_data = full_train_vals(train_mask,:) * train_coeff(:, 1:num_pca);
            val_data = full_train_vals(val_mask,:) * train_coeff(:, 1:num_pca);
            
            % Normalize the data for SVM train
            if (prob >= 4)
               [train_data, train_mins, train_ranges] =  normalize(train_data);
               [val_data] = normalize(val_data, train_mins, train_ranges);
            end
            
            % Training
            [tr_mdl, train_avg_err] = get_mdl(prob, train_mask, ...
                train_data, full_train_res, beta, train_avg_err);
            
            % Validation
            
            val_feats = [val_data, full_train_res(val_mask,:)];
            val_err = 1 - test_model(prob, val_feats, tr_mdl, false(1));
            val_avg_err = val_avg_err + val_err;
        end
        
        train_loss = train_avg_err / 10;
        % Set the rloss value to be the validation loss for min comparison
        rloss = val_avg_err / 10; 
        
        % Get model for the ALL the training data (including validation)
        full_msk = (indices > 0);
        full_coeff = pca(full_train_vals);
        full_new_train = full_train_vals * full_coeff(:, 1:num_pca);
        if (prob >= 4)
           [full_new_train, full_mins, full_ranges] = normalize(full_new_train); 
        end
        [mdl, err] = get_mdl(prob, full_msk, full_new_train, full_train_res, beta, 0);
        
        fprintf('correct_train_cv = %f, correct_val_cv = %f \n', (1-train_loss), (1-rloss));
    else
        mdl = fitcdiscr(train_vals, train_res);
        rloss = resubLoss(mdl);
        train_loss = 0;
        err = rloss;
        fprintf('correct_train = %f ', (1-rloss));
    end

    % Keep track of the lowest loss and the model associated with it
    if min_loss > rloss
        min_loss = rloss;
        min_trainloss = train_loss;
        min_mdl = mdl;
        min_coeff = full_coeff(:, 1:num_pca);
        min_fullloss = err;
        min_mins = full_mins;
        min_ranges = full_ranges;
    end
end

% Get the model and error after PCA 
function [mdl, err] = get_mdl(prob, data_mask, new_train_feats, full_res, beta, avg_err)
    if (prob == 4)
        mdl = svmtrain(full_res(data_mask,:), new_train_feats, ['-t 0 -c ', num2str(beta), ' -q']); 
    elseif (prob == 3)
        mdl = fitcdiscr(new_train_feats, full_res(data_mask,:), 'discrimType', 'quadratic');
    else
        mdl = fitcdiscr(new_train_feats, full_res(data_mask,:));
    end
    
    if (prob < 4)
        err = avg_err + resubLoss(mdl);
    else
        err = 1; %TODO: CHANGE
    end
end