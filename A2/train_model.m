% Train the model - changes depending on the problem
function [min_mdl, min_loss, min_trainloss, min_fullloss, min_coeff] = train_model(prob, full_train_data, num_pca)

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
            
            % Training
            train_coeff = pca(full_train_vals(train_mask,:));
            [tr_mdl, train_avg_err] = get_pca_feats(prob, train_mask, ...
                train_coeff(:, 1:num_pca), full_train_vals, ...
                full_train_res, train_avg_err);
            
            % Validation
            val_feats = [full_train_vals(val_mask,:)*train_coeff(:, 1:num_pca), full_train_res(val_mask,:)];
            val_err = 1 - test_model(val_feats, tr_mdl, false(1));
            val_avg_err = val_avg_err + val_err;
        end
        
        train_loss = train_avg_err / 10;
        % Set the rloss value to be the validation loss for min comparison
        rloss = val_avg_err / 10; 
        
        % Get model for the ALL the training data (including validation)
        full_msk = (indices > 0);
        full_coeff = pca(full_train_vals);
        [mdl, err] = get_pca_feats(prob, full_msk, full_coeff(:, 1:num_pca), ...
            full_train_vals, full_train_res, 0);
        
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
    end
end

% Get the model and error after PCA
function [mdl, err] = get_pca_feats(prob, data_mask, coeff, full_data, full_res, avg_err)
    new_train_feats = full_data(data_mask,:) * coeff;
    if (prob == 3)
        mdl = fitcdiscr(new_train_feats, full_res(data_mask,:), 'discrimType', 'quadratic');
    else
        mdl = fitcdiscr(new_train_feats, full_res(data_mask,:));
    end
    err = avg_err + resubLoss(mdl);
end