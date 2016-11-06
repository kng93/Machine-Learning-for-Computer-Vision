% Train the model - changes depending on the problem
function [min_mdl, min_loss, min_trainloss, min_fullloss, ...
    min_coeff, min_mins, min_ranges] = train_model(prob, ...
    full_train_data, num_pca, beta, gamma)
    
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
                train_data, full_train_res, beta, gamma, train_avg_err);
            
            % Validation
            
            val_feats = [val_data, full_train_res(val_mask,:)];
            val_err = test_model(prob, val_feats, tr_mdl, false(1));
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
        [mdl, err] = get_mdl(prob, full_msk, full_new_train, full_train_res, beta, gamma, 0);
        
        fprintf('%f & %f \\\\\\hline\n', train_loss, rloss);
    else
        mdl = fitcdiscr(full_train_vals, full_train_res);
        rloss = resubLoss(mdl);
        train_loss = 0;
        err = rloss;
        fprintf('correct_train = %f ', rloss);
    end

    % Keep track of the lowest loss and the model associated with it
    if min_loss > rloss
        min_loss = rloss;
        min_trainloss = train_loss;
        min_mdl = mdl;
        min_fullloss = err;
        if (prob >= 2)
            min_coeff = full_coeff(:, 1:num_pca);
        else
            min_coeff = -1;
        end
        if (prob >= 4)
            min_mins = full_mins;
            min_ranges = full_ranges;
        else
            min_mins = -1;
            min_ranges = -1;
        end
    end
end

% Get the model and error after PCA 
function [mdl, err] = get_mdl(prob, data_mask, new_train_feats, full_res, beta, gamma, avg_err)
    if (prob == 5)
        mdl = svmtrain(full_res(data_mask,:), new_train_feats, ['-t 2 -c ', num2str(beta), ' -g ', num2str(gamma), ' -q']);
    elseif (prob == 4)
        mdl = svmtrain(full_res(data_mask,:), new_train_feats, ['-t 0 -c ', num2str(beta), ' -q']); 
    elseif (prob == 3)
        mdl = fitcdiscr(new_train_feats, full_res(data_mask,:), 'discrimType', 'quadratic');
    else
        mdl = fitcdiscr(new_train_feats, full_res(data_mask,:)); 
    end
    
    if (prob < 4)
        err = avg_err + resubLoss(mdl);
    else
        [img_class, acc, dec_values_P] = svmpredict(full_res(data_mask,:), new_train_feats, mdl, '-q');
        err = 1 - (acc(1) / 100);
    end
end