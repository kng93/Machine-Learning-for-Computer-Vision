% Train the model - changes depending on the problem
function [min_mdl, min_loss] = train_model(prob_num, train_data)
    % Separate the data and the labels
    train_res = train_data(:, end);
    train_vals = train_data(:, 1:(end-1));
    min_loss = 1;
    
    % Run the model over a number of k values
    mdl = fitcdiscr(train_vals, train_res);
    rloss = resubLoss(mdl);
    fprintf('correct_train = %f ', (1-rloss));

    % Cross-Validation for problems 3, 4, 5
    if (prob_num >= 3)
        rng(10);
        cvMdl = crossval(mdl);
        kloss = kfoldLoss(cvMdl);
        rloss = kloss; % Set the loss value to be the kloss for min comparison
        fprintf('correct_train_cv = %f\n', (1-kloss));
    end

    % Keep track of the lowest loss and the model associated with it
    if min_loss > rloss
        min_loss = rloss;
        min_mdl = mdl;
    end
end