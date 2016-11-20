% Train the model - changes depending on the problem
function [mdl, loss] = train_model(prob, classifier, full_train_data, ...
    num_split, num_tree, crossVal)

    if nargin < 6
        crossVal = true(1);
    end

    % Separate the data and the labels
    full_train_res = full_train_data(:, end);
    full_train_vals = full_train_data(:, 1:(end-1));
    
    templ = templateTree('MaxNumSplits', num_split);
    
    if (crossVal)
        mdl = fitensemble(full_train_vals, full_train_res, classifier, ...
            num_tree, templ, 'type', 'classification', 'kfold', 5);
        loss = kfoldLoss(mdl, 'Mode','cumulative');
    else
        mdl = fitensemble(full_train_vals, full_train_res, classifier, ...
            num_tree, templ, 'type', 'classification');
        if (prob == 3)
            loss = resubLoss(mdl, 'Mode', 'cumulative');
        else
            loss = resubLoss(mdl);
        end
    end
end