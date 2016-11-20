% Train the model - changes depending on the problem
function [mdl, loss] = train_model(prob, full_train_data, num_split, num_tree, crossVal)
    if nargin < 5
        crossVal = true(1);
    end

    % Separate the data and the labels
    full_train_res = full_train_data(:, end);
    full_train_vals = full_train_data(:, 1:(end-1));
    
    templ = templateTree('MaxNumSplits', num_split);
    
    if (crossVal)
        mdl = fitensemble(full_train_vals, full_train_res, 'Bag', ...
            num_tree, templ, 'type', 'classification', 'kfold', 5);
        loss = kfoldLoss(mdl, 'Mode','cumulative');
    else
        mdl = fitensemble(full_train_vals, full_train_res, 'Bag', ...
            num_tree, templ, 'type', 'classification');
        loss = resubLoss(mdl);
    end
end