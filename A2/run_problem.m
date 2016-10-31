% Control function to run the problems
function [] = run_problem(prob, train_data, test_data, pca_vals)
    % 1=loss, 2=mdl, 3 = pca, 4 = coeff, 5 = full loss
    min_vals = cell(3, 1);
    min_vals{1} = 1; % loss initialized for comparison
    plot_loss = zeros(1, size(pca_vals, 2));
    plot_trainloss = zeros(1, size(pca_vals,2));

    for pca_idx = 1:numel(pca_vals)
        num_pca = pca_vals(pca_idx);
        if (num_pca < size(train_data, 2))
            fprintf('PCA NUM: %d ', num_pca);
            [mdl, loss, train_loss, full_loss, coeff]= train_model(prob, train_data, num_pca);

            plot_loss(pca_idx) = loss;
            plot_trainloss(pca_idx) = train_loss;
            % Keep track of the lowest loss and the model associated with it
            if min_vals{1} > loss
                min_vals = {loss; mdl; num_pca; coeff; full_loss};
            end
        end
    end
    
    fprintf('PCA NUM FOR TEST: %d, correct_train (full) = %f ', min_vals{3}, min_vals{5});
    test_vals = test_data(:, 1:(end-1));
    test_res = test_data(:, end);
    
    test_feats = [test_vals * min_vals{4}, test_res];
    test_model(test_feats, min_vals{2});
    
    plot(pca_vals, plot_trainloss, '-x', pca_vals, plot_loss, '-x');
    legend({'train', 'val'});
end