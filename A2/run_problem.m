% Control function to run the problems
function [] = run_problem(prob, name, train_data, test_data, pca_vals, beta_vals, gamma_vals)
    % 1=loss, 2=mdl, 3 = pca, 4 = coeff, 5 = full loss, 6 = mins, 7 =
    % ranges, 8 = beta, 9 = gamma
    min_vals = cell(3, 1);
    min_vals{1} = 1; % loss initialized for comparison
    plot_loss = zeros(1, size(pca_vals, 2));
    plot_trainloss = zeros(1, size(pca_vals,2));
    
    if (prob < 4)
       beta_vals = [-1]; 
    end
    
    if (prob < 5)
        gamma_vals = [-1];
    end

    for pca_idx = 1:numel(pca_vals)
        num_pca = pca_vals(pca_idx);
        if (num_pca < size(train_data, 2))
            for beta = beta_vals
                for gamma = gamma_vals
                    if (gamma >= 0)
                        fprintf('%s & %d & %f & %f & ', name, num_pca, beta, gamma);
                    elseif (beta >= 0)
                        fprintf('%s & %d & %f & ', name, num_pca, beta);
                    else
                        fprintf('%s & %d & ', name, num_pca);
                    end
                    [mdl, loss, train_loss, full_loss, coeff, mins, ranges]= train_model(prob, train_data, num_pca, beta, gamma);

                    plot_loss(pca_idx) = loss;
                    plot_trainloss(pca_idx) = train_loss;
                    % Keep track of the lowest loss and the model associated with it
                    if min_vals{1} > loss
                        min_vals = {loss; mdl; num_pca; coeff; full_loss; mins; ranges; beta; gamma};
                    end
                end
            end
        end
    end
    
    fprintf('PCA NUM FOR TEST: %d, BETA: %f, GAMMA: %f, train_err (full) = %f ', min_vals{3}, min_vals{8}, min_vals{9}, min_vals{5});
    test_vals = test_data(:, 1:(end-1));
    test_res = test_data(:, end);
    
    % Get new testing features and normalize for SVM
    test_feats = test_vals * min_vals{4};
    if (prob >= 4)
        test_feats = normalize(test_feats, min_vals{6}, min_vals{7}); 
    end
    
    test_feats = [test_feats, test_res];
    test_model(prob, test_feats, min_vals{2});
    
    plot(pca_vals, plot_trainloss, '-x', pca_vals, plot_loss, '-x');
    legend({'train', 'val'});
end