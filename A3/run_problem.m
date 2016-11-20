% Control function to run the problems
function [] = run_problem(prob, train_data, test_data, max_num_splits)
    % 1=loss, 2=mdl, 3=num_tree, 4=num_split
    min_vals = cell(3, 1);
    min_vals{1} = 1; % loss initialized for comparison
    % TODO: plot_loss = zeros(1, size(pca_vals, 2));

    % Cross validation on number of maximum splits
    for num_split = max_num_splits
        [mdl, loss, num_tree]= train_model(prob, train_data, num_split, 50);
        fprintf('Problem %d & %d & %d & %f \n', prob, num_split, num_tree, loss);
        
        % Keep track of the lowest loss and the model associated with it
        if min_vals{1} > loss
            min_vals = {loss; mdl; num_tree; num_split};
        end
    end
    % CROSS VAL RESULTS
    fprintf('CV -- NUM SPLIT: %d, NUM_TREE: %d, train_err (full) = %f\n', ...
        min_vals{4}, min_vals{3}, min_vals{1});
    
    % Full train on chosen parameters
    [full_mdl, full_loss]= train_model(prob, train_data, ...
        min_vals{4}, min_vals{3}, false(1));
    fprintf('FULL TRAIN -- NUM SPLIT: %d, NUM_TREE: %d, train_err (full) = %f\n', ...
        min_vals{4}, min_vals{3}, full_loss);
    
    % Test
    test_model(prob, test_data, full_mdl);
%     plot(pca_vals, plot_trainloss, '-x', pca_vals, plot_loss, '-x');
%    legend({'train', 'val'});
    
end


function [] = run_problem2(prob, name, train_data, test_data, pca_vals, beta_vals, gamma_vals)
    % 1=loss, 2=mdl, 3 = pca, 4 = coeff, 5 = full loss, 6 = mins, 7 =
    % ranges, 8 = beta, 9 = gamma
    min_vals = cell(3, 1);
    min_vals{1} = 1; % loss initialized for comparison
    plot_loss = zeros(1, size(pca_vals, 2));
    plot_trainloss = zeros(1, size(pca_vals,2));

    % Try various values for each paramter
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
    
    % Test the best model and return the results
    test_feats = [test_feats, test_res];
    test_model(prob, test_feats, min_vals{2});
    
    plot(pca_vals, plot_trainloss, '-x', pca_vals, plot_loss, '-x');
    legend({'train', 'val'});
end