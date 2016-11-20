% Control function to run the problems
function [] = run_problem(prob, classifier, train_data, test_data, max_num_splits, max_num_trees)
    % 1=loss, 2=mdl, 3=num_tree, 4=num_split
    min_vals = cell(3, 1);
    min_vals{1} = 1; % loss initialized for comparison
  
    % Clear the figure
    h = figure(prob);
    if ishandle(h)
        clf(prob)
    end
    hold on;
    figure(prob);

    % Cross validation on number of maximum splits
    for num_split = max_num_splits
        [mdl, loss]= train_model(prob, classifier, train_data, num_split, max_num_trees);
        plot(loss, '-x');
        [loss,  num_tree] = min(loss);
        
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
    [full_mdl, full_loss]= train_model(prob, classifier, train_data, ...
        min_vals{4}, min_vals{3}, false(1));
   
    if (prob == 3)
        plot(full_loss, '-x');
    end

    fprintf('FULL TRAIN -- NUM SPLIT: %d, NUM_TREE: %d, train_err (full) = %f\n', ...
        min_vals{4}, min_vals{3}, min(full_loss));
    % Test
    [err, cLoss] = test_model(prob, test_data, full_mdl);


    if (prob < 3)
        legend({'MNS=1', 'MNS=5', 'MNS=10', 'MNS=20'});
    else
        plot(cLoss, '-x');
        legend({'Cross Val', 'Training', 'Testing'});
    end
	axis([0 max_num_trees 0 1]);
	xlabel('Num Trees'); ylabel('Loss'); title(['Question ', num2str(prob)]);
    
end
