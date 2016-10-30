% Control function to run the problems
function [] = run_problem(prob_num, train_data, test_data, codewords)
    % 1=loss, 2=mdl, 3=test_data, 4=k, 5=bin, 6=partition, 7=feature, 
    % 8=codewords
    min_vals = cell(8, 1);
    min_vals{1} = 1; % loss initialized for comparison

    [mdl, loss]= train_model(prob_num, train_data);
    test_model(test_data, mdl);

    fprintf('Done Q1.1, part %0.1f\n', prob_num);
end