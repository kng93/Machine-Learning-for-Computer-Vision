function [err, cLoss] = test_model(prob, test_data, mdl, dispVal)
    if nargin < 4
        dispVal = true(1);
    end

    % Split up the result from the data
    test_res = test_data(:, end);
    test_vals = test_data(:, 1:(end-1));
    num_correct = 0;
    
    for test_idx = 1:size(test_vals,1)
        img_class = predict(mdl, test_vals(test_idx,:));

        % Count the number of correct values
        if img_class == test_res(test_idx)
            num_correct = num_correct + 1; 
        end
    end
    err = 1 - (num_correct/size(test_vals,1));
    cLoss = loss(mdl, test_vals, test_res, 'mode', 'cumulative');

    % Print out the correct percentage
    if (dispVal)
        fprintf('test_err = %f\n', err);
    end
end