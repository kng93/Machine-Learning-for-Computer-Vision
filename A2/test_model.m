% Test the model (on the test data set)
function [acc] = test_model(prob, test_data, mdl, dispVal)

    if nargin < 4
        dispVal = true(1);
    end
    
    % Split up the result from the data
    test_res = test_data(:, end);
    test_vals = test_data(:, 1:(end-1));
    
    % Go over every test value
    num_correct = 0;
    if (prob < 3)
        for test_idx = 1:size(test_vals,1)
            img_class = predict(mdl, test_vals(test_idx,:));
       
            % Count the number of correct values
            if img_class == test_res(test_idx)
                num_correct = num_correct + 1; 
            end
        end
        acc = (num_correct/size(test_vals,1));
    else
        [img_class, acc, dec_values_P] = svmpredict(test_res, test_vals, mdl, '-q');
        acc = acc(1) / 100;
    end
    
    % Print out the correct percentage
    if (dispVal)
        fprintf('correct_test = %f\n', acc);
    end
end
