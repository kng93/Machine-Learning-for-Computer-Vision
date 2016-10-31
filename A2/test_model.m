% Test the model (on the test data set)
function [acc] = test_model(test_data, mdl, dispVal)

    if nargin < 3
        dispVal = true(1);
    end
    
    % Split up the result from the data
    test_res = test_data(:, end);
    test_vals = test_data(:, 1:(end-1));
    
    % Go over every test value
    num_correct = 0;
    for test_idx = 1:size(test_vals,1)
       img_class = predict(mdl, test_vals(test_idx,:));
       
       % Count the number of correct values
       if img_class == test_res(test_idx)
          num_correct = num_correct + 1; 
       end
    end
    
    % Print out the correct percentage
    acc = (num_correct/size(test_vals,1));
    if (dispVal)
        fprintf('correct_test = %f\n', acc);
    end
end
