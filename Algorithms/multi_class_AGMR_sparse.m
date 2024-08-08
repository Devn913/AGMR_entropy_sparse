function [PreLabel,time] = multi_class_AGMR_sparse(params)

    numClasses = size(params.Ylabel, 2);

    % Initialize the matrix to hold the scores for each class
    PreLabel = zeros(size(params.Xtest, 1), numClasses);

    full_time = 0;

    % Create a one-vs-rest classifier for each class
    for i = 1:numClasses

        % Create a binary target vector for the current class
        target = params.Ylabel(:, i);
        % pos_index = (target == 1);
        neg_index = (target ~= 1);

        target(neg_index) = -1;
        outparams = params;
        outparams.Ylabel = target;


        % Train the classifier and get the prediction scores
        [PreLabel(:, i), time] = binary_AGMR_Sparse(outparams);
        full_time = full_time + time;
    end

end