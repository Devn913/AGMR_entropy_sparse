function [data_labelled, target_labelled, data_unlabelled] = multiclass_masking_data(data, target, per)

    % works for multiclass and multilabel data
    [n, c] = size(target); % samples x classes
    [~, f] = size(data); % features

    X = [data target];

    new_X = []; D = []; index = [];

    %separate data according to classes
    for i = 1:c
        fix_per = round((0.1 * n) / c); % percentage of data fixed for each class
        index = X(:, f + i) == 1;
        Dc{i} = X(index, :);
        num_class_pts = size(Dc{i}, 1);
        X(index, :) = [];

        if fix_per > num_class_pts
            fix_per = num_class_pts;
        end

        new_X = [new_X; Dc{i}(1:fix_per, :)];
        Dc{i}(1:fix_per, :) = [];
        D = [D; Dc{i}];
        index = [];
    end

    percen = per - 0.1;

    if (percen ~= 0)
        nper = round(percen * n);
        rndIDX = randperm(size(D, 1));

        if (nper > size(D, 1))
            nper = round(percen * size(D, 1));
        end

        index = rndIDX(1:nper);
        data_labelled = [new_X; D(index, :)];
        D(index, :) = [];
    else
        data_labelled = new_X;
    end

    target_labelled = data_labelled(:, f + 1:end);
    data_labelled(:, f + 1:end) = [];

    data_unlabelled = D;
    data_unlabelled(:, f + 1:end) = [];
end