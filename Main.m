clc;
clear;
datasets = ["emotions.mat"];
kernelParm = 0.3;

for iiii = 1:length(datasets)
    dataset = string(datasets(iiii));

    rng(0);
    load(dataset);

    X = data;
    Y = target';
    clear data ;
    clear target;


    %% Randomly select part of data if very large
    random_select = 1; % set random_select = 1 to select otherwise 0
    max_num = 25000;
    if (size(X,1) > max_num) && random_select
        nRows = size(X,1);
        nSample = max_num;
        rndIDX = randperm(nRows);
        index = rndIDX(1:nSample);
        X = X(index, :);
        Y = Y(index,:);
    end
    
    [~, num_class] = size(Y);
    
    %nomalization of the data
    X = svdatanorm(X, 'ker');
    
    percentage = [0.3];
    num_fold = 5; num_metric = 16; num_method = 1;
    % accuracy = zeros(1, 10);
    
    indices = crossvalind('Kfold', size(X, 1), num_fold);
    Results = zeros(num_metric + 2, num_fold, num_method);
    Results_AGMR = zeros(num_metric + 1, num_fold, num_method);
    %matrix for stroing parameter values for each fold and percentage;\
    
    for i = 1:num_fold
        disp(['Fold ', num2str(i)]);
        test = (indices == i); train = ~test;
        train_data = X(train, :);
        test_data = X(test, :);
        target_train = Y(train, :);
        target_test = Y(test, :);
        
        
        for ii = 1:length(percentage)
            per = percentage(ii);
            
            labelled_data = []; labelled_target = []; train_data_unlabel = [];
            
            if num_class == 1
                [labelled_data, labelled_target, train_data_unlabel] = masking_data(train_data, target_train, per);
            else
                [labelled_data, labelled_target, train_data_unlabel] = multiclass_masking_data(train_data, target_train, per);
            end
            params = struct();
            % data.Y = [labelled_target; zeros(size(train_data_unlabel,1),1) ];

            
            %===================================================================
            
            params.Xlabel = labelled_data;
            params.X = [labelled_data; train_data_unlabel];
            params.Ylabel = labelled_target;
            params.kernel = 'rbf';
            params.kernelparam = kernelParm;
            params.gamma1 = 1;
            params.gamma2 = 1;
            params.max_iter = 100;
            params.eta = 0.1;
            params.eta1 = 0.1;
            params.eta2 = 0.1;
            params.epsilon = 0.01;
            params.Xtest = test_data;
            tic;
            [Pre_Labels,~] = multi_class_AGMR_sparse(params);
            time5 = toc;
            Results_AGMR(1, i, ii) = time5;
            Pre_Labels(Pre_Labels(:, :) == ~1) = -1;
            tmpResult = EvaluationAll(Pre_Labels', Pre_Labels',target_test');
            Results_AGMR(2:end, i, ii) = tmpResult';
            % accuracy(i) = sum(Pre_Labels == target_test) / length(target_test);

            %% ===================================================
        end
    end
    
    
    ignore = []; Results_AGMR(:, :, ignore) = [];
    meanResults_AGMR = squeeze(mean(Results_AGMR, 2));
    stdResults_AGMR = squeeze(std(Results_AGMR, 0, 2) / sqrt(size(Results_AGMR, 2)));
    
    %     ignore = []; Results_LGMPM(:, :, ignore) = [];
    %     meanResults_LGMPM = squeeze(mean(Results_LGMPM, 2));
    %     stdResults_LGMPM = squeeze(std(Results_LGMPM, 0, 2) / sqrt(size(Results_LGMPM, 2)));
    %% Save the evaluation results
    % filename = strcat('results/', dataset, '.mat');
    % % save(filename, 'meanResults', 'stdResults', '-mat');
    % 
    %% Show the experimental results
    disp(dataset);
    % disp("LapSVM");
    % disp(meanResults);
    
    disp("AGMR");
    % disp(meanResults_AGMR);
    % meanAccuracy = mean(accuracy);
    % disp(meanAccuracy*100);
    % std_dev = std(accuracy);
    % disp(std_dev*100);
    disp("AGMR");
    % create a vector of metric 
    metric = ["time", "HammingLoss", "ExampleBasedAccuracy", "ExampleBasedPrecision", "ExampleBasedRecall", "ExampleBasedFmeasure", "SubsetAccuracy", "LabelBasedAccuracy", "LabelBasedPrecision", "LabelBasedRecall", "LabelBasedFmeasure", "MicroF1Measure", "Average_Precision", "OneError", "RankingLoss", "Coverage"];
    for i = 1:length(metric)
        % print in a formatted way
        disp(sprintf('%-20s: %-10.4f %-10.4f', metric(i), meanResults_AGMR(i), stdResults_AGMR(i)));
    end

    %
    %     disp("LGMPM");
    %     disp(meanResults_LGMPM);
    %
    %     curr_dir = pwd;
    %     temp_dataset = "result_"+dataset;
    %     save(fullfile(curr_dir, '/HGSVM_results/', temp_dataset), 'meanResults', 'stdResults', '-mat');
    %     save(fullfile(curr_dir, '/LapMPM_result/', temp_dataset), 'meanResults_LGMPM', 'stdResults_LGMPM', '-mat');
    %     save(fullfile(curr_dir, '/HGMPM_results/', temp_dataset), 'meanResults_HGMPM', 'stdResults_HGMPM', '-mat');
    %     save(fullfile(curr_dir, '/parameter/', temp_dataset), 'parameter_HGMPM', '-mat');
end