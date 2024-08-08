function [labels,time] = binary_AGMR_entropy(params)
    % params.Xlabel: labeled data
    % params.X: All data
    % params.Ylabel: labeled label
    % params.kernel: kernel type
    % params.kernelparam: kernel parameter
    % params.gamma1: gamma1
    % params.gamma2: gamma2
    % params.max_iter: max_iter
    % params.eta: eta
    % params.epsilon: epsilon
    % params.Xtest : test data

    number_of_labelled_examples       = size(params.Xlabel,1);
    number_of_unlabelled_examples     = size(params.X,1) - number_of_labelled_examples;
    number_of_features                = size(params.X,2);
    number_of_classes                 = length(unique(params.Ylabel));
    n                                 = number_of_labelled_examples + number_of_unlabelled_examples;
    K_l                               = evalkernel(params.Xlabel,params.X,params.kernel,params.kernelparam); 
    params.K                          = evalkernel(params.X,params.X,params.kernel,params.kernelparam);
    W                                 = eye(number_of_labelled_examples+number_of_unlabelled_examples);
    L                                 = cal_laplacian(W);
    %inverse of matrix

    alpha                             = inv(params.gamma1*transpose(K_l)*K_l + params.gamma2*params.K + transpose(params.K)*L*params.K)*params.gamma1*K_l'*params.Ylabel;
    iteration                         = 1;
    % set j  to 
    tic;
    while iteration<params.max_iter
        F_x = sign(params.K*alpha);
        F_l = sign(K_l*alpha);
        % write this in code w[i][j] = exp(-((F_x[i]-F_x[j])^2)/eta) / sum((j = 1 to n) exp(-((F_x[i]-F_x[j])^2)/eta))
        old_val = optimization_function(W,F_x,params.Ylabel,params,L,F_l);
        for i = 1:n
            for j = 1:n
                W(i,j) = exp(-((F_x(i)-F_x(j))^2)/params.eta);
            end
        end
        W(i,j)     = W(i,j) / sum(W(i,:));
        L          = cal_laplacian(W);
        alpha      = inv(params.gamma1*transpose(K_l)*K_l + params.gamma2*params.K + transpose(params.K)*L*params.K)*params.gamma1*K_l'*params.Ylabel;
        iteration  = iteration + 1;
        F_x        = sign(params.K * alpha);
        F_l        = sign(K_l * alpha);
        if(optimization_function(W,F_x,params.Ylabel,params,L,F_l) -old_val ) < params.epsilon
            break;
        end
    end
    time = toc;
    labels = sign(evalkernel(params.Xtest,params.X,params.kernel,params.kernelparam)*alpha);
end
function val = optimization_function(W, F,Y,params,L,F_l)
    % sum(w[i][j] * (f(x[i]) - f(x[j]))^2) + gamma1 * sum(i=1:labelled_example ||f(x_i) - y_i||^2) + gamma2 * f(x)^T * L * f(x)
    val = 0;
    for i = 1:size(W,1)
        for j = 1:size(W,2)
            val = val + W(i,j) * (F(i) - F(j))^2;
        end
    end
    val = val  + params.gamma2 * transpose(F) * L * F;
    % val = val + gamma1 * (Summation of (f(x_i) - y_i)^2 for i = 1 to number_of_labelled_examples)
    val = val + params.gamma1 * transpose(F_l - Y) * (F_l - Y);

end

