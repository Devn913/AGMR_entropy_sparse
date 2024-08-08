function [data_labelled, target_labelled, data_unlabelled] = masking_data(data, target, per)
[n c] = size(target); % samples x classes
[~,f] = size(data); % features

X = [data target];

D1 = X(X(:,end)==1,:);
D2 = X(X(:,end)~=1,:);
fix_per = round(0.1*n*0.5);

new_X = [D1(1:fix_per,:); D2(1:fix_per,:)];
D1(1:fix_per,:) = [];
D2(1:fix_per,:) = [];
D = [D1;D2];

per = per - 0.1;


if (per ~= 0)
    nper = round(per*size(D,1));
    rndIDX = randperm(size(D,1));
    index = rndIDX(1:nper);
    data_labelled = [new_X; D(index,:)];
    D(index,:) = [];
else
    data_labelled = new_X;
end

target_labelled = data_labelled(:,end);
data_labelled(:,end) = [];

data_unlabelled = D;
data_unlabelled(:,end) =[]; 