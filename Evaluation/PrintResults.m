
function PrintResults(Result)
fprintf('------------------------------------\n');
fprintf('Evaluation Result For MYproblem  is  as follow\n');
fprintf('Evaluation Metric         Mean         Std\n');
fprintf('------------------------------------\n');
fprintf('HammingLoss                    %.4f  %.4f\r',Result(1,1),Result(1,2));
fprintf('ExampleBasedAccuracy /Accuracy %.4f  %.4f\r',Result(2,1),Result(2,2));
fprintf('ExampleBasedPrecision          %.4f  %.4f\r',Result(3,1),Result(3,2));
fprintf('ExampleBasedRecall             %.4f  %.4f\r',Result(4,1),Result(4,2));
fprintf('ExampleBasedFmeasure/ F1       %.4f  %.4f\r',Result(5,1),Result(5,2));

fprintf('SubsetAccuracy                 %.4f  %.4f\r',Result(6,1),Result(6,2));
fprintf('LabelBasedAccuracy             %.4f  %.4f\r',Result(7,1),Result(7,2));
fprintf('LabelBasedPrecision            %.4f  %.4f\r',Result(8,1),Result(8,2));
fprintf('LabelBasedRecall               %.4f  %.4f\r',Result(9,1),Result(9,2));
fprintf('LabelBasedFmeasure/Macro F     %.4f  %.4f\r',Result(10,1),Result(10,2));
fprintf('MicroF1Measure                 %.4f  %.4f\r',Result(11,1),Result(11,2));
fprintf('Average_Precision              %.4f  %.4f\r',Result(12,1),Result(12,2));
fprintf('OneError                       %.4f  %.4f\r',Result(13,1),Result(13,2));
fprintf('RankingLoss                    %.4f  %.4f\r',Result(14,1),Result(14,2));
fprintf('Coverage                       %.4f  %.4f\r',Result(15,1),Result(15,2));
fprintf('AUC                            %.4f  %.4f\r',Result(16,1),Result(16,2));
fprintf('------------------------------------\n');
NewResult=zeros(7,2);
NewResult(1,:)=Result(1,:);
NewResult(2,:)=Result(6,:);
NewResult(3,:)=Result(11,:);
NewResult(4,:)=Result(12,:);
NewResult(5,:)=Result(13,:);
NewResult(6,:)=Result(14,:);
NewResult(7,:)=Result(16,:);
var='L69:L75';
%xlswrite('D:\Sayed Mortaza\Reports\Eval0.6.xlsx',NewResult(:,1),var);
%xlswrite('D:\Sayed Mortaza\Reports\var0.6eval.xlsx',NewResult(:,2),var);
end