function [TP TN FP FN] = calculate_TP_FP( source, target)
%function [TP TN FP FN] = calculate_TP_FP( source, target)
%Function that computes TP, TN, FP, FN for given source image and target truth image 
%In both images, bg must be 0, fg must be 255

source_positives = find( source == 255);
source_negatives = find( source == 0);
target_positives = find( target == 255);
target_negatives = find( target == 0);
true_positives = intersect( source_positives, target_positives);

TP = size(true_positives, 1);
true_negatives = intersect( source_negatives, target_negatives);
TN = size(true_negatives, 1);
false_positives = intersect( source_positives, target_negatives);
FP = size(false_positives, 1);
false_negatives = intersect( source_negatives, target_positives);
FN = size(false_negatives, 1);
