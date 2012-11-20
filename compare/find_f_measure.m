function f = find_f_measure( source, target)
%function f = find_f_measure( source, target)
%Function that calculates 2.recall.precision/(recall+precision)

source_positives = find( source == 255);
source_negatives = find( source == 0);
target_positives = find( target == 255);
target_negatives = find( target == 0);
true_positives = intersect( source_positives, target_positives);
TP = size(true_positives, 1);
false_positives = intersect( source_positives, target_negatives);
FP = size(false_positives, 1);
false_negatives = intersect( source_negatives, target_positives);
FN = size(false_negatives, 1);
f= 2*TP/(2*TP + FN + FP);

