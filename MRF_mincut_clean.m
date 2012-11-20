function bg_mask = MRF_mincut_clean( input_bg_mask, input_fg_mask, lambda)
%function bg_mask = MRF_mincut_clean( input_mask, lambda)
% returns bg_mask after MRF clean up using min-cut procedure. Each node is connected by lambda in the MRF.
[rows cols] = size(input_bg_mask);
%edges
E = edges4connected( rows, cols);
%connecting edges with weight = lambda
V = lambda;
%sparse representation
A = sparse( E(:,1), E(:,2), V, rows*cols, rows*cols, 4*rows*cols);

%Connect the image nodes to the label nodes (source, sink). Connect to source if likelihood ratio greater than zero or to sink (after multiplying by -1) if less than zero

ratio_img = -log(input_bg_mask./(input_fg_mask+eps));
%positive ratios
plus_indices = find(ratio_img(:)>0);
num_plus = size( plus_indices, 1);
plus_ratios = ratio_img( plus_indices);
%negative ratios
minus_indices = find(ratio_img(:)<=0);
num_minus = size( minus_indices, 1);
minus_ratios = -ratio_img( minus_indices);


T = sparse( [plus_indices; minus_indices], [ones(num_plus,1); ones(num_minus,1)*2], [plus_ratios; minus_ratios]);
%fprintf('performing max flow to filter bg pixels\n');

if num_plus>0 && num_minus>0
    [max_flow_value, labels] = maxflow( A, T);
    bg_mask = double(reshape( labels, [rows cols]));
else
    bg_mask = input_bg_mask;    
end
