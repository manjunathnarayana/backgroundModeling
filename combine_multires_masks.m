function combined_mask = combine_bg_masks( individual_masks_cell)
%function combined_mask = combine_bg_masks( individual_masks_cell)
%function that returns a geometric mean of mask probabilities by indexing into the correct location in the individual_masks_cell

num_resolutions = size( individual_masks_cell, 2);
if num_resolutions == 1
    combined_mask = individual_masks_cell{1};   
    return
end

[num_rows num_cols] = size( individual_masks_cell{1});
geo_mean = individual_masks_cell{1};
for res=2:num_resolutions
    geo_mean = geo_mean.*( imresize( individual_masks_cell{res}, [num_rows num_cols], 'bilinear'));
end
combined_mask = geo_mean.^(1.0/num_resolutions);
