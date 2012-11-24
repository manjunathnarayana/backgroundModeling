%Script that takes the predictions in the workspace, applies a filter that throws away any regions less than 15 pixels in size and computes the f_measure

if ~exist('size_threshold');
    size_threshold = 15;
end
if ~exist('p_threshold')
    p_threshold = .5;
end

for i = 1:size( gt_frames_truth, 3)
    pred_img = gt_frames_prediction(:,:,i);
    truth_img = gt_frames_truth(:,:,i);
    pred_img_threshed = ( pred_img<p_threshold);
    pred_img_filtered = make_small_regions_zero( pred_img_threshed, size_threshold);
    pred_img_filtered = double( pred_img_filtered*255);
    f_measure_clean = find_f_measure( pred_img_filtered, truth_img);
    f_measures_clean( i ) = f_measure_clean;
end
f_measure_cleaned_average = sum(f_measures_clean)/size( gt_frames_truth,3)


