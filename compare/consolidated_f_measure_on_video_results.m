%Script that takes the predictions in the workspace, applies a filter that throws away any regions less than 15 pixels in size and computes the TP,FP, FN in all groundtruth frames and returns overall F-measure (Different from average F-measure over the frames)

if ~exist('size_threshold');
    size_threshold = 15;
end
if ~exist('p_threshold')
    p_threshold = .5;
end
current_video_TP = 0;
current_video_FN = 0;
current_video_FP = 0;

if ~isempty( gt_frames_prediction )
    for i = 1:size( gt_frames_truth, 3)
        pred_img = gt_frames_prediction(:,:,i);
        truth_img = gt_frames_truth(:,:,i);
        pred_img_threshed = ( pred_img<p_threshold);
        pred_img_filtered = make_small_regions_zero( pred_img_threshed, size_threshold);
        pred_img_filtered = double( pred_img_filtered*255);
        [TP TN FP FN] = calculate_TP_FP( pred_img_filtered, truth_img);
        current_video_TP = current_video_TP+TP;
        current_video_FN = current_video_FN+FN;
        current_video_FP = current_video_FP+FP;
    end
end
overall_f_measure = 2*current_video_TP/(2*current_video_TP + current_video_FN + current_video_FP)
