(1) Instructions to run code    
    (i) Main script is run_domain_range_bg_modeling.m (for color feature model only)
    or run_domain_range_bg_modeling_hybrid.m (for color-texture hybrid feature
    model)
    (ii)  Change video number to select appropriate video in above scripts. 
    Note that video_numbers can be an array of video numbers which will be 
    processed in sequence
    (ii)  Change path to input video, input groundtruth, output folder in load_video.m
    (iii) Set algorithm_to_use variable in the main script
    (iv)  Set any optional parameters in the main script
    Setting display_general to 1 displays the segmentation for each frame in a figure
    (v)   Run main script in matlab. 
    Note that it may take several hours for complete video
    sequence to be processed. If you desire to run script for a smaller subset of
    video frames, change the total_num_frames variable in load_video.m
    (vi) Output will be saved in the folder specified by
    output_sequences_folder in load_video.m
    Output is a mat file with all processed frames, all output frames, bg
    masks, and some input parameters. 
    (vii) Output video segmentation may be observed by using 
    step_through_color_image_sequence_pair( image_stack, segmented_image_stack);
    This is automatically invoked if a single video is in video_numbers. If
    multiple videos are processed, then the user has to load the mat files
    that were saved and then call step_through_color_image_sequence_pair
    manually

(2) Credits for third-party code used
    Thanks to Michael Rubinstein for max flow code (obtained from
    matlabcentral)
    Thanks to Mark A. Ruzon for RGB2Lab code (obtained from matlabcentral)
