%/*
%* Copyright (c) 2012, Manjunath Narayana, UMass-Amherst
%* All rights reserved.
%*
%* Redistribution and use in source and binary forms, with or without
%* modification, are permitted provided that the following conditions are met:
%*     * Redistributions of source code must retain the above copyright
%*       notice, this list of conditions and the following disclaimer.
%*     * Redistributions in binary form must reproduce the above copyright
%*       notice, this list of conditions and the following disclaimer in the
%*       documentation and/or other materials provided with the distribution.
%*     * Neither the name of the author nor the organization may be used to 
%*       endorse or promote products derived from this software without specific 
%*       prior written permission.
%*
%* THIS SOFTWARE IS PROVIDED BY Manjunath Narayana ``AS IS'' AND ANY
%* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
%* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
%* DISCLAIMED. IN NO EVENT SHALL <copyright holder> BE LIABLE FOR ANY
%* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
%* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
%* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
%* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
%* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
%* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%*/

%Usage - Change video number and run code. To add more videos,
%change load_video.m file

addpath( genpath ('utils'));
addpath( genpath ('compare'));
addpath( genpath ('max_flow_min_cut_wrapper'));
addpath( genpath ('siltp_color_hybrid'));

%Video number - Can use one number at a time or use an array of numbers
video_numbers = [1]; %3600, 
video_numbers = [2]; %3100
%video_numbers = [3]; %2000
%video_numbers = [4]; %3500
%video_numbers = [5]; %500
%video_numbers = [6]; %1300
%video_numbers = [12]; %1300
%video_numbers = [7]; %1600
%video_numbers = [8]; %1500
%video_numbers = [9]; %600
%video_numbers = [10]; %500
%video_numbers = [11]; %318
%video_numbers = [1:11]; % all videos one by one

%display video_numbers
video_numbers

%Run for all videos listed in video_numbers
for video_number = video_numbers
    
    %Algorithm options
    %(1) BMVC 2012 - joint domain-range + adaptive kernel variance 
    %(2) CVPR 2012 - adaptive kernel variance
    %(3) Our implementation of Sheikh-Shah
    algorithm_to_use = 2;

    %Call load video script
    load_video

    video_number

    disp( videoname);
    disp( input_sequence_files_suffix);
    
    %display options
    display_general = 1;

    %max frames to classify
    max_track_frame = total_num_frames;
    
    %num frames used for bg KDE
    num_bg_frames = 50;

    %num frames used for fg KDE
    num_fg_frames = 5;

    %TODO - explain this
    fg_uniform_factor = 0.5;

    %Algorithm options
    %use LAB or RGB color space
    use_LAB = 1;
    
    %Formulate as a three class problem - bg v/s seen foreground v/s new foreground
    %Must be set to 1 for BMVC 2012 results
    use_bg_fg_new_classes = 0;
    
    %Suboptions for each algorithm
    %For all algorithms
    
    %To choose reinitializing the bg model when sudden intensity change is detected
    use_intensity_change_reasoning = 1;
    %Percent of pixels in the image that must be different to start reinitialization
    lumin_percent_threshold = 50;
    %lumin values that differ by more than threshold are used to detect change
    lumin_value_threshold = 10;
    if use_LAB
        lumin_value_threshold = 2.5;
    end
    
    %Post processing clean-up using MRF
    use_MRF_clean_procedure = 1;
    %MRF lambda parameter, if using MRF clean-up
    MRF_lambda = 1;
    
    %Multiple resolutions (Pyramid) for SILTP computation
    num_siltp_resolutions = 3;
    %number of resolutions for color features (NOTE - Untested for greater than 1 )
    %TODO - Test and make it work for higher resolutions
    use_multi_resolution = 1;
    num_resolutions = 3;
    if use_multi_resolution==0
        num_resolutions = 1;
    end
    
    %Cache method from CVPR 2012 paper
    %this uses center pixel + neighborhood values to decide bg fg lik
    use_efficient_sigma_cache = 1;
    if use_efficient_sigma_cache==1
        bg_sigma_images_old = cell(1, num_resolutions);
        fg_sigma_images_old = cell(1, num_resolutions);
    end
    
    %Flag that is set when a reinitialization is in progress
    reinitialize_mode_flag = 0;
    %Count of number of frames since reinitialization has begun
    num_reinitialize_frames = 0;
    
    %How bg and fg models are updated
    %TODO - Explain the options
    bg_update_version = 3;
    fg_update_version = 1;

    %TODO - Print information about method being used, etc.

    %Threshold for SILTP feature generation
    %TODO - Explain this
    SILTP_threshold = .05;
        
    %How bg and fg models are updated
    %TODO - Explain the options
    bg_update_version = 3;
    fg_update_version = 1;

    disp('Classifying pixels as bg/fg using hybrid feature space (SILTP+color) in joint domain-range model');
    
    fprintf('num resolutions = %d\n', num_resolutions);
    
    if use_LAB
        fprintf('Using LAB colorspace');
    else
        fprintf('Using RGB colorspace');
    end

    % First generate a bg model with num_bg_frames. (rows*cols*num_bg_frames)x5 in size. Reads [y, x, red, green, blue]
    bg_model = cell(1, num_resolutions);
    fg_model = cell(1, num_resolutions);
    
    %indicator images that tell which pixels in the above models are valid pixels for bg/fg
    bg_indicator = cell(1, num_resolutions);
    fg_indicator = cell(1, num_resolutions);
    
    %Read a temporary image to get image size
    temp_image = imread(sprintf('%s/%s%03d.%s', input_video_folder, input_sequence_files_suffix, seq_starts_from+0, file_ext));
    num_rows = size(temp_image, 1);
    num_cols = size(temp_image, 2);
    clear temp_image;
    
    %Initialize a bg model from required number of frames
    bg_frame_num = 0;
    for i = bg_frames_start:min(bg_frames_end, num_bg_frames)
    %for i = 46:50
        img = imread(sprintf('%s/%s%03d.%s', bg_input_video_folder, bg_input_sequence_files_suffix, seq_starts_from+ i, file_ext));
        img_g = rgb2gray( img );
        if use_LAB==1
            img = rgb2LAB(double(img));
        end

        %Compute SILTP features from image
        img_siltp = im2quadsiltp( img_g, SILTP_threshold );
        %Get SILTP pixel values as joint domain-range kernel points
        img_siltp_pixels = get_gray_kernel_points_2d_from_image(img_siltp);
        
        %Compute SILTP features for all resolutions
        for siltp_res = 2:num_siltp_resolutions
            %subsample the image    
            img_g_sampled = subsample_image( img_g, 1.0/siltp_res);
            %compute SILTP features for subsampled image
            img_siltp_sampled = im2quadsiltp( img_g_sampled, SILTP_threshold);
            %resize the SILTP features to be same size as original image
            img_siltp_resampled = imresize( img_siltp_sampled, [num_rows num_cols], 'nearest');
            %Get SILTP pixel values as joint domain-range kernel points
            temp_siltp_pixels_sampled = get_gray_kernel_points_2d_from_image( img_siltp_resampled);
            %Concatinate the SILTP features from subsampled image to KDE samples
            img_siltp_pixels(:,:,end+1) = temp_siltp_pixels_sampled(:,:,3);
        end

        %Get image pixels as joint domain-range kernel points
        temp_pixels_samples = get_kernel_points_2d_from_image(img);
        %Make the last few dimensions of the pixel samples be the SILTP values
        %temp_pixels_samples now has XY, R, G, B, SILTP_res_1, SILTP_res_2, ... SILTP_res_k
        temp_pixels_samples(:,:,6:6+num_siltp_resolutions-1) = img_siltp_pixels(:,:,3:end);
        
        %Update the bg model with pixel samples from above
        bg_model{1}(bg_frame_num+1,:,:,:,:,:,:,:,:) = temp_pixels_samples;
        bg_indicator{1}(bg_frame_num+1,:,:) = ones(num_rows, num_cols);
    
        %Populate models for multiple resolutions as desired
        for res = 2:num_resolutions
            sampled_img = subsample_image( img, 1.0/res );
            [num_sampled_rows num_sampled_cols num_sampled_colors] = size( sampled_img);
            
            %Compute SILTP features from subsampled image
            subsampled_img_g = rgb2gray( sampled_img );
            sub_img_siltp = im2quadsiltp( subsampled_img_g, SILTP_threshold );
            %Get SILTP pixel values as joint domain-range kernel points
            sub_img_siltp_pixels = get_gray_kernel_points_2d_from_image(sub_img_siltp);

            %Compute SILTP features for all resolutions
            for siltp_res = 2:num_siltp_resolutions
                %subsample the image    
                sub_img_g_sampled = subsample_image( subsampled_img_g, 1.0/siltp_res);
                %compute SILTP features for subsampled image
                sub_img_siltp_sampled = im2quadsiltp( sub_img_g_sampled, SILTP_threshold);
                %resize the SILTP features to be same size as original image
                sub_img_siltp_resampled = imresize( sub_img_siltp_sampled, [num_sampled_rows num_sampled_cols], 'nearest');
                %Get SILTP pixel values as joint domain-range kernel points
                sub_temp_siltp_pixels_sampled = get_gray_kernel_points_2d_from_image( sub_img_siltp_resampled);
                %Concatinate the SILTP features from subsampled image to KDE samples
                sub_img_siltp_pixels(:,:,end+1) = sub_temp_siltp_pixels_sampled(:,:,3);
            end

            %Get image pixels for subsampled image as joint domain-range kernel points
            sub_temp_pixels_samples = get_kernel_points_2d_from_image(sampled_img);
            %Make the last few dimensions of the pixel samples be the SILTP values
            %temp_pixels_samples now has XY, R, G, B, SILTP_res_1, SILTP_res_2, ... SILTP_res_k
            sub_temp_pixels_samples(:,:,6:6+num_siltp_resolutions-1) = sub_img_siltp_pixels(:,:,3:end);
            bg_model{res}(bg_frame_num+1,:,:,:,:,:,:,:) = sub_temp_pixels_samples;
            bg_indicator{res}(bg_frame_num+1,:,:) = ones(num_sampled_rows, num_sampled_cols);
        end
        bg_frame_num = bg_frame_num+1;
    end
    
    %TODO- Why is bsr important
    if algorithm_to_use == 1
        bg_prior = ones( num_rows, num_cols)*.98;
        boundary_region_size = 10;
        bsr = boundary_region_size;
        bg_prior(1:bsr,:) = 0;
        bg_prior(end-bsr+1:end,:) = 0;
        bg_prior(:,1:bsr) = 0;
        bg_prior(:, end-bsr+1:end) = 0;
        fg_prior = 1-bg_prior;
    elseif algorithm_to_use == 2
        bg_prior = ones( num_rows, num_cols)*.5;
        fg_prior = ones( num_rows, num_cols)*.5;
    elseif algorithm_to_use == 3
        bg_prior = ones( num_rows, num_cols)*.5;
        fg_prior = ones( num_rows, num_cols)*.5;
    else
        bg_prior = ones( num_rows, num_cols)*.99;
        fg_prior = ones( num_rows, num_cols)*.01;
    end
    
    %Initialize bg and fg masks
    bg_mask = bg_prior;
    fg_mask = fg_prior;
    
    %number of values the color features can take (R, G, B = 0 to 255, hence 256)
    num_color_feature_vals = 256;
    %number of values the siltp features can take (Typically 81)
    num_siltp_feature_vals = 81;
    
    %Sigma values for bg and fg processes
    %For RGB color space sigma values
    if use_LAB == 0
        bg_sigma_XYs{1} = [1/4 3/4];
        bg_sigma_Ls{1}  = [5/4 15/4 45/4];
        bg_sigma_ABs{1}  = [5/4 15/4 45/4];
        bg_sigma_LTPs{1} = [3/4];
        fg_sigma_XYs{1} = [12/4];
        fg_sigma_Ls{1}  = [15/4];
        fg_sigma_ABs{1}  = [15/4];
        fg_sigma_LTPs{1} = [3/4];
    else
        %LAB sigma values
        bg_sigma_XYs{1} = [1/4 3/4];
        bg_sigma_Ls{1}  = [5/4 10/4 20/4];
        bg_sigma_ABs{1}  = [4/4 6/4];
        bg_sigma_LTPs{1} = [3/4];
        fg_sigma_XYs{1} = [12/4];
        fg_sigma_Ls{1}  = [15/4];
        fg_sigma_ABs{1}  = [4/4];
        fg_sigma_LTPs{1} = [3/4];
    end

    fprintf('bg_sigma_XYs{1}\n')
    bg_sigma_XYs{1}(:)'
    fprintf('bg_sigma_Ls{1}\n')
    bg_sigma_Ls{1}(:)'
    fprintf('bg_sigma_ABs{1}\n')
    bg_sigma_ABs{1}(:)'
    fprintf('bg_sigma_LTPs{1}\n')
    bg_sigma_LTPs{1}(:)'
    fprintf('fg_sigma_XYs{1}\n')
    fg_sigma_XYs{1}(:)'
    fprintf('fg_sigma_Ls{1}\n')
    fg_sigma_Ls{1}(:)'
    fprintf('fg_sigma_ABs{1}\n')
    fg_sigma_ABs{1}(:)'
    fprintf('fg_sigma_LTPs{1}\n')
    fg_sigma_LTPs{1}(:)'
    
    %If using multi-resolution, set up sigma values for each resolution
    if use_multi_resolution
        % Resolution 2
        bg_sigma_XYs{2} = [3/4];
        bg_sigma_Ls{2} = [10/4];
        bg_sigma_ABs{2} = [2/4];
        bg_sigma_LTPs{2} = [3/4];
        fg_sigma_XYs{2} = [3/4];
        fg_sigma_Ls{2} = [10/4];
        fg_sigma_ABs{2} = [2/4];
        fg_sigma_LTPs{2} = [3/4];
        % Resolution 3
        bg_sigma_XYs{3} = [3/4];
        bg_sigma_Ls{3} = [10/4];
        bg_sigma_ABs{3} = [2/4];
        bg_sigma_LTPs{3} = [3/4];
        fg_sigma_XYs{3} = [3/4];
        fg_sigma_Ls{3} = [10/4];
        fg_sigma_ABs{3} = [2/4];
        fg_sigma_LTPs{3} = [3/4];
        fprintf('bg_sigma_XYs{2}\n')
        bg_sigma_XYs{2}(:)'
        fprintf('bg_sigma_Ls{2}\n')
        bg_sigma_Ls{2}(:)'
        fprintf('bg_sigma_ABs{2}\n')
        bg_sigma_ABs{2}(:)'
        fprintf('fg_sigma_XYs{2}\n')
        fg_sigma_XYs{2}(:)'
        fprintf('fg_sigma_Ls{2}\n')
        fg_sigma_Ls{2}(:)'
        fprintf('fg_sigma_ABs{2}\n')
        fg_sigma_ABs{2}(:)'
        fprintf('bg_sigma_XYs{3}\n')
        bg_sigma_XYs{3}(:)'
        fprintf('bg_sigma_Ls{3}\n')
        bg_sigma_Ls{3}(:)'
        fprintf('bg_sigma_ABs{3}\n')
        bg_sigma_ABs{3}(:)'
        fprintf('fg_sigma_XYs{3}\n')
        fg_sigma_XYs{3}(:)'
        fprintf('fg_sigma_Ls{3}\n')
        fg_sigma_Ls{3}(:)'
        fprintf('fg_sigma_ABs{3}\n')
        fg_sigma_ABs{3}(:)'
    end

    %Consolidate the sigma values into a cell array for bg and fg
    bg_sigmas{1} = bg_sigma_XYs;
    bg_sigmas{2} = bg_sigma_Ls;
    bg_sigmas{3} = bg_sigma_ABs;
    bg_sigmas{4} = bg_sigma_LTPs;
    fg_sigmas{1} = fg_sigma_XYs;
    fg_sigmas{2} = fg_sigma_Ls;
    fg_sigmas{3} = fg_sigma_ABs;
    fg_sigmas{4} = fg_sigma_LTPs;
    
    if (size(bg_sigma_XYs, 2)~=num_resolutions) || (size(fg_sigma_XYs, 2)~=num_resolutions)
        error(' number of candidate sigmas should match the number or resolutions');
    end
    
    %definition of neighborhood for kde calculation
    %pixels in a range + and - this value are used as samples in computing KDE likelihoods
    bg_near_rows = ceil( max( bg_sigma_XYs{1})*4/2);
    bg_near_cols = bg_near_rows;
    fg_near_rows = ceil( max( fg_sigma_XYs{1})*4/2);
    fg_near_cols = fg_near_rows;
    
    %Variables to store accuracy statistics and groundtruth related data
    f_measure_sum=0;
    f_measures = [];
    gt_count = 0;
    gt_frames_prediction = [];
    gt_frames_truth = [];
    gt_frames_image = [];
    
    %For each frame that has to be classified
    for track_frame = skip_until_frame-seq_starts_from:max_track_frame-seq_starts_from
        frame_start_time = tic;
        tic
        fprintf('%d.', seq_starts_from+track_frame);
        img = imread(sprintf('%s/%s%03d.%s', input_video_folder, input_sequence_files_suffix, seq_starts_from+track_frame, file_ext));
        img_orig = img; 
        img_g = rgb2gray( img );
        img_siltp = im2quadsiltp( img_g, SILTP_threshold );
        img_siltp_pixels = get_gray_kernel_points_2d_from_image(img_siltp);
        
        if use_LAB==1
            img = rgb2LAB(double(img));
        end
    
        %image_stack stores the original image for each frame
        image_stack(:,:,:, track_frame) = img_orig;
        %image_stack_work stores the working image (possibly LAB) for each frame
        image_stack_work(:,:,:, track_frame) = img;
        %bg mask output for each frame
        bg_masks(:, :, track_frame) = zeros(num_rows, num_cols);
        %bg mask before applying any clean-up
        bg_masks_unclean(:, :, track_frame) = zeros(num_rows, num_cols);
        
        %Compute SILTP features for image
        img_siltp = im2quadsiltp( img_g, SILTP_threshold );
        %Get kde samples from SILTP features
        img_siltp_pixels = get_gray_kernel_points_2d_from_image(img_siltp);
        %For all SILTP resolutions
        for siltp_res = 2:num_siltp_resolutions
            img_g_sampled = subsample_image( img_g, 1.0/siltp_res);
            %Compute SILTP features for subsampled image
            img_siltp_sampled = im2quadsiltp( img_g_sampled, SILTP_threshold);
            %Resize the SILTP features image to be same size as original image
            img_siltp_resampled = imresize( img_siltp_sampled, [num_rows num_cols], 'nearest');
            %Get kde samples from SILTP features
            temp_siltp_pixels_sampled = get_gray_kernel_points_2d_from_image( img_siltp_resampled);
            %Concatinate the SILTP features from subsampled image to KDE samples
            img_siltp_pixels(:,:,end+1) = temp_siltp_pixels_sampled(:,:,3);
        end
     
        %Get image pixels as joint domain-range kernel points
        img_pixels{1} = get_kernel_points_2d_from_image(img);
        %Make the last few dimensions of the pixel samples be the SILTP values
        %temp_pixels_samples now has XY, R, G, B, SILTP_res_1, SILTP_res_2, ... SILTP_res_k
        img_pixels{1}(:,:,6:6+num_siltp_resolutions-1) = img_siltp_pixels(:,:,3:end);
        %Populate image pixel values for multiple resolutions as desired
        for res = 2:num_resolutions
            temp_pixels_samples = [];
            sampled_img = subsample_image( img, 1.0/res );
            [num_sampled_rows num_sampled_cols num_sampled_colors] = size( sampled_img);
            
            %Compute SILTP features from subsampled image
            subsampled_img_g = rgb2gray( sampled_img );
            sub_img_siltp = im2quadsiltp( subsampled_img_g, SILTP_threshold );
            %Get SILTP pixel values as joint domain-range kernel points
            sub_img_siltp_pixels = get_gray_kernel_points_2d_from_image(sub_img_siltp);

            %Compute SILTP features for all resolutions
            for siltp_res = 2:num_siltp_resolutions
                %subsample the image    
                sub_img_g_sampled = subsample_image( subsampled_img_g, 1.0/siltp_res);
                %compute SILTP features for subsampled image
                sub_img_siltp_sampled = im2quadsiltp( sub_img_g_sampled, SILTP_threshold);
                %resize the SILTP features to be same size as original image
                sub_img_siltp_resampled = imresize( sub_img_siltp_sampled, [num_sampled_rows num_sampled_cols], 'nearest');
                %Get SILTP pixel values as joint domain-range kernel points
                sub_temp_siltp_pixels_sampled = get_gray_kernel_points_2d_from_image( sub_img_siltp_resampled);
                %Concatinate the SILTP features from subsampled image to KDE samples
                sub_img_siltp_pixels(:,:,end+1) = sub_temp_siltp_pixels_sampled(:,:,3);
            end

            %Get image pixels for subsampled image as joint domain-range kernel points
            sub_temp_pixels_samples = get_kernel_points_2d_from_image(sampled_img);
            %Make the last few dimensions of the pixel samples be the SILTP values
            %temp_pixels_samples now has XY, R, G, B, SILTP_res_1, SILTP_res_2, ... SILTP_res_k
            sub_temp_pixels_samples(:,:,6:6+num_siltp_resolutions-1) = sub_img_siltp_pixels(:,:,3:end);
            %Add these pixel samples to cell array    
            img_pixels{res} = sub_temp_pixels_samples;
        end

        %If a reninitialize is in progress, dont classify this frame.
        %Simply update the bg model with current image pixels
        %bg mask is 1 at all pixels (everything is considered bg)
        if reinitialize_mode_flag==1
            bg_mask = ones(num_rows, num_cols);
            bg_masks_unclean(:,:,track_frame) = bg_mask;
            bg_masks(:,:,track_frame) = bg_mask;
            [bg_model bg_indicator] = update_kde_model( img_pixels, bg_model, bg_indicator, bg_mask, num_bg_frames, bg_update_version); 
            prev_frame_pixels = img_pixels{1};
            %If still not reached num_bg_frames
            if num_reinitialize_frames < num_bg_frames
                num_reinitialize_frames = num_reinitialize_frames + 1;
                fprintf('** bg model reintialization continues at frame %d.. frame took %f seconds\n', seq_starts_from+track_frame, toc(frame_start_time));
            else
                fprintf('** bg model reintialization finished at frame %d.. frame took %f seconds\n. Returning to regular processing\n', seq_starts_from+track_frame, toc(frame_start_time));
                reinitialize_mode_flag = 0;
            end
            continue; 
        end         
        %Here, check if the new frame has significant change in Y from previous frame. If so, then make bg model = []. fg model = []. Set a flag that indicates we are in a reinitializing mode. Once num_bg_frames have passed, set the reinitializing mode to 0. As long as reinitializing_mode is not 0, keep building the bg model, predict all pixels as bg.
        if use_intensity_change_reasoning == 1
            if (exist('prev_frame_pixels'))
                %third dimension is L (in LAB space)
                %compute L value difference between previous and current frame
                prev_frame_lumin = prev_frame_pixels(:,:,3);
                curr_frame_lumin = img_pixels{1}(:,:,3);
                lumin_diff_img = curr_frame_lumin-prev_frame_lumin;
                lumin_threshed_img = abs(lumin_diff_img)>lumin_value_threshold;
                percent_pixels_different = sum(lumin_threshed_img(:))/num_rows/num_cols*100;
                %If more than certain percent pixels have changed much in L values, then reinitialize the bg model by setting the bg model to all zeros
                if percent_pixels_different > lumin_percent_threshold
                    reinitialize_mode_flag = 1;
                    num_reinitialize_frames = 1;
                    %reset bg and fg models
                    bg_model = cell(1, num_resolutions);
                    fg_model = cell(1, num_resolutions);
                    bg_indicator = cell(1, num_resolutions);
                    fg_indicator = cell(1, num_resolutions);
                    bg_mask = ones(num_rows, num_cols);
                    bg_masks_unclean(:,:,track_frame) = bg_mask;
                    bg_masks(:,:,track_frame) = bg_mask;
                    %update the bg model to include pixels from current frame
                    [bg_model bg_indicator] = update_kde_model( img_pixels, bg_model, bg_indicator, bg_mask, num_bg_frames, bg_update_version); 
                    prev_frame_pixels = img_pixels{1};
                    fprintf('********* bg model reintialization began at frame %d.. frame took %f seconds\n', seq_starts_from+track_frame, toc(frame_start_time));
                    continue; 
                end
            end 
        end

        %CVPR 2012 algorithm
        if algorithm_to_use == 2
            if use_bg_fg_new_classes ~=0
                fprintf('For CVPR 2012 algorithm, use_bg_fg_new_classes should be set to 1\n');
                fprintf('press any key to continue with this value reset to 0, Ctrl-C to quit\n');
                pause;
            end
            %The cache method - Uses the old sigma values as input to the 
            %algorithm. The resulting sigma values are saved for use in the next frame
            if use_efficient_sigma_cache
                %Using num_color_feature_vals = 100 for CVPR results. Comment out to use 255 instead
                num_color_feature_vals = 100;
                %Using num_siltp_feature_vals = 81
                num_siltp_feature_vals = 81;
                [bg_mask fg_mask dummy_variable bg_sigma_images fg_sigma_images] = classify_using_lab_siltp_kde_sharpening_sigma_with_cache( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, bg_sigma_images_old, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_prior, fg_sigma_images_old, fg_near_rows, fg_near_cols, fg_uniform_factor, num_color_feature_vals, num_siltp_feature_vals, 0 );
                bg_sigma_images_old = bg_sigma_images;
                fg_sigma_images_old = fg_sigma_images;
            %Else use the full method with adaptive variance selection
            else
                %Using num_color_feature_vals = 100 for CVPR results. Comment out to use 255 instead
                num_color_feature_vals = 100;
                %Using num_siltp_feature_vals = 81
                num_siltp_feature_vals = 81;
                [bg_mask fg_mask] = classify_lab_siltp_using_kde_sharpening_sigma_cvpr( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_prior, fg_near_rows, fg_near_cols, fg_uniform_factor, num_color_feature_vals, num_siltp_feature_vals, 0 );
            end
        end
         
        %BMVC 2012 algorithm - Joint domain-range with improvements, with adaptive kernel selection
        if algorithm_to_use == 1
            if use_bg_fg_new_classes == 1
                [bg_mask fg_mask] = classify_lab_siltp_using_kde_sharpening_sigma_3_classes( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_near_rows, fg_near_cols, epsilon_bayes_threshold, search_window, num_feature_vals, object_tracking_version, track_frame>= 108800 );
            else
                error('Error - No alternative procedure here');
            end
                    
        end
        %Sheikh-Shah method (our implementation) Joint domain-range modeling
        %For BMVC 2012 results
        if algorithm_to_use == 3
            if use_bg_fg_new_classes == 1
                [bg_mask fg_mask] = classify_lab_siltp_using_kde_sharpening_sigma_Sheikh_norm( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_near_rows, fg_near_cols, search_window, num_feature_vals, object_tracking_version, track_frame>= 108800 );

            else
                error('Error - Sheikh Shah normalization without 3 classes not implemented yet');
            end
        end

        if algorithm_to_use > 3 || algorithm_to_use <1
            error('Error - Undefined algorithm_to_use');
        end

        %If post-processing clean-up using MRF is required
        %TODO - Credit MRF code in documentation
        if use_MRF_clean_procedure
            bg_masks_unclean(:,:,track_frame) = bg_mask;
            bg_mask_clean = MRF_mincut_clean( bg_mask, fg_mask, MRF_lambda);
            fg_mask_clean = 1-bg_mask_clean;
            bg_mask = bg_mask.*bg_mask_clean;
            fg_mask = 1-bg_mask;
        end

        bg_masks(:,:,track_frame) = bg_mask;
        
        fg_mask = 1-bg_mask;
            
        if min(fg_mask(:))<0
            disp('fg_mask cannot be negative');
            keyboard;
        end
    
        %Add current frame pixel samples to bg and fg pixel_samples. 
        %Update the indicator matrices
        %Update the bg and fg models
        [bg_model bg_indicator] = update_kde_model( img_pixels, bg_model, bg_indicator, bg_mask, num_bg_frames, bg_update_version); 
        [fg_model fg_indicator] = update_kde_model( img_pixels, fg_model, fg_indicator, fg_mask, num_fg_frames, fg_update_version); 
        
        %If using BMVC algorithm, the posterior probability of bg/fg becomes the
        %prior for the current frame
        %For CVPR algorithm, the prior is kept unchanged (uniform equal priors for bg and fg)
        if use_bg_fg_new_classes
           bg_prior = bg_mask;
           fg_prior = 1-bg_mask;
        end
        
        
        %F-measure calculations
        %Calculate F_measure
        gt_evaluation_threshold = .5;
        %Calculate error (Fmeasure) for I2R data
        if (~isempty(find( ground_truth_frames == track_frame+seq_starts_from)))
           if bootstrap==1
                truth_image = imread(sprintf('%s/%s%05d.%s', input_groundtruth_video_folder, input_groundtruth_sequence_files_suffix, seq_starts_from+track_frame, file_ext));
            else
                truth_image = imread(sprintf('%s/%s%03d.%s', input_groundtruth_video_folder, input_groundtruth_sequence_files_suffix, seq_starts_from+track_frame, file_ext));
            end
            if size(truth_image, 3)==3
               truth_image = rgb2gray(truth_image);
            end
            threshed_fg_mask = double((bg_mask<gt_evaluation_threshold)*255); 
            track_frame+seq_starts_from
            f_measure = find_f_measure( threshed_fg_mask, truth_image) 
            gt_count = gt_count+1;
            f_measures(gt_count) = f_measure;
            f_measure_sum = f_measure_sum+f_measure;
            gt_frames_prediction(:,:,gt_count) = bg_mask;
            gt_frames_truth(:,:,gt_count) = truth_image;
            gt_frames_image(:,:,:,gt_count) = img_orig;
            gt_frame_prediction_threshed(:,:,gt_count) = threshed_fg_mask;
        end
        
        %Display results
        if(display_general==1)
            num_plots = 3;
            figure; subplot(1,num_plots,1); 
            if use_LAB == 1
                imagesc( uint8(img+128));
            else
                imagesc( img);
            end
            title('input frame');
            hold on; 
            subplot(1,num_plots,2); imagesc( bg_mask, [0 1] ); colormap(gray);
            title('bg probabilities');
            subplot(1,num_plots,3); imagesc( bg_mask>.5, [0 1]); colormap('default');
            title('bg segmented mask');
            impixelinfo
            %keyboard
            pause
        end 
        fprintf('frame %d took %f seconds\n', seq_starts_from+track_frame, toc(frame_start_time));

        prev_frame_pixels = img_pixels{1};
    end 

    %Generate a informative output filename
    if use_MRF_clean_procedure
        if MRF_lambda<1
            name_string = sprintf('MRF_lambda_0_%d', MRF_lambda*10);
        else
            name_string = sprintf('MRF_lambda_%d', MRF_lambda);
        end
    else
        name_string = sprintf('No_MRF');
    end 
    if use_LAB==1
        output_sequence_files_suffix = 'LAB_';
    else
        output_sequence_files_suffix = 'RGB_';
    end

    output_sequence_files_suffix = sprintf('%sKDE_bg_near_%d_fg_near_%d_bg_fg_updt_%d_%d', output_sequence_files_suffix, bg_near_rows, fg_near_rows, bg_update_version, fg_update_version);

    if use_multi_resolution == 1
        output_sequence_files_suffix = sprintf('mul_res_%s', output_sequence_files_suffix);
    end
    if use_intensity_change_reasoning == 1
        output_sequence_files_suffix = sprintf('int_reas_%s', output_sequence_files_suffix);
    end
    if use_efficient_sigma_cache == 1
        output_sequence_files_suffix = sprintf('eff_sig_cache_%s', output_sequence_files_suffix);
    end
   
    if algorithm_to_use == 1
        output_sequence_files_suffix = sprintf('bmvc_%s', output_sequence_files_suffix);
    elseif algorithm_to_use == 2
        output_sequence_files_suffix = sprintf('cvpr_%s', output_sequence_files_suffix);
    elseif algorithm_to_use == 3
        output_sequence_files_suffix = sprintf('sheikh_shah_%s', output_sequence_files_suffix);
    end

    output_sequence_files_suffix = sprintf('hybrid_SILTP_%s', output_sequence_files_suffix);
    
    f_measure_sum
    f_measure_sum/20
    f_measure_after_remove_small_regions
    consolidated_f_measure_on_video_results
    current_video_TP
    current_video_FP
    current_video_FN

    save_sequences_filename  = [ output_sequences_folder '/' output_sequence_files_suffix '_' name_string '_' input_sequence_files_suffix '_' num2str(skip_until_frame) '_' num2str(max_track_frame) '_fms_' 'bg_' num2str(num_bg_frames) '_fg_' num2str(num_fg_frames) '_updt_bg_fg_' num2str(bg_update_version) '_' num2str(fg_update_version) '_2729d.mat'];
    %save(save_sequences_filename, 'bg_masks', 'bg_masks_unclean','image_stack_work', 'image_stack', 'all_objects_masks', 'save_sequences_filename', 'f_measure_sum', 'f_measures', 'gt_frames_prediction', 'gt_frames_truth', 'gt_frames_image', 'videoname', 'input_sequence_files_suffix', 'seq_starts_from', 'skip_until_frame', 'total_num_frames', 'ground_truth_frames', 'bg_sigmas', 'fg_sigmas', 'object_sigmas', 'num_resolutions', 'fg_uniform_factor');
    
    %If video_numbers is an array, then delete all variables except video_numbers and then proceed
    if size(video_numbers, 2)>1
        save('del_temp.mat', 'video_numbers', 'run_test_subset');
        clear
        load('del_temp.mat');
    end
end
