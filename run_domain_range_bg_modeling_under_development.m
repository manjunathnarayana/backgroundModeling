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

%Usage - 
%(1) Change video number to select appropriate video. Note that video_numbers
%can be an array of video numbers which will be processed in sequence
%(2) Change path to input video, input groundtruth, output folder in loadVideo.m
%(3) Set algorithm_to_use variable in this script
%(4) Set any optional parameters in this script
%(5) Run video. Note that it may take several hours for complete video
%sequence to be processed. If you desire to run script for a smaller subset of
%video frames, change the total_num_frames variable in loadVideo.m

addpath( genpath ('utils'));
addpath( genpath ('compare'));
addpath( genpath ('max_flow_min_cut_wrapper'));

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

video_numbers

%Run for all videos listed in video_numbers
for video_number = video_numbers

    %Algorithm options
    %(1) BMVC 2012 - joint domain-range + adaptive kernel variance 
    %(2) CVPR 2012 - adaptive kernel variance
    %(3) Our implementation of Sheikh-Shah
    %(4) Ihler code for kernel variance calculation (AMISE criterion in CVPR 2012 paper)
    %(5) knn distance as variance - undocumented algorithm that uses the distance of the estimation point to the k-th nearest neighbor among the kde samples as variance
    %(6) use variance learned from first few frames (no adaptive kernel variance after that)
    algorithm_to_use = 3;

    %Call load video script
    load_video
    video_number

    disp( videoname);
    disp( input_sequence_files_suffix);

    %display options
    display_general = 0;

    %max frames to classify
    max_track_frame = total_num_frames;

    %num frames used for bg KDE
    num_bg_frames = 50;

    %num frames used for fg KDE
    num_fg_frames = 5;

    %TODO - explain this
    fg_uniform_factor = .5;

    %Algorithm options
    %use LAB or RGB color space
    use_LAB = 0;
    
    %Formulate as a three class problem - bg v/s seen foreground v/s new foreground
    %Must be set to 1 for BMVC 2012 results
    use_bg_fg_new_classes = 1;

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
   
    %Multiple resolutions (Pyramid)
    use_multi_resolution = 0;
    num_resolutions = 3;
    if use_multi_resolution==0
        num_resolutions = 1;
    end

    %Cache method from CVPR 2012 paper
    %this uses center pixel + neighborhood values to decide bg fg lik
    use_efficient_sigma_cache = 0;
    if use_efficient_sigma_cache==1
        bg_sigma_images_old = cell(1, num_resolutions);
        fg_sigma_images_old = cell(1, num_resolutions);
    end

    %Algorithm 5
    %k-th neighbor to use (for algorithm 5)
    k_th_neighbor = 2;
    
    %Flag that is set when a reinitialization is in progress
    reinitialize_mode_flag = 0;
    %Count of number of frames since reinitialization has begun
    num_reinitialize_frames = 0;

    %How bg and fg models are updated
    %TODO - Explain the options
    bg_update_version = 3;
    fg_update_version = 1;

    %TODO - Print information about method being used, etc.

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
        img = imread(sprintf('%s/%s%03d.%s', bg_input_video_folder, bg_input_sequence_files_suffix, seq_starts_from+ i, file_ext));
        if use_LAB==1
            img = rgb2LAB(double(img));
        end
    
        %Get image pixels as joint domain-range kernel points
        temp_pixels_samples = get_kernel_points_2d_from_image(img);
        
        %Update the bg model with pixel samples from above
        bg_model{1}(bg_frame_num+1,:,:,:,:,:,:,:) = temp_pixels_samples;
        bg_indicator{1}(bg_frame_num+1,:,:) = ones(num_rows, num_cols);
   
        %Populate models for multiple resolutions as desired
        for res = 2:num_resolutions
            temp_pixels_samples = [];
            sampled_img = subsample_image( img, 1.0/res );
            temp_pixels_samples = get_kernel_points_2d_from_image( sampled_img);
            bg_model{res}(bg_frame_num+1,:,:,:,:,:,:,:) = temp_pixels_samples;
            [num_sampled_rows num_sampled_cols num_sampled_colors] = size( sampled_img);
            bg_indicator{res}(bg_frame_num+1,:,:) = ones(num_sampled_rows, num_sampled_cols);
        end
    
        bg_frame_num = bg_frame_num+1;
    end
  
    %Prior values for various algorithms

    if algorithm_to_use == 1
        %When using bg_fg_new_classes, initially the center region has .98 confidence for bg. The image boundary region has 0 confidence for bg. This is used to arrive at bg, fg, and new priors in the confident and not-confident regions as described in BMVC 2012
        if use_bg_fg_new_classes == 1   
            bg_prior = ones( num_rows, num_cols)*.98;
            boundary_region_size = 10;
            bsr = boundary_region_size;
            bg_prior(1:bsr,:) = 0;
            bg_prior(end-bsr+1:end,:) = 0;
            bg_prior(:,1:bsr) = 0;
            bg_prior(:, end-bsr+1:end) = 0;
            fg_prior = 1-bg_prior;
        else
            bg_prior = ones( num_rows, num_cols)*.98;
            fg_prior = ones( num_rows, num_cols)*.02;
        end
    elseif algorithm_to_use == 2
        bg_prior = ones( num_rows, num_cols)*.5;
        fg_prior = ones( num_rows, num_cols)*.5;
    elseif algorithm_to_use == 3
        bg_prior = ones( num_rows, num_cols)*.5;
        fg_prior = ones( num_rows, num_cols)*.5;
    else
        bg_prior = ones( num_rows, num_cols)*.5;
        fg_prior = ones( num_rows, num_cols)*.5;
    end
   
    %Initialize bg and fg masks
    bg_mask = bg_prior;
    fg_mask = fg_prior;
    
    %number of values the features can take (R, G, B = 0 to 255, hence 256)
    num_feature_vals = 256;
    
    %Sigma values for bg and fg processes
    %For RGB color space sigma values
    if use_LAB == 0
        bg_sigma_XYs{1} = [1/4 3/4];
        bg_sigma_Ys{1}  = [5/4 15/4 45/4];
        bg_sigma_UVs{1} = [5/4 15/4 45/4];
        fg_sigma_XYs{1} = [12/4];
        fg_sigma_Ys{1}  = [15/4];
        fg_sigma_UVs{1} = [15/4];
    else
    %LAB sigma values
        bg_sigma_XYs{1} = [1/4 3/4];
        bg_sigma_Ys{1}  = [5/4 10/4 15/4];
        bg_sigma_UVs{1} = [4/4 6/4];
        fg_sigma_XYs{1} = [12/4];
        fg_sigma_Ys{1}  = [15/4];
        fg_sigma_UVs{1} = [6/4];
    end
    
    fprintf('bg_sigma_XYs{1}\n')
    bg_sigma_XYs{1}(:)'
    fprintf('bg_sigma_Ys{1}\n')
    bg_sigma_Ys{1}(:)'
    fprintf('bg_sigma_UVs{1}\n')
    bg_sigma_UVs{1}(:)'
    fprintf('fg_sigma_XYs{1}\n')
    fg_sigma_XYs{1}(:)'
    fprintf('fg_sigma_Ys{1}\n')
    fg_sigma_Ys{1}(:)'
    fprintf('fg_sigma_UVs{1}\n')
    fg_sigma_UVs{1}(:)'

    %If using multi-resolution, set up sigma values for each resolution
    if use_multi_resolution
        % Resolution 2
        bg_sigma_XYs{2} = [3/4];
        bg_sigma_Ys{2} = [10/4];
        bg_sigma_UVs{2} = [2/4];
        fg_sigma_XYs{2} = [3/4];
        fg_sigma_Ys{2} = [10/4];
        fg_sigma_UVs{2} = [2/4];
        % Resolution 3
        bg_sigma_XYs{3} = [3/4];
        bg_sigma_Ys{3} = [10/4];
        bg_sigma_UVs{3} = [2/4];
        fg_sigma_XYs{3} = [3/4];
        fg_sigma_Ys{3} = [10/4];
        fg_sigma_UVs{3} = [2/4];
        fprintf('bg_sigma_XYs{2}\n')
        bg_sigma_XYs{2}(:)'
        fprintf('bg_sigma_Ys{2}\n')
        bg_sigma_Ys{2}(:)'
        fprintf('bg_sigma_UVs{2}\n')
        bg_sigma_UVs{2}(:)'
        fprintf('fg_sigma_XYs{2}\n')
        fg_sigma_XYs{2}(:)'
        fprintf('fg_sigma_Ys{2}\n')
        fg_sigma_Ys{2}(:)'
        fprintf('fg_sigma_UVs{2}\n')
        fg_sigma_UVs{2}(:)'
        fprintf('bg_sigma_XYs{3}\n')
        bg_sigma_XYs{3}(:)'
        fprintf('bg_sigma_Ys{3}\n')
        bg_sigma_Ys{3}(:)'
        fprintf('bg_sigma_UVs{3}\n')
        bg_sigma_UVs{3}(:)'
        fprintf('fg_sigma_XYs{3}\n')
        fg_sigma_XYs{3}(:)'
        fprintf('fg_sigma_Ys{3}\n')
        fg_sigma_Ys{3}(:)'
        fprintf('fg_sigma_UVs{3}\n')
        fg_sigma_UVs{3}(:)'
    end
    
    %Consolidate the sigma values into a cell array for bg and fg
    bg_sigmas{1} = bg_sigma_XYs;
    bg_sigmas{2} = bg_sigma_Ys;
    bg_sigmas{3} = bg_sigma_UVs;
    fg_sigmas{1} = fg_sigma_XYs;
    fg_sigmas{2} = fg_sigma_Ys;
    fg_sigmas{3} = fg_sigma_UVs;
    
    if (size(bg_sigma_XYs, 2)~=num_resolutions) || (size(fg_sigma_XYs, 2)~=num_resolutions)
        error(' number of candidate sigmas should match the number or resolutions');
    end
    
    %definition of neighborhood for kde calculation
    %pixels in a range + and - this value are used as samples in computing KDE likelihoods
    bg_near_rows = ceil( max( bg_sigma_XYs{1})*4/2);
    bg_near_cols = bg_near_rows;
    fg_near_rows = ceil( max( fg_sigma_XYs{1})*4/2);
    fg_near_cols = fg_near_rows;
            
    %if need to learn a covariance from the first few frames
    optimal_bg_sigma_index = [];
    optimal_fg_sigma_index = [];
    bg_sigma_values = [];
    fg_sigma_values = [];
   
    %If variances are learned from first few frames and held fixed after that (no adaptive variance selection)
    if algorithm_to_use == 6 
        if num_resolutions == 1
            [optimal_bg_sigma_indexes bg_sigma_values] = get_optimal_sigmas_from_model( bg_model{1}, bg_indicator{1}, bg_sigma_XYs{1}, bg_sigma_Ys{1}, bg_sigma_UVs{1}, bg_near_cols, bg_near_rows, 0);
        else
            error('Error - Learning sigmas from model does not work for higher than 1 resolution');
        end
    end
   
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
        
        %Get kde pixel samples from image
        img_pixels{1} = get_kernel_points_2d_from_image( img );
        %For all resolutions
        for res = 2:num_resolutions
            temp_pixels_samples = [];
            sampled_img = subsample_image( img, 1.0/res );
            temp_pixels_samples = get_kernel_points_2d_from_image( sampled_img);
            img_pixels{res} = temp_pixels_samples;
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

        %BMVC 2012 algorithm - Joint domain-range with improvements, with adaptive kernel selection
        if algorithm_to_use == 1
            if use_bg_fg_new_classes == 1
                [bg_mask fg_mask] = classify_using_kde_sharpening_sigma_3_classes( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_prior, fg_near_rows, fg_near_cols, num_feature_vals, 0 );
            else
                if use_efficient_sigma_cache == 1
                    [bg_mask fg_mask bg_sigma_images fg_sigma_images] = classify_using_kde_sharpening_sigma_with_cache( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, bg_sigma_images_old, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_prior, fg_sigma_images_old, fg_near_rows, fg_near_cols, fg_uniform_factor, num_feature_vals, 0 );
                    bg_sigma_images_old = bg_sigma_images;
                    fg_sigma_images_old = fg_sigma_images;
                else
                    [bg_mask fg_mask] = classify_using_kde_sharpening_sigma( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_prior, fg_near_rows, fg_near_cols, fg_uniform_factor, num_feature_vals, 0 );
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
                %Using num_feature_vals = 100 for CVPR results. Comment out to use 255 instead
                num_feature_vals = 100;
                [bg_mask fg_mask bg_sigma_images fg_sigma_images] = classify_using_kde_sharpening_sigma_with_cache_cvpr( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, bg_sigma_images_old, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_prior, fg_sigma_images_old, fg_near_rows, fg_near_cols, fg_uniform_factor, num_feature_vals, 0 );
                bg_sigma_images_old = bg_sigma_images;
                fg_sigma_images_old = fg_sigma_images;

            %Else use the full method with adaptive variance selection
            else
                %Using num_feature_vals = 100 for CVPR results. Comment out to use 255 instead
                num_feature_vals = 100;
                [bg_mask fg_mask] = classify_using_kde_sharpening_sigma_cvpr( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_prior, fg_near_rows, fg_near_cols, fg_uniform_factor, num_feature_vals, 0 );
            end
        end

       %Sheikh-Shah method (our implementation) Joint domain-range modeling
        %For BMVC 2012 results
        if algorithm_to_use == 3
            if use_bg_fg_new_classes == 1
                [bg_mask fg_mask] = classify_using_kde_sharpening_sigma_Sheikh_normalization( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_prior, fg_near_rows, fg_near_cols, num_feature_vals, 0 );
            else
                error('Error - Sheikh Shah normalization without 3 classes not implemented yet');
            end
        end

        %Classic variance selection algorithms (AMISE)
        if algorithm_to_use == 4
            if use_bg_fg_new_classes %if use_ihler_code TODO which of these two are to be used? Also remove reference to ihler in function names. Credit Ihler in your documentation
                [bg_mask fg_mask] = classify_using_ihler_sharpening_sigma_3_classes( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_prior, fg_near_rows, fg_near_cols, fg_uniform_factor, objects, object_near_rows, object_near_cols, search_window, num_feature_vals, object_tracking_version, track_frame>= 12000000 );

            else %if use_ihler_code_3_classes
                [bg_mask fg_mask] = classify_using_ihler_sharpening_sigma( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_prior, fg_near_rows, fg_near_cols, fg_uniform_factor, objects, object_near_rows, object_near_cols, search_window, num_feature_vals, object_tracking_version, track_frame>= 12000000 );
            end
        end

        %Computing variance using distance to k-th nearest neighbor
        if algorithm_to_use == 5
            if use_bg_fg_new_classes
                [bg_mask fg_mask] = classify_using_kde_knn_distance_as_sigma_3_classes( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_prior, fg_near_rows, fg_near_cols, fg_uniform_factor, objects, object_sigmas, object_near_rows, object_near_cols, k_th_neighbor, search_window, num_feature_vals, object_tracking_version, track_frame>= 12000000 );
            else
                [bg_mask fg_mask] = classify_using_kde_knn_distance_as_sigma( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_prior, fg_near_rows, fg_near_cols, fg_uniform_factor, objects, object_sigmas, object_near_rows, object_near_cols, k_th_neighbor, search_window, num_feature_vals, object_tracking_version, track_frame>= 12000000 );
            end
        end

          
        %Compute the best variance only from the first few frames. No adaptive variance selection after that
        if algorithm_to_use == 6
            [bg_mask fg_mask] = classify_using_kde_learned_bg_sigmas_3_classes( img_pixels, bg_model, bg_indicator, bg_sigma_values, optimal_bg_sigma_indexes, bg_prior, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_prior, fg_near_rows, fg_near_cols, search_window, num_feature_vals, object_tracking_version, track_frame>= 12000000 );
        end

        if algorithm_to_use > 6 || algorithm_to_use <1
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
        
        %Use the mask values as new priors if flag is set
        if use_bg_fg_new_classes == 1
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
            gt_frames_image(:,:,:,gt_count) = img;
            gt_frame_prediction_threshed(:,:,gt_count) = threshed_fg_mask;
        end

        %Display results
        if(display_general==1)
            num_plots = 3;
            figure; subplot(1,num_plots,1); 
            if use_LAB
                imagesc( uint8(img+128));
            else
                imagesc(img);
            end
            title('input frame');
            hold on; 
            subplot(1,num_plots,2); imagesc( bg_mask, [0 1] ); colormap(gray);
            title('bg probabilities');
            subplot(1,num_plots,3); imagesc( bg_mask>.5); colormap('default');
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
    elseif algorithm_to_use == 4
        if use_bg_fg_new_classes
            output_sequence_files_suffix = sprintf('knn_%d_3_classes_%s', k_th_neighbor, output_sequence_files_suffix);
        else
            output_sequence_files_suffix = sprintf('knn_%d_%s', k_th_neighbor, output_sequence_files_suffix);
        end
    elseif algorithm_to_use == 5
        if use_bg_fg_new_classes
            output_sequence_files_suffix = sprintf('standard_kde_variance_3_classes_%s', output_sequence_files_suffix);
        else
            output_sequence_files_suffix = sprintf('standard_kde_variance_%s', output_sequence_files_suffix);
        end
     elseif algorithm_to_use == 6
        output_sequence_files_suffix = sprintf('initial_frames_variance_%s', output_sequence_files_suffix);
    end
    
    f_measure_sum
    f_measure_sum/size(f_measures,2);

    %script that f_measure calculation after removing small regions 
    f_measure_after_remove_small_regions
    %TODO - What does this do?
    consolidated_f_measure_on_video_results
    current_video_TP
    current_video_FP
    current_video_FN

    save_sequences_filename  = [ output_sequences_folder '/' output_sequence_files_suffix '_' name_string '_' input_sequence_files_suffix '_' num2str(skip_until_frame) '_' num2str(max_track_frame) '_fms_' 'bg_' num2str(num_bg_frames) '_fg_' num2str(num_fg_frames) '_updt_bg_fg_' num2str(bg_update_version) '_' num2str(fg_update_version) '_1803d.mat'];
%    save(save_sequences_filename, 'bg_masks', 'bg_masks_unclean', 'image_stack_work', 'image_stack', 'save_sequences_filename', 'f_measure_sum', 'f_measures', 'gt_frames_prediction', 'gt_frames_truth', 'gt_frames_image', 'videoname', 'input_sequence_files_suffix', 'seq_starts_from', 'skip_until_frame', 'total_num_frames', 'ground_truth_frames', 'bg_sigmas', 'fg_sigmas', 'object_sigmas', 'num_resolutions', 'fg_uniform_factor', 'optimal_bg_sigma_index', 'optimal_fg_sigma_index', 'bg_sigma_values', 'fg_sigma_values');

    %If video_numbers is an array, then delete all variables except video_numbers and then proceed
    if size(video_numbers, 2)>1
        save('del_temp.mat', 'video_numbers');
        clear
        load('del_temp.mat');
        delete('del_temp.mat');
    end
end
