function [bg_mask fg_mask bg_sigmas_image fg_sigmas_image] = classify_using_kde_sharpening_sigma_with_cache( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, prev_bg_sigma_images, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_prior, prev_fg_sigma_images, fg_near_rows, fg_near_cols, fg_uniform_factor, num_feature_vals, debug_flag)
%function [bg_mask fg_mask bg_sigmas_image fg_sigmas_image] = classify_using_kde_sharpening_sigma_with_cache( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, prev_bg_sigma_images, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_prior, prev_fg_sigma_images, fg_near_rows, fg_near_cols, fg_uniform_factor, num_feature_vals, debug_flag)
%Function that classifies frames as bg/fg with kde likelihoods, but uses cached covariance values from previous frame where possible. Adaptive kernel variance is performed only in required pixels
%This is BMVC 2012 (Narayana et. al) code -- normalizing by number of frames, using uniform fg factor
%Function that classifies pixels as bg/fg based on the kde point samples in bg_model and fg_model
%img_pixels is the r x c x d representation of pixels in the image
%bg_model and fg_model are k x r x c x d in size where k = number of frames for which the model is being kept, r is number of rows in the image, c is number of columns in the image, and d is the number of dimensions of the kernel
%the indicator matrices identify which pixels in each frame are valid bg/fg/object pixels
%the covariance matrix (sigma) and priors for each class are also input to the function
%the prev_X_sigma_image is the sigma from the previous frame which is used to decide which pixels to process with sharpening sigma in the current frame and which to simply use the old sigma sigma values from the previous frame. It is saved as a cell array of images
%the near_rows and near_cols denote how much on each side of the pixel must be considered as neighborhood for kde samples
%fg_uniform_factor is the factor which includes a uniform distribution in the fg model. ( There is no other way to include a uniform distribution using kde samples). p(x|fg) = fg_uniform_factor*uniform_pdf + (1-fg_uniform_factor)*fg_kde_estimate
%num_feature_vals = number of values the features (like color or intensity) can take -- this is used in generating the uniform distributions for likelihood. Techinically should be 255, but this value results in very small uniform likelihoods (many foreground objects go undetected). For CVPR 2012 results, 100 was used instead.
%bg_sigmas and fg_sigmas are cells that have candidate sigma values for XY, Y (or R), and UV (or GB) dimensions

%debug_flag = 1;
temp_bg_model = bg_model{1};
temp_fg_model = fg_model{1};
num_bg_frames = size( temp_bg_model, 1);
num_fg_frames = size( temp_fg_model, 1);
num_resolutions = size( bg_model, 2);

bg_mask = [];
fg_mask = [];

%bg frames to use for likelihood calculation
%One in 10 frames used for bg samples
bg_sample_frames = 1:10:num_bg_frames;
%uncomment below to use all frames in the model
%bg_sample_frames = 1:num_bg_frames;

%Run for all resolutions
for res=1:num_resolutions
    bg_sigma_XYs{res} = bg_sigmas{1}{res};
    bg_sigma_Ys{res} = bg_sigmas{2}{res};
    bg_sigma_UVs{res} = bg_sigmas{3}{res};

    fg_sigma_XYs{res} = fg_sigmas{1}{res};
    fg_sigma_Ys{res} = fg_sigmas{2}{res};
    fg_sigma_UVs{res} = fg_sigmas{3}{res};
    
    bg_model_sampled = bg_model{res}( bg_sample_frames, :,:,:);  
    bg_indicator_sampled = bg_indicator{res}( bg_sample_frames, :,:);

    %First classify any pixels that are obviously bg or fg based on old sigma values
    
    %Calculate KDE likelihoods for bg and fg using old sigma values
    [selective_maps{res} bg_liks_selective{res} selective_bg_sigmas_image{res} fg_liks_selective{res} selective_fg_sigmas_image{res}] = selective_calculate_kde_likelihood_bg_fg_with_cache( img_pixels{res}, bg_model_sampled, bg_indicator_sampled, prev_bg_sigma_images{res}, bg_sigma_XYs{res}, bg_sigma_Ys{res}, bg_sigma_UVs{res}, bg_near_rows, bg_near_cols, 0, fg_model{res}, fg_indicator{res}, prev_fg_sigma_images{res}, fg_sigma_XYs{res}, fg_sigma_Ys{res}, fg_sigma_UVs{res}, fg_near_rows, fg_near_cols, fg_uniform_factor, num_feature_vals, debug_flag);
    [num_res_rows num_res_cols num_res_dims] = size( img_pixels{res});
    num_selected_pixels = sum( selective_maps{res}(:));
    %printing statistics about number of pixels for which the efficient processing was used
    fprintf('Computing sharpening match on %.0f percent of pixels. Others set to cached values from prev frame\n', 100-(num_selected_pixels*100/num_res_rows/num_res_cols));
    
    %The function below reads the selective_map. In places where selective_map is zero, do the full processing as before. In places where selective_map is one, simply copy the value of the likelihoods from the X_liks_selective image. Set the X_sigmas_image to be the same as the prev_X_sigma_image at these pixels
    %Calculate KDE likelihoods for bg in the selected pixels where full computation is desired
    [bg_liks{res} bg_sigmas_image{res} bg_model_sigmas{res}] = selective_calculate_kde_likelihood_sharpening( img_pixels{res}, bg_model_sampled, bg_indicator_sampled, selective_maps{res}, bg_liks_selective{res}, selective_bg_sigmas_image{res}, bg_sigma_XYs{res}, bg_sigma_Ys{res}, bg_sigma_UVs{res}, bg_near_rows, bg_near_cols, 0, num_feature_vals, debug_flag);
    %Calculate KDE likelihoods for fg in the selected pixels where full computation is desired
    [fg_liks{res} fg_sigmas_image{res} fg_model_sigmas{res}] = selective_calculate_kde_likelihood_sharpening( img_pixels{res}, fg_model{res}, fg_indicator{res}, selective_maps{res}, fg_liks_selective{res}, selective_fg_sigmas_image{res}, fg_sigma_XYs{res}, fg_sigma_Ys{res}, fg_sigma_UVs{res}, fg_near_rows, fg_near_cols, fg_uniform_factor, num_feature_vals, debug_flag);
end

%Combine likelihoods from different resolutions
bg_liks_combined = combine_multires_masks( bg_liks);
fg_liks_combined = combine_multires_masks( fg_liks);

%Use prior values
bg_liks_prior = bg_liks_combined.*bg_prior;
fg_liks_prior = fg_liks_combined.*fg_prior;
bg_mask = bg_liks_prior./(bg_liks_prior+fg_liks_prior+eps(0));
fg_mask = (1-bg_mask);
if debug_flag
    keyboard;
end
if sum(isnan(bg_mask(:))~=0)
    disp('nan in bg_mask');
    keyboard;
end
if sum(isnan(fg_mask(:))~=0)
    disp('nan in fg_mask');
    keyboard;
end
