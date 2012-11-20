function [bg_mask fg_mask bg_sigmas_image fg_sigmas_image] = classify_using_kde_sharpening_sigma_with_cache_cvpr( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, prev_bg_sigma_images, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_prior, prev_fg_sigma_images, fg_near_rows, fg_near_cols, fg_uniform_factor, num_color_feature_vals, num_siltp_feature_vals, debug_flag)
%function [bg_mask fg_mask bg_sigmas_image fg_sigmas_image] = classify_using_kde_sharpening_sigma_with_cache_cvpr( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, prev_bg_sigma_images, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_prior, prev_fg_sigma_images, fg_near_rows, fg_near_cols, fg_uniform_factor, num_color_feature_vals, num_siltp_feature_vals, debug_flag)
%Function that classifies pixels as bg/fg/object_i based on the kde point samples in bg_model, fg_model and obj_model
%img_pixels is the r x c x d representation of pixels in the image
%bg_model and fg_model are k x r x c x d in size where k = number of frames for which the model is being kept, r is number of rows in the image, c is number of columns in the image, and d is the number of dimensions of the kernel
%objects_model is a cell of num_objects size with each element being k_i x r x c x d size. k_i is number of frames for which current object has samples
%the indicator matrices identify which pixels in each frame are valid bg/fg/object pixels
%the covariance matrix (sigma) and priors for each class are also input to the function
%the prec_X_sigma_image is the sigma from the previous frame which is used to decide which pixels to process with sharpening sigma in the current frame and which to simply use the old sigma sigma values from the previous frame
%the near_rows and near_cols denote how much on each side of the pixel must be considered as neighborhood for kde samples
%fg_uniform_factor is the factor which includes a uniform distribution in the fg model. (No other way to include a uniform distribution using kde samples). p(x|fg) = fg_uniform_factor*uniform_pdf + (1-fg_uniform_factor)*fg_kde_estimate
%num_feature_vals = number of values the features (like color or intensity) can take -- this is used in generating the uniform distributions for likelihood
%bg_sigmas, fg_sigmas, object_sigmas are cells that have candidate sigma values for XY, Y, and UV dimensions

%debug_flag = 1;
temp_bg_model = bg_model{1};
temp_fg_model = fg_model{1};
num_bg_frames = size( temp_bg_model, 1);
num_fg_frames = size( temp_fg_model, 1);
num_resolutions = size( bg_model, 2);

num_objects = size( objects, 2);

bg_mask = [];
fg_mask = [];
obj_mask = [];

%If there are no objects, then simply classify bg/fg
if num_objects==0
    
    %bg frames to use for likelihood calculation
    bg_sample_frames = 1:10:num_bg_frames;
    %bg_sample_frames = 1:num_bg_frames;
    

    %Run for all resolutions
    for res=1:num_resolutions
        bg_sigma_XYs{res} = bg_sigmas{1}{res};
        bg_sigma_Ls{res} = bg_sigmas{2}{res};
        bg_sigma_ABs{res} = bg_sigmas{3}{res};
        bg_sigma_LTPs{res} = bg_sigmas{3}{res};
    
        fg_sigma_XYs{res} = fg_sigmas{1}{res};
        fg_sigma_Ls{res} = fg_sigmas{2}{res};
        fg_sigma_ABs{res} = fg_sigmas{3}{res};
        fg_sigma_LTPs{res} = fg_sigmas{3}{res};
        
        bg_model_sampled = bg_model{res}( bg_sample_frames, :,:,:);  
        bg_indicator_sampled = bg_indicator{res}( bg_sample_frames, :,:);
    
        %First classify any pixels that are obviously bg or fg based on old sigma values
        
        %debug_flag = 1;
        [selective_maps{res} bg_liks_selective{res} selective_bg_sigmas_image{res} fg_liks_selective{res} selective_fg_sigmas_image{res}] = selective_calculate_lab_siltp_kde_likelihood_bg_fg_with_cache( img_pixels{res}, bg_model_sampled, bg_indicator_sampled, prev_bg_sigma_images{res}, bg_sigma_XYs{res}, bg_sigma_Ls{res}, bg_sigma_ABs{res}, bg_sigma_LTPs{res}, bg_near_rows, bg_near_cols, 0, fg_model{res}, fg_indicator{res}, prev_fg_sigma_images{res}, fg_sigma_XYs{res}, fg_sigma_Ls{res}, fg_sigma_ABs{res}, fg_sigma_LTPs{res}, fg_near_rows, fg_near_cols, fg_uniform_factor, num_feature_vals, debug_flag);
        [num_res_rows num_res_cols num_res_dims] = size( img_pixels{res});
        num_selected_pixels = sum( selective_maps{res}(:));
        fprintf('Computing sharpening match on %.0f percent of pixels. Others set to cached values from prev frame\n', 100-(num_selected_pixels*100/num_res_rows/num_res_cols));
        %figure; imagesc( selective_maps{res}); impixelinfo
        %keyboard
        %pause
        %The function below should read the selective_map. In places where selective_map is zero, do the full processing as before. In places where selective_map is one, simply copy the value of the likelihoods from the X_liks_selective image. Set the X_sigmas_image to be the same as the prev_X_sigma_image at these pixels
        [bg_liks{res} bg_sigmas_image{res} bg_model_sigmas{res}] = selective_calculate_lab_siltp_kde_likelihood_sharpening( img_pixels{res}, bg_model_sampled, bg_indicator_sampled, selective_maps{res}, bg_liks_selective{res}, selective_bg_sigmas_image{res}, bg_sigma_XYs{res}, bg_sigma_Ls{res}, bg_sigma_ABs{res}, bg_sigma_LTPs{res}, bg_near_rows, bg_near_cols, 0, num_feature_vals, debug_flag);
        [fg_liks{res} fg_sigmas_image{res} fg_model_sigmas{res}] = selective_calculate_lab_siltp_kde_likelihood_sharpening( img_pixels{res}, fg_model{res}, fg_indicator{res}, selective_maps{res}, fg_liks_selective{res}, selective_fg_sigmas_image{res}, fg_sigma_XYs{res}, fg_sigma_Ls{res}, fg_sigma_ABs{res}, fg_sigma_LTPs{res}, fg_near_rows, fg_near_cols, fg_uniform_factor, num_feature_vals, debug_flag);

    end
%    bg_liks{1}(:) = 1;
%    bg_liks{2}(:) = 1;
%    fg_liks{1}(:) = 1;
%    fg_liks{2}(:) = 1;
    bg_liks_combined = combine_multires_masks( bg_liks);
    fg_liks_combined = combine_multires_masks( fg_liks);
%    keyboard
    %Use prior values
    bg_liks_prior = bg_liks_combined.*bg_prior;
    fg_liks_prior = fg_liks_combined.*fg_prior;
    bg_mask = bg_liks_prior./(bg_liks_prior+fg_liks_prior+eps(0));
    fg_mask = (1-bg_mask);
    if debug_flag
        keyboard;
    end
%    disp('In classify_using_kde');
%    keyboard
    if sum(isnan(bg_mask(:))~=0)
        disp('nan in bg_mask');
        keyboard;
    end
    if sum(isnan(fg_mask(:))~=0)
        disp('nan in fg_mask');
        keyboard;
    end

%If there are objects, then classify as bg/fg/obj_i
%This part is out of date. Need to reflect multiresolution bg and fg models here
else
    disp('This function classify_using_kde_sharpening_sigma has not been tested since making models multiresolution, test before proceeding');
    keyboard
    if tracking_version == 1
        disp('tracking');
        [bg_mask fg_mask obj_mask] = classify_by_independent_max_obj_likelihood( img_pixels, bg_model, bg_indicator, bg_sigma, bg_prior, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigma, fg_prior, fg_near_rows, fg_near_cols, fg_uniform_factor, objects, object_sigma, object_near_rows, object_near_cols, search_window, num_feature_vals, debug_flag);
    end
end

