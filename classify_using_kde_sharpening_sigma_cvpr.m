function [bg_mask fg_mask] = classify_using_kde_sharpening_sigma_cvpr( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_prior, fg_near_rows, fg_near_cols, fg_uniform_factor, num_feature_vals, debug_flag)
%function [bg_mask fg_mask] = classify_using_kde_sharpening_sigma_cvpr( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_prior, fg_near_rows, fg_near_cols, fg_uniform_factor, num_feature_vals, debug_flag)
%This is CVPR 2012 (Narayana et. al) code -- normalizing by number of frames, using uniform fg factor
%Function that classifies pixels as bg/fg based on the kde point samples in bg_model, fg_model and obj_model
%img_pixels is the r x c x d representation of pixels in the image
%bg_model and fg_model are k x r x c x d in size where k = number of frames for which the model is being kept, r is number of rows in the image, c is number of columns in the image, and d is the number of dimensions of the kernel
%the indicator matrices identify which pixels in each frame are valid bg/fg/object pixels
%the covariance matrix (sigma) and priors for each class are also input to the function
%the near_rows and near_cols denote how much on each side of the pixel must be considered as neighborhood for kde samples
%fg_uniform_factor is the factor which includes a uniform distribution in the fg model. ( There is no other way to include a uniform distribution using kde samples). p(x|fg) = fg_uniform_factor*uniform_pdf + (1-fg_uniform_factor)*fg_kde_estimate
%num_feature_vals = number of values the features (like color or intensity) can take -- this is used in generating the uniform distributions for likelihood. Techinically should be 255, but this value results in very small uniform likelihoods (many foreground objects go undetected). For CVPR 2012 results, 100 was used instead.
%bg_sigmas and fg_sigmas are cells that have candidate sigma values for XY, Y (or R), and UV (or GB) dimensions

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
    %Calculate KDE likelihoods for bg
    [bg_liks{res} bg_sigmas_image{res} bg_model_sigmas{res}] = calculate_kde_likelihood_sharpening_cvpr( img_pixels{res}, bg_model_sampled, bg_indicator_sampled, bg_sigma_XYs{res}, bg_sigma_Ys{res}, bg_sigma_UVs{res}, bg_near_rows, bg_near_cols, 0, num_feature_vals, debug_flag);
    %Calculate KDE likelihoods for fg
    [fg_liks{res} fg_sigmas_image{res} fg_model_sigmas{res}] = calculate_kde_likelihood_sharpening_cvpr( img_pixels{res}, fg_model{res}, fg_indicator{res}, fg_sigma_XYs{res}, fg_sigma_Ys{res}, fg_sigma_UVs{res}, fg_near_rows, fg_near_cols, fg_uniform_factor, num_feature_vals, debug_flag);
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
