function [bg_mask fg_mask obj_mask] = classify_lab_siltp_using_kde_sharpening_sigma_3_classes( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_near_rows, fg_near_cols, num_color_feature_vals, num_siltp_feature_vals, debug_flag)
%function [bg_mask fg_mask obj_mask] = classify_lab_siltp_using_kde_sharpening_sigma_3_classes( img_pixels, bg_model, bg_indicator, bg_sigmas, bg_prior, bg_near_rows, bg_near_cols, fg_model, fg_indicator, fg_sigmas, fg_near_rows, fg_near_cols, num_color_feature_vals, num_siltp_feature_vals, debug_flag)
%Function that classifies pixels as bg/fg/new based on the kde point samples in bg_model, fg_model and bg prior
%This is BMVC 2012 (Narayana et. al) code for hybrid feature space (color+siltp) -- joint domain-range KDE
%Function that classifies pixels as bg/fg based on the kde point samples in bg_model and  fg_model, by using a joint domain-range model as described in BMVC 2012 paper
%img_pixels is the r x c x d representation of pixels in the image
%bg_model and fg_model are k x r x c x d in size where k = number of frames for which the model is being kept, r is number of rows in the image, c is number of columns in the image, and d is the number of dimensions of the kernel
%the indicator matrices identify which pixels in each frame are valid bg/fg pixels in a soft manner
%the covariance values (sigma) and priors for each class are also input to the function. Note that fg_prior is not used. bg_prior is used to decide priors for the three processes.
%the near_rows and near_cols denote how much on each side of the pixel must be considered as neighborhood for kde samples
%num_color_feature_vals = number of values each color channel feature can take -- this is used in generating the uniform distributions for likelihood. Usually, it should be 256.
%num_siltp_feature_vals = number of values siltp features can take (Typically 81)
%bg_sigmas and fg_sigmas are cells that have candidate sigma values for XY, Y (or R), UV (or GB), and SILTP dimensions
%Prior for bg fg newfg set to [.98 .01 .01]p(bg) + [.50 .25 .25](1-p(bg)) or a threshold based choice of one of the two

%debug_flag = 1;
temp_bg_model = bg_model{1};
temp_fg_model = fg_model{1};
num_bg_frames = size( temp_bg_model, 1);
num_fg_frames = size( temp_fg_model, 1);
num_resolutions = size( bg_model, 2);

num_rows = size( img_pixels{1},1);
num_cols = size( img_pixels{1},2);

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
    bg_sigma_Ls{res} = bg_sigmas{2}{res};
    bg_sigma_ABs{res} = bg_sigmas{3}{res};
    bg_sigma_LTPs{res} = bg_sigmas{4}{res};

    fg_sigma_XYs{res} = fg_sigmas{1}{res};
    fg_sigma_Ls{res} = fg_sigmas{2}{res};
    fg_sigma_ABs{res} = fg_sigmas{3}{res};
    fg_sigma_LTPs{res} = fg_sigmas{4}{res};

    bg_model_sampled = bg_model{res}( bg_sample_frames, :,:,:);  
    bg_indicator_sampled = bg_indicator{res}( bg_sample_frames, :,:);
    %Calculate KDE likelihoods for bg
    [bg_liks{res} bg_sigmas_image{res} bg_model_sigmas{res}] = calculate_lab_siltp_kde_likelihood_sharpening( img_pixels{res}, bg_model_sampled, bg_indicator_sampled, bg_sigma_XYs{res}, bg_sigma_Ls{res}, bg_sigma_ABs{res}, bg_sigma_LTPs{res},bg_near_rows, bg_near_cols, 0, num_color_feature_vals, num_siltp_feature_vals, debug_flag);
    %Calculate KDE likelihoods for fg
    [fg_liks{res} fg_sigmas_image{res} fg_model_sigmas{res}] = calculate_lab_siltp_kde_likelihood_sharpening( img_pixels{res}, fg_model{res}, fg_indicator{res}, fg_sigma_XYs{res}, fg_sigma_Ls{res}, fg_sigma_ABs{res}, fg_sigma_LTPs{res}, fg_near_rows, fg_near_cols, 0, num_color_feature_vals, num_siltp_feature_vals, debug_flag);
end

%Combine likelihoods from different resolutions
bg_liks_combined = combine_multires_masks( bg_liks);
fg_liks_combined = combine_multires_masks( fg_liks);

%Figure out the prior values

%First smooth the bg_prior

%First smooth the bg_prior
prior_smooth_length = 7;
bg_prior_filter = fspecial('gaussian', [prior_smooth_length prior_smooth_length], prior_smooth_length/4); 
bg_prior_smoothed =  imfilter( bg_prior, bg_prior_filter, 'replicate');

%Use a "soft" interpretation of a which regions are confident bg and which are not-confident bg regions. Alternative is to use a hard threshold, but this works better
%To use soft threshold, use these values for confident bg pixels and not-confident bg pixels 
confident_bg_priors = [.95 .025 .025];
not_confident_bg_priors = [.5 .25 .25];
bg_prior_3 = (bg_prior_smoothed*confident_bg_priors(1)) + ((1-bg_prior_smoothed).*not_confident_bg_priors(1));
fg_prior_3 = (bg_prior_smoothed*confident_bg_priors(2)) + ((1-bg_prior_smoothed).*not_confident_bg_priors(2));
new_prior_3 = (bg_prior_smoothed*confident_bg_priors(3)) + ((1-bg_prior_smoothed).*not_confident_bg_priors(3));
        
%Use prior values
bg_liks_prior = bg_liks_combined.*bg_prior_3;
fg_liks_prior = fg_liks_combined.*fg_prior_3;

%NOTE - BMVC results had bug - The following line was used
new_liks_prior = (1/num_color_feature_vals/num_color_feature_vals/num_color_feature_vals/num_siltp_feature_vals/num_siltp_feature_vals).*new_prior_3;
%The following is the correct line 
%new_liks_prior = (1/num_color_feature_vals/num_color_feature_vals/num_color_feature_vals/num_siltp_feature_vals/num_siltp_feature_vals/num_siltp_feature_vals).*new_prior_3;

%Compute masks
bg_mask = bg_liks_prior./(bg_liks_prior+fg_liks_prior+new_liks_prior);
fg_mask = fg_liks_prior./(bg_liks_prior+fg_liks_prior+new_liks_prior);
new_mask = new_liks_prior./(bg_liks_prior+fg_liks_prior+new_liks_prior);
fg_mask = (1-bg_mask);

if debug_flag
    figure; 
    subplot(2,3,1); imagesc(bg_prior_3); impixelinfo;
    subplot(2,3,2); imagesc(fg_prior_3); impixelinfo;
    subplot(2,3,3); imagesc(new_prior_3); impixelinfo;
    subplot(2,3,4); imagesc(bg_mask); impixelinfo;
    subplot(2,3,5); imagesc(fg_mask); impixelinfo;
    subplot(2,3,6); imagesc(new_mask); impixelinfo;
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
