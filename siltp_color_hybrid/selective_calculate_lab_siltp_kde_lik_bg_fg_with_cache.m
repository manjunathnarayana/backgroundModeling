function [selection_map bg_liks bg_sigma_image fg_liks fg_sigma_image] = selective_calculate_lab_siltp_kde_lik_bg_fg_with_cache( pixel_samples, bg_model, bg_indicator, prev_bg_sigma_image, bg_sigma_XYs, bg_sigma_Ls, bg_sigma_ABs, bg_sigma_LTPs, bg_neighborhood_rows, bg_neighborhood_cols, bg_uniform_factor, fg_model, fg_indicator, prev_fg_sigma_image, fg_sigma_XYs, fg_sigma_Ls, fg_sigma_ABs, fg_sigma_LTPs, fg_neighborhood_rows, fg_neighborhood_cols, fg_uniform_factor, num_color_vals, num_siltp_vals, debug_flag)
%function [selection_map bg_liks bg_sigma_image fg_liks fg_sigma_image] = selective_calculate_lab_siltp_kde_lik_bg_fg_with_cache( pixel_samples, bg_model, bg_indicator, prev_bg_sigma_image, bg_sigma_XYs, bg_sigma_Ls, bg_sigma_ABs, bg_sigma_LTPs, bg_neighborhood_rows, bg_neighborhood_cols, bg_uniform_factor, fg_model, fg_indicator, prev_fg_sigma_image, fg_sigma_XYs, fg_sigma_Ls, fg_sigma_ABs, fg_sigma_LTPs, fg_neighborhood_rows, fg_neighborhood_cols, fg_uniform_factor, num_vals, debug_flag)
%function that returns the bg and fg likelihoods of the pixel_samples under prev sigma values. Only in pixels where the bg and fg likelihoods are greater than a factor apart are they set to the values based on old sigma. 
%In pixels where the difference is less than factor, this function will not set the likelihoods based on old sigmas. It is upto another function to set these values using a sharpening sigma procedure
%indicator shows (in a soft manner) which pixels in the model belong to this process and which dont. indicator values are used as a weight for each sample in the kde likelihood calculation
%the covariance values (sigma) and priors for each class are also input to the function
%Both pixel_samples are of size r x c x d. model is of size k x r x c x d. indicator is of size k x r x c. sigma is d x d in size
%neighborhood_rows and neighborhood_cols denote the number of pixels to consider on each side as neighbors. A 3x3 neighborhood is defined by neighborhood_rows = neighborhood_cols = 1
%prev_sigma_image is the image of indices of sigma values from the previous frame
%uniform_factor basically is the weight of a uniform distribution mixed to the kde estimate
%num_color_vals  = number of values each color feature can take
%num_siltp_vals  = number of values each siltp feature can tale
%liks = uniform_factor*uniform_pdf + (1-uniform_factor)*kde_estimate
%The algorithm is as described in the adaptive kernel variances caching method from CVPR 2012 (Narayana et. al, CVPR 2012), except the normalization is as described in BMVC 2012

%Procedure:
%(1) Create the bg_sigmas matrix and fg_sigmas matrix 
%(2) Calculate bg and fg likelihoods based on the sigmas picked using the prev_sigma_image
%(3) If bg_lik/fg_lik>2, then set bg_liks and fg_liks at this pixel and set selection_map at this pixel to be 1
%(4) else, set selection_map, bg_liks, fg_liks to 0 at this pixel
%    

if ~exist('num_color_vals','var')
    num_color_vals = 256;
end
if ~exist('num_siltp_vals','var')
    num_siltp_vals = 81;
end

num_rows = size( pixel_samples, 1);
num_cols = size( pixel_samples, 2);
num_bg_model_frames = size( bg_model, 1);
num_fg_model_frames = size( fg_model, 1);
num_dims = size(pixel_samples, 3);
num_siltp_resolutions = num_dims-5;

%Calculate the dec2binary lookup table so that dec2bin() is not called repeatedly
dec2bin_lutable = dec2bin([0:255],  8);

%Compute a covariance matrix from the covariances (sigmas) given
%Compute the uniform likelihood that results from the given spatial neighborhood
%and given covariance values

%For bg sigma values
i=0;
for LTP=bg_sigma_LTPs
    for AB = bg_sigma_ABs
        for L = bg_sigma_Ls
            for XY = bg_sigma_XYs
                i = i+1;
                bg_sigma(1:5,i) = [ XY XY L AB AB]';
                bg_sigma(6:num_dims,i) = LTP;
                bg_sigma_inv(:,i) = 1./bg_sigma(:,i);
                det_s = prod(bg_sigma(:,i));
                bg_const(i) = (det_s^.5)*((2*pi)^(num_dims/2));

                %Uniform distribution for this sigma
                [dx dy] = meshgrid(-bg_neighborhood_cols:bg_neighborhood_cols,-bg_neighborhood_rows:bg_neighborhood_rows);
                uniform_xy_diff = [];
                uniform_xy_diff(:,1) = dx(:);
                uniform_xy_diff(:,2) = dy(:);
                det_xy = bg_sigma(1,i)*bg_sigma(2,i);
                bg_uniform_sigma_inv = [1/bg_sigma(1,i); 1/bg_sigma(2,i)];
                bg_uniform_const = (det_xy^.5)*2*pi;
                %Save the constant for later use in normalization (when normalizing depending on distance of each sample from center)
                bg_xy_const(i) = bg_uniform_const;
                bg_uniform_lik = exp(-.5*(uniform_xy_diff.*uniform_xy_diff)*bg_uniform_sigma_inv);
                bg_uniform_density = 1/num_color_vals/num_color_vals/num_color_vals/(num_siltp_vals^(num_siltp_resolutions));
                bg_uniform_contribution(i) = sum( bg_uniform_lik)/bg_uniform_const*bg_uniform_density;
            end
        end 
    end
end
num_bg_sigmas = i;
bg_sigmas = bg_sigma;

%For fg sigma values
i=0;
for LTP=fg_sigma_LTPs
    for AB = fg_sigma_ABs
        for L = fg_sigma_Ls
            for XY = fg_sigma_XYs
                i = i+1;
                fg_sigma(1:5,i) = [ XY XY L AB AB]';
                fg_sigma(6:num_dims,i) = LTP;
                fg_sigma_inv(:,i) = 1./fg_sigma(:,i);
                det_s = prod(fg_sigma(:,i));
                fg_const(i) = (det_s^.5)*((2*pi)^(num_dims/2));

                %Uniform distribution for this sigma
                [dx dy] = meshgrid(-fg_neighborhood_cols:fg_neighborhood_cols,-fg_neighborhood_rows:fg_neighborhood_rows);
                uniform_xy_diff = [];
                uniform_xy_diff(:,1) = dx(:);
                uniform_xy_diff(:,2) = dy(:);
                det_xy = fg_sigma(1,i)*fg_sigma(2,i);
                fg_uniform_sigma_inv = [1/fg_sigma(1,i); 1/fg_sigma(2,i)];
                fg_uniform_const = (det_xy^.5)*2*pi;
                %Save the constant for later use in normalization (when normalizing depending on distance of each sample from center)
                fg_xy_const(i) = fg_uniform_const;
                fg_uniform_lik = exp(-.5*(uniform_xy_diff.*uniform_xy_diff)*fg_uniform_sigma_inv);
                fg_uniform_density = 1/num_color_vals/num_color_vals/num_color_vals/(num_siltp_vals^(num_siltp_resolutions));
                fg_uniform_contribution(i) = sum( fg_uniform_lik)/fg_uniform_const*fg_uniform_density;
            end
        end 
    end
end
num_fg_sigmas = i;
fg_sigmas = fg_sigma;

if ~exist('num_vals','var')
    num_vals = 256;
end
if ~exist('debug_flag', 'var')
    debug_flag = 0;
end

%Set selection_map to all 0's. Update it to 1 when any pixel's likelihood is decided based on the criteria for confidence
selection_map = zeros( num_rows, num_cols);
%set likelihood and sigma images to 0's
bg_liks = selection_map;
fg_liks = selection_map;
bg_sigma_image = selection_map;
fg_sigma_image = selection_map;

%if there are no data in the bg or fg models, return. No pixel likelihoods have been set
if num_bg_model_frames == 0 || num_fg_model_frames == 0
    return;
end

%For each pixel in the image
for i=1:num_rows*num_cols
    %get the row and column number
    [r c] = get_2D_coordinates(i, num_rows, num_cols);
    %if r==num_rows && rem(c, 50)==0
    %    c
    %end
    %read the sigma index for this pixel from the input sigma image for bg and fg
    bg_sigma_index = prev_bg_sigma_image(r, c);
    fg_sigma_index = prev_fg_sigma_image(r, c);
    %if fg sigma is non-zero (in some pixels, it may be 0, if no fg has been seen in that pixel location in the model so far)
    if fg_sigma_index ~=0
        %compute bg likelihoods
        %find out the indices of the neighbors for bg process
        bg_min_row = max(1, r-bg_neighborhood_rows);
        bg_max_row = min(num_rows, r+bg_neighborhood_rows);
        bg_min_col = max(1, c-bg_neighborhood_cols);
        bg_max_col = min(num_cols, c+bg_neighborhood_cols);
        bg_num_centers = num_bg_model_frames*(bg_max_row-bg_min_row+1)*(bg_max_col-bg_min_col+1);
        %kde data samples
        bg_kde_centers = bg_model(1:num_bg_model_frames, bg_min_row:bg_max_row, bg_min_col:bg_max_col, 1:num_dims-1);
        bg_kde_centers_reshape = reshape(bg_kde_centers, [bg_num_centers num_dims-1]);
        current_sample = pixel_samples(r,c,1:num_dims-1);
        current_sample_bg_repeat = repmat( current_sample(:)', [ bg_num_centers 1]);
        bg_diff = bg_kde_centers_reshape-current_sample_bg_repeat;
        %Calculate the difference in SILTP feature space -- difference in bit values
        for siltp_res=1:num_siltp_resolutions
            bg_siltp_kde_centers = bg_model(1:num_bg_model_frames, bg_min_row:bg_max_row, bg_min_col:bg_max_col, siltp_res+5);
            bg_siltp_kde_centers_reshape = reshape(bg_siltp_kde_centers, [bg_num_centers 1]);
            %siltp_kde_centers_reshape_bin = dec2bin(siltp_kde_centers_reshape,8);
            bg_siltp_kde_centers_reshape_bin = dec2bin_lutable( bg_siltp_kde_centers_reshape+1,:);
            current_sample_siltp = pixel_samples(r,c,siltp_res+5);
            %current_sample_siltp_bin = dec2bin( current_sample_siltp, 8);
            current_sample_siltp_bin = dec2bin_lutable( current_sample_siltp+1, :);
            bg_current_sample_siltp_repeat_bin = repmat( current_sample_siltp_bin, [ bg_num_centers 1]);
    %        current_sample_siltp_repeat_bin = dec2bin( current_sample_siltp_repeat, 8);
            bg_diff_siltp = bg_siltp_kde_centers_reshape_bin-bg_current_sample_siltp_repeat_bin;
            bg_abs_diff_siltp = sum( abs(bg_diff_siltp), 2);
            %concatenate the siltp diff to the diff matrix
            bg_diff(:,siltp_res+5) = bg_abs_diff_siltp;
        end
        
        %Find out which pixels are part of model
        bg_true_mask = bg_indicator(1:num_bg_model_frames, bg_min_row:bg_max_row, bg_min_col:bg_max_col);
        %Reshape to enable efficient multiplication
        bg_true_mask_reshape = reshape(bg_true_mask, [bg_num_centers 1]);
        bg_true_mask_repeat = repmat(bg_true_mask_reshape, [1 num_bg_sigmas]);
    
        %pick the sigma values for this pixel based on the sigma index in the input sigma image 
        bg_sigma_inv_current = bg_sigma_inv(:, bg_sigma_index);
        bg_const_current = bg_const( bg_sigma_index);
        %Compute un-normalized kde likelihood
        bg_lik_indiv = exp(-.5*(bg_diff.*bg_diff)*bg_sigma_inv_current);
        %multiply each sample's contribution by bg mask and then sum all contributions
        bg_lik_sum = sum(bg_lik_indiv.*bg_true_mask_reshape);
        %Normalize by number of frames - Uncomment line below to normalize using CVPR 2012 method
        %normalize by required constant for bg sigma and by number of frames
        %bg_lik_sum_norm = bg_lik_sum/bg_const_current/num_bg_model_frames;
        %The above line will hurt the performance when neighborhood is large. Pixels far away in the neighborhood contribute very little, but will be penalized by the probability of label at that location. Perhaps we should normalize by gaussian distance (in xy dimensions) of these points multiplied by probability of label there
        %likelihood of x y distances alone
        bg_xy_lik = exp(-.5*(bg_diff(:,1:2).*bg_diff(:,1:2))*bg_sigma_inv_current(1:2,:));
        bg_xy_const_current = bg_xy_const( bg_sigma_index );
        bg_xy_const_repeat = repmat( bg_xy_const_current, [bg_num_centers 1]);
        bg_xy_liks_sum = bg_xy_lik./bg_xy_const_repeat.*bg_true_mask_reshape;
        bg_norm_factor = sum( bg_xy_liks_sum);
        %Proper normalization, as described in BMVC 2012
        bg_lik_sum_norm = bg_lik_sum/bg_const_current/bg_norm_factor;
  
        %Add desired uniform factor to the likelihood
        bg_lik = (bg_uniform_factor*bg_uniform_contribution(bg_sigma_index)) + (1-bg_uniform_factor)*bg_lik_sum_norm;

        %Now compute fg likelihoods
        %find out the indices of the neighbors for fg process
        fg_min_row = max(1, r-fg_neighborhood_rows);
        fg_max_row = min(num_rows, r+fg_neighborhood_rows);
        fg_min_col = max(1, c-fg_neighborhood_cols);
        fg_max_col = min(num_cols, c+fg_neighborhood_cols);
        fg_num_centers = num_fg_model_frames*(fg_max_row-fg_min_row+1)*(fg_max_col-fg_min_col+1);
        %kde data samples
        fg_kde_centers = fg_model(1:num_fg_model_frames, fg_min_row:fg_max_row, fg_min_col:fg_max_col, 1:num_dims-1);
        fg_kde_centers_reshape = reshape(fg_kde_centers, [fg_num_centers num_dims-1]);
        current_sample = pixel_samples(r,c,1:num_dims-1);
        current_sample_fg_repeat = repmat( current_sample(:)', [ fg_num_centers 1]);
        fg_diff = fg_kde_centers_reshape-current_sample_fg_repeat;
        %Calculate the difference in SILTP feature space -- difference in bit values
        for siltp_res=1:num_siltp_resolutions
            fg_siltp_kde_centers = fg_model(1:num_fg_model_frames, fg_min_row:fg_max_row, fg_min_col:fg_max_col, siltp_res+5);
            fg_siltp_kde_centers_reshape = reshape(fg_siltp_kde_centers, [fg_num_centers 1]);
            %siltp_kde_centers_reshape_bin = dec2bin(siltp_kde_centers_reshape,8);
            fg_siltp_kde_centers_reshape_bin = dec2bin_lutable( fg_siltp_kde_centers_reshape+1,:);
            current_sample_siltp = pixel_samples(r,c,siltp_res+5);
            %current_sample_siltp_bin = dec2bin( current_sample_siltp, 8);
            current_sample_siltp_bin = dec2bin_lutable( current_sample_siltp+1, :);
            fg_current_sample_siltp_repeat_bin = repmat( current_sample_siltp_bin, [ fg_num_centers 1]);
    %        current_sample_siltp_repeat_bin = dec2bin( current_sample_siltp_repeat, 8);
            fg_diff_siltp = fg_siltp_kde_centers_reshape_bin-fg_current_sample_siltp_repeat_bin;
            fg_abs_diff_siltp = sum( abs(fg_diff_siltp), 2);
            %concatenate the siltp diff to the diff matrix
            fg_diff(:,siltp_res+5) = fg_abs_diff_siltp;
        end
        %Find out which pixels are part of model
        fg_true_mask = fg_indicator(1:num_fg_model_frames, fg_min_row:fg_max_row, fg_min_col:fg_max_col);
        %Reshape to enable efficient multiplication
        fg_true_mask_reshape = reshape(fg_true_mask, [fg_num_centers 1]);
        fg_true_mask_repeat = repmat(fg_true_mask_reshape, [1 num_fg_sigmas]);
        %pick the sigma values for this pixel based on the sigma index in the input sigma image 
        fg_sigma_inv_current = fg_sigma_inv(:, fg_sigma_index);
        fg_const_current = fg_const( fg_sigma_index);
        %Compute un-normalized kde likelihood
        fg_lik_indiv = exp(-.5*(fg_diff.*fg_diff)*fg_sigma_inv_current);
        %multiply each sample's contribution by fg mask and then sum all contributions
        fg_lik_sum = sum(fg_lik_indiv.*fg_true_mask_reshape);
        %Uncomment below to normalize by number of frames in model - CVPR 2012 model 
        %fg_lik_sum_norm = fg_lik_sum/fg_const_current/num_fg_model_frames;
        %fprintf('lik sum const takes %f secs\n', toc);
        %tic 
        %The above line will hurt the performance when neighborhood is large. Pixels far away in the neighborhood contribute very little, but will be penalized by the probability of label at that location. Hence, we should normalize by gaussian distance (in xy dimensions) of these points multiplied by probability of label there
        %likelihood of x y distances alone
        fg_xy_lik = exp(-.5*(fg_diff(:,1:2).*fg_diff(:,1:2))*fg_sigma_inv_current(1:2,:));
        fg_xy_const_current = fg_xy_const( fg_sigma_index );
        fg_xy_const_repeat = repmat( fg_xy_const_current, [fg_num_centers 1]);
        fg_xy_liks_sum = fg_xy_lik./fg_xy_const_repeat.*fg_true_mask_reshape;
        fg_norm_factor = sum( fg_xy_liks_sum);
        %Proper normalization, as described in BMVC 2012
        fg_lik_sum_norm = fg_lik_sum/fg_const_current/fg_norm_factor;
        
        %Add desired uniform factor to the likelihood
        fg_lik = (fg_uniform_factor*fg_uniform_contribution(fg_sigma_index)) + (1-fg_uniform_factor)*fg_lik_sum_norm;

        %If bg likelihood is greater than fg likelihood by a given factor, then 
        %use the above bg and fg likelihoods and set the selection_map to 1.
        selection_factor = 2;
        %if bg_lik/fg_lik > 2 
        if bg_lik/fg_lik > selection_factor
            bg_liks( r, c) = bg_lik;
            fg_liks( r, c) = fg_lik;
            bg_sigma_image( r,c) = bg_sigma_index;
            fg_sigma_image( r,c) = fg_sigma_index;
            selection_map( r, c) = 1;
        end
    end
end
