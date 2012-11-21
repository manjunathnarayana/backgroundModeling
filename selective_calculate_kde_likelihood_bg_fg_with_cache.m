function [selection_map bg_liks bg_sigma_image fg_liks fg_sigma_image] = selective_calculate_kde_likelihood_bg_fg_with_cache( pixel_samples, bg_model, bg_indicator, prev_bg_sigma_image, bg_sigma_XYs, bg_sigma_Ys, bg_sigma_UVs, bg_neighborhood_rows, bg_neighborhood_cols, bg_uniform_factor, fg_model, fg_indicator, prev_fg_sigma_image, fg_sigma_XYs, fg_sigma_Ys, fg_sigma_UVs, fg_neighborhood_rows, fg_neighborhood_cols, fg_uniform_factor, num_vals, debug_flag)
%function [selection_map bg_liks bg_sigma_image fg_liks fg_sigma_image] = selective_calculate_kde_likelihood_bg_fg_with_cache( pixel_samples, bg_model, bg_indicator, prev_bg_sigma_image, bg_sigma_XYs, bg_sigma_Ys, bg_sigma_UVs, bg_neighborhood_rows, bg_neighborhood_cols, bg_uniform_factor, fg_model, fg_indicator, prev_fg_sigma_image, fg_sigma_XYs, fg_sigma_Ys, fg_sigma_UVs, fg_neighborhood_rows, fg_neighborhood_cols, fg_uniform_factor, num_vals, debug_flag)
%function that returns the bg and fg likelihoods of the pixel_samples under prev sigma values. Only in pixels where the bg and fg likelihoods are greater than a factor apart are they set to the values based on old sigma. 
%In pixels where the difference is less than factor, this function will not set the likelihoods based on old sigmas. It is upto another function to set these values using a sharpening sigma procedure. Call selective_calculate_kde_likelihood_sharpening after this function is called to calculate the likelihoods at these pixels.
%indicator shows (in a soft manner) which pixels in the model belong to this process and which dont. indicator values are used as a weight for each sample in the kde likelihood calculation
%the covariance values (sigma) and priors for each class are also input to the function
%Both pixel_samples are of size r x c x d. model is of size k x r x c x d. indicator is of size k x r x c. sigma is d x d in size
%neighborhood_rows and neighborhood_cols denote the number of pixels to consider on each side as neighbors. A 3x3 neighborhood is defined by neighborhood_rows = neighborhood_cols = 1
%prev_sigma_image is the image of indices of sigma values from the previous frame
%uniform_factor basically is the weight of a uniform distribution mixed to the kde estimate
%num_vals  = number of values each dimension can take
%liks = uniform_factor*uniform_pdf + (1-uniform_factor)*kde_estimate
%function returns likelihoods in bg_liks and fg_liks and the covariance values (indexes) in X_sigma_image .
%The algorithm is as described in the adaptive kernel variances caching method from CVPR 2012 (Narayana et. al, CVPR 2012). The model described is from BMVC 2012 (Narayana et. al)

%Procedure:
%(1) Create the bg_sigmas matrix and fg_sigmas matrix 
%(2) Calculate bg and fg likelihoods based on the sigmas picked using the prev_sigma_image
%(3) If bg_lik/fg_lik>2, then set bg_liks and fg_liks at this pixel and set selection_map at this pixel to be 1
%(4) else, set selection_map, bg_liks, fg_liks to 0 at this pixel
%    

if ~exist('num_vals','var')
    num_vals = 256;
end
if ~exist('debug_flag', 'var')
    debug_flag = 0;
end

num_rows = size( pixel_samples, 1);
num_cols = size( pixel_samples, 2);
num_dims = size(pixel_samples, 3); 
%Compute a covariance matrix from the covariances (sigmas) given
%Compute the uniform likelihood that results from the given spatial neighborhood
%and given covariance values

%For bg sigma values
num_bg_model_frames = size( bg_model, 1);
i=0;
for Y=bg_sigma_Ys
    for  UV = bg_sigma_UVs
        for XY = bg_sigma_XYs
            i = i+1;
            bg_sigma(:,i) = [ XY XY Y UV UV]';
            bg_sigma_inv(:,i) = 1./bg_sigma(:,i);
            det_s = prod(bg_sigma(:,i));
            bg_const(i) = (det_s^.5)*((2*pi)^(num_dims/2));

            %Uniform distribution for this sigma
            [dx dy] = meshgrid(-bg_neighborhood_cols:bg_neighborhood_cols,-bg_neighborhood_rows:bg_neighborhood_rows);
            uniform_xy_diff = [];
            uniform_xy_diff(:,1) = dx(:);
            uniform_xy_diff(:,2) = dy(:);
            det_xy = bg_sigma(1,i)*bg_sigma(2,i);
            uniform_sigma_inv = [1/bg_sigma(1,i); 1/bg_sigma(2,i)];
            uniform_const = (det_xy^.5)*2*pi;
            %Save the constant for later use in normalization (when normalizing depending on distance of each sample from center)
            bg_xy_const(i) = uniform_const;
            uniform_lik = exp(-.5*( uniform_xy_diff.*uniform_xy_diff)*uniform_sigma_inv);
            uniform_density = 1/num_vals/num_vals/num_vals;
            bg_uniform_contribution(i) = sum( uniform_lik)/uniform_const*uniform_density;
        end
    end
end
num_bg_sigmas = i;
bg_sigmas = bg_sigma;

%For fg sigma values
num_fg_model_frames = size( fg_model, 1);
i=0;
for Y=fg_sigma_Ys
    for UV = fg_sigma_UVs
        for XY = fg_sigma_XYs
            i = i+1;
            fg_sigma(:,i) = [ XY XY Y UV UV]';
            fg_sigma_inv(:,i) = 1./fg_sigma(:,i);
            det_s = prod(fg_sigma(:,i));
            fg_const(i) = (det_s^.5)*((2*pi)^(num_dims/2));

            %Uniform distribution for this sigma
            [dx dy] = meshgrid(-fg_neighborhood_cols:fg_neighborhood_cols,-fg_neighborhood_rows:fg_neighborhood_rows);
            uniform_xy_diff = [];
            uniform_xy_diff(:,1) = dx(:);
            uniform_xy_diff(:,2) = dy(:);
            det_xy = fg_sigma(1,i)*fg_sigma(2,i);
            uniform_sigma_inv = [1/fg_sigma(1,i); 1/fg_sigma(2,i)];
            uniform_const = (det_xy^.5)*2*pi;
            %Save the constant for later use in normalization (when normalizing depending on distance of each sample from center)
            fg_xy_const(i) = uniform_const;
            uniform_lik = exp(-.5*(  uniform_xy_diff.*uniform_xy_diff )*uniform_sigma_inv);
            uniform_density = 1/num_vals/num_vals/num_vals;
            fg_uniform_contribution(i) = sum( uniform_lik)/uniform_const*uniform_density;
        end
    end
end
num_fg_sigmas = i;
fg_sigmas = fg_sigma;

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
        bg_kde_centers = bg_model(1:num_bg_model_frames, bg_min_row:bg_max_row, bg_min_col:bg_max_col, :);
        bg_kde_centers_reshape = reshape(bg_kde_centers, [bg_num_centers num_dims]);
        current_sample = pixel_samples(r,c,:);
        bg_current_sample_repeat = repmat( current_sample(:)', [ bg_num_centers 1]);
        bg_diff = bg_kde_centers_reshape-bg_current_sample_repeat;
        %Find out which pixels are part of bg model
        bg_true_mask = bg_indicator(1:num_bg_model_frames, bg_min_row:bg_max_row, bg_min_col:bg_max_col);
        %Reshape to enable efficient multiplication
        bg_true_mask_reshape = reshape(bg_true_mask, [bg_num_centers 1]);
        %pick the sigma values for this pixel based on the sigma index in the input sigma image 
        bg_sigma_inv_current = bg_sigma_inv(:, bg_sigma_index);
        bg_const_current = bg_const( bg_sigma_index);
         
        %Compute un-normalized kde likelihood
        bg_lik_indiv = exp(-.5*(bg_diff.*bg_diff)*bg_sigma_inv_current);
        %multiply each sample's contribution by bg mask and then sum all contributions
        bg_lik_sum = sum(bg_lik_indiv.*bg_true_mask_reshape);
        %Uncomment below to normalize by number of frames in model - CVPR 2012 model 
        %bg_lik_sum_norm = bg_lik_sum/bg_const_current/num_bg_model_frames;
        %fprintf('lik sum const takes %f secs\n', toc);
        %tic 
        %The above line will hurt the performance when neighborhood is large. Pixels far away in the neighborhood contribute very little, but will be penalized by the probability of label at that location. Hence, we should normalize by gaussian distance (in xy dimensions) of these points multiplied by probability of label there
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
        fg_kde_centers = fg_model(1:num_fg_model_frames, fg_min_row:fg_max_row, fg_min_col:fg_max_col, :);
        fg_kde_centers_reshape = reshape(fg_kde_centers, [fg_num_centers num_dims]);
        current_sample = pixel_samples(r,c,:);
        fg_current_sample_repeat = repmat( current_sample(:)', [ fg_num_centers 1]);
        fg_diff = fg_kde_centers_reshape-fg_current_sample_repeat;
        %Find out which pixels are part of fg model
        fg_true_mask = fg_indicator(1:num_fg_model_frames, fg_min_row:fg_max_row, fg_min_col:fg_max_col);
        %Reshape to enable efficient multiplication
        fg_true_mask_reshape = reshape(fg_true_mask, [fg_num_centers 1]);
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
