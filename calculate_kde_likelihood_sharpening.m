function [liks sigma_image sigmas] = calculate_kde_likelihood_sharpening( pixel_samples, model, indicator, sigma_XYs, sigma_Ys, sigma_UVs, neighborhood_rows, neighborhood_cols, uniform_factor, num_vals, debug_flag)
%function [liks sigma_image sigmas] = calculate_kde_likelihood_sharpening( pixel_samples, model, indicator, sigma_XYs, sigma_Ys, sigma_UVs, neighborhood_rows, neighborhood_cols, uniform_factor, num_vals, debug_flag)
%function that returns the likelihoods of the pixel_samples under given model. 
%function that returns the likelihoods of the pixel_samples under given model. indicator shows (in a soft manner) which pixels in the model belong to this process and which dont. indicator values are used as a weight for each sample in the kde likelihood calculation
%Both pixel_samples are of size r x c x d. model is of size k x r x c x d. indicator is of size k x r x c. sigma is d x d in size
%neighborhood_rows and neighborhood_cols denote the number of pixels to consider on each side as neighbors. A 3x3 neighborhood is defined by neighborhood_rows = neighborhood_cols = 1
%uniform_factor basically is the weight of a uniform distribution mixed to the kde estimate
%num_vals  = number of values each dimension can take
%liks = uniform_factor*uniform_pdf + (1-uniform_factor)*kde_estimate
%function returns likelihoods in liks and the covariance values (indexes) used in the
%adaptive kernel variances (Narayana et. al, CVPR 2012)

if ~exist('num_vals','var')
    num_vals = 256;
end
if ~exist('debug_flag', 'var')
    debug_flag = 0;
end

num_rows = size( pixel_samples, 1);
num_cols = size( pixel_samples, 2);
num_model_frames = size( model, 1);
num_dims = size(pixel_samples, 3);

%Compute a sigma matrix from the covariances (sigmas) given
%Compute the uniform likelihood that results from the given spatial neighborhood
%and given covariance values

i=0;
for Y=sigma_Ys
    for  UV = sigma_UVs
        for XY = sigma_XYs
            i = i+1;
            sigma(:,i) = [ XY XY Y UV UV]';
            sigma_inv(:,i) = 1./sigma(:,i);
            det_s = prod(sigma(:,i));
            const(i) = (det_s^.5)*((2*pi)^(num_dims/2));

            %Uniform distribution for this sigma
            [dx dy] = meshgrid(-neighborhood_cols:neighborhood_cols,-neighborhood_rows:neighborhood_rows);
            uniform_xy_diff = [];
            uniform_xy_diff(:,1) = dx(:);
            uniform_xy_diff(:,2) = dy(:);
            det_xy = sigma(1,i)*sigma(2,i);
            uniform_sigma_inv = [1/sigma(1,i); 1/sigma(2,i)];
            uniform_const = (det_xy^.5)*2*pi;
            %Save the constant for later use in normalization (when normalizing depending on distance of each sample from center)
            xy_const(i) = uniform_const;
            uniform_lik = exp(-.5*(uniform_xy_diff.*uniform_xy_diff)*uniform_sigma_inv);
            uniform_density = 1/num_vals/num_vals/num_vals;
            uniform_contribution(i) = sum( uniform_lik)/uniform_const*uniform_density;
        end
    end
end
num_sigmas = i;
sigmas = sigma;

%If model is empty, then use zero likelihoods for KDE, but add the required uniformcontribution (max of all possible sigma values)
if num_model_frames == 0
    liks = zeros(num_rows, num_cols);
    liks = (uniform_factor*max(uniform_contribution)) + ((1-uniform_factor)*liks);
    sigma_image = zeros(num_rows, num_cols);
    return;
end

%If model is not empty, use kde likelihood estimation

%For each pixel in the image
for i=1:num_rows*num_cols
    
    %get the row and column number
    [r c] = get_2D_coordinates(i, num_rows, num_cols);
    %if r==num_rows && rem(c, 50)==0
    %    c
    %end
    
    %find out the indices of the neighbors
    min_row = max(1, r-neighborhood_rows);
    max_row = min(num_rows, r+neighborhood_rows);
    min_col = max(1, c-neighborhood_cols);
    max_col = min(num_cols, c+neighborhood_cols);
    num_centers = num_model_frames*(max_row-min_row+1)*(max_col-min_col+1);
    %kde data samples
    kde_centers = model(1:num_model_frames, min_row:max_row, min_col:max_col, :);
    kde_centers_reshape = reshape(kde_centers, [num_centers num_dims]);
    current_sample = pixel_samples(r,c,:);
    current_sample_repeat = repmat( current_sample(:)', [ num_centers 1]);
    diff = kde_centers_reshape-current_sample_repeat;
    %Find out which pixels are part of model
    true_mask = indicator(1:num_model_frames, min_row:max_row, min_col:max_col);
    %Reshape to enable efficient multiplication
    true_mask_reshape = reshape(true_mask, [num_centers 1]);
    true_mask_repeat = repmat(true_mask_reshape, [1 num_sigmas]);

    %lik = exp(-.5*sum((diff*inv(sigma)).*diff, 2));
    
    %Compute un-normalized kde likelihood
    %Optimization for diagonal sigma
    %tic 
    lik_indiv = exp(-.5*(diff.*diff)*sigma_inv);
    %fprintf('exp takes %f secs\n', toc);
    %tic 
    %multiply each sample's contribution by the mask and then sum all contributions
    lik_sum_all_sigmas = sum(lik_indiv.*true_mask_repeat, 1);
    %fprintf('lik sum takes %f secs\n', toc);
    %tic 
    %Various normalization schemes possible - Comment out the desired model

    %Uncomment below to normalize by number of frames in model - CVPR 2012 model
    %lik_sum_all_sigmas = lik_sum_all_sigmas./const/num_model_frames;
    %Normalize by sum of mask probabilities
    %lik_sum_all_sigmas = lik_sum_all_sigmas./const/(sum(true_mask_reshape)+eps(0));
    %fprintf('lik sum const takes %f secs\n', toc);
    %tic 
    %The above line will hurt the performance when neighborhood is large. Pixels far away in the neighborhood contribute very little, but will be penalized by the probability of label at that location. Hence, we should normalize by gaussian distance (in xy dimensions) of these points multiplied by probability of label there
    %likelihood of x y distances alone
    xy_lik = exp(-.5*(diff(:,1:2).*diff(:,1:2))*sigma_inv(1:2,:));
    xy_const_repeat = repmat( xy_const, [num_centers 1]);
    xy_liks_sum = xy_lik./xy_const_repeat.*true_mask_repeat;
    norm_factor = sum( xy_liks_sum);
    %Proper normalization, as described in BMVC 2012
    lik_sum_all_sigmas = lik_sum_all_sigmas./const./norm_factor;

    %Find the covariance that results in highest likelihood
    [lik_sum_max_sigma max_sigma_index] = max( lik_sum_all_sigmas);
    %fprintf('lik sum max takes %f secs\n', toc);
   
    %Add desired uniform factor to the likelihood
    liks(r,c) = (uniform_factor*uniform_contribution(max_sigma_index)) + (1-uniform_factor)*lik_sum_max_sigma;
    %save the covariance index also
    sigma_image( r,c) = max_sigma_index;

    %if liks(r,c)>1
    %    disp('in calc kde lik lik is greater than 1');
    %    keyboard;
    %end

    if(isnan(liks(r,c)))
        liks(r,c) = 0;
    end

    debug_flag = 0;
    if (r==36 && c==57 && debug_flag==1)
        disp('in calculate_kde_likelihood');
        keyboard;
    end
end
