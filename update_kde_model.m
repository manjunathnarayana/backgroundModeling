function [new_model new_indicator] = update_kde_model( img_pixels, model, indicator, mask, max_num_frames, version)
%function [new_model new_indicator] = update_kde_model( img_pixels, model, indicator, mask, max_num_frames, version)
%Function that updates the kde model and indicator matrices given the image pixel samples (z x c x d), the old model, old indicator, mask of labels, and max_num_frames that must be kept in the new_model
% img_pixels is a cell of information from various resolutions, so is model, indicator. mask is the mask in the highest resolution

%version = 1 means remove the last frame info, add new frame info at head of mode
%version = 2 - find least valued indicator frame for each pixel and move the current mask value to that location, move current image pixel values to that location in the model. This is buggy - wont work unless the indicators are weighted by time (even then there are complications, some sort of shifting the model frames down is necessary). Also, is untested with the multiresolution cell representation of images and models. Not recommended.
%version = 3 - move all frames down one frame, but dont throw away the oldest frame. If the current image mask has a probability greater than some threshold, only then update the model pixels at the corresponding pixels.
if ~exist('version')
    version = 3;
end

if version==3
    if ~exist('mask_threshold')
        mask_threshold = .5;
    end
    temp_model = model{1};
    num_frames = size( temp_model, 1);
    num_resolutions = size( model, 2);
    for res = 1:num_resolutions
        res_model = model{res};
        res_indicator = indicator{res};
        res_num_rows = size(res_model, 2);
        res_num_cols = size(res_model, 3);
        res_num_dims = size(res_model, 4);
        res_mask = subsample_image( mask, 1.0/res );
        try
        %If the num_frames in the model is less than max, then add one more layer at the beginning of the model
        if num_frames<max_num_frames
            new_model{res}(1,:,:,:) = img_pixels{res};
            new_indicator{res}(1,:,:) = res_mask;
            if num_frames~=0
                new_model{res}(2:num_frames+1,:,:,:) = res_model;
                new_indicator{res}(2:num_frames+1,:,:) = res_indicator;
            end
        %else, remove the last layer in the model and add one new layer in the beginning of the model
        else    
            new_model{res}(2:max_num_frames,:,:,:) = res_model(1:end-1,:,:,:);
            new_indicator{res}(2:max_num_frames,:,:) = res_indicator(1:end-1, :, :);
            %Read the oldest frame values and indicator. Use these when the mask is less than threshold
            oldest_model_frame_values = reshape( res_model(end, :,:,:), [res_num_rows*res_num_cols res_num_dims]);
            oldest_indicator = reshape(res_indicator(end, :, :), [ res_num_rows*res_num_cols 1]);
            %which pixels to keep from the oldest frame
            old_pixels_to_keep = find(res_mask(:)<mask_threshold);
            %Read current image pixels as new frame values
            new_frame_values = reshape(img_pixels{res}, [res_num_rows*res_num_cols res_num_dims]);
            %replace the required pixels in the new frame values
            new_frame_values( old_pixels_to_keep, :) = oldest_model_frame_values( old_pixels_to_keep, :);
            %reshape to be in desired format
            new_frame_values_reshaped = reshape( new_frame_values, [res_num_rows res_num_cols res_num_dims]);
            %Read current mask pixels as new frame indicator
            new_frame_indicator = reshape(res_mask, [res_num_rows*res_num_cols 1]);
            %replace the required pixels in the new frame indicator
            new_frame_indicator( old_pixels_to_keep) = oldest_indicator( old_pixels_to_keep);
            %reshape to be in desired format
            new_frame_indicator_reshaped = reshape( new_frame_indicator, [ res_num_rows res_num_cols]);
            %Make the change to the first frame in the model and the indicator
            new_model{res}(1,:,:,:) = new_frame_values_reshaped;
            new_indicator{res}(1,:,:) = new_frame_indicator_reshaped;
        end 
        catch e
            disp('Error in update_kde_model');
            keyboard
        end
    end
end
if version==1
    temp_model = model{1};
    num_frames = size( temp_model, 1);
    num_resolutions = size( model, 2);
    for res = 1:num_resolutions
        res_model = model{res};
        res_indicator = indicator{res};
        res_num_rows = size(res_model, 2);
        res_num_cols = size(res_model, 3);
        res_num_dims = size(res_model, 4);
        res_mask = subsample_image( mask, 1.0/res );
        try
        %If the num_frames in the model is less than max, then add one more layer at the beginning of the model
        if num_frames<max_num_frames
            new_model{res}(1,:,:,:) = img_pixels{res};
            new_indicator{res}(1,:,:) = res_mask;
            if num_frames~=0
                new_model{res}(2:num_frames+1,:,:,:) = res_model;
                new_indicator{res}(2:num_frames+1,:,:) = res_indicator;
            end
        %else, remove the last layer in the model and add one new layer in the beginning of the model
        else    
            new_model{res}(1,:,:,:) = img_pixels{res};
            new_indicator{res}(1,:,:) = res_mask;
            new_model{res}(2:max_num_frames,:,:,:) = res_model(1:end-1,:,:,:);
            new_indicator{res}(2:max_num_frames,:,:) = res_indicator(1:end-1, :, :);
        end
        catch e
            disp('Error in update_kde_model');
            keyboard
        end
    end
end


%Note - The code below has not been tested with a multiresolution model, after moving to use of cells to represent multiresolution images

if version == 2
    temp_model = model{1};
    num_frames = size( temp_model, 1);
    num_resolutions = size( model, 2);
    for res=1:num_resolutions
        res_model = model{res};
        res_indicator = indicator{res};
        res_num_rows = size(res_model, 2);
        res_num_cols = size(res_model, 3);
        res_num_dims = size(res_model, 4);
        res_mask = subsample_image( mask, 1.0/res );
        disp('This function update_kde_model has not been tested, test before proceeding');
        keyboard
        try
        %If the num_frames in the model is less than max, then add one more layer at the beginning of the model
        if num_frames<max_num_frames
            new_model{res}(1,:,:,:) = img_pixels{res};
            new_indicator{res}(1,:,:) = res_mask;
            if num_frames~=0
                new_model{res}(2:num_frames+1,:,:,:) = res_model;
                new_indicator{res}(2:num_frames+1,:,:) = res_indicator;
            end
        %else, remove the last layer in the model and add one new layer in the beginning of the model
        else    
            %Find out the minimum indicator value and its index
            [min_v min_i] = min( res_indicator, [], 1);
            min_v = squeeze( min_v);
            min_i = squeeze( min_i);
            %Generate y and x matrices so that addressing into the correct row and column using min_i is efficient
            [x_pos_matrix y_pos_matrix] = meshgrid( 1:res_num_cols, 1:res_num_rows);
            true_1d_indices = min_i(:) + ((y_pos_matrix(:)-1)*num_frames) + ((x_pos_matrix(:)-1)*num_frames*res_num_rows);
%            check = indicator( true_1d_indices) - min_v(:)
%           sum(check(:))
            %Update the indicator value at thes indices
            new_indicator{res} = res_indicator;
            new_indicator{res}( true_1d_indices) = mask(:); 
            %Update the model values at these indices
            new_model_reshape = reshape(res_model, [num_frames*res_num_rows*res_num_cols res_num_dims]);
            img_pixels_reshape = reshape( img_pixels{res}, [res_num_rows*res_num_cols res_num_dims]);
            new_model_reshape( true_1d_indices,:) = img_pixels_reshape;
            new_model{res} = reshape( new_model_reshape, [num_frames res_num_rows res_num_cols res_num_dims]);
            %keyboard
        end
        catch e
            disp('Error in update_kde_model');
            keyboard
        end
    end
end
