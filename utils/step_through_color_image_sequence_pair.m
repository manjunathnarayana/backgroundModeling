function step_through_color_image_sequence_pair( image_sequence1, image_sequence2 )
%Function that allows to step through a pair of image sequences  
%image_sequence is a 4-D matrix rows x cols x 3(colors) x num_images in case of color images
%Note - Both images must be scaled from 0-255 for proper display normalization

    if isempty( image_sequence1 ) || isempty( image_sequence2)
        disp(' Error in step_through_color_image_sequence_pair() : empty image_sequence1 or 2');
        return;
    end

    if (size(image_sequence1,3) ~= 3) ||(size(image_sequence2,3) ~= 3) 
        disp(' Error in step_through_color_image_sequence() : image_sequence1 and 2 should have 3 colors (3rd dimension = 3) for color image sequences');
        return;
    end

 if (ndims(image_sequence1) ~= 4) ||(ndims(image_sequence2) ~= 4) 
        disp(' Error in step_through_color_image_sequence() : image_sequence1 and 2 should have 3 dimensions for gray image sequence');
        return;
    end

    if ~strcmp(class(image_sequence1),'double')
        image_sequence1 = double(image_sequence1);
    end
    if ~strcmp(class(image_sequence2),'double')
        image_sequence2 = double(image_sequence2);
    end
    

    imsize1=[size(image_sequence1, 1) size(image_sequence1, 2)];
    imsize2=[size(image_sequence2, 1) size(image_sequence2, 2)];
    movLength1 = size(image_sequence1, 4);
    movLength2 = size(image_sequence2, 4);

    if movLength1~=movLength2
        disp(' Error in step_through_color_image_sequence_pair() : image_sequence1 and 2 should have same length');
        return;
    end
    movLength = movLength1;

    series1 = image_sequence1;
    series2 = image_sequence2;

    im1=series1(:,:,:,1);
    im2=series2(:,:,:,1);
    % Create the figure with screne position and dimensions
    f = figure('Visible', 'off', 'Position', [500, 500, 400, 400]);
    colormap(gray)
    % Create the axis for drawing the image
    %ha = axes('Units', 'pixels', 'Position', [10, 25, imsize(2), imsize(1)]);

    % Show image on axis
    subplot(2,1,1)
    %imagesc(series1(:,:,:,1)/255.9);
    imagesc(series1(:,:,:,1)/255.9, [ 0 1]);
    axis equal;
    axis tight;
    subplot(2,1,2)
    %imagesc(series2(:,:,:,1)/255.9);
    imagesc(series2(:,:,:,1)/255.9, [ 0 1]);
    axis equal;
    axis tight;
     
    % Initialize parameters
    param1 = .5;
    param1Min = .5;
    param1Max = movLength+.49;
    %param1Pos = [10, imsize(1)+25+40, 200, 15];
    %param1Label = 'Rotate';
    %param1LabelPos = [imsize(2)+10+10, imsize(1)+25+10, 80, 15];

    param1Pos = [ 20, 20, 300, 15];
    param1Label = 'Slide frame num';
    param1LabelPos = [ 20, 0, 80, 15];


    % Create the parameter adjustment sliders
    param1Text = uicontrol('Style', 'Text', 'String', param1Label, ...
                            'Position', param1LabelPos);
    param1Slider = uicontrol('Style', 'slider', ...
                             'Min', param1Min, 'Max', param1Max, ...
                             'Value', param1, 'Position', param1Pos, ...
                             'Callback', @param1Callback);

    slider_step = get(param1Slider, 'SliderStep');
    %Set the slider arrow to step through one frame at a time
    slider_step(1) = 1.0/movLength;
    set(param1Slider, 'SliderStep', slider_step);

    align([param1Label, param1Slider], 'Center', 'None');

    % Add action listeners to get real-time adjustment callbacks
    param1Listener = handle.listener(param1Slider, 'ActionEvent', ...
                                        @param1Callback);

    
    % Perform computation on image
    %val = computeImage(im);
    %fprintf('Computation: %f\n', val);
    fprintf('Frame - %d ', round(param1));
    
    % Show the layed out figure
    set(f, 'Visible', 'on');


    old_param_val=1;

    % Create callback functions for sliders
    function param1Callback(hObj, event, ax)
        % Retrieve the updated parameter value
        param1 = get(hObj, 'Value');

        % Reshow modified image
        if round(param1)~=old_param_val
          [imMod1 imMod2] = param1ModifyImage();
          old_param_val=round(param1)
          imageModified(imMod1, imMod2);
        end

    end

    % Updated image and computation
    function imageModified(imNew1, imNew2)
        subplot(2,1,1);
        %imagesc(imNew1/255.9);
        imagesc(imNew1/255.9, [ 0 1]);
        axis equal;
        axis tight;
        subplot(2,1,2);
        %imagesc(imNew2/255.9);
        imagesc(imNew2/255.9, [ 0 1]);
        axis equal;
        axis tight;
        
        % Update computation
        %val = computeImage(imNew);     
        %fprintf('Computation: %f\n', val);
        fprintf('Frame - %d ', round(param1));
    end

    % Create functions for updating image and computation
    function [imMod1 imMod2] = param1ModifyImage()
        % Use parameters to modify image
          imMod1 = series1(:,:,:,round(param1));
          imMod2 = series2(:,:,:,round(param1));
    end

    function val = computeImage(im)
        val = round(param1);
    end
end
