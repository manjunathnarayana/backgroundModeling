function pixels_info = get_kernel_points_2d_from_image( img)
%function pixels_info = get_kernel_points_2d_from_image( img)
%Function that returns pixel_info for each pixel in the image in the format
%[ y1 x1 r g b, y1 x2 r g b, ... , y1 xc r g b;
%  y2 x1 r g b, y2 x2 r g b, ... , y2 xc r g b;
%  y3 x1 r g b, y3 x2 r g b, ... , y3 xc r g b;
%  . . . 
%  yr x1 r g b, yr x2 r g b, ... , yr xc r g b;]

if (ndims(img)~=3 || size(img, 3)~=3)
    error('Error - Input image must have 3 colors in get_kernel_points_2d_from_image function');
end

if (~exist('mask'))
    [num_rows num_cols num_colors] = size( img );
    img_linear = reshape( img, [num_rows*num_cols, 3]);
    [x y] = meshgrid(1:num_cols, 1:num_rows);
    y = y(:);
    x = x(:);
    pixels_info_linear(:,1) = y;
    pixels_info_linear(:,2) = x;
    pixels_info_linear(:,3:5) = img_linear;
    pixels_info = reshape( pixels_info_linear, [num_rows num_cols 5]);
else
    error('Error - Functionality not yet implemented in get_kernel_points_2d_from_image function');
end
