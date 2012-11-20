function img_lab = rgb2lab(img)
%function img_lab = rgb2lab(img)
[num_rows num_cols num_colors] = size(img);
r = img(:,:,1);
g = img(:,:,2);
b = img(:,:,3);

[l a b] = RGB2Lab( r(:), g(:), b(:)); 

l_reshape = reshape( l, [num_rows num_cols]);
a_reshape = reshape( a, [num_rows num_cols]);
b_reshape = reshape( b, [num_rows num_cols]);

img_lab(:, :, 1) = l_reshape;
img_lab(:, :, 2) = a_reshape;
img_lab(:, :, 3) = b_reshape;

