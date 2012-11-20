function sampled_image = subsample_image( img , scale)
%function sampled_image = subsample_image( img , scale)
%function that returns a subsampled image where each pixel is a result of average of scale X scale pixels in img

sampled_image = imresize( img, scale, 'box');

