function result_im = make_small_regions_zero( img, size_threshold)
%function result_im = make_small_regions_zero( img, size_threshold)
%Function that takes a binary image and makes any 1 regions of size less than size_threshold = 0

result_im = img;
%8 connected regions
regions = regionprops( img, 'PixelIdxList', 'Area' );

for con_comp_id=1:size(regions,1)
    if regions(con_comp_id).Area<size_threshold
        result_im(regions(con_comp_id).PixelIdxList) = 0;
    end
end



