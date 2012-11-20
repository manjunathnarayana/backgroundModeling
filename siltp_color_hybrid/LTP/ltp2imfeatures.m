function features = ltp2imfeatures( ltp, image_size )
%function features = ltp2imfeatures( ltp, image_size )
%Function that returns the ltp as decimal feature numbers in 2-d format, shose size is given by the image_size. The ltp is  represented as a binary number

features = binary2dec( ltp );

features = reshape( features, image_size);

