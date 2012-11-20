function feature_image = im2quadsilpt( img, t )
%function feature_image = im2quadsilpt( img, t )
%Function that converts img to 4-digit siltp with threshold t

ltp = im2siltp_quad( img, t);
feature_image = ltp2imfeatures( ltp, size(img));
