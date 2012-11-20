function ltp = im2ltp_quad( img, t)
%function ltp = im2ltp_quad( img, t)
%Function that returns a ltp representation for the img. The ltp is defined as follows (This is SILTP by Shengcai Liao et. al. CVPR 2010)
% For the k (8) neighbors of each pixel c, 
%    ltp_k_th_bit = +1, if I_k > (1+t)I_c
%    ltp_k_th_bit = -1, if I_k < (1-t)I_c
%    ltp_k_th_bit = 0, otherwise
% t is set to 0 if undefined
% Number of features returned = 4 per pixel
% - 2 -
% 3 - 1
% - 4 -
%

if ~strcmp(class(img), 'double')
    img = double(img);
end

if(~exist('t'))
    t = 0;
end

ltp = ones(numel(img), 4);
%ltp_temp = ones(numel(img), 9);
ltp_temp = zeros(numel(img), 18);
ltp_plus = zeros(numel(img), 8);
ltp_minus = zeros(numel(img), 8);

% check each of the 8 neighbors 
h_plus = zeros(1, 9);
h_plus(5) = -(1+t);

h_minus = zeros(1, 9);
h_minus(5) = -(1-t);

for i=1:9
    curr_h_plus = h_plus;
    curr_h_minus = h_minus;
    if i ~= 5
        %create the filter
        curr_h_plus(ind2sub([3, 3], i)) = 1;    
        curr_h_minus(ind2sub([3, 3], i)) = 1;    
        %convolve image with filter to find out plus part 
        img_c_plus = imfilter(img, reshape(curr_h_plus, 3, 3), 'same');
        %convolve image with filter to find out minus part 
        img_c_minus = imfilter(img, reshape(curr_h_minus, 3, 3), 'same');

        %find out where ltp_plus is active
        ltp_plus(:, i) = (img_c_plus(:) > 0);
        %find out where ltp_minus is active
        ltp_minus(:, i) = (img_c_minus(:) < 0);
        %keyboard
    end;
    
end;
for i=1:9

    %Change the ltp whereever the ltp_plus is active
    plus_indices = find(ltp_plus(:,i)>0);
    ltp_temp( plus_indices, (i-1)*2+1 ) = 0;
    ltp_temp( plus_indices, (i-1)*2+2 ) = 1;

    %Change the ltp whereever the ltp_minus is active
    minus_indices = find(ltp_minus(:,i)>0);
    ltp_temp(minus_indices, (i-1)*2+1 ) = 1;
    ltp_temp(minus_indices, (i-1)*2+2 ) = 0;
end

% rearrange the columns to get the relevant 8 t-bits from the 18 columns
% 01,02 07,08 13,14        -  3,4  -
% 03,04 09,10 15,16  ==>  5,6  -  1,2 
% 05,06 11,12 17,18        -  7,8  -

ltp = [ltp_temp(:, 11) ltp_temp(:, 12) ltp_temp(:, 3) ltp_temp(:, 4) ltp_temp(:, 7) ltp_temp(:, 8) ltp_temp(:, 15)  ltp_temp(:, 16)];

