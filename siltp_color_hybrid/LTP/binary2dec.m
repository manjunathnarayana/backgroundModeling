function decimal = binary2dec( binary )
%function decimal = binary2dec( binary )
%Function that returns a decimal representation of the numbers that are represented as binary numbers. The last dimension of the binary matrix is the number of digits in the representation (bits).

dims = ndims( binary );
if dims>2
    error('Error in binary2dec. Cannot handle numbers not in 2 dimensional matrix');
end
num_bits = size( binary, 2);

binary_factor = 2.^[num_bits-1:-1:0]';
decimal = binary * binary_factor;
