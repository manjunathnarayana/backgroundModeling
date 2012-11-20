function [Y X] = get_2D_coordinates(linear_index, number_of_rows, number_of_columns)
%function [Y X] = get_2D_coordinates(linear_index, number_of_rows, number_of_columns)
%function that returns the coordinates that correspond to the row Y and column X in an array of number_of_rows x number_of_columns where the elements are stored in column major format
% 1 4 7 10 
% 2 5 8 11
% 3 6 9 12

    X = ceil(linear_index/number_of_rows);
    Y = rem(linear_index, number_of_rows);
    if Y == 0
        Y = number_of_rows;
    end
