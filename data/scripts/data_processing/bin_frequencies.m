function [Y_out ] = bin_frequencies( f, Y, note_freq)
%bin_frequencies Bins the frequencies into notes.
%   From a musical table, bins the frequencies into discrete notes.


size_f = size(f,2);
size_y = size(Y,2);
Y_out = zeros(1,size(note_freq,2));


%assumes size_f and size_y are equal
%k = dsearchn(note_freq', f');


for i = 1:size_f
    [c_unused, index] = min(abs(note_freq - f(i)));
    Y_out(index) = Y_out(index) + Y(i);
end



end

