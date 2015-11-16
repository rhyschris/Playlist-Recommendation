function fft_read_song_helper( wav_file, Fs, outfile, time_quanta, mode )
 %set step_size equal to .5s:
    L = size(wav_file, 1);
    fprintf('           Total samples: %d\n', L);

    num_notes = 96;
    notes = linspace(1,num_notes,num_notes);
    notes_shifted = notes - 49;
    note_freq = (2^(1/12)) .^ notes_shifted * 440;
    dlmwrite(outfile, note_freq, 'delimiter', ' ');

    if strcmp(mode, 'linear') 
        step_size = 2^nextpow2(time_quanta*Fs);
        seconds = step_size/Fs;
        num = L/step_size;
        i=0;

        for v=1:step_size:L-step_size
        i=i+1;
        NFFT = step_size; % Next power of 2 from length of y
        Y = fft(wav_file(v:v+step_size),NFFT)/L;
        f = Fs/2*linspace(0,1,NFFT/2+1);
        out_frequency = 2*abs(Y(1:NFFT/2+1));

        Y_out = bin_frequencies(f, out_frequency, note_freq);
        dlmwrite(outfile, Y_out, '-append', 'delimiter', ' ');


        end
        return;
    end
    if strcmp(mode, 'quadratic')
        time_quanta = time_quanta/4;
        update = 1;
        i = 0;
        v = 1;
        while (v <= L)
            i=i+update;
            step_size = 2^nextpow2(i*time_quanta*Fs);
            NFFT = step_size; % Next power of 2 from length of y
            Y = fft(wav_file(v:v+step_size),NFFT)/L;
            f = Fs/2*linspace(0,1,NFFT/2+1);
            out_frequency = 2*abs(Y(1:NFFT/2+1));

            Y_out = bin_frequencies(f, out_frequency, note_freq);
            dlmwrite(outfile, Y_out, '-append', 'delimiter', ' ');
            v = v+step_size;
            if (v >= L/2)
                update = -1;
            end
        end
                
    end


end

