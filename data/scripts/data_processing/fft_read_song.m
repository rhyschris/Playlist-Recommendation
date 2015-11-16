function fft_read_song (filename, out_dir, basename, time_quanta, mode)
    
    
    delim = '_';
    tq = num2str(time_quanta);
     info = audioinfo(filename);
     ts = info.TotalSamples;
     iter = 0;
     space = 5.5*10^7;
     for s=1:space:ts
         end_sample = s+space;
         if end_sample > ts
             end_sample = ts;
         end
        [wav_file,Fs] = audioread(filename, [s end_sample]);
        iter = iter + 1;
        outfile = fullfile(out_dir, [basename delim num2str(iter) delim mode delim tq '.fft']);
        fft_read_song_helper(wav_file, Fs, outfile, time_quanta, mode);
     end
end