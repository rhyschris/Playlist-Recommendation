import sys
import subprocess
import shlex
import re
import glob
''' Splices raw data.

    Input parameters:
    python splicer.py [playlist name] [requested splits]

    playlist name: the name of the playlist (assumes its in ../raw)
    requested splits: a file containing timestamp | song title pairs

    Outputs a directory of the playlist under its original name (with .plylst extension)
    and the song names in LINEAR order.

    Ex. 
    python splicer.py Celldweller_-_Frozen_\(Celldweller_vs_Blue_Stahli\).mp3 new.txt

'''


filename = sys.argv[1]
print filename
# Youtube comment of description
description = sys.argv[2]

timestamp = re.compile('\d+:\d+')

path = "../raw/plylst.{0}".format(filename)
mkdir = "mkdir -p " + path

subprocess.call(shlex.split(mkdir))

times = []
till_next = []
titles = []

with open(description, 'r') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        l = lines[i]
        match = timestamp.search(l)
        if not match:
            print "No match: " + l
        end =  match.span()[1]
        times.append( match.group() )
        
        ind = l[end:].find(' ')
        title = l[end + ind + 1:]

        titles.append('_'.join(title.split(' ')))
    
# Calculate the time needed for each clip

real_start = []
real_duration = []

for i in range(len(times)):
    minutes, seconds = (float(t) for t in times[i].split(':'))
    real_start.append(60 * minutes + seconds)
    if i > 0:
        real_duration.append(real_start[i] - real_start[i - 1]) 

print real_start, real_duration

# last song is 1000 seconds (we can overestimate)
real_duration.append(1000.0)
for i in range(len(real_start)):
    outputf = path + "/{0}-".format(i) + titles[i].rstrip() + ".mp3"
    print "output file: " + outputf
    cmd = "ffmpeg -ss {0} -i {1} -t {2} -c:a copy {3} ".format(real_start[i], "../raw/" + filename, real_duration[i], outputf)
    print cmd
    subprocess.call(shlex.split(cmd))
    
