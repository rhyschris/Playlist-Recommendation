__author__ = "rhyschris"

import soundcloud
import shlex
import subprocess
import sys
import signal
import os
import re
# Fake threading only used for network I/O
import multiprocessing.dummy as mp


# Hide API key from github
_public_key = None
_api_entry = "http://api.soundcloud.com"

"""
   Scrapes for commercially licensed music, for use with unsupervised
   feature learning.

   Does nothing yet.
"""

def download_from_uri(arg, sizelimit=10*1024**2):
    """ Start method for the threads to download to file """

    uri, title = arg
<<<<<<< HEAD

    filepath = ur"../raw/{0}".format(unicode( '_'.join(title.split(' '))))

    cmd = "wget -O {0}.mp3 {1}?client_id={2}".format(filepath, uri, _public_key)

    proc = subprocess.Popen(shlex.split(cmd))
    proc.wait()
=======
    title = re.sub(r"(\'|\")", "", title)
    filepath = '_'.join(title.split(' ')) + ".mp3"
    if os.path.exists(filepath):
        print "ALREADY EXISTS"
        return

    cmd = "curl -L --max-filesize {3} -o {0} {1}?client_id={2}".format(filepath, uri, _public_key, sizelimit)

    proc = subprocess.call(shlex.split(cmd))
>>>>>>> f6d5806be297276177f0f39d9319bbc01dd5fc0d
    print "DONE"

def get_tracks(pool, count, query, gen=None):
    print gen
    client = soundcloud.Client(client_id=_public_key)
<<<<<<< HEAD
    tracks = client.get('/tracks', q=query, genre=gen, limit=count)

=======
    tracks = client.get('/tracks', q=query, genres=gen, limit=count)
    
>>>>>>> f6d5806be297276177f0f39d9319bbc01dd5fc0d
    args = [(track.stream_url, track.title.encode("utf-8")) for track in tracks if track.stream_url]
    for arg in args:
        print "{0}: Stream from {1}".format(arg[1],
                                              arg[0])

<<<<<<< HEAD

=======
>>>>>>> f6d5806be297276177f0f39d9319bbc01dd5fc0d
    try:
        pool.map_async(download_from_uri, args).get()

        pool.close()
        print "POOL CLOSING"
        pool.join()

    except KeyboardInterrupt:
        print "Stopping all downloads"
        pool.terminate()
        return

if __name__ == '__main__':

    # default query
<<<<<<< HEAD

    query = "Blue Stahli"
    genre = None
=======
    query = None
    genres = ["Dance & EDM"]
>>>>>>> f6d5806be297276177f0f39d9319bbc01dd5fc0d
    if len(sys.argv) > 1:
        query = sys.argv[1]
    with open("API_KEY", 'r') as f:
        _public_key = f.readline().rstrip()

    # Change to the "raw" directory for wget a
    # for more seamless use of --content-disposition
<<<<<<< HEAD
    os.chdir("/media/Elements/raw")
    get_tracks (mp.Pool(4), 10, query, genre)
=======
    os.chdir("../raw")
    get_tracks (mp.Pool(4), 200, query, ",".join(genres))
>>>>>>> f6d5806be297276177f0f39d9319bbc01dd5fc0d
    print "DONE ALL"
    os.chdir("../scripts")

