__author__ = "rhyschris"

import soundcloud
import shlex
import subprocess
import sys
import signal
import os
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

def download_from_uri(arg):
    """ Start method for the threads to download to file """

    uri, title = arg

    filepath = ur"../raw/{0}".format(unicode( '_'.join(title.split(' '))))

    cmd = "wget -O {0}.mp3 {1}?client_id={2}".format(filepath, uri, _public_key)

    proc = subprocess.Popen(shlex.split(cmd))
    proc.wait()
    print "DONE"

def get_tracks(pool, count, query, gen=None):
    print _public_key
    client = soundcloud.Client(client_id=_public_key)
    tracks = client.get('/tracks', q=query, genre=gen, limit=count)

    args = [(track.stream_url, track.title.encode("utf-8")) for track in tracks if track.stream_url]
    for arg in args:
        print "{0}: Stream from {1}".format(arg[1],
                                              arg[0])


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

    query = "Blue Stahli"
    genre = None
    if len(sys.argv) > 1:
        query = sys.argv[1]
    with open("API_KEY", 'r') as f:
        _public_key = f.readline().rstrip()

    # Change to the "raw" directory for wget a
    # for more seamless use of --content-disposition
    os.chdir("/media/Elements/raw")
    get_tracks (mp.Pool(4), 10, query, genre)
    print "DONE ALL"
    os.chdir("../scripts")

