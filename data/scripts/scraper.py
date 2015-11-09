__author__ = "rhyschris"

import soundcloud
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
    filepath = "../raw/{0}".format(title)
    proc = subprocess.Popen("wget --content-disposition " + 
                               "{0}?client_id={1}".format(uri, _public_key), shell=True)
    proc.wait()
                      
def get_tracks(pool, count, query):
    print _public_key
    client = soundcloud.Client(client_id=_public_key)
    tracks = client.get('/tracks', q=query, limit=count, license='cc-by')
    
    args = [(track.download_url, track.title) for track in tracks if track.download_url]
    for arg in args:
        print "{0}: Download from {1}".format(arg[1].encode("utf-8"), 
                                              arg[0])

    try:
        pool.map_async(download_from_uri, args).get()
    except KeyboardInterrupt:
        print "Stopping all downloads"
        pool.terminate()
        return

if __name__ == '__main__':
    
    # default query
    query = "fetty wap"
    
    if len(sys.argv) > 1:
        query = sys.argv[1]
    with open("API_KEY", 'r') as f:
        _public_key = f.readline().rstrip()
    
    # Change to the "raw" directory for wget a
    # for more seamless use of --content-disposition
    os.chdir("../raw")
    get_tracks (mp.Pool(4), 50, query)
    os.chdir("../scripts")
    
