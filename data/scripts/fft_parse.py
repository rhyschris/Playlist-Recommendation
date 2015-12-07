#!/usr/bin/python
import sys
import os
import re
import collections
import subprocess
import shlex
import numpy as np
np.set_printoptions(threshold='nan')

''' Parses a File Directory structure of the kind
	Data/name_of_playlist/linear to extract the linear FFTs
	from the files of FFTs in the playlist directoies and then outputs a
	list of tuples (x_i,y_i), where x_i is a numpy matrix and y_i is
	an int.

Usage: python fft_parse.py [input_directory_path] [output_directory_path]
[num_fft_samples_from_first_song_of_transition_pair]
[num_fft_samples_from_second_song_of_transition_pair]
[num_bins_in_fft]

Output: fftTransitions.txt file in the output directory folder having
		a list of tuples (x_i,y_i)

'''

def main():
	args = sys.argv
	inputDir = args[1]
	outputDir = args[2]
	numPrev = args[3]
	numNext = args[4]
	numBins = args[5]
	fftCompleteList = parseInput(inputDir)
	transitionList = findTransitions(fftCompleteList,numPrev,numNext,numBins)
	outputf = open(os.path.join(outputDir, "fftTransitions.txt"), 'w')
	print(outputf)
	print(len(transitionList))
	for item in transitionList:
		outputf.write("%s\n" % str(item))



def parseInput(inputDir):
	fftCompleteList = []
	for root, subFolders, files in os.walk(inputDir):
		fftFile = []
		count = 0;
		for fname in files:
			print("Found " + fname)
			if 'linear' in fname or 'Linear' in fname:
				fullpath = os.path.abspath(os.path.join(root,fname))
				f = open(fullpath, 'r')
				fftFile.append(f.readlines())
		if fftFile:
			fftCompleteList.append(fftFile)

	return fftCompleteList

def findTransitions(fftCompleteList,numPrev,numNext,numBins):
	numPlaylists = len(fftCompleteList)
	outerCount = 0
	transitionList = []
	numPrev = int(numPrev)
	numNext = int(numNext)
	numBins = int(numBins)
	i = iter(fftCompleteList)
	while outerCount < numPlaylists:
		innerCount = 0
		innerList = i.next()
		maxInner = len(innerList)
		j = iter(innerList)
		prev = j.next()
		while innerCount < maxInner-1:
			next = j.next()
			maxLines = len(prev)
			combList =  np.zeros(shape=(numPrev + numNext)* numBins, dtype=float)
			start = 0;
			end = 0

			if(numPrev > maxLines):
				print("The number of FFT samples requested is greater than the number of FFT samples available")
			for a in range(maxLines-numPrev,maxLines):
				end = start + numBins
				splitRes = prev[a].split(" ")
				finalRes = [float(e) for e in splitRes]
				combList[start:end] = finalRes
				start = end

			maxLines = len(next)
			if(numNext > maxLines):
				print("The number of FFT samples requested is greater than the number of FFT samples available")
			for a in range(1,numNext+1):
				end = start + numBins
				splitRes = next[a].split(" ")
				finalRes = [float(e) for e in splitRes]
				combList[start:end] = finalRes
				start = end

			combList = " ".join(map(str,list(combList)))
			transitionList.append((combList,1))
			prev = next
			innerCount = innerCount + 1

		outerCount = outerCount + 1

	return transitionList









if __name__ == "__main__":
    main()



