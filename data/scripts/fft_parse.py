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
	list of pair (x_i,y_i), where x_i is a numpy matrix and y_i is
	an int, 0 or 1.

Usage: python fft_parse.py [input_directory_path] [output_directory_path]
[num_fft_samples_from_first_song_of_transition_pair]
[num_fft_samples_from_second_song_of_transition_pair]
[num_bins_in_fft]
[label]
[unique_name_output_file]

Output: fftTransitions.txt file in the output directory folder having
		a list of tuples (x_i,y_i)

'''
numbers = re.compile(r'(\d+)')

def main():
	args = sys.argv
	inputDir = args[1]
	outputDir = args[2]
	numPrev = int(args[3])
	numNext = int(args[4])
	numBins = int(args[5])
	label = args[6]
	unique_name = args[7]
	fftCompleteList = parseInput(inputDir)
	transitionList = findTransitions(fftCompleteList,numPrev,numNext,numBins,label)
	if unique_name.endswith('/'):
		unique_name = unique_name[:-1]
	name = unique_name + "-prev-" + str(numPrev) + "-next-" + str(numNext) + "-bins-" + str(numBins) + "-label-" + label
	outputf = open(outputDir + "/" + name, 'w')
	outputf.write(str( numBins * (numPrev + numNext) ) + "\n")
	for item in transitionList:
		outputf.write("%s\n" % str(item))
	outputf.close()



def parseInput(inputDir):
	fftCompleteList = []
	for root, subFolders, files in os.walk(inputDir):
		fftFile = []
		count = 0;
		for fname in sorted(files,key=numericalSort):
			print("Found " + fname)
			if 'linear' in fname or 'Linear' in fname:
				fullpath = os.path.abspath(os.path.join(root,fname))
				f = open(fullpath, 'r')
				fftFile.append(f.readlines())
		if fftFile:
			fftCompleteList.append(fftFile)

	return fftCompleteList

def numericalSort(value):
	parts = numbers.split(value)
	parts[1::2] = map(int, parts[1::2])
	return parts

def findTransitions(fftCompleteList,numPrev,numNext,numBins,label):
	numPlaylists = len(fftCompleteList)
	outerCount = 0
	transitionList = []
	i = iter(fftCompleteList)
	numTrans = 0
	print "Number of playlists: " + str(numPlaylists)
	while outerCount < numPlaylists:
		innerCount = 0
		innerList = i.next()
		maxInner = len(innerList)
		j = iter(innerList)
		prev = j.next()
		print "Number of songs in playlist: " + str(maxInner)
		while innerCount < maxInner-1:
			next = j.next()
			maxLines = len(prev)
			combList =  np.zeros(shape=(numPrev + numNext)* numBins, dtype=float)
			start = 0;
			end = 0

			print(maxLines)
			if(numPrev > maxLines):
				print "The number of FFT samples requested is greater than the number of FFT samples available" 
				print "Failed at song number:" + str(innerCount + 1) + " in playlist number: " + str(outerCount +1) 
				sys.exit(1)
			for a in range(maxLines-numPrev,maxLines):
				end = start + numBins
				splitRes = prev[a].split(" ")
				finalRes = [float(e) for e in splitRes]
				combList[start:end] = finalRes
				start = end

			maxLines = len(next)
			print(maxLines)
			if(numNext > maxLines):
				print "The number of FFT samples requested is greater than the number of FFT samples available"
				print "Failed at song number:" + str(innerCount + 1) + " in playlist number: " + str(outerCount +1) 
				sys.exit(1)
			for a in range(1,numNext+1):
				end = start + numBins
				splitRes = next[a].split(" ")
				finalRes = [float(e) for e in splitRes]
				combList[start:end] = finalRes
				start = end

			numTrans = numTrans + 1
			combList = " ".join(map(str,list(combList)))
			transitionList.append(combList + " " + label)
			prev = next
			innerCount = innerCount + 1
			print "Finished transition " + str(numTrans)

		outerCount = outerCount + 1

	print "Number of transitions created: " + str(numTrans)
	return transitionList

if __name__ == "__main__":
    main()



