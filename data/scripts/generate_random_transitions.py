#!/usr/bin/python
import sys
import os
import re
import collections
import subprocess
import shlex
import random
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
[num_samples_needed]

Output: unique_name.txt file in the output directory folder having
		a list of pairs (x_i,y_i)

'''

def main():
	args = sys.argv
	inputDir = args[1]
	outputDir = args[2]
	numPrev = int(args[3])
	numNext = int(args[4])
	numBins = int(args[5])
	label = args[6]
	unique_name = args[7]
	numExamples= int(args[8])
	fftCompleteList = parseInput(inputDir)
	transitionList = findTransitions(fftCompleteList,numPrev,numNext,numBins,label,numExamples)
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
		count = 0;
		for fname in files:
			print("Found " + fname)
			fullpath = os.path.abspath(os.path.join(root,fname))
			f = open(fullpath, 'r')
			fftCompleteList.append(f.readlines())

	return fftCompleteList

def findTransitions(fftCompleteList,numPrev,numNext,numBins,label, numExamples):
	numPlaylists = len(fftCompleteList)
	outerCount = 0
	transitionList = []
	numTrans = 0
	pairsCompleted = []
	numFiles = len(fftCompleteList)
	while outerCount < numExamples:
		first = (random.randint(0,1e5))%numFiles
		second = (random.randint(0,1e5))%numFiles
		tupleExist = [i for i, v in enumerate(pairsCompleted) if v[0] == first and v[1] == second]
		if tupleExist:
			continue

		prev = fftCompleteList[first]
		next = fftCompleteList[second]
		maxLines = len(prev)
		combList =  np.zeros(shape=(numPrev + numNext)* numBins, dtype=float)
		start = 0;
		end = 0

		if(numPrev > maxLines):
			print "The number of FFT/MFCC samples requested is greater than the number of FFT/MFCC samples available" 
			print "Failed at song number:" + str(innerCount + 1) + " in playlist number: " + str(outerCount +1) 
			sys.exit(1)

		for a in range(maxLines-numPrev,maxLines):
			end = start + numBins
			splitRes = prev[a].split(" ")
			finalRes = [float(e) for e in splitRes]
			combList[start:end] = finalRes
			start = end

		maxLines = len(next)
		if(numNext > maxLines):
			print "The number of FFT/MFCC samples requested is greater than the number of FFT/MFCC samples available"
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

		pairsCompleted.append((first,second))
		outerCount = outerCount + 1

	print "Number of transitions created: " + str(numTrans)
	return transitionList

if __name__ == "__main__":
    main()



