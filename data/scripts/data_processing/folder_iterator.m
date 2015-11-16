%iterate over everything in folder and turn it into data


% Start with a folder and get a list of all subfolders.
% Finds and prints names of all PNG, JPG, and TIF images in 
% that folder and all of its subfolders.
clc;    % Clear the command window.
workspace;  % Make sure the workspace panel is showing.
format longg;
format compact;

% Define a starting folder.
start_path = fullfile(matlabroot, '\toolbox\images\imdemos');
% Ask user to confirm or change.
topLevelFolder = uigetdir(start_path);
if topLevelFolder == 0
	return;
end
topLevelOutput = uigetdir(topLevelFolder);
if topLevelOutput == 0
	return;
end
% Get list of all subfolders.
allSubFolders = genpath(topLevelFolder);
% Parse into a cell array.
remain = allSubFolders;
listOfFolderNames = {};
while true
	[singleSubFolder, remain] = strtok(remain, ';');
	if isempty(singleSubFolder)
		break;
	end
	listOfFolderNames = [listOfFolderNames singleSubFolder];
end
numberOfFolders = length(listOfFolderNames)

% Process all image files in those folders.
for k = 1 : numberOfFolders
	% Get this folder and print it out.
	thisFolder = listOfFolderNames{k};
	fprintf('Processing folder %s\n', thisFolder);
	
	% Get PNG files.
	filePattern = sprintf('%s/*.wav', thisFolder);
	baseFileNames = dir(filePattern);
	% Add on TIF files.
	filePattern = sprintf('%s/*.mp3', thisFolder);
	baseFileNames = [baseFileNames; dir(filePattern)];
	% Add on JPG files.
	filePattern = sprintf('%s/*.mpeg', thisFolder);
	baseFileNames = [baseFileNames; dir(filePattern)];
	numberOfAudioFiles = length(baseFileNames);
	% Now we have a list of all files in this folder.
	
	if numberOfAudioFiles >= 1
		% Go through all those image files.
		for f = 1 : numberOfAudioFiles
			fullFileName = fullfile(thisFolder, baseFileNames(f).name);
            fprintf('     Processing %s.\n', baseFileNames(f).name);
            [~,name,~] = fileparts(baseFileNames(f).name);
            tic;
            fft_read_song(fullFileName, topLevelOutput, name, .1, 'quadratic');
            time = toc;
            fprintf('           Took %d seconds\n', time);
			
		end
	else
		fprintf('     Folder %s has no audio files in it.\n', thisFolder);
	end
end
