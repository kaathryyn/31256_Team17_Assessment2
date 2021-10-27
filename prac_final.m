clc
clear all
close all

doTraining = true;

if ~doTraining
    preTrainedDetector = downloadPretrainedYOLOv3Detector();    
end

data = load('gTruth.mat');
PPEDataset = data.gTruth;

% Add the full path to the local vehicle data folder.
PPEDataset.imageFilename = fullfile(pwd, PPEDataset.imageFilename);


    
