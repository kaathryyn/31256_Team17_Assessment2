clc
clear all
close all

imagesize = [224 224 3];
load gTruth
numClasses = 5;
anchorBoxes = [
    43 59
    18 22
    23 29
    84 109
    ];

%residual network
base = resnet50('Weights','none');
inputlayer = base.Layers(1)
middle = base.Layers(2:174)
finallayer = base.Layers(175:end)
baseNetwork = [inputlayer middle finallayer]

featureLayer = 'activation_40_relu';

lgraph = yolov2Layers(imagesize,numClasses,anchorBoxes,base,featureLayer);

options = trainingOptions('sgdm',...
            'MiniBatchSize', 128, ...
            'InitialLearnRate', 1e-3,...
            'MaxEpochs',1,...
            'CheckpointPath',tempdir,...
            'Shuffle','every-epoch');
        
PPEDataset = gTruth;
[detector,info] = trainYOLOv2ObjectDetector(PPEDataset,lgraph,options);

matlabroot ='C:\Users\kwonr\MATLAB\Projects\IPPR1\test'
DatasetPath=fullfile(matlabroot);
Data = imageDatastore(DatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
           
CountLabel = Data.countEachLabel;

trainData1 = Data;
[trainData] = Data;

%create resnet
netWidth =16;
layers = [
    imageInputLayer([224 224 3],'Name', 'input')
    convolution2dLayer(3, netWidth, 'Padding','same','Name','convInp')
    batchNormalizationLayer('Name','BNInp')
    reluLayer('Name','reluInp')
    
    convolutionalUnit(netWidth,1,'S1U1')
    additionLayer(2,'Name','add11')
    reluLayer('Name','relu11')
    convolutionalUnit(netWidth,1,'S1U2')
    additionLayer(2,'Name','add12')
    reluLayer('Name','relu12')
    
    convolutionalUnit(2*netWidth,2,'S2U1')
    additionLayer(2,'Name','add21')
    reluLayer('Name','relu21')
    convolutionalUnit(2*netWidth,1,'S2U2')
    additionLayer(2,'Name','add22')
    reluLayer('Name','relu22')

    convolutionalUnit(4*netWidth,2,'S3U1')
    additionLayer(2,'Name','add31')
    reluLayer('Name','relu31')
    convolutionalUnit(4*netWidth,1,'S3U2')
    additionLayer(2,'Name','add32')
    reluLayer('Name','relu32')
    
    averagePooling2dLayer(8,'Name','globalPool')
    fullyConnectedLayer(4,'Name','fcFinal')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
    
    ];
    

    
    
    













