clc
clear all
close all

doTraining = true;

if ~doTraining
    preTrainedDetector = downloadPretrainedYOLOv3Detector();    
end

%Loading the groundTruth data and dataset
data = load("gTruth.mat");
covidDataset = data.gTruth;
covidDataset.imageFilename = data.gTruth.imageFilename;

% split the data into training and test set
rng(0);
shuffledIndices = randperm(height(covidDataset));
idx = floor(0.6 * length(shuffledIndices));
trainingDataTbl = covidDataset(shuffledIndices(1:idx), :);
testDataTbl = covidDataset(shuffledIndices(idx+1:end), :);

% Creating datastores for image and label data
imdsTrain = imageDatastore(trainingDataTbl.imageFilename);
imdsTest = imageDatastore(testDataTbl.imageFilename);

bldsTrain = boxLabelDatastore(trainingDataTbl(:, 2:end));
bldsTest = boxLabelDatastore(testDataTbl(:, 2:end));

trainingData = combine(imdsTrain, bldsTrain);
testData = combine(imdsTest, bldsTest);

%augment training data
augmentedTrainingData = transform(trainingData, @augmentData);
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1,1}, 'Rectangle', data{1,2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData, 'BorderSize', 10)

networkInputSize = [227 227 3];

rng(0)
trainingDataForEstimation = transform(trainingData, @(data)preprocessData(data, networkInputSize));
numAnchors = 6;
[anchors, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);

area = anchors(:, 1).*anchors(:, 2);
[~, idx] = sort(area, 'descend');
anchors = anchors(idx, :);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    };

%Getting pre-trained network
baseNetwork = squeezenet;
classNames = trainingDataTbl.Properties.VariableNames(2:end);

yolov3Detector = yolov3ObjectDetector(baseNetwork, classNames, anchorBoxes, 'DetectionNetworkSource', {'fire9-concat', 'fire5-concat'});

% Data preprocessing
preprocessedTrainingData = transform(augmentedTrainingData, @(data)preprocess(yolov3Detector, data));
data = read(preprocessedTrainingData);

I = data{1,1};
bbox = data{1,2};
annotatedImage = insertShape(I, 'Rectangle', bbox);
annotatedImage = imresize(annotatedImage,3);
figure
imshow(annotatedImage)
%reset(preprocessedTrainingData);

numEpochs = 80;
miniBatchSize = 8;
learningRate = 0.001;
warmupPeriod = 1000;
l2Regularization = 0.0005;
penaltyThreshold = 0.5;
velocity = [];

%train model
if canUseParallelPool
   dispatchInBackground = true;
else
   dispatchInBackground = false;
end

% Error - related to transform() function --> Index in position 2 exceeds
% aray bounds
mbqTrain = minibatchqueue(preprocessedTrainingData, 2,...
        "MiniBatchSize", miniBatchSize,...
        "MiniBatchFcn", @(images, boxes, labels) createBatchData(images, boxes, labels, classNames), ...
        "MiniBatchFormat", ["SSCB", ""],...
        "DispatchInBackground", dispatchInBackground,...
        "OutputCast", ["", "double"]);
    
%Detecting objects
data = read(testData);
I = data{1};
[bboxes, scores, labels] = detect(yolov3Detector,I);
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I);

%Evaluating model - using average precision metric
results = detect(yolov3Detector, testData, 'MiniBatchSize',8);
[ap,recall,precision] = evaluateDetectionPrecision(results,testData);

%Plotting precision-recall curve
figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))
