clc
clear all
close all

doTraining = true;

if ~doTraining
    preTrainedDetector = downloadPretrainedYOLOv3Detector();    
end

%Loading the groundTruth data and dataset
data = load("gTruth.mat");
covidDataset = data.covidDataset;
covidDataset.imageFilename = data.covidDataset.imageFilename;

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
preprocessedTrainingData = transform(trainingData, @(data)preprocess(yolov3Detector, data));
data = read(preprocessedTrainingData);

I = data{1,1};
bbox = data{1,2};
annotatedImage = insertShape(I, 'Rectangle', bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)
reset(preprocessedTrainingData);

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

mbqTrain = minibatchqueue(preprocessedTrainingData, 2,...
        "MiniBatchSize", miniBatchSize,...
        "MiniBatchFcn", @(images, boxes, labels) createBatchData(images, boxes, labels, classNames), ...
        "MiniBatchFormat", ["SSCB", ""],...
        "DispatchInBackground", dispatchInBackground,...
        "OutputCast", ["", "double"]);
    
if doTraining
    
    % Create subplots for the learning rate and mini-batch loss.
    fig = figure;
    [lossPlotter, learningRatePlotter] = configureTrainingProgressPlotter(fig);

    iteration = 0;
    % Custom training loop.
    for epoch = 1:numEpochs
          
        reset(mbqTrain);
        shuffle(mbqTrain);
        
        while(hasdata(mbqTrain))
            iteration = iteration + 1;
           
            [XTrain, YTrain] = next(mbqTrain);
            
            % Evaluate the model gradients and loss using dlfeval and the
            % modelGradients function.
            [gradients, state, lossInfo] = dlfeval(@modelGradients, yolov3Detector, XTrain, YTrain, penaltyThreshold);
    
            % Apply L2 regularization.
            gradients = dlupdate(@(g,w) g + l2Regularization*w, gradients, yolov3Detector.Learnables);
    
            % Determine the current learning rate value.
            currentLR = piecewiseLearningRateWithWarmup(iteration, epoch, learningRate, warmupPeriod, numEpochs);
    
            % Update the detector learnable parameters using the SGDM optimizer.
            [yolov3Detector.Learnables, velocity] = sgdmupdate(yolov3Detector.Learnables, gradients, velocity, currentLR);
    
            % Update the state parameters of dlnetwork.
            yolov3Detector.State = state;
              
            % Display progress.
            displayLossInfo(epoch, iteration, currentLR, lossInfo);  
                
            % Update training plot with new points.
            updatePlots(lossPlotter, learningRatePlotter, iteration, currentLR, lossInfo.totalLoss);
        end        
    end
else
    yolov3Detector = preTrainedDetector;
end
    
%Detecting objects
data = read(testData);
I = data{1};
[bboxes, scores, labels] = detect(yolov3Detector,I);
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I);

%Evaluating model - using average precision metric
results = detect(yolov3Detector, testData, 'MiniBatchSize',1);
[ap,recall,precision] = evaluateDetectionPrecision(results,testData);

    %Plotting precision-recall curve
figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))

%Calculating mean IoU
blds = boxLabelDatastore(gTruth(:,2));
numAnchors = 5;
[anchorBoxes,meanIoU] = estimateAnchorBoxes(blds,numAnchors);

%Evaluating model - number of anchors vs. mean IoU
maxNumAnchors = 15;
meanIoU = zeros([maxNumAnchors,1]);
anchorBoxes = cell(maxNumAnchors,1);
for k = 1:maxNumAnchors
    [anchorBoxes{k},meanIoU(k)] = estimateAnchorBoxes(blds,k);
end

    %Plotting mean IoU vs. number of anchors
figure
plot(1:maxNumAnchors,meanIoU,'-o')
ylabel("Mean IoU")
xlabel("Number of Anchors")
title("Number of Anchors vs. Mean IoU")
