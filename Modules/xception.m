
lgraph = layerGraph();


tempLayers = [
    imageInputLayer([299 299 3],"Name","input_1","Normalization","rescale-symmetric")
    convolution2dLayer([3 3],32,"Name","block1_conv1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","block1_conv1_bn","Epsilon",0.001)
    reluLayer("Name","block1_conv1_act")
    convolution2dLayer([3 3],64,"Name","block1_conv2","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block1_conv2_bn","Epsilon",0.001)
    reluLayer("Name","block1_conv2_act")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_1","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batch_normalization_1","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([3 3],1,64,"Name","block2_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],128,"Name","block2_sepconv1_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block2_sepconv1_bn","Epsilon",0.001)
    reluLayer("Name","block2_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,128,"Name","block2_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],128,"Name","block2_sepconv2_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block2_sepconv2_bn","Epsilon",0.001)
    maxPooling2dLayer([3 3],"Name","block2_pool","Padding","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","conv2d_2","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batch_normalization_2","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block3_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,128,"Name","block3_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],256,"Name","block3_sepconv1_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block3_sepconv1_bn","Epsilon",0.001)
    reluLayer("Name","block3_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,256,"Name","block3_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],256,"Name","block3_sepconv2_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block3_sepconv2_bn","Epsilon",0.001)
    maxPooling2dLayer([3 3],"Name","block3_pool","Padding","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block4_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,256,"Name","block4_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block4_sepconv1_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block4_sepconv1_bn","Epsilon",0.001)
    reluLayer("Name","block4_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block4_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block4_sepconv2_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block4_sepconv2_bn","Epsilon",0.001)
    maxPooling2dLayer([3 3],"Name","block4_pool","Padding","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],728,"Name","conv2d_3","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batch_normalization_3","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block5_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block5_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block5_sepconv1_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block5_sepconv1_bn","Epsilon",0.001)
    reluLayer("Name","block5_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block5_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block5_sepconv2_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block5_sepconv2_bn","Epsilon",0.001)
    reluLayer("Name","block5_sepconv3_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block5_sepconv3_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block5_sepconv3_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block5_sepconv3_bn","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block6_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block6_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block6_sepconv1_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block6_sepconv1_bn","Epsilon",0.001)
    reluLayer("Name","block6_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block6_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block6_sepconv2_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block6_sepconv2_bn","Epsilon",0.001)
    reluLayer("Name","block6_sepconv3_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block6_sepconv3_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block6_sepconv3_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block6_sepconv3_bn","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block7_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block7_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block7_sepconv1_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block7_sepconv1_bn","Epsilon",0.001)
    reluLayer("Name","block7_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block7_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block7_sepconv2_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block7_sepconv2_bn","Epsilon",0.001)
    reluLayer("Name","block7_sepconv3_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block7_sepconv3_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block7_sepconv3_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block7_sepconv3_bn","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block8_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block8_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block8_sepconv1_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block8_sepconv1_bn","Epsilon",0.001)
    reluLayer("Name","block8_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block8_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block8_sepconv2_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block8_sepconv2_bn","Epsilon",0.001)
    reluLayer("Name","block8_sepconv3_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block8_sepconv3_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block8_sepconv3_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block8_sepconv3_bn","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block9_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block9_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block9_sepconv1_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block9_sepconv1_bn","Epsilon",0.001)
    reluLayer("Name","block9_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block9_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block9_sepconv2_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block9_sepconv2_bn","Epsilon",0.001)
    reluLayer("Name","block9_sepconv3_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block9_sepconv3_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block9_sepconv3_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block9_sepconv3_bn","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block10_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block10_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block10_sepconv1_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block10_sepconv1_bn","Epsilon",0.001)
    reluLayer("Name","block10_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block10_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block10_sepconv2_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block10_sepconv2_bn","Epsilon",0.001)
    reluLayer("Name","block10_sepconv3_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block10_sepconv3_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block10_sepconv3_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block10_sepconv3_bn","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block11_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block11_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block11_sepconv1_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block11_sepconv1_bn","Epsilon",0.001)
    reluLayer("Name","block11_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block11_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block11_sepconv2_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block11_sepconv2_bn","Epsilon",0.001)
    reluLayer("Name","block11_sepconv3_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block11_sepconv3_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block11_sepconv3_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block11_sepconv3_bn","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block12_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block12_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block12_sepconv1_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block12_sepconv1_bn","Epsilon",0.001)
    reluLayer("Name","block12_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block12_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block12_sepconv2_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block12_sepconv2_bn","Epsilon",0.001)
    reluLayer("Name","block12_sepconv3_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block12_sepconv3_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block12_sepconv3_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block12_sepconv3_bn","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_11");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block13_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block13_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],728,"Name","block13_sepconv1_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block13_sepconv1_bn","Epsilon",0.001)
    reluLayer("Name","block13_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block13_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],1024,"Name","block13_sepconv2_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block13_sepconv2_bn","Epsilon",0.001)
    maxPooling2dLayer([3 3],"Name","block13_pool","Padding","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],1024,"Name","conv2d_4","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batch_normalization_4","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_12")
    groupedConvolution2dLayer([3 3],1,1024,"Name","block14_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],1536,"Name","block14_sepconv1_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block14_sepconv1_bn","Epsilon",0.001)
    reluLayer("Name","block14_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,1536,"Name","block14_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same")
    convolution2dLayer([1 1],2048,"Name","block14_sepconv2_point-wise","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","block14_sepconv2_bn","Epsilon",0.001)
    reluLayer("Name","block14_sepconv2_act")
    globalAveragePooling2dLayer("Name","avg_pool")
    fullyConnectedLayer(5,"Name","predictions")
    softmaxLayer("Name","predictions_softmax")
    classificationLayer("Name","ClassificationLayer_predictions")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;


lgraph = connectLayers(lgraph,"block1_conv2_act","conv2d_1");
lgraph = connectLayers(lgraph,"block1_conv2_act","block2_sepconv1_channel-wise");
lgraph = connectLayers(lgraph,"batch_normalization_1","add_1/in2");
lgraph = connectLayers(lgraph,"block2_pool","add_1/in1");
lgraph = connectLayers(lgraph,"add_1","conv2d_2");
lgraph = connectLayers(lgraph,"add_1","block3_sepconv1_act");
lgraph = connectLayers(lgraph,"batch_normalization_2","add_2/in2");
lgraph = connectLayers(lgraph,"block3_pool","add_2/in1");
lgraph = connectLayers(lgraph,"add_2","block4_sepconv1_act");
lgraph = connectLayers(lgraph,"add_2","conv2d_3");
lgraph = connectLayers(lgraph,"batch_normalization_3","add_3/in2");
lgraph = connectLayers(lgraph,"block4_pool","add_3/in1");
lgraph = connectLayers(lgraph,"add_3","block5_sepconv1_act");
lgraph = connectLayers(lgraph,"add_3","add_4/in2");
lgraph = connectLayers(lgraph,"block5_sepconv3_bn","add_4/in1");
lgraph = connectLayers(lgraph,"add_4","block6_sepconv1_act");
lgraph = connectLayers(lgraph,"add_4","add_5/in2");
lgraph = connectLayers(lgraph,"block6_sepconv3_bn","add_5/in1");
lgraph = connectLayers(lgraph,"add_5","block7_sepconv1_act");
lgraph = connectLayers(lgraph,"add_5","add_6/in2");
lgraph = connectLayers(lgraph,"block7_sepconv3_bn","add_6/in1");
lgraph = connectLayers(lgraph,"add_6","block8_sepconv1_act");
lgraph = connectLayers(lgraph,"add_6","add_7/in2");
lgraph = connectLayers(lgraph,"block8_sepconv3_bn","add_7/in1");
lgraph = connectLayers(lgraph,"add_7","block9_sepconv1_act");
lgraph = connectLayers(lgraph,"add_7","add_8/in2");
lgraph = connectLayers(lgraph,"block9_sepconv3_bn","add_8/in1");
lgraph = connectLayers(lgraph,"add_8","block10_sepconv1_act");
lgraph = connectLayers(lgraph,"add_8","add_9/in2");
lgraph = connectLayers(lgraph,"block10_sepconv3_bn","add_9/in1");
lgraph = connectLayers(lgraph,"add_9","block11_sepconv1_act");
lgraph = connectLayers(lgraph,"add_9","add_10/in2");
lgraph = connectLayers(lgraph,"block11_sepconv3_bn","add_10/in1");
lgraph = connectLayers(lgraph,"add_10","block12_sepconv1_act");
lgraph = connectLayers(lgraph,"add_10","add_11/in2");
lgraph = connectLayers(lgraph,"block12_sepconv3_bn","add_11/in1");
lgraph = connectLayers(lgraph,"add_11","block13_sepconv1_act");
lgraph = connectLayers(lgraph,"add_11","conv2d_4");
lgraph = connectLayers(lgraph,"block13_pool","add_12/in1");
lgraph = connectLayers(lgraph,"batch_normalization_4","add_12/in2");
