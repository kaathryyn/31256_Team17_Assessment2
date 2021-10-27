
lgraph = layerGraph();
tempLayers = [
    imageInputLayer([299 299 3],"Name","input_1","Normalization","rescale-symmetric")
    convolution2dLayer([3 3],32,"Name","conv2d_1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","batch_normalization_1","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_1")
    convolution2dLayer([3 3],32,"Name","conv2d_2","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batch_normalization_2","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_2")
    convolution2dLayer([3 3],64,"Name","conv2d_3","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_3","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_3")
    maxPooling2dLayer([3 3],"Name","max_pooling2d_1","Stride",[2 2])
    convolution2dLayer([1 1],80,"Name","conv2d_4","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batch_normalization_4","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_4")
    convolution2dLayer([3 3],192,"Name","conv2d_5","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batch_normalization_5","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_5")
    maxPooling2dLayer([3 3],"Name","max_pooling2d_2","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv2d_9","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_9","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_9")
    convolution2dLayer([3 3],96,"Name","conv2d_10","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_10","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_10")
    convolution2dLayer([3 3],96,"Name","conv2d_11","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_11","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_11")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],96,"Name","conv2d_6","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_6","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],48,"Name","conv2d_7","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_7","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_7")
    convolution2dLayer([5 5],64,"Name","conv2d_8","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_8","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","average_pooling2d_1","Padding","same")
    convolution2dLayer([1 1],64,"Name","conv2d_12","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_12","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_12")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","mixed_5b");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_13","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_13","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_13")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_16","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_16","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_16")
    convolution2dLayer([3 3],48,"Name","conv2d_17","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_17","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_17")
    convolution2dLayer([3 3],64,"Name","conv2d_18","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_18","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_18")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_14","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_14","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_14")
    convolution2dLayer([3 3],32,"Name","conv2d_15","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_15","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_15")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(3,"Name","block35_1_mixed")
    convolution2dLayer([1 1],320,"Name","block35_1_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block35_1_scale",0.17)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block35_1")
    reluLayer("Name","block35_1_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_22","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_22","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_22")
    convolution2dLayer([3 3],48,"Name","conv2d_23","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_23","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_23")
    convolution2dLayer([3 3],64,"Name","conv2d_24","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_24","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_24")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_20","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_20","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_20")
    convolution2dLayer([3 3],32,"Name","conv2d_21","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_21","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_21")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_19","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_19","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_19")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(3,"Name","block35_2_mixed")
    convolution2dLayer([1 1],320,"Name","block35_2_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block35_2_scale",0.17)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block35_2")
    reluLayer("Name","block35_2_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_28","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_28","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_28")
    convolution2dLayer([3 3],48,"Name","conv2d_29","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_29","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_29")
    convolution2dLayer([3 3],64,"Name","conv2d_30","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_30","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_30")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_26","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_26","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_26")
    convolution2dLayer([3 3],32,"Name","conv2d_27","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_27","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_27")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_25","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_25","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_25")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(3,"Name","block35_3_mixed")
    convolution2dLayer([1 1],320,"Name","block35_3_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block35_3_scale",0.17)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block35_3")
    reluLayer("Name","block35_3_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_31","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_31","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_31")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_34","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_34","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_34")
    convolution2dLayer([3 3],48,"Name","conv2d_35","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_35","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_35")
    convolution2dLayer([3 3],64,"Name","conv2d_36","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_36","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_36")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_32","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_32","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_32")
    convolution2dLayer([3 3],32,"Name","conv2d_33","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_33","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_33")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(3,"Name","block35_4_mixed")
    convolution2dLayer([1 1],320,"Name","block35_4_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block35_4_scale",0.17)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block35_4")
    reluLayer("Name","block35_4_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_37","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_37","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_37")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_40","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_40","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_40")
    convolution2dLayer([3 3],48,"Name","conv2d_41","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_41","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_41")
    convolution2dLayer([3 3],64,"Name","conv2d_42","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_42","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_42")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_38","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_38","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_38")
    convolution2dLayer([3 3],32,"Name","conv2d_39","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_39","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_39")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(3,"Name","block35_5_mixed")
    convolution2dLayer([1 1],320,"Name","block35_5_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block35_5_scale",0.17)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block35_5")
    reluLayer("Name","block35_5_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_43","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_43","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_43")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_44","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_44","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_44")
    convolution2dLayer([3 3],32,"Name","conv2d_45","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_45","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_45")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_46","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_46","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_46")
    convolution2dLayer([3 3],48,"Name","conv2d_47","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_47","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_47")
    convolution2dLayer([3 3],64,"Name","conv2d_48","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_48","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_48")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(3,"Name","block35_6_mixed")
    convolution2dLayer([1 1],320,"Name","block35_6_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block35_6_scale",0.17)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block35_6")
    reluLayer("Name","block35_6_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_49","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_49","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_49")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_52","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_52","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_52")
    convolution2dLayer([3 3],48,"Name","conv2d_53","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_53","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_53")
    convolution2dLayer([3 3],64,"Name","conv2d_54","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_54","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_54")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_50","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_50","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_50")
    convolution2dLayer([3 3],32,"Name","conv2d_51","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_51","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_51")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(3,"Name","block35_7_mixed")
    convolution2dLayer([1 1],320,"Name","block35_7_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block35_7_scale",0.17)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block35_7")
    reluLayer("Name","block35_7_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_58","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_58","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_58")
    convolution2dLayer([3 3],48,"Name","conv2d_59","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_59","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_59")
    convolution2dLayer([3 3],64,"Name","conv2d_60","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_60","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_60")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_56","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_56","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_56")
    convolution2dLayer([3 3],32,"Name","conv2d_57","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_57","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_57")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_55","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_55","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_55")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(3,"Name","block35_8_mixed")
    convolution2dLayer([1 1],320,"Name","block35_8_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block35_8_scale",0.17)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block35_8")
    reluLayer("Name","block35_8_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_61","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_61","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_61")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_64","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_64","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_64")
    convolution2dLayer([3 3],48,"Name","conv2d_65","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_65","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_65")
    convolution2dLayer([3 3],64,"Name","conv2d_66","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_66","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_66")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_62","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_62","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_62")
    convolution2dLayer([3 3],32,"Name","conv2d_63","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_63","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_63")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(3,"Name","block35_9_mixed")
    convolution2dLayer([1 1],320,"Name","block35_9_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block35_9_scale",0.17)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block35_9")
    reluLayer("Name","block35_9_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_70","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_70","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_70")
    convolution2dLayer([3 3],48,"Name","conv2d_71","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_71","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_71")
    convolution2dLayer([3 3],64,"Name","conv2d_72","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_72","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_72")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_68","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_68","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_68")
    convolution2dLayer([3 3],32,"Name","conv2d_69","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_69","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_69")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv2d_67","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_67","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_67")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(3,"Name","block35_10_mixed")
    convolution2dLayer([1 1],320,"Name","block35_10_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block35_10_scale",0.17)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block35_10")
    reluLayer("Name","block35_10_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = maxPooling2dLayer([3 3],"Name","max_pooling2d_3","Stride",[2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","conv2d_74","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_74","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_74")
    convolution2dLayer([3 3],256,"Name","conv2d_75","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_75","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_75")
    convolution2dLayer([3 3],384,"Name","conv2d_76","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","batch_normalization_76","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_76")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],384,"Name","conv2d_73","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","batch_normalization_73","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_73")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(3,"Name","mixed_6a");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_77","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_77","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_77")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_78","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_78","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_78")
    convolution2dLayer([1 7],160,"Name","conv2d_79","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_79","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_79")
    convolution2dLayer([7 1],192,"Name","conv2d_80","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_80","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_80")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_1_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_1_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_1_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_1")
    reluLayer("Name","block17_1_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_81","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_81","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_81")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_82","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_82","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_82")
    convolution2dLayer([1 7],160,"Name","conv2d_83","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_83","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_83")
    convolution2dLayer([7 1],192,"Name","conv2d_84","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_84","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_84")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_2_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_2_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_2_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_2")
    reluLayer("Name","block17_2_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_86","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_86","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_86")
    convolution2dLayer([1 7],160,"Name","conv2d_87","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_87","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_87")
    convolution2dLayer([7 1],192,"Name","conv2d_88","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_88","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_88")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_85","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_85","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_85")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_3_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_3_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_3_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_3")
    reluLayer("Name","block17_3_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_89","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_89","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_89")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_90","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_90","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_90")
    convolution2dLayer([1 7],160,"Name","conv2d_91","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_91","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_91")
    convolution2dLayer([7 1],192,"Name","conv2d_92","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_92","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_92")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_4_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_4_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_4_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_4")
    reluLayer("Name","block17_4_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_94","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_94","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_94")
    convolution2dLayer([1 7],160,"Name","conv2d_95","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_95","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_95")
    convolution2dLayer([7 1],192,"Name","conv2d_96","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_96","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_96")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_93","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_93","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_93")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_5_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_5_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_5_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_5")
    reluLayer("Name","block17_5_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_97","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_97","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_97")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_98","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_98","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_98")
    convolution2dLayer([1 7],160,"Name","conv2d_99","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_99","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_99")
    convolution2dLayer([7 1],192,"Name","conv2d_100","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_100","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_100")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_6_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_6_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_6_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_6")
    reluLayer("Name","block17_6_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_101","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_101","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_101")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_102","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_102","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_102")
    convolution2dLayer([1 7],160,"Name","conv2d_103","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_103","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_103")
    convolution2dLayer([7 1],192,"Name","conv2d_104","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_104","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_104")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_7_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_7_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_7_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_7")
    reluLayer("Name","block17_7_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_105","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_105","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_105")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_106","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_106","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_106")
    convolution2dLayer([1 7],160,"Name","conv2d_107","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_107","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_107")
    convolution2dLayer([7 1],192,"Name","conv2d_108","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_108","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_108")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_8_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_8_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_8_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_8")
    reluLayer("Name","block17_8_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_109","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_109","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_109")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_110","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_110","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_110")
    convolution2dLayer([1 7],160,"Name","conv2d_111","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_111","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_111")
    convolution2dLayer([7 1],192,"Name","conv2d_112","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_112","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_112")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_9_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_9_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_9_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_9")
    reluLayer("Name","block17_9_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_113","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_113","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_113")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_114","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_114","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_114")
    convolution2dLayer([1 7],160,"Name","conv2d_115","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_115","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_115")
    convolution2dLayer([7 1],192,"Name","conv2d_116","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_116","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_116")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_10_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_10_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_10_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_10")
    reluLayer("Name","block17_10_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_118","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_118","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_118")
    convolution2dLayer([1 7],160,"Name","conv2d_119","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_119","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_119")
    convolution2dLayer([7 1],192,"Name","conv2d_120","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_120","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_120")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_117","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_117","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_117")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_11_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_11_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_11_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_11")
    reluLayer("Name","block17_11_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_122","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_122","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_122")
    convolution2dLayer([1 7],160,"Name","conv2d_123","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_123","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_123")
    convolution2dLayer([7 1],192,"Name","conv2d_124","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_124","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_124")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_121","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_121","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_121")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_12_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_12_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_12_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_12")
    reluLayer("Name","block17_12_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_126","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_126","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_126")
    convolution2dLayer([1 7],160,"Name","conv2d_127","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_127","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_127")
    convolution2dLayer([7 1],192,"Name","conv2d_128","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_128","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_128")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_125","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_125","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_125")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_13_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_13_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_13_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_13")
    reluLayer("Name","block17_13_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_129","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_129","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_129")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_130","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_130","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_130")
    convolution2dLayer([1 7],160,"Name","conv2d_131","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_131","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_131")
    convolution2dLayer([7 1],192,"Name","conv2d_132","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_132","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_132")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_14_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_14_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_14_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_14")
    reluLayer("Name","block17_14_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_134","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_134","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_134")
    convolution2dLayer([1 7],160,"Name","conv2d_135","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_135","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_135")
    convolution2dLayer([7 1],192,"Name","conv2d_136","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_136","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_136")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_133","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_133","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_133")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_15_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_15_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_15_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_15")
    reluLayer("Name","block17_15_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_138","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_138","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_138")
    convolution2dLayer([1 7],160,"Name","conv2d_139","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_139","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_139")
    convolution2dLayer([7 1],192,"Name","conv2d_140","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_140","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_140")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_137","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_137","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_137")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_16_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_16_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_16_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_16")
    reluLayer("Name","block17_16_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_141","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_141","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_141")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_142","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_142","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_142")
    convolution2dLayer([1 7],160,"Name","conv2d_143","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_143","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_143")
    convolution2dLayer([7 1],192,"Name","conv2d_144","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_144","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_144")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_17_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_17_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_17_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_17")
    reluLayer("Name","block17_17_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_146","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_146","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_146")
    convolution2dLayer([1 7],160,"Name","conv2d_147","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_147","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_147")
    convolution2dLayer([7 1],192,"Name","conv2d_148","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_148","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_148")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_145","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_145","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_145")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_18_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_18_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_18_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_18")
    reluLayer("Name","block17_18_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_149","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_149","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_149")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_150","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_150","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_150")
    convolution2dLayer([1 7],160,"Name","conv2d_151","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_151","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_151")
    convolution2dLayer([7 1],192,"Name","conv2d_152","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_152","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_152")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_19_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_19_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_19_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_19")
    reluLayer("Name","block17_19_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_154","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_154","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_154")
    convolution2dLayer([1 7],160,"Name","conv2d_155","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_155","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_155")
    convolution2dLayer([7 1],192,"Name","conv2d_156","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_156","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_156")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_153","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_153","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_153")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block17_20_mixed")
    convolution2dLayer([1 1],1088,"Name","block17_20_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block17_20_scale",0.1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block17_20")
    reluLayer("Name","block17_20_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","conv2d_159","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_159","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_159")
    convolution2dLayer([3 3],288,"Name","conv2d_160","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","batch_normalization_160","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_160")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","conv2d_157","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_157","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_157")
    convolution2dLayer([3 3],384,"Name","conv2d_158","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","batch_normalization_158","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_158")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","conv2d_161","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_161","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_161")
    convolution2dLayer([3 3],288,"Name","conv2d_162","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_162","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_162")
    convolution2dLayer([3 3],320,"Name","conv2d_163","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","batch_normalization_163","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_163")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = maxPooling2dLayer([3 3],"Name","max_pooling2d_4","Stride",[2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","mixed_7a");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_164","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_164","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_164")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_165","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_165","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_165")
    convolution2dLayer([1 3],224,"Name","conv2d_166","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_166","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_166")
    convolution2dLayer([3 1],256,"Name","conv2d_167","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_167","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_167")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block8_1_mixed")
    convolution2dLayer([1 1],2080,"Name","block8_1_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block8_1_scale",0.2)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block8_1")
    reluLayer("Name","block8_1_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_169","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_169","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_169")
    convolution2dLayer([1 3],224,"Name","conv2d_170","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_170","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_170")
    convolution2dLayer([3 1],256,"Name","conv2d_171","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_171","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_171")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_168","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_168","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_168")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block8_2_mixed")
    convolution2dLayer([1 1],2080,"Name","block8_2_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block8_2_scale",0.2)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block8_2")
    reluLayer("Name","block8_2_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_172","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_172","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_172")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_173","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_173","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_173")
    convolution2dLayer([1 3],224,"Name","conv2d_174","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_174","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_174")
    convolution2dLayer([3 1],256,"Name","conv2d_175","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_175","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_175")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block8_3_mixed")
    convolution2dLayer([1 1],2080,"Name","block8_3_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block8_3_scale",0.2)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block8_3")
    reluLayer("Name","block8_3_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_176","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_176","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_176")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_177","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_177","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_177")
    convolution2dLayer([1 3],224,"Name","conv2d_178","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_178","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_178")
    convolution2dLayer([3 1],256,"Name","conv2d_179","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_179","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_179")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block8_4_mixed")
    convolution2dLayer([1 1],2080,"Name","block8_4_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block8_4_scale",0.2)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block8_4")
    reluLayer("Name","block8_4_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_180","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_180","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_180")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_181","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_181","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_181")
    convolution2dLayer([1 3],224,"Name","conv2d_182","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_182","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_182")
    convolution2dLayer([3 1],256,"Name","conv2d_183","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_183","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_183")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block8_5_mixed")
    convolution2dLayer([1 1],2080,"Name","block8_5_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block8_5_scale",0.2)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block8_5")
    reluLayer("Name","block8_5_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_185","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_185","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_185")
    convolution2dLayer([1 3],224,"Name","conv2d_186","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_186","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_186")
    convolution2dLayer([3 1],256,"Name","conv2d_187","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_187","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_187")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_184","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_184","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_184")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block8_6_mixed")
    convolution2dLayer([1 1],2080,"Name","block8_6_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block8_6_scale",0.2)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block8_6")
    reluLayer("Name","block8_6_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_189","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_189","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_189")
    convolution2dLayer([1 3],224,"Name","conv2d_190","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_190","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_190")
    convolution2dLayer([3 1],256,"Name","conv2d_191","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_191","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_191")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_188","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_188","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_188")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block8_7_mixed")
    convolution2dLayer([1 1],2080,"Name","block8_7_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block8_7_scale",0.2)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block8_7")
    reluLayer("Name","block8_7_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_192","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_192","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_192")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_193","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_193","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_193")
    convolution2dLayer([1 3],224,"Name","conv2d_194","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_194","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_194")
    convolution2dLayer([3 1],256,"Name","conv2d_195","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_195","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_195")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block8_8_mixed")
    convolution2dLayer([1 1],2080,"Name","block8_8_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block8_8_scale",0.2)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block8_8")
    reluLayer("Name","block8_8_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_197","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_197","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_197")
    convolution2dLayer([1 3],224,"Name","conv2d_198","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_198","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_198")
    convolution2dLayer([3 1],256,"Name","conv2d_199","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_199","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_199")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_196","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_196","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_196")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block8_9_mixed")
    convolution2dLayer([1 1],2080,"Name","block8_9_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block8_9_scale",0.2)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block8_9")
    reluLayer("Name","block8_9_ac")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_200","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_200","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_200")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_201","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_201","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_201")
    convolution2dLayer([1 3],224,"Name","conv2d_202","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_202","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_202")
    convolution2dLayer([3 1],256,"Name","conv2d_203","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_203","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","activation_203")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","block8_10_mixed")
    convolution2dLayer([1 1],2080,"Name","block8_10_conv","Padding","same")
    nnet.inceptionresnetv2.layer.ScalingFactorLayer("block8_10_scale",1)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block8_10")
    convolution2dLayer([1 1],1536,"Name","conv_7b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","conv_7b_bn","Epsilon",0.001,"ScaleLearnRateFactor",0)
    reluLayer("Name","conv_7b_ac")
    globalAveragePooling2dLayer("Name","avg_pool")
    fullyConnectedLayer(5,"Name","predictions")
    softmaxLayer("Name","predictions_softmax")
    classificationLayer("Name","ClassificationLayer_predictions")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"max_pooling2d_2","conv2d_9");
lgraph = connectLayers(lgraph,"max_pooling2d_2","conv2d_6");
lgraph = connectLayers(lgraph,"max_pooling2d_2","conv2d_7");
lgraph = connectLayers(lgraph,"max_pooling2d_2","average_pooling2d_1");
lgraph = connectLayers(lgraph,"activation_8","mixed_5b/in2");
lgraph = connectLayers(lgraph,"activation_11","mixed_5b/in3");
lgraph = connectLayers(lgraph,"activation_6","mixed_5b/in1");
lgraph = connectLayers(lgraph,"activation_12","mixed_5b/in4");
lgraph = connectLayers(lgraph,"mixed_5b","conv2d_13");
lgraph = connectLayers(lgraph,"mixed_5b","conv2d_16");
lgraph = connectLayers(lgraph,"mixed_5b","conv2d_14");
lgraph = connectLayers(lgraph,"mixed_5b","block35_1/in1");
lgraph = connectLayers(lgraph,"activation_13","block35_1_mixed/in1");
lgraph = connectLayers(lgraph,"activation_18","block35_1_mixed/in3");
lgraph = connectLayers(lgraph,"activation_15","block35_1_mixed/in2");
lgraph = connectLayers(lgraph,"block35_1_scale","block35_1/in2");
lgraph = connectLayers(lgraph,"block35_1_ac","conv2d_22");
lgraph = connectLayers(lgraph,"block35_1_ac","conv2d_20");
lgraph = connectLayers(lgraph,"block35_1_ac","conv2d_19");
lgraph = connectLayers(lgraph,"block35_1_ac","block35_2/in1");
lgraph = connectLayers(lgraph,"activation_24","block35_2_mixed/in3");
lgraph = connectLayers(lgraph,"activation_21","block35_2_mixed/in2");
lgraph = connectLayers(lgraph,"activation_19","block35_2_mixed/in1");
lgraph = connectLayers(lgraph,"block35_2_scale","block35_2/in2");
lgraph = connectLayers(lgraph,"block35_2_ac","conv2d_28");
lgraph = connectLayers(lgraph,"block35_2_ac","conv2d_26");
lgraph = connectLayers(lgraph,"block35_2_ac","conv2d_25");
lgraph = connectLayers(lgraph,"block35_2_ac","block35_3/in1");
lgraph = connectLayers(lgraph,"activation_27","block35_3_mixed/in2");
lgraph = connectLayers(lgraph,"activation_30","block35_3_mixed/in3");
lgraph = connectLayers(lgraph,"activation_25","block35_3_mixed/in1");
lgraph = connectLayers(lgraph,"block35_3_scale","block35_3/in2");
lgraph = connectLayers(lgraph,"block35_3_ac","conv2d_31");
lgraph = connectLayers(lgraph,"block35_3_ac","conv2d_34");
lgraph = connectLayers(lgraph,"block35_3_ac","conv2d_32");
lgraph = connectLayers(lgraph,"block35_3_ac","block35_4/in1");
lgraph = connectLayers(lgraph,"activation_31","block35_4_mixed/in1");
lgraph = connectLayers(lgraph,"activation_36","block35_4_mixed/in3");
lgraph = connectLayers(lgraph,"activation_33","block35_4_mixed/in2");
lgraph = connectLayers(lgraph,"block35_4_scale","block35_4/in2");
lgraph = connectLayers(lgraph,"block35_4_ac","conv2d_37");
lgraph = connectLayers(lgraph,"block35_4_ac","conv2d_40");
lgraph = connectLayers(lgraph,"block35_4_ac","conv2d_38");
lgraph = connectLayers(lgraph,"block35_4_ac","block35_5/in1");
lgraph = connectLayers(lgraph,"activation_37","block35_5_mixed/in1");
lgraph = connectLayers(lgraph,"activation_39","block35_5_mixed/in2");
lgraph = connectLayers(lgraph,"activation_42","block35_5_mixed/in3");
lgraph = connectLayers(lgraph,"block35_5_scale","block35_5/in2");
lgraph = connectLayers(lgraph,"block35_5_ac","conv2d_43");
lgraph = connectLayers(lgraph,"block35_5_ac","conv2d_44");
lgraph = connectLayers(lgraph,"block35_5_ac","conv2d_46");
lgraph = connectLayers(lgraph,"block35_5_ac","block35_6/in1");
lgraph = connectLayers(lgraph,"activation_43","block35_6_mixed/in1");
lgraph = connectLayers(lgraph,"activation_45","block35_6_mixed/in2");
lgraph = connectLayers(lgraph,"activation_48","block35_6_mixed/in3");
lgraph = connectLayers(lgraph,"block35_6_scale","block35_6/in2");
lgraph = connectLayers(lgraph,"block35_6_ac","conv2d_49");
lgraph = connectLayers(lgraph,"block35_6_ac","conv2d_52");
lgraph = connectLayers(lgraph,"block35_6_ac","conv2d_50");
lgraph = connectLayers(lgraph,"block35_6_ac","block35_7/in1");
lgraph = connectLayers(lgraph,"activation_49","block35_7_mixed/in1");
lgraph = connectLayers(lgraph,"activation_51","block35_7_mixed/in2");
lgraph = connectLayers(lgraph,"activation_54","block35_7_mixed/in3");
lgraph = connectLayers(lgraph,"block35_7_scale","block35_7/in2");
lgraph = connectLayers(lgraph,"block35_7_ac","conv2d_58");
lgraph = connectLayers(lgraph,"block35_7_ac","conv2d_56");
lgraph = connectLayers(lgraph,"block35_7_ac","conv2d_55");
lgraph = connectLayers(lgraph,"block35_7_ac","block35_8/in1");
lgraph = connectLayers(lgraph,"activation_57","block35_8_mixed/in2");
lgraph = connectLayers(lgraph,"activation_55","block35_8_mixed/in1");
lgraph = connectLayers(lgraph,"activation_60","block35_8_mixed/in3");
lgraph = connectLayers(lgraph,"block35_8_scale","block35_8/in2");
lgraph = connectLayers(lgraph,"block35_8_ac","conv2d_61");
lgraph = connectLayers(lgraph,"block35_8_ac","conv2d_64");
lgraph = connectLayers(lgraph,"block35_8_ac","conv2d_62");
lgraph = connectLayers(lgraph,"block35_8_ac","block35_9/in1");
lgraph = connectLayers(lgraph,"activation_66","block35_9_mixed/in3");
lgraph = connectLayers(lgraph,"activation_63","block35_9_mixed/in2");
lgraph = connectLayers(lgraph,"activation_61","block35_9_mixed/in1");
lgraph = connectLayers(lgraph,"block35_9_scale","block35_9/in2");
lgraph = connectLayers(lgraph,"block35_9_ac","conv2d_70");
lgraph = connectLayers(lgraph,"block35_9_ac","conv2d_68");
lgraph = connectLayers(lgraph,"block35_9_ac","conv2d_67");
lgraph = connectLayers(lgraph,"block35_9_ac","block35_10/in1");
lgraph = connectLayers(lgraph,"activation_72","block35_10_mixed/in3");
lgraph = connectLayers(lgraph,"activation_69","block35_10_mixed/in2");
lgraph = connectLayers(lgraph,"activation_67","block35_10_mixed/in1");
lgraph = connectLayers(lgraph,"block35_10_scale","block35_10/in2");
lgraph = connectLayers(lgraph,"block35_10_ac","max_pooling2d_3");
lgraph = connectLayers(lgraph,"block35_10_ac","conv2d_74");
lgraph = connectLayers(lgraph,"block35_10_ac","conv2d_73");
lgraph = connectLayers(lgraph,"max_pooling2d_3","mixed_6a/in3");
lgraph = connectLayers(lgraph,"activation_76","mixed_6a/in2");
lgraph = connectLayers(lgraph,"activation_73","mixed_6a/in1");
lgraph = connectLayers(lgraph,"mixed_6a","conv2d_77");
lgraph = connectLayers(lgraph,"mixed_6a","conv2d_78");
lgraph = connectLayers(lgraph,"mixed_6a","block17_1/in1");
lgraph = connectLayers(lgraph,"activation_77","block17_1_mixed/in1");
lgraph = connectLayers(lgraph,"activation_80","block17_1_mixed/in2");
lgraph = connectLayers(lgraph,"block17_1_scale","block17_1/in2");
lgraph = connectLayers(lgraph,"block17_1_ac","conv2d_81");
lgraph = connectLayers(lgraph,"block17_1_ac","conv2d_82");
lgraph = connectLayers(lgraph,"block17_1_ac","block17_2/in1");
lgraph = connectLayers(lgraph,"activation_81","block17_2_mixed/in1");
lgraph = connectLayers(lgraph,"activation_84","block17_2_mixed/in2");
lgraph = connectLayers(lgraph,"block17_2_scale","block17_2/in2");
lgraph = connectLayers(lgraph,"block17_2_ac","conv2d_86");
lgraph = connectLayers(lgraph,"block17_2_ac","conv2d_85");
lgraph = connectLayers(lgraph,"block17_2_ac","block17_3/in1");
lgraph = connectLayers(lgraph,"activation_88","block17_3_mixed/in2");
lgraph = connectLayers(lgraph,"activation_85","block17_3_mixed/in1");
lgraph = connectLayers(lgraph,"block17_3_scale","block17_3/in2");
lgraph = connectLayers(lgraph,"block17_3_ac","conv2d_89");
lgraph = connectLayers(lgraph,"block17_3_ac","conv2d_90");
lgraph = connectLayers(lgraph,"block17_3_ac","block17_4/in1");
lgraph = connectLayers(lgraph,"activation_92","block17_4_mixed/in2");
lgraph = connectLayers(lgraph,"activation_89","block17_4_mixed/in1");
lgraph = connectLayers(lgraph,"block17_4_scale","block17_4/in2");
lgraph = connectLayers(lgraph,"block17_4_ac","conv2d_94");
lgraph = connectLayers(lgraph,"block17_4_ac","conv2d_93");
lgraph = connectLayers(lgraph,"block17_4_ac","block17_5/in1");
lgraph = connectLayers(lgraph,"activation_93","block17_5_mixed/in1");
lgraph = connectLayers(lgraph,"activation_96","block17_5_mixed/in2");
lgraph = connectLayers(lgraph,"block17_5_scale","block17_5/in2");
lgraph = connectLayers(lgraph,"block17_5_ac","conv2d_97");
lgraph = connectLayers(lgraph,"block17_5_ac","conv2d_98");
lgraph = connectLayers(lgraph,"block17_5_ac","block17_6/in1");
lgraph = connectLayers(lgraph,"activation_100","block17_6_mixed/in2");
lgraph = connectLayers(lgraph,"activation_97","block17_6_mixed/in1");
lgraph = connectLayers(lgraph,"block17_6_scale","block17_6/in2");
lgraph = connectLayers(lgraph,"block17_6_ac","conv2d_101");
lgraph = connectLayers(lgraph,"block17_6_ac","conv2d_102");
lgraph = connectLayers(lgraph,"block17_6_ac","block17_7/in1");
lgraph = connectLayers(lgraph,"activation_101","block17_7_mixed/in1");
lgraph = connectLayers(lgraph,"activation_104","block17_7_mixed/in2");
lgraph = connectLayers(lgraph,"block17_7_scale","block17_7/in2");
lgraph = connectLayers(lgraph,"block17_7_ac","conv2d_105");
lgraph = connectLayers(lgraph,"block17_7_ac","conv2d_106");
lgraph = connectLayers(lgraph,"block17_7_ac","block17_8/in1");
lgraph = connectLayers(lgraph,"activation_105","block17_8_mixed/in1");
lgraph = connectLayers(lgraph,"activation_108","block17_8_mixed/in2");
lgraph = connectLayers(lgraph,"block17_8_scale","block17_8/in2");
lgraph = connectLayers(lgraph,"block17_8_ac","conv2d_109");
lgraph = connectLayers(lgraph,"block17_8_ac","conv2d_110");
lgraph = connectLayers(lgraph,"block17_8_ac","block17_9/in1");
lgraph = connectLayers(lgraph,"activation_109","block17_9_mixed/in1");
lgraph = connectLayers(lgraph,"activation_112","block17_9_mixed/in2");
lgraph = connectLayers(lgraph,"block17_9_scale","block17_9/in2");
lgraph = connectLayers(lgraph,"block17_9_ac","conv2d_113");
lgraph = connectLayers(lgraph,"block17_9_ac","conv2d_114");
lgraph = connectLayers(lgraph,"block17_9_ac","block17_10/in1");
lgraph = connectLayers(lgraph,"activation_116","block17_10_mixed/in2");
lgraph = connectLayers(lgraph,"activation_113","block17_10_mixed/in1");
lgraph = connectLayers(lgraph,"block17_10_scale","block17_10/in2");
lgraph = connectLayers(lgraph,"block17_10_ac","conv2d_118");
lgraph = connectLayers(lgraph,"block17_10_ac","conv2d_117");
lgraph = connectLayers(lgraph,"block17_10_ac","block17_11/in1");
lgraph = connectLayers(lgraph,"activation_120","block17_11_mixed/in2");
lgraph = connectLayers(lgraph,"activation_117","block17_11_mixed/in1");
lgraph = connectLayers(lgraph,"block17_11_scale","block17_11/in2");
lgraph = connectLayers(lgraph,"block17_11_ac","conv2d_122");
lgraph = connectLayers(lgraph,"block17_11_ac","conv2d_121");
lgraph = connectLayers(lgraph,"block17_11_ac","block17_12/in1");
lgraph = connectLayers(lgraph,"activation_121","block17_12_mixed/in1");
lgraph = connectLayers(lgraph,"activation_124","block17_12_mixed/in2");
lgraph = connectLayers(lgraph,"block17_12_scale","block17_12/in2");
lgraph = connectLayers(lgraph,"block17_12_ac","conv2d_126");
lgraph = connectLayers(lgraph,"block17_12_ac","conv2d_125");
lgraph = connectLayers(lgraph,"block17_12_ac","block17_13/in1");
lgraph = connectLayers(lgraph,"activation_125","block17_13_mixed/in1");
lgraph = connectLayers(lgraph,"activation_128","block17_13_mixed/in2");
lgraph = connectLayers(lgraph,"block17_13_scale","block17_13/in2");
lgraph = connectLayers(lgraph,"block17_13_ac","conv2d_129");
lgraph = connectLayers(lgraph,"block17_13_ac","conv2d_130");
lgraph = connectLayers(lgraph,"block17_13_ac","block17_14/in1");
lgraph = connectLayers(lgraph,"activation_129","block17_14_mixed/in1");
lgraph = connectLayers(lgraph,"activation_132","block17_14_mixed/in2");
lgraph = connectLayers(lgraph,"block17_14_scale","block17_14/in2");
lgraph = connectLayers(lgraph,"block17_14_ac","conv2d_134");
lgraph = connectLayers(lgraph,"block17_14_ac","conv2d_133");
lgraph = connectLayers(lgraph,"block17_14_ac","block17_15/in1");
lgraph = connectLayers(lgraph,"activation_133","block17_15_mixed/in1");
lgraph = connectLayers(lgraph,"activation_136","block17_15_mixed/in2");
lgraph = connectLayers(lgraph,"block17_15_scale","block17_15/in2");
lgraph = connectLayers(lgraph,"block17_15_ac","conv2d_138");
lgraph = connectLayers(lgraph,"block17_15_ac","conv2d_137");
lgraph = connectLayers(lgraph,"block17_15_ac","block17_16/in1");
lgraph = connectLayers(lgraph,"activation_137","block17_16_mixed/in1");
lgraph = connectLayers(lgraph,"activation_140","block17_16_mixed/in2");
lgraph = connectLayers(lgraph,"block17_16_scale","block17_16/in2");
lgraph = connectLayers(lgraph,"block17_16_ac","conv2d_141");
lgraph = connectLayers(lgraph,"block17_16_ac","conv2d_142");
lgraph = connectLayers(lgraph,"block17_16_ac","block17_17/in1");
lgraph = connectLayers(lgraph,"activation_141","block17_17_mixed/in1");
lgraph = connectLayers(lgraph,"activation_144","block17_17_mixed/in2");
lgraph = connectLayers(lgraph,"block17_17_scale","block17_17/in2");
lgraph = connectLayers(lgraph,"block17_17_ac","conv2d_146");
lgraph = connectLayers(lgraph,"block17_17_ac","conv2d_145");
lgraph = connectLayers(lgraph,"block17_17_ac","block17_18/in1");
lgraph = connectLayers(lgraph,"activation_145","block17_18_mixed/in1");
lgraph = connectLayers(lgraph,"activation_148","block17_18_mixed/in2");
lgraph = connectLayers(lgraph,"block17_18_scale","block17_18/in2");
lgraph = connectLayers(lgraph,"block17_18_ac","conv2d_149");
lgraph = connectLayers(lgraph,"block17_18_ac","conv2d_150");
lgraph = connectLayers(lgraph,"block17_18_ac","block17_19/in1");
lgraph = connectLayers(lgraph,"activation_149","block17_19_mixed/in1");
lgraph = connectLayers(lgraph,"activation_152","block17_19_mixed/in2");
lgraph = connectLayers(lgraph,"block17_19_scale","block17_19/in2");
lgraph = connectLayers(lgraph,"block17_19_ac","conv2d_154");
lgraph = connectLayers(lgraph,"block17_19_ac","conv2d_153");
lgraph = connectLayers(lgraph,"block17_19_ac","block17_20/in1");
lgraph = connectLayers(lgraph,"activation_156","block17_20_mixed/in2");
lgraph = connectLayers(lgraph,"activation_153","block17_20_mixed/in1");
lgraph = connectLayers(lgraph,"block17_20_scale","block17_20/in2");
lgraph = connectLayers(lgraph,"block17_20_ac","conv2d_159");
lgraph = connectLayers(lgraph,"block17_20_ac","conv2d_157");
lgraph = connectLayers(lgraph,"block17_20_ac","conv2d_161");
lgraph = connectLayers(lgraph,"block17_20_ac","max_pooling2d_4");
lgraph = connectLayers(lgraph,"activation_160","mixed_7a/in2");
lgraph = connectLayers(lgraph,"activation_158","mixed_7a/in1");
lgraph = connectLayers(lgraph,"max_pooling2d_4","mixed_7a/in4");
lgraph = connectLayers(lgraph,"activation_163","mixed_7a/in3");
lgraph = connectLayers(lgraph,"mixed_7a","conv2d_164");
lgraph = connectLayers(lgraph,"mixed_7a","conv2d_165");
lgraph = connectLayers(lgraph,"mixed_7a","block8_1/in1");
lgraph = connectLayers(lgraph,"activation_167","block8_1_mixed/in2");
lgraph = connectLayers(lgraph,"activation_164","block8_1_mixed/in1");
lgraph = connectLayers(lgraph,"block8_1_scale","block8_1/in2");
lgraph = connectLayers(lgraph,"block8_1_ac","conv2d_169");
lgraph = connectLayers(lgraph,"block8_1_ac","conv2d_168");
lgraph = connectLayers(lgraph,"block8_1_ac","block8_2/in1");
lgraph = connectLayers(lgraph,"activation_168","block8_2_mixed/in1");
lgraph = connectLayers(lgraph,"activation_171","block8_2_mixed/in2");
lgraph = connectLayers(lgraph,"block8_2_scale","block8_2/in2");
lgraph = connectLayers(lgraph,"block8_2_ac","conv2d_172");
lgraph = connectLayers(lgraph,"block8_2_ac","conv2d_173");
lgraph = connectLayers(lgraph,"block8_2_ac","block8_3/in1");
lgraph = connectLayers(lgraph,"activation_172","block8_3_mixed/in1");
lgraph = connectLayers(lgraph,"activation_175","block8_3_mixed/in2");
lgraph = connectLayers(lgraph,"block8_3_scale","block8_3/in2");
lgraph = connectLayers(lgraph,"block8_3_ac","conv2d_176");
lgraph = connectLayers(lgraph,"block8_3_ac","conv2d_177");
lgraph = connectLayers(lgraph,"block8_3_ac","block8_4/in1");
lgraph = connectLayers(lgraph,"activation_176","block8_4_mixed/in1");
lgraph = connectLayers(lgraph,"activation_179","block8_4_mixed/in2");
lgraph = connectLayers(lgraph,"block8_4_scale","block8_4/in2");
lgraph = connectLayers(lgraph,"block8_4_ac","conv2d_180");
lgraph = connectLayers(lgraph,"block8_4_ac","conv2d_181");
lgraph = connectLayers(lgraph,"block8_4_ac","block8_5/in1");
lgraph = connectLayers(lgraph,"activation_180","block8_5_mixed/in1");
lgraph = connectLayers(lgraph,"activation_183","block8_5_mixed/in2");
lgraph = connectLayers(lgraph,"block8_5_scale","block8_5/in2");
lgraph = connectLayers(lgraph,"block8_5_ac","conv2d_185");
lgraph = connectLayers(lgraph,"block8_5_ac","conv2d_184");
lgraph = connectLayers(lgraph,"block8_5_ac","block8_6/in1");
lgraph = connectLayers(lgraph,"activation_187","block8_6_mixed/in2");
lgraph = connectLayers(lgraph,"activation_184","block8_6_mixed/in1");
lgraph = connectLayers(lgraph,"block8_6_scale","block8_6/in2");
lgraph = connectLayers(lgraph,"block8_6_ac","conv2d_189");
lgraph = connectLayers(lgraph,"block8_6_ac","conv2d_188");
lgraph = connectLayers(lgraph,"block8_6_ac","block8_7/in1");
lgraph = connectLayers(lgraph,"activation_188","block8_7_mixed/in1");
lgraph = connectLayers(lgraph,"activation_191","block8_7_mixed/in2");
lgraph = connectLayers(lgraph,"block8_7_scale","block8_7/in2");
lgraph = connectLayers(lgraph,"block8_7_ac","conv2d_192");
lgraph = connectLayers(lgraph,"block8_7_ac","conv2d_193");
lgraph = connectLayers(lgraph,"block8_7_ac","block8_8/in1");
lgraph = connectLayers(lgraph,"activation_192","block8_8_mixed/in1");
lgraph = connectLayers(lgraph,"activation_195","block8_8_mixed/in2");
lgraph = connectLayers(lgraph,"block8_8_scale","block8_8/in2");
lgraph = connectLayers(lgraph,"block8_8_ac","conv2d_197");
lgraph = connectLayers(lgraph,"block8_8_ac","conv2d_196");
lgraph = connectLayers(lgraph,"block8_8_ac","block8_9/in1");
lgraph = connectLayers(lgraph,"activation_196","block8_9_mixed/in1");
lgraph = connectLayers(lgraph,"activation_199","block8_9_mixed/in2");
lgraph = connectLayers(lgraph,"block8_9_scale","block8_9/in2");
lgraph = connectLayers(lgraph,"block8_9_ac","conv2d_200");
lgraph = connectLayers(lgraph,"block8_9_ac","conv2d_201");
lgraph = connectLayers(lgraph,"block8_9_ac","block8_10/in1");
lgraph = connectLayers(lgraph,"activation_200","block8_10_mixed/in1");
lgraph = connectLayers(lgraph,"activation_203","block8_10_mixed/in2");
lgraph = connectLayers(lgraph,"block8_10_scale","block8_10/in2");
