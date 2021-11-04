
lgraph = layerGraph();

tempLayers = [
    imageInputLayer([224 224 3],"Name","Input_gpu_0|data_0","Normalization","zscore")
    convolution2dLayer([3 3],24,"Name","node_1","Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","node_2")
    reluLayer("Name","node_3")
    maxPooling2dLayer([3 3],"Name","node_4","Padding",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = averagePooling2dLayer([3 3],"Name","node_15","Padding",[1 1 1 1],"Stride",[2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 1],28,4,"Name","node_5")
    batchNormalizationLayer("Name","node_6")
    reluLayer("Name","node_7")
    helperNnetShufflenetLayerChannelShufflingLayer("shuffle_8to10",4,"in","out")
    groupedConvolution2dLayer([3 3],1,112,"Name","node_11","Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","node_12")
    groupedConvolution2dLayer([1 1],28,4,"Name","node_13")
    batchNormalizationLayer("Name","node_14")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","node_16")
    reluLayer("Name","node_17")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 1],34,4,"Name","node_18")
    batchNormalizationLayer("Name","node_19")
    reluLayer("Name","node_20")
    helperNnetShufflenetLayerChannelShufflingLayer("shuffle_21to23",4,"in","out")
    groupedConvolution2dLayer([3 3],1,136,"Name","node_24","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","node_25")
    groupedConvolution2dLayer([1 1],34,4,"Name","node_26")
    batchNormalizationLayer("Name","node_27")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","node_28")
    reluLayer("Name","node_29")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 1],34,4,"Name","node_30")
    batchNormalizationLayer("Name","node_31")
    reluLayer("Name","node_32")
    helperNnetShufflenetLayerChannelShufflingLayer("shuffle_33to35",4,"in","out")
    groupedConvolution2dLayer([3 3],1,136,"Name","node_36","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","node_37")
    groupedConvolution2dLayer([1 1],34,4,"Name","node_38")
    batchNormalizationLayer("Name","node_39")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","node_40")
    reluLayer("Name","node_41")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 1],34,4,"Name","node_42")
    batchNormalizationLayer("Name","node_43")
    reluLayer("Name","node_44")
    helperNnetShufflenetLayerChannelShufflingLayer("shuffle_45to47",4,"in","out")
    groupedConvolution2dLayer([3 3],1,136,"Name","node_48","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","node_49")
    groupedConvolution2dLayer([1 1],34,4,"Name","node_50")
    batchNormalizationLayer("Name","node_51")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","node_52")
    reluLayer("Name","node_53")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 1],34,4,"Name","node_54")
    batchNormalizationLayer("Name","node_55")
    reluLayer("Name","node_56")
    helperNnetShufflenetLayerChannelShufflingLayer("shuffle_57to59",4,"in","out")
    groupedConvolution2dLayer([3 3],1,136,"Name","node_60","Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","node_61")
    groupedConvolution2dLayer([1 1],34,4,"Name","node_62")
    batchNormalizationLayer("Name","node_63")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = averagePooling2dLayer([3 3],"Name","node_64","Padding",[1 1 1 1],"Stride",[2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","node_65")
    reluLayer("Name","node_66")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 1],68,4,"Name","node_67")
    batchNormalizationLayer("Name","node_68")
    reluLayer("Name","node_69")
    helperNnetShufflenetLayerChannelShufflingLayer("shuffle_70to72",4,"in","out")
    groupedConvolution2dLayer([3 3],1,272,"Name","node_73","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","node_74")
    groupedConvolution2dLayer([1 1],68,4,"Name","node_75")
    batchNormalizationLayer("Name","node_76")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","node_77")
    reluLayer("Name","node_78")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 1],68,4,"Name","node_79")
    batchNormalizationLayer("Name","node_80")
    reluLayer("Name","node_81")
    helperNnetShufflenetLayerChannelShufflingLayer("shuffle_82to84",4,"in","out")
    groupedConvolution2dLayer([3 3],1,272,"Name","node_85","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","node_86")
    groupedConvolution2dLayer([1 1],68,4,"Name","node_87")
    batchNormalizationLayer("Name","node_88")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","node_89")
    reluLayer("Name","node_90")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 1],68,4,"Name","node_91")
    batchNormalizationLayer("Name","node_92")
    reluLayer("Name","node_93")
    helperNnetShufflenetLayerChannelShufflingLayer("shuffle_94to96",4,"in","out")
    groupedConvolution2dLayer([3 3],1,272,"Name","node_97","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","node_98")
    groupedConvolution2dLayer([1 1],68,4,"Name","node_99")
    batchNormalizationLayer("Name","node_100")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","node_101")
    reluLayer("Name","node_102")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 1],68,4,"Name","node_103")
    batchNormalizationLayer("Name","node_104")
    reluLayer("Name","node_105")
    helperNnetShufflenetLayerChannelShufflingLayer("shuffle_106to108",4,"in","out")
    groupedConvolution2dLayer([3 3],1,272,"Name","node_109","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","node_110")
    groupedConvolution2dLayer([1 1],68,4,"Name","node_111")
    batchNormalizationLayer("Name","node_112")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","node_113")
    reluLayer("Name","node_114")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 1],68,4,"Name","node_115")
    batchNormalizationLayer("Name","node_116")
    reluLayer("Name","node_117")
    helperNnetShufflenetLayerChannelShufflingLayer("shuffle_118to120",4,"in","out")
    groupedConvolution2dLayer([3 3],1,272,"Name","node_121","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","node_122")
    groupedConvolution2dLayer([1 1],68,4,"Name","node_123")
    batchNormalizationLayer("Name","node_124")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","node_125")
    reluLayer("Name","node_126")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 1],68,4,"Name","node_127")
    batchNormalizationLayer("Name","node_128")
    reluLayer("Name","node_129")
    helperNnetShufflenetLayerChannelShufflingLayer("shuffle_130to132",4,"in","out")
    groupedConvolution2dLayer([3 3],1,272,"Name","node_133","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","node_134")
    groupedConvolution2dLayer([1 1],68,4,"Name","node_135")
    batchNormalizationLayer("Name","node_136")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","node_137")
    reluLayer("Name","node_138")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 1],68,4,"Name","node_139")
    batchNormalizationLayer("Name","node_140")
    reluLayer("Name","node_141")
    helperNnetShufflenetLayerChannelShufflingLayer("shuffle_142to144",4,"in","out")
    groupedConvolution2dLayer([3 3],1,272,"Name","node_145","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","node_146")
    groupedConvolution2dLayer([1 1],68,4,"Name","node_147")
    batchNormalizationLayer("Name","node_148")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","node_149")
    reluLayer("Name","node_150")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = averagePooling2dLayer([3 3],"Name","node_161","Padding",[1 1 1 1],"Stride",[2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 1],68,4,"Name","node_151")
    batchNormalizationLayer("Name","node_152")
    reluLayer("Name","node_153")
    helperNnetShufflenetLayerChannelShufflingLayer("shuffle_154to156",4,"in","out")
    groupedConvolution2dLayer([3 3],1,272,"Name","node_157","Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","node_158")
    groupedConvolution2dLayer([1 1],68,4,"Name","node_159")
    batchNormalizationLayer("Name","node_160")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","node_162")
    reluLayer("Name","node_163")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 1],136,4,"Name","node_164")
    batchNormalizationLayer("Name","node_165")
    reluLayer("Name","node_166")
    helperNnetShufflenetLayerChannelShufflingLayer("shuffle_167to169",4,"in","out")
    groupedConvolution2dLayer([3 3],1,544,"Name","node_170","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","node_171")
    groupedConvolution2dLayer([1 1],136,4,"Name","node_172")
    batchNormalizationLayer("Name","node_173")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","node_174")
    reluLayer("Name","node_175")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 1],136,4,"Name","node_176")
    batchNormalizationLayer("Name","node_177")
    reluLayer("Name","node_178")
    helperNnetShufflenetLayerChannelShufflingLayer("shuffle_179to181",4,"in","out")
    groupedConvolution2dLayer([3 3],1,544,"Name","node_182","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","node_183")
    groupedConvolution2dLayer([1 1],136,4,"Name","node_184")
    batchNormalizationLayer("Name","node_185")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","node_186")
    reluLayer("Name","node_187")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 1],136,4,"Name","node_188")
    batchNormalizationLayer("Name","node_189")
    reluLayer("Name","node_190")
    helperNnetShufflenetLayerChannelShufflingLayer("shuffle_191to193",4,"in","out")
    groupedConvolution2dLayer([3 3],1,544,"Name","node_194","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","node_195")
    groupedConvolution2dLayer([1 1],136,4,"Name","node_196")
    batchNormalizationLayer("Name","node_197")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","node_198")
    reluLayer("Name","node_199")
    globalAveragePooling2dLayer("Name","node_200")
    fullyConnectedLayer(5,"Name","node_202")
    softmaxLayer("Name","node_203")
    classificationLayer("Name","ClassificationLayer_node_203")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;


lgraph = connectLayers(lgraph,"node_4","node_15");
lgraph = connectLayers(lgraph,"node_4","node_5");
lgraph = connectLayers(lgraph,"node_15","node_16/in2");
lgraph = connectLayers(lgraph,"node_14","node_16/in1");
lgraph = connectLayers(lgraph,"node_17","node_18");
lgraph = connectLayers(lgraph,"node_17","node_28/in2");
lgraph = connectLayers(lgraph,"node_27","node_28/in1");
lgraph = connectLayers(lgraph,"node_29","node_30");
lgraph = connectLayers(lgraph,"node_29","node_40/in2");
lgraph = connectLayers(lgraph,"node_39","node_40/in1");
lgraph = connectLayers(lgraph,"node_41","node_42");
lgraph = connectLayers(lgraph,"node_41","node_52/in2");
lgraph = connectLayers(lgraph,"node_51","node_52/in1");
lgraph = connectLayers(lgraph,"node_53","node_54");
lgraph = connectLayers(lgraph,"node_53","node_64");
lgraph = connectLayers(lgraph,"node_64","node_65/in2");
lgraph = connectLayers(lgraph,"node_63","node_65/in1");
lgraph = connectLayers(lgraph,"node_66","node_67");
lgraph = connectLayers(lgraph,"node_66","node_77/in2");
lgraph = connectLayers(lgraph,"node_76","node_77/in1");
lgraph = connectLayers(lgraph,"node_78","node_79");
lgraph = connectLayers(lgraph,"node_78","node_89/in2");
lgraph = connectLayers(lgraph,"node_88","node_89/in1");
lgraph = connectLayers(lgraph,"node_90","node_91");
lgraph = connectLayers(lgraph,"node_90","node_101/in2");
lgraph = connectLayers(lgraph,"node_100","node_101/in1");
lgraph = connectLayers(lgraph,"node_102","node_103");
lgraph = connectLayers(lgraph,"node_102","node_113/in2");
lgraph = connectLayers(lgraph,"node_112","node_113/in1");
lgraph = connectLayers(lgraph,"node_114","node_115");
lgraph = connectLayers(lgraph,"node_114","node_125/in2");
lgraph = connectLayers(lgraph,"node_124","node_125/in1");
lgraph = connectLayers(lgraph,"node_126","node_127");
lgraph = connectLayers(lgraph,"node_126","node_137/in2");
lgraph = connectLayers(lgraph,"node_136","node_137/in1");
lgraph = connectLayers(lgraph,"node_138","node_139");
lgraph = connectLayers(lgraph,"node_138","node_149/in2");
lgraph = connectLayers(lgraph,"node_148","node_149/in1");
lgraph = connectLayers(lgraph,"node_150","node_161");
lgraph = connectLayers(lgraph,"node_150","node_151");
lgraph = connectLayers(lgraph,"node_161","node_162/in2");
lgraph = connectLayers(lgraph,"node_160","node_162/in1");
lgraph = connectLayers(lgraph,"node_163","node_164");
lgraph = connectLayers(lgraph,"node_163","node_174/in2");
lgraph = connectLayers(lgraph,"node_173","node_174/in1");
lgraph = connectLayers(lgraph,"node_175","node_176");
lgraph = connectLayers(lgraph,"node_175","node_186/in2");
lgraph = connectLayers(lgraph,"node_185","node_186/in1");
lgraph = connectLayers(lgraph,"node_187","node_188");
lgraph = connectLayers(lgraph,"node_187","node_198/in2");
lgraph = connectLayers(lgraph,"node_197","node_198/in1");
