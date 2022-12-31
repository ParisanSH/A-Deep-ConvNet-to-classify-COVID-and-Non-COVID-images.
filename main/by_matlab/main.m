clear
clc

alex=alexnet;
layers = alex.Layers 


layers(23)=fullyConnectedLayer(2);
layers(25)=classificationLayer;

Itrain=imageDatastore('train','IncludeSubfolders',true,'LabelSource','foldernames');
Itest=imageDatastore('test','IncludeSubfolders',true,'LabelSource','foldernames');



option=trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',30,'Minibatchsize',64,'Plots','training-progress');
NET=trainNetwork(Itrain,layers,option);


TEST=classify(NET,Itest);
accuracy=mean(TEST==Itest.Labels)





A=Itest.Labels;

Lcov=A(1:252);
Lnon=A(253:end);
Tcov=TEST(1:252);
Tnon=TEST(253:end);

TP=sum(Tcov==Lcov)
FN=252-TP
TN=sum(Tnon==Lnon)
FP=230-TN

F1=TP/(TP+0.5*(FP+FN))
%F1=2*(precision*recall)/(precision+recall)

precision=TP/(TP+FP)
recall=TP/(TP+FN)





sensitivity=TP/(TP+FN); %recall
specificity=TN/(TN+FP);
%ROC=plot(sensitivity,specificity)
%AUC=trapz(sensitivity,specificity)




