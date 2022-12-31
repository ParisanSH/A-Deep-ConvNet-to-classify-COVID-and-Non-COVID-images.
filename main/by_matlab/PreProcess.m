
clc
close all
clear all
imagefiles = dir('E:\mahdi\Personal\OmureKlasi\Phd\Dorus\DeepLearning\projects\HW4\COVID,Non COVID-CT Images\COVID,Non COVID-CT Images\test\COVID\*.png');
nfiles = length(imagefiles);    % Number of files found
for ImNum=1:nfiles
    
    cd('E:\mahdi\Personal\OmureKlasi\Phd\Dorus\DeepLearning\projects\HW4\COVID,Non COVID-CT Images\COVID,Non COVID-CT Images\test\COVID')
    baseFileNameIn = imagefiles(ImNum).name;

   input = imread(baseFileNameIn);
   i1 = rgb2gray(input);
   i2= imresize(i1,[28 28]);
   i2=mat2gray(i2);
   file_name=sprintf('%d.jpeg',ImNum);
   cd('E:\mahdi\Personal\OmureKlasi\Phd\Dorus\DeepLearning\projects\HW4\code\COVID_mod_Test')
    imwrite(i2, file_name,'quality',100);
    
end