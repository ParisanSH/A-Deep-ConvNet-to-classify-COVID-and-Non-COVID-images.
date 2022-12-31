clear
clc

train1=imageDatastore('C:\Users\Nik\Desktop\NNDL_Pr4\COVID,Non COVID-CT Images\train\COVID');
train2=imageDatastore('C:\Users\Nik\Desktop\NNDL_Pr4\COVID,Non COVID-CT Images\train\Non-COVID');
test1=imageDatastore('C:\Users\Nik\Desktop\NNDL_Pr4\COVID,Non COVID-CT Images\test\COVID');
test2=imageDatastore('C:\Users\Nik\Desktop\NNDL_Pr4\COVID,Non COVID-CT Images\test\Non-COVID');

A=test2;

for ii=1:252
    AA=readimage(A,ii);
    AA=imresize(AA,[227,227]);
    
    imwrite(AA,sprintf('%d.jpg',ii))
    
end






































