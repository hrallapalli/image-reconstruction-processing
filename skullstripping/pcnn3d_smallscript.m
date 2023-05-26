%%
clear all
close all
clc
%%
addpath('C:\Users\rallapallih2\Documents\GitHub\image-reconstruction-processing\skullstripping\NIfTI_20140122\')
addpath('C:\Users\rallapallih2\Documents\GitHub\image-reconstruction-processing\skullstripping\PCNN3D_matlab\')

%% PCNN3D auto brain extraction
% save as *_mask.nii.gz
% requires nifti toolbox
%
% 2017/07/27 ver1.0
% 2017/08/29 ver1.1 bug fix; add ZoomFactor; use mean image for 4D data
% 2017/09/12 ver1.2 save as 8-bit mask
% 2018/01/23 ver1.3 use with command line bash script
% Kai-Hsiang Chuang, QBI/UQ

%% init setup
basepath ='\\nindsdirfs2\shares\LFMI\FMM\Shared\rallapallih\ScanData'; % data path 
[file,path] = uigetfile(basepath);
datpath = strcat(path,file);
% BrSize=[150,600]; % brain size range for MOUSE (mm3).
BrSize=[1500,5000]; % brain size range for RAT (mm3)
StrucRadius=7; % use =3 for low resolution, use 5 or 7 for highres data
ZoomFactor=1; % resolution magnification factor

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% run PCNN
[nii] = load_untouch_nii(datpath);
mtx=size(nii.img);
if length(mtx)==4
    disp('Data is 4D, use the average image to generate mask')
    nii.img=mean(nii.img,4);
end
voxdim=nii.hdr.dime.pixdim(2:4);
[I_border, G_I, optG] = PCNN3D(single(nii.img), StrucRadius, voxdim, BrSize*ZoomFactor^3);
V=zeros(mtx);
for n=1:mtx(3)
    V(:,:,n)=I_border{optG-1}{n};
end
%%

V = bwmorph3(V,"fill");

for i = 1:mtx(3)
V(:,:,i) = bwconvhull(V(:,:,i));
end


%% save data
disp(['Saving mask at ',datpath(1:end-7),'_mask.nii.gz....'])
nii.img=V;
nii.hdr.dime.dim(1)=3; nii.hdr.dime.dim(5)=1;
nii.hdr.dime.datatype=2; nii.hdr.dime.bitpix=8; % save as unsigned char
p=strfind(datpath,'.nii');
save_untouch_nii(nii,[datpath(1:p-1),'_mask.nii.gz'])

disp('Done')

orthosliceViewer(V)
volshow(V)

% PCNN3D_run_v1_3.m
% Displaying PCNN3D_run_v1_3.m.