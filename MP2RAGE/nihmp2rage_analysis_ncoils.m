
% mp2rage image and t1 map generation
% ok for multiple receivers with no acceleration
% linear encoding selected in the method

%readnum=256;
%ncoils=2;
%fidnum = ceil(readnum*ncoils/128.0)*128.0;
%phasenum=256;
%slicenum=128;
trainnum=2;
%nseg = 4;

% pick experiment number for directory name
directoryname = uigetdir('/home/doddst/data/1322587_Rat-2.U41', 'Pick a Directory');

nseg = readpvpar('##$SegmNumber=', directoryname);
segduration = readpvpar('##$SegmDuration=', directoryname)/1000.0;
ncoils = readpvpar('##$PVM_EncNReceivers=', directoryname);
ti1=readpvpar('##$PVM_InversionTime=', directoryname)/1000.0;
ti2 = readpvpar('##$TI2=', directoryname)/1000.0;
alphadeg = readpvpar('##$PVM_ExcPulseAngle=', directoryname);
alpha2deg = readpvpar('##$MP2RAGE_2ndExcPulseAngle=', directoryname);
slicethickness = readpvpar('##$PVM_SliceThick=',directoryname);
slicenum = readpvpar('##$PVM_SPackArrNSlices=',directoryname);
phaseoffset = readpvpar('##$PVM_SPackArrPhase1Offset=',directoryname);
sliceoffset = readpvpar('##$PVM_SPackArrSliceOffset=', directoryname);
mp2ragetr = readpvpar('##$SegmRepTime=', directoryname)/1000.0;
tr = readpvpar('##$EchoRepTime=', directoryname)/1000.0;

slicefov = slicenum*slicethickness;

fmethod = fopen(strcat(directoryname,'/method'));
str1 = '';
while (strncmp(str1,'##$PVM_Fov=',11) ~= 1),
      str1 = fgets(fmethod);
end
str1 = fgetl(fmethod);
str2 = '';
str2 = cat(2,str2,strsplit(strtrim(str1)));
readfov = str2double(str2(1));
phasefov = str2double(str2(2));

str1 = '';
while (strncmp(str1,'##$PVM_Matrix=',14) ~= 1),
      str1 = fgets(fmethod);
end
str1 = fgetl(fmethod);
str2 = '';
str2 = cat(2,str2,strsplit(strtrim(str1)));
readnum = str2double(str2(1));
phasenum = str2double(str2(2));
fidnum = ceil(readnum*ncoils/128.0)*128.0;

fclose(fmethod);

phasenum = 255;

str1 = '';
fmethod = fopen(strcat(directoryname,'/method'));

while (strncmp(str1,'##$PVM_EncSteps1=',17) ~= 1),
      str1 = fgets(fmethod);
end
str2= '';

while (size(str2,2) ~= phasenum),    
      str1 = fgetl(fmethod);
      str2 = cat(2,str2,strsplit(strtrim(str1)));
end
phaseorder = str2double(str2);
fclose(fmethod);

a=zeros(1,2*fidnum*phasenum*slicenum*trainnum);
d=zeros(readnum,phasenum,slicenum,1);

    file1 = fopen(strcat(directoryname,'/fid'));
    a = fread(file1,size(a),'int32');
    fclose(file1);
    
    a=reshape(a,2,fidnum*phasenum*slicenum*trainnum);
    a = complex(a(1,:),a(2,:));
    a=reshape(a,fidnum,phasenum/nseg,trainnum,nseg,slicenum);
    a1 =a(1:readnum*ncoils,:,:);
    a=reshape(a1,readnum,ncoils,phasenum/nseg,trainnum,nseg,slicenum);
    %rescale coil 2 if necessary
    %a(:,2,:,:,:,:) = a(:,2,:,:,:,:);
    a1=1;
    atrain1 = zeros(readnum,ncoils,phasenum,slicenum);
    atrain2 = atrain1;
    % split up images and reorder phase encodes
    for ii = 1:nseg
        for jj = 1:phasenum/nseg 
            atrain1(:,:,phaseorder(1,jj+(ii-1)*phasenum/nseg)+floor(phasenum/2)+1,:) = a(:,:,jj,1,ii,:);
        end
    end
    for ii = 1:nseg
        for jj = 1:phasenum/nseg 
            atrain2(:,:,phaseorder(1,jj+(ii-1)*phasenum/nseg)+floor(phasenum/2)+1,:) = a(:,:,jj,2,ii,:);
        end
    end 

    atrain1fft = fftshift(fft(fftshift(atrain1(1:readnum,:,:,:)),[],1),1); 
    atrain1fft = fftshift(fft(atrain1fft(1:readnum,:,:,:),[],3),3);
    atrain1fft = fftshift(fft(atrain1fft(1:readnum,:,:,:),[],4),4);
    atrain2fft = fftshift(fft(fftshift(atrain2(1:readnum,:,:,:)),[],1),1); 
    atrain2fft = fftshift(fft(atrain2fft(1:readnum,:,:,:),[],3),3);
    atrain2fft = fftshift(fft(atrain2fft(1:readnum,:,:,:),[],4),4);
    
    %combine for mprage
    maxsignal1 = max(max(max(max(abs(atrain1fft)))));
    maxsignal2 = max(max(max(max(abs(atrain2fft)))));
  
    
    mp2rage = (real((conj(atrain1fft).*atrain2fft))./((abs(atrain1fft).*abs(atrain1fft) + abs(atrain2fft).*abs(atrain2fft))));

    %mp2ragethresh = mp2rage .* or((abs(atrain1fft) > 0.05*maxsignal1), (abs(atrain2fft) > 0.05*maxsignal2));
    
    %mp2ragesos = zeros(readnum,phasenum,slicenum);
    %mp2ragewa = zeros(readnum,phasenum,slicenum);
    %atrain2sos =  zeros(readnum,phasenum,slicenum);
    
    
    % sum of squares calculation, just using sum at the moment
    for ii=1:readnum
        for jj=1:phasenum
            for kk=1:slicenum
     %           mp2ragesos(ii,jj,kk) = sqrt(squeeze(mp2rage(ii,:,jj,kk)) * transpose(conj(squeeze(mp2rage(ii,:,jj,kk)))));
     %           atrain2sos(ii,jj,kk) = sqrt(squeeze(atrain2fft(ii,:,jj,kk)) * transpose(conj(squeeze(atrain2fft(ii,:,jj,kk)))));
     %           mp2ragewa(ii,jj,kk) = (mp2rage(ii,1,jj,kk)*atrain2fft(ii,1,jj,kk) + mp2rage(ii,2,jj,kk)*atrain2fft(ii,2,jj,kk))/atrain2sos(ii,jj,kk);
            end
        end
    end

    
    atrain2sum = squeeze(abs(atrain2fft(:,1,:,:)) + abs(atrain2fft(:,2,:,:)));
    
    %atrain2sum = atrain2sos;
    
    %threshold mask    
%    atrain2sumfilt = medfilt1(atrain2sum,3,[],1);
%    atrain2sumfilt = medfilt1(atrain2sumfilt,3,[],2);
%    atrain2sumfilt = medfilt1(atrain2sumfilt,3,[],3);
%    maxsum2 = max(max(max(atrain2sumfilt)));
    
    % 5 perscent threshold on second image
%    mp2ragemask = abs(atrain2sumfilt) > 0.05*maxsum2;
    
    mp2ragewa = squeeze((mp2rage(:,1,:,:).*abs(atrain2fft(:,1,:,:)) +  mp2rage(:,2,:,:).*abs(atrain2fft(:,2,:,:))))./atrain2sum;
    
  %  mp2ragewamask = (mp2ragewa+0.5).*mp2ragemask - 0.5;
    
    %t1map
        
    t1map = t1calc(mp2ragewa, alphadeg, alpha2deg, segduration, phasenum, ti1, ti2, nseg, tr, mp2ragetr);
    
    
%      t1map = interp1(mp2ragecalc,t1array,mp2ragewa);

     t1mapwmask = t1map.*mp2ragemask;
     
     fid = fopen(strcat(directoryname,'/t1map'), 'wb');
       % write out data in ms
    fwrite(fid,t1mapwmask*1000.0,'float');
    fclose(fid);

% output to afni
     cd(directoryname);
     
     %orientation is correct for coronal 3d slab from Bruker, readout head
     % to foot (at least I think so)
     afnicmd = strcat(['to3d -anat -prefix r1 ',' -xFOV ', num2str(readfov/2.0),'S-I -yFOV ',num2str(phasefov/2.0),'L-R -zFOV ',num2str(slicefov/2.0),'A-P 3Df:0:0:',int2str(readnum),':',int2str(phasenum),':',int2str(slicenum),':t1map'])
     status = system(afnicmd);

     