%%  3GPP TR 38.901 release 15
% generate 3D MIMO channel matrix, random pathdelays, pathgains, AoA, AoD,
% ZoA, ZoD
clear

Nc=64;
L=3; % number of path
Nt=32;
Nr=16;
samplingrate=1e8;
num_fre=2;
num_sta=50;
num_ffading=200;
time_steps=2;
sample_num=num_sta*num_ffading;

load ch_sta_mtx % channel statistics for training channel data
% load ch_sta_mtx2  % different channel statistics from training stage
for l=1:10   
fprintf('l=%d\n',l);
ChannelData_fre=zeros(Nt,time_steps*Nr,num_fre,sample_num);
for m=1:num_sta
sta_mtx=channel_statistic(:,:,m);
LOSangle=60;
philosAoA=LOSangle;
philosAoD=LOSangle;
thetalosZoA=LOSangle;
thetalosZoD=LOSangle;
cdl = nr5gCDLChannel;
cdl.CarrierFrequency=28e9;
cdl.TransmitAntennaArray.Size = [Nt 1 1 1 1];
cdl.ReceiveAntennaArray.Size = [Nr 1 1 1 1];
fc=cdl.CarrierFrequency/1e9;
cdl.MaximumDopplerShift = 2.8e3;
cdl.SampleDensity=0.5;
cdl.ChannelFiltering=false;    
cdl.DelayProfile='Custom'; 
cdl.PathDelays=sta_mtx(1,:);
cdl.AveragePathGains=sta_mtx(2,:);
cdl.AnglesAoA=sta_mtx(3,:);
cdl.AnglesAoD=sta_mtx(4,:);
cdl.AnglesZoA=sta_mtx(5,:);
cdl.AnglesZoD=sta_mtx(6,:);
for n=1:num_ffading
    cdl.Seed = (l-1)*num_ffading+n;
    [pathgains,sampletimes]=step(cdl);
    pathgains=sqrt(Nr)*pathgains;
    for t=1:time_steps        
        for nt=1:Nt            
            for nr=1:Nr            
                pathpower=pathgains(t,:,nt,nr);
                h=zeros(1,1024);
                I=floor(cdl.PathDelays*samplingrate)+1;
                I_uniq=unique(I);
                Power_sum=[];
                for i=1:length(I_uniq)
                    Power_sum(i)=sum(pathpower(I==I_uniq(i)));
                end
                h(I_uniq)=Power_sum;
                h_ntnr=h(1:Nc);
                h_fre=fft(h_ntnr);                
                %fre=randi(Nc-num_fre+1);
                fre=32;
                pathgains_fre(:,nt,(t-1)*Nr+nr)=h_fre(fre:fre+num_fre-1);   %  frequency domain channel matrix
            end            
        end
    end
    for j=1:num_fre
        pathgains_fre2(:,:,j)=pathgains_fre(j,:,:);
    end
    ChannelData_fre(:,:,:,(m-1)*num_ffading+n)=pathgains_fre2;
    release(cdl);
end
end
save(['channel_2fre2time_training_data',num2str(l)],'ChannelData_fre')
end