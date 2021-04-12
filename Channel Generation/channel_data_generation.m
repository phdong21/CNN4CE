%%  3GPP TR 38.901 release 15
% generate 3D MIMO channel matrix, random pathdelays, pathgains, AoA, AoD,
% ZoA, ZoD
clear

Nc=64;
N=3; % number of path
Nt=32;
Nr=16;
L=3;
samplingrate=1e8;
sample_num=1000;
num_fre=2;
num_sta=100;
num_ffading=100;


load ch_sta_mtx2
for m=11:20
ChannelData_fre=zeros(Nt,Nr,num_fre,sample_num);
sta_mtx=channel_statistic(:,:,m);
LOSangle=60;
philosAoA=LOSangle;
philosAoD=LOSangle;
thetalosZoA=LOSangle;
thetalosZoD=LOSangle;
cdl = nr5gCDLChannel;
cdl.CarrierFrequency=28e9;
cdl.TransmitAntennaArray.Size = [Nt 1 1 1 1];
%cdl.TransmitAntennaArray.ElementSpacing = [0.5 0 0 0];
cdl.ReceiveAntennaArray.Size = [Nr 1 1 1 1];
%cdl.ReceiveAntennaArray.ElementSpacing = [0.5 0 0 0];
fc=cdl.CarrierFrequency/1e9;
cdl.MaximumDopplerShift = 0;
%cdl.SampleRate=30.72e6;
cdl.ChannelFiltering=false;    
cdl.DelayProfile='Custom'; 
cdl.PathDelays=sta_mtx(1,:);
cdl.AveragePathGains=sta_mtx(2,:);
cdl.AnglesAoA=sta_mtx(3,:);
cdl.AnglesAoD=sta_mtx(4,:);
cdl.AnglesZoA=sta_mtx(5,:);
cdl.AnglesZoD=sta_mtx(6,:);
for n=1:sample_num
    cdl.Seed = 2000+n;
    [pathgains,sampletimes]=step(cdl);
    pathgains=sqrt(Nr)*pathgains;
    for nt=1:Nt
        for nr=1:Nr
            pathpower=pathgains(:,:,nt,nr);
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
            %pathgains_time(:,:,nt,nr)=h_ntnr;   %  time domain channel matrix
            %fre=randi(Nc-num_fre+1);
            fre=32;
            pathgains_fre(:,nt,nr)=h_fre(fre:fre+num_fre-1);   %  frequency domain channel matrix
        end
    end
    for j=1:num_fre
        pathgains_fre2(:,:,j)=pathgains_fre(j,:,:);
    end
    ChannelData_fre(:,:,:,n)=pathgains_fre2;
    release(cdl);
end
%save channel_matrix_timedomain2 ChannelData_time
save(['channel_2bands_1by1000_sta',num2str(m)],'ChannelData_fre')
%save channel_2band_MMSE2 ChannelData_fre
%save channel_2bands_2000_MMSE_modify_mismatch2 ChannelData_fre
end