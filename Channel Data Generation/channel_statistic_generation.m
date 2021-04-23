%%  3GPP TR 38.901 release 15
% generate candidates for channel statistics

clear
Nc=64;
L=3; % number of path
Nt=32;
Nr=16;
num_para=6;
num_sta=50;
channel_statistic=zeros(num_para,L,num_sta);

for m=1:num_sta
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
cdl.MaximumDopplerShift = 0;
cdl.ChannelFiltering=false;    
cdl.DelayProfile='Custom';        
lgDSmean=-0.24*log10(1+ fc) - 6.83;   %UMi - Street Canyon  NLOS
lgDSstanvar=0.16*log10(1+ fc) + 0.28;
DS=10^(normrnd(lgDSmean,lgDSstanvar));  % DS
r_tau=2.1;
tau=-r_tau*DS*log(rand(1,L));
cdl.PathDelays=sort(tau-min(tau));   % path delays 
save_PD=cdl.PathDelays;
Power=exp(-(r_tau-1)/(r_tau*DS)*cdl.PathDelays).*10.^(-normrnd(0,3,1,L)/10);
P=Power/sum(Power);
cdl.AveragePathGains=10*log10(P);  % average path gains
save_P=cdl.AveragePathGains;
lgASAmean=-0.08*log10(1+ fc) + 1.81;   %UMi - Street Canyon  NLOS
lgASAstanvar=0.05*log10(1+ fc) + 0.3;
ASA=10^(normrnd(lgASAmean,lgASAstanvar));  % ASA
Cphi=0.779; % N=4
Phi=2*ASA/1.4*sqrt(-log(P/max(P)))/Cphi;
cdl.AnglesAoA=(2*randi([0,1],1,L)-1).*Phi+normrnd(0,ASA/7,1,L)+philosAoA;  % AoA
save_AoA=cdl.AnglesAoA;
lgASDmean=-0.23*log10(1+ fc) + 1.53;   %UMi - Street Canyon  NLOS
lgASDstanvar=0.11*log10(1+ fc) + 0.33;
ASD=10^(normrnd(lgASDmean,lgASDstanvar));  % ASD
Cphi=0.779; % N=4
Phi=2*ASD/1.4*sqrt(-log(P/max(P)))/Cphi;
cdl.AnglesAoD=(2*randi([0,1],1,L)-1).*Phi+normrnd(0,ASD/7,1,L)+philosAoD;  % AoD
save_AoD=cdl.AnglesAoD;
lgZSAmean=-0.04*log10(1+ fc) + 0.92;   %UMi - Street Canyon  NLOS
lgZSAstanvar=-0.07*log10(1+ fc) + 0.41;
ZSA=10^(normrnd(lgZSAmean,lgZSAstanvar));  % ZSA
Ctheta=0.889; % N=8
Theta=-ZSA*log(P/max(P))/Ctheta;
cdl.AnglesZoA=(2*randi([0,1],1,L)-1).*Theta+normrnd(0,ZSA/7,1,L)+thetalosZoA;  % ZoA
save_ZoA=cdl.AnglesZoA;
d2D=50;
hUT=0;
hBS=10;
lgZSDmean=max(-0.5, -3.1*(d2D/1000)+ 0.01*max(hUT-hBS,0) +0.2);   %UMi - Street Canyon  NLOS
lgZSDstanvar=0.35;
ZSD=10^(normrnd(lgZSDmean,lgZSDstanvar));  % ZSD
Ctheta=0.889; % N=8
Theta=-ZSD*log(P/max(P))/Ctheta;
cdl.AnglesZoD=(2*randi([0,1],1,L)-1).*Theta+normrnd(0,ZSD/7,1,L)-10^(-1.5*log10(max(10, d2D))+3.3)+thetalosZoD;  % ZoD
save_ZoD=cdl.AnglesZoD;
sta_mtx=[save_PD;save_P;save_AoA;save_AoD;save_ZoA;save_ZoD];
channel_statistic(:,:,m)=sta_mtx;
end
save ch_sta_mtx channel_statistic