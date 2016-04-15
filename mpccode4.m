% This script is intended to test the applicability of using prewhitened error and u

clear
clc
% load('e.mat'); % For comparison, use same white noise sequence

z=tf('z',1); %transfer function model
G=0.05/(1-0.6*z^(-1))*z^(-4);
H=1;

model=ss([G H],'min');
model=setmpcsignals(model,'MV',1,'UD',2);

mpcobj=mpc(model,1,20,10); %2 is sampling time
% mpcobj.MV.Min=-10;
% mpcobj.MV.max=10;
% 
% mpcobj.OV.Min=-10;
% mpcobj.OV.max=10;

plant=1.2*0.05/(1-0.6*z^(-1))*z^(-4);

psim=ss([plant H],'min'); %minimal realisation
psim=setmpcsignals(psim,'MV',1,'UD',2);

%variying set point:
Tf=1000;
ref=ones(1000,1);
%ref=[1*ones(100,1); 5*ones(100,1);3*ones(100,1); 6*ones(100,1);8*ones(100,1); 0*ones(100,1);6*ones(100,1); 4*ones(100,1); 1*ones(100,1); 5*ones(100,1)];
% distmodel=1/(1-1*z^(-1)); % Origial Model
e=randn(Tf,1)*0.01;
distmodel=(1+0.5*z^(-1))/(1+0.3*z^(-1));
d=lsim(distmodel,e); % Integrated white noise
% d=e+[zeros(200,1);ones(Tf-200,1)]*0; % step disturbance plus white noise
% d=randn(Tf,1)*0.1+0.1*sin(2*pi*0.05*[1:Tf]')*1;
%d=d+1*sin(2*pi*0.02*[1:Tf]')*1;




options=mpcsimopt(mpcobj);
options.unmeas=d; %d is unmeasured disturbance
options.model=psim;


[y,t,u]=sim(mpcobj,Tf,ref,options);
y_hat=lsim(G,u);
error=y-y_hat;


% %%
% 
% % [F u_res]=whiten(u-mean(u),5);
% F = z^5/(z^5-0.6257*z^4+0.3454*z^3-0.1844*z^2+0.1053*z-0.0557)
% error_w=lsim(1/F,error-mean(error));
% 
% u_res = lsim(1/F,u-mean(u));
% 
% % plot(y)
% figure(1)
% [xww lags]=xcorr(u_res-mean(u_res),error_w-mean(error_w),50,'coeff');
% plot(lags,xww,'-b.','MarkerSize',20,'linewidth',2)
% ylim([-1 1])
% xlim([-50 5])
% grid on
% hold on
% 

% xlim([-10 4])
% 
% xlabel('\tau','fontsize',12)
% ylabel('R_{u,\epsilon}(\tau)','fontsize',12)
% legend('No Mismatch','50% Gain Mismatch','100% Gain Mismatch','150% Gain Mismatch','-50% Gain Mismatch')


% xlabel('\tau','fontsize',12)
% ylabel('R_{u,\epsilon}(\tau)','fontsize',12)
% legend('No Mismatch','No Mismatch + High Freq. Deterministic Dist.','100% Gain Mismatch','100% Gain Mismatch + High Freq. Deterministic Dist.')
