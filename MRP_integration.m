% MRP integration
clc;
clear;
close all;

s0=[0.4;0.2;-0.1];
w=@(t) [sin(0.1*t);0.01;cos(0.1*t)]*deg2rad(20);
dt = 0.01;
time = 0:dt:42;

t_stop=0;
j = 0;
gh = cell(1);
options = odeset('Events',@singularity_prevention);

while t_stop<time(end)
    
    [T,Y]=ode45(@(t,y) dsfunc(t,y,w),time,s0,options);
    j=j+1;
    gh{j}=Y;
    s0 = Y(end,:);
    s0 = -s0/norm(s0)^2;
    t_stop=T(end);
    time = t_stop:dt:42;
end
%%
Y=[gh{1}(1:end-1,:);gh{2}(1:end-1,:);gh{3}(1:end-1,:);gh{4}];
norms_s=zeros(size(Y,1),1);
for i=1:size(Y,1)
    norms_s(i)=norm(Y(i,:));
end
figure()
plot(norms_s);


disp(norm(Y(end,:)));


%% Functions
function dy = dsfunc(t,s,w)
w_t = w(t);
x = s;
s_skew=[0 -x(3) x(2) ; x(3) 0 -x(1) ; -x(2) x(1) 0 ];

B = (1-norm(s)^2)*eye(3)+2*s_skew+2*(s*s');

dy = 1/4*B*w_t;

end

function [check,isterminal,direction] = singularity_prevention(~,y,~)
check = norm(y)-1;
isterminal=1;
direction = 0;
end