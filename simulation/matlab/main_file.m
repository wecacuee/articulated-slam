% Creating links and associated rotation and translation
clc;%clear all;close all;
%{
data = [2.33,0,0,1.866,0.833,-0.5;...
    2.11,0,0,1.866,0.611,-0.5;...
    2.55,0,0,1.86,1.05,-0.5];

data = [1,0,0,1.7071,1.7071,0;...
    2,0,0,2.4142,2.4142,0;...
    3,0,0,3.1213,3.1213,0];


%}


data = [1.11111,0,0,1.09622504,0,-0.0555556;...
    1.22222,0,0,1.1924509,0,-0.111111;...
    1.33333,0,0,1.28867513,0,-0.16666];

[R,T,status] = prosecutes_analysis(data);

% Testing the transformation
if status
    R*data(1,1:3)'+T
end
