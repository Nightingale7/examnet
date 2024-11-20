echo off;

% 1 - Add to path "AeroBenchVV-develop" and all its subfolders
addpath(genpath("f16n"));

% 2 - Deactivate warning
warning off Stateflow:cdr:UnusedDataOrEvent;

% 3 - Define model name and load it
global sys;
sys = "AeroBenchSim_2019a";
load_system(sys);

% 4 - Load system configurations parameters
curDir = pwd;
cd('f16n/AeroBenchVV-develop/src/main');
evalin('base','SimConfig;');
cd(curDir);

% 5 - Assign manually parameters
initAlt = 2338;
simTime = 15;

assignin('base','InitAlt',initAlt);
assignin('base','model_err',false);
assignin('base','analysisOn',false);
assignin('base','printOn',false);
assignin('base','plotOn',false);
assignin('base','backCalculateBestSamp',false);
assignin('base','t_end',simTime);
evalin('base','initialState(12) = InitAlt;');   % Inital Altitude of the System set appropriately
evalin('base','x_f16_0(12) = InitAlt;');        % Inital Altitude of the System set appropriately

