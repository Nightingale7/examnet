function [T,YT] = run_f16(roll,pitch,yaw)

% Define model
global sys;

% Define initial conditions
evalin('base',"initialState(4) = "+string(roll)+";");   % Roll angle from wings level (rad)
evalin('base',"initialState(5) = "+string(pitch)+";");  % Pitch angle from nose level (rad)
evalin('base',"initialState(6) = "+string(yaw)+";");    % Yaw angle from North (rad)

evalin('base',"x_f16_0(4) = "+string(roll)+";");        % Roll angle from wings level (rad)
evalin('base',"x_f16_0(5) = "+string(pitch)+";");       % Pitch angle from nose level (rad)
evalin('base',"x_f16_0(6) = "+string(yaw)+";");         % Yaw angle from North (rad)

set_param(sys, 'SimulationCommand', 'update');

%cd('AeroBenchVV-develop/src/main/Simulink');

% Execute the simulink model
[T, XT] = sim(strcat(sys,'.slx'));

%cd('../../../..');

T = t_out;
   
% YT becomes the predicates state trajectory, [Altitude, GCAS_mode, Roll, Pitch]
YT = [x_f16_out(:,12) GCAS_mode_out x_f16_out(:,4) x_f16_out(:,5)];
end
