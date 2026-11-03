%% Test du RNN identifié face au simulateur UAV_model.slx (boucle ouverte, même u)
clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(fullfile(here, '..', '..', 'Matlab files'));   % UAV_model.slx, quadcopter_model.p, random_init.m

%% 1. Paramètres de simulation
random_init;              % seeds du bruit (écrase Ts)
Ts = 0.05;
g_noise = 0.1;              % 0 d'abord, puis 0.1 comme dans EINT_RL_eval.m

%% 2. Chargement du RNN
d   = fullfile('..', 'model_rnn', 'lightning_logs', 'version_3', 'export_NN');   % poids exportés par RNN_drone.py
Wih = str2num(fileread(fullfile(d, 'weight_ih_l0.txt')));    % 16x1
Whh = str2num(fileread(fullfile(d, 'weight_hh_l0.txt')));    % 16x16
bh  = str2num(fileread(fullfile(d, 'bias_h.txt')))';         % 16x1
Wo  = str2num(fileread(fullfile(d, 'head_weight.txt')));     % 1x16
bo  = str2num(fileread(fullfile(d, 'head_bias.txt')));       % scalaire
nrm = str2num(fileread(fullfile(d, 'normalization.txt')));   % [u_scale, y_offset, y_scale]

%% 3. Signal de test : 2 s au repos puis paliers aléatoires (comme le dataset)
N = 400;  t = (0:N-1)' * Ts;
u = zeros(N, 1);  k = 41;
while k <= N
    L = randi([5 39]);  u(k:min(k+L-1, N)) = 2*rand - 1;  k = k + L;
end
u_ts = [t u];             % lu par le bloc From Workspace

%% 4. Simulation du drone (test_RNN_harness.slx)
out = sim('test_RNN_harness', 'StopTime', num2str((N-1)*Ts));
az_sim = out.az_sim(:);   % 1x1xN (3-D) -> vecteur colonne Nx1


%% 5. RNN pas à pas
h = zeros(16, 1);  az_rnn = zeros(N, 1);
for k = 1:N
    h = tanh(Wih * (u(k)/nrm(1)) + Whh * h + bh);
    az_rnn(k) = (Wo * h + bo) * nrm(3) + nrm(2);
end

%% 6. Comparaison
rmse = sqrt(mean((az_rnn - az_sim).^2));
figure('Name', 'RNN vs UAV_model');
subplot(2, 1, 1); stairs(t, u); grid on; ylabel('u');
title(sprintf('g\\_noise = %g : RMSE = %.3f m/s^2', g_noise, rmse));
subplot(2, 1, 2); plot(t, az_sim, t, az_rnn, '--', 'LineWidth', 1.2); grid on;
legend('UAV\_model', 'RNN'); ylabel('a_z [m/s^2]'); xlabel('t [s]');
