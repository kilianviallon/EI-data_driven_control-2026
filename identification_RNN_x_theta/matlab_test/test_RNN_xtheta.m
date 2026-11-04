%% Test des RNN de tangage (a_theta et ax) face au simulateur UAV_model.slx, même u_theta
%  Comme test_RNN.m pour az : un signal de test jamais vu (2 s au repos puis paliers aléatoires) excite la
%  boucle de tangage du simulateur (boucle_tangage.slx, mêmes réglages que la génération des bases) ;
%  le u_theta réellement appliqué est envoyé aux deux RNN et leurs sorties sont comparées au simulateur.
clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
gen = fullfile(here, '..', '..', 'generation_donnees_x_theta');
addpath(fullfile(here, '..', '..', 'Matlab files'), gen);      % UAV_model.slx, random_init.m, boucle_tangage.slx
% Compilation de Simulink dans un dossier au chemin court (limite de 260 caractères de Windows)
Simulink.fileGenControl('set', 'CacheFolder', fullfile(tempdir, 'sluav'), ...
                        'CodeGenFolder', fullfile(tempdir, 'sluav'), 'createDir', true);

%% 1. Paramètres de simulation (identiques à generer_bdd_x_theta.m)
random_init;              % graines du bruit (écrase Ts)
Ts = 0.05;
g_noise = 0.01;
u0 = 0.386; Kp = 0.02; Kd = 0.015; Kpsi = 0.04; Kdpsi = 0.03;

%% 2. Signal de test : 2 s au repos puis paliers aléatoires (comme la base)
N = 400;  t = (0:N-1)' * Ts;
exc = zeros(N, 1);  k = 41;
while k <= N
    L = randi([5 39]);  exc(k:min(k+L-1, N)) = 0.007 * (2*rand - 1);  k = k + L;
end
EXC = [t exc];            % lu par le bloc From Workspace de boucle_tangage.slx

%% 3. Simulation du drone
out = sim('boucle_tangage', 'StopTime', num2str((N-1)*Ts));
u = out.utheta(:);        % u_theta réellement appliqué (PD + excitation)
at_sim = out.atheta(:);
ax_sim = out.ax(:);

%% 4. RNN pas à pas (poids exportés par RNN_drone_xtheta.py, meilleure version de chaque modèle)
at_rnn = rnn(dossier_poids('RNN_a_theta_16neurones'), u);
ax_rnn = rnn(dossier_poids('RNN_ax_16neurones'), u);
ax32 = isfolder(fullfile(here, '..', 'RNN_ax_32neurones'));
if ax32, ax_rnn32 = rnn(dossier_poids('RNN_ax_32neurones'), u); end

%% 5. Comparaison
rmse_at = sqrt(mean((at_rnn - at_sim).^2));
rmse_ax = sqrt(mean((ax_rnn - ax_sim).^2));
if ax32, rmse_ax32 = sqrt(mean((ax_rnn32 - ax_sim).^2)); end
figure('Name', 'RNN de tangage vs UAV_model', 'Position', [100 100 900 750]);
subplot(3, 1, 1); stairs(t, u); hold on; stairs(t, exc, '--'); grid on;
ylabel('u_\theta'); legend('appliqué (PD + excitation)', 'excitation', 'Location', 'best');
title(sprintf('Signal de test jamais vu, g\\_noise = %g', g_noise));
subplot(3, 1, 2); plot(t, at_sim, t, at_rnn, '--', 'LineWidth', 1.2); grid on;
ylabel('a_\theta [rad/s^2]'); legend('UAV\_model', 'RNN', 'Location', 'best');
title(sprintf('a_\\theta : RMSE = %.3f rad/s^2 (écart-type du signal : %.2f)', rmse_at, std(at_sim)));
subplot(3, 1, 3); plot(t, ax_sim, t, ax_rnn, '--', 'LineWidth', 1.2); grid on; hold on;
ylabel('a_x [m/s^2]'); xlabel('t [s]');
if ax32
    plot(t, ax_rnn32, '-.', 'LineWidth', 1.2);
    legend('UAV\_model', 'RNN 16 neurones', 'RNN 32 neurones', 'Location', 'best');
    title(sprintf('a_x : RMSE = %.3f m/s^2 (16 neurones), %.3f m/s^2 (32 neurones) ; écart-type du signal : %.2f', ...
                  rmse_ax, rmse_ax32, std(ax_sim)));
else
    legend('UAV\_model', 'RNN 16 neurones', 'Location', 'best');
    title(sprintf('a_x : RMSE = %.3f m/s^2 (écart-type du signal : %.2f)', rmse_ax, std(ax_sim)));
end
exportgraphics(gcf, fullfile(here, '..', 'figures', 'test_RNN_xtheta.png'));
fprintf('RMSE a_theta = %.3f rad/s² (std %.3f) ; RMSE ax = %.3f m/s² (16 neurones)', rmse_at, std(at_sim), rmse_ax);
if ax32, fprintf(', %.3f m/s² (32 neurones)', rmse_ax32); end
fprintf(' (std %.3f)\n', std(ax_sim));


%% Fonctions locales
function d = dossier_poids(nom)
% Dossier export_NN de la version retenue (la seule exportée) du dossier de poids nom (ex. RNN_ax_16neurones)
here = fileparts(mfilename('fullpath'));
v = dir(fullfile(here, '..', nom, 'lightning_logs', 'version_*', 'export_NN'));
if isempty(v), error('Pas de poids exportés dans %s : lancer RNN_drone_xtheta.py', nom); end
[~, i] = max([v.datenum]);
d = v(i).folder;
end

function y = rnn(d, u)
% Même équations que test_RNN.m : h = tanh(Wih.u_n + Whh.h + bh), y = (Wo.h + bo).y_scale + y_offset
Wih = str2num(fileread(fullfile(d, 'weight_ih_l0.txt')));     %#ok<ST2NM>
Whh = str2num(fileread(fullfile(d, 'weight_hh_l0.txt')));     %#ok<ST2NM>
bh  = str2num(fileread(fullfile(d, 'bias_h.txt')))';          %#ok<ST2NM>
Wo  = str2num(fileread(fullfile(d, 'head_weight.txt')));      %#ok<ST2NM>
bo  = str2num(fileread(fullfile(d, 'head_bias.txt')));        %#ok<ST2NM>
nrm = str2num(fileread(fullfile(d, 'normalization.txt')));    %#ok<ST2NM>  [u_scale, y_offset, y_scale]
h = zeros(size(Whh, 1), 1);  y = zeros(numel(u), 1);
for k = 1:numel(u)
    h = tanh(Wih * (u(k) / nrm(1)) + Whh * h + bh);
    y(k) = (Wo * h + bo) * nrm(3) + nrm(2);
end
end
