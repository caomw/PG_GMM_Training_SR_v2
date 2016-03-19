function [gmm_conf, data_conf] = ParaSet

% gmm_conf.step = 3;
% gmm_conf.delta = [0.00002, 0.0000002];
% gmm_conf.win = 15;
% gmm_conf.ps  =  8;
gmm_conf.nlsp = 1;
gmm_conf.cls_num     =   1024;
% gmm_conf.data=[];

data_conf.scale = 3; % scale-up factor
data_conf.level = 1; % # of scale-ups to perform
data_conf.window = [3 3]; % low-res. window size
data_conf.border = [1 1]; % border of the image (to ignore)

% High-pass filters for feature extraction (defined for upsampled low-res.)
data_conf.upsample_factor = 3;       % upsample low-res. into mid-res.
O = zeros(1, data_conf.upsample_factor-1);
G = [1 O -1];               % Gradient
L = [1 O -2 O 1]/2;       % Laplacian
data_conf.filters = {G, G.', L, L.'}; % 2D versions
data_conf.interpolate_kernel = 'bicubic';
data_conf.overlap = [1 1]; % partial overlap (for faster training)

end