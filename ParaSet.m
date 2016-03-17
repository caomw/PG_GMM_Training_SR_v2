function [gmmConf, dataConf] = ParaSet

 gmmConf.step = 3;
 gmmConf.delta = [0.00002, 0.0000002];
gmmConf.win = 15;
gmmConf.ps  =  8;
gmmConf.nlsp = 10;
gmmConf.cls_num     =   32;

dataConf.scale = 3; % scale-up factor
dataConf.level = 1; % # of scale-ups to perform
dataConf.window = [3 3]; % low-res. window size
dataConf.border = [1 1]; % border of the image (to ignore)

% High-pass filters for feature extraction (defined for upsampled low-res.)
dataConf.upsample_factor = 3;       % upsample low-res. into mid-res.
O = zeros(1, dataConf.upsample_factor-1);
G = [1 O -1];               % Gradient
L = [1 O -2 O 1]/2;       % Laplacian
dataConf.filters = {G, G.', L, L.'}; % 2D versions
dataConf.interpolate_kernel = 'bicubic';
dataConf.overlap = [1 1]; % partial overlap (for faster training)
end