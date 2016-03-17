function [hiresImgs, loresImgs]=prepareImg(imgs, dataConf)

filters=dataConf.filters;       % Ò»¹²4¸öÂË²¨Æ÷
numImg=size(imgs);
hiresImgs = cell(numImg);
loresImgs=  cell(numImg);

% Load training high-res. image set and resample it
hires = modcrop(imgs, dataConf.scale); % crop a bit (to simplify scaling issues)
% Scale down images
lores = resize(hires, 1/dataConf.scale, dataConf.interpolate_kernel);
midres = resize(lores, dataConf.upsample_factor, dataConf.interpolate_kernel);
interpolated = resize(lores, dataConf.scale, dataConf.interpolate_kernel);
clear lores;

for i = 1: numImg % Remove low frequencies
     hiresImgs{i} = hires{i} - interpolated{i};
%      hiresImgs{i} = hires{i};
     for j = 1:numel(filters)
        loresIm(:,:,j)= conv2(midres{i}, filters{j}, 'same');
     end
     loresImgs{i}=loresIm;
     clear loresIm;
end

name_mat = sprintf('prepareImgs.mat');
save(name_mat,'hires','hiresImgs','loresImgs');

clear hires interpolated midres;

end