%% neural network bpp 

clearvars;
code_path = '/home/yw/Desktop/latent_3d_points_entropy/data/recon_pc';
code_list = dir(fullfile(code_path,'chair_*.txt'));
code_list = {code_list.name};
bpp = [];
for i=1:length(code_list)
    code = importdata(fullfile(code_path,code_list{i}));
    xC{1} = code;
    [y, Res] = Arith07(xC);
    bpp(i) = Res(1,3)/2048;
end


%% neural network PSNR
pc_path = '/home/yw/Desktop/latent_3d_points_entropy/data/recon_pc';
ori_path = '/home/yw/Desktop/latent_3d_points_entropy/data/ori_pc';
ori_list = dir(fullfile('/home/yw/Desktop/latent_3d_points_entropy/data/ori_pc','ori_chair_*.ply'));
ori_list = {ori_list.name};
rec_list = dir(fullfile(pc_path,'rec_chair_*.ply'));
rec_list = {rec_list.name};
assert(length(ori_list)==length(rec_list));
for i=1:length(ori_list)
   [status, cmdout] = system(['./pc_error -a ',fullfile(ori_path,ori_list{i}), ' -b ',fullfile(pc_path,rec_list{i})]);
   psnr = str2double(char(regexp(cmdout,'mseF,PSNR \(p2point\): (\S*)','tokens','once')));
   PSNR(i) = psnr;
end

fprintf('mean bpp:%f\n',mean(bpp));
fprintf('mean psnr:%f\n',mean(psnr));
