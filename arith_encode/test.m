% This scrip using tmc13 compute bpp and psnr

% original point cloud files in ori_pc
% original big point cloud files in ori_big_pc
clearvars;

current_path = pwd;
ori_path = fullfile(current_path,'ori_pc');
% record all the ply file name in the path
ori_list = dir(fullfile(ori_path,'*.ply'));
ori_list = {ori_list.name};
if  exist('ori_big_pc','dir')==0
    mkdir(current_path,'ori_big_pc');
end


% enlarge original point cloud
for i=1:length(ori_list)
    
    pc = pcread(fullfile(ori_path,ori_list{i}));
    pc_location = pc.Location;
    pc_location = pc_location*100;
    pc2 = pointCloud(pc_location);
   % pcshow(pc2);
    pcwrite(pc2,['ori_big_pc/big_',ori_list{i}]);
end

big_ori_path = fullfile(current_path,'ori_big_pc');
big_ori_list = dir(fullfile(big_ori_path,'*.ply'));
big_ori_list = {big_ori_list.name};
if exist('rec_big_pc','dir')==0
    mkdir(current_path,'rec_big_pc');
end
if exist('tmc3_bin','dir')==0
   mkdir(current_path,'tmc3_bin'); 
end

% compute bpp
 % some tmc3 config path
config = '/home/yw/Desktop/latent_3d_points/arith_encode/encoder.cfg';
%positionQuantizationScale = [0.1 0.2 0.3 0.4 1];
positionQuantizationScale = 1;
reconstructedDataPath = fullfile(current_path,'rec_big_pc');
uncompressedDataPath = big_ori_path;
compressedStreamPath = fullfile(current_path,'tmc3_bin');
bpp=[];
psnr=[];
for i=1:length(big_ori_list)
    input_file = big_ori_list{i};
   
    for j=1:length(positionQuantizationScale)
        qp = positionQuantizationScale(j);

        cmd_line = ['./tmc3 --config=',config,' --reconstructedDataPath=',...
            fullfile(reconstructedDataPath,[num2str(qp),'_',input_file]),...
            ' --uncompressedDataPath=',fullfile(uncompressedDataPath,input_file),...
            ' --compressedStreamPath=',fullfile(compressedStreamPath,[input_file(1:end-4),'.bin']),...
            ' --positionQuantizationScale=',num2str(qp)];
        [status, cmdout] = system(cmd_line);
        bpp(i,j) = str2double(char(regexp(cmdout,'positions bitstream size .* B \((\S*) bpp\)','tokens','once')));
        
        [status, cmdout_2] = system(['./pc_error -a ',fullfile(uncompressedDataPath,input_file),...
            ' -b ',fullfile(reconstructedDataPath,[num2str(qp),'_',input_file])]);
        psnr(i,j) = str2double(char(regexp(cmdout_2,'mseF,PSNR \(p2point\): (\S*)','tokens','once')));
    end
    
end

fprintf('mean bpp:%f\n',mean(bpp));
fprintf('mean psnr:%f\n',mean(psnr));

