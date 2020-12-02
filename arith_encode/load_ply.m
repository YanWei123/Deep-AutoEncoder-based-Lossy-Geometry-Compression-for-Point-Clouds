filename = '/home/yw/Desktop/latent_3d_points/data/recon_pc/ori_chair_1.ply';
pc1 = pcread(filename);
pc1_location = pc1.Location;
pc1_location = pc1_location*100;
pc2 = pointCloud(pc1_location);
pcshow(pc2);
pcwrite(pc2,'ori_chair_1_big.ply');
