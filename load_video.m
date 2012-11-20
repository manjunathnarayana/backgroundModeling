%Script that reads video folder names, sets up output folder names, etc
%/*
%* Copyright (c) 2012, Manjunath Narayana, UMass-Amherst
%* All rights reserved.
%

output_main_folder = '/scratch1/narayana/output_data/dist_fields/kde_output';

%Video number 
if(video_number == 1)
    %bootstrap flag - set to 1 only for video number 2
    bootstrap = 0;
    %Input video frames foldername
    input_video_folder = '/nfs/orac/people/narayana/data/videos/I2R/Hall';
    %Videoname for display and output purposes
    videoname = 'Hall';
    %Folder for output
    output_sequences_folder = sprintf('%s/I2R/Hall', output_main_folder);
    %Input groundtruth images folder
    input_groundtruth_video_folder = '/nfs/orac/people/narayana/data/videos/I2R/GroundTruth';
    %Suffix that is used in filename for input videos
    input_sequence_files_suffix = 'airport';
    %Suffix that is used in filename for input videos groundtruth images
    input_groundtruth_sequence_files_suffix = 'gt_new_airport';
    %File extension of input files
    file_ext = 'bmp';
    %Folder from which to read images for initial background model
    bg_input_video_folder = input_video_folder;
    %Suffix that is used in filename for background initialization images
    bg_input_sequence_files_suffix = input_sequence_files_suffix;
    %Background model will be initialized from this frame number
    bg_frames_start = 1;
    %to this frame number
    bg_frames_end = 50;
    %Sequence filename begins from this number
    seq_starts_from = 1000;
    %To skip to a particular frame, use skip_frames
    %Classification begins at skip_frames frame
    skip_until_frame = 1051;
    %Total number of frames to classify
    total_num_frames = 4583;
    %TODO - Remove this
    skip_until_frame = 1654;
    total_num_frames = 1658;

    %Frame numbers for which groundtruth is available
    ground_truth_frames = [ 1656 2180 2289 2810 2823 2926 2961 3049 3409 3434 3800 3872 3960 4048 4257 4264 4333 4348 4388 4432];
end

if(video_number == 2)
    %bootstrap flag - set to 1 only for video number 2 
    bootstrap = 1;
    input_video_folder = '/nfs/orac/people/narayana/data/videos/I2R/Bootstrap';
    videoname = 'Bootstrap';
    output_sequences_folder = sprintf('%s/I2R/Bootstrap', output_main_folder);
    input_groundtruth_video_folder = '/nfs/orac/people/narayana/data/videos/I2R/GroundTruth';
    input_sequence_files_suffix = 'b0';
    input_groundtruth_sequence_files_suffix = 'gt_new_b';
    file_ext = 'bmp';
    bg_input_video_folder = input_video_folder;
    bg_input_sequence_files_suffix = input_sequence_files_suffix;
    bg_frames_start = 1;
    bg_frames_end = 10;
    seq_starts_from = 1000;
    skip_until_frame = 1011;
    total_num_frames = 3054;
    ground_truth_frames = [ 1021 1119 1285 1362 1408 1416 1558 1724 1832 1842 1912 2238 2262 2514 2624 2667 2832 2880 2890 2918];
end

if(video_number == 3)
    %bootstrap flag - set to 1 only for video number 2
    bootstrap = 0;
    input_video_folder = '/nfs/orac/people/narayana/data/videos/I2R/Curtain';
    videoname = 'Curtain';
    input_groundtruth_video_folder = '/nfs/orac/people/narayana/data/videos/I2R/GroundTruth';
    output_sequences_folder = sprintf('%s/I2R/Curtain', output_main_folder);
    input_sequence_files_suffix = 'Curtain';
    input_groundtruth_sequence_files_suffix = 'gt_new_Curtain';
    file_ext = 'bmp';
    bg_input_video_folder = input_video_folder;
    bg_input_sequence_files_suffix = input_sequence_files_suffix;
    bg_frames_start = 1;
    bg_frames_end = 50;
    seq_starts_from = 22100;
    skip_until_frame = 22151;
    total_num_frames = 23963;
    ground_truth_frames = [ 22772 22774 22847 22849 22890 23206 23222 23226 23233 23242 23257 23266 23786 23790 23801 23817 23852 23854 23857 23893];
end

if(video_number == 4)
    %bootstrap flag - set to 1 only for video number 2
    bootstrap = 0;
    input_video_folder = '/nfs/orac/people/narayana/data/videos/I2R/Escalator';
    videoname = 'Escalator';
    input_groundtruth_video_folder = '/nfs/orac/people/narayana/data/videos/I2R/GroundTruth';
    output_sequences_folder = sprintf('%s/I2R/Escalator', output_main_folder);
    input_sequence_files_suffix = 'airport';
    input_groundtruth_sequence_files_suffix = 'gt_new_Escalator';
    file_ext = 'bmp';
    bg_input_video_folder = input_video_folder;
    bg_input_sequence_files_suffix = input_sequence_files_suffix;
    bg_frames_start = 1;
    bg_frames_end = 50;
    seq_starts_from = 1398;
    skip_until_frame = 1449;
    total_num_frames = 4814;
    ground_truth_frames = [2424 2532 2678 2805 2913 2952 2978 3007 3078 3260 3279 3353 3447 3585 3743 4277 4558 4595 4769 4787];
end

if(video_number == 5)
    %bootstrap flag - set to 1 only for video number 2
    bootstrap = 0;
    input_video_folder = '/nfs/orac/people/narayana/data/videos/I2R/Fountain';
    videoname = 'Fountain';
    input_groundtruth_video_folder = '/nfs/orac/people/narayana/data/videos/I2R/GroundTruth';
    output_sequences_folder = sprintf('%s/I2R/Fountain', output_main_folder);
    input_sequence_files_suffix = 'Fountain';
    input_groundtruth_sequence_files_suffix = 'gt_new_Fountain';
    file_ext = 'bmp';
    bg_input_video_folder = input_video_folder;
    bg_input_sequence_files_suffix = input_sequence_files_suffix;
    bg_frames_start = 1;
    bg_frames_end = 50;
    seq_starts_from = 1000;
    skip_until_frame = 1051;
    total_num_frames = 1522;
    ground_truth_frames = [1157 1158 1165 1179 1184 1189 1190 1196 1202 1204 1422 1426 1430 1440 1453 1465 1477 1489 1494 1509];
end

if(video_number == 6)
    %bootstrap flag - set to 1 only for video number 2
    bootstrap = 0;
    input_video_folder = '/nfs/orac/people/narayana/data/videos/I2R/ShoppingMall';
    videoname = 'ShoppingMall';
    input_groundtruth_video_folder = '/nfs/orac/people/narayana/data/videos/I2R/GroundTruth';
    output_sequences_folder = sprintf('%s/I2R/ShoppingMall', output_main_folder);
    input_sequence_files_suffix = 'ShoppingMall';
    input_groundtruth_sequence_files_suffix = 'gt_new_ShoppingMall';
    file_ext = 'bmp';
    bg_input_video_folder = input_video_folder;
    bg_input_sequence_files_suffix = input_sequence_files_suffix;
    bg_frames_start = 1;
    bg_frames_end = 50;
    seq_starts_from = 1000;
    skip_until_frame = 1051;
    total_num_frames = 2285;
    ground_truth_frames = [1433 1535 1553 1581 1606 1649 1672 1740 1750 1761 1780 1827 1862 1892 1899 1920 1980 2018 2055 2123];
end

if(video_number == 7)
    %bootstrap flag - set to 1 only for video number 2
    bootstrap = 0;
    input_video_folder = '/nfs/orac/people/narayana/data/videos/I2R/Lobby';
    videoname = 'Lobby';
    output_sequences_folder = sprintf('%s/I2R/Lobby', output_main_folder);
    input_groundtruth_video_folder = '/nfs/orac/people/narayana/data/videos/I2R/GroundTruth';
    input_sequence_files_suffix = 'SwitchLight';
    input_groundtruth_sequence_files_suffix = 'gt_new_SwitchLight';
    file_ext = 'bmp';
    bg_input_video_folder = input_video_folder;
    bg_input_sequence_files_suffix = input_sequence_files_suffix;
    bg_frames_start = 1;
    bg_frames_end = 50;
    seq_starts_from = 1000;
    skip_until_frame = 1051;
    total_num_frames = 2545;
    ground_truth_frames = [ 1349 1353 1368 1634 1649 2019 2245 2247 2260 2265 2388 2436 2446 2457 2466 2469 2497 2507 2509 2514];
end

if(video_number == 8)
    %bootstrap flag - set to 1 only for video number 2
    bootstrap = 0;
    input_video_folder = '/nfs/orac/people/narayana/data/videos/I2R/Campus';
    videoname = 'Campus';
    input_groundtruth_video_folder = '/nfs/orac/people/narayana/data/videos/I2R/GroundTruth';
    output_sequences_folder = sprintf('%s/I2R/Campus', output_main_folder);
    input_sequence_files_suffix = 'trees';
    input_groundtruth_sequence_files_suffix = 'gt_new_trees';
    file_ext = 'bmp';
    bg_input_video_folder = input_video_folder;
    bg_input_sequence_files_suffix = input_sequence_files_suffix;
    bg_frames_start = 1;
    bg_frames_end = 50;
    seq_starts_from = 1000;
    skip_until_frame = 1051;
    total_num_frames = 2438;
    ground_truth_frames = [1372 1392 1394 1450 1451 1489 1650 1695 1698 1758 1785 1812 1831 1839 1845 2011 2013 2029 2032 2348];
end

if(video_number == 9)
    %bootstrap flag - set to 1 only for video number 2
    bootstrap = 0;
    input_video_folder = '/nfs/orac/people/narayana/data/videos/I2R/WaterSurface';
    videoname = 'WaterSurface';
    input_groundtruth_video_folder = '/nfs/orac/people/narayana/data/videos/I2R/GroundTruth';
    output_sequences_folder = sprintf('%s/I2R/WaterSurface', output_main_folder);
    input_sequence_files_suffix = 'WaterSurface';
    input_groundtruth_sequence_files_suffix = 'gt_new_WaterSurface';
    file_ext = 'bmp';
    bg_input_video_folder = input_video_folder;
    bg_input_sequence_files_suffix = input_sequence_files_suffix;
    bg_frames_start = 1;
    bg_frames_end = 50;
    seq_starts_from = 1000;
    skip_until_frame = 1051;
    total_num_frames = 1632;
    ground_truth_frames = [1499 1515 1523 1547 1548 1553 1554 1559 1575 1577 1594 1597 1601 1605 1615 1616 1619 1620 1621 1624];
end
