aligned_path = '../EvaluationDatasetAligned/';
list=dir(aligned_path);
test_files = ["trans_mef.jpg", "PASMEF_uw_w.jpg", "GhostFreeMEF_DSIFT_GF_uw_w.jpg", "dual_ill_est.jpg", "trans_mef_unaligned.jpg", "fmmef_uw_w.jpg", "mertens_uw_w.jpg"];
new_val_series = [];
for dir_num = 1:length(list)
    if list(dir_num).isdir && (list(dir_num).name ~= "." && list(dir_num).name ~= "..")
        path = strcat(aligned_path, list(dir_num).name, '/')
        ref = double(hdrread(append(path,  'gt_mertens.hdr')));
        [ref_x, ref_y, ref_c] = size(ref);
        new_val_set = [];
        for test_img = 1:length(test_files)
            if test_files(test_img) == "dual_ill_est.jpg"
                test = double(imread(strcat('../result_imgs/input_1_aligned_0_', int2str(dir_num), '.jpg')));
            else
                test = double(imread(strcat(path, test_files(test_img))));
            end
            
            [test_x, test_y, test_c] = size(test);
            if test_x == ref_y && test_y == ref_x && test_x ~= ref_x
                    test = imrotate(test, 270);
                    size(test)
            end

            if (test_x ~= ref_x && test_y ~= ref_y && test_y ~= ref_x && test_x ~= ref_y)
                if (test_x >= test_y  && ref_x >= ref_y) || (test_x <= test_y && ref_x <= ref_y)
                    test = imrotate(test, 270);
                    [test_x, test_y, test_c] = size(test);
                end
                
                new_img = imresize(ref, [test_x, test_y]);
                size(new_img)
                %[ref_x, ref_y, ref_c] = size(ref);
                size(test)
                new_vals = hdrvdp3('quality', test, new_img, 'RGB-native', 30);
            else
                size(test)
                new_vals = hdrvdp3('quality', test, ref, 'RGB-native', 30);
            end
            
            new_val_set = [new_val_set, new_vals];
        end
        new_val_series = [new_val_series; new_val_set];
    end
end