aligned_path = '..\EvaluationDatasetAligned\';
list=dir(aligned_path);
test_files = ["trans_mef.jpg", "PASMEF.jpg", "GhostFreeMEF_DSIFT_GF.jpg", "dual_ill_est.jpg", "trans_mef_aligned.jpg", "fmmef.jpg"];
for dir_num = 1:length(list)
    if list(dir_num).isdir && (list(dir_num).name ~= "." && list(dir_num).name ~= "..")
        path = strcat(aligned_path, list(dir_num).name, '\')
        ref = double(hdrread(append(path,  'gt_mertens.hdr')));
        [ref_x, ref_y, ref_c] = size(ref);
        for test_img = 1:length(test_files)
            if test_files(test_img) == "dual_ill_est.jpg"
                test = double(imread(strcat('../result_imgs/input_1_aligned_0_', int2str(dir_num), '.jpg')));
            else
                test = double(imread(strcat(path, test_files(test_img))));
            end
            
            [test_x, test_y, test_c] = size(test);
            if test_x == ref_y && test_y == ref_x && test_x ~= ref_x
                    test = imrotate(test, 270);
                    test_files(test_img)
                    imshow(test)
            end
            
            if (test_x ~= ref_x && test_y ~= ref_y && test_y ~= ref_x && test_x ~= ref_y)
                if (test_x >= test_y  && ref_x >= ref_y) || (test_x <= test_y && ref_x <= ref_y)
                    test = imrotate(test, 270);
                    [test_x, test_y, test_c] = size(test);
                    imshow(test)
                    test_files(test_img)
                end
                new_img = imresize(ref, [test_x, test_y]);
            else
                new_img = ref;
            end
            new_vals = hdrvdp3('civdm', test, new_img, 'RGB-native', 30).civdm;
            imwrite(new_vals.loss, strcat(path, 'hdr_vdp_3_loss_', test_files(test_img)))
            imwrite(new_vals.rev, strcat(path, 'hdr_vdp_3_rev_', test_files(test_img)))
            imwrite(new_vals.ampl, strcat(path, 'hdr_vdp_3_ampl_', test_files(test_img)))
        end
    end
end