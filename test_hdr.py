import cv2
from skimage.metrics import structural_similarity as ssim
import os
import time
from PIL import Image, ExifTags
import numpy as np
from transmef import hdr_fuse
import pandas as pd
# TODO: Dockerise program, add ability for single image pair processing, clean code, documentation, etc.
def single_save_results(image, gt, metrics, single_data_range = False):
    if single_data_range:
        ssim_image =  ssim(image, gt, channel_axis=2, data_range=1)
    else:
        ssim_image = ssim(image, gt, channel_axis=2)
    metrics.append(ssim_image*100)
    metrics.append(cv2.PSNR(image, gt))
    if single_data_range:
        metrics.append(cv2.Laplacian(np.clip(image*255, 0, 255).astype('uint8'), cv2.CV_64F).var())
    else:
        metrics.append(cv2.Laplacian(image, cv2.CV_64F).var())
    return metrics
def get_image(path_name):
    if os.path.exists(path_name):
        return cv2.imread(path_name)
    else:
        return False
    
def read_col_names(init_names, number_string = ''):
    return [name + number_string for name in init_names]

def graph_all_results():
    first_path = '../EvaluationDataset/SSIM_PSNR_Misaligned.csv'
    print(os.path.exists(first_path))
    second_path = '../EvaluationDatasetAligned/SSIM_PSNR_Results.csv'
    first_dataframe = evaluate_hdr(True) if not os.path.exists((first_path)) else pd.read_csv(first_path)
    cols = ['Mertens', 'Dual Illumination Estimation', 'TransMEF', 'PASMEF', 'GhostFreeMEF', 'FMMEF']
    second_dataframe = evaluate_hdr(False) if not os.path.exists((second_path)) else pd.read_csv(second_path)
    print(first_dataframe.columns)
    print(second_dataframe.columns)
def evaluate_hdr(evaluate_misaligned = False):
    test_path = os.path.abspath('../EvaluationDataset' if evaluate_misaligned else '../EvaluationDatasetAligned')
    single_hdr_path = os.path.abspath('../code/result_imgs')
    unaligned_ev=-2
    aligned_ev=2
    results = []
    merge_mertens = cv2.createMergeMertens()
    merge_robertson = cv2.createMergeRobertson()
    merge_debevec = cv2.createMergeDebevec()
    results =[ ]
    column_names = ['Mertens' if evaluate_misaligned else 'Our Solution', 'Mertens (unaligned)', 'Dual Illumination Estimation', 'TransMEF (unaligned)', 'TransMEF (aligned)', 'PASMEF (aligned)', 'PASMEF (unaligned)', 'GhostFreeMEF (aligned)', 'GhostFreeMEF (unaligned)', 'FMMEF (aligned)', 'FMMEF (unaligned)']
    data_col = pd.MultiIndex.from_product([column_names, ['SSIM', 'PSNR', 'Blur']])
    
    for photo_dir in sorted(os.listdir(test_path)):
        if not photo_dir.__contains__('.'):
            start_time=time.time()
            print(f'photo dir = {photo_dir}')
            try:
                # ORDER = OUR MERTENS(UN), DIE, TRANS(AL), TRANS(UN), PAS, GF, PAS, FM(AL)
                metrics = []
                unaligned_images = [cv2.imread("/".join([test_path, photo_dir, f"input_2_unaligned_{unaligned_ev}.jpg"])), cv2.imread("/".join([test_path, photo_dir, f"input_1_aligned_{aligned_ev}.jpg"]))]
                images = [cv2.imread("/".join([test_path, photo_dir, f"input_1_aligned_{i}.jpg"])) for i in [-2, 0, 2]]
                
                trans_mef = cv2.imread("/".join([test_path, photo_dir, "trans_mef.jpg"]))
                images = [cv2.resize(image, (trans_mef.shape[1], trans_mef.shape[0])) for image in images]
                
                single_image_hdr = cv2.resize(cv2.imread("/".join([single_hdr_path, f"input_1_aligned_0_{photo_dir}.jpg"])), (trans_mef.shape[1], trans_mef.shape[0]))
                
                res_mertens_gt = merge_mertens.process(images)
                res_mertens_unaligned = merge_mertens.process(unaligned_images)
                

                if os.path.exists("/".join([test_path, photo_dir, "gt_mertens.jpg"])):
                    res_mertens_gt_jpg = cv2.imread("/".join([test_path, photo_dir, "gt_mertens.jpg"]))
                else:
                    res_mertens_gt_jpg = np.clip(res_mertens_gt*255, 0, 255).astype('uint8')
                    cv2.imwrite("/".join([test_path, photo_dir, "gt_mertens.jpg"]), res_mertens_gt_jpg)
                if os.path.exists("/".join([test_path, photo_dir, "mertens_uw_w.jpg"])):
                    res_mertens_unaligned_jpg = cv2.imread("/".join([test_path, photo_dir, "mertens_uw_w.jpg"]))
                else:
                    res_mertens_unaligned_jpg = np.clip(res_mertens_unaligned*255, 0, 255).astype('uint8')
                    cv2.imwrite("/".join([test_path, photo_dir, "mertens_uw_w.jpg"]), res_mertens_unaligned_jpg)
                if os.path.exists("/".join([test_path, photo_dir, "trans_mef_unaligned.jpg"])):
                    trans_mef_unaligned = cv2.imread("/".join([test_path, photo_dir, "trans_mef_unaligned.jpg"]))
                else:
                    trans_mef_unaligned = hdr_fuse.hdr_fuse(unaligned_images)
                    cv2.imwrite("/".join([test_path, photo_dir, "trans_mef_unaligned.jpg"]), res_mertens_unaligned_jpg)
                if os.path.exists("/".join([test_path, photo_dir, "trans_mef_aligned.jpg"])):
                    trans_mef_aligned = cv2.imread("/".join([test_path, photo_dir, "trans_mef_aligned.jpg"]))
                else:
                    trans_mef_aligned = hdr_fuse.hdr_fuse([images[0], images[2]])
                    cv2.imwrite("/".join([test_path, photo_dir, "trans_mef_aligned.jpg"]), trans_mef_aligned)
                cv2.imwrite("/".join([test_path, photo_dir, "gt_mertens.hdr"]), res_mertens_gt_jpg)
                # MERTENS
                if evaluate_misaligned:
                    metrics = single_save_results(res_mertens_gt_jpg, trans_mef, metrics)
                    res_mertens_gt_jpg = trans_mef
                    metrics = single_save_results(res_mertens_unaligned_jpg, res_mertens_gt_jpg, metrics)
                # OUR RESULT
                else:
                    metrics = single_save_results(trans_mef, res_mertens_gt_jpg, metrics)
                    # MERTENS UNALIGNED
                    metrics = single_save_results(res_mertens_unaligned, res_mertens_gt, metrics, True)
                # Dual Illumination Estimation
                metrics = single_save_results(single_image_hdr, res_mertens_gt_jpg, metrics)
                # TransMEF (unaligned)
                metrics = single_save_results(trans_mef_unaligned, res_mertens_gt_jpg, metrics)
                # TransMEF (aligned)
                metrics = single_save_results(trans_mef_aligned, res_mertens_gt_jpg, metrics)
                if os.path.exists("/".join([test_path, photo_dir, "PASMEF.jpg"])):
                    pas_mef = cv2.imread("/".join([test_path, photo_dir, "PASMEF.jpg"]))
                    metrics = single_save_results(pas_mef, res_mertens_gt_jpg, metrics)
                if os.path.exists("/".join([test_path, photo_dir, "PASMEF_uw_w.jpg"])):
                    pas_mef_uw_w = cv2.imread("/".join([test_path, photo_dir, "PASMEF_uw_w.jpg"]))
                    metrics = single_save_results(pas_mef_uw_w, res_mertens_gt_jpg, metrics)
                if os.path.exists("/".join([test_path, photo_dir, "GhostFreeMEF_DSIFT_GF.jpg"])):
                    ghost_free_mef = cv2.imread("/".join([test_path, photo_dir, "GhostFreeMEF_DSIFT_GF.jpg"]))
                    metrics = single_save_results(ghost_free_mef, res_mertens_gt_jpg, metrics)
                if os.path.exists("/".join([test_path, photo_dir, "GhostFreeMEF_DSIFT_GF_uw_w.jpg"])):
                    ghost_free_mef_uw_w = cv2.imread("/".join([test_path, photo_dir, "GhostFreeMEF_DSIFT_GF_uw_w.jpg"]))
                    metrics = single_save_results(ghost_free_mef_uw_w, res_mertens_gt_jpg, metrics)
                if os.path.exists("/".join([test_path, photo_dir, "fmmef.jpg"])):
                    fmmef = cv2.imread("/".join([test_path, photo_dir, "fmmef.jpg"]))
                    metrics = single_save_results(fmmef, cv2.resize(res_mertens_gt_jpg, (fmmef.shape[1], fmmef.shape[0])), metrics)
                if os.path.exists("/".join([test_path, photo_dir, "fmmef_uw_w.jpg"])):
                    fmmef_uw_w = cv2.imread("/".join([test_path, photo_dir, "fmmef_uw_w.jpg"]))
                    metrics = single_save_results(fmmef_uw_w, cv2.resize(res_mertens_gt_jpg, (fmmef.shape[1], fmmef.shape[0])), metrics)
                print(metrics)
                results.append(metrics)
            except:
                raise ValueError()
    df = pd.DataFrame(results, columns=data_col)
    df.to_csv("/".join([test_path, "SSIM_PSNR_Misaligned.csv" if evaluate_misaligned else "SSIM_PSNR_Results.csv"]))
    return df

if __name__ == "__main__":
    graph_all_results()