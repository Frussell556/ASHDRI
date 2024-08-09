import os
import torch
import argparse
import imageio
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit
from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from model_selection import select_model
from utils_flow.util_optical_flow import flow_to_image
from utils_flow.visualization_utils import overlay_semantic_mask
from validation.test_parser import define_model_parser
from hesaffnet import affinewarp
from transmef import hdr_fuse as hdr
from skimage.metrics import structural_similarity as ssim
import time
import color_matcher
import pandas as pd
from PIL import Image, ExifTags
from typing import Optional

def pad_to_same_shape(im1, im2):
    # pad to same shape both images with zero
    if im1.shape[0] <= im2.shape[0]:
        pad_y_1 = im2.shape[0] - im1.shape[0]
        pad_y_2 = 0
    else:
        pad_y_1 = 0
        pad_y_2 = im1.shape[0] - im2.shape[0]
    if im1.shape[1] <= im2.shape[1]:
        pad_x_1 = im2.shape[1] - im1.shape[1]
        pad_x_2 = 0
    else:
        pad_x_1 = 0
        pad_x_2 = im1.shape[1] - im2.shape[1]
    im1 = cv2.copyMakeBorder(im1, 0, pad_y_1, 0, pad_x_1, cv2.BORDER_CONSTANT)
    im2 = cv2.copyMakeBorder(im2, 0, pad_y_2, 0, pad_x_2, cv2.BORDER_CONSTANT)

    return im1, im2


# Argument parsing
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def hist_eq_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    split_img = cv2.split(img)
    equalised_img = [split_img[0], split_img[1], cv2.equalizeHist(split_img[2])]
    img = cv2.cvtColor(cv2.merge(equalised_img), cv2.COLOR_HSV2BGR)
    # TODO: clean this up! (make it function automatically)
    if img.shape[1] > 4000 or img.shape[0] > 4000:
        img = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)))
    elif img.shape[1] > 2000 or img.shape[0] > 2000:
        img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    return img
def test_model_on_image_pair(args, query_image, reference_image, sr, second_pass = True):
    with torch.no_grad():
        start_time = time.time()
        network, estimate_uncertainty = select_model(
            args.model, args.pre_trained_model, args, args.optim_iter, local_optim_iter,
            path_to_pre_trained_models=args.path_to_pre_trained_models)
        
        # save original ref image shape
        ref_image_shape = reference_image.shape[:2]

        # pad both images to the same size, to be processed by network
        query_image_, reference_image_ = pad_to_same_shape(query_image, reference_image)
        query_image_ = hist_eq_img(query_image_)
        reference_image_ = hist_eq_img(reference_image_)
        ref_matcher = color_matcher.ColorMatcher()
        reference_image_ = np.clip(ref_matcher.transfer(reference_image_, query_image_, 'mkl'), 0, 255).astype('uint8')
        if query_image.shape != query_image_.shape:
            query_image = cv2.resize(query_image, (query_image_.shape[1], query_image_.shape[0]))
        if reference_image_.shape != reference_image.shape:
            reference_image = cv2.resize(reference_image, (reference_image_.shape[1], reference_image_.shape[0]))
        # convert numpy to torch tensor and put it in right format
        query_image_ = torch.from_numpy(query_image_).permute(2, 0, 1).unsqueeze(0)
        reference_image_ = torch.from_numpy(reference_image_).permute(2, 0, 1).unsqueeze(0)
        preprocess_time = time.time() - start_time
        print(f'preprocess = {preprocess_time} seconds')
        # ATTENTION, here source and target images are Torch tensors of size 1x3xHxW, without further pre-processing
        # specific pre-processing (/255 and rescaling) are done within the function.

        # pass both images to the network, it will pre-process the images and ouput the estimated flow
        # in dimension 1x2xHxW
        start_time = time.time()
        if estimate_uncertainty:
            if args.flipping_condition:
                raise NotImplementedError('No flipping condition with PDC-Net for now')

            estimated_flow, uncertainty_components = network.estimate_flow_and_confidence_map(query_image_,
                                                                                              reference_image_,
                                                                                              mode='channel_first')
            confidence_map = uncertainty_components['p_r'].squeeze().detach().cpu().numpy()
            confidence_map = confidence_map[:ref_image_shape[0], :ref_image_shape[1]]
        else:
            if args.flipping_condition and 'GLUNet' in args.model:
                estimated_flow = network.estimate_flow_with_flipping_condition(query_image_, reference_image_,
                                                                               mode='channel_first')
            else:
                estimated_flow = network.estimate_flow(query_image_, reference_image_, mode='channel_first')
        estimated_flow_numpy = estimated_flow.squeeze().permute(1, 2, 0).cpu().numpy()
        estimated_flow_numpy = estimated_flow_numpy[:ref_image_shape[0], :ref_image_shape[1]]
        # removes the padding
        flow_estimation_time = time.time()-start_time
        print(f'flow estimation = {flow_estimation_time} seconds')
        start_time = time.time()
        
        # estimated_flow_numpy = sr.upsample(estimated_flow_numpy)
        # estimated_flow_numpy = cv2.resize(estimated_flow_numpy, (reference_image.shape[1], reference_image.shape[0]))
        warped_query_image = remap_using_flow_fields(query_image, estimated_flow_numpy[:, :, 0],
                                                     estimated_flow_numpy[:, :, 1]).astype(np.uint8)
        remapping_time = time.time()-start_time
        print(f'remapping = {remapping_time} seconds')
        print(warped_query_image.shape)
        # save images
        start_time = time.time()
        if args.save_ind_images:
            imageio.imwrite(os.path.join(args.save_dir, 'query.png'), query_image)
            imageio.imwrite(os.path.join(args.save_dir, 'reference.png'), reference_image)
            imageio.imwrite(os.path.join(args.save_dir, 'warped_query_{}_{}.png'.format(args.model, args.pre_trained_model)),
                            warped_query_image)

        if estimate_uncertainty:
            color = [255, 102, 51]
            fig, axis = plt.subplots(1, 6, figsize=(30, 30))
            confident_mask = (confidence_map > 0.20).astype(np.uint8)
            confident_warped = overlay_semantic_mask(warped_query_image, ann=255 - confident_mask*255, color=color)
            axis[2].imshow(confident_warped)
            axis[2].set_title('Confident warped query image according to \n estimated flow by {}_{}'
                              .format(args.model, args.pre_trained_model))
            axis[4].imshow(confidence_map, vmin=0.0, vmax=1.0)
            axis[4].set_title('Confident regions')
        else:
            fig, axis = plt.subplots(1, 4, figsize=(30, 30))
            axis[2].imshow(warped_query_image)
            axis[2].set_title(
                'Warped query image according to estimated flow by {}_{}'.format(args.model, args.pre_trained_model))
        axis[0].imshow(cv2.resize(query_image, (warped_query_image.shape[1], warped_query_image.shape[0])))
        axis[0].set_title('Query image')
        axis[1].imshow(cv2.resize(reference_image, (warped_query_image.shape[1], warped_query_image.shape[0])))
        axis[1].set_title('Reference image')

        axis[3].imshow(flow_to_image(cv2.resize(estimated_flow_numpy, (warped_query_image.shape[1], warped_query_image.shape[0]))))
        axis[3].set_title('Estimated flow {}_{}'.format(args.model, args.pre_trained_model))
        
        if second_pass:
            fig.savefig(
                os.path.join(args.save_dir, 'Warped_query_image_{}_{}.png'.format(args.model, args.pre_trained_model)),
                bbox_inches='tight')
        else:
            fig.savefig(
                os.path.join(args.save_dir, 'Warped_query_image_{}_{}_first_pass.png'.format(args.model, args.pre_trained_model)),
                bbox_inches='tight')
        plt.close(fig)
        print('Saved image!')
        save_graph_time = time.time() - start_time
        print(f'saving/generating graph = {save_graph_time} seconds')
        start_time = time.time() 
        # should calculate all this at a lower res!
        # confident_mask = (confidence_map > 0.2).astype(np.uint8)
        
        confident_mask =  cv2.resize(confident_mask, (reference_image.shape[1], reference_image.shape[0]))
        
        # confidence_channels = sr.upsample(np.clip(255*expit(20*(np.array(confidence_map)-0.25)), 0, 255).astype('uint8'))/255
        confidence_channels = expit(20*(np.array(confidence_map)-0.25))
        confidence_channels.shape
        axis[5].imshow(cv2.resize(confidence_channels,(warped_query_image.shape[1], warped_query_image.shape[0])), vmin=0.0, vmax=1.0)
        axis[5].set_title('B(X) blended confidence map')
        # confidence_channels = cv2.resize(expit(20*(np.array(confidence_map)-0.25)), (reference_image.shape[1], reference_image.shape[0]))
        confidence_map = cv2.resize(confidence_map, (reference_image.shape[1], reference_image.shape[0]))
        
        warp_query_hsv = cv2.split(cv2.cvtColor(warped_query_image, cv2.COLOR_RGB2YUV))
        # mask regions >= 250 (i.e. remove oversaturated pixels)
        if second_pass:
            ref_hsv = np.clip(ref_matcher.transfer(reference_image, query_image, 'hm-mkl-hm'), 0, 255).astype('uint8')
        else:
            ref_hsv = np.clip(ref_matcher.transfer(reference_image, query_image), 0, 255).astype('uint8')
        if second_pass:
            _, Y_mask = cv2.threshold(cv2.split(cv2.cvtColor(reference_image, cv2.COLOR_RGB2YUV))[0], 240, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5,5), np.uint8)
            
            Y_mask = np.repeat(cv2.dilate(cv2.morphologyEx(cv2.erode(Y_mask, kernel), cv2.MORPH_OPEN, kernel), kernel )[..., np.newaxis], 3, axis=2)
            if args.save_ind_images:
                cv2.imwrite(os.path.join(args.save_dir, 'path.png'), Y_mask)
            
            ref_hsv = cv2.bitwise_and(ref_hsv, ~Y_mask) + cv2.bitwise_and(Y_mask, query_image)
        ref_hsv = cv2.split(cv2.cvtColor(ref_hsv, cv2.COLOR_RGB2YUV))
        warp_query_hsv_replace = cv2.merge([
            (warp_query_hsv[0]*confidence_channels + ref_hsv[0]*(np.ones_like(confidence_channels) - confidence_channels)).astype('uint8'),
            (warp_query_hsv[1]*confidence_map+ref_hsv[1]*(np.ones_like(confidence_map) - confidence_map)).astype('uint8'), 
            (warp_query_hsv[2] * confidence_map+ref_hsv[2]*(np.ones_like(confidence_map) - confidence_map)).astype('uint8'), 
            ])
        img_blend_time = time.time() - start_time
        print(f'image blending = {img_blend_time} seconds')
        if args.save_ind_images:
            cv2.imwrite(os.path.join(args.save_dir, 'warped_query_{}_{}_blend_hsv_replace.png'.format(args.model, args.pre_trained_model)), cv2.cvtColor(sr.upsample(warp_query_hsv_replace), cv2.COLOR_YUV2BGR))
        return warp_query_hsv_replace, [preprocess_time, flow_estimation_time, remapping_time, img_blend_time, save_graph_time]
def hdr_gen(args, img_ue, img_oe, exposure_times, sr):
    img_pair = [cv2.cvtColor(img_ue, cv2.COLOR_YUV2BGR), cv2.cvtColor(img_oe, cv2.COLOR_RGB2BGR)]
    start_time = time.time()
    res_hdr = hdr.hdr_fuse(img_pair)
    cv2.imwrite(os.path.join(args.save_dir, 'trans_mef.jpg'), res_hdr)
    
    hdr_time = time.time() - start_time
    return res_hdr, hdr_time

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Test models on a pair of images')
    define_model_parser(parser)  # model parameters
    args = parser.parse_args()

    torch.cuda.empty_cache()
    torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # either gpu or cpu

    local_optim_iter = args.optim_iter if not args.local_optim_iter else int(args.local_optim_iter)

    
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    test_path = args.load_dir
    # test_path = os.path.abspath('../EvaluationDatasetAligned')
    unaligned_ev=-2
    aligned_ev=2
    results = []
    merge_mertens = cv2.createMergeMertens()
    merge_robertson = cv2.createMergeRobertson()
    merge_debevec = cv2.createMergeDebevec()
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("pretrained/FSRCNN_x4.pb")
    sr.setModel("fsrcnn", 4)
    for photo_dir in os.listdir(test_path):
        if os.path.isdir("\\".join([test_path, photo_dir,])):
            start_time=time.time()
            print(f'photo dir = {photo_dir}')
            args.save_dir = "\\".join([test_path, photo_dir,])
            img_pair = [ f"input_2_unaligned_{unaligned_ev}.jpg", f"input_1_aligned_{aligned_ev}.jpg"]
            # "\\".join([test_path, photo_dir, f"input_2_unaligned_{unaligned_ev}.jpg"]), Image.open("\\".join([test_path, photo_dir, f"input_1_aligned_{aligned_ev}.jpg"]
            img_paths = ["\\".join([test_path, photo_dir, img_name]) for img_name in img_pair]
            img_list = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_paths]
            images = [Image.open(img_path) for img_path in img_paths]
            exposure_times = np.array([[v for k, v in img._getexif().items() if k in ExifTags.TAGS and ExifTags.TAGS[k] == "ExposureTime" ][0] for img in  images], dtype=np.float32)
            warped_image, metrics = test_model_on_image_pair(args, img_list[0], img_list[1], sr=sr, second_pass=False)
            warped_image = sr.upsample(warped_image)
            if args.save_ind_images:
                cv2.imwrite(os.path.join(args.save_dir, 'warped_query_{}_{}_first_step.png'.format(args.model, args.pre_trained_model)), cv2.cvtColor(warped_image, cv2.COLOR_YUV2BGR))
            query_image = affinewarp.convert_to_grayscale_var(img_list[0])
            reference_image = affinewarp.convert_to_grayscale_var(warped_image)
            query_image, _ = affinewarp.affine_warp(reference_image, query_image, img_paths[0])
            final_warped_image, final_metrics = test_model_on_image_pair(args, cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB), img_list[1], sr)
            final_warped_image=sr.upsample(final_warped_image)
            hdr_image, hdr_time = hdr_gen(args, final_warped_image, img_list[1], exposure_times, sr=sr)
            metrics.extend(final_metrics)
            metrics.append(hdr_time)
            total_time = time.time() - start_time
            metrics.append(total_time)
            results.append(metrics)
            end_time = time.time()
            print(f"model takes {end_time-start_time} seconds to run")
    print(results)
    result_csv = pd.DataFrame(results)
    result_csv.to_csv(path_or_buf="\\".join([test_path, f"results_uw_{unaligned_ev}_w_{aligned_ev}.csv"]))



