import cv2 as cv
import os
# (x1, y1), (x2, y2)
# (y1, y2), (x1, x2)
coordinates=[[0, 1200], [3024-1352, 3024]]
subdir="8"
repo = "../EvaluationDataset/" + subdir
newrepo ="../EvaluationDatasetInset/"

if not os.listdir(newrepo).__contains__(subdir):
    os.mkdir(newrepo + subdir)
newrepo += subdir
images = ["input_1_aligned_2.jpg", "input_2_unaligned_-2.jpg", "fmmef.jpg", "PASMEF.jpg", "gt_mertens.jpg", "trans_mef.jpg", "trans_mef_aligned.jpg", "GhostFreeMEF_DSIFT_GF.jpg"]
ref_image = "input_1_aligned_0.jpg"
new_coords = coordinates
cv.imwrite(newrepo + "/demonstrate_loc_"+ref_image, cv.rectangle(cv.imread(repo + "/" + ref_image), (new_coords[1][0], new_coords[0][0]), (new_coords[1][1], new_coords[0][1]), (0,0,255), 20))
for image in images:
    img = cv.imread(repo+"/"+image)
    print(img.shape)
    if img.shape == (1024, 768, 3):
        new_coords = [[int(xy/4) for xy in coordinate] for coordinate in coordinates]
        print(new_coords)
        # print(cv.imread(repo+"/"+image)[[[int(xy/4) for xy in coordinate] for coordinate in coordinates]].shape)
        # cv.imwrite(newrepo+"/inset_"+image, cv.imread(repo+"/"+image)[new_coords[0][0]:new_coords[0][1], new_coords[1][0]:new_coords[1][1]])
    else :
        new_coords = coordinates
    cv.imwrite(newrepo+"/inset_"+image, cv.imread(repo+"/"+image)[new_coords[0][0]:new_coords[0][1], new_coords[1][0]:new_coords[1][1]])
