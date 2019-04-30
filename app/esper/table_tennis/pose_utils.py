from esper.table_tennis.utils import *
import cv2
import random
import pickle
import numpy as np


def find_nearby_bg_frame(target_fid, player_bbox, match_scene_cls, fid2bbox):
    def check_bbox_overlap(bboxA, bboxB):
        # def check_pt_in_box(bbox, pt):
        #     return (pt[0] > bbox[0] and pt[0] < bbox[2] \
        #             and pt[1] > bbox[1] and pt[1] < bbox[3])
        
        # return check_pt_in_box(bboxA, (bboxB[0], bboxB[2])) or \
        #     check_pt_in_box(bboxA, (bboxB[0], bboxB[3])) or \
        #     check_pt_in_box(bboxA, (bboxB[1], bboxB[2])) or \
        #     check_pt_in_box(bboxA, (bboxB[1], bboxB[3]))
        if bboxA[0] > bboxB[2] or bboxB[0] > bboxA[2]: 
            return False
        if bboxA[1] > bboxB[3] or bboxB[1] > bboxA[3]: 
            return False
        return True

        
    def test_bg_frame(fid):
        if not match_scene_cls[fid]:
            return False

        for person in fid2bbox[fid]:
            if check_bbox_overlap(player_bbox, person['bbox']):
                return False
        return True
    
    for idx in range(1000):
        test_fid = target_fid + idx
        if test_bg_frame(test_fid):
            return test_fid
        test_fid = target_fid - idx
        if test_bg_frame(test_fid):
            return test_fid
    return None
    

def group_pose_from_interval(interval):
    video_id, sfid, efid = interval[:3]
    
    poses = Pose.objects.filter(frame__video_id=video_id) \
    .filter(frame__number__gte=sfid) \
    .filter(frame__number__lte=efid) \
    
    # group pose by fid
    fid2pose = {}
    for pose in poses:
        fid = pose.frame.number
        if fid not in fid2pose:
            fid2pose[fid] = []
        fid2pose[fid].append(pose)
    
    # separate pose into foreground and background        
    fid2pose_fg = {}
    fid2pose_bg = {}
    for fid in sorted(fid2pose):
        pose_list = fid2pose[fid]
        if len(pose_list) < 2:
            continue
        # filter two player
        neck_shoulder_dist = []
        for pose in pose_list:
            kp = pose._format_keypoints()
            if tuple(kp[Pose.Neck, :2]) == (0, 0) or tuple(kp[Pose.LShoulder, :2]) == (0, 0) or tuple(kp[Pose.RShoulder, :2]) == (0, 0):
                neck_shoulder_dist += [-1]
            else:
                neck_shoulder_dist += [(np.linalg.norm(kp[Pose.Neck, :2] - kp[Pose.LShoulder, :2]) + 
                    np.linalg.norm(kp[Pose.Neck, :2] - kp[Pose.RShoulder, :2])) / 2]
        top2 = np.argsort(neck_shoulder_dist)[-2:]

        poseA, poseB = pose_list[top2[0]], pose_list[top2[1]]
        poseA_neck = poseA._format_keypoints()[Pose.Neck]
        poseB_neck = poseB._format_keypoints()[Pose.Neck]
        if poseA_neck[1] >= poseB_neck[1]:
            fid2pose_fg[fid] = poseA 
            fid2pose_bg[fid] = poseB 
        else:
            fid2pose_fg[fid] = poseB 
            fid2pose_bg[fid] = poseA 
    return fid2pose_fg, fid2pose_bg


def collect_histogram(pose_list):
    def get_histogram(i):
        fid, pose = pose_list[i]
        img = load_frame(video, fid, [])
        H, W = img.shape[:2]
        keypoints = pose._format_keypoints()
        poly_vertices =[keypoints[Pose.LShoulder][:2], keypoints[Pose.LHip][:2], \
                        keypoints[Pose.RHip][:2], keypoints[Pose.RShoulder][:2]]
        poly_vertices = np.array([(int(pt[0]*W), int(pt[1]*H)) for pt in poly_vertices])
        mask = np.zeros((H, W))
        cv2.fillConvexPoly(mask, poly_vertices, 1)
        # for visualization
#         cv2.fillConvexPoly(img, poly_vertices, (255, 255, 255))
#         imshow(img)
        
        cloth = img[mask > 0]
        width = int(np.sqrt(cloth.shape[0]))
        cloth = cloth[:width * width].reshape((width, width, 3))
        hist_channel = []
        for i in range(3):
            hist_channel.append(cv2.calcHist([cloth], [i], None, [16], [0,256]))
        hist = np.vstack((hist_channel[0], hist_channel[1], hist_channel[2])).reshape(48)
        return hist / (1.*width*width)
#     get_histogram(0)    
    hist_all = par_for(get_histogram, [i for i in range(len(pose_list))])
    return hist_all


def get_openpose_dist(poseA, poseB):
	kpA = poseA._format_keypoints()
	kpB = poseB._format_keypoints()
	dist = 0
	num_valid_kp = 0
	for i in range(Pose.POSE_KEYPOINTS):
		if tuple(kpA[i, :2]) == (0, 0) or tuple(kpB[i, :2]) == (0, 0):
			continue
		dist += np.linalg.norm(kpA[i, :2] - kpB[i, :2])
		num_valid_kp += 1
	return dist / num_valid_kp


def get_nearest_openpose(target_pose, fid2pose):
	best_dist = np.inf
	best_key = None
	for fid, pose in fid2pose.items():
		dist = get_openpose_dist(target_pose, pose)
		if dist < best_dist:
			best_dist = dist
			best_key = (fid, pose)
	print("smallest distance: ", best_dist)
	return best_key


def visualize_openpose_stick(img, pose, color):
	def draw_stick(pt1, pt2):
		if pt1 != (0, 0) and pt2 != (0, 0):
			cv2.line(img, pt1, pt2, color, 3)

	H, W = img.shape[:2]
	kp = pose._format_keypoints()
	kp = [(int(pt[0] * W), int(pt[1] * H)) for pt in kp]
	for i in range(Pose.POSE_KEYPOINTS - 1):
		if tuple(kp[i][:2]) != (0, 0):
			cv2.circle(img, kp[i], 8, color, -1)

	# draw_stick(kp[Pose.Nose], kp[Pose.Neck])

	draw_stick(kp[Pose.Neck], kp[Pose.LShoulder])
	draw_stick(kp[Pose.LShoulder], kp[Pose.LElbow])
	draw_stick(kp[Pose.LElbow], kp[Pose.LWrist])
	
	draw_stick(kp[Pose.Neck], kp[Pose.RShoulder])
	draw_stick(kp[Pose.RShoulder], kp[Pose.RElbow])
	draw_stick(kp[Pose.RElbow], kp[Pose.RWrist])

	draw_stick(kp[Pose.Neck], kp[Pose.LHip])
	draw_stick(kp[Pose.LHip], kp[Pose.LKnee])
	draw_stick(kp[Pose.LKnee], kp[Pose.LAnkle])
	
	draw_stick(kp[Pose.Neck], kp[Pose.RHip])
	draw_stick(kp[Pose.RHip], kp[Pose.RKnee])
	draw_stick(kp[Pose.RKnee], kp[Pose.RAnkle])


def get_densepose_dist(poseA, poseB):
    dist = 0
    num_valid_kp = 0
    for i in range(17):
        if tuple(poseA[:2, i]) == (0, 0) or tuple(poseB[:2, i]) == (0, 0):
            continue
        dist += np.linalg.norm(poseA[:2, i] - poseB[:2, i])
        num_valid_kp += 1
    return dist / num_valid_kp


def get_nearest_densepose(target_pose, fid2pose):
    best_dist = np.inf
    best_key = None
    for fid, pose in fid2pose.items():
        dist = get_densepose_dist(target_pose, pose)
        if dist < best_dist:
            best_dist = dist
            best_key = (fid, pose)
    print("smallest distance: ", best_dist)
    return best_key


def visualize_densepose_stick(img, keyps, color):
    def draw_stick(pt1, pt2):
        pt1, pt2 = tuple(pt1), tuple(pt2)
        if pt1 != (0, 0) and pt2 != (0, 0):
            cv2.line(img, pt1, pt2, color, 3)

    H, W = img.shape[:2]
    for i in range(17):
        if tuple(keyps[:2, i]) != (0, 0):
            cv2.circle(img, tuple(keyps[:2, i]), 8, color, -1)

    # draw_stick(kp[Pose.Nose], kp[Pose.Neck])

    # draw_stick(keyps[:2, 0], keyps[:2, Pose.LShoulder])
    draw_stick(keyps[:2, 5], keyps[:2, 7])
    draw_stick(keyps[:2, 7], keyps[:2, 9])
    
    # draw_stick(keyps[:2, 0], keyps[:2, Pose.RShoulder])
    draw_stick(keyps[:2, 6], keyps[:2, 8])
    draw_stick(keyps[:2, 8], keyps[:2, 10])

    # draw_stick(keyps[:2, 0], keyps[:2, Pose.LHip])
    draw_stick(keyps[:2, 11], keyps[:2, 13])
    draw_stick(keyps[:2, 13], keyps[:2, 15])
    
    # draw_stick(keyps[:2, 0], keyps[:2, Pose.RHip])
    draw_stick(keyps[:2, 12], keyps[:2, 14])
    draw_stick(keyps[:2, 14], keyps[:2, 16])
