from esper.utils import *
import cv2
import random
import pickle
import numpy as np

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
    foreground_pose = []
    background_pose = []
    for fid in sorted(fid2pose):
        pose_list = fid2pose[fid]

        # need some filter for more than two poses case
        # neck_list = [pose._format_keypoints()[Pose.Neck] for pose in pose_list]
        # for neck in neck_list:
        # 	if neck[1] < 0 or neck[1] > 1080:
        # 		print(neck)	
        if len(pose_list) == 2:
	        poseA, poseB = pose_list[0], pose_list[1]
	        poseA_neck = poseA._format_keypoints()[Pose.Neck]
	        poseB_neck = poseB._format_keypoints()[Pose.Neck]
	        if poseA_neck[1] >= poseB_neck[1]:
	            foreground_pose += [(fid, poseA)] 
	            background_pose += [(fid, poseB)] 
	        else:
	            foreground_pose += [(fid, poseB)] 
	            background_pose += [(fid, poseA)] 
    return foreground_pose, background_pose


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


def get_pose_dist(poseA, poseB):
	kpA = poseA._format_keypoints()
	kpB = poseB._format_keypoints()
	dist = 0
	num_valid_kp = 0
	for i in range(Pose.POSE_KEYPOINTS):
		if tuple(kpA[i][:2]) == (0, 0) or tuple(kpB[i][:2]) == (0, 0):
			continue
		dist += np.linalg.norm(kpA[i][:2] - kpB[i][:2])
		num_valid_kp += 1
	return dist / num_valid_kp


def get_nearest_pose(pose, pose_list):
	best_dist = np.inf
	best_key = None
	for (fid, p) in pose_list:
		dist = get_pose_dist(pose, p)
		if dist < best_dist:
			best_dist = dist
			best_key = (fid, p)
	print("smallest distance: ", best_dist)
	return best_key


def visualize_pose_stick(img, pose, color):
	def draw_stick(pt1, pt2):
		if pt1 != (0, 0) and pt2 != (0, 0):
			cv2.line(img, pt1, pt2, color, 3)

	H, W = img.shape[:2]
	kp = pose._format_keypoints()
	kp = [(int(pt[0] * W), int(pt[1] * H)) for pt in kp]
	for i in range(Pose.POSE_KEYPOINTS):
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
	# add sticks 