from scannerpy import Client, DeviceType
from scannerpy.storage import NamedVideoStream, NamedStream
from scannertools import maskrcnn_detection
from scannertools import densepose_detection
from query.models import Video
from esper.table_tennis.utils import *

from detectron.utils.vis import vis_keypoints

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


def get_openpose_dist(poseA, poseB, size=None, shift=None):
    kpA = poseA._format_keypoints().copy()
    kpB = poseB._format_keypoints().copy()
    if not size is None:
        kpA[:, 0] *= size[0]
        kpA[:, 1] *= size[1]
        kpB[:, 0] *= size[0]
        kpB[:, 1] *= size[1]
    if not shift is None:
        # shift is in pixel
        kpA[:, 0] += shift[0]
        kpA[:, 1] += shift[1]

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


def get_openpose_by_fid(video_name, fid):
    pose_list = list(Pose.objects.filter(frame__video__path__contains=video_name, frame__number=fid))
    if len(pose_list) < 2:
        print(fid, len(pose_list))
        return None, None
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
        pose_fg = poseA 
        pose_bg = poseB 
    else:
        pose_fg = poseB 
        pose_bg = poseA 
    return pose_fg, pose_bg


def get_maskrcnn_by_fid(sc, video_name, fid):
    maskrcnn_stream = NamedStream(sc, video_name + '_maskrcnn')
    seq = sc.sequence(maskrcnn_stream._name)
    obj = seq.load(workers=1, rows=[fid])
    metadata = next(obj)

    if len(metadata) < 2:
        print(fid, len(metadata))
        return None, None
    PERSON_CATEGORY = maskrcnn_detection.CATEGORIES.index('person')
    # filter two player
    bbox_size_list = []
    for m in metadata:
        if int(m['label']) == PERSON_CATEGORY and m['score'] > 0.9:
            bbox_size_list += [m['bbox']['x2'] - m['bbox']['x1']]
        else:
            bbox_size_list += [0]

    top2 = np.argsort(bbox_size_list)[-2:]
    playerA, playerB = metadata[top2[0]], metadata[top2[1]]
    if playerA['bbox']['y1'] > playerB['bbox']['y1']:
        mask_fg = playerA['mask'] 
        mask_bg = playerB['mask'] 
    else:
        mask_fg = playerB['mask'] 
        mask_bg = playerA['mask'] 
    return mask_fg, mask_bg


class Person(object):
    Nose = 0
    Leye = 1
    Reye = 2
    Lear = 3
    Rear = 4
    LShoulder = 5
    RShoulder = 6
    LElbow = 7
    RElbow = 8
    LWrist = 9
    RWrist = 10
    LHip = 11
    RHip = 12
    LKnee = 13
    RKnee = 14
    LAnkle = 15
    RAnkle = 16
    KP_THRESH = 0
    KP_NUM = 17

    def __init__(self, bbox, keyp, mask, body=None, score=None):
        self.bbox = bbox.astype(np.int64)
        self.keyp = keyp
        self.mask = mask
        self.body = body
        self.score = score

    def align_keypoint(self, other, key=None):
        '''Align other's keyp to self. by joint key'''
        if not key is None and self.keyp[2, key] > self.KP_THRESH and other.keyp[2, key] > self.KP_THRESH:
            shift = self.keyp[:2, key] - other.keyp[:2, key]
        else:
            shift = (self.keyp[:2, self.LShoulder] + self.keyp[:2, self.RShoulder]) / 2 - \
                (other.keyp[:2, self.LShoulder] + other.keyp[:2, self.RShoulder]) / 2
        aligned_keyp = other.keyp.copy()
        aligned_keyp[:, 0] += shift[0]    
        aligned_keyp[:, 1] += shift[1]
        return aligned_keyp    

    def blend_keypoint(self, other, weight, key=None):
        blend_keyp = self.keyp.copy()
        # blend_keyp[:2] = self.keyp[:2] * weight + self.align_keypoint(other, key)[:2] * (1 - weight)
        self_keyp_avg = np.mean(self.keyp[:2, self.LShoulder: self.RShoulder+1], axis=0)
        other_keyp_avg = np.mean(other.keyp[:2, self.LShoulder: self.RShoulder+1], axis=0)
        for i in range(self.KP_NUM):
            if other.keyp[2, i] > self.KP_THRESH:
                blend_keyp[:2, i] = self_keyp_avg + (self.keyp[:2, i] - self_keyp_avg) * weight + (other.keyp[:2, i] - other_keyp_avg) * (1 - weight)
        # blend_keyp[2:] = np.min((self.keyp[2:], other.keyp[2:]), axis=0)
        return blend_keyp

    def draw_keypoint(self, image):
        return vis_keypoints(image, self.keyp.astype(np.int64), kp_thresh=2, alpha=1)

    def get_crop_box(self, im_size=(1080, 1920)):
        cx, cy = (self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2
        crop_width = 640
        crop_box = [cx - crop_width // 2, min(self.bbox[3] + 30, im_size[0]) - crop_width]
        crop_box += [crop_box[0] + crop_width, crop_box[1] + crop_width]
        if crop_box[0] < 0 or crop_box[2] > im_size[1] or crop_box[1] < 0 or crop_box[3] > im_size[0]:
            print(crop_box)
            return None
        else:
            return crop_box

### hacky
hit_annotation = pickle.load(open('/app/data/pkl/hit_annotation.pkl', 'rb'))
frame_ids_dict = {}
for k, v in hit_annotation.items():
    hit_dict = []
    for h in v.values():
        hit_dict += h
    frame_ids = [i for point in hit_dict for i in range(point[0]['fid']-25, point[-1]['fid']+25) ]
    frame_ids.sort()
    frame_ids_dict[k] = frame_ids
### hacky
def get_densepose_by_fid(sc, video_name, fid):
    def fid2idx(fid):
        return frame_ids_dict[video_name].index(fid)

    densepose_stream = NamedStream(sc, video_name + '_densepose')
    seq = sc.sequence(densepose_stream._name)
    obj = seq.load(workers=1, rows=[fid2idx(fid)])
    metadata = next(obj)

    assert len(metadata) >= 2, "Player not detected!"
    
    # filter two player by shoulder dist
    # shoulder_dist = []
    # for person in metadata:
    #     if person['keyp'][2, Person.LShoulder] < Person.KP_THRESH or person['keyp'][2, Person.RShoulder] < Person.KP_THRESH:
    #         shoulder_dist += [-1]
    #     else:
    #         shoulder_dist += [np.linalg.norm(person['keyp'][:2, Person.LShoulder] - person['keyp'][:2, Person.RShoulder])] 
    # top2 = np.argsort(shoulder_dist)[-2:]
    # personA = Person(metadata[top2[0]]['bbox'], metadata[top2[0]]['keyp'], metadata[top2[0]]['mask'], score=metadata[top2[0]]['score'])
    # personB = Person(metadata[top2[1]]['bbox'], metadata[top2[1]]['keyp'], metadata[top2[1]]['mask'], score=metadata[top2[1]]['score'])
    # if personA.keyp[1, 5] >= personB.keyp[1, 5]:
    #     person_fg = personA
    #     person_bg = personB 
    # else:
    #     person_fg = personB
    #     person_bg = personA 

    # filter two player by bbox area
    bbox_area = []
    for person in metadata:
        area = (person['bbox'][2] - person['bbox'][0]) * (person['bbox'][3] - person['bbox'][1])
        bbox_area.append(area)
    top2 = np.argsort(bbox_area)[-2:]
    person_fg = person_bg = None
    personA = Person(metadata[top2[0]]['bbox'], metadata[top2[0]]['keyp'], metadata[top2[0]]['mask'], score=metadata[top2[0]]['score'])
    personB = Person(metadata[top2[1]]['bbox'], metadata[top2[1]]['keyp'], metadata[top2[1]]['mask'], score=metadata[top2[1]]['score'])
    if personA.keyp[1, Person.LShoulder] >= personB.keyp[1, Person.LShoulder]:
        person_fg = personA
        person_bg = personB 
    else:
        person_fg = personB
        person_bg = personA 

    return person_fg, person_bg

