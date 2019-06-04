from esper.table_tennis.utils import *
from esper.table_tennis.motion_control import *
from esper.table_tennis.pose_utils import * 

import pycocotools.mask as mask_util

import numpy as np
import cv2
import pickle


def L2(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def median(pt1, pt2):
    return ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)


def add(pt1, pt2):
    return (pt1[0] + pt2[0], pt1[1] + pt2[1])


def minus(pt1, pt2):
    return (pt1[0] - pt2[0], pt1[1] - pt2[1])


def interpolate_trajectory_from_hit(hit_traj):
    ball_traj = []
    start = hit_traj[0]
    i = 1
    while i != len(hit_traj):
        end = hit_traj[i]
        nframe = end['fid'] - start['fid']
        step_x = 1. * (end['pos'][0] - start['pos'][0]) / nframe
        step_y = 1. * (end['pos'][1] - start['pos'][1]) / nframe 
        for idx in range(nframe):
            ball_traj.append({'fid' : start['fid']+idx, 
                              'pos': (int(np.round(start['pos'][0] + step_x * idx)), 
                              int(np.round(start['pos'][1] + step_y * idx)))})
        i += 1
        start = end
    ball_traj.append({'fid' : start['fid'], 'pos': start['pos']})
    return ball_traj


#########################################################################
##### Generate motion from simple left/right control
#########################################################################

def group_motion(interval, fid2openpose, step_size=5, fps=25):
    _, sfid, efid, _ = interval
    x_list = [fid2openpose_A[fid]._format_keypoints()[Pose.Neck, 0] for fid in range(sfid, efid) if fid in fid2openpose]
    x_smooth = savgol_filter(x_list, fps, 2)
    motion_dict = {'left': [], 'right': [], 'still': []}
   
    for i in range(0, efid - sfid, step_size):
        for j in range(i + fps, efid - sfid, step_size):
            x_seg = x_smooth[i : j]
            xmin, xmax = min(x_seg), max(x_seg)
            argmin = np.argmin(x_seg)
            argmax = np.argmax(x_seg)
            if xmax - xmin < 0.01:
                avgx = np.average(x_seg)
                motion_dict['still'] += [{'start_x': avgx, 'end_x': avgx, 'start_fid': i+sfid, 'end_fid' : j+sfid, 
                                          'duration': (j-i)/video.fps}]
            elif (xmax - xmin) > 0.05 and argmin == 0 and argmax == len(x_seg) - 1:
                # decide whether the slope is smooth
                valid = True
                for k in range(0, len(x_seg) - fps, fps):
                    if x_seg[k+fps] - x_seg[k] < 0.01:
                        valid = False
                        break
                if valid:
                    motion_dict['right'] += [{'start_x': xmin, 'end_x': xmax, 'start_fid': i+sfid, 'end_fid' : j+sfid, 
                                          'duration': (j-i)/video.fps}]
            elif (xmax - xmin) > 0.04 and argmax == 0 and argmin == len(x_seg) - 1:
                # decide whether the slope is smooth
                valid = True
                for k in range(0, len(x_seg) - fps, fps):
                    if x_seg[k] - x_seg[k+fps] < 0.01:
                        valid = False
                        break
                if valid:
                    motion_dict['left'] += [{'start_x': xmax, 'end_x': xmin, 'start_fid': i+sfid, 'end_fid' : j+sfid, 
                                          'duration': (j-i)/video.fps}] 
    def merge_overlap_seg(motion_list):
        if len(motion_list) == 0:
            return []
        motion_list.sort(key = lambda x : x['start_fid'])
        motion_list_merge = []
        old_sfid = motion_list[0]['start_fid']
        for seg in motion_list:
            if seg['start_fid'] == old_sfid:
                longest_seg = seg
            elif old_sfid != -1:
                motion_list_merge.append(longest_seg)
                old_sfid = -1
            if seg['start_fid'] > longest_seg['end_fid']:
                old_sfid = seg['start_fid']
                longest_seg = seg
        if len(motion_list_merge) == 0:
            motion_list_merge.append(longest_seg)
        elif longest_seg['start_fid'] != motion_list_merge[-1]['start_fid']:
            motion_list_merge.append(longest_seg)
        return motion_list_merge
    
    motion_dict_filter = {'left': merge_overlap_seg(motion_dict['left']), 'right': merge_overlap_seg(motion_dict['right']), 
                     'still': merge_overlap_seg(motion_dict['still'])}

    return motion_dict_filter


def find_motion(motion_dict, start_x, end_x, duration, motion_type, weight=(1,1,1)):
    best_dist = np.inf
    best_motion = None
    for motion in motion_dict[motion_type]:
        dist = abs(start_x - motion['start_x']) * weight[0] + \
                abs(end_x - motion['end_x']) * weight[1] + \
                abs(duration - motion['duration']) * weight[2]
        if dist < best_dist:
            best_dist = dist
            best_motion = motion
    return best_motion


#########################################################################
##### Generate motion without hit label
#########################################################################

def generate_motion_without_hitlabel(sc, video, fid2densepose, motion_dict, hit_traj, out_path):
    '''generate motion by looking up each triple on the query trajectory, and match any possible clip from motion database'''
    # hacky start
    background = load_frame(video, 39050, [])
    video_name = video.item_name()
    # hacky end

    videowriter = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('M','J','P','G'), 8, (video.width, video.height))
    i = 0
    ball_traj = interpolate_trajectory_from_hit(hit_traj)
    while i+2 < len(hit_traj):
        hit_start = hit_traj[i]
        if hit_start['fg'] == 0:
            i += 1
            continue
            
        hit_end = hit_traj[i+2]
        nframe = hit_end['fid'] - hit_start['fid']
        best_dist = float('inf')
        best_match = None
        for motion_traj in motion_dict:
            for fid in sorted(motion_traj):
                if fid + nframe in motion_traj:
                    hit_median = median(hit_start['hit'], hit_end['hit'])
                    motion_median = median(motion_traj[fid]['RWrist'], motion_traj[fid+nframe]['RWrist'])
                    shift = (hit_median[0] - motion_median[0], 0)
    #                 shift = minus(hit_median, motion_median)
                    d = L2(hit_start['hit'], add(motion_traj[fid]['RWrist'], shift)) \
                        + L2(hit_end['hit'], add(motion_traj[fid+nframe]['RWrist'], shift))
                    # filter still clips
    #                 if L2(motion_traj[fid]['LAnkle'], motion_traj[fid+nframe]['LAnkle']) < 100 or \
    #                     L2(motion_traj[fid]['RAnkle'], motion_traj[fid+nframe]['RAnkle']) < 100:
    #                     continue
                    if d < best_dist:
                            best_dist = d
                            best_match = {'sfid':fid, 'efid':fid + nframe, 'shift':shift}
        i += 2
        print(best_dist, best_match)
        offset = hit_start['fid'] - hit_traj[0]['fid']
        for idx, source_fid in enumerate(range(best_match['sfid'], best_match['efid']+1)):
            target_frame = background.copy()
            source_frame = load_frame(video, source_fid, [])    
            source_mask = mask_util.decode(fid2densepose[source_fid]['segms'])[..., 0]
            # with shift
            source_frame = np.roll(source_frame, best_match['shift'], axis=(1, 0))
            source_mask = np.roll(source_mask, best_match['shift'], axis=(1, 0))
            # with shift
            target_frame[source_mask == 1] = source_frame[source_mask == 1]
            cv2.circle(target_frame, ball_traj[offset+idx]['pt'], 12, (255, 255, 255), -1)
            if idx == 0 or idx == nframe:
                cv2.circle(target_frame, ball_traj[offset+idx]['pt'], 12, (0, 0, 255), -1)
            videowriter.write(target_frame)
    videowriter.release()


#########################################################################
##### Generate motion with hit label
#########################################################################

def render_motion(sc, video, query2result, out_path):
    # hacky start
    background = load_frame(video, 39050, [])
    video_name = video.item_name()
    # hacky end
    videowriter = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('M','J','P','G'), 8, (video.width, video.height))

    for entry in query2result:
        hit_start, hit_med, hit_end = entry['query']
        motion_start, motion_med, motion_end = entry['result']
        shift_start = entry['shift']

        nframe = hit_end['fid'] - hit_start['fid']
        ball_traj = interpolate_trajectory_from_hit(list(entry['query']))
        # for interpolation
        time_step = 1. * (motion_end['fid'] - motion_start['fid']) / nframe
        shift_step = 1. * (hit_end['pos'][0] - motion_end['pos'][0] - shift_start[0]) / nframe
        
        for idx in range(nframe + 1):
            target_frame = background.copy()
            
            # interpolate motion to match hit
            source_fid = motion_start['fid'] + int(np.round(idx * time_step))
            source_frame = load_frame(video, source_fid, []) 

            # load mask from maskrcnn database
            mask_fg, mask_bg = get_maskrcnn_by_fid(sc, video_name, source_fid)
            source_mask = mask_util.decode(mask_fg)

            # load player mask with shift
            shift = add(shift_start, (int(idx * shift_step), 0))
            source_frame = np.roll(source_frame, shift, axis=(1, 0))
            source_mask = np.roll(source_mask, shift, axis=(1, 0))
            
            # stitch player to background
            target_frame[source_mask == 1] = source_frame[source_mask == 1]
            
            # draw ball
            cv2.circle(target_frame, ball_traj[idx]['pos'], 12, (255, 255, 255), -1)
            if idx == 0 or idx == nframe:
                cv2.circle(target_frame, ball_traj[idx]['pos'], 12, (0, 0, 255), -1)
            
            videowriter.write(target_frame)
    videowriter.release()


def generate_motion_local(sc, video, motion_dict, hit_traj, out_path):
    '''generate motion by looking up single triangle in hit trajectory'''

    def evaluate(query, result, shift):
        (hit_start, hit_med, hit_end) = query
        (motion_start, motion_med, motion_end) = result
        d = L2(hit_start['pos'], add(motion_start['pos'], shift)) \
                    + L2(hit_med['pos'], add(motion_med['pos'], shift)) \
                    + L2(hit_end['pos'], add(motion_end['pos'], shift))
        return d

    query2result = []
    i = 0
    while i+2 < len(hit_traj):
        hit_start = hit_traj[i]
        if not hit_start['fg']:
            i += 1
            continue
        hit_med = hit_traj[i + 1]
        hit_end = hit_traj[i + 2]
        nframe = hit_end['fid'] - hit_start['fid']
        query = (hit_start, hit_med, hit_end)

        best_dist = float('inf')
        best_match = None
        for motion_traj in motion_dict:
            for j, motion_start in enumerate(motion_traj):
                if not motion_start['fg']:
                    continue
                if j + 2 >= len(motion_traj):
                    break
                motion_med = motion_traj[j + 1]
                motion_end = motion_traj[j + 2]
                result = (motion_start, motion_med, motion_end)
                if motion_start['pos'] is None or motion_med['pos'] is None or motion_end['pos'] is None:
                    continue
                
                shift = (hit_start['pos'][0] - motion_start['pos'][0], 0)
                d = evaluate(query, result, shift)
                if d < best_dist:
                        best_dist = d
                        best_match = {'query': query, 'result': result, 'shift':shift}
        query2result += [best_match]
        print("best_match_distance", best_dist)
        print("num frames of query", nframe)
        print("num frames of result", best_match['result'][2]['fid'] - best_match['result'][0]['fid'])
        i += 2

    render_motion(sc, video, query2result, out_path)  


def generate_motion_global(sc, video, motion_dict, hit_traj, out_path):
    '''generate motion by looking up whole query'''
    POSE_WEIGHT = 5
    def evaluate(query, result, shift, last_pose_end, pose_start):
        (hit_start, hit_med, hit_end) = query
        (motion_start, motion_med, motion_end) = result
        d = L2(hit_start['pos'], add(motion_start['pos'], shift)) \
                    + L2(hit_med['pos'], add(motion_med['pos'], shift)) \
                    + L2(hit_end['pos'], add(motion_end['pos'], shift))
        if not last_pose_end is None and not pose_start is None:
            d += get_openpose_dist(last_pose_end, pose_start, size=(video.width, video.height)) * POSE_WEIGHT
        return d

    query2result = []
    i = 0
    last_pose_end = None
    while i+2 < len(hit_traj):
        hit_start = hit_traj[i]
        if not hit_start['fg']:
            i += 1
            continue
        hit_med = hit_traj[i + 1]
        hit_end = hit_traj[i + 2]
        nframe = hit_end['fid'] - hit_start['fid']
        query = (hit_start, hit_med, hit_end)

        best_dist = float('inf')
        best_match = None
        for motion_traj in motion_dict:
            for j, motion_start in enumerate(motion_traj):
                if not motion_start['fg']:
                    continue
                if j + 2 >= len(motion_traj):
                    break
                motion_med = motion_traj[j + 1]
                motion_end = motion_traj[j + 2]
                result = (motion_start, motion_med, motion_end)
                if motion_start['pos'] is None or motion_med['pos'] is None or motion_end['pos'] is None:
                    continue
                # load motion from openpose
                pose_fg, _ = get_openpose_by_fid(video, motion_start['fid'])
                shift = (hit_start['pos'][0] - motion_start['pos'][0], 0)
                d = evaluate(query, result, shift, last_pose_end, pose_fg)
                if d < best_dist:
                        best_dist = d
                        best_match = {'query': query, 'result': result, 'shift':shift}
        
        query2result += [best_match]
        pose_fg, _ = get_openpose_by_fid(video, best_match['result'][2]['fid'])
        last_pose_end = pose_fg
        print("best_match_distance", best_dist)
        print("num frames of query", nframe)
        print("num frames of result", best_match['result'][2]['fid'] - best_match['result'][0]['fid'])
        i += 2

    render_motion(sc, video, query2result, out_path)    