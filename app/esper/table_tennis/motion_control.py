from esper.table_tennis.utils import *
from esper.table_tennis.pose_utils import * 
from esper.table_tennis.search import *
import pycocotools.mask as mask_util
from detectron.utils.vis import vis_keypoints

import numpy as np
import cv2
import pickle
from scipy import ndimage


INTERPOLATION_WINDOW_SIZE = 6
BACKGROUND_FRAME_PATH = '/app/data/tabletennis_background.jpg'
FRAME_H, FRAME_W = 1080, 1920
VIDEO_DIR = '/app/data/videos'


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
##### Generate motion with hit label using triangle query offline
#########################################################################

# Not upgraded
def generate_motion_local(sc, video, motion_dict, hit_traj, out_path):
    '''generate motion by looking up single triangle in hit trajectory'''

    def evaluate(query, result, shift):
        (hit_start, hit_med, hit_end) = query
        (motion_start, motion_med, motion_end) = result
        (shift_start, _, shift_end) = shift 
        d = L2(hit_start['pos'], add(motion_start['pos'], shift_start)) \
            + L2(hit_med['pos'], add(motion_med['pos'], shift_start)) \
            + L2(hit_end['pos'], add(motion_end['pos'], shift_start))
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
                shift_start = (hit_start['pos'][0] - motion_start['pos'][0], 0)
                shift_end = (hit_end['pos'][0] - motion_end['pos'][0], 0)
                shift = (shift_start, None, shift_end) 
                
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


# Not upgraded
def generate_motion_global(sc, video, motion_dict, hit_traj, out_path):
    '''generate motion by looking up whole query'''
    POSE_WEIGHT = 5
    SHIFT_WEIGHT = 5
    TIME_WEIGHT = 5
    def evaluate(query, result, last_pose_end=None, pose_start=None):
        (hit_start, hit_med, hit_end) = query
        (motion_start, motion_med, motion_end) = result
        shift_start = (hit_start['pos'][0] - motion_start['pos'][0], 0)
        shift_end = (hit_end['pos'][0] - motion_end['pos'][0], 0)
        # penalize shifted hit position distance
        d = L2(hit_start['pos'], add(motion_start['pos'], shift_start)) \
            + L2(hit_med['pos'], add(motion_med['pos'], shift_start)) \
            + L2(hit_end['pos'], add(motion_end['pos'], shift_start))
        # penalize sliding shift
        # d += L2(shift_start, shift_end) * SHIFT_WEIGHT
        # penalize timing difference
        # d += np.abs(((hit_end['fid'] - hit_start['fid']) - (motion_end['fid'] - motion_start['fid']))) * TIME_WEIGHT
        if not last_pose_end is None and not pose_start is None:
            # consider different shift of pose
            d += get_openpose_dist(last_pose_end[0], pose_start, size=(video.width, video.height), shift=minus(last_pose_end[1], shift_start)) * POSE_WEIGHT
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
        # dist_candidates = []
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
                shift_start = (hit_start['pos'][0] - motion_start['pos'][0], 0)
                shift_end = (hit_end['pos'][0] - motion_end['pos'][0], 0)
                shift = (shift_start, None, shift_end)    
                # load motion from openpose
                pose_fg, _ = get_openpose_by_fid(video, motion_start['fid'])
                d = evaluate(query, result, last_pose_end, pose_fg)
                if d < best_dist:
                        best_dist = d
                        best_match = {'query': query, 'result': result, 'shift':shift}
                # dist_candidates.append(d)
        # dist_candidates.sort()
        # print(dist_candidates)
        
        query2result += [best_match]
        pose_fg, _ = get_openpose_by_fid(video, best_match['result'][2]['fid'])
        last_pose_end = (pose_fg, best_match['shift'][2])
        print("best_match_distance", best_dist)
        print("num frames of query", nframe)
        print("num frames of result", best_match['result'][2]['fid'] - best_match['result'][0]['fid'])
        i += 2

    render_motion(sc, video, query2result, out_path)    


def generate_motion_dijkstra(sc, motion_dict, hit_traj, out_path=None, interpolation=False, draw_stick=False):
    '''generate motion by looking up whole query'''
    POSE_WEIGHT = 5
    SHIFT_WEIGHT = 5
    TIME_WEIGHT = 10
    def evaluate(query, result, last_pose_end=None):
        # hit_start, hit_med, hit_end = query
        # (motion_start, motion_med, motion_end) = result
        shift_start = (query['hit_start']['pos'][0] - result['motion_start']['pos'][0], 0)
        shift_end = (query['hit_end']['pos'][0] - result['motion_end']['pos'][0], 0)
        # penalize shifted hit position distance
        d = L2(query['hit_start']['pos'], add(result['motion_start']['pos'], shift_start)) \
            + L2(query['hit_med']['pos'], add(result['motion_med']['pos'], shift_start)) \
            + L2(query['hit_end']['pos'], add(result['motion_end']['pos'], shift_start))
        # penalize sliding shift
        # d += L2(shift_start, shift_end) * SHIFT_WEIGHT
        # penalize timing difference
        # d += np.abs(((hit_end['fid'] - hit_start['fid']) - (motion_end['fid'] - motion_start['fid']))) * TIME_WEIGHT
        if not last_pose_end is None and not result['pose_start'] is None:
            # align two pose by the right wrist in X
            shift = pose_start._format_keypoints()[Pose.RWrist] - last_pose_end._format_keypoints()[Pose.RWrist]
            d += get_openpose_dist(last_pose_end, result['pose_start'], size=(FRAME_W, FRAME_H), shift=(int(shift[0]*FRAME_W), 0)) * POSE_WEIGHT
        return d


    # build graph
    graph = Graph()
    query_list = []
    result_list = []
    # collect query
    i = 0
    while i+2 < len(hit_traj):
        hit_start = hit_traj[i]
        if not hit_start['fg']:
            i += 1
            continue
        hit_med = hit_traj[i + 1]
        hit_end = hit_traj[i + 2]
        nframe = hit_end['fid'] - hit_start['fid']
        query = {'hit_start': hit_start, 'hit_med': hit_med, 'hit_end': hit_end}
        query_list.append(query)
        i += 2

    # collect motion (should move out of this)
    for video_name, motion_dict_sub in motion_dict.items():
        for motion_traj in motion_dict_sub:
            for j, motion_start in enumerate(motion_traj):
                if j == 0 or not motion_start['fg']:
                    continue
                if j + 2 >= len(motion_traj):
                    break
                motion_med = motion_traj[j + 1]
                motion_end = motion_traj[j + 2]
                if motion_start['pos'] is None or motion_med['pos'] is None or motion_end['pos'] is None:
                    continue
                pose_start, _ = get_openpose_by_fid(video_name, motion_start['fid'])
                pose_end, _ = get_openpose_by_fid(video_name, motion_end['fid'])
                result = {'video_name': video_name,
                          'motion_start': motion_start, 
                          'motion_med': motion_med, 
                          'motion_end': motion_end,
                          'pose_start': pose_start,
                          'pose_end': pose_end
                          }
                result_list.append(result)

    graph.add_node('S')
    for query_idx, query in enumerate(query_list):
        # (hit_start, hit_med, hit_end) = query
        for result_idx, result in enumerate(result_list):
            # (motion_start, motion_med, motion_end) = result
            # shift_end = (hit_end['pos'][0] - motion_end['pos'][0], 0)
            # shift = (shift_start, None, shift_end) 
            graph.add_node((query_idx, result_idx))
            if query_idx == 0:
                graph.add_edge('S', (query_idx, result_idx), evaluate(query, result))
            else:
                for result_idx_prev, result_prev in enumerate(result_list):
                    graph.add_edge((query_idx-1, result_idx_prev), (query_idx, result_idx), evaluate(query, result, result_prev['pose_end']))        
    graph.add_node('E')
    for result_idx, result in enumerate(result_list):
        graph.add_edge((query_idx, result_idx), 'E', 0)

    distances, shortest_path = dijkstra(graph, 'S', 'E')
    query2result = []
    for idx, (query_idx, result_idx) in enumerate(shortest_path[1 : -1]):
        # print(query_idx)
        query2result += [{'query': query_list[query_idx], 'result': result_list[result_idx]}]
        print("best_match_distance", distances[idx])

    if not interpolation:
        render_motion(sc, query2result, out_path, interpolation, draw_stick)
    else:
        interpolate_motion(sc, query2result)
        return query2result


def render_motion(sc, query2result, out_path, interpolation=False, draw_stick=False):
    # hacky start
    window_size = INTERPOLATION_WINDOW_SIZE
    pix2pix_dir = '/app/result/pix2pixHD/stick2human'
    # background = load_frame(video, 39050, [])
    background = cv2.imread(BACKGROUND_FRAME_PATH)
    # hacky end

    def load_interpolation(person, hash, fid):
        path = '{}/test_B/{}_{}_synthesized_image.jpg'.format(pix2pix_dir, hash, fid)
        if not os.path.exists(path):
            return None
        image = cv2.imread(path)
        # assert image is not None, 'Cannot load {}'.format(path)
        crop_box = person.get_crop_box(im_size=(video.height, video.width))
        if crop_box is None:
            return None
        frame = np.zeros((video.height, video.width, 3), dtype=np.uint8)
        frame[crop_box[1]: crop_box[3], crop_box[0]: crop_box[2]] = image
        return frame

    videowriter = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('M','J','P','G'), 25, (FRAME_W, FRAME_H))

    for entry_idx, entry in enumerate(query2result):
        hit_start, hit_med, hit_end = entry['query']['hit_start'], entry['query']['hit_med'], entry['query']['hit_end']
        motion_start, motion_end = entry['result']['motion_start'], entry['result']['motion_end']
        video_name = entry['result']['video_name']
        shift_start = (hit_start['pos'][0] - motion_start['pos'][0], 0)
        shift_end = (hit_end['pos'][0] - motion_end['pos'][0], 0)

        nframe = hit_end['fid'] - hit_start['fid']
        ball_traj = interpolate_trajectory_from_hit([hit_start, hit_med, hit_end])

        time_step = 1. * (motion_end['fid'] - motion_start['fid']) / nframe
        shift_step = 1. * (shift_end[0] - shift_start[0]) / nframe
        
        for fid in range(nframe + 1):
            target_frame = background.copy()
            
            # interpolate motion to match hit
            source_fid = motion_start['fid'] + int(np.round(fid * time_step))

            # load mask from maskrcnn database
            person_fg, person_bg = get_densepose_by_fid(sc, video_name, source_fid)
            source_mask = mask_util.decode(person_fg.mask)
            # source_mask = ndimage.binary_dilation((source_mask > 0), iterations=30)
            
            if draw_stick:
                source_frame = np.ones_like(background) * 255
                visualize_densepose_stick(source_frame, person_fg.keyp, (0, 255, 0))           
            else:
                source_frame = load_frame_by_path('{}/{}.mp4'.format(VIDEO_DIR, video_name), source_fid) 

            interpolation_done = False
            if interpolation:
                if (entry_idx > 0 and source_fid - motion_start['fid'] < window_size) or \
                   (entry_idx < len(query2result) - 1 and motion_end['fid'] - source_fid < window_size):
                    source_frame_interpolation = load_interpolation(person_fg, hash_query(video.id, entry['query']), source_fid)
                    if not source_frame_interpolation is None:
                        interpolation_done = True
                        source_frame = source_frame_interpolation
                        source_mask = ndimage.binary_dilation((source_mask > 0), iterations=30)

            # load player mask with shift
            shift = add(shift_start, (int(fid * shift_step), 0))
            source_frame = np.roll(source_frame, shift, axis=(1, 0))
            source_mask = np.roll(source_mask, shift, axis=(1, 0))
            
            # stitch player to background
            target_frame[source_mask == 1] = source_frame[source_mask == 1]

            if interpolation_done:
                contours, _ = cv2.findContours(
                source_mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(target_frame, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)
            
            # draw ball
            cv2.circle(target_frame, ball_traj[fid]['pos'], 12, (255, 255, 255), -1)
            if fid == 0 or fid == nframe:
                cv2.circle(target_frame, ball_traj[fid]['pos'], 12, (0, 0, 255), -1)
            
            videowriter.write(target_frame)
    videowriter.release()


def interpolate_motion(sc, video, query2result):
    # hacky start
    window_size = INTERPOLATION_WINDOW_SIZE
    pix2pix_dir = '/app/result/pix2pixHD/stick2human'
    # hacky end

    for entry_idx, entry in enumerate(query2result):
        if entry_idx == 0:
            continue
        query_prev = query2result[entry_idx-1]['query']
        query_next = entry['query']
        result_prev = query2result[entry_idx-1]['result']
        result_next = entry['result']

        for i in range(window_size):
            weight = 1. * i / window_size + 0.5
            # blend prev
            fid_prev = result_prev['motion_end']['fid'] - i
            person_prev, _ = get_densepose_by_fid(sc, video_name, fid_prev)
            fid_next = result_next['motion_start']['fid'] - i
            person_next, _ = get_densepose_by_fid(sc, video_name, fid_next)
            
            blend_keyp = person_prev.blend_keypoint(person_next, weight, key=Person.RWrist)
            image = np.ones((video.height, video.width, 3), dtype=np.uint8) * 255
            image = vis_keypoints(image, blend_keyp.astype(np.int64), kp_thresh=2, alpha=1)
            # image2 = np.ones((video.height, video.width, 3), dtype=np.uint8) * 255
            # image2 = person_prev.draw_keypoint(image2)
            # image3 = np.ones((video.height, video.width, 3), dtype=np.uint8) * 255
            # image3 = person_next.draw_keypoint(image3)
            # return (image, image2, image3)
            crop_box = person_prev.get_crop_box(im_size=(video.height, video.width))
            path = '{}/test_A/{}_{}.jpg'.format(pix2pix_dir, hash_query(video.id, query_prev), fid_prev)
            # print(path)
            if not crop_box is None:
                cv2.imwrite(path, image[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2], :])

            # blend next
            fid_prev = result_prev['motion_end']['fid'] + i
            person_prev, _ = get_densepose_by_fid(sc, video_name, fid_prev)
            fid_next = result_next['motion_start']['fid'] + i
            person_next, _ = get_densepose_by_fid(sc, video_name, fid_next)

            blend_keyp = person_next.blend_keypoint(person_prev, weight, key=Person.RWrist)
            image = np.ones((video.height, video.width, 3), dtype=np.uint8) * 255
            image = vis_keypoints(image, blend_keyp.astype(np.int64), kp_thresh=0, alpha=1)
            crop_box = person_next.get_crop_box(im_size=(video.height, video.width))
            path = '{}/test_A/{}_{}.jpg'.format(pix2pix_dir, hash_query(video.id, query_next), fid_next)
            # print(path)
            if not crop_box is None:
                cv2.imwrite(path, image[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2], :])


def hash_query(video_id, query):
    return video_id * sum([h['fid'] + h['pos'][0] + h['pos'][1] for h in query])