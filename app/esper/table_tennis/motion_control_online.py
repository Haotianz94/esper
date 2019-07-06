from esper.table_tennis.utils import *
from esper.table_tennis.pose_utils import * 

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


def interpolate_trajectory_from_hit(motion):
    # hit_traj = [{'fid': motion['fid_start'], 'pos': motion['hit_start']},
    #             {'fid': motion['fid_med'], 'pos': motion['hit_med']},
    #             {'fid': motion['fid_end'], 'pos': motion['hit_end']}]
    # ball_traj = []
    # start = hit_traj[0]
    # i = 1
    # while i != len(hit_traj):
    #     end = hit_traj[i]
    #     nframe = end['fid'] - start['fid']
    #     step_x = 1. * (end['pos'][0] - start['pos'][0]) / nframe
    #     step_y = 1. * (end['pos'][1] - start['pos'][1]) / nframe 
    #     for idx in range(nframe):
    #         ball_traj.append({'fid' : start['fid']+idx, 
    #                           'pos': (int(np.round(start['pos'][0] + step_x * idx)), 
    #                           int(np.round(start['pos'][1] + step_y * idx)))})
    #     i += 1
    #     start = end
    # ball_traj.append({'fid' : start['fid'], 'pos': start['pos']})
    # return ball_traj

    ball_traj = []
    for fid in range(motion['fid_med'] - motion['fid_start'] + 1):
         ball_traj.append( (motion['hit_start'] + motion['ball_speed_input'] * fid).astype(np.int) )

    for fid in range(1, motion['fid_end'] - motion['fid_med'] + 1):
        ball_traj.append( (motion['hit_med'] + motion['ball_speed_output'] * fid).astype(np.int) )
    return ball_traj

#########################################################################
##### Generate motion with hit label online
#########################################################################
def render_motion(sc, motion_all, out_path, interpolation=False, draw_stick=False):
    # hacky start
    window_size = INTERPOLATION_WINDOW_SIZE
    pix2pix_dir = '/app/result/pix2pixHD/stick2human'
    # background = load_frame(video, 39050, [])
    background = cv2.imread(BACKGROUND_FRAME_PATH)
    # hacky end

    # def load_interpolation(person, hash, fid):
    #     path = '{}/test_B/{}_{}_synthesized_image.jpg'.format(pix2pix_dir, hash, fid)
    #     if not os.path.exists(path):
    #         return None
    #     image = cv2.imread(path)
    #     # assert image is not None, 'Cannot load {}'.format(path)
    #     crop_box = person.get_crop_box(im_size=(video.height, video.width))
    #     if crop_box is None:
    #         return None
    #     frame = np.zeros((video.height, video.width, 3), dtype=np.uint8)
    #     frame[crop_box[1]: crop_box[3], crop_box[0]: crop_box[2]] = image
    #     return frame

    videowriter = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('M','J','P','G'), 25, (FRAME_W, FRAME_H))

    for motion_idx, motion in enumerate(motion_all):
        # hit_start, hit_med, hit_end = entry['query']['hit_start'], entry['query']['hit_med'], entry['query']['hit_end']
        # fid_start, fid_end = entry['result']['motion_start'], entry['result']['motion_end']
        video_name = motion['video_name']
        # shift_start = (hit_start['pos'][0] - motion_start['pos'][0], 0)
        # shift_end = (hit_end['pos'][0] - motion_end['pos'][0], 0)
        # shift = motion['shift']

        if motion_idx == 0:
            nframes = motion['fid_end'] - motion['fid_med']
        else:    
            nframes = motion['fid_end'] - motion['fid_start']
            nframes_med = motion['fid_med'] - motion['fid_start']
            ball_traj = interpolate_trajectory_from_hit(motion)
            ball_start = ball_traj[0]
            ball_med = ball_traj[nframes_med]
            # print(ball_traj[-1])
            # continue
        # shift_step = 1. * (shift_end[0] - shift_start[0]) / nframe

        for fid in range(nframes + 1):
            target_frame = background.copy()
            
            # interpolate motion to match hit
            motion['speed'] = 1
            if motion_idx == 0:
                source_fid = motion['fid_med'] + int(np.round(fid * motion['speed']))
            else:
                source_fid = motion['fid_start'] + int(np.round(fid * motion['speed']))

            # load mask from maskrcnn database
            person_fg, person_bg = get_densepose_by_fid(sc, video_name, source_fid)
            source_mask = mask_util.decode(person_fg.mask)
            # source_mask = ndimage.binary_dilation((source_mask > 0), iterations=30)
            
            if draw_stick:
                source_frame = np.ones_like(background) * 255
                person_fg.draw_keypoint(source_frame) # not work           
            else:
                source_frame = load_frame_by_path('{}/{}.mp4'.format(VIDEO_DIR, video_name), source_fid) 

            # interpolation_done = False
            # if interpolation:
            #     if (entry_idx > 0 and source_fid - motion_start['fid'] < window_size) or \
            #        (entry_idx < len(query2result) - 1 and motion_end['fid'] - source_fid < window_size):
            #         source_frame_interpolation = load_interpolation(person_fg, hash_query(video.id, entry['query']), source_fid)
            #         if not source_frame_interpolation is None:
            #             interpolation_done = True
            #             source_frame = source_frame_interpolation
            #             source_mask = ndimage.binary_dilation((source_mask > 0), iterations=30)

            # load player mask with shift
            # shift = add(shift_start, (int(fid * shift_step), 0))
            shift = motion['shift']
            source_frame = np.roll(source_frame, shift, axis=(1, 0))
            source_mask = np.roll(source_mask, shift, axis=(1, 0))
            
            # stitch player to background
            target_frame[source_mask == 1] = source_frame[source_mask == 1]

            # if interpolation_done:
            #     contours, _ = cv2.findContours(
            #     source_mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            #     cv2.drawContours(target_frame, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)

            # draw ball
            if motion_idx != 0:
                cv2.circle(target_frame, tuple(ball_traj[fid]), 12, (255, 255, 255), -1)
                if fid == 0 or fid == nframes or fid == nframes_med:
                    cv2.circle(target_frame, tuple(ball_traj[fid]), 12, (0, 0, 255), -1)
                if fid < nframes_med:
                    cv2.line(target_frame, tuple(ball_start), tuple(ball_traj[fid]), (0, 255, 0), 5)
                elif fid > nframes_med:
                    cv2.line(target_frame, tuple(ball_med), tuple(ball_traj[fid]), (0, 255, 255), 5)

            videowriter.write(target_frame)
    videowriter.release()


def generate_motion_online(sc, motion_dict_raw, hit_candidates, out_path=None, interpolation=False, draw_stick=False, num_hits=10):
    random.seed(0)
    # preprocess motion_dict
    motion_dict = []
    for video_name, motion_dict_sub in motion_dict_raw.items():
        for point in motion_dict_sub:
            for j, motion_start in enumerate(point):
                if j == 0 or j + 2 >= len(point) - 1 or motion_start['fg']:
                    continue
                motion_med = point[j + 1]
                motion_end = point[j + 2]
                if motion_start['pos'] is None or motion_med['pos'] is None or motion_end['pos'] is None:
                    continue
                pose_start, _ = get_densepose_by_fid(sc, video_name, motion_start['fid'])
                pose_end, _ = get_densepose_by_fid(sc, video_name, motion_end['fid'])
                motion_clip = {'video_name': video_name,
                          'hit_start': np.array(motion_start['pos']), 
                          'hit_med': np.array(motion_med['pos']), 
                          'hit_end': np.array(motion_end['pos']),
                          'fid_start': motion_start['fid'],
                          'fid_med': motion_med['fid'],
                          'fid_end': motion_end['fid'],
                          'pose_start': pose_start,
                          'pose_end': pose_end
                          }
                motion_dict.append(motion_clip)

    POSE_WEIGHT = 1#10
    SHIFT_WEIGHT = 5
    TIME_WEIGHT = 0#5
    def evaluate(motion_last, motion_next, hit):
        # spatial location with shift
        pose_last_center = motion_last['pose_end'].get_body_center()
        pose_next_center = motion_next['pose_start'].get_body_center()
        shift = np.array([pose_last_center[0] - pose_next_center[0], 0])
        d = L2(motion_next['hit_med'] + shift, hit['pos'])

        # similiar pose transition
        d += motion_last['pose_end'].get_pose_dist(motion_next['pose_start'], shift) * POSE_WEIGHT

        # timing difference
        d += np.abs(((motion_next['fid_med'] - motion_next['fid_start']) - (hit['nframes']))) * TIME_WEIGHT
        return d

    def search_hit(motion_last):
        def is_on_table(pos):
            pos = (pos[0] / FRAME_W, pos[1] / FRAME_H)
            return (pos[0] > 0.38 and pos[0] < 0.63 and pos[1] > 0.45 and pos[1] < 0.62)
        ball_start = np.array(motion_last['hit_end'])
        attempt = 0
        seg_num = 10
        while attempt < len(hit_candidates):
            hit = random.choice(hit_candidates)
            if np.linalg.norm(motion_last['hit_med'] - np.array(hit['pos'])) < 50:
                continue
            ball_end = np.array(hit['pos'])
            for idx in range(0, seg_num+1):
                pos = ball_start + 1. * (ball_end - ball_start) * idx / seg_num
                if is_on_table(pos):
                    return hit
            attempt += 1
        raise Exception("Can not find good hit")


    def search_motion(motion_last, hit):
        # get ball trajectory by interpolation
        # ball_traj = interpolate_trajectory_from_hit(motion_last['hit_next'], hit['pos'], hit['nframes'], extrapolation=True)
        best_dist = 1e9
        best_match = None
        for motion_next in motion_dict:
            dist = evaluate(motion_last, motion_next, hit)
            if dist < best_dist:
                best_dist = dist
                best_match = motion_next
        print('best_match: ', best_match['fid_start'], hit, best_dist)
        return best_match        

    # find a serving clip
    motion_all = []
    video_name = 'Tabletennis_2012_Olympics_men_single_semi_final_2'
    hit_med = motion_dict_raw[video_name][3][0]
    hit_end = motion_dict_raw[video_name][3][1]
    pose_end, _ = get_densepose_by_fid(sc, video_name, hit_end['fid'])
    motion_first = {'video_name': video_name,
                    'fid_start': None, 'fid_med': hit_med['fid'], 'fid_end': hit_end['fid'],
                    'hit_start': None, 'hit_med': hit_med['pos'], 'hit_end': hit_end['pos'],
                    'pose_start': None, 'pose_end': pose_end,
                    'shift': (0, 0), 'speed': 1}
    motion_all.append(motion_first)

    # loop to search clips
    hit_idx = 0
    while hit_idx < num_hits: 
        motion_last = motion_all[-1]
        hit = search_hit(motion_last)
        motion_next = search_motion(motion_last, hit)

        pose_last_center = motion_all[-1]['pose_end'].get_body_center()
        pose_next_center = motion_next['pose_start'].get_body_center()
        motion_next['shift'] = np.array([pose_last_center[0] - pose_next_center[0], 0])
        motion_next['speed'] = 1. * (motion_next['fid_med'] - motion_next['fid_start'] + 1) / hit['nframes']

        motion_next['hit_start'] = motion_last['hit_end']
        motion_next['hit_end'] = motion_next['hit_end'] + motion_next['shift']

        ## only consider ball speed
        # motion_next['ball_speed_input'] = 1. * (np.array(hit['pos']) - np.array(motion_next['hit_start'])) / hit['nframes']
        # motion_next['hit_med'] = motion_next['hit_start'] + (motion_next['fid_med'] - motion_next['fid_start']) * motion_next['ball_speed_input']
        ## only consider ball hit location
        motion_next['hit_med'] = np.array(hit['pos'])
        motion_next['ball_speed_input'] = 1. * (motion_next['hit_med'] - motion_next['hit_start']) / (motion_next['fid_med'] - motion_next['fid_start'])

        motion_next['ball_speed_output'] = 1. * (motion_next['hit_end'] - motion_next['hit_med']) / (motion_next['fid_end'] - motion_next['fid_med'])
        motion_all.append(motion_next)
        hit_idx += 1

    # call render function
    render_motion(sc, motion_all, out_path, interpolation, draw_stick)
    