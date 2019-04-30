# import query sets
from query.models import Video, Face, Pose
from django.db.models import F, Q

# import esper utils
from esper.prelude import *

# import rekall
from esper.rekall import *
from rekall.interval_list import Interval, IntervalList
from rekall.video_interval_collection import VideoIntervalCollection
from rekall.temporal_predicates import *
from rekall.spatial_predicates import *
from rekall.logical_predicates import *
from rekall.parsers import in_array, bbox_payload_parser
from rekall.merge_ops import payload_plus
from rekall.payload_predicates import payload_satisfies
from rekall.list_predicates import length_exactly


# ============== Visualization help functions ============== 

def create_montage_from_images(imgs,
                 output_path=None,
                 width=1600,
                 num_cols=8,
                 target_height=None):
    target_width = int(width / num_cols)
    target_height = target_width * imgs[0].shape[0] // imgs[0].shape[1] 
    num_rows = int(math.ceil(float(len(imgs)) / num_cols))

    montage = np.zeros((num_rows * target_height, width, 3), dtype=np.uint8)
    for row in range(num_rows):
        for col in range(num_cols):
            i = row * num_cols + col
            if i >= len(imgs):
                break
            img = cv2.resize(imgs[i], (target_width, target_height))
            montage[row * target_height:(row + 1) * target_height, col * target_width:(col + 1) *
                    target_width, :] = img
        else:
            continue
        break

    if output_path is not None:
        cv2.imwrite(output_path, montage)
    else:
        return montage


def create_montage_from_intervals(intervals, output_path=None, **kwargs):
    videos, frames = [], []
    id2video = {}
    for i in intervals:
        video_id, sfid, efid = i[:3]
        if not video_id in id2video:
            id2video[video_id] = Video.objects.filter(id=video_id)[0]
    
    for i in intervals:
        video_id, sfid, efid = i[:3]
        videos.append(id2video[video_id])
        frame = (sfid + efid) / 2
        frames.append(int(frame))

    montage = make_montage(videos, frames, **kwargs)
    if not out_path is None:
        cv2.imwrite(out_path, montage)


# ============== Rekall help functions ============== 

def count_intervals(intrvlcol):
    num_intrvl = 0
    for intrvllist in intrvlcol.get_allintervals().values():
        num_intrvl += intrvllist.size()
    return num_intrvl


def count_duration(intrvlcol):
    if type(intrvlcol) == IntervalList:
        intrvllist = intrvlcol
        if intrvllist.size() > 0:
            duration = sum([i.end - i.start for i in intrvllist.get_intervals()])
        else:
            duration = 0
    else:
        if count_intervals(intrvlcol) > 0:
            duration = sum([i.end - i.start for _, intrvllist in intrvlcol.get_allintervals().items() \
                            for i in intrvllist.get_intervals() ])
        else:
            duration = 0
    return duration


def intrvlcol2list(intrvlcol, with_duration=True, sort_by_duration=False):
    interval_list = []
    if type(intrvlcol) == IntervalList:
        intrvllist = intrvlcol
        if with_duration:
            video = Video.objects.filter(id=video_id)[0]
        for i in intrvllist.get_intervals():
            if with_duration:
                interval_list.append((video_id, i.start, i.end, (i.end - i.start) / video.fps))
            else:
                interval_list.append((video_id, i.start, i.end))
    else:            
        for video_id, intrvllist in intrvlcol.get_allintervals().items():
            video = Video.objects.filter(id=video_id)[0]
            for i in intrvllist.get_intervals():
                if i.start > video.num_frames:
                    continue
                if with_duration:
                    interval_list.append((video_id, i.start, i.end, (i.end - i.start) / video.fps))
                else:
                    interval_list.append((video_id, i.start, i.end))
    if sort_by_duration and with_duration:
        interval_list.sort(key=lambda i: i[-1])
        interval_list = interval_list[::-1]

    print("Get {} intervals from interval collection".format(len(interval_list)))
    return interval_list


def intrvlcol2result(intrvlcol, flat=False):
    if not flat:
        return intrvllists_to_result(intrvlcol.get_allintervals())
    else:
        return interval2result(intrvlcol2list(intrvlcol))
    

def interval2result(intervals):
    materialized_result = [
        {'video': video_id,
#             'track': t.id,
         'min_frame': sfid,
         'max_frame': efid }
        for video_id, sfid, efid, duration in intervals ]
    count = len(intervals)
    groups = [{'type': 'flat', 'label': '', 'elements': [r]} for r in materialized_result]
    return {'result': groups, 'count': count, 'type': 'Video'}


def intrvlcol_frame2second(intrvlcol):
    intrvllists_second = {}
    for video_id, intrvllist in intrvlcol.get_allintervals().items():
        video = Video.objects.filter(id=video_id)[0]
        fps = video.fps
        intrvllists_second[video_id] = IntervalList([(i.start / fps, i.end / fps, i.payload) \
                                                  for i in intrvllist.get_intervals()] )
    return VideoIntervalCollection(intrvllists_second)


def intrvlcol_second2frame(intrvlcol):
    intrvllists_frame = {}
    for video_id, intrvllist in intrvlcol.get_allintervals().items():
        video = Video.objects.filter(id=video_id)[0]
        fps = video.fps
        intrvllists_frame[video_id] = IntervalList([(int(i.start * fps), int(i.end * fps), i.payload) \
                                                  for i in intrvllist.get_intervals()] )
    return VideoIntervalCollection(intrvllists_frame)