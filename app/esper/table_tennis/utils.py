from django.db.models import F, Q
from operator import itemgetter, attrgetter

# import query sets
from query.models import Video, Face, Pose

# import esper utils
from esper.prelude import *
# from esper.supercut import *

from rekall.stdlib.merge_ops import payload_plus
from rekall.predicates import payload_satisfies, length_exactly
from rekall.bounds import Bounds1D, Bounds3D

from rekall import Interval, IntervalSet, IntervalSetMapping
from vgrid import VGridSpec, VideoMetadata, VideoBlockFormat, LabelState
from vgrid_jupyter import VGridWidget

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


def create_video_supercut(intervals, out_path):
    stitch_video_temporal(intervals, out_path)


def create_video_montage(intervals, out_path, width=1920, num_cols=10, aspect_ratio=16./9, align=False, decrease_volume=10):
    video_path = '/app/result/pose_montage.avi'
    audio_path = '/app/result/pose_montage.wav'
    stitch_video_spatial(intervals, out_path=video_path, align=align, 
                     width=width, num_cols=num_cols, target_height = int(1. * width / num_cols / aspect_ratio))
    mix_audio(intervals_selected, out_path=audio_path, decrease_volume=decrease_volume, align=align)
    concat_video_audio(video_path, audio_path, out_path)


# ============== Rekall help functions ============== 

def count_intervals(ism):
    return sum([intervalSet.size() for intervalSet in ism.get_grouped_intervals().values() ])


def count_duration(ism):
    return sum([intervalSet.duration() for intervalSet in ism.get_grouped_intervals().values() ])


def list_to_IntervalSetMapping(interval_list):
    ism = {}
    for video_id, start, end, duration in interval_list:
        if not video_id in ism:
            ism[video_id] = []
        ism[video_id].append(Interval(Bounds3D(start, end)))
    return IntervalSetMapping({video_id: IntervalSet(intervalSet) for video_id, intervalSet in ism.items()}) 


# def intrvlcol2list(intrvlcol, with_duration=True, sort_by_duration=False):
#     interval_list = []
#     if type(intrvlcol) == IntervalList:
#         intrvllist = intrvlcol
#         if with_duration:
#             video = Video.objects.filter(id=video_id)[0]
#         for i in intrvllist.get_intervals():
#             if with_duration:
#                 interval_list.append((video_id, i.start, i.end, (i.end - i.start) / video.fps))
#             else:
#                 interval_list.append((video_id, i.start, i.end))
#     else:            
#         for video_id, intrvllist in intrvlcol.get_allintervals().items():
#             video = Video.objects.filter(id=video_id)[0]
#             for i in intrvllist.get_intervals():
#                 if i.start > video.num_frames:
#                     continue
#                 if with_duration:
#                     interval_list.append((video_id, i.start, i.end, (i.end - i.start) / video.fps))
#                 else:
#                     interval_list.append((video_id, i.start, i.end))
#     if sort_by_duration and with_duration:
#         interval_list.sort(key=lambda i: i[-1])
#         interval_list = interval_list[::-1]

#     print("Get {} intervals from interval collection".format(len(interval_list)))
#     return interval_list


def IntervalSetMapping_to_vgrid(ism, flat=False):
    video_meta = []
    if flat:
        video_meta_id = 0
        ism_final = {}
        for video_id, intervalSet in ism.items():
            video = Video.objects.filter(id=video_id)[0] 
            for i in intervalSet.get_intervals():
                ism_final[video_meta_id] = IntervalSet([i])
                video_meta.append(VideoMetadata(path=video.path, id=video_meta_id))
                video_meta_id += 1
    else:
        for video_id, intervalSet in ism.items():
            video = Video.objects.filter(id=video_id)[0] 
            video_meta.append(VideoMetadata(path=video.path, id=video_id))
        ism_final = ism

    vgrid_spec = VGridSpec(
      video_meta=video_meta,
      video_endpoint="http://sora.stanford.edu/system_media/",
      vis_format=VideoBlockFormat([('test', ism_final)]),
      show_timeline=True)

    return VGridWidget(vgrid_spec=vgrid_spec.to_json())


def IntervalSetMapping_frame_to_second(ism):
    intervalSets_second = {}
    for video_id, intervalSet in ism.get_grouped_intervals().items():
        video = Video.objects.filter(id=video_id)[0]
        fps = video.fps
        intervalSets_second[video_id] = IntervalSet( 
            [Interval(Bounds1D(i.bounds['t1'] / fps, i.bounds['t2'] / fps), i.payload) \
            for i in intervalSet.get_intervals()] )
    return IntervalSetMapping(intervalSets_second)


def IntervalSetMapping_second_to_frame(ism):
    intervalSets_frame = {}
    for video_id, intervalSet in ism.get_grouped_intervals().items():
        video = Video.objects.filter(id=video_id)[0]
        fps = video.fps
        intervalSets_frame[video_id] = IntervalSet(
            [Interval(Bounds1D(int(i.bounds['t1'] * fps), int(i.bounds['t2'] * fps)), i.payload) \
            for i in intervalSet.get_intervals()] )
    return IntervalSetMapping(intervalSets_frame)
