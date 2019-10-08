import scannerpy
from scannerpy import Client, DeviceType
from scannerpy.storage import NamedVideoStream, NamedStream
from django.db.models import Q
from query.models import Video, Frame, Face, Labeler, Tag, VideoTag, Shot
from esper.prelude import Notifier
from scannertools import densepose_detection

import os
import sys
import math
import numpy as np
import pickle
from tqdm import tqdm
import six.moves.urllib as urllib


def main():

	print("Prepare videos and frames")
	video = Video.objects.filter(path__contains='wim')[0]
	video_path = video.path
	video_name = video.item_name()

	sc = Client()
	stride = 1
	input_stream = NamedVideoStream(sc, video_name, path=video_path)
	frame = sc.io.Input([input_stream])

	# run on all frames
	# running_frame = sc.streams.Stride(frame, [stride])
	
	hit_annotation = pickle.load(open('/app/data/pkl/hit_annotation_tennis.pkl', 'rb'))[video_name + '.mp4']
	# run on selected frames

# 	hit_dict = []
# 	for h in hit_annotation.values():
# 		hit_dict += h
	hit_dict = hit_annotation
    
# 	frame_ids = [i for point in hit_dict for i in range(point[0]['fid']-25, point[-1]['fid']+25) ]
	frame_ids = [i for point in hit_dict for i in range(point[0] - 25, point[-1] + 25) ]

	frame_ids.sort()
	running_frame = sc.streams.Gather(frame, [frame_ids])

	print('Running Scanner DensePose op on %d frames' %(len(frame_ids)))
	densepose_frame = sc.ops.DensePoseDetectPerson(
	    frame=running_frame,
	    device=DeviceType.GPU,
	    batch=1,
	    confidence_threshold=0.5,
        nms_threshold = 0.2
	    )
	densepose_stream = NamedStream(sc, video_name + '_densepose')
	output_op = sc.io.Output(densepose_frame, [densepose_stream])
	sc.run(output_op, scannerpy.common.PerfParams.estimate(), cache_mode=scannerpy.CacheMode.Overwrite)

	exit()

	print('Writing DensePose metadata into frames')
	drawn_frame = sc.ops.DrawDensePose(frame=running_frame,
	                                 bundled_data=sc.io.Input([densepose_stream]),
	                                 min_score_thresh=0.5,
	                                 show_body=True)
	drawn_stream = NamedVideoStream(sc, video_name + '_densepose_draw_uvbody')
	output_op = sc.io.Output(drawn_frame, [drawn_stream])
	sc.run(output_op, scannerpy.common.PerfParams.estimate(), cache_mode=scannerpy.CacheMode.Overwrite)
	drawn_stream.save_mp4('/app/result/' + video_name + '_densepose_uvbody')

	drawn_frame = sc.ops.DrawDensePose(frame=running_frame,
	                                 bundled_data=sc.io.Input([densepose_stream]),
	                                 min_score_thresh=0.5,
	                                 show_body=False)
	drawn_stream = NamedVideoStream(sc, video_name + '_densepose_draw_full')
	output_op = sc.io.Output(drawn_frame, [drawn_stream])
	sc.run(output_op, scannerpy.common.PerfParams.estimate(), cache_mode=scannerpy.CacheMode.Overwrite)
	drawn_stream.save_mp4('/app/result/' + video_name + '_densepose_full')


if __name__ == '__main__':
    main()
