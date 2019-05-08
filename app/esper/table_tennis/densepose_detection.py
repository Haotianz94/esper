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
from tqdm import tqdm
import six.moves.urllib as urllib


def main():

	print("Prepare videos and frames")
	movie_path = '/app/data/videos/sample-clip.mp4'
	movie_name = os.path.splitext(os.path.basename(movie_path))[0]
	# video = Video.objects.filter(path__contains='men_single_final_gold')[0]
	# movie_path = video.path
	# movie_name = os.path.splitext(os.path.basename(movie_path))[0]

	sc = Client()
	stride = 1
	input_stream = NamedVideoStream(sc, movie_name, path=movie_path)
	frame = sc.io.Input([input_stream])
	strided_frame = sc.streams.Stride(frame, [stride])

	print('Running Scanner DensePose op')
	densepose_frame = sc.ops.DensePoseDetectPerson(
	    frame=strided_frame,
	    device=DeviceType.GPU,
	    batch=1,
	    )
	densepose_stream = NamedStream(sc, movie_name + '_densepose')
	output_op = sc.io.Output(densepose_frame, [densepose_stream])
	sc.run(output_op, scannerpy.common.PerfParams.estimate(), cache_mode=scannerpy.CacheMode.Overwrite)


	print('Writing DensePose metadata into frames')
	drawn_frame = sc.ops.DrawDensePose(frame=strided_frame,
	                                 bundled_data=sc.io.Input([densepose_stream]),
	                                 min_score_thresh=0.5,
	                                 show_body=True)
	drawn_stream = NamedVideoStream(sc, movie_name + '_densepose_draw_uvbody')
	output_op = sc.io.Output(drawn_frame, [drawn_stream])
	sc.run(output_op, scannerpy.common.PerfParams.estimate(), cache_mode=scannerpy.CacheMode.Overwrite)
	drawn_stream.save_mp4('/app/result/' + movie_name + '_densepose_uvbody')

	drawn_frame = sc.ops.DrawDensePose(frame=strided_frame,
	                                 bundled_data=sc.io.Input([densepose_stream]),
	                                 min_score_thresh=0.5,
	                                 show_body=False)
	drawn_stream = NamedVideoStream(sc, movie_name + '_densepose_draw_full')
	output_op = sc.io.Output(drawn_frame, [drawn_stream])
	sc.run(output_op, scannerpy.common.PerfParams.estimate(), cache_mode=scannerpy.CacheMode.Overwrite)
	drawn_stream.save_mp4('/app/result/' + movie_name + '_densepose_full')


if __name__ == '__main__':
    main()