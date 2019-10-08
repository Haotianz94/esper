import scannerpy
from scannerpy import Client, DeviceType
from scannerpy.storage import NamedVideoStream, NamedStream
from scannertools import maskrcnn_detection
from query.models import Video, Frame, Face, Labeler, Tag, VideoTag, Shot
from esper.prelude import Notifier

import os
import sys
import math
import numpy as np
from tqdm import tqdm
import six.moves.urllib as urllib
from django.db.models import Q


def main():

	print("Prepare videos and frames")
	# video = Video.objects.filter(path__contains='men_single_final_gold')[0]
	# movie_path = video.path
	# movie_name = video.item_name()

	sc = Client()
	stride = 1
	input_stream = NamedVideoStream(sc, movie_name, path=movie_path)
	frame = sc.io.Input([input_stream])
	strided_frame = sc.streams.Stride(frame, [stride])


	print('Running Scanner MaskRCNN op')
	maskrcnn_frame = sc.ops.MaskRCNNDetectObjects(
	    frame=strided_frame,
	    device=DeviceType.GPU, #if sc.has_gpu() else DeviceType.CPU,
	    batch=8,
	    confidence_threshold = 0.5,
	    min_image_size = 800
	    )
	maskrcnn_stream = NamedStream(sc, movie_name + '_maskrcnn')
	output_op = sc.io.Output(maskrcnn_frame, [maskrcnn_stream])
	sc.run(output_op, scannerpy.common.PerfParams.estimate(), cache_mode=scannerpy.CacheMode.Ignore)


	print('Writing MaskRCNN metadata into frames')
	drawn_frame = sc.ops.DrawMaskRCNN(frame=strided_frame,
	                                 bundled_data=sc.io.Input([maskrcnn_stream]),
	                                 min_score_thresh=0.5)
	drawn_stream = NamedVideoStream(sc, movie_name + '_maskrcnn_draw')
	output_op = sc.io.Output(drawn_frame, [drawn_stream])
	sc.run(output_op, scannerpy.common.PerfParams.estimate(), cache_mode=scannerpy.CacheMode.Overwrite)
	drawn_stream.save_mp4('/app/result/' + movie_name + '_maskrcnn')


if __name__ == '__main__':
    main()