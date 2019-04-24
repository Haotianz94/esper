import scannerpy
from scannerpy import Client, DeviceType
from scannerpy.storage import NamedVideoStream, NamedStream
from scannertools import maskrcnn_detection
from django.db.models import Q
from query.models import Video, Frame, Face, Labeler, Tag, VideoTag, Shot
from esper.prelude import Notifier

import os
import sys
import math
import numpy as np
from tqdm import tqdm
import six.moves.urllib as urllib


def main():

	# movie_path = '/app/data/videos/sample-clip.mp4'
	# movie_name = os.path.splitext(os.path.basename(movie_path))[0]
	print("Prepare videos and frames")
	video = Video.objects.filter(path__contains='Tabletennis').order_by('id')[0]
	movie_path = video.path
	movie_name = os.path.splitext(os.path.basename(movie_path))[0]

	sc = Client()

	stride = 1
	input_stream = NamedVideoStream(sc, movie_name, path=movie_path)
	frame = sc.io.Input([input_stream])
	strided_frame = sc.streams.Stride(frame, [stride])

	object_frame = sc.ops.MaskRCNNDetectObjects(
	    frame=strided_frame,
	    device=DeviceType.GPU, #if sc.has_gpu() else DeviceType.CPU,
	    batch=8,
	    # min_score_thresh = 0.5,
	    # min_image_size = 800,
	    )

	segmentation_stream = NamedStream(sc, movie_name + '_segment')
	output_op = sc.io.Output(object_frame, [segmentation_stream])
	sc.run(output_op, scannerpy.common.PerfParams.estimate(), cache_mode=scannerpy.CacheMode.Overwrite)

	exit()

	print('Extracting data from Scanner output...')

	# bundled_data_list is a list of bundled_data
	# bundled data format: [box position(x1 y1 x2 y2), box class, box score]
	seq = sc.sequence(segmentation_stream._name)
	for i, obj in tqdm(enumerate(seq.load(workers=1))):	
		print(i, len(obj))
	#bundled_data_list = list(tqdm())
	print('Successfully extracted data from Scanner output!')

	# run non-maximum suppression
	# bundled_np_list = bundled_data_list
	# bundled_np_list = instance_segmentation.nms_bulk(bundled_data_list)
	# bundled_np_list = instance_segmentation.smooth_box(bundled_np_list, min_score_thresh=0.5)

	print('Writing frames to {:s}_instance_segment.mp4'.format(movie_name))

	frame = sc.io.Input([input_stream])
	# bundled_data = sc.io.Input([PythonStream(bundled_np_list)])
	strided_frame = sc.streams.Stride(frame, [stride])
	drawn_frame = sc.ops.TorchDrawBoxes(frame=strided_frame,
	                                 bundled_data=segmentation_stream,
	                                 min_score_thresh=0.5)
	drawn_stream = NamedVideoStream(sc, movie_name + '_drawn_frames')
	output_op = sc.io.Output(drawn_frame, [drawn_stream])
	sc.run(output_op, scannerpy.common.PerfParams.estimate(), cache_mode=scannerpy.CacheMode.Ignore)

	drawn_stream.save_mp4(movie_name + '_instance_segment')

	input_stream.delete(sc)
	detect_stream.delete(sc)
	drawn_stream.delete(sc)

	print('Successfully generated {:s}_instance_segment.mp4'.format(movie_name))


if __name__ == '__main__':
    main()