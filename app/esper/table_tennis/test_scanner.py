import scannerpy as sp
import scannertools.face_detection

def main():
  # Compute face bounding boxes and draw them on a sample video
  with sp.utils.sample_video() as video_path:
    cl = sp.Client()
    video = sp.NamedVideoStream(cl, 'example', path='/app/data/videos/sample-clip.mp4')
    frames = cl.io.Input([video])
    faces = cl.ops.MTCNNFaceDetect(frame=frames)
    drawn_faces = cl.ops.DrawBboxes(frame=frames, bboxes=faces)
    output_video = sp.NamedVideoStream(cl, 'example_faces')
    output_op = cl.io.Output.drawn_faces([output_video])
    cl.run(output_op, sp.PerfParams.estimate())
    output_video.save_mp4('example_faces')
    # output video is saved to 'example_faces.mp4'

if __name__ == "__main__":
    main()