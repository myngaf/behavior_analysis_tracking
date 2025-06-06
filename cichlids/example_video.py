from video_analysis_toolbox.video import Video
import cv2

path = r"J:\_Projects\Cichlid_Group\Duncan\prey_capture_experiments\a_burtoni\videos\2020_09_09\fish07\15-31-15.avi"
v = Video.open(path)
# v.scroll(first_frame=350, last_frame=2850)

frames = v.return_frames(350, 2850)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
output = r"D:\DATA\cichlids\l_attenuatus\plots\burtoni_example.avi"
writer = cv2.VideoWriter(output, fourcc, 80., frames[0].shape[::-1], False)
for frame in frames:
    writer.write(frame)
writer.release()
