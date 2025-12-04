ffmpeg -r 30 -f image2 -s 640x480 -i ./fig_out/%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ./iris_vid.mp4
