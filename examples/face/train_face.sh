~/git/caffe-face-caffe-face/build/tools/caffe train \
    --solver=/home/licheng/git/caffe-face-caffe-face/examples/face/face_solver.prototxt \
    --gpu 0,1,2,3 \
    >run_log.out 2>&1
