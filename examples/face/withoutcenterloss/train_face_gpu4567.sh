~/git/caffe-face-caffe-face/build/tools/caffe train \
    --solver=/home/licheng/git/caffe-face-caffe-face/examples/face/face_solver_gpu4567_without_center_loss.prototxt \
    --gpu 4,5,6,7 \
    >run_log_without_center_loss_gpu4567.out 2>&1
