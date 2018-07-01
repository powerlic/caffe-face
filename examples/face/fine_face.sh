~/git/caffe-face-caffe-face/build/tools/caffe train \
    --solver=/home/licheng/git/caffe-face-caffe-face/examples/face/finetune_face_solver.prototxt \
    --weights=/home/licheng/git/caffe-face-caffe-face/models/face/face_resnet_iter_96000.caffemodel\
    --gpu 1,2,3,4 \
    >fine_run_log.out 2>&1
