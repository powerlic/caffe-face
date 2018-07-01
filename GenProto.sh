protoc --proto_path /home/licheng/git/caffe-face-caffe-face/src/caffe/proto --cpp_out=/home/licheng/git/caffe-face-caffe-face/include/caffe/proto/ /home/licheng/git/caffe-face-caffe-face/src/caffe/proto/caffe.proto 
MV_CPP_OK=false
while [ "$MV_CPP_OK" == false ]
do
  sleep 1s
  if [ -f /home/licheng/git/caffe-face-caffe-face/include/caffe/proto/caffe.pb.cc ]
  then
	mv /home/licheng/git/caffe-face-caffe-face/include/caffe/proto/caffe.pb.cc /home/licheng/git/caffe-face-caffe-face/src/caffe/proto/
	MV_CPP_OK=true
  fi
done
