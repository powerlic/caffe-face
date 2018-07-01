#ifndef CAFFE_SELECT_LAYER_HPP_
#define CAFFE_SELECT_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

template <typename Dtype>
	class SelectLayer:public Layer<Dtype>{
	public:
		explicit SelectLayer(const LayerParameter&param) : Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual inline int ExactNumBottomBlobs() const { return -1;}
		virtual inline int ExactNumTopBlobs() const { return 2; }
		virtual inline const char* type() const{return "Select";}
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom);

		int valid_counts_;
		int M_;
		int select_labels_size_;
		int label_bottom_index_;
	};
}
#endif
