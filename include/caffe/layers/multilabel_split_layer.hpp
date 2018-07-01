#ifndef CAFFE_MULTILABEL_SPLIT_LAYER_HPP_
#define CAFFE_MULTILABEL_SPLIT_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

template <typename Dtype>
class MultilabelSplitLayer:public Layer<Dtype>{
	public:
		explicit MultilabelSplitLayer(const LayerParameter&param) : Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual inline int ExactNumBottomBlobs() const { return -1;}
		virtual inline int MinTopBlobs() const { return 1; }
  		virtual inline int MaxTopBlobs() const { return 12; }
		virtual inline const char* type() const{return "MultilabelSplit";}
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom);
		int label_size_;
		int label_index_;
	};

}//caffe namespace

#endif