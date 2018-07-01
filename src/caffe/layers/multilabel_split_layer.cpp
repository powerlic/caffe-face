#include "caffe/layers/multilabel_split_layer.hpp"
#include <iostream>
namespace caffe{
template <typename Dtype>
void MultilabelSplitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top){
	this->label_size_=this->layer_param_.multilabel_split_param().label_size();
	this->label_index_=-1;
	if (this->layer_param_.multilabel_split_param().has_label_index())
	{
		this->label_index_=this->layer_param_.multilabel_split_param().label_index();
	}
	
}

template <typename Dtype>
void MultilabelSplitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	int M=bottom[0]->num();//batch size
	if (label_index_>=0)
	{
		top[0]->Reshape(M,1,1,1);
	}
	else
	{
		for (int i = 0; i < label_size_; ++i)
		{
			top[i]->Reshape(M,1,1,1);
		}
	}
}

template <typename Dtype>
void MultilabelSplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
	int M = bottom[0]->num();
	const Dtype *bottom_data = bottom[0]->cpu_data();
	if (label_index_>=0)
	{
		Dtype *top_data = top[0]->mutable_cpu_data();
		for (int i = 0; i < M; ++i)
		{
			top_data[i] = bottom_data[i*label_size_+label_index_];
		}
	}
	else
	{
		for (int i = 0; i < M; ++i)
		{
			for (int j = 0; j < label_size_; ++j)
			{
				Dtype *top_data = top[j]->mutable_cpu_data();
				top_data[i] = bottom_data[i*label_size_+j];
			}
		}
	}
}

template <typename Dtype>
void MultilabelSplitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){
	Forward_cpu(bottom, top);
}

template <typename Dtype>
void MultilabelSplitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& top)
{
	
}

template <typename Dtype>
void MultilabelSplitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom)
{

}

#ifdef CPU_ONLY
STUB_GPU(MultilabelSplitLayer);
#endif

INSTANTIATE_CLASS(MultilabelSplitLayer);
REGISTER_LAYER_CLASS(MultilabelSplit);


}//caffe namespace 