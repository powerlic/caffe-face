#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/cluster_loss_layer.hpp"

namespace caffe
{

template <typename Dtype>
void ClusterLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	//LOG(INFO)<<"forward gpu";
	const Dtype* multilabel_data = bottom[1]->cpu_data();
	int batch_size = bottom[0]->num();
    Dtype *cls_diff_square_sum_data = cls_diff_square_sum_.mutable_cpu_data();
    for (int i = 0; i < batch_size; ++i)
	{
		int label = (int)multilabel_data[i*3+1];
		int index = (int)multilabel_data[i*3];
		Dtype d;
		ReadCenterFromDB(label);
		
		caffe_gpu_sub(K_,bottom[0]->gpu_data()+i*K_,cls_center_blob_.gpu_data(),t_diff_.mutable_gpu_data());
		caffe_gpu_dot(K_,t_diff_.gpu_data(),t_diff_.gpu_data(),&d);
		cls_diff_square_sum_data[label]+=d;
		//-old
		ReadFeatureFromDB(index);
		caffe_gpu_sub(K_,feature_blob_.gpu_data(),cls_center_blob_.gpu_data(),t_diff_.mutable_gpu_data());
		caffe_gpu_dot(K_,t_diff_.gpu_data(),t_diff_.gpu_data(),&d);
		cls_diff_square_sum_data[label]-=d;
		//更新特征向量
		WriteFeatureToDB(index,label,bottom[0]->cpu_data()+i*K_);
	}
	Dtype loss;
	//caffe_gpu_memcpy(C_,cls_diff_square_sum_.cpu_data(),cls_diff_square_sum_.mutable_gpu_data());
	loss=caffe_cpu_asum(C_,cls_diff_square_sum_data);
	top[0]->mutable_cpu_data()[0]=loss/N_;	
}

template<typename Dtype>
void ClusterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{	
	const Dtype* multilabel_data = bottom[1]->cpu_data();
	int batch_size=bottom[0]->num();
	caffe_gpu_set(K_,(Dtype)0,sample_bp_diff_.mutable_gpu_data());
	if (propagate_down[0])
	{	
		for (int i = 0; i < batch_size; ++i)
		{
			int label = (int)multilabel_data[i*3+1];

			ReadCenterFromDB(label);
			caffe_gpu_sub(K_,bottom[0]->gpu_data()+i*K_,cls_center_blob_.gpu_data(),t_diff_.mutable_gpu_data());
			caffe_gpu_axpy(K_,(Dtype)2.0/(Dtype)N_,t_diff_.gpu_data(),sample_bp_diff_.mutable_gpu_data());
			caffe_copy(K_,sample_bp_diff_.gpu_data(),bottom[0]->mutable_gpu_diff()+i*K_);
		}
	}
	if (propagate_down[1]) 
	{
   	 	LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to index inputs.";
 	}
 	if (propagate_down[2]) 
	{
   	 	LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
 	}
}
INSTANTIATE_LAYER_GPU_FUNCS(ClusterLossLayer);
}// namespace caffe