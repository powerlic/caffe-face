#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());

  if(use_hard_mining_)
  {
    caffe_sqr(count,diff_.cpu_data(),diff_sqr_.mutable_cpu_data());
    const Dtype* begin = diff_sqr_.cpu_data();
    const Dtype* end = diff_sqr_.cpu_data()+count;
    std::vector<Dtype> diff_sqr_list(begin,end);
    int hard_size  = round((1-hard_ratio_)*count);
    std::sort(diff_sqr_list.begin(), diff_sqr_list.end());
    CHECK_LT(hard_size,count)<<"hard_size_ must be less than count"<<std::endl;
    threshold_ = diff_sqr_list[hard_size];
  }

  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      int count = bottom[0]->count();
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          count,              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
       if(use_hard_mining_)
        {
          const Dtype *diff_sqr_data=diff_sqr_.cpu_data();
          for (int ii = 0; ii < count; ++ii)
          { 
            if (*(diff_sqr_data+ii)<threshold_)
            {
              *(bottom[i]->mutable_cpu_diff()+ii)=Dtype(0);
            }
          }
        }

    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
