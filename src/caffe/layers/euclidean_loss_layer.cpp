#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);


  hard_ratio_ = 0.;
  use_hard_mining_ = this->layer_param_.euclidean_loss_param().use_hard_mining();
  if (use_hard_mining_)
  {
    diff_sqr_.ReshapeLike(*bottom[0]);
    hard_ratio_ = this->layer_param_.euclidean_loss_param().hard_ratio();
  }
  threshold_ = Dtype(0);
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());

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
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
        const Dtype sign = (i == 0) ? 1 : -1;
        const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
        caffe_cpu_axpby(
        count,              // count
        alpha,                              // alpha
        diff_.cpu_data(),                   // a
        Dtype(0),                           // beta
        bottom[i]->mutable_cpu_diff());  // b
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

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe
