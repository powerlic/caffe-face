#include <algorithm>

#include "caffe/layers/clsselect_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>

namespace caffe {
    
template <typename Dtype>
void ClsSelectLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top)
{

}

template <typename Dtype>
void ClsSelectLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    //std::cout << bottom[0]->num() << " " << bottom[0]->channels() << " " << bottom[0]->height() << " " << bottom[0]->width() << std::endl;
    //std::cout << bottom[1]->num() << " " << bottom[1]->channels() << " " << bottom[1]->height() << " " << bottom[1]->width() << std::endl;
    M_ = bottom[0]->num();
    valid_counts_ = 0;
    const Dtype* label = bottom[1]->cpu_data();
    for (int i = 0; i < M_; i++)
    {
        const int label_value = static_cast<int>(label[i]);
        if (label_value != -1)
        {
            valid_counts_++;
        }
    }
    //std::cout << "valid_counts_ = " << valid_counts_ << std::endl;
    //top[0]->ReshapeLike(*bottom[0]);
    //top[1]->ReshapeLike(*bottom[1]);
    top[0]->Reshape(valid_counts_, 2, 1, 1);
    top[1]->Reshape(valid_counts_, 1, 1, 1);
}

/*
layer {
  name: "cls_select"
  type: "ClsSelect"
  bottom: "conv4-1"
  bottom: "label"
  top: "conv4-1-valid"
  top: "label-valid"
  propagate_down: 1
  propagate_down: 0
}
*/

template <typename Dtype>
void ClsSelectLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    caffe_set(valid_counts_ * 2, (Dtype)0., top[0]->mutable_cpu_data());
    caffe_set(valid_counts_ * 1, (Dtype)0, top[1]->mutable_cpu_data());
    const Dtype* label = bottom[1]->cpu_data();
    int index = 0;
    for (int i = 0; i < M_; i++)
    {
        const int label_value = static_cast<int>(label[i]);
        if (label_value == -1)
        {

        } else
        {
            caffe_copy(2, bottom[0]->cpu_data() + i * 2, top[0]->mutable_cpu_data() + index * 2);
            caffe_copy(1, bottom[1]->cpu_data() + i * 1, top[1]->mutable_cpu_data() + index * 1);
            index += 1;
        }
    }
}

template <typename Dtype>
void ClsSelectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    if(propagate_down[0])
    {
        //std::cout << "propagate_down[0]: true" << std::endl;
        caffe_set(M_ * 2, (Dtype)0., bottom[0]->mutable_cpu_diff());
        const Dtype* label = bottom[1]->cpu_data();
        int index = 0;
        for (int i = 0; i < M_; i++)
        {
            const int label_value = static_cast<int>(label[i]);
            if (label_value == -1)
            {

            } else
            {
                caffe_copy(2, top[0]->cpu_diff() + index * 2, bottom[0]->mutable_cpu_diff() + i * 2);
                index += 1;
            }
        }
    }
}

template <typename Dtype>
void ClsSelectLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

    Forward_cpu(bottom, top);
}

template <typename Dtype>
void ClsSelectLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    Backward_cpu(top, propagate_down, bottom);
}

#ifdef CPU_ONLY
STUB_GPU(ClsSelectLayer);
#endif

INSTANTIATE_CLASS(ClsSelectLayer);
REGISTER_LAYER_CLASS(ClsSelect);

}
