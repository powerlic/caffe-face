#include<vector>
#include<string>
#include<fstream>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/cluster_loss_layer.hpp"
#include <boost/thread.hpp>
namespace caffe {

template<typename Dtype>
void ClusterLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  
   N_=this->layer_param_.cluster_loss_param().num_sample();
   C_=this->layer_param_.cluster_loss_param().num_cls();
   K_=this->layer_param_.cluster_loss_param().feature_size();

   C_K_shape_.resize(2);
   C_K_shape_[0]=C_;
   C_K_shape_[1]=K_;

   C_shape_.resize(1);
   C_shape_[0]=C_;

   K_shape_.resize(1);
   K_shape_[0]=K_;

   //load the LMDB
  feature_db_.reset(db::GetDB("leveldb"));
  feature_db_->Open(this->layer_param_.cluster_loss_param().feature_lmdb(),db::WRITE);
 

  cls_center_db_.reset(db::GetDB("leveldb"));
  cls_center_db_->Open(this->layer_param_.cluster_loss_param().cls_center_lmdb(),db::READ);

   //test
   cls_diff_square_sum_.Reshape(C_shape_);
   Dtype *cls_diff_square_sum_data=cls_diff_square_sum_.mutable_cpu_data();
   std::ifstream cls_diff_square_sum_file(this->layer_param_.cluster_loss_param().cls_diff_square_sum_file().c_str());
   int c=0;
   std::string line;
   while(getline(cls_diff_square_sum_file,line))
   {
    cls_diff_square_sum_data[c++]=(Dtype)atof(line.c_str());
   }
   CHECK_EQ(c,C_)<<"cls diff square sum size must be equal to the setting "<<std::endl;

   //data blob
   feature_blob_.Reshape(K_shape_);
   cls_center_blob_.Reshape(K_shape_);
   t_diff_.Reshape(K_shape_);
   sample_bp_diff_.Reshape(K_shape_);
}

template <typename Dtype>
void ClusterLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{ 
  LossLayer<Dtype>::Reshape(bottom,top);
}

template<typename Dtype>
void ClusterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{

  const Dtype* bottom_feature_data = bottom[0]->cpu_data();
  const Dtype* multilabel_data = bottom[1]->cpu_data();
  int batch_size = bottom[0]->num();
  const Dtype *cls_center_data=cls_center_blob_.cpu_data();
  const Dtype *feature_data=feature_blob_.cpu_data();
  Dtype *t_diff_data = t_diff_.mutable_cpu_data();
  Dtype *cls_diff_square_sum_data = cls_diff_square_sum_.mutable_cpu_data();
  for (int i = 0; i < batch_size; ++i)
  {
    int label = (int)multilabel_data[i*3+1];
    int index = (int)multilabel_data[i*3];
    Dtype d;



    // LOG(INFO)<<"Bottom feature "<<i;
    // for (int ii = 0; ii < K_; ++ii)
    // {
    //   std::cout<<bottom[0]->cpu_data()[i*K_+ii]<<" ";
    // }
    // std::cout<<std::endl;


    ReadCenterFromDB(label);

    // LOG(INFO)<<"Center "<<label;
    // for (int ii = 0; ii < K_; ++ii)
    // {
    //     std::cout<<cls_center_blob_.cpu_data()[ii]<<" ";
    // }
    // std::cout<<std::endl;

    //+new
    caffe_sub(K_,bottom_feature_data+i*K_,cls_center_data,t_diff_data);
    d = caffe_cpu_dot(K_,t_diff_data,t_diff_data);
    cls_diff_square_sum_data[label]+=d;


    // LOG(INFO)<<"diff_data "<<i;
    // for (int ii = 0; ii < K_; ++ii)
    // {
    //     std::cout<<t_diff_.cpu_data()[ii]<<" ";
    // }
    // std::cout<<std::endl;

    //-old
    ReadFeatureFromDB(index);
    caffe_sub(K_,feature_data,cls_center_data,t_diff_data);
    d = caffe_cpu_dot(K_,t_diff_data,t_diff_data);
    cls_diff_square_sum_data[label]-=d;
    //更新特征向量
    WriteFeatureToDB(index,label,bottom_feature_data+i*K_);
    //count the label
  }
  Dtype loss = caffe_cpu_asum(C_,cls_diff_square_sum_data)/N_;
  top[0]->mutable_cpu_data()[0]=loss;   
}

template<typename Dtype>
void ClusterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{ 
  const Dtype* multilabel_data = bottom[1]->cpu_data();
  const Dtype* bottom_feature_data = bottom[0]->cpu_data();
  int batch_size=bottom[0]->num();
  Dtype* bottom_diff_data = bottom[0]->mutable_cpu_diff();
  Dtype *sample_bp_diff_data=sample_bp_diff_.mutable_cpu_data();
  caffe_set(K_,(Dtype)0,sample_bp_diff_data);
  Dtype *t_diff_data = t_diff_.mutable_cpu_data();
  if (propagate_down[0])
  { 

    for (int i = 0; i < batch_size; ++i)
    {
      int label = (int)multilabel_data[i*3+1];
      ReadCenterFromDB(label);

      // //
      // LOG(INFO)<<"sample "<<i;
      // LOG(INFO)<<"center "<<label;
      // for (int ii = 0; ii < K_; ++ii)
      // {
      //   std::cout<<cls_center_blob_.cpu_data()[ii]<<" ";
      // }
      // std::cout<<std::endl;

      caffe_sub(K_,bottom_feature_data+i*K_,cls_center_blob_.cpu_data(),t_diff_data);

      // LOG(INFO)<<"bottom feature "<<label;
      // for (int ii = 0; ii < K_; ++ii)
      // {
      //   std::cout<<bottom_feature_data[i*K_+ii]<<" ";
      // }
      // std::cout<<std::endl;

      caffe_cpu_axpby(K_,(Dtype)2.0/(Dtype)N_,t_diff_data,(Dtype)0.0,sample_bp_diff_data);

      // LOG(INFO)<<"sample_bp_diff_data "<<label;
      // for (int ii = 0; ii < K_; ++ii)
      // {
      //   std::cout<<sample_bp_diff_data[ii]<<" ";
      // }
      // std::cout<<std::endl;

      caffe_copy(K_,sample_bp_diff_data,bottom_diff_data+i*K_);
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

// template<typename Dtype>
// void ClusterLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top)
// {
//   Forward_cpu(bottom,top);
// }
// template<typename Dtype>
// void ClusterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
//       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
// {
//   Backward_cpu(top,propagate_down,bottom);
// }





template<typename Dtype>
void ClusterLossLayer<Dtype>::ReadFeatureFromDB(int index)
{ 
  FeatureDatum feature_datum;
  std::string value;
  std::stringstream stream;
    stream << index;
    std::string key=stream.str();
    feature_db_->Retrieve(key,value);
  feature_datum.ParseFromString(value);
  Dtype* data=feature_blob_.mutable_cpu_data();
  for (int i = 0; i < K_; ++i)
  {
    data[i]=static_cast<Dtype>(feature_datum.feature_data(i));
  }
  feature_blob_.gpu_data();
  //caffe_gpu_memcpy(K_,feature_blob_.cpu_data(),feature_blob_.mutable_gpu_data());
}

template<typename Dtype>
void ClusterLossLayer<Dtype>::ReadCenterFromDB(int label)
{ 
  FeatureDatum center_datum;
  std::string value;
  std::stringstream stream;
    stream << label;
    std::string key=stream.str();
    cls_center_db_->Retrieve(key,value);
  center_datum.ParseFromString(value);
  Dtype* data=cls_center_blob_.mutable_cpu_data();
  for (int i = 0; i < K_; ++i)
  {
    data[i]=static_cast<Dtype>(center_datum.feature_data(i));
  }
  cls_center_blob_.gpu_data();
  //caffe_gpu_memcpy(K_,cls_center_blob_.cpu_data(),cls_center_blob_.mutable_gpu_data());
  //center_datum.Clear();
}
template<typename Dtype>
void ClusterLossLayer<Dtype>::WriteFeatureToDB(int index, int label, const Dtype *feature_blob_data)
{ 
  FeatureDatum feature_datum;
  std::stringstream stream;
    stream << index;
    std::string key=stream.str();

    feature_datum.set_size(K_);
    feature_datum.set_label(label);
    feature_datum.set_index(index);
    for (int ii = 0; ii < K_; ++ii)
    {
      feature_datum.add_feature_data(static_cast<float>(feature_blob_data[ii]));
    }
    std::string feature_out;
    feature_datum.SerializeToString(&feature_out);
    feature_db_->UpdateValue(key,feature_out);
}

#ifdef CPU_ONLY
STUB_GPU(ClusterLossLayer);
#endif

INSTANTIATE_CLASS(ClusterLossLayer);
REGISTER_LAYER_CLASS(ClusterLoss);

}  // namespace caffe