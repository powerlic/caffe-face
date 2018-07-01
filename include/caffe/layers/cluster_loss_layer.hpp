#ifndef CAFFE_CLUSTER_LOSS_LAYER_HPP_
#define CAFFE_CLUSTER_LOSS_LAYER_HPP_


#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/db_lmdb.hpp"
#include "boost/shared_ptr.hpp"


namespace caffe {

template <typename Dtype>
class ClusterLossLayer : public LossLayer<Dtype> {
 public:
  explicit ClusterLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "ClusterLoss"; }
    virtual inline int ExactNumBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return -1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int N_;//样本数量
  int C_;//类别数量
  int K_;//特征向量维度
  vector<int> cls_N_;//每个类别训练样本数量
  Blob<Dtype> cls_diff_square_sum_;//j类样本与中心点距离平方和

  shared_ptr<db::DB> feature_db_;
  shared_ptr<db::DB> cls_center_db_;

  Blob<Dtype> feature_blob_;
  Blob<Dtype> cls_center_blob_;
  Blob<Dtype> t_diff_;
  Blob<Dtype> sample_bp_diff_;
  

  std::vector<int> C_K_shape_;
  std::vector<int> K_shape_;
  std::vector<int> C_shape_;

private:
  void ReadFeatureFromDB(int index);
  void WriteFeatureToDB(int index,int label, const Dtype *feature_blob_data);
  void ReadCenterFromDB(int label);
  //shared_ptr<boost::mutex> lmdb_mutex_;
};


}// namespace caffe
#endif  // CAFFE_CLUSTER_LOSS_LAYER_HPP_