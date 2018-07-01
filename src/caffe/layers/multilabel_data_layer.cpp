#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/multilabel_data_layer.hpp"
#include "caffe/util/benchmark.hpp"


namespace caffe
{

template <typename Dtype>
MultilabelDataLayer<Dtype>::MultilabelDataLayer(const LayerParameter& param)
  : BasePrefetchingMultilabelDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
MultilabelDataLayer<Dtype>::~MultilabelDataLayer()
{
  this->StopInternalThread();
}



template <typename Dtype>
void MultilabelDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  const int batch_size = this->layer_param_.multilabel_data_param().batch_size();


  this->label_size_ = this->layer_param_.multilabel_data_param().label_size();
  this->pts_size_ = this->layer_param_.multilabel_data_param().pts_size();


  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.

  //data
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);

  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  //multilabel top[1] multi_labels
  vector<int> multilabel_shape(2,1);
  multilabel_shape[0]=batch_size;
  multilabel_shape[1]=label_size_;
  if (this->output_labels_&&label_size_>0)
  {
    top[1]->Reshape(multilabel_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) 
    {
      this->prefetch_[i].label_.Reshape(multilabel_shape);
      this->prefetch_[i].has_label_=true;
    }
    LOG(INFO)<<"output label size "<<top[1]->shape_string();
  }

  //pts top[2]
  vector<int> pts_shape(2,1);
  pts_shape[0]=batch_size;
  pts_shape[1]=pts_size_;
   if (this->output_labels_&&pts_size_>0)
   {
    top[2]->Reshape(pts_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) 
    {
      this->prefetch_[i].pts_.Reshape(pts_shape);
      this->prefetch_[i].has_pts_=true;
    }
    LOG(INFO)<<"output pts size "<<top[2]->shape_string();
   }
   LOG(INFO)<<"MultilabelDataLayer setup down!";
}

// This function is called on prefetch thread
// Btach is the memory address for batch training data
template<typename Dtype>
void MultilabelDataLayer<Dtype>::load_batch(MultilabelBatch<Dtype>* batch)
{
	CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
 	CPUTimer timer;
 	CHECK(batch->data_.count());
 	CHECK(this->transformed_data_.count());

 	 // Reshape according to the first datum of each batch
    // on single input batches allows for inputs of varying dimension.
    const int batch_size = this->layer_param_.multilabel_data_param().batch_size();
    Datum& datum = *(reader_.full().peek());
    //test
    //std::cout<<datum.channels()<<" "<<datum.height()<<" "<<datum.width()<<std::endl;


    ///test
   // Use data_transformer to infer the expected blob shape from datum.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
    this->transformed_data_.Reshape(top_shape);
   // Reshape batch according to the batch_size.
   top_shape[0] = batch_size;
   batch->data_.Reshape(top_shape);
   //LOG(INFO)<<"batch_size "<<batch_size;
   //LOG(INFO)<<"top_data_shape: "<<batch->data_.shape_string();

    Dtype* top_data = batch->data_.mutable_cpu_data();
    Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
    Dtype* top_pts = NULL;
    if (this->output_labels_) 
    {
      if (label_size_)
      {
        top_label = batch->label_.mutable_cpu_data();
      }
      if (pts_size_>0)
      {
        top_pts = batch->pts_.mutable_cpu_data();
      }
  	}
  	for (int item_id = 0; item_id < batch_size; ++item_id)
  	{
	    timer.Start();
	    // get a datum
	    Datum& datum = *(reader_.full().pop("Waiting for data"));
	    read_time += timer.MicroSeconds();
	    timer.Start();
     
	    // Apply data transformations (mirror, scale, crop...)
      //LOG(INFO)<<"item_id "<<item_id<<" offset "<<batch->data_.offset(item_id);
	    int offset = batch->data_.offset(item_id);
	    this->transformed_data_.set_cpu_data(top_data + offset);
      //test
     // LOG_IF(INFO,datum.encoded())<<" Datum is encdoded ";
      //test
	    this->data_transformer_->Transform(datum, &(this->transformed_data_));
      //test
      // LOG(INFO)<<"Datum Shape "<<datum.channels()<<" "<<datum.height()<<" "<<datum.width()<<std::endl;
      // const std::string& data_str=datum.data();
      // for (int c = 0; c < datum.channels(); ++c) 
      // {
      //   for (int h = 0; h < datum.height(); ++h) 
      //   {
      //        for (int w = 0; w < datum.width(); ++w)
      //       { 
      //         int data_index = (c * datum.height() + h) * datum.width() + w;
      //         float datum_element =
      //          static_cast<float>(static_cast<uint8_t>(data_str[data_index]));
      //          LOG(INFO)<<datum_element<<" "<<" transformed_data: "<<this->transformed_data_.cpu_data()[data_index];
      //       }
      //   }
      // }
      //test
	   /* // Copy label.
	    if (this->output_labels_)
	     {
	      top_label[item_id] = datum.label();
	    }*/
	    if (this->output_labels_)
	    {
        
        for (int i = 0; i < label_size_; ++i)
        {
            top_label[item_id*label_size_+i]=datum.multi_label(i);
	    
        }
	//LOG(INFO)<<"label "<<datum.multi_label(0)<<std::endl;
        for (int i = 0; i < pts_size_; ++i)
        {
            top_pts[item_id*pts_size_+i]=datum.pts(i);        
	}
	//LOG(INFO)<<"pts ";
	//for(int i=0;i<pts_size_;++i)
	//{
	 // LOG(INFO)<<datum.pts(i)<<" ";
	//}

		 }
	    trans_time += timer.MicroSeconds();

	    reader_.free().push(const_cast<Datum*>(&datum));
  	}
    //test
    // LOG(INFO)<<"Data Shape "<<batch->data_.shape_string()<<std::endl;
    // const Dtype* data=batch->data_.cpu_data();
    // LOG(INFO)<<"Data<<<<<<<<";
    // for (int i = 0; i < batch->data_.count(); ++i)
    // {
    //   LOG(INFO)<<"index "<<i<<" "<<static_cast<float>(data[i]);
    // }
    // LOG(INFO)<<"Data>>>>>>>";
    //test
  timer.Stop();
  batch_timer.Stop();
  //DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  //DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  //DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";

}


INSTANTIATE_CLASS(MultilabelDataLayer);
REGISTER_LAYER_CLASS(MultilabelData);

} // namespace caffe

