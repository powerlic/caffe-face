#include "caffe/layers/select_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>
namespace caffe{
template <typename Dtype>
void SelectLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top){
	const SelectParameter& select_param=this->layer_param_.select_param();
	select_labels_size_=select_param.select_label().size();
	label_bottom_index_=select_param.label_bottom_index();
	CHECK_GT(select_labels_size_, 0)<<" Select one label at least."<<std::endl;
	CHECK_EQ(select_param.copy_pair().size(),2)<<" copy pairs must be equal to 2."<<std::endl;
}


//Always remember the layer that has learned param is bottom[0], label_input is the bottom[last_index]
// And correspongd the top and bottom as this setting
//clsselect_layer_param
/*

layer
{
	name: "cls_select"
  	type: "Select"
	bottom: "conv4-1"
  	bottom: "label"
  	top: "conv4-1-valid"
  	top: "label-valid"
  	propagate_down: 1
 	propagate_down: 0
	select_param
	{
		select_label:0
		select_label:1
		label_bottom_index:1
		//bottom[0]:conv4_1 -> top[0]:conv4_1_valid  copysize 2 (score of two class)
		copy_pair
		{
			bottom_index:0
			top_index:0
			each_copy_size:2
		}
		//bottom[1]:label -> top[1]:label_valid  copysize 1 (only one label data )
		copy_pair
		{
			bottom_index:1
			top_index:1
			each_copy_size:1
		}
	}
}
	
*/

//roiselect_layer_param
/*
layer{

	name: "roi_select"
  	type: "Select"
 	bottom: "conv4-2"
 	bottom: "hdf_data" #roi
 	bottom: "label"
 	propagate_down: 1
 	propagate_down: 0
  	propagate_down: 0
	select_param
	{
		//part_sample
		select_label:-1 
		//positive_sample
		select_label:1
		label_bottom_index:2
		//bottom[0]:conv4_2 -> top[0]:conv4_2_valid  copysize 4 (x1,y1,x2,y2)
		copy_pair
		{
			bottom_index:0
			top_index:0
			each_copy_size:4
		}
		//bottom[1]:hdf_data -> top[1]:roi_valid  copysize 4 (x1,y1,x2,y2)
		copy_pair
		{
			bottom_index:1
			top_index:1
			each_copy_size:4
		}
	}
}
	
*/
//ptsselect_layer_param
/*
layer
{
	name: "pst_select"
	type: "Select"
	bottom: "conv4-3"
 	bottom: "hdf_data" #pts
 	bottom: "label"
 	propagate_down: 1
 	propagate_down: 0
  	propagate_down: 0
	select_param
	{
		//landmark_sample
		select_label:-2
		label_bottom_index:2
		//bottom[0]:conv4_3 -> top[0]:conv4_3_valid  copysize 10 (l_eye_x,l_eye_y,r_eye_x,r_eye_y....)
		copy_pair
		{
			bottom_index:0
			top_index:0
			each_copy_size:10
		}
		//bottom[1]:hdf_data -> top[1]:pts_valid  copysize 10 (l_eye_x,l_eye_y,r_eye_x,r_eye_y....)
		copy_pair
		{
			bottom_index:1
			top_index:1
			each_copy_size:10
		}
	}
}

*/

template <typename Dtype>
void SelectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    //std::cout << bottom[0]->num() << " " << bottom[0]->channels() << " " << bottom[0]->height() << " " << bottom[0]->width() << std::endl;
    //std::cout << bottom[1]->num() << " " << bottom[1]->channels() << " " << bottom[1]->height() << " " << bottom[1]->width() << std::endl;
    //std::cout << bottom[2]->num() << " " << bottom[2]->channels() << " " << bottom[2]->height() << " " << bottom[2]->width() << std::endl;
    M_ = bottom[0]->num();
    valid_counts_ = 0;
    const SelectParameter& select_param=this->layer_param_.select_param();
    const Dtype* label = bottom[label_bottom_index_]->cpu_data();
    for (int i = 0; i < M_; i++)
    {
        const int label_value = static_cast<int>(label[i]);
        bool selected=false;
        for (int j = 0; j < select_labels_size_; ++j)
        {
        	if (select_param.select_label(j)==label_value)
        	{
        		 selected=true;
        		 break;
        	}
        }
        if (selected)
        {
            valid_counts_++;
        }
    }
    //top 0 ,top 1 set size
  	top[0]->Reshape(valid_counts_,select_param.copy_pair(0).each_copy_size(),1,1);
  	top[1]->Reshape(valid_counts_,select_param.copy_pair(1).each_copy_size(),1,1);
  	//LOG(INFO)<<this->layer_param_.name()<<" valid_counts_ "<<valid_counts_<<std::endl;
}



template <typename Dtype>
void SelectLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    const SelectParameter& select_param=this->layer_param_.select_param();
    for (int i = 0; i < 2; ++i)
    {
    	caffe_set(valid_counts_ * select_param.copy_pair(i).each_copy_size(), (Dtype)0, top[i]->mutable_cpu_data());
    }

    const Dtype* label = bottom[label_bottom_index_]->cpu_data();
    int index = 0;
    for (int i = 0; i < M_; i++)
    {
        const int label_value = static_cast<int>(label[i]);
        bool selected = false;
        for (int j = 0; j < select_labels_size_; ++j)
        {
        	if (select_param.select_label(j)==label_value)
        	{
        		 selected = true;
        		 break;
        	}
        }
        if (selected)
        {
            for (int j = 0; j < 2; ++j)
            {
            	int bottom_index=select_param.copy_pair(j).bottom_index();
            	int top_index=select_param.copy_pair(j).top_index();
            	int copy_size=select_param.copy_pair(j).each_copy_size();
         //   	int bottom_dim=bottom[bottom_index]->count(1);
            	int bottom_dim =  select_param.copy_pair(j).bottom_dim();
		 		caffe_copy(copy_size, bottom[bottom_index]->cpu_data()+i*bottom_dim,top[top_index]->mutable_cpu_data()+ index*copy_size);
	 	 //if(this->layer_param_.name()=="roi_select")
		//{
		//LOG(INFO)<<"count "<<bottom[bottom_index]->count()<<" M_ "<<M_<<std::endl;
		//LOG(INFO)<<"bottom idx "<<bottom_index<<" dim "<<bottom_dim<<std::endl;
		//}
            }
		 // if(this->layer_param_.name()=="pts_select")
		 //  {
			// LOG(INFO)<<"label "<<label_value<<std::endl;	
			// LOG(INFO)<<"copy_size "<<select_param.copy_pair(0).each_copy_size()<<std::endl;
			// LOG(INFO)<<" pts data ";
			// for(int ii=0;ii<select_param.copy_pair(1).each_copy_size();ii++)
			// {
			//   LOG(INFO)<<*(bottom[1]->cpu_data()+i*select_param.copy_pair(1).bottom_dim()+ii)<<" ";
			// }
			// LOG(INFO)<<std::endl;
		 // }
         index += 1;
        }
    }
    //label landmark corre 
    // LOG(INFO)<<"<<<<<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
    // int hdf5_dim=bottom[label_bottom_index_-1]->count()/M_;
    // for (int i = 0; i < M_; ++i)
    // {
    // 	 const int label_value = static_cast<int>(label[i]);
    // 	 LOG(INFO)<<"label: "<<label_value<<" hdf5: ";
    // 	 const Dtype *hdf5_data=bottom[label_bottom_index_-1]->cpu_data();
    // 	 for (int ii = 0; ii < hdf5_dim; ++ii)
    // 	 {
    // 	 	LOG(INFO)<<hdf5_data[i*hdf5_dim+ii]<<" ";
    // 	 }
    // 	 LOG(INFO)<<std::endl;
    // }
    // LOG(INFO)<<">>>>>>>>>>>>>>>>>>>>>>>>>>"<<std::endl;
}


template <typename Dtype>
void SelectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	//only propagate conv or iner_product diff
    if(propagate_down[0])
    {
        //std::cout << "propagate_down[0]: true" << std::endl;
        //int bottom_dim= bottom[0]->count()/M_;
        //std::cout<< this->layer_param_.name()<<" bottom_dim "<<bottom_dim<<std::endl;
        const SelectParameter& select_param=this->layer_param_.select_param();
        int bottom_dim = select_param.copy_pair(0).bottom_dim();
        caffe_set(M_ * bottom_dim, (Dtype)0., bottom[0]->mutable_cpu_diff());
        const Dtype* label = bottom[label_bottom_index_]->cpu_data();
        int index = 0;
        for (int i = 0; i < M_; i++)
	    {
	        const int label_value = static_cast<int>(label[i]);
	        bool selected = false;
	        for (int j = 0; j < select_labels_size_; ++j)
	        {
	        	if (select_param.select_label(j)==label_value)
	        	{
	        		 selected = true;
	        		 break;
	        	}
	        }
	        if (selected)
	        {
	        	int bottom_index=select_param.copy_pair(0).bottom_index();
	        	int top_index=select_param.copy_pair(0).top_index();
	        	int copy_size=select_param.copy_pair(0).each_copy_size();
	       		caffe_copy(copy_size, top[top_index]->cpu_diff()+index*copy_size,
	       			bottom[bottom_index]->mutable_cpu_diff() + i * bottom_dim);
	            index += 1;
	        }
	    }
    }
}
template <typename Dtype>
void SelectLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    Forward_cpu(bottom, top);
}

template <typename Dtype>
void SelectLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    Backward_cpu(top, propagate_down, bottom);
}

#ifdef CPU_ONLY
STUB_GPU(SelectLayer);
#endif

INSTANTIATE_CLASS(SelectLayer);
REGISTER_LAYER_CLASS(Select);
}
