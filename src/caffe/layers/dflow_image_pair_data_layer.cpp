#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <map>
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/directory.hpp"

namespace caffe {

template <typename Dtype>
DecoupleFlowImagePairDataLayer<Dtype>::~DecoupleFlowImagePairDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void DecoupleFlowImagePairDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  DecoupleFlowImagePairDataParameter data_param = this->layer_param_.dflow_image_pair_data_param();
  const string& source = data_param.source();
  const int batch_size = data_param.batch_size();
  const string& mean_file = data_param.image_mean_file();
  num_stack_frames_ = data_param.num_stack_frames();

  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
  image_mean_.FromProto(blob_proto);

  int verb_label, obj_label, action_label;
  string flow_folder;
  string image_folder;

  LOG(INFO) << "Opening file " << source;
  LOG(INFO) << "Mean file " << mean_file;
  std::ifstream in_file(source.c_str());
  std::map<string, vector<FlowImageGroup> > instances;
  int max_instance_count = 0;
  int min_instance_count = 0;
  while(in_file >> flow_folder >> image_folder \
                >> verb_label >> obj_label >> action_label){
    FlowImageGroup group;
    string bx_folder = join_path(flow_folder, "bx");
    string by_folder = join_path(flow_folder, "by");
    string hx_folder = join_path(flow_folder, "hx");
    string hy_folder = join_path(flow_folder, "hy");
    ls_files(group.bx_images, bx_folder, "jpg");
    ls_files(group.by_images, by_folder, "jpg");
    ls_files(group.hx_images, hx_folder, "jpg");
    ls_files(group.hy_images, hy_folder, "jpg");
    ls_files(group.images, image_folder, "jpg");
    group.verb_label = verb_label;
    group.obj_label = obj_label;
    group.action_label = action_label;
    //flow_image_groups_.push_back(group);
    
    string foldername = get_basename(image_folder);
    int start = foldername.find_first_of("_");
    int end = foldername.find_last_of("_");
    string action = foldername.substr(start + 1, end - start - 1);
    if(instances.count(action) == 0){
      instances[action] = vector<FlowImageGroup>();
    }
    instances[action].push_back(group);
    min_instance_count = std::min(min_instance_count, (int)instances[action].size());
    max_instance_count = std::max(max_instance_count, (int)instances[action].size());
    LOG(INFO) << action << ": " << instances[action].size();
  }

  LOG(INFO) << "Min number of instances: " << min_instance_count;
  LOG(INFO) << "Max number of instances: " << max_instance_count;

  typename std::map<string, vector<FlowImageGroup> >::iterator it;
  for(it = instances.begin(); it != instances.end(); ++it){
    int size = 0;
    int index = 0;
    while(size < max_instance_count){
        flow_image_groups_.push_back(it->second[index++]);
        if(index > it->second.size() - 1){
            index = 0;
        }
        size++;
    }
  }

  LOG(INFO) << "Total groups: " << flow_image_groups_.size();
  flow_image_pair_id_ = 0;
  prefetch_rng_.reset(new Caffe::RNG(caffe_rng_rand()));
  image_rng_.reset(new Caffe::RNG(caffe_rng_rand()));

  ShuffleImages();

  cv::Mat I = cv::imread(flow_image_pairs_[0].image);
  vector<int> I_shape = this->data_transformer_->InferBlobShape(I);
  const int height = I_shape[2];
  const int width = I_shape[3];
  
  this->transformed_data_.Reshape(1, num_stack_frames_ * 4 + 3, height, width);

  int shape_array[4] = {batch_size, num_stack_frames_ * 4 + 3, height, width};
  vector<int> top_shape(&shape_array[0], &shape_array[0] + 4);
  for(int i = 0; i < this->PREFETCH_COUNT; ++i){
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  int label_shape_array[4] = {batch_size, 3, 1, 1};
  vector<int> label_shape(&label_shape_array[0], &label_shape_array[0] + 4);
  //vector<int> label_shape(1, batch_size);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
  top[1]->Reshape(label_shape);
  LOG(INFO) << "label size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width();
}

template <typename Dtype>
inline int DecoupleFlowImagePairDataLayer<Dtype>::get_frame_num(const string& path){
  int end = path.find_last_of(".");
  int start = path.find_last_of("_");
  string s = path.substr(start + 1, end - start - 1);
  return atoi(s.c_str());
  //return std::stoi(s);
}

template <typename Dtype>
void DecoupleFlowImagePairDataLayer<Dtype>::ShuffleImages() {
  LOG(INFO) << "Shuffle images...";
  
  flow_image_pair_id_ = 0;
  flow_image_pairs_.clear();
  
  for(int i = 0; i < flow_image_groups_.size(); i++){ 
    vector<string> &images = flow_image_groups_[i].images;
    vector<string> &bx_images = flow_image_groups_[i].bx_images;
    vector<string> &by_images = flow_image_groups_[i].by_images;
    vector<string> &hx_images = flow_image_groups_[i].hx_images;
    vector<string> &hy_images = flow_image_groups_[i].hy_images;
    
    int flow_min_num = get_frame_num(hx_images[0]);
    int flow_max_num = flow_min_num + hx_images.size() - 1;
    for(int j = 0; j < images.size(); j++){
      int frame_num = get_frame_num(images[j]);
      int min_num = std::max(flow_min_num, frame_num - num_stack_frames_ + 1);
      int max_num = std::min(frame_num, flow_max_num - num_stack_frames_ + 1);
      for(int k = 0; k < max_num - min_num + 1; k++){
        FlowImagePair pair;
        int index = min_num - flow_min_num + k;
        for(int offset = 0; offset < num_stack_frames_; offset++){
          pair.bx_images.push_back(bx_images[index + offset]);
          pair.by_images.push_back(by_images[index + offset]);
          pair.hx_images.push_back(hx_images[index + offset]);
          pair.hy_images.push_back(hy_images[index + offset]);
        }
        pair.image = images[j];
        pair.verb_label = flow_image_groups_[i].verb_label;
        pair.obj_label = flow_image_groups_[i].obj_label;
        pair.action_label = flow_image_groups_[i].action_label;
        flow_image_pairs_.push_back(pair);
      }
    }  
  }

  caffe::rng_t* prefetch_rng =
    static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(flow_image_pairs_.begin(), flow_image_pairs_.end(), prefetch_rng);

  LOG(INFO) << flow_image_pairs_.size() << " flow + image pairs";

}

// This function is called on prefetch thread
template <typename Dtype>
void DecoupleFlowImagePairDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  DecoupleFlowImagePairDataParameter data_param = this->layer_param_.dflow_image_pair_data_param();
  const int batch_size = data_param.batch_size();
  const int flow_mean = data_param.flow_mean();
  const int show_level = data_param.show_level();

  cv::Mat I = cv::imread(flow_image_pairs_[flow_image_pair_id_].image);
  vector<int> I_shape = this->data_transformer_->InferBlobShape(I);
  int input_height = I.rows;
  int input_width = I.cols;
  int height = I_shape[2];
  int width = I_shape[3];

  this->transformed_data_.Reshape(1, num_stack_frames_ * 4 + 3, height, width);
  batch->data_.Reshape(batch_size, num_stack_frames_ * 4 + 3, height, width);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  for(int item_id = 0; item_id < batch_size; item_id++){
    
    FlowImagePair &pair = flow_image_pairs_[flow_image_pair_id_];

    // set label
    prefetch_label[3 * item_id + 0] = pair.verb_label;
    prefetch_label[3 * item_id + 1] = pair.obj_label;
    prefetch_label[3 * item_id + 2] = pair.action_label;

    // pack a datum
    Datum datum;
    datum.set_channels(num_stack_frames_ * 4 + 3);
    datum.set_height(input_height);
    datum.set_width(input_width);
    datum.clear_data();
    datum.clear_float_data();

    // flow data
    for(int i = 0; i < num_stack_frames_; i++){
        cv::Mat Ibx = cv::imread(pair.bx_images[i], CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat Iby = cv::imread(pair.by_images[i], CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat Ihx = cv::imread(pair.hx_images[i], CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat Ihy = cv::imread(pair.hy_images[i], CV_LOAD_IMAGE_GRAYSCALE);


        if(show_level > 0){
          cv::imshow("Ibx", Ibx);
          cv::imshow("Iby", Iby);
          cv::imshow("Ihx", Ihx);
          cv::imshow("Ihy", Ihy);
          if(show_level == 1){
            cv::waitKey(30);
          }else{
            LOG(INFO) << pair.bx_images[i];
            LOG(INFO) << pair.by_images[i];
            LOG(INFO) << pair.hx_images[i];
            LOG(INFO) << pair.hy_images[i];
            LOG(INFO) << pair.verb_label << "," << pair.obj_label << "," << pair.action_label;
            cv::waitKey(0);
          }
        }

        for(int h = 0; h < Ibx.rows; ++h){
          for(int w = 0; w < Ibx.cols; ++w){
            float val = (float)Ibx.at<uchar>(h, w) - flow_mean;
            datum.add_float_data(val);
          }
        }
        for(int h = 0; h < Iby.rows; ++h){
          for(int w = 0; w < Iby.cols; ++w){
            float val = (float)Iby.at<uchar>(h, w) - flow_mean;
            datum.add_float_data(val);
          }
        }
        for(int h = 0; h < Ihx.rows; ++h){
          for(int w = 0; w < Ihx.cols; ++w){
            float val = (float)Ihx.at<uchar>(h, w) - flow_mean;
            datum.add_float_data(val);
          }
        }
        for(int h = 0; h < Ihy.rows; ++h){
          for(int w = 0; w < Ihy.cols; ++w){
            float val = (float)Ihy.at<uchar>(h, w) - flow_mean;
            datum.add_float_data(val);
          }
        }
   }

    //LOG(INFO) << flow_q[0].first;
    //LOG(INFO) << "rgb: " << flow_image_pair.second;
    //LOG(INFO) << "label: " << labels[1];

    // rgb image
    cv::Mat I = cv::imread(pair.image);
    for(int c = 0; c < I.channels(); ++c){
      for(int h = 0; h < I.rows; ++h){
        for(int w = 0; w < I.cols; ++w){
          datum.add_float_data((float)I.at<cv::Vec3b>(h, w)[c] - \
                               image_mean_.data_at(0, c, h, w));         
        }
      }
    }
        // set data
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));

    flow_image_pair_id_++;
    if(flow_image_pair_id_ >= flow_image_pairs_.size()){
      flow_image_pair_id_ = 0;
      ShuffleImages();
    }

  }
}

INSTANTIATE_CLASS(DecoupleFlowImagePairDataLayer);
REGISTER_LAYER_CLASS(DecoupleFlowImagePairData);

}  // namespace caffe


