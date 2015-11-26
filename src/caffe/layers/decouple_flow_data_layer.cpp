#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/directory.hpp"

namespace caffe {

template <typename Dtype>
DecoupleFlowDataLayer<Dtype>::~DecoupleFlowDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void DecoupleFlowDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.decouple_flow_data_param().batch_size();
  const int num_stack_frames = this->layer_param_.decouple_flow_data_param().num_stack_frames();
  const int data_type = this->layer_param_.decouple_flow_data_param().data_type();

  const string& source = this->layer_param_.decouple_flow_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label;

  vector<std::pair<std::string, int> > lines;
  while (infile >> filename >> label) {
    lines.push_back(std::make_pair(filename, label));
  }
  LOG(INFO) << "A total of " << lines.size() << " videos.";
  
  std::map<string, vector<FlowBlob> > instances;
  int max_instance_count = 0;
  int min_instance_count = 0;
  
  for(int i = 0; i < (int)lines.size(); i++){
    FlowBlob blob;
    blob.label = lines[i].second;
    ls_files(blob.bx_images, lines[i].first + "/bx", "jpg");
    if(blob.bx_images.size() < num_stack_frames){
        continue;
    }
    ls_files(blob.by_images, lines[i].first + "/by", "jpg");
    ls_files(blob.hx_images, lines[i].first + "/hx", "jpg");
    ls_files(blob.hy_images, lines[i].first + "/hy", "jpg");

    string basename = get_basename(lines[i].first);
    int start = basename.find_first_of("_");
    int end = basename.find_last_of("_");
    string action = basename.substr(start + 1, end - start - 1);
    if(instances.count(action) == 0){
       instances[action] = vector<FlowBlob>();
    }
    instances[action].push_back(blob);
    max_instance_count = std::max(max_instance_count, (int)instances[action].size());
    min_instance_count = std::min(min_instance_count, (int)instances[action].size()); 
  }

  LOG(INFO) << "Max instance count: " << max_instance_count;
  LOG(INFO) << "Min instance count: " << min_instance_count;

  // up sampling to balance training data
  typename std::map<string, vector<FlowBlob> >::iterator it;
  vector<FlowBlob> balanced_instances;
  for(it = instances.begin(); it != instances.end(); ++it){
    int index = 0;
    int size = 0;
    //while(size < max_instance_count){
    while(size < it->second.size()){
      balanced_instances.push_back(it->second[index++]);
      if(index > it->second.size() - 1){
        index = 0;
      }
      size++;
    }
  }

  typename std::vector<FlowBlob>::iterator b_it = balanced_instances.begin();
  for(; b_it != balanced_instances.end(); ++b_it){
    for(int index = 0; index < b_it->bx_images.size() - num_stack_frames; index++){
        FlowBlob blob;
        blob.label = b_it->label;
        for(int i = 0; i < num_stack_frames; i++){
            blob.bx_images.push_back(b_it->bx_images[index + i]);
            blob.by_images.push_back(b_it->by_images[index + i]);
            blob.hx_images.push_back(b_it->hx_images[index + i]);
            blob.hy_images.push_back(b_it->hy_images[index + i]);
        }
        flow_blobs_.push_back(blob);
    }
  }

  LOG(INFO) << "Total number of stacked blobs: " << flow_blobs_.size();

  flow_set_id_ = 0;
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  ShuffleImages();

  cv::Mat I = cv::imread(flow_blobs_[0].bx_images[0]);
  vector<int> I_shape = this->data_transformer_->InferBlobShape(I);
  int height = I_shape[2];
  int width = I_shape[3];

  int channels = num_stack_frames * 2;
  if(data_type == 2){
    channels = num_stack_frames * 4;
  }
  this->transformed_data_.Reshape(1, channels, height, width);
  int shape_array[4] = {batch_size, channels, height, width};

  vector<int> top_shape(&shape_array[0], &shape_array[0] + 4);
  for(int i = 0; i < this->PREFETCH_COUNT; ++i){
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  vector<int> label_shape(1, batch_size);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
  top[1]->Reshape(label_shape);
  LOG(INFO) << "label size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width();
}

template <typename Dtype>
void DecoupleFlowDataLayer<Dtype>::ShuffleImages() {
    LOG(INFO) << "shuffle images";
    caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    shuffle(flow_blobs_.begin(), flow_blobs_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void DecoupleFlowDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

  DecoupleFlowDataParameter flow_data_param = this->layer_param_.decouple_flow_data_param();
  const int batch_size = flow_data_param.batch_size();
  const int num_stack_frames = flow_data_param.num_stack_frames();
  const int mean = flow_data_param.mean();
  const int show_level = flow_data_param.show_level();
  const int data_type = flow_data_param.data_type();

  cv::Mat I = cv::imread(flow_blobs_[0].bx_images[0]);
  vector<int> I_shape = this->data_transformer_->InferBlobShape(I);
  int input_height = I.rows;
  int input_width = I.cols;
  int height = I_shape[2];
  int width = I_shape[3];

  int channels = num_stack_frames * 2;
  if(data_type == 2){
   channels = num_stack_frames * 4;
  }
  this->transformed_data_.Reshape(1, channels, height, width);
  batch->data_.Reshape(batch_size, channels, height, width);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  for(int item_id = 0; item_id < batch_size; item_id++){
    FlowBlob &blob = flow_blobs_[flow_set_id_];
    // set label
    prefetch_label[item_id] = blob.label;
    // pack a datum
    Datum datum;
    datum.set_channels(channels);
    datum.set_height(input_height);
    datum.set_width(input_width);
    datum.clear_data();
    datum.clear_float_data();

    for(int i = 0; i < num_stack_frames; i++){
        cv::Mat Ibx, Iby, Ihx, Ihy;
        if(data_type == 0 || data_type == 2){
            Ibx = cv::imread(blob.bx_images[i], CV_LOAD_IMAGE_GRAYSCALE);
            Iby = cv::imread(blob.by_images[i], CV_LOAD_IMAGE_GRAYSCALE);
        }
        if(data_type == 1 || data_type == 2){
            Ihx = cv::imread(blob.hx_images[i], CV_LOAD_IMAGE_GRAYSCALE);
            Ihy = cv::imread(blob.hy_images[i], CV_LOAD_IMAGE_GRAYSCALE);
        }

        if(show_level > 0){
          if(data_type == 0 || data_type == 2){
            cv::imshow("Ibx", Ibx);
            cv::imshow("Iby", Iby);
          }
          if(data_type == 1 || data_type == 2){
            cv::imshow("Ihx", Ihx);
            cv::imshow("Ihy", Ihy);
          }
          if(show_level == 1){
            cv::waitKey(30);
          }else{
            if(data_type == 0 || data_type == 2){
              LOG(INFO) << blob.bx_images[i];
              LOG(INFO) << blob.by_images[i];
            }
            if(data_type == 1 || data_type == 2){
              LOG(INFO) << blob.hx_images[i];
              LOG(INFO) << blob.hy_images[i];
            }
            LOG(INFO) << "label: " << blob.label << ", data_type: " << data_type;
            cv::waitKey(0);
          }
        }

        if(data_type == 0 || data_type == 2){
          for(int h = 0; h < Ibx.rows; ++h){
            for(int w = 0; w < Ibx.cols; ++w){
              float val = (float)Ibx.at<uchar>(h, w) - mean;
              datum.add_float_data(val);
            }
          }
          for(int h = 0; h < Iby.rows; ++h){
            for(int w = 0; w < Iby.cols; ++w){
              float val = (float)Iby.at<uchar>(h, w) - mean;
              datum.add_float_data(val);
            }
          }
        }
 
        if(data_type == 1 || data_type == 2){ 
          for(int h = 0; h < Ihx.rows; ++h){
            for(int w = 0; w < Ihx.cols; ++w){
              float val = (float)Ihx.at<uchar>(h, w) - mean;
              datum.add_float_data(val);
            }
          }
          for(int h = 0; h < Ihy.rows; ++h){
            for(int w = 0; w < Ihy.cols; ++w){
              float val = (float)Ihy.at<uchar>(h, w) - mean;
              datum.add_float_data(val);
            }
          }
        }
   }

    // set data
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));

    flow_set_id_++;
    if(flow_set_id_ >= flow_blobs_.size()){
      flow_set_id_ = 0;
      ShuffleImages();
    }
  }
}

INSTANTIATE_CLASS(DecoupleFlowDataLayer);
REGISTER_LAYER_CLASS(DecoupleFlowData);

}  // namespace caffe

