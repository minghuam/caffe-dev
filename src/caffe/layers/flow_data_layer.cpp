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
FlowDataLayer<Dtype>::~FlowDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void FlowDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.flow_data_param().batch_size();
  const int num_stack_frames = this->layer_param_.flow_data_param().num_stack_frames();

  const string& source = this->layer_param_.flow_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label;

  vector<std::pair<std::string, int> > lines;
  while (infile >> filename >> label) {
    lines.push_back(std::make_pair(filename, label));
  }
  LOG(INFO) << "A total of " << lines.size() << " videos.";

  std::vector<std::string> flow_x_images;
  std::vector<std::string> flow_y_images;
  for(int i = 0; i < (int)lines.size(); i++){
    flow_x_images.clear();
    flow_y_images.clear();
    ls_files(flow_x_images, lines[i].first + "/x", "jpg");
    if(flow_x_images.size() < num_stack_frames){
        continue;
    }
    ls_files(flow_y_images, lines[i].first + "/y", "jpg");

    int index = 0;
    std::deque<std::pair<std::string, std::string> > image_set;
    while(true){
      while(image_set.size() < num_stack_frames){
          image_set.push_back(std::make_pair(flow_x_images[index], flow_y_images[index]));
          index++;
      }
      flow_images_.push_back(std::make_pair(image_set, lines[i].second));
      image_set.pop_front();
      if(index >= flow_x_images.size()){
          break;
      }
    }
  }
  LOG(INFO) << "Total number of stacked blobs: " << flow_images_.size();

  flow_set_id_ = 0;
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  ShuffleImages();

  cv::Mat I = cv::imread(flow_x_images[0]);
  vector<int> I_shape = this->data_transformer_->InferBlobShape(I);
  int height = I_shape[2];
  int width = I_shape[3];

  this->transformed_data_.Reshape(1, num_stack_frames * 2, height, width);

  int shape_array[4] = {batch_size, num_stack_frames * 2, height, width};
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
int FlowDataLayer<Dtype>::Rand(int n){
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(augmentation_rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
void FlowDataLayer<Dtype>::ShuffleImages() {
    LOG(INFO) << "shuffle images";
    caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    shuffle(flow_images_.begin(), flow_images_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void FlowDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

  FlowDataParameter flow_data_param = this->layer_param_.flow_data_param();
  const int batch_size = flow_data_param.batch_size();
  const int num_stack_frames = flow_data_param.num_stack_frames();
  const int mean = flow_data_param.mean();
  const bool show_level = flow_data_param.show_level();

  cv::Mat I = cv::imread(flow_images_[0].first[0].first);
  vector<int> I_shape = this->data_transformer_->InferBlobShape(I);
  int input_height = I.rows;
  int input_width = I.cols;
  int height = I_shape[2];
  int width = I_shape[3];

  this->transformed_data_.Reshape(1, num_stack_frames * 2, height, width);
  batch->data_.Reshape(batch_size, num_stack_frames * 2, height, width);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  for(int item_id = 0; item_id < batch_size; item_id++){
    // set label
    prefetch_label[item_id] = flow_images_[flow_set_id_].second;

    // pack a datum
    std::deque<std::pair<std::string, std::string> >& image_set =
            flow_images_[flow_set_id_].first;
    Datum datum;
    datum.set_channels(num_stack_frames * 2);
    datum.set_height(input_height);
    datum.set_width(input_width);
    datum.clear_data();
    datum.clear_float_data();

    for(int i = 0; i < (int)image_set.size(); i++){
        cv::Mat Ix = cv::imread(image_set[i].first, CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat Iy = cv::imread(image_set[i].second, CV_LOAD_IMAGE_GRAYSCALE);

        if(show_level > 0){
          cv::imshow("Ix", Ix);
          cv::imshow("Iy", Iy);
          if(show_level == 1){
            cv::waitKey(30);
          }else{
            cv::waitKey(0);
          }
        }

        for(int h = 0; h < Ix.rows; ++h){
          for(int w = 0; w < Ix.cols; ++w){
            float val = (float)Ix.at<uchar>(h, w) - mean;
            datum.add_float_data(val);
          }
        }
        for(int h = 0; h < Iy.rows; ++h){
          for(int w = 0; w < Iy.cols; ++w){
            float val = (float)Iy.at<uchar>(h, w) - mean;
            datum.add_float_data(val);
          }
        }
    }

    // set data
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));

    flow_set_id_++;
    if(flow_set_id_ >= flow_images_.size()){
      flow_set_id_ = 0;
      ShuffleImages();
    }
  }
}

INSTANTIATE_CLASS(FlowDataLayer);
REGISTER_LAYER_CLASS(FlowData);

}  // namespace caffe

