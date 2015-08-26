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
  const int image_height = this->layer_param_.flow_data_param().image_height();
  const int image_width = this->layer_param_.flow_data_param().image_width();
  const int new_height = this->layer_param_.flow_data_param().new_height();
  const int new_width  = this->layer_param_.flow_data_param().new_width();
  const int num_stack_frames = this->layer_param_.flow_data_param().num_stack_frames();

  const string& source = this->layer_param_.flow_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label;
  while (infile >> filename >> label) {
    lines_.push_back(std::make_pair(filename, label));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " videos.";

  lines_id_ = get_next_image_set(0, num_stack_frames);
  images_id_ = 0;

  int width = image_width;
  int height = image_height;
  if(new_height * new_width != 0){
    width = new_width;
    height = new_height;
  }

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
void FlowDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
int FlowDataLayer<Dtype>::get_next_image_set(int lines_id, int num_stack_frames) {
  int start_lines_id = lines_id;
  while(true){
    flow_x_images_.clear();
    flow_y_images_.clear();
    ls_files(flow_x_images_, lines_[lines_id].first + "/x", "jpg");
    ls_files(flow_y_images_, lines_[lines_id].first + "/y", "jpg");
    if(flow_x_images_.size() >= num_stack_frames){
      break;
    }
    lines_id++;
    if(lines_id >= lines_.size()){
      lines_id = 0;
    }
    if(lines_id == start_lines_id){
      LOG(ERROR) << "None of the video has enough frames!";
      return -1;
    }
  }
  return lines_id;
}

// This function is called on prefetch thread
template <typename Dtype>
void FlowDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

  FlowDataParameter flow_data_param = this->layer_param_.flow_data_param();
  const int batch_size = flow_data_param.batch_size();
  const int image_height = flow_data_param.image_height();
  const int image_width = flow_data_param.image_width();
  const int new_height = flow_data_param.new_height();
  const int new_width = flow_data_param.new_width();
  const int num_stack_frames = flow_data_param.num_stack_frames();
  const int mean = flow_data_param.mean();
  const bool shuffle = flow_data_param.shuffle();

  int width = image_width;
  int height = image_height;
  if(new_height * new_width != 0){
    width = new_width;
    height = new_height;
  }

  int num_stack_images = num_stack_frames * 2;
  this->transformed_data_.Reshape(1, num_stack_images, height, width);
  batch->data_.Reshape(batch_size, num_stack_images, height, width);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  for(int item_id = 0; item_id < batch_size; item_id++){
    // set labels
    prefetch_label[item_id] = lines_[lines_id_].second;

    // fill flow images
    while(flow_matrices_.size() < num_stack_images){
      cv::Mat Ix = cv::imread(flow_x_images_[images_id_]);
      cv::Mat Iy = cv::imread(flow_y_images_[images_id_]);

      if(new_height * new_width != 0){
        cv::resize(Ix, Ix, cv::Size(new_width, new_height));
        cv::resize(Iy, Iy, cv::Size(new_width, new_height));
      }

      flow_matrices_.push_back(Ix);
      flow_matrices_.push_back(Iy);
      images_id_++;
    }
    // pack to a datum
    Datum datum;
    datum.set_channels(num_stack_images);
    datum.set_height(height);
    datum.set_width(width);

    for(int i = 0; i < (int)flow_matrices_.size(); i++){
      cv::Mat& I = flow_matrices_[i];
      for(int h = 0; h < I.rows; ++h){
        for(int w = 0; w < I.cols; ++w){
          float val = static_cast<float>(I.at<uchar>(h, w, 0)) - mean;
          datum.add_float_data(val);
        }
      }
    }

    // set data
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));

    // remove oldest frame
    flow_matrices_.pop_front();
    flow_matrices_.pop_front();

    if(images_id_ >= flow_x_images_.size()){
      // load next set of images
      int prev_lines_id = lines_id_;
      lines_id_++;
      if(lines_id_ >= lines_.size()){
        lines_id_ = 0;
      }
      lines_id_ = get_next_image_set(lines_id_, num_stack_frames);
      if(lines_id_ < prev_lines_id && shuffle){
        ShuffleImages();
      }
      flow_matrices_.clear();
      images_id_ = 0;
    }

  }
}

INSTANTIATE_CLASS(FlowDataLayer);
REGISTER_LAYER_CLASS(FlowData);

}  // namespace caffe

