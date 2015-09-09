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
ImageFlowDataLayer<Dtype>::~ImageFlowDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageFlowDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  ImageFlowDataParameter data_param = this->layer_param_.image_flow_data_param();
  const string& image_folder = data_param.image_folder();
  const string& flow_folder = data_param.flow_folder();
  const string& source = data_param.source();
  const int batch_size = data_param.batch_size();
  const int num_stack_frames = data_param.num_stack_frames();
  const string& mean_file = data_param.image_mean_file();

  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
  image_mean_.FromProto(blob_proto);

  string folder_name;
  int verb_label, obj_label, action_label;
  vector<std::string> folder_names;
  vector<vector<int> > labels;

  LOG(INFO) << "Opening file " << source;
  LOG(INFO) << "Mean file " << mean_file;
  std::ifstream in_file(source.c_str());
  while(in_file >> folder_name >> verb_label >> obj_label >> action_label){
      folder_names.push_back(folder_name);
      vector<int> v;
      v.push_back(verb_label);
      v.push_back(obj_label);
      v.push_back(action_label);
      labels.push_back(v);
  }

  std::vector<std::string> rgb_images;
  std::vector<std::string> flow_x_images;
  std::vector<std::string> flow_y_images;
  for(int i = 0; i < (int)folder_names.size(); i++){
    rgb_images.clear();
    flow_x_images.clear();
    flow_y_images.clear();

    std::string rgb_folder = join_path(image_folder, folder_names[i]);
    ls_files(rgb_images, rgb_folder, "jpg");

    std::string flow_path = join_path(flow_folder, folder_names[i]);
    ls_files(flow_x_images, join_path(flow_path, "x"), "jpg");
    if(flow_x_images.size() < num_stack_frames){
        continue;
    }
    ls_files(flow_y_images, join_path(flow_path, "y"), "jpg");

    int images_index = 0;
    int flow_index = 0;
    FLOW_Q flow_q;
    while(true){
      while(flow_q.size() < num_stack_frames){
          flow_q.push_back(std::make_pair(flow_x_images[flow_index], \
                                          flow_y_images[flow_index]));
          flow_index++;
      }

      std::pair<FLOW_Q, std::string> flow_img_pair = std::make_pair(\
                  flow_q, rgb_images[images_index]);

      images_index++;

      image_flow_pairs_.push_back(std::make_pair(flow_img_pair, labels[i]));

      /*
      LOG(INFO) << flow_img_pair.first.size();
      LOG(INFO) << flow_img_pair.first[num_stack_frames-1].first;
      LOG(INFO) << flow_img_pair.second;
      LOG(INFO) << folder_names[i] << ": " << labels[i][0] << "," \
        << labels[i][1] << "," << labels[i][2];

      cv::Mat Ix = cv::imread(flow_img_pair.first[num_stack_frames-1].first);
      cv::Mat Iy = cv::imread(flow_img_pair.first[num_stack_frames-1].second);
      cv::Mat I = cv::imread(flow_img_pair.second);
      cv::imshow("Ix", Ix);
      cv::imshow("Iy", Iy);
      cv::imshow("I", I);
      cv::waitKey(0);
      */

      flow_q.pop_front();
      if(flow_index >= flow_x_images.size()){
          break;
      }
    }
  }
  LOG(INFO) << "Total number of stacked blobs: " << image_flow_pairs_.size();

  image_flow_pair_id_ = 0;
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));

  ShuffleImages();

  cv::Mat I = cv::imread(rgb_images[0]);
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

  int label_shape_array[4] = {batch_size, 1, 1, 1};
  vector<int> label_shape(&label_shape_array[0], &label_shape_array[0] + 4);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
  top[1]->Reshape(label_shape);
  LOG(INFO) << "label size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width();
}

template <typename Dtype>
int ImageFlowDataLayer<Dtype>::Rand(int n){
  //caffe::rng_t* rng =
  //    static_cast<caffe::rng_t*>(augmentation_rng_->generator());
  //return ((*rng)() % n);
  return 0;
}

template <typename Dtype>
void ImageFlowDataLayer<Dtype>::ShuffleImages() {
    LOG(INFO) << "shuffle images";
    caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    shuffle(image_flow_pairs_.begin(), image_flow_pairs_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageFlowDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  ImageFlowDataParameter data_param = this->layer_param_.image_flow_data_param();
  const int batch_size = data_param.batch_size();
  const int num_stack_frames = data_param.num_stack_frames();
  const int flow_mean = data_param.flow_mean();
  const bool show_level = data_param.show_level();

  cv::Mat I = cv::imread(image_flow_pairs_[image_flow_pair_id_].first.second);
  vector<int> I_shape = this->data_transformer_->InferBlobShape(I);
  int height = I_shape[2];
  int width = I_shape[3];

  this->transformed_data_.Reshape(1, num_stack_frames * 2, height, width);
  batch->data_.Reshape(batch_size, num_stack_frames * 2, height, width);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  for(int item_id = 0; item_id < batch_size; item_id++){
    
    std::pair<FLOW_Q, std::string>& flow_image_pair = image_flow_pairs_[image_flow_pair_id_].first;
    std::vector<int>& labels = image_flow_pairs_[image_flow_pair_id_].second;

    // set label
    //prefetch_label[item_id] = flow_images_[flow_set_id_].second;
    prefetch_label[1 * item_id + 0] = labels[0];
    //prefetch_label[3 * item_id + 1] = labels[1];
    //prefetch_label[3 * item_id + 2] = labels[3];

    // pack a datum
    Datum datum;
    datum.set_channels(num_stack_frames * 2);
    datum.set_height(height);
    datum.set_width(width);
    datum.clear_data();
    datum.clear_float_data();

    // flow data

    std::deque<std::pair<std::string, std::string> >& flow_q = flow_image_pair.first;
    for(int i = 0; i < (int)flow_q.size(); i++){

        //LOG(INFO) << "i: " << i << ": " << flow_q[i].first;
        //LOG(INFO) << "i: " << i << ": " << flow_q[i].second;

        cv::Mat Ix = cv::imread(flow_q[i].first, CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat Iy = cv::imread(flow_q[i].second, CV_LOAD_IMAGE_GRAYSCALE);

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
            float val = (float)Ix.at<uchar>(h, w) - flow_mean;
            datum.add_float_data(val);
          }
        }
        for(int h = 0; h < Iy.rows; ++h){
          for(int w = 0; w < Iy.cols; ++w){
            float val = (float)Iy.at<uchar>(h, w) - flow_mean;
            datum.add_float_data(val);
          }
        }
    }

    //LOG(INFO) << "rgb: " << flow_image_pair.second;

    // rgb image
    cv::Mat I = cv::imread(flow_image_pair.second);
    for(int c = 0; c < I.channels(); ++c){
      for(int h = 0; h < I.rows; ++h){
        for(int w = 0; w < I.cols; ++w){
         ;//datum.add_float_data((float)I.at<cv::Vec3b>(h, w)[c] - \
                               image_mean_.data_at(0, c, h, w));         
          //int index = ((20 + c) * I.rows + h) * I.cols + w;
          //float val = (float)I.at<cv::Vec3b>(h,w)[c] - \
                                image_mean_.data_at(0, c, h, w);
          //datum.set_float_data(index, val);
        }
      }
    }

    // set data
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));

    image_flow_pair_id_++;
    if(image_flow_pair_id_ >= image_flow_pairs_.size()){
      image_flow_pair_id_ = 0;
      ShuffleImages();
    }
  }
}

INSTANTIATE_CLASS(ImageFlowDataLayer);
REGISTER_LAYER_CLASS(ImageFlowData);

}  // namespace caffe


