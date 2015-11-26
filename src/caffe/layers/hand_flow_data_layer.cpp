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
HandFlowDataLayer<Dtype>::~HandFlowDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void HandFlowDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.hand_flow_data_param().batch_size();
  const int num_stack_frames = this->layer_param_.hand_flow_data_param().num_stack_frames();
  const string hand_mean_file = this->layer_param_.hand_flow_data_param().hand_mean_file();

  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(hand_mean_file.c_str(), &blob_proto);
  Blob<Dtype> mean_blob;
  mean_blob.FromProto(blob_proto);
 
  hand_mean_ = cv::Mat::zeros(mean_blob.height(), mean_blob.width(), CV_32F);
  for(int h = 0; h < mean_blob.height(); ++h){
    for(int w = 0; w < mean_blob.width(); ++w){
      hand_mean_.at<float>(h, w) = mean_blob.data_at(0, 0, h, w);
    }
  }

  const string& source = this->layer_param_.hand_flow_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string flow_folder;
  string hand_folder;
  string label_s;

  vector<vector<string> > lines;
  while (infile >> flow_folder >> hand_folder >> label_s) {
    vector<string> line;
    line.push_back(flow_folder);
    line.push_back(hand_folder);
    line.push_back(label_s);
    lines.push_back(line);
  }
  LOG(INFO) << "A total of " << lines.size() << " videos.";
  
  std::map<string, vector<FlowBlob> > instances;
  int max_instance_count = 0;
  int min_instance_count = 0;
  
  for(int i = 0; i < (int)lines.size(); i++){
    FlowBlob blob;
    blob.label = atoi(lines[i][2].c_str());
    ls_files(blob.flow_x_images, lines[i][0] + "/x", "jpg");
    if(blob.flow_x_images.size() < num_stack_frames){
        continue;
    }
    ls_files(blob.flow_y_images, lines[i][0] + "/y", "jpg");
    ls_files(blob.hand_images, lines[i][1], "jpg");

    string basename = get_basename(lines[i][0]);
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
    for(int index = 0; index < b_it->flow_x_images.size() - num_stack_frames; index++){
        FlowBlob blob;
        blob.label = b_it->label;
        for(int i = 0; i < num_stack_frames; i++){
            blob.flow_x_images.push_back(b_it->flow_x_images[index + i]);
            blob.flow_y_images.push_back(b_it->flow_y_images[index + i]);
            blob.hand_images.push_back(b_it->hand_images[index + i]);
        }
        flow_blobs_.push_back(blob);
    }
  }

  LOG(INFO) << "Total number of stacked blobs: " << flow_blobs_.size();

  flow_set_id_ = 0;
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  ShuffleImages();

  cv::Mat I = cv::imread(flow_blobs_[0].flow_x_images[0]);
  vector<int> I_shape = this->data_transformer_->InferBlobShape(I);
  int height = I_shape[2];
  int width = I_shape[3];

  this->transformed_data_.Reshape(1, num_stack_frames * 3, height, width);

  int shape_array[4] = {batch_size, num_stack_frames * 3, height, width};
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
float HandFlowDataLayer<Dtype>::RandWeight(){
  return 2.0f*((float)rand()/(float)RAND_MAX) -1.0f;
}

template <typename Dtype>
void HandFlowDataLayer<Dtype>::ShuffleImages() {
    LOG(INFO) << "shuffle images";
    caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    shuffle(flow_blobs_.begin(), flow_blobs_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void HandFlowDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

  CPUTimer batch_timer;
  batch_timer.Start();

  HandFlowDataParameter flow_data_param = this->layer_param_.hand_flow_data_param();
  const int batch_size = flow_data_param.batch_size();
  const int num_stack_frames = flow_data_param.num_stack_frames();
  const int flow_mean = flow_data_param.flow_mean();
  const int show_level = flow_data_param.show_level();

  cv::Mat I = cv::imread(flow_blobs_[0].flow_x_images[0]);
  vector<int> I_shape = this->data_transformer_->InferBlobShape(I);
  int input_height = I.rows;
  int input_width = I.cols;
  int height = I_shape[2];
  int width = I_shape[3];

  this->transformed_data_.Reshape(1, num_stack_frames * 3, height, width);
  batch->data_.Reshape(batch_size, num_stack_frames * 3, height, width);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  for(int item_id = 0; item_id < batch_size; item_id++){
    FlowBlob &blob = flow_blobs_[flow_set_id_];
    // set label
    prefetch_label[item_id] = blob.label;
    // pack a datum
    Datum datum;
    datum.set_channels(num_stack_frames * 3);
    datum.set_height(input_height);
    datum.set_width(input_width);
    datum.clear_data();
    datum.clear_float_data();

    for(int i = 0; i < num_stack_frames; i++){
        cv::Mat Ix = cv::imread(blob.flow_x_images[i], CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat Iy = cv::imread(blob.flow_y_images[i], CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat Ihand = cv::imread(blob.hand_images[i], CV_LOAD_IMAGE_GRAYSCALE);

        if(show_level > 0){
          cv::imshow("Ix", Ix);
          cv::imshow("Iy", Iy);
          cv::imshow("Ihand", Ihand);
          if(show_level == 1){
            cv::waitKey(30);
          }else{
            LOG(INFO) << blob.flow_x_images[i];
            LOG(INFO) << blob.flow_y_images[i];
            LOG(INFO) << blob.hand_images[i];
            LOG(INFO) << blob.label;;
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
        for(int h = 0; h < Ihand.rows; ++h){
          for(int w = 0; w < Ihand.cols; ++w){
            //float val = (float)Ihand.at<uchar>(h, w) -\
            //    hand_mean_.data_at(0, 0, h, w);
            float val = (float)Ihand.at<uchar>(h, w) - \
                  hand_mean_.at<float>(h, w);
            datum.add_float_data(val);
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

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}

INSTANTIATE_CLASS(HandFlowDataLayer);
REGISTER_LAYER_CLASS(HandFlowData);

}  // namespace caffe

