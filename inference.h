#ifndef YOLO_INFERENCE_H_
#define YOLO_INFERENCE_H_

#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

namespace yolo {

struct Detection {
    cv::Rect box;
    float confidence;
};

class Inference {
public:
    Inference(const std::string &model_path, 
              const float &confidence_threshold, 
              const float &nms_threshold,
              const cv::Size &input_shape = cv::Size(640, 640));

    void RunInference(cv::Mat &frame);
    const std::vector<Detection>& GetDetections() const { return detections_; }

private:
    void InitializeModel(const std::string &model_path);
    void Preprocessing(const cv::Mat &frame);
    void PostProcessing(const cv::Mat &frame);
    
    cv::Point2f scale_factor_;
    cv::Size model_input_shape_;
    cv::Size model_output_shape_;
    
    ov::InferRequest inference_request_;
    ov::CompiledModel compiled_model_;
    
    float confidence_threshold_;
    float nms_threshold_;
    
    std::vector<Detection> detections_;
    
    // Pre-allocated buffers to avoid allocations in hot path
    cv::Mat resized_;
    std::vector<cv::Rect> boxes_;
    std::vector<float> confidences_;
    std::vector<int> indices_;
};

} // namespace yolo

#endif // YOLO_INFERENCE_H_