#include "inference.h"
#include <opencv2/dnn.hpp>
#include <iostream>
#include <openvino/core/preprocess/pre_post_process.hpp>

namespace yolo {

Inference::Inference(const std::string &model_path, 
                     const float &confidence_threshold, 
                     const float &nms_threshold,
                     const cv::Size &input_shape)
    : model_input_shape_(input_shape)
    , confidence_threshold_(confidence_threshold)
    , nms_threshold_(nms_threshold) {
    
    // Pre-allocate buffers
    boxes_.reserve(100);
    confidences_.reserve(100);
    detections_.reserve(100);
    indices_.reserve(100);
    
    InitializeModel(model_path);
}

void Inference::InitializeModel(const std::string &model_path) {
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    
    // Reshape if dynamic
    if (model->is_dynamic()) {
        model->reshape({1, 3, 
            static_cast<long int>(model_input_shape_.height), 
            static_cast<long int>(model_input_shape_.width)});
    }
    
    // Setup preprocessing
    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);
    
    ppp.input().preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale({255.0f, 255.0f, 255.0f});
    
    ppp.input().model().set_layout("NCHW");
    ppp.output().tensor().set_element_type(ov::element::f32);
    
    model = ppp.build();
    
    // *** Enable multi-threading for parallel inference computation ***
    compiled_model_ = core.compile_model(
        model, 
        "CPU",                                                              // LATENCY mode for better single-thread performance
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),  // THROUGHPUT mode for better multi-threading performance
        // ov::hint::inference_precision(ov::element::f32),               // removed this just for int8 inference
        ov::num_streams(1)  // 1 stream for sequential frame processing
    );
    
    inference_request_ = compiled_model_.create_infer_request();
    
    // Get output shape
    const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
    const ov::Shape output_shape = outputs[0].get_shape();
    model_output_shape_ = cv::Size(output_shape[2], output_shape[1]);
    
    std::cout << "Model initialized. Output shape: " 
              << model_output_shape_.width << "x" 
              << model_output_shape_.height << "\n";

    // Print inference precision
    try {
        std::cout << "Inference precision: " 
                  << compiled_model_.get_property(ov::hint::inference_precision).to_string() << "\n";
    } catch (...) {
        std::cout << "Could not query inference precision\n";
    }
    
    // Print performance info
    std::cout << "Optimal number of inference requests: " 
              << compiled_model_.get_property(ov::optimal_number_of_infer_requests) << "\n";
    std::cout << "Number of CPU streams: " 
              << compiled_model_.get_property(ov::num_streams) << "\n";
    
    // Print threading info
    try {
        std::cout << "Number of threads: " 
                  << compiled_model_.get_property(ov::inference_num_threads) << "\n";
    } catch (...) {
        std::cout << "Could not query thread count\n";
    }
}

void Inference::RunInference(cv::Mat &frame) {
    Preprocessing(frame);
    inference_request_.infer();
    PostProcessing(frame);
}

void Inference::Preprocessing(const cv::Mat &frame) {
    // Reuse pre-allocated Mat
    cv::resize(frame, resized_, model_input_shape_, 0, 0, cv::INTER_LINEAR);
    
    // Calculate scale factors for later use
    scale_factor_.x = static_cast<float>(frame.cols) / model_input_shape_.width;
    scale_factor_.y = static_cast<float>(frame.rows) / model_input_shape_.height;
    
    // Get input tensor and copy data
    ov::Tensor input_tensor = inference_request_.get_input_tensor();
    uint8_t* input_data = input_tensor.data<uint8_t>();
    
    // Fast memcpy - ensure continuous memory
    if (resized_.isContinuous()) {
        std::memcpy(input_data, resized_.data, resized_.total() * resized_.elemSize());
    } else {
        resized_ = resized_.clone();
        std::memcpy(input_data, resized_.data, resized_.total() * resized_.elemSize());
    }
}

void Inference::PostProcessing(const cv::Mat &frame) {
    // Clear previous results (doesn't deallocate memory due to reserve)
    detections_.clear();
    boxes_.clear();
    confidences_.clear();
    
    ov::Tensor output_tensor = inference_request_.get_output_tensor();
    const float* output_data = output_tensor.data<const float>();
    auto out_shape = output_tensor.get_shape(); // [1, 5, N]
    
    if (out_shape.size() != 3 || out_shape[1] != 5) {
        std::cerr << "Unexpected output shape\n";
        return;
    }
    
    const int num_boxes = static_cast<int>(out_shape[2]);
    const float inv_scale_x = 1.0f / scale_factor_.x;
    const float inv_scale_y = 1.0f / scale_factor_.y;
    
    // First pass: filter by confidence and scale coordinates
    for (int i = 0; i < num_boxes; ++i) {
        const float score = output_data[4 * num_boxes + i];
        
        if (score < confidence_threshold_) continue;
        
        const float x = output_data[0 * num_boxes + i];
        const float y = output_data[1 * num_boxes + i];
        const float w = output_data[2 * num_boxes + i];
        const float h = output_data[3 * num_boxes + i];
        
        // Scale to original frame size
        const float cx = x * scale_factor_.x;
        const float cy = y * scale_factor_.y;
        const float bw = w * scale_factor_.x;
        const float bh = h * scale_factor_.y;
        
        const int x1 = std::max(0, std::min(static_cast<int>(cx - bw * 0.5f), frame.cols - 1));
        const int y1 = std::max(0, std::min(static_cast<int>(cy - bh * 0.5f), frame.rows - 1));
        const int x2 = std::max(0, std::min(static_cast<int>(cx + bw * 0.5f), frame.cols - 1));
        const int y2 = std::max(0, std::min(static_cast<int>(cy + bh * 0.5f), frame.rows - 1));
        
        if (x2 <= x1 || y2 <= y1) continue;
        
        boxes_.emplace_back(x1, y1, x2 - x1, y2 - y1);
        confidences_.push_back(score);
    }
    
    // Apply NMS using OpenCV (hardware optimized)
    if (!boxes_.empty()) {
        cv::dnn::NMSBoxes(boxes_, confidences_, confidence_threshold_, nms_threshold_, indices_);
        
        detections_.reserve(indices_.size());
        for (int idx : indices_) {
            detections_.push_back({boxes_[idx], confidences_[idx]});
        }
    }
}

} // namespace yolo