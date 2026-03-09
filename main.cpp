#include "inference.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <numeric>

void DrawDetections(cv::Mat &frame, const std::vector<yolo::Detection> &detections) {
    for (const auto &det : detections) {
        cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, 
                    cv::format("%.2f", det.confidence),
                    det.box.tl() + cv::Point(0, -5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 2);
    }
}

int main() {
    std::string video_path = "brt_presentation.mp4";
    //std::string model_dir = "yolo_nano_v2_1_class_640_no_filter_openvino_model";
    //std::string model_xml = model_dir + "/yolo_nano_v2_1_class_640_no_filter.xml";
    std::string model_dir = "yolo_nano_v2_1_class_640_no_filter_int8_openvino_model";
    std::string model_xml = model_dir + "/yolo_nano_v2_1_class_640_no_filter.xml";
    std::string output_video = "output_cpp.avi";
    
    const float CONFIDENCE = 0.2f;
    const float NMS_THRESHOLD = 0.45f;
    
    std::cout << "OpenVINO YOLO Inference (Optimized)\n";
    std::cout << "Model: " << model_xml << "\n";
    std::cout << "Video: " << video_path << "\n";
    
    // Initialize inference engine
    yolo::Inference inference(model_xml, CONFIDENCE, NMS_THRESHOLD);
    
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video.\n";
        return 1;
    }
    
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    std::cout << "Total frames: " << total_frames << "\n";
    std::cout << "Resolution: " << width << "x" << height << ", FPS: " << fps << "\n";
    
    // Use H264 codec if available (faster than MJPEG)
    int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
    cv::VideoWriter writer(output_video, fourcc, fps > 0 ? fps : 25.0, 
                          cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cout << "H264 codec not available, falling back to MJPEG\n";
        fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        writer.open(output_video, fourcc, fps > 0 ? fps : 25.0, cv::Size(width, height));
        if (!writer.isOpened()) {
            std::cerr << "Error: Could not open output file.\n";
            return 1;
        }
    }
    
    std::vector<double> times_total, times_inference, times_io;
    times_total.reserve(total_frames);
    times_inference.reserve(total_frames);
    times_io.reserve(total_frames);
    
    int frame_count = 0;
    
    while (true) {
        auto t_frame_start = std::chrono::high_resolution_clock::now();
        
        cv::Mat frame;
        auto t_read_start = std::chrono::high_resolution_clock::now();
        if (!cap.read(frame)) break;
        auto t_read_end = std::chrono::high_resolution_clock::now();
        
        frame_count++;
        
        // Run inference
        auto t_infer_start = std::chrono::high_resolution_clock::now();
        inference.RunInference(frame);
        const auto &detections = inference.GetDetections();
        auto t_infer_end = std::chrono::high_resolution_clock::now();
        
        // Draw results
        DrawDetections(frame, detections);
        
        // Write frame
        auto t_write_start = std::chrono::high_resolution_clock::now();
        writer.write(frame);
        auto t_write_end = std::chrono::high_resolution_clock::now();
        
        auto t_frame_end = std::chrono::high_resolution_clock::now();
        
        double dt_total = std::chrono::duration<double>(t_frame_end - t_frame_start).count();
        double dt_infer = std::chrono::duration<double>(t_infer_end - t_infer_start).count();
        double dt_io = std::chrono::duration<double>(t_read_end - t_read_start).count() +
                       std::chrono::duration<double>(t_write_end - t_write_start).count();
        
        times_total.push_back(dt_total);
        times_inference.push_back(dt_infer);
        times_io.push_back(dt_io);
        
        if (frame_count % 50 == 0)
            std::cout << "Processed " << frame_count << "/" << total_frames 
                      << " - Inference: " << dt_infer * 1000 << "ms, I/O: " << dt_io * 1000 << "ms\n";
    }
    
    cap.release();
    writer.release();
    
    if (!times_total.empty()) {
        double avg_total = std::accumulate(times_total.begin(), times_total.end(), 0.0) / times_total.size();
        double avg_infer = std::accumulate(times_inference.begin(), times_inference.end(), 0.0) / times_inference.size();
        double avg_io = std::accumulate(times_io.begin(), times_io.end(), 0.0) / times_io.size();
        
        std::cout << "\n==================================================\n";
        std::cout << "Frames processed: " << frame_count << "\n";
        std::cout << "Average total time: " << avg_total * 1000.0 << " ms\n";
        std::cout << "Average inference time: " << avg_infer * 1000.0 << " ms (" 
                  << (avg_infer/avg_total*100) << "%)\n";
        std::cout << "Average I/O time: " << avg_io * 1000.0 << " ms (" 
                  << (avg_io/avg_total*100) << "%)\n";
        std::cout << "FPS (total): " << 1.0 / avg_total << "\n";
        std::cout << "FPS (inference only): " << 1.0 / avg_infer << "\n";
        std::cout << "Output: " << output_video << "\n";
        std::cout << "==================================================\n";
    }
    
    return 0;
}