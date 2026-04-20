#include "inference.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <numeric>
#include <fstream>

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
    std::string model_dir  = "models/mapillary_yolo_v8_nano_1_class_640_no_filter_v2_fp16_openvino_model";
    std::string model_xml  = model_dir + "/mapillary_yolo_v8_nano_1_class_640_no_filter_v2.xml";

    const float CONFIDENCE    = 0.2f;
    const float NMS_THRESHOLD = 0.45f;

    const bool SAVE_VIDEO = false;          // set to false to not save videos
    const std::string LOG_FILE = "perf_log.txt";

    std::cout << "OpenVINO YOLO Inference (Optimized)\n";
    std::cout << "Model: " << model_xml << "\n";
    std::cout << "Video: " << video_path << "\n";

    std::cout << "Initializing model...\n";
    yolo::Inference inference(model_xml, CONFIDENCE, NMS_THRESHOLD);

    int run_index = 1;

    // Outer loop: run the whole video over and over
    while (true) {
        std::cout << "\n==================================================\n";
        std::cout << "Starting run " << run_index << "\n";
        std::cout << "==================================================\n";

        std::string output_video = "output_cpp_" + std::to_string(run_index) + ".avi";

        std::cout << "Initializing video capture...\n";
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video.\n";
            return 1;
        }

        std::cout << "Video capture initialized.\n";

        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        double fps       = cap.get(cv::CAP_PROP_FPS);
        int width        = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height       = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        std::cout << "Total frames: " << total_frames << "\n";
        std::cout << "Resolution: " << width << "x" << height << ", FPS: " << fps << "\n";

        cv::VideoWriter writer;
        int fourcc = 0;

        if (SAVE_VIDEO) {
            // Use H264 codec if available (faster than MJPEG)
            fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
            writer.open(output_video, fourcc, fps > 0 ? fps : 25.0,
                        cv::Size(width, height));
            if (!writer.isOpened()) {
                std::cout << "H264 codec not available, falling back to MJPEG\n";
                fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
                writer.open(output_video, fourcc, fps > 0 ? fps : 25.0,
                            cv::Size(width, height));
                if (!writer.isOpened()) {
                    std::cerr << "Error: Could not open output file.\n";
                    return 1;
                }
            }
        }

        std::vector<double> times_total, times_inference, times_io;
        times_total.reserve(total_frames > 0 ? total_frames : 1000);
        times_inference.reserve(total_frames > 0 ? total_frames : 1000);
        times_io.reserve(total_frames > 0 ? total_frames : 1000);

        int frame_count = 0;

        // Inner loop: per-frame processing for this run
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

            // Draw results (optional)
            //DrawDetections(frame, detections);

            // Write frame (optional)
            auto t_write_start = std::chrono::high_resolution_clock::now();
            if (SAVE_VIDEO) {
                // You can also comment this out if you want:
                //writer.write(frame);
            }
            auto t_write_end = std::chrono::high_resolution_clock::now();

            auto t_frame_end = std::chrono::high_resolution_clock::now();

            double dt_total = std::chrono::duration<double>(t_frame_end - t_frame_start).count();
            double dt_infer = std::chrono::duration<double>(t_infer_end - t_infer_start).count();
            double dt_io    = std::chrono::duration<double>(t_read_end - t_read_start).count() +
                              std::chrono::duration<double>(t_write_end - t_write_start).count();

            times_total.push_back(dt_total);
            times_inference.push_back(dt_infer);
            times_io.push_back(dt_io);

            if (frame_count % 50 == 0) {
                std::cout << "Run " << run_index
                          << " - Processed " << frame_count << "/" << total_frames
                          << " - Inference: " << dt_infer * 1000 << " ms, I/O: "
                          << dt_io * 1000 << " ms\n";
            }
        }

        cap.release();
        if (SAVE_VIDEO && writer.isOpened()) {
            writer.release();
        }

        if (!times_total.empty()) {
            double avg_total = std::accumulate(times_total.begin(), times_total.end(), 0.0) / times_total.size();
            double avg_infer = std::accumulate(times_inference.begin(), times_inference.end(), 0.0) / times_inference.size();
            double avg_io    = std::accumulate(times_io.begin(), times_io.end(), 0.0) / times_io.size();

            std::cout << "\n================ Run " << run_index << " stats ================\n";
            std::cout << "Frames processed: " << frame_count << "\n";
            std::cout << "Average total time: " << avg_total * 1000.0 << " ms\n";
            std::cout << "Average inference time: " << avg_infer * 1000.0 << " ms ("
                      << (avg_infer / avg_total * 100.0) << "%)\n";
            std::cout << "Average I/O time: " << avg_io * 1000.0 << " ms ("
                      << (avg_io / avg_total * 100.0) << "%)\n";
            std::cout << "FPS (total): " << 1.0 / avg_total << "\n";
            std::cout << "FPS (inference only): " << 1.0 / avg_infer << "\n";
            if (SAVE_VIDEO) {
                std::cout << "Output: " << output_video << "\n";
            }
            std::cout << "==================================================\n";

            // Append per-run stats to log file
            std::ofstream log(LOG_FILE, std::ios::app);
            if (log) {
                log << "run=" << run_index
                    << " frames=" << frame_count
                    << " avg_total_ms=" << avg_total * 1000.0
                    << " avg_infer_ms=" << avg_infer * 1000.0
                    << " avg_io_ms="    << avg_io * 1000.0
                    << " fps_total="    << (1.0 / avg_total)
                    << " fps_infer="    << (1.0 / avg_infer)
                    << " save_video="   << (SAVE_VIDEO ? 1 : 0)
                    << " output="       << (SAVE_VIDEO ? output_video : "none")
                    << "\n";
            }
        } else {
            std::cout << "No frames processed in run " << run_index << ".\n";
        }

        ++run_index;  // next run will use output_cpp_2.avi, 3, ...
    }

    return 0;
}