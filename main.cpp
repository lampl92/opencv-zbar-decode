#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <sys/time.h>
#include <zbar.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include "ZXingOpenCV.h"
//#define WINDOW_DEBUG 1
int threshold_value = 100;
int threshold_type = 0;
int const max_value = 255;
int const max_type = 4;
int const max_binary_value = 255;
int const sigma = 3;

const char* window_name = "Threshold Demo";
const char* trackbar_value = "Value";

using namespace std::chrono;
using namespace std;
using namespace cv;
using namespace zbar;

class Parallel_process : public cv::ParallelLoopBody
{

private:
    cv::Mat img;
    cv::Mat& retVal;
    int size;
    int diff;

public:
    Parallel_process(cv::Mat inputImgage, cv::Mat& outImage,
                     int sizeVal, int diffVal)
                : img(inputImgage), retVal(outImage),
                  size(sizeVal), diff(diffVal){}

    virtual void operator()(const cv::Range& range) const
    {
        for(int i = range.start; i < range.end; i++)
        {
            /* divide image in 'diff' number
            of parts and process simultaneously */

            cv::Mat in(img, cv::Rect(0, (img.rows/diff)*i,
                       img.cols, img.rows/diff));
            cv::Mat out(retVal, cv::Rect(0, (retVal.rows/diff)*i,
                                retVal.cols, retVal.rows/diff));
	    adaptiveThreshold(in, out, threshold_value, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 41 ,2 );
        }
    }
};

void getCamInfo(VideoCapture cam) {
    cout<<"CAP_PROP_FRAME_WIDTH " << cam.get(CAP_PROP_FRAME_WIDTH) << endl;
    cout<<"CAP_PROP_FRAME_HEIGHT " << cam.get(CAP_PROP_FRAME_HEIGHT) << endl;
    cout<<"CAP_PROP_FPS " << cam.get(CAP_PROP_FPS) << endl;
    cout<<"CAP_PROP_EXPOSURE " << cam.get(CAP_PROP_EXPOSURE) << endl;
    cout<<"CAP_PROP_FORMAT " << cam.get(CAP_PROP_FORMAT) << endl; //deafult CV_8UC3?!
    cout<<"CAP_PROP_CONTRAST " << cam.get(CAP_PROP_CONTRAST) << endl;
    cout<<"CAP_PROP_BRIGHTNESS "<< cam.get(CAP_PROP_BRIGHTNESS) << endl;
    cout<<"CAP_PROP_SATURATION "<< cam.get(CAP_PROP_SATURATION) << endl;
    cout<<"CAP_PROP_HUE "<< cam.get(CAP_PROP_HUE) << endl;
    cout<<"CAP_PROP_BUFFERSIZE "<< cam.get(CAP_PROP_BUFFERSIZE) << endl;
    cout<<"CAP_PROP_FOURCC "<< cam.get(CAP_PROP_FOURCC) << endl;

    int ex = static_cast<int>(cam.get(CAP_PROP_FOURCC));     // Get Codec Type- Int form
    char EXT[] = {(char)(ex & 255) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};
    cout << "Input codec type: " << EXT << endl;

}
uint64_t last_duration;
uint64_t timeSinceEpochMillisec() {
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

void start_track() {
     last_duration = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}
uint64_t mduration() {
	uint64_t last = last_duration;
	start_track();
	return timeSinceEpochMillisec() - last;
}

#define FRAME_RATE_LIMIT 10
#define CROP_VALUE 0
int main(int argc, char **argv) {
    last_duration = 0;
    int cam_idx = 0;
    int debug = 1;
    uint64_t prev_time, cur_time;
    char file_path[100];
    if (argc == 2) {
        cam_idx = atoi(argv[1]);
    }

    VideoCapture cap;
    int deviceID = cam_idx;
    int apiID = CAP_ANY;
    cap.open(deviceID, apiID);
    if (!cap.isOpened()) {
        cerr << "Could not open camera." << endl;
        exit(EXIT_FAILURE);
    }
    
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_BUFFERSIZE, 0);
    cap.set(CAP_PROP_FOURCC, CV_FOURCC('Y', 'U', 'Y', 'V'));
    getCamInfo(cap);
    
    // Create a zbar reader
    ImageScanner scanner;
    
    // Configure the reader
    scanner.set_config(ZBAR_QRCODE, ZBAR_CFG_ENABLE, 1);
    int counter = 0;
    float scale = 1;
    prev_time = timeSinceEpochMillisec();
    for (;;) {
        // Capture an OpenCV frame
        Mat raw_frame, frame, frame_grayscale, frame_process;
	cur_time = timeSinceEpochMillisec();
	cap.read(raw_frame);
	if (raw_frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
	    continue;
        }
	if((cur_time - prev_time) < (1000 / FRAME_RATE_LIMIT)) {
	    continue;
	}
	prev_time = timeSinceEpochMillisec();
	counter++;
	if(debug)
	   start_track();
	frame = raw_frame(Range(CROP_VALUE, 480 - CROP_VALUE), Range(CROP_VALUE, 640 - CROP_VALUE));
	//resize(frame, frame, Size(), (float) 1/scale, (float) 1/ scale);
	
	if(debug)
	   cout << "resize:" << mduration() << endl;
    	//Convert to grayscale
    	cvtColor(frame, frame_grayscale, COLOR_BGR2GRAY);
	if(debug)
	   cout << "cvtColor:" << mduration() << endl;
	GaussianBlur(frame_grayscale, frame_grayscale, Size(5, 5), 0, 0);
	if(debug)
	   cout << "gauss:" << mduration() << endl;
	//adaptiveThreshold(frame_grayscale, frame_process, threshold_value, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 41 ,2 );
	cv::parallel_for_(cv::Range(0, 4), Parallel_process(frame_grayscale, frame_process, 5, 8));
	if(debug)
	   cout << "adapt:" << mduration() << endl;
#ifdef ZBAR_DECODE
// Obtain image idata
        int width = frame_process.cols;
        int height = frame_process.rows;
        uchar *raw = (uchar *)(frame_process.data);

        // Wrap image data
        Image image(width, height, "Y800", raw, width * height);
        // Scan the image for barcodes
        if(scanner.scan(image) == 0) {
	    sprintf(file_path, "/tmp/opencv_%d_fail.jpg", counter);
	}
	if(debug)
	   cout << "decode:" << mduration() << endl;
        // Extract results
        for (Image::SymbolIterator symbol = image.symbol_begin(); symbol != image.symbol_end(); ++symbol) {
            time_t now;
            tm *current;
            now = time(0);
            current = localtime(&now);
	    sprintf(file_path, "/tmp/opencv_%d_pass.jpg", counter);
	    cout << "total: " << timeSinceEpochMillisec() - prev_time << " ms"<< endl;
            // do something useful with results
            cout    << "[" << current->tm_hour << ":" << current->tm_min << ":" << setw(2) << setfill('0') << current->tm_sec << "] " << " "
                    << "decoded " << symbol->get_type_name()
                    << " symbol \"" << symbol->get_data() << '"' << endl;
	    line(frame_process, Point(symbol->get_location_x(0), symbol->get_location_y(0)), Point(symbol->get_location_x(1), symbol->get_location_y(1)), Scalar(0, 255, 0), 2, 8, 0);
            line(frame_process, Point(symbol->get_location_x(1), symbol->get_location_y(1)), Point(symbol->get_location_x(2), symbol->get_location_y(2)), Scalar(0, 255, 0), 2, 8, 0);
            line(frame_process, Point(symbol->get_location_x(2), symbol->get_location_y(2)), Point(symbol->get_location_x(3), symbol->get_location_y(3)), Scalar(0, 255, 0), 2, 8, 0);
            line(frame_process, Point(symbol->get_location_x(3), symbol->get_location_y(3)), Point(symbol->get_location_x(0), symbol->get_location_y(0)), Scalar(0, 255, 0), 2, 8, 0);

        }
        image.set_data(NULL, 0);
#else
		auto results = ReadBarcodes(frame_process);
		if(debug)
		    cout << "decode:" << mduration() << endl;
		for(auto& result : results) {
			cout << timeSinceEpochMillisec() << "- text: " << result.text() << endl;
	    		cout << "total: " << timeSinceEpochMillisec() - prev_time << " ms"<< endl;
		}
#endif
        // Show captured frame, now with overlays!
        // clean up
	//imwrite(file_path, frame_process);
#ifdef WINDOW_DEBUG
		imshow("debug", frame_process);
		imshow("raw", raw_frame);
		waitKey(1);
#endif
    }

    return 0;
}
