#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include "Conv2D.hpp"

using namespace cv;

int nthreads[] = { 1, 2, 4, 8, 16, 32 };
int chunk[] = { 1, 8, 16, 32, 64, 128 };

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "./filter img.jpg" << std::endl;
    }

    std::string image_path = samples::findFile(argv[1]);

    Mat img = imread(image_path, IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    std::cout << "Image width: " << img.cols << std::endl;
    std::cout << "Image height: " << img.rows << std::endl;
    std::cout << "Image channels: " << img.channels() << std::endl << std::endl;

    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            std::cout << "Actual nthreads: " << nthreads[i] << std::endl;
            std::cout << "Actual chunk: " << chunk[j] << std::endl;

            // Edge detection
            auto filter_h = Conv2D(
                new float[9] { -1, 0, 1, -2, 0, 2, -1, 0, 1 }, 
                3, 3,
                nthreads[i],
                chunk[j]
            );
            auto filter_v = Conv2D(
                new float[9] { -1, -2, -1, 0, 0, 0, 1, 2, 1 }, 
                3, 3,
                nthreads[i],
                chunk[j]
            );
            auto filtered_edge = (filter_h(img)/2) + (filter_v(img)/2);

            imwrite("filter.jpg", filtered_edge);
            /*imshow("Display window 1", img);
            imshow("Display window 2", filtered_edge);

            std::cout << "--------------------------------------------------------------\n";

            // Gaussian blur
            auto filter_blur = Conv2D(
                new float[9] { 0.075, 0.124, 0.075, 0.124, 0.204, 0.124, 0.075, 0.124, 0.075 },
                3, 3,
                nthreads[i],
                chunk[j]
            );
            
            Mat filtered_blur;
            img.copyTo(filtered_blur);
            for (int t = 0; t < 10; t++)
                filtered_blur = filter_blur(filtered_blur);

            imshow("Display window 3", filtered_blur);

            int k = waitKey(0); // Wait for a keystroke in the window
            if (k == 'q') return 0;  // exit*/

            std::cout << "--------------------------------------------------------------\n";
        }
    }

    return 0;
}
