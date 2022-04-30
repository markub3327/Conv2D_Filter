#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <omp.h>
#include <fstream>

using namespace cv;


class Conv2D {
    private:
        float *kernel;
        int kwidth, kheight;
        int nthreads, chunk;
        std::ofstream MyFile;
        
        int circular(int M, int x)
        {
            if (x < 0)
                return x + M;
            if (x >= M)
                return x - M;
            
            return x;
        }
 
        Vec3f min(Mat image) {
            int x, y, d, k;
            double t_start, t_end;
            float min_val[image.channels()];
            
            // Find Min-max value per channel
            t_start = omp_get_wtime();
            for (d = 0; d < image.channels(); d++) {
                min_val[d] = image.at<Vec3f>(0, 0)[d];
            }
            #pragma omp parallel num_threads(this->nthreads) shared(k, image, chunk) private(x, y, d)
            {
                #pragma omp for reduction(min:min_val) schedule(guided, chunk)
                    for (k = 0; k < image.rows * image.cols * image.channels(); k++) {
                        x = k % image.cols;
                        y = (k / image.cols) % image.rows;
                        d = k / (image.rows * image.cols);
                        //std::cout << "x = " << x << ", y = " << y << ", d = " << d << ", k = " << k << std::endl;

                        if (min_val[d] > image.at<Vec3f>(y, x)[d])
                            min_val[d] = image.at<Vec3f>(y, x)[d];
                    }
            }
            t_end = omp_get_wtime(); 

            std::cout << "Min: Work took " << (t_end - t_start) << " seconds" << std::endl;            
            std::cout << "Min value " << min_val[0] << ", " << min_val[1] << ", " << min_val[2] << std::endl;

            MyFile << (t_end - t_start) << ";";

            return Vec3f(min_val);
        }

        Vec3f max(Mat image) {
            int x, y, d, k;
            double t_start, t_end;
            float max_val[image.channels()];

            // Find Min-max value per channel
            t_start = omp_get_wtime();
            for (d = 0; d < image.channels(); d++) {
                max_val[d] = image.at<Vec3f>(0, 0)[d];
            }
            #pragma omp parallel num_threads(this->nthreads) shared(k, image, chunk) private(x, y, d)
            {
                #pragma omp for reduction(max:max_val) schedule(guided, chunk)
                    for (k = 0; k < image.rows * image.cols * image.channels(); k++) {
                        x = k % image.cols;
                        y = (k / image.cols) % image.rows;
                        d = k / (image.rows * image.cols);
                        //std::cout << "x = " << x << ", y = " << y << ", d = " << d << ", k = " << k << std::endl;

                        if (max_val[d] < image.at<Vec3f>(y, x)[d])
                            max_val[d] = image.at<Vec3f>(y, x)[d];
                    }
            }
            t_end = omp_get_wtime(); 

            std::cout << "Max: Work took " << (t_end - t_start) << " seconds" << std::endl;            
            std::cout << "Max value " << max_val[0] << ", " << max_val[1] << ", " << max_val[2] << std::endl;

            MyFile << (t_end - t_start) << ";";

            return Vec3f(max_val);
        }

        void normalize(Mat image, Vec3f min_val, Vec3f max_val) {
            int x, y, d, k;
            double t_start, t_end;

            // Normalize
            t_start = omp_get_wtime();
            #pragma omp parallel num_threads(this->nthreads) shared(k, image, min_val, max_val, chunk) private(x, y, d)
            {
                #pragma omp for schedule(guided, chunk)
                    for (k = 0; k < image.rows * image.cols * image.channels(); k++) {
                        x = k % image.cols;
                        y = (k / image.cols) % image.rows;
                        d = k / (image.rows * image.cols);
                        //std::cout << "x = " << x << ", y = " << y << ", d = " << d << ", k = " << k << std::endl;

                        image.at<Vec3f>(y, x)[d] = (image.at<Vec3f>(y, x)[d] - min_val[d]) * 255.0 / (max_val[d] - min_val[d]);
                    }
            }
            t_end = omp_get_wtime(); 

            std::cout << "NORM: Work took " << (t_end - t_start) << " seconds" << std::endl;

            MyFile << (t_end - t_start) << ";";
        }

        Mat filter(Mat image) {
            Mat dst;
            int i, j, k, x, y, d, gx, gy;
            double t_start, t_end;
            float sum[image.channels()];

            // convert to float32
            image.clone().convertTo(dst, CV_32F);

            // Filter
            t_start = omp_get_wtime();
            #pragma omp parallel num_threads(this->nthreads) shared(k, image, dst, chunk) private(gx, gy, i, j, x, y, d, sum)
            {
                #pragma omp for schedule(guided, chunk)
                    for (k = 0; k < image.rows * image.cols * image.channels(); k++) {
                        x = k % image.cols;
                        y = (k / image.cols) % image.rows;
                        d = k / (image.rows * image.cols);
                        sum[d] = 0.0;
                        //std::cout << "x = " << x << ", y = " << y << ", d = " << d << ", k = " << k << std::endl;
            
                        for (i = 0; i < kheight; i++) {
                            for (j = 0; j < kwidth; j++) {
                                gx = circular(image.cols, (x - (this->kwidth / 2) + j));
                                gy = circular(image.rows, (y - (this->kheight / 2) + i));
                                sum[d] += (float)image.at<Vec3b>(gy, gx)[d] * kernel[(i * kwidth) + j];
                            }
                        }

                        dst.at<Vec3f>(y, x)[d] = sum[d];
                    }
            }
            t_end = omp_get_wtime(); 
            std::cout << "FILTER: Work took " << (t_end - t_start) << " seconds" << std::endl << std::endl;

            MyFile << (t_end - t_start) << ";";

            return dst;
        }

    public:
        Conv2D(float* kernel, int kwidth, int kheight, int nthreads, int chunk) : 
            MyFile("log.csv", std::ios_base::app),
            kernel(kernel),
            kwidth(kwidth), 
            kheight(kheight), 
            nthreads(nthreads), 
            chunk(chunk) {};
        ~Conv2D() {
            // Close the file
            MyFile.close();
        };

        Mat operator()(Mat input) {
            MyFile << this->nthreads << ";" << this->chunk << ";";

            auto y = this->filter(input);

            auto min_val = this->min(y);
            auto max_val = this->max(y);
            std::cout << std::endl;

            this->normalize(y, min_val, max_val);
            std::cout << std::endl;

            // convert to uchar
            Mat y_new;
            y.convertTo(y_new, CV_8U);

            MyFile << std::endl;

            return y_new;
        }
};