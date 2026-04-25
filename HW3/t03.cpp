#include <opencv2/opencv.hpp>
#include "hw3utils.hpp"

int main(int argc, char* argv[]) {
    const cv::Vec3d rvec{0.1, -0.2, 0.3}, t{1.1, 2.0, 3.};
    const cv::Affine3d affineTransform{rvec, t};
    const double fx{300.}, fy{300.}, cx{320.}, cy{320.};
    const cv::Matx33d K{fx, 0., cx, 0., fy, cy, 0., 0., 1.};

	cv::Mat Rt{cv::Mat{affineTransform.matrix}.rowRange(0, 3)};
	const cv::Mat P{K * Rt};
	

	const std::vector<cv::Point3d> scenePoints = {
		{0.0, 0.0, 5.0},
		{0.5, 0.0, 5.0},
		{-0.5, 0.0, 5.0},
		{0.0, 0.5, 5.0},
		{0.0, -0.5, 5.0},

		{1.0, 1.0, 7.0},
		{-1.0, 1.0, 7.0},
		{1.0, -1.0, 7.0},
		{-1.0, -1.0, 7.0},

		{2.0, 0.0, 6.0},
		{-2.0, 0.0, 6.0},
		{0.0, 2.0, 6.0},
		{0.0, -2.0, 6.0},

		{1.5, -1.5, 8.0},
		{-1.5, 1.5, 8.0},
		{2.0, 2.0, 9.0},
		{-2.0, -2.0, 9.0},

		{0.3, -0.3, 3.0},
		{-0.3, 0.3, 3.0},
		{0.0, 0.0, 4.0},

		{3.0, 1.0, 7.0},
		{-3.0, -1.0, 7.0},
		{1.0, 3.0, 8.0},
		{-1.0, -3.0, 8.0},
		{2.5, -2.5, 10.0},
	};

	const auto ptsTrue{hw3::projectPoints(scenePoints, P)};




}

