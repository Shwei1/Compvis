#include <opencv2/opencv.hpp>
#include <iostream>
#include <utility>
#include <vector>

std::pair<cv::Mat, cv::Mat> findPose(const std::vector<cv::Point2d>& pts1,
		const std::vector<cv::Point2d>& pts2, const cv::Matx33d K) {
	auto E{cv::findEssentialMat(pts1, pts2, K)};

	cv::Mat R{}, t{};
	cv::recoverPose(E, pts1, pts2, K, R, t);
	return {R, t};
}


int main(int argc, char* argv[]) {

    cv::Vec3d rvec{0.1, -0.2, 0.3}, t{1.1, 2.0, 3.};
    cv::Affine3d affineTransform{rvec, t};
    double fx{300.}, fy{300.}, cx{320.}, cy{320.};
    cv::Matx33d K{fx, 0., cx, 0., fy, cy, 0., 0., 1.};

	cv::Mat Rt1;
	cv::hconcat(K, cv::Mat::zeros(3, 1, CV_64F), Rt1);
	cv::Mat P1{Rt1}; /* Since it is the reference frame*/

	cv::Mat Rt2{cv::Mat{affineTransform.matrix}.rowRange(0, 3)}; /* [R|t] */
	cv::Mat P2{K * Rt2}; /* actual tranformation */

	std::vector<cv::Point3d> points3D{{0.0,  0.0,  5.0},
		{1.0, 0.5, 6.0},
		{-1.0, 1.0, 7.0},
		{2.0, -1.0, 5.5},
		{-2.0, -0.5, 6.5},
		{0.5, 2.0, 8.0},
		{-1.5, -2.0, 7.5},
		{1.5, 1.5, 9.0}
	};

	std::vector<cv::Point2d> pts1{}, pts2{};

	for (const auto& p3d : points3D) {
		cv::Mat X{(cv::Mat_<double>(4, 1) << p3d.x, p3d.y, p3d.z, 1.0)};

		cv::Mat x1{P1 * X};
		pts1.emplace_back(x1.at<double>(0) / x1.at<double>(2),
				x1.at<double>(1) / x1.at<double>(2));

		cv::Mat x2{P2 * X};
		pts2.emplace_back(x2.at<double>(0) / x2.at<double>(2),
				x2.at<double>(1) / x2.at<double>(2));
	}

	auto [R_ret, t_ret]{findPose(pts1, pts2, K)};

	cv::Mat rvec_ret{};
	cv::Rodrigues(R_ret, rvec_ret);

	std::cout << "Rotation matrix:\n";
	std::cout << R_ret << "\n";
	
	std::cout << "Rotation vector\n";
	std::cout << rvec_ret << "\n";

	std::cout << "Translation direction:\n";
	std::cout << t_ret << "\n";

	cv::Mat points4D{};
	cv::triangulatePoints(P1, P2, pts1, pts2, points4D);

	std::vector<cv::Point3d> points3D_ret{};
	for (auto i{0uz}; i < points4D.cols; i++) {
		cv::Mat col{points4D.col(i)};
		col /= col.at<double>(3);
		points3D_ret.emplace_back(
				col.at<double>(0),
				col.at<double>(1),
				col.at<double>(2)
				);
	}

	// const double scale{cv::norm(t)/cv::norm(t_ret)};
	constexpr double scale{1.}; // Seems like OpenCV finds the scale for me
	std::cout << scale << "\n";

	for (auto& pt : points3D_ret) {
		pt.x *= scale;
		pt.y *= scale;
		pt.z *= scale;
	}

	for (auto i{0uz}; i < points4D.cols; i++) {
		std::cout << "Point " << i << "\n";
		std::cout << "Ground truth: " << points3D[i] << "\n";
		std::cout << "Retrieved: " << points3D_ret[i] << "\n";
	}

	return 0;
}
