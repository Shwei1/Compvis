#pragma once
#include <utility>
#include <vector>
#include <variant>

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif


namespace hw3 {

	struct LinearCoefficients {
		double a, b;
	};


	struct QuadraticCoefficients {
		double a, b, c;
	};


	struct ExponentialCoefficients {
		double a, b;
	};


	using FitResult = std::variant<std::monostate, LinearCoefficients, QuadraticCoefficients, ExponentialCoefficients>;

/* For task 1 */
	std::pair<double, double> linearEstimateRANSAC(const std::vector<std::pair<double, double>>& points,
			double tau, double p, std::size_t k=2);

/* For task 1 and 4 */
		std::pair<double, double> minimizeLinear(const std::vector<double>& X, const std::vector<double>& Y);


		std::pair<double, double> minimizeExponential(const std::vector<double>& X, const std::vector<double>& Y);


		FitResult estimateBestModelRANSAC(const std::vector<std::pair<double, double>>& points);


/* For task 3 */
#ifdef USE_OPENCV	
	std::vector<cv::Point2d> projectPoints(const std::vector<cv::Point3d>& points3D, const cv::Matx34d& projectionMat);


	struct Correspondence {
		cv::Point3d worldPoint;
		cv::Point2d imagePoint;
	};

	std::pair<cv::Matx33d, cv::Vec3d> estimateTransformRANSAC(const std::vector<Correspondence>& correspondences, const cv::Matx33d& K, double tau, double p=.99);

void testReprojection(const cv::Point3d& pt3d, const cv::Point2d& pt2d, const cv::Matx33d& R, const cv::Vec3d& t, const cv::Matx33d& K);

#endif


}
