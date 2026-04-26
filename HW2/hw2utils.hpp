#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

namespace hw2 {
	std::pair<cv::Mat, cv::Mat> findPose(const std::vector<cv::Point2d>& pts1,
			const std::vector<cv::Point2d>& pts2, const cv::Matx33d& K); 

	std::pair<cv::Mat, cv::Mat> findPosePnP(const std::vector<cv::Point3d>& ptsReal, const std::vector<cv::Point2d>& pts2, const cv::Matx33d& K);

	cv::Matx33d GramSchmidtProcess(const cv::Matx33d& R);

	std::vector<cv::Point2d> projectPoints(const std::vector<cv::Point3d>& points3D, const cv::Matx34d& projectionMat);

		cv::Mat findEssentialMat(const std::vector<cv::Point2d>& pts1,
				const std::vector<cv::Point2d>& pts2, const cv::Matx33d& K);
}
