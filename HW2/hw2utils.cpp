#include "hw2utils.hpp"
#include <algorithm>

std::pair<cv::Mat, cv::Mat> hw2::findPose(const std::vector<cv::Point2d>& pts1,
		const std::vector<cv::Point2d>& pts2, const cv::Matx33d& K) {
	auto E{cv::findEssentialMat(pts1, pts2, K)};

	cv::Mat R{}, t{};
	cv::recoverPose(E, pts1, pts2, K, R, t);
	return {R, t};
}


std::pair<cv::Mat, cv::Mat> hw2::findPosePnP(const std::vector<cv::Point3d>& ptsReal, const std::vector<cv::Point2d>& pts2, const cv::Matx33d& K) {
	cv::Mat r{}, t{};
	cv::solvePnP(ptsReal, pts2, K, cv::noArray(), r, t);
	return {r, t};
}


cv::Matx33d hw2::GramSchmidtProcess(const cv::Matx33d& R) {
	cv::Matx33d res{};
	cv::Vec3d r1{R(0,0), R(1,0), R(2,0)};
    cv::Vec3d r2{R(0,1), R(1,1), R(2,1)};
	cv::Vec3d r3{};


	r1 /= cv::norm(r1);
	r2 = r2 - r1.dot(r2) * r1;
	r2 /= cv::norm(r2);
	r3 = r1.cross(r2);

    for (int i = 0; i < 3; i++) {
        res(i, 0) = r1(i);
        res(i, 1) = r2(i);
        res(i, 2) = r3(i);
    }
	return res;
}


std::vector<cv::Point2d> hw2::projectPoints(const std::vector<cv::Point3d>& points3D, const cv::Matx34d& projectionMat) {
	std::vector<cv::Point2d> res{};

	for (const auto& p3d : points3D) {
		cv::Vec4d X{p3d.x, p3d.y, p3d.z, 1.0};
	
		cv::Vec3d x1{projectionMat * X};
		res.emplace_back(x1(0)/x1(2), x1(1)/x1(2));
	}

	return res;
}


static cv::Mat eightPointAlgorithm(const std::vector<cv::Point3d>& pts1,
			const std::vector<cv::Point3d>& pts2) {
	std::array<std::array<double, 9>, 8> linSystem{};
	for (auto i{0uz}; i < 8; i++) {
		double y1{pts1[i].x};
		double y2{pts1[i].y};
		double y1t{pts2[i].x};
		double y2t{pts2[i].y};

		linSystem[i][0] = y1t*y1;
		linSystem[i][1] = y1t*y2;
		linSystem[i][2] = y1t;
		linSystem[i][3] = y2t*y1;
		linSystem[i][4] = y2t*y2;
		linSystem[i][5] = y2t;
		linSystem[i][6] = y1;
		linSystem[i][7] = y2;
		linSystem[i][8] = 1.;
	}

	cv::Mat Y(8, 9, CV_64F);
	std::memcpy(Y.ptr<double>(), linSystem.data()->data(), 9 * 8 * sizeof(double));

	cv::Mat W{}, U{}, Vt{};
	cv::SVD::compute(Y, W, U, Vt, cv::SVD::FULL_UV);

	cv::Mat e{Vt.row(8)};
	cv::Mat E{e.reshape(1, 3).clone()};
	
	cv::Mat W2{}, U2{}, Vt2{};
	cv::SVD::compute(E, W2, U2, Vt2);
	cv::Mat WDiag{(cv::Mat_<double>(3,3) << 1,0,0, 0,1,0, 0,0,0)};
	E = U2 * WDiag * Vt2;

	return E;
}


static cv::Vec3d homogenizePoint2D(const cv::Point2d& value) {
	return cv::Vec3d{value.x, value.y, 1.};
}


cv::Mat hw2::findEssentialMat(const std::vector<cv::Point2d>& pts1,
			const std::vector<cv::Point2d>& pts2, const cv::Matx33d& K) {

	std::vector<cv::Vec3d> homogenousPts1{};
	std::vector<cv::Vec3d> homogenousPts2{};

	std::transform(pts1.begin(), pts1.end(), std::back_inserter(homogenousPts1),
		homogenizePoint2D);
	std::transform(pts2.begin(), pts2.end(), std::back_inserter(homogenousPts2),
		homogenizePoint2D);

	std::vector<cv::Point3d> calibratedPts1{};
	std::vector<cv::Point3d> calibratedPts2{};

	std::transform(homogenousPts1.begin(), homogenousPts1.end(), std::back_inserter(calibratedPts1),
			[&K](const cv::Vec3d& pt){
				cv::Vec3d calibPt{K.inv()*pt};
				return cv::Point3d{calibPt[0], calibPt[1], calibPt[2]};
			});

	std::transform(homogenousPts2.begin(), homogenousPts2.end(), std::back_inserter(calibratedPts2),
			[&K](const cv::Vec3d& pt){
				cv::Vec3d calibPt{ K.inv()*pt };
				return cv::Point3d{ calibPt[0], calibPt[1], calibPt[2] };
			});

	cv::Mat res{eightPointAlgorithm(calibratedPts1, calibratedPts2)};

	return res;
}
