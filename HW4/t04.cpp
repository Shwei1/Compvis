#include <vector>
#include <opencv2/opencv.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Projection_traits_xy_3.h>
#include <CGAL/Delaunay_triangulation_2.h>

using CGALKernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Gt = CGAL::Projection_traits_xy_3<CGALKernel>;
using Delaunay = CGAL::Delaunay_triangulation_2<Gt>;
using CGALPoint3 = CGALKernel::Point_3;


std::vector<cv::Point3d> projectPoints(const std::vector<cv::Point3d>& points3D, const cv::Matx34d& projectionMat) {
	std::vector<cv::Point3d> res{};

	for (const auto& p3d : points3D) {
		cv::Vec4d X{p3d.x, p3d.y, p3d.z, 1.0};

		cv::Vec3d x1{projectionMat * X};
		res.emplace_back(x1(0)/x1(2), x1(1)/x1(2), x1(2));
	}

	return res;
}


int main(int argc, char* argv[]) {
	const std::vector<cv::Point3d> points3D{{0.0,  0.0,  5.0},
		{1.0, 0.5, 6.0},
		{-1.0, 1.0, 7.0},
		{2.0, -1.0, 5.5},
		{-2.0, -0.5, 6.5},
		{0.5, 2.0, 8.0},
		{-1.5, -2.0, 7.5},
		{1.5, 1.5, 9.0}
	};

    constexpr double fx{300.}, fy{300.}, cx{320.}, cy{320.};
    const cv::Matx33d K1{fx, 0., cx, 0., fy, cy, 0., 0., 1.};

	const cv::Vec3d r1Vec{0.25, 0.33, 0.8};
	cv::Mat R_W_C1_{};
	cv::Rodrigues(r1Vec, R_W_C1_);
	const cv::Matx33d R_W_C1{R_W_C1_};
	const cv::Vec3d camera1Coords{3.0, 3.0, 3.0};
	const cv::Vec3d t_W_C1{-R_W_C1 * camera1Coords};
	cv::Matx34d T_W_C1{};
	cv::hconcat(R_W_C1, t_W_C1, T_W_C1);
	const cv::Matx34d P1A{K1*T_W_C1};

	const auto projectedPoints{projectPoints(points3D, P1A)};
	Delaunay dt{};

	for (const auto& [u, v, depth] : projectedPoints) {
		dt.insert(CGALPoint3{u, v, depth});
	}

	cv::Mat canvas{2300, 2300, CV_8UC3, {30, 30, 30}};
	for (auto f{dt.finite_faces_begin()}; f != dt.finite_faces_end(); f++) {
		std::array<cv::Point2i, 3> pts;
		for (int i = 0; i < 3; ++i) {
			pts[i] = {
				(int)f->vertex(i)->point().x() + 500,  // u
				(int)f->vertex(i)->point().y() + 1000   // v
			};
		}
		cv::line(canvas, pts[0], pts[1], {0, 255, 0}, 1);
		cv::line(canvas, pts[1], pts[2], {0, 255, 0}, 1);
		cv::line(canvas, pts[2], pts[0], {0, 255, 0}, 1);
	}

	cv::imshow("Triangulated points", canvas);
	cv::waitKey(0);

	return 0;
}
