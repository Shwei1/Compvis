#include <ceres/ceres.h>
#include <iostream>
#include <print>

template<typename T>
concept Addable = requires(T x, T y) {
	{ x + y } -> std::convertible_to<T>;
};


struct Foo1 {
	template<Addable T>
	bool operator()(const T* parameters, T* cost) const {
		const T x{parameters[0]};
		const T y{parameters[1]};
		const T z{parameters[2]};
		cost[0] = ceres::sin(x) + ceres::cos(x) + (z-2.)*(z-2.);
		return true;
	}

	static ceres::FirstOrderFunction* create() {
		constexpr int kNumParameters{3};
		auto* functor{new Foo1{}};
		return new ceres::AutoDiffFirstOrderFunction<Foo1, kNumParameters>{functor};
	}
};


struct Foo2 {
	template<Addable T>
	bool operator()(const T* parameters, T* cost) const {
		const T x{parameters[0]};
		const T y{parameters[1]};
		const T z{parameters[2]};
		cost[0] = (1.-x)*(1.-x) + 100. * (y-x*x)*(y-x*x) + (z-1.)*(z-1.);
		return true;
	}

	static ceres::FirstOrderFunction* create() {
		constexpr int kNumParameters{3};
		auto* functor{new Foo2{}};
		return new ceres::AutoDiffFirstOrderFunction<Foo2, kNumParameters>{functor};
	}

};

int main(int argc, char* argv[]) {
	std::println("Function 1:");

	{
	std::vector<double> parameters{11., 3., -0.52};
	auto copied{parameters};


	ceres::GradientProblem problem{Foo1::create()};

	ceres::GradientProblemSolver::Options options{};
	options.minimizer_progress_to_stdout = true;

	ceres::GradientProblemSolver::Summary summary{};

	ceres::Solve(options, problem, parameters.data(), &summary);
	
	std::cout << summary.FullReport() << std::endl;


	std::println("Initial parameters:");
	for (double &v : copied) std::println("{} ", v);
	std::println("Final parameters:");
	for (double &v : parameters) std::println("{} ", v);
	}

	std::println("Function 2:");
	
	std::vector<double> parameters{0.55, 10., -3.};
	auto copied{parameters};

	ceres::GradientProblem problem{Foo2::create()};

	ceres::GradientProblemSolver::Options options{};
	options.minimizer_progress_to_stdout = true;

	ceres::GradientProblemSolver::Summary summary{};

	ceres::Solve(options, problem, parameters.data(), &summary);

	std::cout << summary.FullReport() << std::endl;
	
	std::println("Initial parameters:");
	for (double &v : copied) std::println("{} ", v);
	std::println("Final parameters:");
	for (double &v : parameters) std::println("{} ", v);

	return 0;
}
