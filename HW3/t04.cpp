#include "hw3utils.hpp"
#include <print>
#include <ranges>
#include <random>

std::mt19937& getRNG(void) {
	static std::random_device dev{};
	static std::mt19937 rng{dev()};
	return rng;
}


void testLinear(){
	std::println("\033[1;32m=== Test 1: linear model, noise and outliers ===\033[0m");
	auto&& rng{getRNG()};
    std::uniform_real_distribution<double> dist{-10., 10.};
    std::normal_distribution<double> noise{0., 1.};
	std::uniform_real_distribution<double> prob{0.0, 1.0};
	std::uniform_real_distribution<double> outlierY{-100.0, 100.0};

	constexpr double aTrue{5.8};
	constexpr double bTrue{0.44};


	std::vector<double> X(50);
	std::vector<double> Y(50);
	
	const auto genX = [&](){
		return dist(rng);
	};

	std::size_t i{0};
	const auto genY = [&](){
		return prob(rng) < 0.3 ? i++, outlierY(rng) : aTrue * X[i++] + bTrue + noise(rng);
	};

	std::generate(X.begin(), X.end(), genX);
	std::generate(Y.begin(), Y.end(), genY);

	std::vector<std::pair<double, double>> points{
		std::views::zip(X, Y) | std::ranges::to<decltype(points)>()
	};

	const double tau{3.};
	auto&& estimate{hw3::estimateBestModelRANSAC(points, tau)};

	std::println("Real model: y={}x+{}", aTrue, bTrue);

	std::print("Estimated model: ");
	std::println("{}", hw3::toString(estimate));
}

void testQuadratic() {
	std::println("\033[1;32m=== Test 2: quadratic model, noise and outliers ===\033[0m");
	auto&& rng{getRNG()};
    std::uniform_real_distribution<double> dist{-10., 10.};
    std::normal_distribution<double> noise{0., 1.};
	std::uniform_real_distribution<double> prob{0.0, 1.0};
	std::uniform_real_distribution<double> outlierY{-100.0, 100.0};

	constexpr double aTrue{3.};
	constexpr double bTrue{.22};
	constexpr double cTrue{.4};


	std::vector<double> X(50);
	std::vector<double> Y(50);
	
	const auto genX = [&](){
		return dist(rng);
	};

	auto i{0uz};
	const auto genY = [&](){
		auto val{prob(rng) < 0.3 ? outlierY(rng) : aTrue * X[i] * X[i] + bTrue * X[i] + cTrue + noise(rng)};
		++i;
		return val;
	};

	std::generate(X.begin(), X.end(), genX);
	std::generate(Y.begin(), Y.end(), genY);

	std::vector<std::pair<double, double>> points{
		std::views::zip(X, Y) | std::ranges::to<decltype(points)>()
	};

	const double tau{3.};
	auto&& estimate{hw3::estimateBestModelRANSAC(points, tau)};

	
	std::println("Real model: y={}x^2+{}x+{}", aTrue, bTrue, cTrue);
	std::print("Estimated model: ");
	std::println("{}", hw3::toString(estimate));
}


void testExponential() {
	std::println("\033[1;32m=== Test 3: exponential model, x range (1, 4), noise and outliers ===\033[0m");
	auto&& rng{getRNG()};
    std::uniform_real_distribution<double> dist{1., 4.};
    std::normal_distribution<double> noise{0., 0.5};
	std::uniform_real_distribution<double> prob{0., 1.};
	std::uniform_real_distribution<double> outlierY{-50., 50.};

	constexpr double aTrue{5.};
	constexpr double bTrue{1.5};


	std::vector<double> X(50);
	std::vector<double> Y(50);
	
	const auto genX = [&](){
		return dist(rng);
	};

	auto i{0uz};
	const auto genY = [&](){
		auto val{prob(rng) < 0.3 ? outlierY(rng) : aTrue * std::exp(bTrue * X[i]) + noise(rng)};
		++i;
		return val;
	};

	std::generate(X.begin(), X.end(), genX);
	std::generate(Y.begin(), Y.end(), genY);

	std::vector<std::pair<double, double>> points{
		std::views::zip(X, Y) | std::ranges::to<decltype(points)>()
	};

	const double tau{5.};
	auto&& estimate{hw3::estimateBestModelRANSAC(points, tau)};

	std::println("Real model: y={}*e^({}x)", aTrue, bTrue);

	std::print("Estimated model: ");
	std::println("{}", hw3::toString(estimate));
}

void testExponential2() {
	std::println("\033[1;32m=== Test 4: exponential model, x range (-3, 3), noise and outliers ===\033[0m");
	auto&& rng{getRNG()};
    std::uniform_real_distribution<double> dist{-3., 3.};
    std::normal_distribution<double> noise{0., 0.5};
	std::uniform_real_distribution<double> prob{0., 1.};
	std::uniform_real_distribution<double> outlierY{-50., 50.};

	constexpr double aTrue{5.};
	constexpr double bTrue{1.5};


	std::vector<double> X(50);
	std::vector<double> Y(50);
	
	const auto genX = [&](){
		return dist(rng);
	};

	auto i{0uz};
	const auto genY = [&](){
		auto val{prob(rng) < 0.3 ? outlierY(rng) : aTrue * std::exp(bTrue * X[i]) + noise(rng)};
		++i;
		return val;
	};

	std::generate(X.begin(), X.end(), genX);
	std::generate(Y.begin(), Y.end(), genY);

	std::vector<std::pair<double, double>> points{
		std::views::zip(X, Y) | std::ranges::to<decltype(points)>()
	};

	const double tau{5.};
	auto&& estimate{hw3::estimateBestModelRANSAC(points, tau)};

	std::println("Real model: y={}*e^({}x)", aTrue, bTrue);

	std::print("Estimated model: ");
	std::println("{}", hw3::toString(estimate));
}


void badTest() {
	std::println("\033[1;32m=== Test 5: rubbish data ===\033[0m");
	auto&& rng{getRNG()};

    std::uniform_real_distribution<double> dist{10., 40.};
	std::uniform_real_distribution<double> outlierY{-50., 50.};

	std::vector<double> X(50);
	std::vector<double> Y(50);

	std::generate(X.begin(), X.end(), [&](){return dist(rng);});
	std::generate(Y.begin(), Y.end(), [&](){return outlierY(rng);});

	std::vector<std::pair<double, double>> points{
		std::views::zip(X, Y) | std::ranges::to<decltype(points)>()
	};

	const double tau{5.};
	auto&& estimate{hw3::estimateBestModelRANSAC(points, tau)};
	
	std::print("Estimated model: ");
	std::println("{}", hw3::toString(estimate));
}




int main(int argc, char* argv[]) { 
	extern bool logWarnings;
	logWarnings = false;

	testLinear();
	std::println("");
	testQuadratic();
	std::println("");
	testExponential();
	std::println("");
	testExponential2();
	std::println("");
	badTest();
	return 0;	
}
