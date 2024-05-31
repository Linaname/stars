#include <iostream>
#include <cstdint>
#include <vector>

#include <opencv2/opencv.hpp>
#include <random>
#include <optional>

struct Star {
    using TCoords = cv::Vec3d;
    using TColor = cv::Vec3d;
    TCoords Coords;
    TColor Color;
};

Star::TColor RandomColor() {
    std::random_device rd;
    std::minstd_rand0 gen(rd());
    std::uniform_real_distribution<Star::TColor::value_type> distrib(0.001, 1);
    auto c1 = distrib(gen);
    auto c2 = distrib(gen);
    auto c3 = distrib(gen);
    Star::TColor::value_type scale = 255. / (c1 + c2 + c3);
    Star::TColor color = {c1*scale, c2*scale, c3*scale};
    return color;
}

struct Cube {
    int32_t XFrom;
    int32_t XTo;
    int32_t YFrom;
    int32_t YTo;
    int32_t ZFrom;
    int32_t ZTo;
};

std::vector<Star> CreateStar(size_t n, const Cube& area) {
    std::vector<Star> ret;
    ret.reserve(n);
    std::random_device rd;
    std::minstd_rand0 gen(rd());
    std::uniform_real_distribution<> distribX(area.XFrom, area.XTo);
    std::uniform_real_distribution<> distribY(area.YFrom, area.YTo);
    std::uniform_real_distribution<> distribZ(area.ZFrom, area.ZTo);
    for (size_t i = 0; i < n; ++i) {
        Star::TCoords coords = {distribX(gen), distribY(gen), distribZ(gen)};
        Star::TColor color = RandomColor();
        ret.push_back({coords, color});
    }
    return ret;
}

std::optional<cv::Point2i> ImagePosition(const Star::TCoords& coords, cv::Size imsize, int f) {
    if (coords[2] <= 0) {
        return std::nullopt;
    }
    double scale = (double)f / coords[2];
    cv::Point2i proj = cv::Point2i((int)(coords[0]*scale + imsize.width / 2.), (int)(coords[1]*scale + imsize.height / 2.));
    if (!cv::Rect2i(cv::Point2i{0, 0}, imsize).contains(proj)) {
        return std::nullopt;
    }
    return proj;
}

double SqrDist(const Star::TCoords& point) {
    return point[0]*point[0] + point[1]*point[1] + point[2]*point[2];
}

int main(int, char**){
    Cube area = {
        -100000, 100000,
        -2000, 2000,
        0, 30000
    };
    auto stars = CreateStar(4800000, area);
    Star::TCoords offsetPoint = {0, 0, 0};
    for (int offset = 0; offset <= area.ZTo; offset += 50) {
        offsetPoint[2] = offset;
        cv::Mat im = cv::Mat::zeros(512, 512, CV_64FC3);
        for (size_t i=0; i<stars.size(); ++i) {
            auto& star = stars[i];
            auto proj = ImagePosition(star.Coords - offsetPoint, im.size(), 500);
            if (!proj.has_value()) {
                continue;
            }
            double brightness = 100000 / SqrDist(star.Coords - offsetPoint);
            im.at<Star::TColor>(*proj) += star.Color * brightness;
        }
        cv::GaussianBlur(im, im, {25, 25}, 0.5);
        cv::imshow("main", im);
        if (cv::waitKey(10) != -1) {
            return 0;
        }
    }
    return 0;
}
