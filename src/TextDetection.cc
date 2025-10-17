#include "TextDetection.h"
#include <fstream>
#include <sstream>
#include <experimental/filesystem>
#include <iostream>

namespace fs = std::experimental::filesystem;

namespace ORB_SLAM2 {

std::vector<Detection::Ptr> TextDetectionsFromFile::detect(const std::string &image_filename) const {
    std::vector<Detection::Ptr> results;
    std::string basename = fs::path(image_filename).stem(); // e.g. "000015"
    std::string txt_file = folder_ + "/" + basename + "_quads.txt";
    std::string words_file = folder_ + "/" + basename + "_texts.txt";

    std::ifstream fin_txt(txt_file);
    std::ifstream fin_words(words_file);
    if (!fin_txt.is_open() || !fin_words.is_open()) {
        // 没有 OCR 文件，返回空
        // std::cerr << "No OCR files for " << basename << std::endl;
        return results;
    }

    // 读 quads
    std::vector<std::vector<Eigen::Vector2d>> quads;
    std::string line;
    while (std::getline(fin_txt, line)) {
        if(line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::stringstream ss(line);
        std::vector<double> coords(8);
        for (int i = 0; i < 8; ++i) ss >> coords[i];
        std::vector<Eigen::Vector2d> quad;
        quad.push_back(Eigen::Vector2d(coords[0], coords[1]));
        quad.push_back(Eigen::Vector2d(coords[2], coords[3]));
        quad.push_back(Eigen::Vector2d(coords[4], coords[5]));
        quad.push_back(Eigen::Vector2d(coords[6], coords[7]));
        quads.push_back(quad);
    }

    // 读 words
    std::vector<std::pair<std::string,double>> words;
    while (std::getline(fin_words, line)) {
        if(line.empty()) continue;
        std::stringstream ss(line);
        std::string word; double score;
        // 格式是: TEXT,score
        if (std::getline(ss, word, ',') && (ss >> score)) {
            // trim spaces from word
            while(!word.empty() && isspace(word.back())) word.pop_back();
            while(!word.empty() && isspace(word.front())) word.erase(word.begin());
            words.push_back({word, score});
        } else {
            // 兼容可能只有单词一行（无score）
            if(!word.empty()){
                words.push_back({word, 1.0});
            }
        }
    }

    // 组合 (以最小长度为准)
    size_t n = std::min(quads.size(), words.size());
    for (size_t i = 0; i < n; ++i) {
        // 计算 quad 对应的 bbox (min/max)
        double xmin = std::numeric_limits<double>::max();
        double ymin = std::numeric_limits<double>::max();
        double xmax = std::numeric_limits<double>::lowest();
        double ymax = std::numeric_limits<double>::lowest();
        
        // 打印quad的所有点坐标
        //std::cout << "Quad " << i << " points: ";
        for (auto &p : quads[i]) {
            //std::cout << "(" << p.x() << ", " << p.y() << ") ";
            xmin = std::min(xmin, static_cast<double>(p.x()));
            ymin = std::min(ymin, static_cast<double>(p.y()));
            xmax = std::max(xmax, static_cast<double>(p.x()));
            ymax = std::max(ymax, static_cast<double>(p.y()));
        }
        //std::cout << std::endl;
        
        // 检查计算结果
        if (ymin == ymax && ymin == 0) {
            std::cerr << "WARNING: Quad " << i << " has all y-coordinates = 0!" << std::endl;
        }
        
        BBox2 bb;
        bb << xmin, ymin, xmax, ymax;

        // 调试输出bbox
        //std::cout << "Calculated bbox: " << xmin << ", " << ymin << ", " << xmax << ", " << ymax << std::endl;

        // 转换 quad 点到 Eigen::Vector2d
        std::vector<Eigen::Vector2d> quad_pts;
        for (auto &p : quads[i]) {
            quad_pts.emplace_back(p.x(), p.y());
        }   

        // category_id: 给一个特殊类别号（比如 1000），或 0。可在 Tracking 判定 text 类别
        unsigned int text_cat = 1000;

        auto det = std::make_shared<Detection>(text_cat, words[i].second, bb,
                                               words[i].first, quads[i]);
        results.push_back(det);
    }

    return results;
}

} // namespace ORB_SLAM2
