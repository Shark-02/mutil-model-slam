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
        // 假设格式是: TEXT,score  （注意你提供的样例里是 "WORD,0.933..."; 若不同请调整解析）
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
        double xmin=1e9, ymin=1e9, xmax=-1e9, ymax=-1e9;
        for (auto &p : quads[i]) {
            xmin = std::min(xmin, p.x());
            ymin = std::min(ymin, p.y());
            xmax = std::max(xmax, p.x());
            ymax = std::max(ymax, p.y());
        }
        BBox2 bb;
        bb << xmin, ymin, xmax, ymax;

        // category_id: 给一个特殊类别号（比如 1000），或 0。你后续可在 Tracking 判定 text 类别
        unsigned int text_cat = 1000;

        auto det = std::make_shared<Detection>(text_cat, words[i].second, bb,
                                               words[i].first, words[i].second, quads[i]);
        results.push_back(det);
    }

    return results;
}

} // namespace ORB_SLAM2
