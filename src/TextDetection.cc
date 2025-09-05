// TextDetectionsFromFile.cc
#include "TextDetection.h"
#include <fstream>
#include <sstream>
#include <experimental/filesystem>
#include <iostream>

namespace fs = std::experimental::filesystem;

namespace ORB_SLAM2 {

TextDetectionsFromFile::TextDetectionsFromFile(const std::string &folder) : folder_(folder) {}

std::vector<TextDetection::Ptr> TextDetectionsFromFile::detect(const std::string &image_filename) const {
    std::vector<TextDetection::Ptr> results;

    std::string basename = fs::path(image_filename).stem(); // "00001" from "00001.png"
    std::string txt_file = folder_ + "/" + basename + "_quads.txt";
    std::string words_file = folder_ + "/" + basename + "_texts.txt";

    std::ifstream fin_txt(txt_file), fin_words(words_file);
    if (!fin_txt.is_open() || !fin_words.is_open()) {
        std::cerr << "⚠️ Missing OCR files for " << basename << std::endl;
        return results;
    }

    std::vector<std::vector<cv::Point2f>> quads;
    std::string line;
    while (std::getline(fin_txt, line)) {
        std::stringstream ss(line);
        std::vector<float> coords(8);
        for (int i = 0; i < 8; ++i) ss >> coords[i];
        std::vector<cv::Point2f> quad = {
            {coords[0], coords[1]},
            {coords[2], coords[3]},
            {coords[4], coords[5]},
            {coords[6], coords[7]}
        };
        quads.push_back(quad);
    }

    std::vector<std::pair<std::string,float>> words;
    while (std::getline(fin_words, line)) {
        std::stringstream ss(line);
        std::string word; float score;
        if (std::getline(ss, word, ',') && (ss >> score))
            words.push_back({word, score});
    }

    size_t n = std::min(quads.size(), words.size());
    for (size_t i = 0; i < n; ++i) {
        auto det = std::make_shared<TextDetection>();
        det->quad = quads[i];
        det->text = words[i].first;
        det->score = words[i].second;
        results.push_back(det);
    }

    return results;
}

}
