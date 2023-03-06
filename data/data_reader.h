#ifndef DATA_READER_H
#define DATA_READER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

std::pair<std::vector<std::vector<float>>, std::vector<int>> read(const std::string& filename){
  std::ifstream file(filename);
  std::string line, header;
  std::vector<std::vector<float>> features;
  std::vector<int> targets;

  std::getline(file, header); // Skip header

  features.reserve(10000);
  targets.reserve(10000);

  while (std::getline(file, line)) {
    std::vector<float> image;
    std::istringstream iss(line);
    std::string field;
    std::getline(iss, field, ','); // Extract label from first field
    targets.emplace_back(std::stoi(field));
    while (std::getline(iss, field, ',')) { // Extract image data from remaining fields
      image.emplace_back(std::stof(field));
    }
    features.emplace_back(std::move(image));
  }
  file.close();
  return std::make_pair(std::move(features), std::move(targets));
}

#endif