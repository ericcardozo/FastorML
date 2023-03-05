#ifndef DATA_READER_H
#define DATA_READER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

std::vector<std::vector<float>> read_csv(const std::string& filename){
  std::ifstream file(filename);
  std::string line, header;
  std::vector<std::vector<float>> data;
  std::getline(file, header);
  while(std::getline(file, line)){
    std::vector<float> row;
    std::string cell;
    std::stringstream line_stream(line);
    while(std::getline(line_stream, cell, ',')){
      row.push_back(std::stof(cell));
    }
    data.push_back(row);
  }
  return data;
}

#endif