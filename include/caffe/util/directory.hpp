#ifndef DIRECTORY_HPP
#define DIRECTORY_HPP

#include <vector>
#include <string>

namespace caffe{

void ls_all(std::vector<std::string> &files, \
            const std::string& path, \
            bool recursive = false);

void ls_files(std::vector<std::string> &files, \
              const std::string& path, \
              const std::string& extension, \
              bool recursive = false);

}

std::string join_path(std::string &parent, std::string &child);

#endif // DIRECTORY_HPP

