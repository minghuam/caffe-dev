#include "caffe/util/directory.hpp"

#include "caffe/common.hpp"
#include "Poco/File.h"
#include "Poco/Path.h"

#include <algorithm>

namespace caffe{

void ls_all(std::vector<std::string> &files, \
            const std::string& path, \
            bool recursive){
    Poco::File dir(path);
    if(!dir.exists() || !dir.isDirectory()){
        LOG(ERROR) << "Invalid directory: " << path;
        return;
    }

    std::vector<Poco::File> poco_files;
    dir.list(poco_files);

    std::vector<Poco::File>::iterator it = poco_files.begin();
    while(it != poco_files.end()){
        if(it->isHidden()){
            ++it;
            continue;
        }
        files.push_back(it->path());
        if(it->isDirectory() && recursive){
            ls_all(files, it->path(), recursive);
        }

        it++;
    }

    std::sort(files.begin(), files.end());
}

void ls_files(std::vector<std::string>& files, \
                                  const std::string& path, \
                                  const std::string& extension, \
                                  bool recursive){
    ls_all(files, path, recursive);

    std::vector<std::string>::iterator it = files.begin();
    while(it != files.end()){
        Poco::Path p(*it);
        if(p.getExtension() != extension){
            it = files.erase(it);
        }else{
            ++it;
        }
    }

    std::sort(files.begin(), files.end());
}

} // end of namespace
