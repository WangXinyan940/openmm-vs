#ifndef OPENMM_TFSINGLETON_H_
#define OPENMM_TFSINGLETON_H_


#include "TFUtils.h"
#include <vector>
#include <string>
#include <map>


class  TFSingleton {
public:
    static TFSingleton& getInstance(){
        static TFSingleton instance;
        return instance;
    };
    std::vector<float> calcVSPos(std::string, int, std::vector<float>);
    std::vector<float> calcVSGrad(int);
private:
    TFSingleton();
    TFUtils TFU;
    std::string modelPath;
    std::map<int,std::vector<float>> gradMem;
    void calcVS_TF(const std::vector<float>& inputCrd, std::vector<float>& outputCrd, std::vector<float>& outputGrad);
};



#endif /*OPENMM_TFSINGLETON_H_*/