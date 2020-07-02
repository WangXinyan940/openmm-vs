#include "TFSingleton.h"
#include <vector>
#include <string>
#include <map>
#include <iostream>

using namespace std;

TFSingleton::TFSingleton(){
    string init_string("");
    modelPath = init_string;
}


vector<float> TFSingleton::calcVSPos(string modelName, int vsid, vector<float> vspos){
    if(modelPath != modelName){
        TFUtils::STATUS status = TFU.LoadModel(modelName);
        if (status == TFUtils::SUCCESS) {
            modelPath = modelName;
        }
    }
    // init output vectors
    vector<float> outCrd(9);
    vector<float> outGrad(27);
    // calcVS_TF to get result
    calcVS_TF(vspos, outCrd, outGrad);
    // save VSGrad to gradMem
    gradMem[vsid] = outGrad;
    // return pos
    return outCrd;
}

vector<float> TFSingleton::calcVSGrad(int vsid){
    // call gradMem to get the grad
    map<int,vector<float>>::iterator grad_it = gradMem.find(vsid);
    if(grad_it == gradMem.end()){
        // raise exception
        throw "Please calculate virtual site position before calc force.";
    };
    vector<float> grad = gradMem[vsid];
    return grad;
}

void TFSingleton::calcVS_TF(const vector<float>& inputCrd, vector<float>& outputCrd, vector<float>& outputGrad){
    // build input dim
    const vector<int64_t> input_dims = {1, 9};
    // build input tensor
    const vector<float> input_vals(inputCrd);
    const vector<TF_Tensor*> input_tensors = {TFUtils::CreateTensor(TF_FLOAT, input_dims, input_vals)};
    // build input op
    const vector<TF_Output> input_ops = {TFU.GetOperationByName("positions", 0)};
    // build output op
    const vector<TF_Output> output_ops = {TFU.GetOperationByName("vs", 0), TFU.GetOperationByName("dmx_dp", 0), TFU.GetOperationByName("dmy_dp", 0), TFU.GetOperationByName("dmz_dp", 0)};
    // build output tensor
    vector<TF_Tensor*> output_tensors = {nullptr, nullptr, nullptr, nullptr};
    TFUtils::STATUS status = TFU.RunSession(input_ops, input_tensors,
                                              output_ops, output_tensors);
    const vector<vector<float>> data = TFUtils::GetTensorsData<float>(output_tensors);
    outputCrd[0] = data[0][0];
    outputCrd[1] = data[0][1];
    outputCrd[2] = data[0][2];
    for(int i=0;i<9;i++){
        outputGrad[i] = data[1][i];
    }
    for(int i=0;i<9;i++){
        outputGrad[i+9] = data[2][i];
    }
    for(int i=0;i<9;i++){
        outputGrad[i+18] = data[3][i];
    }
}