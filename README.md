# triton backend for fastllm
fastllm is a large language model inference speeding up solution  
this repo is integration of fastllm and triton backend, just like fastertransformer triton backend  
this repo is also a good example for how to develop a triton c++ custom backend, with string in and string out,   
and how to use triton dynamic batch support  

## build
### 1. download dependencies
download these repos to local folder  
https://github.com/triton-inference-server/backend  
https://github.com/triton-inference-server/core  
https://github.com/triton-inference-server/common  
https://github.com/ztxz16/fastllm  

### 2. build fastllm
build fastllm, see fastllm readme and you can get a lib file: libfastllm.so

### 3. modify my CMakeLists.txt
open CMakeLists.txt of this repo, modify:  
1. modify SOURCE_DIR field to the local path where you downloaded.  
FetchContent_Declare(  
    repo-common  
    PREFIX repo-common  
    SOURCE_DIR ../../common-main  
)  
FetchContent_Declare(  
    repo-core  
    PREFIX repo-core  
    SOURCE_DIR ../../core-main  
)  
FetchContent_Declare(  
    repo-backend  
    PREFIX repo-backend  
    SOURCE_DIR ../../backend-main  
)  

2. modify /models/glm/fastllm-master/build/libfastllm.so to where you build fastllm libfastllm.so  
target_link_libraries(  
    triton_glmbackend  
    PRIVATE  
    triton-core-serverapi  
    triton-core-backendapi  
    triton-core-serverstub  
    triton-backend-utils  
    /models/glm/fastllm-master/build/libfastllm.so  
)  

### 4. build my repo
cd to-my-repo-root-path  
mkdir build  
cd build  
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install .. -Wno-dev  
make install  
then you will get file: libtriton_glmbackend.so  

## deploy to triton inference server
cd to-triton-inference-server-repository-path  
ask chatgpt if you don't know what is triton inference server repository path  
mkdir glmmodel  
cd glmmodel  
cp config.pbtxt-in-my-repo .  
mkdir 1  
cd 1  
cp libtriton_glmbackend.so-which-you-just-built  
start triton inference server  

## request to triton inference server
just http post req:  
url: http://ip:8000/v2/models/glmmodel/infer  
body:  
{  
    inputs: [  
        {  
            "name": "PROMPT",  
            "shape": [1,1],  
            "datatype":"BYTES",  
            "data":["are you a large language model?"]  
        },  
        {  
            "name": "RESPONSE_LIMIT",  
            "shape": [1,1],  
            "datatype":"INT32",  
            "data": [2048]  
        }  
    ]  
}
