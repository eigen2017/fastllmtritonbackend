#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"
#include <algorithm>
#include "model.h"
#include <iostream>

namespace triton::backend::glmbackend
{

#define RESPOND_ALL_AND_RETURN_FALSE_IF_ERROR(REQUESTS, REQUEST_COUNT, X)                   \
    do                                                                                      \
    {                                                                                       \
        TRITONSERVER_Error *rarie_err__ = (X);                                              \
        if (rarie_err__ != nullptr)                                                         \
        {                                                                                   \
            for (uint32_t r = 0; r < REQUEST_COUNT; ++r)                                    \
            {                                                                               \
                TRITONBACKEND_Request *request = REQUESTS[r];                               \
                TRITONBACKEND_Response *response;                                           \
                TRITONSERVER_Error *newRes = TRITONBACKEND_ResponseNew(&response, request); \
                if (newRes != nullptr)                                                      \
                {                                                                           \
                    TRITONSERVER_ErrorDelete(newRes);                                       \
                    continue;                                                               \
                }                                                                           \
                LOG_IF_ERROR(                                                               \
                    TRITONBACKEND_ResponseSend(                                             \
                        response, TRITONSERVER_RESPONSE_COMPLETE_FINAL,                     \
                        rarie_err__),                                                       \
                    "failed to send error response");                                       \
            }                                                                               \
            TRITONSERVER_ErrorDelete(rarie_err__);                                          \
            return false;                                                                   \
        }                                                                                   \
    } while (false);

    class ModelState : public BackendModel
    {
    public:
        static TRITONSERVER_Error *Create(TRITONBACKEND_Model *triton_model, ModelState **state);
        virtual ~ModelState() = default;

    private:
        ModelState(TRITONBACKEND_Model *triton_model) : BackendModel(triton_model) {}
    };

    TRITONSERVER_Error *
    ModelState::Create(TRITONBACKEND_Model *triton_model, ModelState **state)
    {
        try
        {
            *state = new ModelState(triton_model);
        }
        catch (const BackendModelException &ex)
        {
            RETURN_ERROR_IF_TRUE(
                ex.err == nullptr, TRITONSERVER_ERROR_INTERNAL,
                std::string("unexpected nullptr in BackendModelException"));
            RETURN_IF_ERROR(ex.err_);
        }

        return nullptr;
    }

    extern "C"
    {
        TRITONSERVER_Error *
        TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model *model)
        {
            ModelState *model_state;
            RETURN_IF_ERROR(ModelState::Create(model, &model_state));
            RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(model, reinterpret_cast<void *>(model_state)));
            return nullptr;
        }
        TRITONSERVER_Error *
        TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model *model)
        {
            void *vstate;
            RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
            ModleState *model_state = reinterpret_cast<ModelState *>(vstate);
            delete model_state;
            return nullptr;
        }

    } // extern "C"

    class FlmCall
    {
    public:
        FlmCall() {}
        void init()
        {
            const char *flmPathName = std::getenv("FLM_PATH_NAME");
            if (flmPathName == nullptr)
            {
                LOG_MESSAGE(TRITONSERVER_LOG_INFO, "FLM_PATH_NAME is not env var.");
                return;
            }
            std::string flmPathNameStr(flmPathName);
            flmModel = fastllm::CreateLLMModelFromFile(flmPathNameStr);
        }
        void predict(const std::vector<std::string> &inputs, int &output_token_limit, std::vector<std::string> &outputs)
        {
            fastllm::GenerationConfig generationConfig;
            generationConfig.output_token_limit = output_token_limit;

            flmModel->ResponseBatch(
                inputs, outputs, [](int index, std::vector<std::string> &contents) {}, generationConfig);
        }

    private:
        std::unique_ptr<fastllm::basellm> flmModel;
    };

    class ModelInstanceState : public BackendModelInstance
    {
    public:
        static TRITONSERVER_Error *Create(ModelState *model_state, TRITONBACKEND_ModelInstance *triton_model_instance, ModelInstanceState **state);
        virtual ~ModelInstanceState() = default;
        ModelState *StateForModel() const { return model_state_; }
        void ProcessRequest(TRITONBACKEND_Request **requests, const uint32_t request_count);
        TRITONSERVER_Error *byteListToStr(const char *byteList, const uint64_t &byteCnt, std::string &targetStr);
        TRITONSERVER_Error *byteListToInt(const char *byteList, const uint64_t &byteCnt, int &targetInt);
        bool parseInput(TRITONBACKEND_Request **requests, const uint32_t request_count, std::vector<std::string> &promptStrs, std::vector<int> &respLimitInts);
        bool getResponseStrs(const std::vector<std::string> &promptStrs, const std::vector<int> &respLimitInts, std::vector<std::string> &responseStrs);
        bool sendResp(TRITONBACKEND_Request **requests, const uint32_t request_count, std::vector<std::string> &responseStrs);
        TRITONSERVER_Error *returnErrorIfFalse(const bool &myAssert, const TRITONSERVER_Error_Code &errCode, const std::string &msg);
        std::string getRound(const int &round, const std::string &query, const std::string &answer);

    private:
        std::unique_ptr<FlmCall> flmCall;
        ModelInstanceState(ModelState *model_state, TRITONBACKEND_ModelInstance *triton_model_instance)
            : BackendModelInstance(model_state, triton_model_instance), model_state_(model_state)
        {
            flmCall.reset(new FlmCall());
            FlmCall->init();
        }
        ModelState *model_state_;
    };

    void ModelInstanceState::ProcessRequest(TRITONBACKEND_Request **requests, const uint32_t request_count)
    {
        std::vector<std::string> promptStrs;
        std::vector<int> respLimitInts;
        if (!parseInput(requests, request_count, promptStrs, respLimitInts))
        {
            return;
        }
        std::vector<std::string> responseStrs;
        if (!getResponseStrs(promptStrs, respLimitInts, responseStrs))
        {
            return;
        }
        sendResp(requests, request_count, getResponseStrs);
    }

    bool ModelInstanceState::sendResp(TRITONBACKEND_Request **requests, const uint32_t request_count, std::vector<std::string> &responseStrs)
    {
        if (request_count != (uint32_t)responseStrs.size())
        {
            return false;
        }
        std::vector<TRITONBACKEND_Response *> responses;
        responses.reserve(request_count);
        for (uint32_t r = 0; r < request_count; ++r)
        {
            TRITONBACKEND_Request *request = requests[r];
            std::string responseStr = responseStrs[r];
            TRITONBACKEND_Response *response;
            RESPOND_ALL_AND_RETURN_FALSE_IF_ERROR(requests, request_count, TRITONBACKEND_ResponseNew(&response, request));

            TRITONBACKEND_Output *output;
            std::vector<int64_t> output_shape;
            output_shape.push_back(1);
            output_shape.push_back(1);
            RESPOND_ALL_AND_RETURN_FALSE_IF_ERROR(requests, request_count, TRITONBACKEND_ResponseOutput(response, output, "RESPONSE", TRITONSERVER_TYPE_BYTES, output_shape.data(), output_shape.size()));

            void *output_buffer;
            TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
            int64_t output_memory_type_id = 0;
            RESPOND_ALL_AND_RETURN_FALSE_IF_ERROR(requests, request_count, TRITONBACKEND_OutputBuffer(output, &output_buffer, responseStr.size() + 4, &output_memory_type, &output_memory_type_id));

            uint8_t *output_buffer_unit8 = reinterpret_cast<uint8_t *>(output_buffer);
            int responseStrSize = int(responseStr.size());

            std::memcpy(output_buffer_unit8 + 4, responseStr.c_str(), responseStr.size());
            std::memcpy(output_buffer_unit8, &(responseStrSize), 4);

            responses.push_back(response);
        }

        for (auto &response : responses)
        {
            if (response != nullptr)
            {
                LOG_IF_ERROR(TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr), "falsed");
            }
        }
        return true;
    }

    TRITONSERVER_Error *
    ModelInstanceState::returnErrorIfFalse(const bool &myAssert, const TRITONSERVER_Error_Code &errCode, const std::string &msg)
    {
        RETURN_ERROR_IF_FALSE(myAssert, errCode, msg);
        return nullptr;
    }

    bool ModelInstanceState::getResponseStrs(const std::vector<std::string> &promptStrs, const std::vector<int> &respLimitInts, std::vector<std::string> &responseStrs)
    {
        if (promptStrs.size() != respLimitInts.size() || promptStrs.size() == 0 || respLimitInts.size() == 0)
        {
            return fase;
        }
        int output_token_limit = *std::max_element(respLimitInts.begin(), respLimitInts.end());
        std::vector<std::string> promptStrsRound;
        for (const std::string &promptStr : promptStrs)
        {
            promptStrsRound.push_back(getRound(0, promptStr, ""));
        }

        flmCall->predict(promptStrsRound, output_token_limit, responseStrs);

        for (int ii = 0; ii < responseStrs.size(); ii++)
        {
            responseStrs[ii].erase(0, responseStrs[ii].find_first_not_of(" "));
        }

        if (promptStrs.size() != responseStrs.size())
        {
            return false;
        }

        if (promptStrsRound.size() != responseStrs.size())
        {
            return false;
        }

        return true;
    }

    std::string ModelInstanceState::getRound(const int &round, const std::string &query, const std::string &answer)
    {
        if (answer.empty())
        {
            return "[Round " + std::to_string(round) + "]\n问：" + query + "\n答：";
        }
        else
        {
            return "[Round " + std::to_string(round) + "]\n问：" + query + "\n答：" + answer + "\n";
        }
    }

    bool ModelInstanceState::parseInput(TRITONBACKEND_Request **requests, const uint32_t request_count, std::vector<std::string> &promptStrs, std::vector<int> &respLimitInts)
    {
        for (uint32_t r = 0; r < request_count; ++r)
        {
            TRITONBACKEND_Request *request = requests[r];

            TRITONBACKEND_Input *prompt;
            TRITONBACKEND_Input *respLimit;

            RESPOND_ALL_AND_RETURN_FALSE_IF_ERROR(requests, request_count, TRITONBACKEND_RequestInput(request, "PROMPT", &prompt));
            RESPOND_ALL_AND_RETURN_FALSE_IF_ERROR(requests, request_count, TRITONBACKEND_RequestInput(request, "RESPONSE_LIMIT", &respLimit));

            const int16_t *promptShape;
            uint32_t promptShapeDim;
            uint64_t promptByteCnt;
            RESPOND_ALL_AND_RETURN_FALSE_IF_ERROR(requests, request_count, TRITONBACKEND_InputProperties(prompt, nullptr, nullptr, &promptShape, &promptShapeDim, &promptByteCnt, nullptr));

            const int64_t *respLimitShape;
            uint32_t respLimitShapeDim;
            uint64_t respLimitByteCnt;
            RESPOND_ALL_AND_RETURN_FALSE_IF_ERROR(requests, request_count, TRITONBACKEND_InputProperties(respLimit, nullptr, nullptr, &respLimitShape, &respLimitShapeDim, &respLimitByteCnt, nullptr));

            std::unique_ptr<char[]> promptBuff(new char[promptByteCnt]);
            RESPOND_ALL_AND_RETURN_FALSE_IF_ERROR(requests, request_count, ReadInputTensor(request, "PROMPT", promptBuff.get(), reinterpret_cast<size_t *>(&promptByteCnt)));

            std::unique_ptr<char[]> respLimitBuff(new char[respLimitByteCnt]);
            RESPOND_ALL_AND_RETURN_FALSE_IF_ERROR(requests, request_count, ReadInputTensor(request, "RESPONSE_LIMIT", respLimitBuff.get(), reinterpret_cast<size_t *>(&respLimitByteCnt)));

            std::string promptStr("");
            RESPOND_ALL_AND_RETURN_FALSE_IF_ERROR(requests, request_count, byteListToStr(promptBuff.get(), promptByteCnt, promptStr));

            int respLimitInt = 0;
            RESPOND_ALL_AND_RETURN_FALSE_IF_ERROR(requests, request_count, byteListToInt(respLimitBuff.get(), respLimitByteCnt, respLimitInt));

            promptStrs.push_back(promptStr);
            respLimitInts.push_back(respLimitInt);
        }
        return true;
    }

    TRITONSERVER_Error *
    ModelInstanceState::byteListToStr(const char *byteList, const uint64_t &byteCnt, std::string &targetStr)
    {
        RETURN_ERROR_IF_FALSE(
            byteCnt > 4, TRITONSERVER_ERROR_INVALID_ARG,
            std::string("byteCnt must bigger than 4"));
        targetStr.assign(byteList + 4, byteList + byteCnt);
        return nullptr;
    }
    TRITONSERVER_Error *
    ModelInstanceState::byteListToInt(const char *byteList, const uint64_t &byteCnt, int &targetInt)
    {
        RETURN_ERROR_IF_FALSE(
            byteCnt == 4, TRITONSERVER_ERROR_INVALID_ARG,
            std::string("byteCnt must equals to 4"));
        std::memcpy(&targetInt, byteList, 4);
        return nullptr;
    }

    TRITONSERVER_Error *
    ModelInstanceState::Create(ModelState *model_state, TRITONBACKEND_ModelInstance *triton_model_instance, ModelInstanceState **state)
    {
        try
        {
            *state = new ModelInstanceState(model_state, triton_model_instance);
        }
        catch (const BackendModelInstanceException &ex)
        {
            RETURN_ERROR_IF_TRUE(
                ex.err == nullptr, TRITONSERVER_ERROR_INTERNAL,
                std::string("unexpected nullptr in BackendModelException"));
            RETURN_IF_ERROR(ex.err_);
        }

        return nullptr;
    }

    extern "C"
    {
        TRITONSERVER_Error *
        TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance *instance)
        {
            TRITONBACKEND_Model *model;
            RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

            void *vmodelstate;
            RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, vmodelstate));
            ModelState *model_state = reinterpret_cast<ModelState *>(vmodelstate);

            ModelInstanceState *instance_state;
            RETURN_IF_ERROR(ModelInstanceState::Create(model_state, instance, &instance_state));
            RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(instance, reinterpret_cast<void *>(instance_state)));

            return nullptr;
        }

        TRITONSERVER_Error *
        TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance *instance)
        {
            void *vstate;
            RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
            ModelInstanceState *instance_state = reinterpret_cast<ModelInstanceState *>(vstate);
            delete instance_state;
            return nullptr;
        }
    } // extern "C"

    extern "C"
    {
        TRITONSERVER_Error *
        TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance *instance, TRITONBACKEND_Request **requests, const uint32_t request_count)
        {
            ModelInstanceState *instance_state;
            RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void **>(&instance_state)));
            instance_state->ProcessRequest(requests, request_count);
            for (uint32_t r = 0; r < request_count; ++r)
            {
                TRITONBACKEND_Request *request = requests[r];
                LOG_IF_ERROR(TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL), "failed");
            }
            return nullptr;
        }
    } // extern "C"

} // namespace triton::backend::glmbackend
