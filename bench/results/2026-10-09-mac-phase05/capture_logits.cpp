// Standalone diagnostic using the pinned public llama API; llama.cpp is not modified.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("model.gguf cpu|metal prompts.json output_dir");
        const bool metal = std::string(argv[2]) == "metal";
        if (!metal && std::string(argv[2]) != "cpu") throw std::runtime_error("Invalid backend");
        std::ifstream promptFile(argv[3]);
        const json configuration = json::parse(promptFile);
        const std::filesystem::path directory(argv[4]);
        std::filesystem::create_directories(directory);
        ggml_backend_load_all();
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = metal ? 99 : 0;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        mp.lazy_mode = LLAMA_LAZY_MODE_ON;
        mp.load_mtp = false;
        const llama_model_tensor_buft_override overrides[] = {
            {"per_layer_token_embd.weight", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}
        };
        mp.tensor_buft_overrides = overrides;
        const auto loadStart = Clock::now();
        llama_model* model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("Model loading failed");
        const double loadSeconds = seconds(loadStart);
        const llama_vocab* vocab = llama_model_get_vocab(model);
        const int nVocab = llama_vocab_n_tokens(vocab);
        json report = {{"backend",argv[2]}, {"load_seconds",loadSeconds}, {"vocabulary_size",nVocab},
                       {"logits_format","little-endian FP32; rows are pre-greedy full-vocabulary logits"},
                       {"runs",json::array()}};
        for (const auto& prompt : configuration.at("prompts")) {
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = 512; cp.n_batch = 64; cp.n_ubatch = 64;
            cp.n_threads = 8; cp.n_threads_batch = 8;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            cp.no_perf = false;
            llama_context* ctx = llama_init_from_model(model, cp);
            if (!ctx) throw std::runtime_error("Context creation failed");
            const std::string text = prompt.at("text");
            const std::string id = prompt.at("id");
            int count = llama_tokenize(vocab,text.data(),text.size(),nullptr,0,true,true);
            if (count >= 0) throw std::runtime_error("Token sizing failed");
            std::vector<llama_token> input(-count);
            count = llama_tokenize(vocab,text.data(),text.size(),input.data(),input.size(),true,true);
            if (count < 1 || count > 64) throw std::runtime_error("Prompt does not fit fixed batch");
            input.resize(count);
            const auto promptStart = Clock::now();
            if (llama_decode(ctx,llama_batch_get_one(input.data(),input.size())) != 0)
                throw std::runtime_error("Prompt decode failed");
            const double promptSeconds = seconds(promptStart);
            std::ofstream logits(directory/(id+".logits.bin"),std::ios::binary);
            std::vector<llama_token> generated;
            std::string generatedText;
            const auto decodeStart = Clock::now();
            bool eog = false;
            for (int step=0; step<configuration.at("generation").at("max_generated_tokens").get<int>(); ++step) {
                const float* row = llama_get_logits_ith(ctx,-1);
                if (!row) throw std::runtime_error("Missing logits");
                logits.write(reinterpret_cast<const char*>(row),nVocab*sizeof(float));
                if (!logits) throw std::runtime_error("Logit write failed");
                llama_token token = 0;
                for (int i=0;i<nVocab;++i) {
                    if (!std::isfinite(row[i])) throw std::runtime_error("Nonfinite logit");
                    if (row[i]>row[token]) token=i;
                }
                generated.push_back(token);
                if (llama_vocab_is_eog(vocab,token)) { eog=true; break; }
                std::vector<char> piece(256);
                int length = llama_token_to_piece(vocab,token,piece.data(),piece.size(),0,true);
                if (length<0) {
                    piece.resize(-length);
                    length=llama_token_to_piece(vocab,token,piece.data(),piece.size(),0,true);
                }
                if (length<0) throw std::runtime_error("Token text conversion failed");
                generatedText.append(piece.data(),length);
                if (step+1<configuration.at("generation").at("max_generated_tokens").get<int>()) {
                    if (llama_decode(ctx,llama_batch_get_one(&token,1))!=0)
                        throw std::runtime_error("Token decode failed");
                }
            }
            const double captureSeconds = seconds(decodeStart);
            json run = {{"id",id},{"input_token_ids",input},{"generated_token_ids",generated},
                        {"generated_text",generatedText},{"stopped_at_eog",eog},
                        {"logits_rows",generated.size()},{"logits_columns",nVocab},{"all_logits_finite",true},
                        {"prompt_seconds",promptSeconds},{"capture_seconds_including_logit_IO",captureSeconds}};
            report["runs"].push_back(run);
            std::ofstream(directory/"capture.json") << report.dump(2) << '\n';
            llama_perf_context_print(ctx);
            llama_free(ctx);
        }
        llama_model_free(model);
        llama_backend_free();
        std::cout << "Capture complete: " << directory << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "capture_logits: " << e.what() << '\n';
        return 1;
    }
}
