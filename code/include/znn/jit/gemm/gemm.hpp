//
// Copyright (C) 2018 Aleksandar Zlateski <zlateski@mit.edu>
// Copyright (C) 2018 Zhen Jia <zhenj@princeton.edu>
// ---------------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#pragma once

#if defined(ZNN_AVX2)
#include "avx2.hpp"
#elif defined(ZNN_AVX512)
#include "avx512.hpp"
#else
#error "needs either avx2 or avx512 support"
#endif

#include "type.hpp"

#include <fstream>
#include <map>
#include <mutex>

#include <dlfcn.h>

namespace znn::jit
{

struct znn_gemm_cache_t
{
    std::mutex                           lock;
    std::map<std::string, weak_gemm_t>   cache;
    std::map<std::string, shared_gemm_t> speedup;
};

inline znn_gemm_cache_t znn_gemm_cache;

inline shared_gemm_t get_znn_gemm(int_t ArCr, int_t AcBr, int_t BcCc, int_t LDA,
                                  int_t LDB, int_t LDC, int_t alpha, int_t beta,
                                  bool Apf, bool Bpf, bool Cpf,
                                  bool Apf0 = true, bool Bpf0 = true,
                                  bool Cpf0 = false, bool Cscatter = false,
                                  int_t SC_LD0 = 0, int_t SC_LD1 = 0)

{

    ZNN_JIT_ASSERT(LDA % 16 == 0);
    ZNN_JIT_ASSERT(LDB % 16 == 0);
    ZNN_JIT_ASSERT(LDC % 16 == 0);
    ZNN_JIT_ASSERT(AcBr % 16 == 0);
    ZNN_JIT_ASSERT(BcCc % 16 == 0);
    ZNN_JIT_ASSERT(alpha == -1 || alpha == 1);
    ZNN_JIT_ASSERT(beta == 0 || beta == 1);

    if (!Cscatter)
    {
        SC_LD0 = 0;
        SC_LD1 = 0;
    }

    std::string name;

#if defined(ZNN_AVX2)
    name = "znn_gemm_avx2_";
#elif defined(ZNN_AVX512)
    name = "znn_gemm_avx512_";
#endif

    name += std::to_string(ArCr) + "_" + std::to_string(AcBr) + "_" +
            std::to_string(BcCc) + "_" + std::to_string(LDA) + "_" +
            std::to_string(LDB) + "_" + std::to_string(LDC) + "_" +
            std::to_string(alpha + 1) + "_" + std::to_string(beta) + "_" +
            std::to_string(Apf) + "_" + std::to_string(Bpf) + "_" +
            std::to_string(Cpf) + "_" + std::to_string(Apf0) + "_" +
            std::to_string(Bpf0) + "_" + std::to_string(Cpf0) + "_" +
            std::to_string(Cscatter) + "_" + std::to_string(SC_LD0) + "_" +
            std::to_string(SC_LD1);

    {
        std::unique_lock<std::mutex> guard(znn_gemm_cache.lock);

        if (znn_gemm_cache.speedup.size() > 765)
        {
            znn_gemm_cache.speedup.clear();
        }

        if (auto ret = znn_gemm_cache.cache[name].lock(); ret)
        {
            return ret;
        }
    }

    auto fn_code =
        znn_gemm(ArCr, AcBr, BcCc, LDA * 4, LDB * 4, LDC * 4, alpha, beta, Apf,
                 Bpf, Cpf, Apf0, Bpf0, Cpf0, Cscatter, SC_LD0 * 4, SC_LD1 * 4);

    {
        std::unique_lock<std::mutex> guard(znn_gemm_cache.lock);

        if (auto ret = znn_gemm_cache.cache[name].lock(); ret)
        {
            return ret;
        }

        // std::system("mkdir -p gen");

        std::string fname = std::string("/tmp/") + name + ".c";

        std::ofstream ofs(fname.c_str());

        ofs << "void " + name +
                   "(float const * const A, float const * const B, float * "
                   "const  "
                   "C, float const * const A_prefetch, float const * const "
                   "B_prefetch, float const * const C_prefetch, float * const "
                   "C_scatter) { \n    __asm__ __volatile__ (\n";

        ofs << fn_code;

        ofs << "        :\n"
               "        : \"m\"(A), \"m\"(B), \"m\"(C), \"m\"(A_prefetch), "
               "\"m\"(B_prefetch), \"m\"(C_prefetch), \"m\"(C_scatter)\n"
               "        : \"rax\", \"rbx\", \"rcx\", \"rdx\", "
               "\"rdi\", \"rsi\", \"r8\", \"r9\", \"r10\",\n"
               "          \"r11\", \"r12\", \"r13\", \"r14\", \"r15\", ";

#if defined(ZNN_AVX2)

        ofs << "\"ymm0\", \"ymm1\", \"ymm2\", \"ymm3\",\n"
               "          \"ymm4\", \"ymm5\", \"ymm6\", \"ymm7\", \"ymm8\", "
               "\"ymm9\", \"ymm10\", \"ymm11\",\n"
               "          \"ymm12\", \"ymm13\", \"ymm14\", \"ymm15\");\n}\n";

#elif defined(ZNN_AVX512)

        ofs << "\"zmm0\", \"zmm1\", \"zmm2\", \"zmm3\",\n"
               "          \"zmm4\", \"zmm5\", \"zmm6\", \"zmm7\", \"zmm8\", "
               "\"zmm9\", \"zmm10\", \"zmm11\",\n"
               "          \"zmm12\", \"zmm13\", \"zmm14\", \"zmm15\", "
               "\"zmm16\", \"zmm17\", \"zmm18\",\n"
               "          \"zmm19\", \"zmm20\", \"zmm21\", \"zmm22\", "
               "\"zmm23\", \"zmm24\", \"zmm25\",\n"
               "          \"zmm26\", \"zmm27\", \"zmm28\", \"zmm29\", "
               "\"zmm30\", \"zmm31\");\n}\n";

#endif

        std::string compile_command = std::string("gcc -shared -Wl,-soname,") +
                                      name + ".so -O3 -DNDEBUG -fPIC " + fname +
                                      " ";

#if defined(ZNN_AVX2)
        compile_command += "-mavx2 -mfma";
#elif defined(ZNN_AVX512)
        compile_command += "-mavx512f -mavx512pf";
#endif

        compile_command += " -o /tmp/" + name + ".so";

        ofs.flush();

        system(compile_command.c_str());

        std::string so_name = std::string("/tmp/") + name + ".so";

        void* myso = dlopen(so_name.c_str(), RTLD_NOW);

        if (!myso)
        {
            std::printf("%s\n", dlerror());
            exit(EXIT_FAILURE);
        }

        ZNN_JIT_ASSERT(myso);

        void* myfn = dlsym(myso, name.c_str());

        ZNN_JIT_ASSERT(myfn);

        std::string rm_c_file  = std::string("rm ") + fname;
        std::string rm_so_file = std::string("rm ") + so_name;

        system(rm_c_file.c_str());
        system(rm_so_file.c_str());

        auto ret = shared_gemm_t(reinterpret_cast<znn_gemm_t>(myfn), myso);

        znn_gemm_cache.speedup[name] = ret;
        znn_gemm_cache.cache[name] = ret;

        return ret;
    }
}

} // namespace znn::jit
