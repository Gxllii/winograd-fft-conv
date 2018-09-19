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

#include "../frame.hpp"

namespace znn::jit
{

inline static constexpr int_t zmm_bytes = 16 * 4;

inline void znn_gemm_row_col_blocked(
    std::shared_ptr<frame> const& pf, int_t ArCr, int_t AcBr, int_t LDA,
    int_t LDB, int_t LDC, int_t alpha, int_t beta, reg const& A_reg,
    reg const& B_reg, reg const& C_reg, reg const& Apf_reg, reg const& Bpf_reg,
    reg const& Cpf_reg, bool Apf0, bool Bpf0, bool Cpf0,
    reg const& Cscatter_reg, int_t SC_LD0 = 0)
{
    static_cast<void>(Cpf0);

    auto f = pf->spawn();

    f->uses_register(A_reg);
    f->uses_register(B_reg);
    f->uses_register(C_reg);

    if (Apf_reg)
    {
        f->uses_register(Apf_reg);
    }
    if (Bpf_reg)
    {
        f->uses_register(Bpf_reg);
    }
    if (Cpf_reg)
    {
        f->uses_register(Cpf_reg);
    }
    if (Cscatter_reg)
    {
        f->uses_register(Cscatter_reg);
    }

    if (beta)
    {
        for (int_t cr = 0; cr < ArCr; ++cr)
        {
            f->vmov(ptr(cr * LDC, C_reg), zmm(cr + 2));
        }
    }
    else
    {
        f->set0(zmm(2));
        for (int_t cr = 1; cr < ArCr; ++cr)
        {
            f->vmov(zmm(2), zmm(cr + 2));
        }
    }

    reg A_index_1, A_index_3, A_index_5, A_index_7;

    if (ArCr > 1)
    {
        f->mov(val(LDA), A_index_1 = f->get_register());
    }
    if (ArCr > 3)
    {
        f->mov(val(LDA * 3), A_index_3 = f->get_register());
    }
    if (ArCr > 5)
    {
        f->mov(val(LDA * 5), A_index_5 = f->get_register());
    }
    if (ArCr > 7)
    {
        f->mov(val(LDA * 7), A_index_7 = f->get_register());
    }

    reg A_base_9, A_base_18, A_base_27;

    if (ArCr > 9)
    {
        f->mov(A_reg, A_base_9 = f->get_register());
        f->add(val(LDA * 9), A_base_9);
    }
    if (ArCr > 18)
    {
        f->mov(A_reg, A_base_18 = f->get_register());
        f->add(val(LDA * 18), A_base_18);
    }
    if (ArCr > 27)
    {
        f->mov(A_reg, A_base_27 = f->get_register());
        f->add(val(LDA * 27), A_base_27);
    }

    auto unrolled = [&](int_t i) {
        if (Bpf_reg)
        {
            f->prefetcht1(ptr(i * LDB, Bpf_reg));
        }
        if (ArCr > 0)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_reg), zmm(i % 2), zmm(2),
                           alpha == -1);
            if (Apf0 && i == 0)
            {
                f->prefetcht0(ptr(zmm_bytes, A_reg));
            }
        }
        if (ArCr > 1)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_reg, A_index_1, 1), zmm(i % 2),
                           zmm(3), alpha == -1);
            if (Apf0 && i == 1)
            {
                f->prefetcht0(ptr(zmm_bytes, A_reg, A_index_1, 1));
            }
        }
        if (ArCr > 2)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_reg, A_index_1, 2), zmm(i % 2),
                           zmm(4), alpha == -1);
            if (Apf0 && i == 2)
            {
                f->prefetcht0(ptr(zmm_bytes, A_reg, A_index_1, 2));
            }
        }
        if (ArCr > 3)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_reg, A_index_3, 1), zmm(i % 2),
                           zmm(5), alpha == -1);
            if (Apf0 && i == 3)
            {
                f->prefetcht0(ptr(zmm_bytes, A_reg, A_index_3, 1));
            }
        }

        if (Bpf0)
        {
            f->prefetcht0(ptr((i + 16) * LDB, B_reg));
        }

        if (ArCr > 4)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_reg, A_index_1, 4), zmm(i % 2),
                           zmm(6), alpha == -1);
            if (Apf0 && i == 4)
            {
                f->prefetcht0(ptr(zmm_bytes, A_reg, A_index_1, 4));
            }
        }
        if (ArCr > 5)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_reg, A_index_5, 1), zmm(i % 2),
                           zmm(7), alpha == -1);
            if (Apf0 && i == 5)
            {
                f->prefetcht0(ptr(zmm_bytes, A_reg, A_index_5, 1));
            }
        }
        if (ArCr > 6)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_reg, A_index_3, 2), zmm(i % 2),
                           zmm(8), alpha == -1);
            if (Apf0 && i == 6)
            {
                f->prefetcht0(ptr(zmm_bytes, A_reg, A_index_3, 2));
            }
        }
        if (ArCr > 7)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_reg, A_index_7, 1), zmm(i % 2),
                           zmm(9), alpha == -1);
            if (Apf0 && i == 7)
            {
                f->prefetcht0(ptr(zmm_bytes, A_reg, A_index_7, 1));
            }
        }

        if (Bpf0)
        {
            f->prefetcht0(ptr(i * LDB + zmm_bytes, B_reg));
        }

        if (ArCr > 8)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_reg, A_index_1, 8), zmm(i % 2),
                           zmm(10), alpha == -1);
            if (Apf0 && i == 8)
            {
                f->prefetcht0(ptr(zmm_bytes, A_reg, A_index_1, 8));
            }
        }
        if (ArCr > 9)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_9), zmm(i % 2), zmm(11),
                           alpha == -1);
            if (Apf0 && i == 9)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_9));
            }
        }
        if (ArCr > 10)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_9, A_index_1, 1), zmm(i % 2),
                           zmm(12), alpha == -1);
            if (Apf0 && i == 10)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_9, A_index_1, 1));
            }
        }
        if (ArCr > 11)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_9, A_index_1, 2), zmm(i % 2),
                           zmm(13), alpha == -1);
            if (Apf0 && i == 11)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_9, A_index_1, 2));
            }
        }
        if (ArCr > 12)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_9, A_index_3, 1), zmm(i % 2),
                           zmm(14), alpha == -1);
            if (Apf0 && i == 12)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_9, A_index_3, 1));
            }
        }
        if (ArCr > 13)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_9, A_index_1, 4), zmm(i % 2),
                           zmm(15), alpha == -1);
            if (Apf0 && i == 13)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_9, A_index_1, 4));
            }
        }
        if (ArCr > 14)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_9, A_index_5, 1), zmm(i % 2),
                           zmm(16), alpha == -1);
            if (Apf0 && i == 14)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_9, A_index_5, 1));
            }
        }
        if (ArCr > 15)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_9, A_index_3, 2), zmm(i % 2),
                           zmm(17), alpha == -1);
            if (Apf0 && i == 15)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_9, A_index_3, 2));
            }
        }
        if (ArCr > 16)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_9, A_index_7, 1), zmm(i % 2),
                           zmm(18), alpha == -1);
            if (Apf0 && i == 0)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_9, A_index_7, 1));
            }
        }
        if (ArCr > 17)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_9, A_index_1, 8), zmm(i % 2),
                           zmm(19), alpha == -1);
            if (Apf0 && i == 1)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_9, A_index_1, 8));
            }
        }
        if (ArCr > 18)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_18), zmm(i % 2), zmm(20),
                           alpha == -1);
            if (Apf0 && i == 2)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_18));
            }
        }
        if (ArCr > 19)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_18, A_index_1, 1),
                           zmm(i % 2), zmm(21), alpha == -1);
            if (Apf0 && i == 3)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_18, A_index_1, 1));
            }
        }
        if (ArCr > 20)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_18, A_index_1, 2),
                           zmm(i % 2), zmm(22), alpha == -1);
            if (Apf0 && i == 4)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_18, A_index_1, 2));
            }
        }
        if (ArCr > 21)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_18, A_index_3, 1),
                           zmm(i % 2), zmm(23), alpha == -1);
            if (Apf0 && i == 5)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_18, A_index_3, 1));
            }
        }
        if (ArCr > 22)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_18, A_index_1, 4),
                           zmm(i % 2), zmm(24), alpha == -1);
            if (Apf0 && i == 6)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_18, A_index_1, 4));
            }
        }
        if (ArCr > 23)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_18, A_index_5, 1),
                           zmm(i % 2), zmm(25), alpha == -1);
            if (Apf0 && i == 7)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_18, A_index_5, 1));
            }
        }
        if (ArCr > 24)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_18, A_index_3, 2),
                           zmm(i % 2), zmm(26), alpha == -1);
            if (Apf0 && i == 8)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_18, A_index_3, 2));
            }
        }
        if (ArCr > 25)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_18, A_index_7, 1),
                           zmm(i % 2), zmm(27), alpha == -1);
            if (Apf0 && i == 9)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_18, A_index_7, 1));
            }
        }
        if (ArCr > 26)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_18, A_index_1, 8),
                           zmm(i % 2), zmm(28), alpha == -1);
            if (Apf0 && i == 10)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_18, A_index_1, 8));
            }
        }
        if (ArCr > 27)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_27), zmm(i % 2), zmm(29),
                           alpha == -1);
            if (Apf0 && i == 11)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_27));
            }
        }
        if (ArCr > 28)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_27, A_index_1, 1),
                           zmm(i % 2), zmm(30), alpha == -1);
            if (Apf0 && i == 12)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_27, A_index_1, 1));
            }
        }
        if (ArCr > 29)
        {
            f->vfmadd231ps(ptr_1to16(i * 4, A_base_27, A_index_1, 2),
                           zmm(i % 2), zmm(31), alpha == -1);
            if (Apf0 && i == 13)
            {
                f->prefetcht0(ptr(zmm_bytes, A_base_27, A_index_1, 2));
            }
        }
    };

    auto body = [&](bool inc_regs) {

        // First unrolled
        f->vmov(ptr(0, B_reg), zmm(0));
        f->vmov(ptr(LDB, B_reg), zmm(1));
        unrolled(0);

        // Mid unrolled
        for (int_t u = 1; u < 16 - 1; ++u)
        {
            f->vmov(ptr((u + 1) * LDB, B_reg), zmm((u + 1) % 2));
            unrolled(u);
        }

        // Last unrolled
        unrolled(15);

        if (inc_regs)
        {

            f->add(val(zmm_bytes), A_reg);
            f->add(val(16 * LDB), B_reg);

            if (ArCr > 9)
            {
                f->add(val(zmm_bytes), A_base_9);
            }
            if (ArCr > 18)
            {
                f->add(val(zmm_bytes), A_base_18);
            }
            if (ArCr > 27)
            {
                f->add(val(zmm_bytes), A_base_27);
            }
            if (Bpf_reg)
            {
                f->add(val(16 * LDB), Bpf_reg);
            }
        }
    };

    // meat

    auto full = AcBr / 16;

    ZNN_JIT_ASSERT(full > 0);

    if (full > 1)
    {

        auto loop_reg = f->get_register();
        f->mov(val(0), loop_reg);
        auto loop_lab = f->add_label();
        f->add(val(1), loop_reg);

        body(true);

        f->cmp(val(full), loop_reg);
        f->jl(loop_lab);

        f->sub(val(full * zmm_bytes), A_reg);
        f->sub(val(full * 16 * LDB), B_reg);
        if (Bpf_reg)
        {
            f->sub(val(full * 16 * LDB), Bpf_reg);
        }
    }
    else if (full == 1)
    {
        body(false);
    }

    for (int_t cr = 0; cr < ArCr; ++cr)
    {
        if (Cscatter_reg)
        {
            f->vmovnt(zmm(cr + 2), ptr(cr * SC_LD0, Cscatter_reg));
        }
        else
        {
            f->vmov(zmm(cr + 2), ptr(cr * LDC, C_reg));
        }

        if (Apf_reg)
        {
            f->prefetcht1(ptr(cr * LDA, Apf_reg));
        }
        if (Cpf_reg)
        {
            f->prefetcht1(ptr(cr * LDC, Cpf_reg));
        }
    }
}

inline void znn_gemm_row_blocked(
    std::shared_ptr<frame> const& f, int_t ArCr, int_t AcBr, int_t BcCc,
    int_t LDA, int_t LDB, int_t LDC, int_t alpha, int_t beta, reg const& A_reg,
    reg const& B_reg, reg const& C_reg, reg const& Apf_reg, reg const& Bpf_reg,
    reg const& Cpf_reg, bool Apf0, bool Bpf0, bool Cpf0,
    reg const& Cscatter_reg, int_t SC_LD0 = 0, int_t SC_LD1 = 0)
{
    static constexpr int_t col_block = 1;

    int_t full = BcCc / 16 / col_block;

    auto body = [&](bool inc) {

        znn_gemm_row_col_blocked(f, ArCr, AcBr, LDA, LDB, LDC, alpha, beta,
                                 A_reg, B_reg, C_reg, Apf_reg, Bpf_reg, Cpf_reg,
                                 Apf0, Bpf0, Cpf0, Cscatter_reg, SC_LD0);

        if (inc)
        {
            f->add(val(zmm_bytes), B_reg);
            f->add(val(zmm_bytes), C_reg);

            if (Apf_reg)
            {
                f->add(val(zmm_bytes), Apf_reg);
            }

            if (Bpf_reg)
            {
                f->add(val(zmm_bytes), Bpf_reg);
            }

            if (Cpf_reg)
            {
                f->add(val(zmm_bytes), Cpf_reg);
            }

            if (Cscatter_reg)
            {
                f->add(val(SC_LD1), Cscatter_reg);
            }
        }
    };

    if (full > 1)
    {
        auto loop_reg = f->get_register();
        f->mov(val(0), loop_reg);
        auto loop_lab = f->add_label();
        f->add(val(1), loop_reg);

        body(true);

        f->cmp(val(full), loop_reg);
        f->jl(loop_lab);
        f->return_register(loop_reg);

        f->sub(val(full * zmm_bytes), B_reg);
        f->sub(val(full * zmm_bytes), C_reg);

        if (Apf_reg)
        {
            f->sub(val(full * zmm_bytes), Apf_reg);
        }

        if (Bpf_reg)
        {
            f->sub(val(full * zmm_bytes), Bpf_reg);
        }

        if (Cpf_reg)
        {
            f->sub(val(full * zmm_bytes), Cpf_reg);
        }

        if (Cscatter_reg)
        {
            f->smart_sub(SC_LD1 * full, Cscatter_reg);
        }
    }
    else if (full == 1)
    {
        body(false);
    }
}

inline std::string znn_gemm(int_t ArCr, int_t AcBr, int_t BcCc, int_t LDA,
                            int_t LDB, int_t LDC, int_t alpha, int_t beta,
                            bool Apf, bool Bpf, bool Cpf, bool Apf0 = true,
                            bool Bpf0 = true, bool Cpf0 = false,
                            bool Cscatter = false, int_t SC_LD0 = 0,
                            int_t SC_LD1 = 0)
{
    auto f = std::make_shared<frame>();

    reg A_reg = f->get_register();
    reg B_reg = f->get_register();
    reg C_reg = f->get_register();

    f->mov(c_arg(0), A_reg);
    f->mov(c_arg(1), B_reg);
    f->mov(c_arg(2), C_reg);

    reg Apf_reg, Bpf_reg, Cpf_reg, Cscatter_reg;

    if (Apf)
    {
        Apf_reg = f->get_register();
        f->mov(c_arg(3), Apf_reg);
    }

    if (Bpf)
    {
        Bpf_reg = f->get_register();
        f->mov(c_arg(4), Bpf_reg);
    }

    if (Cpf)
    {
        Cpf_reg = f->get_register();
        f->mov(c_arg(5), Cpf_reg);
    }

    if (Cscatter)
    {
        Cscatter_reg = f->get_register();
        f->mov(c_arg(6), Cscatter_reg);
    }

    auto body = [&](int_t rb, bool inc_regs) {
        znn_gemm_row_blocked(f, rb, AcBr, BcCc, LDA, LDB, LDC, alpha, beta,
                             A_reg, B_reg, C_reg, Apf_reg, Bpf_reg, Cpf_reg,
                             Apf0, Bpf0, Cpf0, Cscatter_reg, SC_LD0, SC_LD1);

        if (inc_regs)
        {
            f->add(val(rb * LDA), A_reg);
            f->add(val(rb * LDC), C_reg);

            if (Apf)
            {
                f->add(val(rb * LDA), Apf_reg);
            }

            if (Cpf)
            {
                f->add(val(rb * LDC), Cpf_reg);
            }

            if (Cscatter)
            {
                f->add(val(rb * SC_LD0), Cscatter_reg);
            }
        }
    };

    int_t row_block = std::min(static_cast<int_t>(30), ArCr);

    for (int_t r = row_block, best_rem = ArCr % 30; r >= 16; --r)
    {
        if (ArCr % r == 0)
        {
            row_block = r;
            break;
        }
        else if (ArCr % r > best_rem)
        {
            best_rem  = ArCr % r;
            row_block = r;
        }
    }

    auto full = ArCr / row_block;
    auto rest = ArCr % row_block;

    if (full > 1)
    {
        auto loop_reg = f->get_register();
        f->mov(val(0), loop_reg);
        auto loop_lab = f->add_label();
        f->add(val(1), loop_reg);

        body(row_block, true);

        f->cmp(val(full), loop_reg);
        f->jl(loop_lab);
        f->return_register(loop_reg);
    }
    else if (full == 1)
    {
        body(row_block, rest > 0);
    }

    if (rest)
    {
        body(rest, false);
    }

    return f->code();
}

} // namespace znn::jit
