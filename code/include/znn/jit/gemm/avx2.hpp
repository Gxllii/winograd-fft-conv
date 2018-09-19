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

inline static constexpr int_t ymm_bytes = 8 * 4;

inline void znn_gemm_row_col_blocked(
    std::shared_ptr<frame> const& f, int_t ArCr, int_t AcBr, int_t BcCc,
    int_t LDA, int_t LDB, int_t LDC, int_t alpha, int_t beta, reg const& A_reg,
    reg const& B_reg, reg const& C_reg, reg const& Apf_reg, reg const& Bpf_reg,
    reg const& Cpf_reg, bool Apf0, bool Bpf0, bool Cpf0,
    reg const& Cscatter_reg, int_t SC_LD0 = 0, int_t SC_LD1 = 0)
{

    // ignored for now
    static_cast<void>(Bpf0);
    static_cast<void>(Cpf0);

    ymm a[16];
    ymm b[16];
    ymm c[16][16];

    int_t yidx = 0;

    for (int_t cr = 0; cr < ArCr; ++cr)
    {
        for (int_t cc = 0; cc < BcCc; ++cc)
        {
            c[cr][cc] = ymm(yidx++);
            if (beta == 1)
            {
                f->vmov(ptr(cr * LDC + cc * ymm_bytes, C_reg), c[cr][cc]);
            }
        }
    }

    if (beta == 0)
    {
        f->set0(c[0][0]);

        for (int_t cr = 0; cr < ArCr; ++cr)
        {
            for (int_t cc = 0; cc < BcCc; ++cc)
            {
                if (cr || cc)
                {
                    f->vmov(c[0][0], c[cr][cc]);
                }
            }
        }
    }

    for (int_t ar = 0; ar < ArCr; ++ar)
    {
        a[ar] = ymm(yidx++);
    }

    for (int_t bc = 0; bc < BcCc; ++bc)
    {
        b[bc] = ymm(yidx++);
    }

    auto body = [&](int_t count, bool inc) {

        for (int_t i = 0; i < count; ++i)
        {
            for (int_t ar = 0; ar < ArCr; ++ar)
            {
                f->vbroadcastss(ptr(i * 4 + LDA * ar, A_reg), a[ar]);
                if ((i % 16 == ar * 4) && Apf0)
                {
                    f->prefetcht0(
                        ptr((i / 16) * ymm_bytes * 2 + LDA * ar + ymm_bytes * 2,
                            A_reg));
                }
            }

            for (int_t bc = 0; bc < BcCc; ++bc)
            {
                f->vmov(ptr(bc * ymm_bytes + LDB * i, B_reg), b[bc]);
                if (Bpf_reg)
                {
                    f->prefetcht1(ptr(bc * ymm_bytes + LDB * i, Bpf_reg));
                }
            }

            for (int_t cr = 0; cr < ArCr; ++cr)
            {
                for (int_t cc = 0; cc < BcCc; ++cc)
                {
                    f->vfmadd231ps(a[cr], b[cc], c[cr][cc], alpha == -1);
                }
            }
        }

        if (inc)
        {
            f->add(val(count * 4), A_reg);
            f->add(val(count * LDB), B_reg);

            if (Bpf_reg)
            {
                f->add(val(count * LDB), Bpf_reg);
            }
        }
    };

    static constexpr int_t unroll_count = 16;

    if (AcBr <= unroll_count * 2)
    {
        body(AcBr, false);
    }
    else
    {
        auto full = AcBr / unroll_count;
        auto rest = AcBr % unroll_count;

        auto loop_reg = f->get_register();
        f->mov(val(0), loop_reg);
        auto loop_lab = f->add_label();
        f->add(val(1), loop_reg);

        body(unroll_count, true);

        f->cmp(val(full), loop_reg);
        f->jl(loop_lab);
        f->return_register(loop_reg);

        body(rest, false);

        f->sub(val(unroll_count * full * 4), A_reg);
        f->sub(val(unroll_count * full * LDB), B_reg);

        if (Bpf_reg)
        {
            f->sub(val(unroll_count * full * LDB), Bpf_reg);
        }
    }

    for (int_t cr = 0; cr < ArCr; ++cr)
    {
        for (int_t cc = 0; cc < BcCc; ++cc)
        {
            if (Cscatter_reg)
            {
                f->vmovnt(c[cr][cc], ptr(SC_LD0 * cr + SC_LD1 * (cc / 2) +
                                             ymm_bytes * (cc % 2),
                                         Cscatter_reg));
            }
            else
            {
                f->vmov(c[cr][cc], ptr(cr * LDC + cc * ymm_bytes, C_reg));
            }

            if (Apf_reg)
            {
                f->prefetcht1(ptr(cr * LDA + cc * ymm_bytes, Apf_reg));
            }
            if (Cpf_reg)
            {
                f->prefetcht1(ptr(cr * LDC + cc * ymm_bytes, Cpf_reg));
            }
        }
    }
}

inline void znn_gemm_row_blocked(
    std::shared_ptr<frame> const& pf, int_t ArCr, int_t AcBr, int_t BcCc,
    int_t LDA, int_t LDB, int_t LDC, int_t alpha, int_t beta, reg const& A_reg,
    reg const& B_reg, reg const& C_reg, reg const& Apf_reg, reg const& Bpf_reg,
    reg const& Cpf_reg, bool Apf0, bool Bpf0, bool Cpf0,
    reg const& Cscatter_reg, int_t SC_LD0 = 0, int_t SC_LD1 = 0)
{
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

    // We have total of 16 registers and need to keep ArCr * col_block
    // regs with results, ArCr loaded/scattered from A and col_block loaded from
    // B, thus the formula
    int_t col_block = (16 - ArCr) / (ArCr + 1);

    // If we perform scattering, we need even number for col_block, due to the
    // memory layout
    if (Cscatter_reg)
    {
        col_block -= (col_block % 2);
    }

    ZNN_JIT_ASSERT(col_block > 0);

    auto full = (BcCc / 8) / col_block;
    auto rest = (BcCc / 8) % col_block;

    auto body = [&](int_t cb, bool inc) {

        znn_gemm_row_col_blocked(f, ArCr, AcBr, cb, LDA, LDB, LDC, alpha, beta,
                                 A_reg, B_reg, C_reg, Apf_reg, Bpf_reg, Cpf_reg,
                                 Apf0, Bpf0, Cpf0, Cscatter_reg, SC_LD0,
                                 SC_LD1);

        if (inc)
        {
            f->add(val(cb * ymm_bytes), B_reg);
            f->add(val(cb * ymm_bytes), C_reg);

            if (Apf_reg)
            {
                f->add(val(cb * ymm_bytes), Apf_reg);
            }

            if (Bpf_reg)
            {
                f->add(val(cb * ymm_bytes), Bpf_reg);
            }

            if (Cpf_reg)
            {
                f->add(val(cb * ymm_bytes), Cpf_reg);
            }

            if (Cscatter_reg)
            {
                f->add(val(SC_LD1 * (cb / 2)), Cscatter_reg);
            }
        }
    };

    if (full > 1)
    {
        auto loop_reg = f->get_register();
        f->mov(val(0), loop_reg);
        auto loop_lab = f->add_label();
        f->add(val(1), loop_reg);

        body(col_block, true);

        f->cmp(val(full), loop_reg);
        f->jl(loop_lab);
        f->return_register(loop_reg);
    }
    else if (full == 1)
    {
        body(col_block, rest > 0);
    }

    if (rest)
    {
        body(rest, false);
    }

    if (full > 1 || (full == 1 && rest > 0))
    {
        f->sub(val(full * col_block * ymm_bytes), B_reg);
        f->sub(val(full * col_block * ymm_bytes), C_reg);

        if (Apf_reg)
        {
            f->sub(val(full * col_block * ymm_bytes), Apf_reg);
        }

        if (Bpf_reg)
        {
            f->sub(val(full * col_block * ymm_bytes), Bpf_reg);
        }

        if (Cpf_reg)
        {
            f->sub(val(full * col_block * ymm_bytes), Cpf_reg);
        }

        if (Cscatter_reg)
        {
            f->smart_sub(SC_LD1 * full * (col_block / 2), Cscatter_reg);
        }
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

    static constexpr int_t row_block = 4;

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
