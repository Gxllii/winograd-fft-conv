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

inline static constexpr int_t ymm_bytes       = 8 * 4;
inline static constexpr int_t cacheline_bytes = 16 * 4;

inline void znn_cgemm_row_col_blocked(
    std::shared_ptr<frame> const& f, int_t ArCr, int_t AcBr, int_t LDA,
    int_t LDB, int_t LDC, int_t alpha, int_t beta, reg const& A_reg,
    reg const& B_reg, reg const& C_reg, reg const& Apf_reg, reg const& Bpf_reg,
    reg const& Cpf_reg, bool Apf0, bool Bpf0, bool Cpf0,
    reg const& Cscatter_reg, int_t SC_LD0 = 0)
{

    // ignored for now
    static_cast<void>(Cpf0);
    static_cast<void>(Bpf_reg);

    int_t yidx = 0;

    ymm br1(yidx++);
    ymm br2(yidx++);
    ymm bi1(yidx++);
    ymm bi2(yidx++);

    ymm ar[2];
    ymm ai[2];

    ymm cr1[2];
    ymm cr2[2];
    ymm ci1[2];
    ymm ci2[2];

    for (int_t cr = 0; cr < ArCr; ++cr)
    {
        ar[cr] = ymm(yidx++);
        ai[cr] = ymm(yidx++);

        cr1[cr] = ymm(yidx++);
        cr2[cr] = ymm(yidx++);
        ci1[cr] = ymm(yidx++);
        ci2[cr] = ymm(yidx++);

        if (beta == 1)
        {
            f->vmov(ptr(cr * LDC, C_reg), cr1[cr]);
            f->vmov(ptr(cr * LDC + ymm_bytes, C_reg), cr2[cr]);
            f->vmov(ptr(cr * LDC + cacheline_bytes, C_reg), ci1[cr]);
            f->vmov(ptr(cr * LDC + cacheline_bytes + ymm_bytes, C_reg),
                    ci2[cr]);
        }
    }

    if (beta == 0)
    {
        f->set0(cr1[0]);
        f->vmov(cr1[0], cr2[0]);
        f->vmov(cr1[0], ci1[0]);
        f->vmov(cr1[0], ci2[0]);

        for (int_t cr = 1; cr < ArCr; ++cr)
        {
            f->vmov(cr1[0], cr1[cr]);
            f->vmov(cr1[0], cr2[cr]);
            f->vmov(cr1[0], ci1[cr]);
            f->vmov(cr1[0], ci2[cr]);
        }
    }

    auto body = [&](bool inc) {

        for (int_t i = 0; i < 16; ++i)
        {
            for (int_t r = 0; r < ArCr; ++r)
            {
                if (i == 0)
                {
                    f->vbroadcastss(ptr(i * 4 + LDA * r, A_reg), ar[r]);
                }

                f->vbroadcastss(ptr(i * 4 + LDA * r + cacheline_bytes, A_reg),
                                ai[r]);

                if ((i == r * 8) && Apf0)
                {
                    f->prefetcht0(ptr(LDA * r + cacheline_bytes * 2, A_reg));
                }
                if ((i == r * 8 + 4) && Apf0)
                {
                    f->prefetcht0(ptr(LDA * r + cacheline_bytes * 3, A_reg));
                }
            }

            f->vmov(ptr(i * LDB, B_reg), br1);
            f->vmov(ptr(i * LDB + ymm_bytes, B_reg), br2);
            f->vmov(ptr(i * LDB + cacheline_bytes, B_reg), bi1);
            f->vmov(ptr(i * LDB + cacheline_bytes + ymm_bytes, B_reg), bi2);

            if (Bpf0)
            {
                f->prefetcht0(ptr(cacheline_bytes * 2, B_reg));
                f->prefetcht0(ptr(cacheline_bytes * 3, A_reg));
            }

            for (int_t r = 0; r < ArCr; ++r)
            {
                f->vfmadd231ps(ar[r], br1, cr1[r], alpha == -1);
                f->vfmadd231ps(ar[r], br2, cr2[r], alpha == -1);
                f->vfmadd231ps(ar[r], bi1, ci1[r], alpha == -1);
                f->vfmadd231ps(ar[r], bi2, ci2[r], alpha == -1);

                if (i < 16 - 1)
                {
                    f->vbroadcastss(ptr((i + 1) * 4 + LDA * r, A_reg), ar[r]);
                }

                f->vfmadd231ps(ai[r], br1, ci1[r], alpha == -1);
                f->vfmadd231ps(ai[r], br2, ci2[r], alpha == -1);
                f->vfmadd231ps(ai[r], bi1, cr1[r], alpha == 1);
                f->vfmadd231ps(ai[r], bi2, cr2[r], alpha == 1);
            }
        }

        if (inc)
        {
            f->add(val(cacheline_bytes * 2), A_reg);
            f->add(val(16 * LDB), B_reg);

            if (Bpf_reg)
            {
                f->add(val(16 * LDB), Bpf_reg);
            }
        }
    };

    static constexpr int_t unroll_count = 16;

    int_t full = AcBr / unroll_count;

    if (full == 1)
    {
        body(false);
    }
    else
    {
        auto loop_reg = f->get_register();
        f->mov(val(0), loop_reg);
        auto loop_lab = f->add_label();
        f->add(val(1), loop_reg);

        body(true);

        f->cmp(val(full), loop_reg);
        f->jl(loop_lab);
        f->return_register(loop_reg);

        f->sub(val(full * cacheline_bytes * 2), A_reg);
        f->sub(val(full * 16 * LDB), B_reg);

        if (Bpf_reg)
        {
            f->sub(val(full * 16 * LDB), Bpf_reg);
        }
    }

    for (int_t cr = 0; cr < ArCr; ++cr)
    {
        if (Cscatter_reg)
        {
            f->vmovnt(cr1[cr], ptr(SC_LD0 * cr, Cscatter_reg));
            f->vmovnt(cr2[cr], ptr(SC_LD0 * cr + ymm_bytes, Cscatter_reg));
            f->vmovnt(ci1[cr],
                      ptr(SC_LD0 * cr + cacheline_bytes, Cscatter_reg));
            f->vmovnt(ci2[cr], ptr(SC_LD0 * cr + cacheline_bytes + ymm_bytes,
                                   Cscatter_reg));
        }
        else
        {
            f->vmov(cr1[cr], ptr(cr * LDC, C_reg));
            f->vmov(cr2[cr], ptr(cr * LDC + ymm_bytes, C_reg));
            f->vmov(ci1[cr], ptr(cr * LDC + cacheline_bytes, C_reg));
            f->vmov(ci2[cr],
                    ptr(cr * LDC + cacheline_bytes + ymm_bytes, C_reg));
        }

        if (Apf_reg)
        {
            f->prefetcht1(ptr(cr * LDA, Apf_reg));
            f->prefetcht1(ptr(cr * LDA + cacheline_bytes, Apf_reg));
        }
        if (Cpf_reg)
        {
            f->prefetcht1(ptr(cr * LDC, Cpf_reg));
            f->prefetcht1(ptr(cr * LDC + cacheline_bytes, Cpf_reg));
        }
    }
}

inline void znn_cgemm_row_blocked(
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

    auto body = [&](bool inc) {

        znn_cgemm_row_col_blocked(
            f, ArCr, AcBr, LDA, LDB, LDC, alpha, beta, A_reg, B_reg, C_reg,
            Apf_reg, Bpf_reg, Cpf_reg, Apf0, Bpf0, Cpf0, Cscatter_reg, SC_LD0);

        if (inc)
        {
            f->add(val(cacheline_bytes * 2), B_reg);
            f->add(val(cacheline_bytes * 2), C_reg);

            if (Apf_reg)
            {
                f->add(val(cacheline_bytes * 2), Apf_reg);
            }

            if (Bpf_reg)
            {
                f->add(val(cacheline_bytes * 2), Bpf_reg);
            }

            if (Cpf_reg)
            {
                f->add(val(cacheline_bytes * 2), Cpf_reg);
            }

            if (Cscatter_reg)
            {
                f->add(val(SC_LD1), Cscatter_reg);
            }
        }
    };

    auto full = (BcCc / 16);

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

        f->sub(val(full * cacheline_bytes * 2), B_reg);
        f->sub(val(full * cacheline_bytes * 2), C_reg);

        if (Apf_reg)
        {
            f->sub(val(full * cacheline_bytes * 2), Apf_reg);
        }

        if (Bpf_reg)
        {
            f->sub(val(full * cacheline_bytes * 2), Bpf_reg);
        }

        if (Cpf_reg)
        {
            f->sub(val(full * cacheline_bytes * 2), Cpf_reg);
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

inline std::string znn_cgemm(int_t ArCr, int_t AcBr, int_t BcCc, int_t LDA,
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

        znn_cgemm_row_blocked(f, rb, AcBr, BcCc, LDA, LDB, LDC, alpha, beta,
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

    static constexpr int_t row_block = 2;

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
