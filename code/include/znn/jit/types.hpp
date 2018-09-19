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

#include "util.hpp"

#include <cstdint>
#include <string>

namespace znn::jit
{

using int_t = std::int64_t;

class label
{
private:
    std::string str_;

public:
    explicit label(std::string const& str = "")
        : str_(str)
    {
    }

    label(label const&) = default;
    label& operator=(label const&) = default;

    operator bool() const { return str_ != ""; }

    std::string to_arg() const
    {
        ZNN_JIT_ASSERT(str_ != "");
        return str_ + "b";
    }

    std::string to_lab() const { return str_ + ":"; }
};

class reg
{
private:
    std::string str_;

public:
    explicit reg(std::string const& r = "")
        : str_(r)
    {
        ZNN_JIT_ASSERT(r == "" || r == "rax" || r == "rbx" || r == "rcx" ||
                       r == "rdx" || r == "rsi" || r == "rdi" || r == "r8" ||
                       r == "r9" || r == "r10" || r == "r11" || r == "r12" ||
                       r == "r13" || r == "r14" || r == "r15");
    }

    reg(reg const&) = default;
    reg& operator=(reg const&) = default;

    operator bool() const { return str_ != ""; }

    bool operator!() const { return str_ == ""; }

    friend bool operator<(reg const& a, reg const& b)
    {
        return a.str_ < b.str_;
    }

    std::string to_arg() const
    {
        ZNN_JIT_ASSERT(str_ != "");
        return std::string("%%") + str_;
    }
};

class val
{
private:
    int_t val_;

public:
    explicit val(int_t v)
        : val_(v)
    {
        // TODO (zlateski): check whether constants are considered unsigned or
        // not?  The code used to be ZNN_JIT_ASSERT((v >> 31) == 0), and had
        // assumed singed 32bit integers, but the previous non-checked code
        // worked wihtout problems.  More than 32 bits cause compile error, as
        // the immidiate value was too large, but not sure if the code was
        // correct before, as the correctness checks were done on larger values
        // of F(m,r) for the Winograd transforms.  Easy way to check is to
        // re-run the accuracy code on avx2 machine, as it was the one that
        // asserted here.  Should also try to understand why avx2 asserts here,
        // and avx512 doesn't.
        ZNN_JIT_ASSERT((v >> 32) == 0);
    }

    val(val const&) = default;
    val& operator=(val const&) = default;

    std::string to_arg() const
    {
        return std::string("$") + std::to_string(val_);
    }
};

class c_arg
{
private:
    int_t no_;

public:
    explicit c_arg(int_t n)
        : no_(n)
    {
    }

    c_arg(c_arg const&) = default;
    c_arg& operator=(c_arg const&) = default;

    std::string to_arg() const
    {
        return std::string("\%") + std::to_string(no_);
    }
};

class ptr
{
private:
    std::string str_;

public:
    explicit ptr(reg const& r)
        : str_("(" + r.to_arg() + ")")
    {
        ZNN_JIT_ASSERT(r);
    }

    ptr(int_t off, reg const& r)
        : str_(std::to_string(off) + "(" + r.to_arg() + ")")
    {
        ZNN_JIT_ASSERT((off >> 31) == 0);
        ZNN_JIT_ASSERT(r);
    }

    ptr(int_t off, reg const& base_reg, reg const& index_reg, int_t mult = 1)
        : str_(std::to_string(off) + "(" + base_reg.to_arg() + "," +
               index_reg.to_arg() + "," + std::to_string(mult) + ")")
    {
        ZNN_JIT_ASSERT((off >> 31) == 0);
        ZNN_JIT_ASSERT(base_reg);
        ZNN_JIT_ASSERT(index_reg);
        ZNN_JIT_ASSERT(mult == 1 || mult == 2 || mult == 4 || mult == 8);
    }

    ptr(ptr const&) = default;
    ptr& operator=(ptr const&) = default;

    std::string to_arg() const { return str_; }
};

class ptr_1to16
{
private:
    std::string str_;

public:
    explicit ptr_1to16(reg const& r)
        : str_("(" + r.to_arg() + ")%{1to16%}")
    {
        ZNN_JIT_ASSERT(r);
    }

    ptr_1to16(int_t off, reg const& r)
        : str_(std::to_string(off) + "(" + r.to_arg() + ")%{1to16%}")
    {
        ZNN_JIT_ASSERT((off >> 31) == 0);
        ZNN_JIT_ASSERT(r);
    }

    ptr_1to16(int_t off, reg const& base_reg, reg const& index_reg,
              int_t mult = 1)
        : str_(std::to_string(off) + "(" + base_reg.to_arg() + "," +
               index_reg.to_arg() + "," + std::to_string(mult) + ")%{1to16%}")
    {
        ZNN_JIT_ASSERT((off >> 31) == 0);
        ZNN_JIT_ASSERT(base_reg);
        ZNN_JIT_ASSERT(index_reg);
        ZNN_JIT_ASSERT(mult == 1 || mult == 2 || mult == 4 || mult == 8);
    }

    ptr_1to16(ptr const& p)
        : str_(p.to_arg() + "%{1to16%}")
    {
    }

    ptr_1to16(ptr_1to16 const&) = default;
    ptr_1to16& operator=(ptr_1to16 const&) = default;

    std::string to_arg() const { return str_; }
};

class ymm
{
private:
    int_t which_;

public:
    ymm()
        : which_(-1)
    {
    }

    ymm(int_t n)
        : which_(n < 0 ? n + 16 : n)
    {
        ZNN_JIT_ASSERT(-16 <= n && n < 16);
    }

    ymm(ymm const&) = default;
    ymm& operator=(ymm const&) = default;

    operator bool() const { return which_ != -1; }

    std::string to_arg() const
    {
        ZNN_JIT_ASSERT(which_ != -1);
        return "%%ymm" + std::to_string(which_);
    }
};

class zmm
{
private:
    int_t which_;

public:
    zmm()
        : which_(-1)
    {
    }

    zmm(int_t n)
        : which_(n < 0 ? n + 32 : n)
    {
        ZNN_JIT_ASSERT(-32 <= n && n < 32);
    }

    zmm(zmm const&) = default;
    zmm& operator=(zmm const&) = default;

    operator bool() const { return which_ != -1; }

    std::string to_arg() const
    {
        ZNN_JIT_ASSERT(which_ != -1);
        return "%%zmm" + std::to_string(which_);
    }
};

} // namespace znn::jit
