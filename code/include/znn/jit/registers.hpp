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

#include "types.hpp"
#include "util.hpp"

#include <cassert>
#include <cstdint>
#include <set>
#include <string>

namespace znn::jit
{

inline reg rax("rax");
inline reg rbx("rbx");
inline reg rcx("rcx");
inline reg rdx("rdx");
inline reg rsi("rsi");
inline reg rdi("rdi");
inline reg r8("r8");
inline reg r9("r9");
inline reg r10("r10");
inline reg r11("r11");
inline reg r12("r12");
inline reg r13("r13");
inline reg r14("r14");
inline reg r15("r15");

class registers
{
private:
    std::set<reg> available_;
    std::set<reg> used_;

public:
    registers(registers const&) = delete;
    registers& operator=(registers const&) = delete;
    registers(registers&&)                 = delete;
    registers& operator=(registers&&) = delete;

    registers()
    {
        available_.insert(rax);
        available_.insert(rcx);
        available_.insert(rdx);
        available_.insert(rbx);
        available_.insert(rsi);
        available_.insert(rdi);
        available_.insert(r8);
        available_.insert(r9);
        available_.insert(r10);
        available_.insert(r11);
        available_.insert(r12);
        available_.insert(r13);
        available_.insert(r14);
        available_.insert(r15);
    }

    void use(reg const& r)
    {
        ZNN_JIT_ASSERT(available_.count(r));
        available_.erase(r);
        used_.insert(r);
    }

    void unuse(reg const& r)
    {
        ZNN_JIT_ASSERT(used_.count(r));
        available_.insert(r);
        used_.erase(r);
    }

    std::set<reg> const& available() const { return available_; }

    reg get_register()
    {
        reg r;
        if (available_.size())
        {
            r = *(available_.rbegin());
            use(r);
        }
        return r;
    }

    std::set<reg> const& used() const { return used_; }
};

} // namespace znn::jit
