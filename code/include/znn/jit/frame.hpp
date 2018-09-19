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

#include "any_of.hpp"
#include "registers.hpp"
#include "types.hpp"
#include "util.hpp"

#include <atomic>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

namespace znn::jit
{

class frame : public std::enable_shared_from_this<frame>
{
private:
    std::shared_ptr<frame> parent_;
    std::weak_ptr<frame>   child_;

    std::shared_ptr<std::ostringstream> ss_;
    std::shared_ptr<registers>          regs_;

    std::set<reg>    possible_regs_;
    std::set<reg>    used_regs_;
    std::vector<reg> pushed_;

    inline static std::atomic<std::uint64_t> next_label_{0};

    template <class First, class... Rest>
    void instruction_args(First const& first, Rest const&... rest)
    {
        *ss_ << ' ' << first;
        if constexpr (sizeof...(Rest) > 0)
        {
            *ss_ << ((std::string(",") + rest) + ...);
        }
    }

    template <class... Args>
    void instruction(std::string const& inst, Args const&... args)
    {
        ZNN_JIT_ASSERT(!child_.lock());
        *ss_ << "        \"" << inst;
        if constexpr (sizeof...(Args) > 0)
        {
            instruction_args((args.to_arg())...);
        }
        *ss_ << "\\n\\t\"\n";
    }

    frame(std::shared_ptr<frame> parent)
        : parent_(parent)
        , child_()
        , ss_(parent->ss_)
        , regs_(parent->regs_)
        , possible_regs_(regs_->used())
    {
    }

    frame(frame const&) = delete;
    frame& operator=(frame const&) = delete;

public:
    frame()
        : parent_()
        , child_()
        , ss_(std::make_shared<std::ostringstream>())
        , regs_(std::make_shared<registers>())
    {
    }

    ~frame()
    {
        for (auto r = pushed_.rbegin(); r != pushed_.rend(); ++r)
        {
            pop(*r);
        }
        for (auto r : used_regs_)
        {
            regs_->unuse(r);
        }
    }

    std::shared_ptr<frame> spawn()
    {
        ZNN_JIT_ASSERT(!child_.lock());

        auto ret = std::shared_ptr<frame>(new frame(shared_from_this()));
        child_   = ret;

        return ret;
    }

    reg get_register()
    {
        auto r = regs_->get_register();
        if (r)
        {
            used_regs_.insert(r);
            return r;
        }

        ZNN_JIT_ASSERT(possible_regs_.size());

        r = *possible_regs_.begin();

        possible_regs_.erase(r);
        push(r);
        pushed_.push_back(r);
        return r;
    }

    void uses_register(reg const& r)
    {
        ZNN_JIT_ASSERT(possible_regs_.count(r));
        possible_regs_.erase(r);
    }

    void return_register(reg const& r)
    {
        ZNN_JIT_ASSERT(used_regs_.count(r));
        used_regs_.erase(r);
        regs_->unuse(r);
    }

    void mov(any_of<ptr, val, reg, c_arg> const& s, reg const& d)
    {
        instruction("movq", s, d);
    }

    void vmov(any_of<ptr, ymm> const& s, ymm const& d)
    {
        instruction("vmovaps", s, d);
    }

    void vmov(ymm const& s, ptr const& d) { instruction("vmovaps", s, d); }

    void vmov(any_of<ptr, zmm> const& s, zmm const& d)
    {
        instruction("vmovaps", s, d);
    }

    void vmov(zmm const& s, ptr const& d) { instruction("vmovaps", s, d); }

    void vmovnt(ymm const& s, ptr const& d) { instruction("vmovntps", s, d); }

    void vmovnt(zmm const& s, ptr const& d) { instruction("vmovntps", s, d); }

    void add(any_of<ptr, val, reg> const& s, reg const& d)
    {
        instruction("addq", s, d);
    }

    void inc(reg const& r) { instruction("incq", r); }

    void sub(any_of<ptr, val, reg> const& s, reg const& d)
    {
        instruction("subq", s, d);
    }

    void shl(any_of<ptr, val, reg> const& s, reg const& d)
    {
        instruction("shlq", s, d);
    }

    void shr(any_of<ptr, val, reg> const& s, reg const& d)
    {
        instruction("shrq", s, d);
    }

    void smart_sub(int_t a, reg const& b)
    {
        if (a >> 31)
        {
            shr(val(4), b);
            sub(val(a >> 4), b);
            shl(val(4), b);
        }
        else
        {
            sub(val(a), b);
        }
    }

    void cmp(any_of<ptr, val, reg> const& a, reg const& b)
    {
        instruction("cmpq", a, b);
    }

    void jl(label const& l) { instruction("jl", l); }

    label add_label()
    {
        ZNN_JIT_ASSERT(!child_.lock());

        std::uint64_t l = next_label_++;

        *ss_ << "        \"" << l << ":";
        *ss_ << "\\n\\t\"\n";

        return label(std::to_string(l));
    }

    label future_label()
    {
        ZNN_JIT_ASSERT(!child_.lock());

        std::uint64_t l = next_label_++;
        return label(std::to_string(l));
    }

    void add_label(label const& l)
    {
        *ss_ << "        \"" << l.to_lab() << "\\n\\t\"\n";
    }

    void push(reg const& r) { instruction("pushq", r); }

    void pop(reg const& r) { instruction("popq", r); }

    void vbroadcastss(ptr const& a, ymm const& b)
    {
        instruction("vbroadcastss", a, b);
    }

    void prefetcht1(ptr const& a) { instruction("prefetcht1", a); }

    void prefetcht0(ptr const& a) { instruction("prefetcht0", a); }

    void ret() { instruction("ret"); }

    void set0(ymm const& r) { instruction("vxorps", r, r, r); }

    void set0(zmm const& r) { instruction("vpxord", r, r, r); }

    void vfmadd231ps(ymm const& a, ymm const& b, ymm const& c,
                     bool negate = false)
    {
        if (negate)
        {
            instruction("vfnmadd231ps", a, b, c);
        }
        else
        {
            instruction("vfmadd231ps", a, b, c);
        }
    }

    void vfmadd231ps(any_of<ptr, ptr_1to16, zmm> const& a, zmm const& b,
                     zmm const& c, bool negate = false)
    {
        if (negate)
        {
            instruction("vfnmadd231ps", a, b, c);
        }
        else
        {
            instruction("vfmadd231ps", a, b, c);
        }
    }

    std::string code() const { return ss_->str(); }
};

} // namespace znn::jit
