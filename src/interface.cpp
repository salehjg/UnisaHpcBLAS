//
// Created by saleh on 3/29/25.
//

#include "interface.h"

template <typename Impl>
interface<Impl>::interface(): impl(std::make_unique<Impl>()) {}

