/*
 * CoDiPack, a Code Differentiation Package
 *
 * Copyright (C) 2015-2021 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
 *
 * Lead developers: Max Sagebaum, Johannes Blühdorn (SciComp, TU Kaiserslautern)
 *
 * This file is part of CoDiPack (http://www.scicomp.uni-kl.de/software/codi).
 *
 * CoDiPack is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * CoDiPack is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU
 * General Public License along with CoDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * For other licensing options please contact us.
 *
 * Authors:
 *  - SciComp, TU Kaiserslautern:
 *    - Max Sagebaum
 *    - Johannes Blühdorn
 *    - Former members:
 *      - Tim Albring
 */
#pragma once

#include <type_traits>
#include <utility>

#include "../misc/macros.hpp"
#include "../config.h"

/** \copydoc codi::Namespace */
namespace codi {

  template<std::size_t N>
  struct num {
      static const constexpr auto value = N;
  };

  template<class F, std::size_t... Is>
  CODI_INLINE void static_for(F func, std::index_sequence<Is...>) {
    using expander = int[];
    (void)expander{0, ((void)func(num<Is>{}), 0)...};
  }

  template<std::size_t N, typename F>
  CODI_INLINE void static_for(F func) {
    static_for(func, std::make_index_sequence<N>());
  }
}
