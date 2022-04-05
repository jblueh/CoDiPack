/*
 * CoDiPack, a Code Differentiation Package
 *
 * Copyright (C) 2015-2022 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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

#include "../../misc/macros.hpp"
#include "../../config.h"

/** \copydoc codi::Namespace */
namespace codi {

  struct StatementSizes {
      size_t const maxActiveArgs;
      size_t const maxConstantArgs;
      size_t const maxOutputArgs;

      StatementSizes(size_t const maxActiveArgs, size_t const maxConstantArgs, size_t const maxOutputArgs) :
        maxActiveArgs(maxActiveArgs),
        maxConstantArgs(maxConstantArgs),
        maxOutputArgs(maxOutputArgs) {}

      template<typename Expr>
      static StatementSizes create() {
        using Lhs = typename Expr::Lhs;
        using Rhs = typename Expr::Rhs;

        size_t constexpr maxActiveArgs = ExpressionTraits::NumberOfActiveTypeArguments<Rhs>::value;
        size_t constexpr maxConstantArgs = ExpressionTraits::NumberOfConstantTypeArguments<Rhs>::value;
        size_t constexpr maxOutputArgs = ExpressionTraits::NumberOfActiveTypeArguments<Lhs>::value;

        return StatementSizes(maxActiveArgs, maxConstantArgs, maxOutputArgs);
      }
  };
}
