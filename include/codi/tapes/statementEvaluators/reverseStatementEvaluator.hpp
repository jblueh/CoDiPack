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

#include <algorithm>
#include <functional>
#include <type_traits>

#include "../../expressions/activeType.hpp"
#include "../../expressions/assignExpression.hpp"
#include "../../misc/exceptions.hpp"
#include "../../misc/macros.hpp"
#include "../../misc/memberStore.hpp"
#include "statementEvaluatorInterface.hpp"
#include "statementEvaluatorTapeInterface.hpp"

/** \copydoc codi::Namespace */
namespace codi {

  /**
   * @brief Only stores the function handle for the reverse evaluation.
   *
   * Uses the StatementEvaluatorTapeInterface.
   */
  struct ReverseStatementEvaluator : public StatementEvaluatorInterface {
    public:

      /*******************************************************************************/
      /// @name StatementEvaluatorInterface implementation
      /// @{

      using Handle = void*;  ///< Function pointer to the reverse evaluation.

      /// \copydoc StatementEvaluatorInterface::call
      template<StatementCall type, typename Tape, typename... Args>
      static void call(Handle const& h, Args&&... args) {
        using Expr = AssignExpression<ActiveType<Tape>, ActiveType<Tape>>;
        using CallGen = typename Tape::template StatementCallGenerator<type, Expr>;

        using Function = decltype(&CallGen::evaluate);

        Function func = (Function)h;

        if (StatementCall::Reverse == type) {
          func(std::forward<Args>(args)...);
        } else {
          CODI_EXCEPTION("ReverseStatementEvaluator only supports reverse evaluation calls.");
        }
      }

      /// \copydoc StatementEvaluatorInterface::createHandle
      template<typename Tape, typename Generator, typename Expr>
      static Handle createHandle() {
        return (Handle*)Generator::template statementEvaluateReverse<Expr>;
      }

      /// @}
  };
}
