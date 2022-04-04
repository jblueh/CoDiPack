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
#include "../../misc/memberStore.hpp"
#include "../misc/statementSizes.hpp"

/** \copydoc codi::Namespace */
namespace codi {

  /**
   * @brief Tape side interface for StatementEvaluatorInterface.
   *
   * See StatementEvaluatorInterface for a full description.
   *
   * In every method the full evaluation of the statement needs to be done.
   * - 1. Load expression specific data
   * - 2. Call expression specific function
   */
  struct StatementEvaluatorTapeInterface {
    public:

      /*******************************************************************************/
      /// @name Interface definition

      /// Clear the adjoints of the expression.
      template<typename Expr, typename... Args>
      static void statementClearAdjoint(Args&&... args);

      /// Evaluate expression in a forward mode.
      template<typename Expr, typename... Args>
      static void statementEvaluateForward(Args&&... args);

      /// Evaluate primal expression.
      template<typename Expr, typename... Args>
      static void statementEvaluatePrimal(Args&&... args);

      /// Reset the primal values of the expression.
      template<typename Expr, typename... Args>
      static void statementResetPrimal(Args&&... args);

      /// Evaluate expression in a reverse mode.
      template<typename Expr, typename... Args>
      static void statementEvaluateReverse(Args&&... args);
  };

  /**
   * @brief Tape side interface for StatementEvaluatorInterface.
   *
   * See StatementEvaluatorInterface for a full description.
   *
   * The `statementEvaluate*Inner` methods needs to be stored by the StatementEvaluatorInterface. These methods
   * perform the `Call expression specific function` logic.
   *
   * The `statementEvaluate*Full` functions are called by the StatementEvaluatorInterface on a `call*` function call.
   * This performs the step `Load expression specific data` in an inline context. `inner` is the stored function pointer
   * in the handle.
   */
  struct StatementEvaluatorInnerTapeInterface {
    public:

      /*******************************************************************************/
      /// @name Interface definition

      /// Load the expression data and clear the adjoints of the expression
      template<typename Func, typename... Args>
      static void statementClearAdjointFull(Func const& inner, StatementSizes stmtSizes, Args&&... args);

      /// Load the expression data and evaluate the expression in a forward mode.
      template<typename Func, typename... Args>
      static void statementEvaluateForwardFull(Func const& inner, StatementSizes stmtSizes, Args&&... args);

      /// Load the expression data and evaluate the expression in a primal setting.
      template<typename Func, typename... Args>
      static void statementEvaluatePrimalFull(Func const& inner, StatementSizes stmtSizes, Args&&... args);

      /// Load the expression data and reset the primal values of the expression
      template<typename Func, typename... Args>
      static void statementResetPrimalFull(Func const& inner, StatementSizes stmtSizes, Args&&... args);

      /// Load the expression data and evaluate the expression in a reverse mode.
      template<typename Func, typename... Args>
      static void statementEvaluateReverseFull(Func const& inner, StatementSizes stmtSizes, Args&&... args);

      /// Clear the adjoints of the expression.
      template<typename Expr, typename... Args>
      static void statementClearAdjointInner(Args&&... args);

      /// Evaluate expression in a forward mode.
      template<typename Expr, typename... Args>
      static void statementEvaluateForwardInner(Args&&... args);

      /// Evaluate expression in a primal setting.
      template<typename Expr, typename... Args>
      static void statementEvaluatePrimalInner(Args&&... args);

      /// Reset the primal values of the expression.
      template<typename Expr, typename... Args>
      static void statementResetPrimalInner(Args&&... args);

      /// Evaluate expression in a reverse mode.
      template<typename Expr, typename... Args>
      static void statementEvaluateReverseInner(Args&&... args);
  };
}
