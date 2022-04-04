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

#include "../../misc/macros.hpp"
#include "../../expressions/activeType.hpp"
#include "../../expressions/assignExpression.hpp"
#include "../../traits/expressionTraits.hpp"
#include "../misc/statementSizes.hpp"
#include "directStatementEvaluator.hpp"
#include "statementEvaluatorInterface.hpp"

/** \copydoc codi::Namespace */
namespace codi {

  /**
   * @brief Additional data required by an InnerStatementEvaluator.
   */
  struct InnerPrimalTapeStatementData {
    public:

      using Base = PrimalTapeStatementFunctions;  ///< Base class abbreviation.

      PrimalTapeStatementFunctions functions;
      StatementSizes stmtSizes;

      /// Constructor
      InnerPrimalTapeStatementData(PrimalTapeStatementFunctions functions, StatementSizes stmtSizes)
          : functions(functions),
            stmtSizes(stmtSizes) {}
  };

  /// Store InnerPrimalTapeStatementData as static variables for each combination of generator (tape) and expression
  /// used in the program.
  template<typename Tape, typename Expr>
  struct InnerStatementEvaluatorStaticStore {
    public:

      /// Static storage. Static construction is done by instantiating the statementEvaluate*Inner functions of the
      /// generator with Expr. Also evaluates the number of active type arguments and constant type arguments.
      static InnerPrimalTapeStatementData const staticStore;  ///< Static storage.
  };

  template<typename Generator, typename Expr>
  InnerPrimalTapeStatementData const InnerStatementEvaluatorStaticStore<Generator, Expr>::staticStore(
      PrimalTapeStatementFunctions(
        (typename PrimalTapeStatementFunctions::Handle)Generator::template statementClearAdjointInner<Expr>,
        (typename PrimalTapeStatementFunctions::Handle)Generator::template statementEvaluateForwardInner<Expr>,
        (typename PrimalTapeStatementFunctions::Handle)Generator::template statementEvaluatePrimalInner<Expr>,
        (typename PrimalTapeStatementFunctions::Handle)Generator::template statementResetPrimalInner<Expr>,
        (typename PrimalTapeStatementFunctions::Handle)Generator::template statementEvaluateReverseInner<Expr>),
      StatementSizes::create<Expr>());

  /**
   * @brief Expression evaluation in the inner function. Data loading in the compilation context of the tape.
   * Storing in static context.
   *
   * Data loading is performed in the compilation context of the tape. The tape will then call the handle for the
   * evaluation of the expression after the data is loaded. This evaluator stores expression specific data and the
   * inner function handles.
   *
   * See StatementEvaluatorInterface for details.
   *
   * @tparam T_Real  The computation type of a tape, usually chosen as ActiveType::Real.
   */
  template<typename T_Real>
  struct InnerStatementEvaluator : public StatementEvaluatorInterface<T_Real> {
    public:

      using Real = CODI_DD(T_Real, double);  ///< See InnerStatementEvaluator.

      /*******************************************************************************/
      /// @name StatementEvaluatorInterface implementation
      /// @{

      using Handle = InnerPrimalTapeStatementData const*;  ///< Pointer to static storage location.

      /// \copydoc StatementEvaluatorInterface::callClearAdjoint
      template<typename Tape, typename... Args>
      static void callClearAdjoint(Handle const& h, Args&&... args) {
        Tape::statementClearAdjointFull((FunctionReverse<Tape>)h->functions.clearAdjoints, h->stmtSizes,
                                          std::forward<Args>(args)...);
      }

      /// \copydoc StatementEvaluatorInterface::callForward
      template<typename Tape, typename... Args>
      static void callForward(Handle const& h, Args&&... args) {
        Tape::statementEvaluateForwardFull((FunctionForward<Tape>)h->functions.forward, h->stmtSizes,
                                                  std::forward<Args>(args)...);
      }

      /// \copydoc StatementEvaluatorInterface::callPrimal
      template<typename Tape, typename... Args>
      static void callPrimal(Handle const& h, Args&&... args) {
        Tape::statementEvaluatePrimalFull((FunctionPrimal<Tape>)h->functions.primal, h->stmtSizes,
                                                 std::forward<Args>(args)...);
      }

      /// \copydoc StatementEvaluatorInterface::callResetPrimal
      template<typename Tape, typename... Args>
      static void callResetPrimal(Handle const& h, Args&&... args) {
        Tape::statementResetPrimalFull((FunctionReverse<Tape>)h->functions.resetPrimal, h->stmtSizes,
                                         std::forward<Args>(args)...);
      }

      /// \copydoc StatementEvaluatorInterface::callReverse
      template<typename Tape, typename... Args>
      static void callReverse(Handle const& h, Args&&... args) {
        Tape::statementEvaluateReverseFull((FunctionReverse<Tape>)h->functions.reverse, h->stmtSizes,
                                           std::forward<Args>(args)...);
      }

      /// \copydoc StatementEvaluatorInterface::createHandle
      template<typename Tape, typename Generator, typename Expr>
      static Handle createHandle() {
        return &InnerStatementEvaluatorStaticStore<Generator, Expr>::staticStore;
      }

      /// @}

    protected:

      /// Full clear adjoints function type.
      template<typename Tape>
      using FunctionClearAdjoint = decltype(&Tape::template statementClearAdjointInner<AssignExpression<ActiveType<Tape>, ActiveType<Tape>>>);

      /// Full forward function type.
      template<typename Tape>
      using FunctionForward = decltype(&Tape::template statementEvaluateForwardInner<AssignExpression<ActiveType<Tape>, ActiveType<Tape>>>);

      /// Full primal function type.
      template<typename Tape>
      using FunctionPrimal = decltype(&Tape::template statementEvaluatePrimalInner<AssignExpression<ActiveType<Tape>, ActiveType<Tape>>>);

      /// Full reset primals function type.
      template<typename Tape>
      using FunctionResetPrimal = decltype(&Tape::template statementResetPrimalInner<AssignExpression<ActiveType<Tape>, ActiveType<Tape>>>);

      /// Full reverse function type.
      template<typename Tape>
      using FunctionReverse = decltype(&Tape::template statementEvaluateReverseInner<AssignExpression<ActiveType<Tape>, ActiveType<Tape>>>);
  };
}
