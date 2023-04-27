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
#include <type_traits>

#include "../config.h"
#include "../expressions/activeType.hpp"
#include "../expressions/lhsExpressionInterface.hpp"
#include "../expressions/logic/compileTimeTraversalLogic.hpp"
#include "../expressions/logic/traversalLogic.hpp"
#include "../misc/macros.hpp"
#include "../traits/expressionTraits.hpp"
#include "data/chunk.hpp"
#include "indices/linearIndexManager.hpp"
#include "interfaces/reverseTapeInterface.hpp"
#include "jacobianBaseTape.hpp"

/** \copydoc codi::Namespace */
namespace codi {

  enum class ElemeniationMissingOutput {
    Ignore = 0,
    Add,
    Throw
  };

  /**
   * @brief Final implementation for a Jacobian tape with a linear index management.
   *
   * This class implements the interface methods from the JacobianBaseTape.
   *
   * @tparam T_TapeTypes  JacobianTapeTypes definition.
   */
  template<typename T_TapeTypes>
  struct JacobianLinearTape : public JacobianBaseTape<T_TapeTypes, JacobianLinearTape<T_TapeTypes>> {
    public:

      using TapeTypes = CODI_DD(T_TapeTypes,
                                CODI_T(JacobianTapeTypes<double, double, IndexManagerInterface<int>,
                                                         DefaultChunkedData>));  ///< See JacobianLinearTape.

      using Base = JacobianBaseTape<TapeTypes, JacobianLinearTape>;  ///< Base class abbreviation.
      friend Base;  ///< Allow the base class to call protected and private methods.

      using Real = typename TapeTypes::Real;                  ///< See TapeTypesInterface.
      using Gradient = typename TapeTypes::Gradient;          ///< See TapeTypesInterface.
      using IndexManager = typename TapeTypes::IndexManager;  ///< See TapeTypesInterface.
      using Identifier = typename TapeTypes::Identifier;      ///< See TapeTypesInterface.
      using Position = typename Base::Position;               ///< See TapeTypesInterface.

      CODI_STATIC_ASSERT(IndexManager::IsLinear, "This class requires an index manager with a linear scheme.");

      /// Constructor
      JacobianLinearTape() : Base() {}

      using Base::clearAdjoints;

      /// \copydoc codi::PositionalEvaluationTapeInterface::clearAdjoints
      void clearAdjoints(Position const& start, Position const& end) {
        using IndexPosition = CODI_DD(typename IndexManager::Position, int);
        IndexPosition startIndex = this->externalFunctionData.template extractPosition<IndexPosition>(start);
        IndexPosition endIndex = this->externalFunctionData.template extractPosition<IndexPosition>(end);

        startIndex = std::min(startIndex, (IndexPosition)this->adjoints.size() - 1);
        endIndex = std::min(endIndex, (IndexPosition)this->adjoints.size() - 1);

        for (IndexPosition curPos = endIndex + 1; curPos <= startIndex; curPos += 1) {
          this->adjoints[curPos] = Gradient();
        }
      }

    protected:

      /// \copydoc codi::JacobianBaseTape::pushStmtData <br><br>
      /// Only the number of arguments is required for linear index managers.
      CODI_INLINE void pushStmtData(Identifier const& index, Config::ArgumentSize const& numberOfArguments) {
        CODI_UNUSED(index);

        this->statementData.pushData(numberOfArguments);
      }

      /// \copydoc codi::JacobianBaseTape::internalEvaluateForward_Step3_EvalStatements
      template<typename Adjoint>
      CODI_INLINE static void internalEvaluateForward_Step3_EvalStatements(
          /* data from call */
          JacobianLinearTape& tape, Adjoint* adjointVector,
          /* data from jacobian vector */
          size_t& curJacobianPos, size_t const& endJacobianPos, Real const* const rhsJacobians,
          Identifier const* const rhsIdentifiers,
          /* data from statement vector */
          size_t& curStmtPos, size_t const& endStmtPos, Config::ArgumentSize const* const numberOfJacobians,
          /* data from index handler */
          size_t const& startAdjointPos, size_t const& endAdjointPos) {
        CODI_UNUSED(endJacobianPos, endStmtPos);

        size_t curAdjointPos = startAdjointPos;

        while (curAdjointPos < endAdjointPos) {
          curAdjointPos += 1;

          Config::ArgumentSize const argsSize = numberOfJacobians[curStmtPos];

          if (Config::StatementInputTag != argsSize) {
            Adjoint lhsAdjoint = Adjoint();
            Base::incrementTangents(adjointVector, lhsAdjoint, argsSize, curJacobianPos, rhsJacobians, rhsIdentifiers);
            adjointVector[curAdjointPos] = lhsAdjoint;

            EventSystem<JacobianLinearTape>::notifyStatementEvaluateListeners(
                tape, curAdjointPos, GradientTraits::dim<Adjoint>(), GradientTraits::toArray(lhsAdjoint).data());
          }

          curStmtPos += 1;
        }
      }

      /// \copydoc codi::JacobianBaseTape::internalEvaluateReverse_Step3_EvalStatements
      template<typename Adjoint>
      CODI_INLINE static void internalEvaluateReverse_Step3_EvalStatements(
          /* data from call */
          JacobianLinearTape& tape, Adjoint* adjointVector,
          /* data from jacobianData */
          size_t& curJacobianPos, size_t const& endJacobianPos, Real const* const rhsJacobians,
          Identifier const* const rhsIdentifiers,
          /* data from statementData */
          size_t& curStmtPos, size_t const& endStmtPos, Config::ArgumentSize const* const numberOfJacobians,
          /* data from index handler */
          size_t const& startAdjointPos, size_t const& endAdjointPos) {
        CODI_UNUSED(endJacobianPos, endStmtPos);

        size_t curAdjointPos = startAdjointPos;

        while (curAdjointPos > endAdjointPos) {
          curStmtPos -= 1;
          Config::ArgumentSize const argsSize = numberOfJacobians[curStmtPos];

          if (Config::StatementInputTag != argsSize) {
            // No input value, perform regular statement evaluation.

            Adjoint const lhsAdjoint = adjointVector[curAdjointPos];  // We do not use the zero index, decrement of
                                                                      // curAdjointPos at the end of the loop.

            EventSystem<JacobianLinearTape>::notifyStatementEvaluateListeners(
                tape, curAdjointPos, GradientTraits::dim<Adjoint>(), GradientTraits::toArray(lhsAdjoint).data());

            if (Config::ReversalZeroesAdjoints) {
              adjointVector[curAdjointPos] = Adjoint();
            }

            Base::incrementAdjoints(adjointVector, lhsAdjoint, argsSize, curJacobianPos, rhsJacobians, rhsIdentifiers);
          }

          curAdjointPos -= 1;
        }
      }

      CODI_INLINE static bool getIncomingDependencies(
          std::map<Identifier, std::map<Identifier, Real>>& dependencies,
          Identifier const& lhsIentifier,
          std::map<Identifier, Real>& result,
          ElemeniationMissingOutput missingOutputHandling) {

        auto iter = dependencies.find(lhsIentifier);
        if(iter != dependencies.end()) {
          result = std::move(iter->second);
          dependencies.erase(iter);

          return true;
        } else {
          switch (missingOutputHandling) {
          case ElemeniationMissingOutput::Ignore:
            return false;
            break;
          case ElemeniationMissingOutput::Add:
            result.clear();
            result[lhsIentifier] = Real(1.0); // Add a self reference.
            return true;
            break;
          case ElemeniationMissingOutput::Throw:
            CODI_EXCEPTION("Node for '%d' not in depency map. It needs to be declared as an output.", (int)lhsIentifier);
            return false;
            break;
          default:
            CODI_EXCEPTION("Missing case '%d'.", (int)missingOutputHandling);
            return false;
            break;
          }
        }

      }

      CODI_INLINE static void internalEliminateTapeReverse(
          /* data from call */
          std::map<Identifier, std::map<Identifier, Real>>& dependencies, ElemeniationMissingOutput missingOutputHandling,
          /* data from jacobianData */
          size_t& curJacobianPos, size_t const& endJacobianPos, Real const* const rhsJacobians,
          Identifier const* const rhsIdentifiers,
          /* data from statementData */
          size_t& curStmtPos, size_t const& endStmtPos, Config::ArgumentSize const* const numberOfJacobians,
          /* data from index handler */
          size_t const& startAdjointPos, size_t const& endAdjointPos) {
        CODI_UNUSED(endJacobianPos, endStmtPos);

        size_t curAdjointPos = startAdjointPos;

        std::map<Identifier, Real> incomingDependencies;

        while (curAdjointPos > endAdjointPos) {
          curStmtPos -= 1;
          Config::ArgumentSize const argsSize = numberOfJacobians[curStmtPos];
          curJacobianPos -= argsSize;

          if (Config::StatementInputTag != argsSize) {
            // Skip input values, they would clear the map.

            if(getIncomingDependencies(dependencies, curAdjointPos, incomingDependencies, missingOutputHandling)) {
              for(auto& curEntry : incomingDependencies) {
                for(Config::ArgumentSize curOut = 0; curOut < argsSize; curOut += 1) {
                  Identifier const rhsIdentifier = rhsIdentifiers[curJacobianPos + curOut];
                  Real const rhsJacobian = rhsJacobians[curJacobianPos + curOut];

                  dependencies[rhsIdentifier][curEntry.first] += curEntry.second * rhsJacobian;
                }
              }
            }
          }

          curAdjointPos -= 1;
        }
      }

      CODI_NO_INLINE void internalEliminateTape(
          Position const& start,
          Position const& end,
          std::map<Identifier, std::map<Identifier, Real>>& dependencies,
          ElemeniationMissingOutput missingOutputHandling = ElemeniationMissingOutput::Ignore) {

        Base::jacobianData.evaluateReverse(start.inner, end.inner, internalEliminateTapeReverse, dependencies,
                                           missingOutputHandling);
      }

    public:

      CODI_NO_INLINE std::map<Identifier, std::map<Identifier, Real>>  eliminateTape(Position const& start,
                                                Position const& end,
                                                std::vector<Identifier>& outputs,
                                                ElemeniationMissingOutput missingOutputHandling = ElemeniationMissingOutput::Ignore) {

        std::map<Identifier, std::map<Identifier, Real>> dependencies;

        // Add a self reference to outputs. This are the starting points for the elimination. All other outputs are
        // ignored or an exception is thrown.
        for(Identifier curOutput : outputs) {
          dependencies[curOutput][curOutput] = Real(1.0);
        }

        internalEliminateTape(start, end, dependencies, missingOutputHandling);

        return dependencies;
      }

      CODI_NO_INLINE void eliminateTapeAndReplace(
          Position const& start,
          std::vector<ActiveType<JacobianLinearTape>*>& outputs,
          ElemeniationMissingOutput missingOutputHandling = ElemeniationMissingOutput::Ignore) {

        using LhsExpr = ActiveType<JacobianLinearTape>;

        std::map<Identifier, std::map<Identifier, Real>> dependencies;

        // Add a self reference to outputs. This are the starting points for the elimination. All other outputs are
        // ignored or an exception is thrown.
        for(LhsExpr* lhs : outputs) {
          dependencies[lhs->getIdentifier()][lhs->getIdentifier()] = Real(1.0);
        }
        internalEliminateTape(this->getPosition(), start, dependencies, missingOutputHandling);

        // Transpose dependencies.
        std::map<Identifier, std::map<Identifier, Real>> dependenciesTransposed;
        for(auto const& rhsEntry : dependencies) {
          for(auto const& lhsEntry : rhsEntry.second) {
            dependenciesTransposed[lhsEntry.first][rhsEntry.first] += lhsEntry.second;
          }
        }

        this->resetTo(start, false);

        // Add Jacobian to the tape.
        for(LhsExpr* curOutput: outputs) {

          Identifier curLhsIdentifier = curOutput->getIdentifier();
          auto const& rhsEntries = dependenciesTransposed[curLhsIdentifier];
          int entriesLeftToPUsh = (int)rhsEntries.size();
          int curEntriesToPush = 0;
          bool staggeringActive = false;

          for(auto const& rhsEntry : rhsEntries) {
            // Add a new statement if we added all from the last staggering iteration.
            if(0 == curEntriesToPush) {
              curEntriesToPush = entriesLeftToPUsh;
              if (curEntriesToPush > (int)Config::MaxArgumentSize) {
                curEntriesToPush = (int)Config::MaxArgumentSize - 1;
                if (staggeringActive) {  // Except in the first round, one Jacobian is reserved for the staggering.
                  curEntriesToPush -= 1;
                }
              }

              Identifier storedIdentifier = curLhsIdentifier;
              this->storeManual(curOutput->getValue(), curLhsIdentifier, curEntriesToPush + (int)staggeringActive);
              if(staggeringActive) {
                this->pushJacobianManual(Real(1.0), Real(), storedIdentifier);
              }
            }

            this->pushJacobianManual(rhsEntry.second, Real(), rhsEntry.first);

            entriesLeftToPUsh -= 1;
            curEntriesToPush -= 1;
          }

          curOutput->getIdentifier() = curLhsIdentifier;
        }
      }
  };
}
