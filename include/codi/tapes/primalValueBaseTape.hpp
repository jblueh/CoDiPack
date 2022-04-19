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
#include <cmath>
#include <functional>
#include <type_traits>
#include <utility>

#include "../misc/macros.hpp"
#include "../misc/memberStore.hpp"
#include "../config.h"
#include "../expressions/aggregatedExpressionType.hpp"
#include "../expressions/assignExpression.hpp"
#include "../expressions/lhsExpressionInterface.hpp"
#include "../expressions/logic/compileTimeTraversalLogic.hpp"
#include "../expressions/logic/constructStaticContext.hpp"
#include "../expressions/logic/helpers/forEachLeafLogic.hpp"
#include "../expressions/logic/helpers/jacobianComputationLogic.hpp"
#include "../expressions/logic/traversalLogic.hpp"
#include "../traits/expressionTraits.hpp"
#include "misc/primalAdjointVectorAccess.hpp"
#include "misc/statementSizes.hpp"
#include "commonTapeImplementation.hpp"
#include "data/chunk.hpp"
#include "data/chunkedData.hpp"
#include "indices/indexManagerInterface.hpp"
#include "statementEvaluators/statementEvaluatorInterface.hpp"
#include "statementEvaluators/statementEvaluatorTapeInterface.hpp"


/** \copydoc codi::Namespace */
namespace codi {

  template<typename T>
  using StackArray = std::array<T, Config::MaxArgumentSize>;

  /**
   * @brief Type definitions for the primal value tapes.
   *
   * @tparam T_Real                See TapeTypesInterface.
   * @tparam T_Gradient            See TapeTypesInterface.
   * @tparam T_IndexManager        Index manager for the tape. Needs to implement IndexManagerInterface.
   * @tparam T_StatementEvaluator  Statement handle generator. Needs to implement StatementEvaluatorInterface and
   *                               StatementEvaluatorInnerTapeInterface.
   * @tparam T_Data                See TapeTypesInterface.
   */
  template<typename T_Real, typename T_Gradient, typename T_IndexManager, typename T_StatementEvaluator,
           template<typename, typename> class T_Data>
  struct PrimalValueTapeTypes : public TapeTypesInterface {
    public:

      using Real = CODI_DD(T_Real, double);                                              ///< See PrimalValueTapeTypes.
      using Gradient = CODI_DD(T_Gradient, double);                                      ///< See PrimalValueTapeTypes.
      using IndexManager = CODI_DD(T_IndexManager, CODI_T(IndexManagerInterface<int>));  ///< See PrimalValueTapeTypes.
      using StatementEvaluator = CODI_DD(T_StatementEvaluator,
                                         StatementEvaluatorInterface);  ///< See PrimalValueTapeTypes.
      template<typename Chunk, typename Nested>
      using Data = CODI_DD(CODI_T(T_Data<Chunk, Nested>),
                           CODI_T(DataInterface<Nested>));  ///< See PrimalValueTapeTypes.

      using Identifier = typename IndexManager::Index;    ///< See IndexManagerInterface.
      using PassiveReal = RealTraits::PassiveReal<Real>;  ///< Basic computation type.

      constexpr static bool IsLinearIndexHandler = IndexManager::IsLinear;  ///< True if the index manager is linear.
      constexpr static bool IsStaticIndexHandler =
          !IsLinearIndexHandler;  ///< For reuse index management, a static index manager is used.

      using EvalHandle = typename StatementEvaluator::Handle;  ///< Handle type returned by the statement generator.

      using FixedSizeDataChunk = Chunk1<char>;  ///< Chunk for byte values.
      using DynamicSizeDataChunk = Chunk1<char>;  ///< Chunk for byte values.

      /// Per statement data that is always read. E.g. the handle for the statement.
      using FixedSizeData = Data<FixedSizeDataChunk, IndexManager>;

      /// Per statement data that depends on the statement itself. E.g. rhs identifiers.
      using DynamicSizeData = Data<DynamicSizeDataChunk, FixedSizeData>;

      using NestedData = DynamicSizeData;  ///< See TapeTypesInterface.
  };

  /**
   * @brief Base class for all standard Primal value tape implementations.
   *
   * This class provides nearly a full implementation of the FullTapeInterface. There are just a few internal methods
   * left which need to be implemented by the final classes. These methods depend significantly on the index management
   * scheme and are performance critical.
   *
   * Tape evaluations are performed in three steps with two wrapper steps beforehand. Each methods calls the next
   * method:
   * - evaluate
   * - internalEvaluate*
   * - internalEvaluate*_Step1_ExtFunc
   * - internalEvaluate*_Step2_DataExtraction
   * - internalEvaluate*_Step3_EvalStatements
   * The placeholder stands for Reverse, Forward, or Primal.
   *
   * @tparam T_TapeTypes needs to implement PrimalValueTapeTypes.
   * @tparam T_Impl Type of the final implementations
   */
  template<typename T_TapeTypes, typename T_Impl>
  struct PrimalValueBaseTape : public CommonTapeImplementation<T_TapeTypes, T_Impl>,
                               public StatementEvaluatorTapeInterface,
                               public StatementEvaluatorInnerTapeInterface {
    public:

      /// See PrimalValueBaseTape.
      using TapeTypes = CODI_DD(T_TapeTypes,
                                CODI_T(PrimalValueTapeTypes<double, double, IndexManagerInterface<int>,
                                                            StatementEvaluatorInterface, DefaultChunkedData>));
      /// See PrimalValueBaseTape.
      using Impl = CODI_DD(T_Impl, CODI_T(FullTapeInterface<double, double, int, EmptyPosition>));

      using Base = CommonTapeImplementation<TapeTypes, Impl>;  ///< Base class abbreviation.
      friend Base;  ///< Allow the base class to call protected and private methods.

      using Real = typename TapeTypes::Real;                              ///< See TapeTypesInterface.
      using Gradient = typename TapeTypes::Gradient;                      ///< See TapeTypesInterface.
      using IndexManager = typename TapeTypes::IndexManager;              ///< See TapeTypesInterface.
      using StatementEvaluator = typename TapeTypes::StatementEvaluator;  ///< See PrimalValueTapeTypes.
      using Identifier = typename TapeTypes::Identifier;                  ///< See PrimalValueTapeTypes.

      using EvalHandle = typename TapeTypes::EvalHandle;  ///< See PrimalValueTapeTypes.

      using FixedSizeData = typename TapeTypes::FixedSizeData;
      using DynamicSizeData = typename TapeTypes::DynamicSizeData;

      using PassiveReal = RealTraits::PassiveReal<Real>;  ///< Basic computation type.

      using NestedPosition = typename DynamicSizeData::Position;  ///< See PrimalValueTapeTypes.
      using Position = typename Base::Position;                     ///< See TapeTypesInterface.

      template<typename Adjoint>
      using VectorAccess =
          PrimalAdjointVectorAccess<Real, Identifier, Adjoint>;  ///< Vector access type generated by this tape.

      static bool constexpr AllowJacobianOptimization = false;  ///< See InternalStatementRecordingTapeInterface.
      static bool constexpr HasPrimalValues = true;             ///< See PrimalEvaluationTapeInterface.
      static bool constexpr LinearIndexHandling =
          TapeTypes::IsLinearIndexHandler;  ///< See IdentifierInformationTapeInterface.
      static bool constexpr RequiresPrimalRestore =
          !TapeTypes::IsLinearIndexHandler;  ///< See PrimalEvaluationTapeInterface.

    protected:

      static EvalHandle const jacobianExpressions[Config::MaxArgumentSize];

      MemberStore<IndexManager, Impl, TapeTypes::IsStaticIndexHandler> indexManager;  ///< Index manager.
      FixedSizeData fixedSizeData;
      DynamicSizeData dynamicSizeData;

      struct ManualStatementData {
        public:
          Real* jacobians;
          Identifier* identifiers;

          size_t pos;
      };

      ManualStatementData manualStatementData;

      std::vector<Gradient> adjoints;  ///< Evaluation vector for AD.
      std::vector<Real> primals;       ///< Current state of primal values in the program.
      std::vector<Real> primalsCopy;   ///< Copy of primal values for AD evaluations.

    private:

      CODI_INLINE Impl const& cast() const {
        return static_cast<Impl const&>(*this);
      }

      CODI_INLINE Impl& cast() {
        return static_cast<Impl&>(*this);
      }

    protected:

      /*******************************************************************************/
      /// @name Interface definition
      /// @{

      /// Perform a forward evaluation of the tape. Arguments are from the recursive eval methods of the DataInterface.
      template<typename... Args>
      static void internalEvaluateForward_Step3_EvalStatements(Args&&... args);

      /// Perform a primal evaluation of the tape. Arguments are from the recursive eval methods of the DataInterface.
      template<typename... Args>
      static void internalEvaluatePrimal_Step3_EvalStatements(Args&&... args);

      /// Perform a reverse evaluation of the tape. Arguments are from the recursive eval methods of the DataInterface.
      template<typename... Args>
      static void internalEvaluateReverse_Step3_EvalStatements(Args&&... args);

      /// Reset the primal values to the given position.
      void internalResetPrimalValues(Position const& pos);

      /// @}

    public:

      /// Constructor
      PrimalValueBaseTape()
          : Base(),
            indexManager(Config::MaxArgumentSize),  // Reserve first items for passive values.
            fixedSizeData(Config::ChunkSize * 8),
            dynamicSizeData(Config::ChunkSize * 8),
            manualStatementData(),
            adjoints(1),  // Ensure that adjoint[0] exists, see its use in gradient() const.
            primals(0),
            primalsCopy(0) {
        checkPrimalSize(true);

        fixedSizeData.setNested(&indexManager.get());
        dynamicSizeData.setNested(&fixedSizeData);

        Base::init(&dynamicSizeData);

        Base::options.insert(TapeParameters::AdjointSize);
        Base::options.insert(TapeParameters::DynamicSizeDataSize);
        Base::options.insert(TapeParameters::LargestIdentifier);
        Base::options.insert(TapeParameters::PrimalSize);
        Base::options.insert(TapeParameters::FixedSizeDataSize);

      }

      /*******************************************************************************/
      /// @name Functions from GradientAccessTapeInterface
      /// @{

      /// \copydoc codi::GradientAccessTapeInterface::gradient(Identifier const&)
      CODI_INLINE Gradient& gradient(Identifier const& identifier) {
        checkAdjointSize(identifier);

        return adjoints[identifier];
      }

      /// \copydoc codi::GradientAccessTapeInterface::gradient(Identifier const&) const
      CODI_INLINE Gradient const& gradient(Identifier const& identifier) const {
        if (identifier > (Identifier)adjoints.size()) {
          return adjoints[0];
        } else {
          return adjoints[identifier];
        }
      }

      /// @}
      /*******************************************************************************/
      /// @name Functions from InternalStatementRecordingTapeInterface
      /// @{

      /// \copydoc codi::InternalStatementRecordingTapeInterface::initIdentifier()
      template<typename Real>
      CODI_INLINE void initIdentifier(Real& value, Identifier& identifier) {
        CODI_UNUSED(value);

        identifier = IndexManager::InactiveIndex;
      }

      /// \copydoc codi::InternalStatementRecordingTapeInterface::destroyIdentifier()
      template<typename Real>
      CODI_INLINE void destroyIdentifier(Real& value, Identifier& identifier) {
        CODI_UNUSED(value);

        indexManager.get().freeIndex(identifier);
      }

      /// @}

      struct LinearStmtPackHelper {
        public:
          size_t&  __restrict__ linearAdjointPos;  ///< Position of the lhs adjoint.
          Config::ArgumentSize numberOfPassiveArguments;  ///< Number of passive arguments.
      };

      struct ReuseStmtPackHelper {
        public:
          static size_t tempAdjointPos;  ///< Position of the lhs adjoint.
          Config::ArgumentSize numberOfPassiveArguments;  ///< Number of passive arguments.
      };

      using StmtPackHelper = typename std::conditional<LinearIndexHandling, LinearStmtPackHelper, ReuseStmtPackHelper>::type;

      /// Statement evaluation arguments for linear index handlers.
      struct StatementEvalArgumentsBase {
        public:
          size_t&  __restrict__ linearAdjointPos;  ///< Position of the lhs adjoint.
          Config::ArgumentSize numberOfPassiveArguments;  ///< Number of passive arguments.
          size_t& __restrict__ curDynamicSizePos;  ///< Position for the dynamic size statement data.
          char* const __restrict__ dynamicSizeValues;  ///< Pointer to the dynamic size statement data.

          StatementEvalArgumentsBase(
              size_t&  __restrict__ linearAdjointPos,
              Config::ArgumentSize numberOfPassiveArguments,
              size_t& __restrict__ curDynamicSizePos,
              char* const __restrict__ dynamicSizeValues) :
            linearAdjointPos(linearAdjointPos),
            numberOfPassiveArguments(numberOfPassiveArguments),
            curDynamicSizePos(curDynamicSizePos),
            dynamicSizeValues(dynamicSizeValues)
          {}

          StatementEvalArgumentsBase(
              Config::ArgumentSize numberOfPassiveArguments,
              size_t& __restrict__ curDynamicSizePos,
              char* const __restrict__ dynamicSizeValues) :
            StatementEvalArgumentsBase(ReuseStmtPackHelper::tempAdjointPos, numberOfPassiveArguments, curDynamicSizePos, dynamicSizeValues) {}

          template<typename ... Args>
          StatementEvalArgumentsBase(LinearStmtPackHelper packHelper, Args&& ... args) :
            StatementEvalArgumentsBase(packHelper.linearAdjointPos, packHelper.numberOfPassiveArguments, std::forward<Args>(args)...) {}

          template<typename ... Args>
          StatementEvalArgumentsBase(ReuseStmtPackHelper packHelper, Args&& ... args) :
            StatementEvalArgumentsBase(ReuseStmtPackHelper::tempAdjointPos, packHelper.numberOfPassiveArguments, std::forward<Args>(args)...) {}
      };

      struct LinearStatementEvalArguments : public StatementEvalArgumentsBase {
        public:
          using StatementEvalArgumentsBase::StatementEvalArgumentsBase;

          /// Moves the linear adjoint position by the number of output arguments.
          CODI_INLINE void updateAdjointPosForward(size_t nOutputArgs) {
            this->linearAdjointPos += nOutputArgs;
          }

          /// Moves the linear adjoint position by the number of output arguments.
          CODI_INLINE void updateAdjointPosReverse(size_t nOutputArgs) {
            this->linearAdjointPos -= nOutputArgs;
          }

          /// Computes the lhs identifier based on the linear adjoint position.
          CODI_INLINE Identifier getLhsIdentifier(size_t pos, Identifier const* __restrict__ identifiers) {
            CODI_UNUSED(identifiers);

            return this->linearAdjointPos + 1 + pos;
          }

          CODI_INLINE LinearStmtPackHelper unpackVariadic() {
            return LinearStmtPackHelper{this->linearAdjointPos, this->numberOfPassiveArguments};
          }
      };

      struct ReuseStatementEvalArguments : public StatementEvalArgumentsBase {
        public:
          using StatementEvalArgumentsBase::StatementEvalArgumentsBase;

          /// Does nothing.
          CODI_INLINE void updateAdjointPosForward(size_t nOutputArgs) {
            CODI_UNUSED(nOutputArgs);
          }

          /// Does nothing.
          CODI_INLINE void updateAdjointPosReverse(size_t nOutputArgs) {
            CODI_UNUSED(nOutputArgs);
          }

          /// Return the lhs identifier from the identifier vector.
          CODI_INLINE Identifier getLhsIdentifier(size_t pos, Identifier const* __restrict__ identifiers) {
            return identifiers[pos];
          }

          CODI_INLINE ReuseStmtPackHelper unpackVariadic() {
            return ReuseStmtPackHelper{this->numberOfPassiveArguments};
          }
      };

      using StatementEvalArguments = typename std::conditional<LinearIndexHandling, LinearStatementEvalArguments, ReuseStatementEvalArguments>::type;

#define STMT_ARGS StmtPackHelper packHelper, size_t& __restrict__ curDynamicSizePos, char* const __restrict__ dynamicSizeValues
#define STMT_ARGS_UNPACK(arg) (arg).unpackVariadic(), (arg).curDynamicSizePos, (arg).dynamicSizeValues
#define STMT_ARGS_PACK StatementEvalArguments{packHelper, curDynamicSizePos, dynamicSizeValues}
#define STMT_ARGS_FORWARD packHelper, curDynamicSizePos, dynamicSizeValues

      /// Helper structure for reading and writing the fixed size statement data.
      ///
      /// A call to readForward or readReverse will populate the member data.
      struct FixedSizeStatementData {
        public:
          EvalHandle handle;  ///< The currently read handle.
          Config::ArgumentSize numberOfPassiveArguments;  ///< The currently read number of passive arguments.

          /// See FixedSizeStatementData.
          ///
          /// @return The new position of in values.
          CODI_INLINE size_t readForward(char const* const values, size_t pos) {
            numberOfPassiveArguments = *((Config::ArgumentSize const*)(&values[pos]));
            pos += sizeof(Config::ArgumentSize);
            handle = *((EvalHandle const*)(&values[pos]));
            pos += sizeof(EvalHandle);

            return pos;
          }

          /// See FixedSizeStatementData.
          ///
          /// @return The new position of in values.
          CODI_INLINE size_t readReverse(char const* const values, size_t pos) {
            pos -= sizeof(EvalHandle);
            handle = *((EvalHandle const*)(&values[pos]));
            pos -= sizeof(Config::ArgumentSize);
            numberOfPassiveArguments = *((Config::ArgumentSize const*)(&values[pos]));

            return pos;
          }

          /// Reserve the data for the fixed size data stream.
          CODI_INLINE static void reserve(FixedSizeData& data) {
            data.reserveItems(sizeof(Config::ArgumentSize) + sizeof(EvalHandle));
          }

          /// Store the data for the fixed data stream.
          CODI_INLINE static void store(FixedSizeData& data, Config::ArgumentSize numberOfPassiveArguments, EvalHandle handle) {
            data.pushData(numberOfPassiveArguments);
            data.pushData(handle);
          }
      };

      /// Helper structure for reading the dynamic statement data.
      ///
      /// A call to readForward or readReverse will create a data structure.
      struct DynamicStatementData {
        public:
          Real* __restrict__ oldPrimalValues;  ///< Overwritten primal values. Only used for reuse index handling.
          Identifier* __restrict__ lhsIdentifiers;  ///< Lhs identifiers. Only used for reuse index handling.

          PassiveReal* __restrict__ constantValues;  ///< Constant values in the statement.
          Identifier* __restrict__ rhsIdentifiers;   ///< Rhs identifiers.
          Real* __restrict__ passiveValues;          ///< Passive values from active arguments of the statement.

          CODI_INLINE static DynamicStatementData readForward(StatementSizes const& stmtSizes, StatementEvalArguments& stmtArgs) {

            return readForward(stmtSizes, stmtArgs.dynamicSizeValues, stmtArgs.curDynamicSizePos, stmtArgs.numberOfPassiveArguments);
          }
          /// The dynamic data position in stmtArgs is updated.
          CODI_INLINE static DynamicStatementData readForward(StatementSizes const& stmtSizes,
                                                              char* __restrict__ dynamicSizeValues,
                                                              size_t& __restrict__ curDynamicSizePos,
                                                              size_t const& __restrict__ numberOfPassiveArguments) {
            DynamicStatementData data;

            size_t pos = curDynamicSizePos;

            data.passiveValues = (Real*)(&dynamicSizeValues[pos]);
            pos += numberOfPassiveArguments * sizeof(Real);
            data.rhsIdentifiers = (Identifier*)(&dynamicSizeValues[pos]);
            pos += stmtSizes.maxActiveArgs * sizeof(Identifier);
            data.constantValues = (PassiveReal*)(&dynamicSizeValues[pos]);
            pos += stmtSizes.maxConstantArgs * sizeof(PassiveReal);

            if(!LinearIndexHandling) {
              data.lhsIdentifiers = ((Identifier*)(&dynamicSizeValues[pos]));
              pos += stmtSizes.maxOutputArgs * sizeof(Identifier);
              data.oldPrimalValues = ((Real*)(&dynamicSizeValues[pos]));
              pos += stmtSizes.maxOutputArgs * sizeof(Real);
            }

            curDynamicSizePos = pos;

            return data;
          }

          /// The dynamic data position in stmtArgs is updated.
          CODI_INLINE static DynamicStatementData readReverse(StatementSizes const& stmtSizes, StatementEvalArguments& stmtArgs) {
            DynamicStatementData data;

            size_t pos = stmtArgs.curDynamicSizePos;

            if(!LinearIndexHandling) {
              pos -= stmtSizes.maxOutputArgs * sizeof(Real);
              data.oldPrimalValues = ((Real*)(&stmtArgs.dynamicSizeValues[pos]));
              pos -= stmtSizes.maxOutputArgs * sizeof(Identifier);
              data.lhsIdentifiers = ((Identifier*)(&stmtArgs.dynamicSizeValues[pos]));
            }

            pos -= stmtSizes.maxConstantArgs * sizeof(PassiveReal);
            data.constantValues = (PassiveReal*)(&stmtArgs.dynamicSizeValues[pos]);
            pos -= stmtSizes.maxActiveArgs * sizeof(Identifier);
            data.rhsIdentifiers = (Identifier*)(&stmtArgs.dynamicSizeValues[pos]);
            pos -= stmtArgs.numberOfPassiveArguments * sizeof(Real);
            data.passiveValues = (Real*)(&stmtArgs.dynamicSizeValues[pos]);

            stmtArgs.curDynamicSizePos = pos;

            return data;
          }

          /// Overwrite the old primal values with new ones. Usually called in a forward tape evaluation.
          CODI_INLINE void updateOldPrimalValues(StatementSizes const& stmtSizes, StatementEvalArguments& revArgs,
                                     Real* __restrict__ primalVector) {
            for(size_t iLhs = 0; iLhs < stmtSizes.maxOutputArgs; iLhs += 1) {
              Identifier const lhsIdentifier = revArgs.getLhsIdentifier(iLhs, lhsIdentifiers);
              oldPrimalValues[iLhs] = primalVector[lhsIdentifier];
            }
          }

          /// Copies the passive values into the first entries of the primal value vector.
          CODI_INLINE void copyPassiveValuesIntoPrimalVector(StatementEvalArguments& revArgs,
                                                             Real* __restrict__ primalVector) {
            for (Config::ArgumentSize curPos = 0; curPos < revArgs.numberOfPassiveArguments; curPos += 1) {
              primalVector[curPos] = passiveValues[curPos];
            }
          }

          /// Reserve the data for the dynamic size data stream.
          CODI_INLINE static void reserve(DynamicSizeData& data, StatementSizes const& sizes, size_t const& activeArguments) {
            data.reserveItems(computeSize(sizes, activeArguments));
          }

          /// Store the data for the fixed data stream.
          CODI_INLINE static DynamicStatementData store(DynamicSizeData& data, StatementSizes const& sizes, size_t const& activeArguments) {
            char* dataPointer = nullptr;
            data.getDataPointer(dataPointer);
            data.addDataSize(computeSize(sizes, activeArguments));

            size_t pos = 0;
            return readForward(sizes, dataPointer, pos, sizes.maxActiveArgs - activeArguments);
          }

        private:

          /// Size in bytes of the dynamic statement data.
          CODI_INLINE static size_t computeSize(StatementSizes const& sizes, size_t const& activeArguments) {
            size_t dynamicSize = (sizes.maxActiveArgs - activeArguments) * sizeof(Real)
                                  + (sizes.maxActiveArgs) * sizeof(Identifier)
                                  + sizes.maxConstantArgs * sizeof(PassiveReal);
            if(!LinearIndexHandling) {
              dynamicSize += sizes.maxOutputArgs * (sizeof(Identifier) + sizeof(Real));
            }

            return dynamicSize;
          }
      };

    protected:

      /// Count all arguments that have non-zero index.
      struct CountActiveArguments : public ForEachLeafLogic<CountActiveArguments> {
        public:

          /// \copydoc codi::ForEachLeafLogic::handleActive
          template<typename Node>
          CODI_INLINE void handleActive(Node const& node, size_t& numberOfActiveArguments) {
            if (CODI_ENABLE_CHECK(Config::CheckZeroIndex, IndexManager::InactiveIndex != node.getIdentifier())) {
              numberOfActiveArguments += 1;
            }
          }
      };

      /// Push all data for each argument.
      struct PushStatementData : public ForEachLeafLogic<PushStatementData> {
        public:

          /// \copydoc codi::ForEachLeafLogic::handleActive
          template<typename Node>
          CODI_INLINE void handleActive(Node const& node,
                                        DynamicStatementData& dynamicPointers,
                                        size_t& curPassiveArgument,
                                        size_t& idPos,
                                        size_t& constantPos) {
            CODI_UNUSED(constantPos);

            Identifier rhsIndex = node.getIdentifier();
            if (CODI_ENABLE_CHECK(Config::CheckZeroIndex, IndexManager::InactiveIndex == rhsIndex)) {
              rhsIndex = curPassiveArgument;

              dynamicPointers.passiveValues[curPassiveArgument] = node.getValue();
              curPassiveArgument += 1;
            }

            dynamicPointers.rhsIdentifiers[idPos] = rhsIndex;
            idPos += 1;
          }

          /// \copydoc codi::ForEachLeafLogic::handleConstant
          template<typename Node>
          CODI_INLINE void handleConstant(Node const& node,
                                          DynamicStatementData& dynamicPointers,
                                          size_t& curPassiveArgument,
                                          size_t& idPos,
                                          size_t& constantPos) {
            CODI_UNUSED(curPassiveArgument, idPos);

            using AggregatedTraits = codi::RealTraits::AggregatedTypeTraits<typename Node::Real>;
            using ConversionOperator = typename Node::template ConversionOperator<PassiveReal>;

            typename Node::Real v = node.getValue();

            static_for<AggregatedTraits::Elements>([&](auto i) CODI_LAMBDA_INLINE {
              dynamicPointers.constantValues[constantPos] = ConversionOperator::toDataStore(AggregatedTraits::template arrayAccess<i.value>(v));
              constantPos += 1;
            });
          }
      };

    public:

      /// @{

      template<typename Aggregated, typename Type, typename Lhs, typename Rhs>
      CODI_INLINE void store(AggregatedExpressionType<Aggregated, Type, Lhs>& lhs,
                             ExpressionInterface<Aggregated, Rhs> const& rhs) {

        using Expr = AssignExpression<Lhs, Rhs>;
        using AggregatedTraits = RealTraits::AggregatedTypeTraits<Aggregated>;
        int constexpr Elements = AggregatedTraits::Elements;

        bool primalsStored = false;
        if (CODI_ENABLE_CHECK(Config::CheckTapeActivity, cast().isActive())) {
          CountActiveArguments countActiveArguments;
          PushStatementData pushStatement;

          codiAssert(ExpressionTraits::NumberOfActiveTypeArguments<Rhs>::value < Config::MaxArgumentSize);
          codiAssert(ExpressionTraits::NumberOfConstantTypeArguments<Rhs>::value < Config::MaxArgumentSize);
          codiAssert(ExpressionTraits::NumberOfActiveTypeArguments<Lhs>::value < Config::MaxArgumentSize);

          size_t activeArguments = 0;
          countActiveArguments.eval(rhs.cast(), activeArguments);

          if (CODI_ENABLE_CHECK(Config::CheckEmptyStatements, 0 != activeArguments)) {
            FixedSizeStatementData::reserve(fixedSizeData);
            DynamicStatementData::reserve(dynamicSizeData, StatementSizes::create<Expr>(), activeArguments);

            size_t passiveArguments = 0;
            size_t idPos = 0;
            size_t constantPos = 0;
            DynamicStatementData dynamicPointers = DynamicStatementData::store(dynamicSizeData, StatementSizes::create<Expr>(), activeArguments);
            pushStatement.eval(rhs.cast(), dynamicPointers, passiveArguments, idPos, constantPos);

            bool generatedNewIndex = false;
            static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
              generatedNewIndex |= indexManager.get().assignIndex(lhs.arrayValue[i.value].getIdentifier());
            });
            checkPrimalSize(generatedNewIndex);

            Aggregated real = rhs.cast().getValue();
            static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
              Identifier lhsIdentifier = lhs.arrayValue[i.value].getIdentifier();
              Real& primalEntry = primals[lhsIdentifier];

              if(!LinearIndexHandling) {
                dynamicPointers.lhsIdentifiers[i.value] = lhsIdentifier;
                dynamicPointers.oldPrimalValues[i.value] = primalEntry;
              }

              lhs.arrayValue[i.value].value() = AggregatedTraits::template arrayAccess<i.value>(real);
              primalEntry = lhs.arrayValue[i.value].getValue();
            });

            FixedSizeStatementData::store(fixedSizeData, (Config::ArgumentSize)passiveArguments, StatementEvaluator::template createHandle<Impl, Impl, Expr>());

            primalsStored = true;
          }
        }

        if(!primalsStored) {
          Aggregated real = rhs.cast().getValue();

          static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
            lhs.arrayValue[i.value].value() = AggregatedTraits::template arrayAccess<i.value>(real);
            indexManager.get().freeIndex(lhs.arrayValue[i.value].getIdentifier());
          });
        }
      }

      /// \copydoc codi::InternalStatementRecordingTapeInterface::store()
      template<typename Lhs, typename Rhs>
      CODI_INLINE void store(LhsExpressionInterface<Real, Gradient, Impl, Lhs>& lhs,
                             ExpressionInterface<Real, Rhs> const& rhs) {
        using Expr = AssignExpression<Lhs, Rhs>;

        if (CODI_ENABLE_CHECK(Config::CheckTapeActivity, cast().isActive())) {
          CountActiveArguments countActiveArguments;
          PushStatementData pushStatement;

          codiAssert(ExpressionTraits::NumberOfActiveTypeArguments<Rhs>::value < Config::MaxArgumentSize);
          codiAssert(ExpressionTraits::NumberOfConstantTypeArguments<Rhs>::value < Config::MaxArgumentSize);

          size_t activeArguments = 0;
          countActiveArguments.eval(rhs.cast(), activeArguments);

          if (CODI_ENABLE_CHECK(Config::CheckEmptyStatements, 0 != activeArguments)) {
            FixedSizeStatementData::reserve(fixedSizeData);
            DynamicStatementData::reserve(dynamicSizeData, StatementSizes::create<Expr>(), activeArguments);

            size_t passiveArguments = 0;
            size_t idPos = 0;
            size_t constantPos = 0;
            DynamicStatementData dynamicPointers = DynamicStatementData::store(dynamicSizeData, StatementSizes::create<Expr>(), activeArguments);
            pushStatement.eval(rhs.cast(), dynamicPointers, passiveArguments, idPos, constantPos);

            bool generatedNewIndex = indexManager.get().assignIndex(lhs.cast().getIdentifier());
            checkPrimalSize(generatedNewIndex);

            Real& primalEntry = primals[lhs.cast().getIdentifier()];
            if(!LinearIndexHandling) {
              dynamicPointers.lhsIdentifiers[0] = lhs.cast().getIdentifier();
              dynamicPointers.oldPrimalValues[0] = primalEntry;
            }

            FixedSizeStatementData::store(fixedSizeData, (Config::ArgumentSize)passiveArguments, StatementEvaluator::template createHandle<Impl, Impl, Expr>());

            primalEntry = rhs.cast().getValue();
          } else {
            indexManager.get().freeIndex(lhs.cast().getIdentifier());
          }
        } else {
          indexManager.get().freeIndex(lhs.cast().getIdentifier());
        }

        lhs.cast().value() = rhs.cast().getValue();
      }

      /// \copydoc codi::InternalStatementRecordingTapeInterface::store() <br>
      /// Optimization for copy statements.
      template<typename Lhs, typename Rhs>
      CODI_INLINE void store(LhsExpressionInterface<Real, Gradient, Impl, Lhs>& lhs,
                             LhsExpressionInterface<Real, Gradient, Impl, Rhs> const& rhs) {
        if (CODI_ENABLE_CHECK(Config::CheckTapeActivity, cast().isActive())) {
          if (IndexManager::CopyNeedsStatement || !Config::CopyOptimization) {
            store<Lhs, Rhs>(lhs, static_cast<ExpressionInterface<Real, Rhs> const&>(rhs));
          } else {
            indexManager.get().copyIndex(lhs.cast().getIdentifier(), rhs.cast().getIdentifier());
          }
        } else {
          indexManager.get().freeIndex(lhs.cast().getIdentifier());
        }

        lhs.cast().value() = rhs.cast().getValue();
      }

      /// \copydoc codi::InternalStatementRecordingTapeInterface::store() <br>
      /// Specialization for passive assignments.
      template<typename Lhs>
      CODI_INLINE void store(LhsExpressionInterface<Real, Gradient, Impl, Lhs>& lhs, PassiveReal const& rhs) {
        indexManager.get().freeIndex(lhs.cast().getIdentifier());

        lhs.cast().value() = rhs;
      }

      /// @}
      /*******************************************************************************
       * Protected helper function for ReverseTapeInterface
       */

    protected:

      /// Add a new input to the tape and update the primal value vector.
      template<typename Lhs>
      CODI_INLINE Real internalRegisterInput(LhsExpressionInterface<Real, Gradient, Impl, Lhs>& value,
                                             bool unusedIndex) {


        if (TapeTypes::IsLinearIndexHandler) {
          FixedSizeStatementData::reserve(fixedSizeData);
        }

        bool generatedNewIndex;
        if (unusedIndex) {
          generatedNewIndex = indexManager.get().assignUnusedIndex(value.cast().getIdentifier());
        } else {
          generatedNewIndex = indexManager.get().assignIndex(value.cast().getIdentifier());
        }
        checkPrimalSize(generatedNewIndex);

        Real& primalEntry = primals[value.cast().getIdentifier()];

        if (TapeTypes::IsLinearIndexHandler) {
          FixedSizeStatementData::store(fixedSizeData, Config::StatementInputTag, StatementEvaluator::template createHandle<Impl, Impl, AssignExpression<Lhs, Lhs>>());
        }

        Real oldValue = primalEntry;
        primalEntry = value.cast().value();

        return oldValue;
      }

    public:

      /// @name Functions from ReverseTapeInterface
      /// @{

      /// \copydoc codi::ReverseTapeInterface::registerInput()
      template<typename Lhs>
      CODI_INLINE void registerInput(LhsExpressionInterface<Real, Gradient, Impl, Lhs>& value) {
        internalRegisterInput(value, true);
      }

      /// \copydoc codi::ReverseTapeInterface::clearAdjoints()
      CODI_INLINE void clearAdjoints() {
        for (Gradient& gradient : adjoints) {
          gradient = Gradient();
        }
      }

      /// \copydoc codi::ReverseTapeInterface::reset()
      CODI_INLINE void reset(bool resetAdjoints = true) {
        for (Real& primal : primals) {
          primal = Real();
        }

        Base::reset(resetAdjoints);
      }

      /// @}

    protected:

      /// Adds data from all streams, the size of the adjoint vector, the size of the primal vector, and index manager
      /// data.
      CODI_INLINE TapeValues internalGetTapeValues() const {
        std::string name;
        if (TapeTypes::IsLinearIndexHandler) {
          name = "CoDi Tape Statistics ( PrimalValueLinearTape )";
        } else {
          name = "CoDi Tape Statistics ( PrimalValueReuseTape )";
        }
        TapeValues values = TapeValues(name);

        size_t nAdjoints = indexManager.get().getLargestCreatedIndex();
        double memoryAdjoints = static_cast<double>(nAdjoints) * static_cast<double>(sizeof(Gradient));

        size_t nPrimals = indexManager.get().getLargestCreatedIndex();
        double memoryPrimals = static_cast<double>(nPrimals) * static_cast<double>(sizeof(Real));

        values.addSection("Adjoint vector");
        values.addUnsignedLongEntry("Number of adjoints", nAdjoints);
        values.addDoubleEntry("Memory allocated", memoryAdjoints, true, true);

        values.addSection("Primal vector");
        values.addUnsignedLongEntry("Number of primals", nPrimals);
        values.addDoubleEntry("Memory allocated", memoryPrimals, true, true);

        values.addSection("Index manager");
        indexManager.get().addToTapeValues(values);

        values.addSection("Fixed size entries");
        fixedSizeData.addToTapeValues(values);
        values.addSection("Dynamic size entries");
        dynamicSizeData.addToTapeValues(values);

        return values;
      }

      /******************************************************************************
       * Protected helper function for CustomAdjointVectorEvaluationTapeInterface
       */

      /// Select the configured adjoint vector, see codi::Config::VariableAdjointInterfaceInPrimalTapes.
      template<typename Adjoint>
      ADJOINT_VECTOR_TYPE* selectAdjointVector(VectorAccess<Adjoint>* vectorAccess, Adjoint* data) {
        CODI_UNUSED(vectorAccess, data);

#if CODI_VariableAdjointInterfaceInPrimalTapes
        return vectorAccess;
#else
        static_assert(std::is_same<Adjoint, Gradient>::value,
                      "Please enable 'CODI_VariableAdjointInterfaceInPrimalTapes' in order"
                      " to use custom adjoint vectors in the primal value tapes.");

        return data;
#endif
      }

      /// Additional wrapper that triggers compiler optimizations.
      CODI_WRAP_FUNCTION(Wrap_internalEvaluateForward_Step3_EvalStatements,
                         Impl::internalEvaluateForward_Step3_EvalStatements);

      /// Forward evaluation of an inner tape part between two external functions.
      CODI_INLINE static void internalEvaluateForward_Step2_DataExtraction(NestedPosition const& start,
                                                                           NestedPosition const& end, Real* primalData,
                                                                           ADJOINT_VECTOR_TYPE* data,
                                                                           DynamicSizeData& dynamicSizeData) {
        Wrap_internalEvaluateForward_Step3_EvalStatements evalFunc{};
        dynamicSizeData.evaluateForward(start, end, evalFunc, primalData, data);
      }

      /// Internal method for the forward evaluation of the whole tape.
      template<bool copyPrimal, typename Adjoint>
      CODI_NO_INLINE void internalEvaluateForward(Position const& start, Position const& end, Adjoint* data) {
        std::vector<Real> primalsCopy(0);
        Real* primalData = primals.data();

        if (copyPrimal) {
          primalsCopy = primals;
          primalData = primalsCopy.data();
        }

        VectorAccess<Adjoint> vectorAccess(data, primalData);

        ADJOINT_VECTOR_TYPE* dataVector = selectAdjointVector(&vectorAccess, data);

        Base::internalEvaluateForward_Step1_ExtFunc(start, end, internalEvaluateForward_Step2_DataExtraction,
                                                    &vectorAccess, primalData, dataVector, dynamicSizeData);
      }

      /// Additional wrapper that triggers compiler optimizations.
      CODI_WRAP_FUNCTION(Wrap_internalEvaluateReverse_Step3_EvalStatements,
                         Impl::internalEvaluateReverse_Step3_EvalStatements);

      /// Reverse evaluation of an inner tape part between two external functions.
      CODI_INLINE static void internalEvaluateReverse_Step2_DataExtraction(NestedPosition const& start,
                                                                           NestedPosition const& end, Real* primalData,
                                                                           ADJOINT_VECTOR_TYPE* data,
                                                                           DynamicSizeData& dynamicSizeData) {
        Wrap_internalEvaluateReverse_Step3_EvalStatements evalFunc;
        dynamicSizeData.evaluateReverse(start, end, evalFunc, primalData, data);
      }

      /// Internal method for the reverse evaluation of the whole tape.
      template<bool copyPrimal, typename Adjoint>
      CODI_INLINE void internalEvaluateReverse(Position const& start, Position const& end, Adjoint* data) {
        Real* primalData = primals.data();

        if (copyPrimal) {
          primalsCopy = primals;
          primalData = primalsCopy.data();
        }

        VectorAccess<Adjoint> vectorAccess(data, primalData);

        ADJOINT_VECTOR_TYPE* dataVector = selectAdjointVector(&vectorAccess, data);

        Base::internalEvaluateReverse_Step1_ExtFunc(start, end, internalEvaluateReverse_Step2_DataExtraction,
                                                    &vectorAccess, primalData, dataVector, dynamicSizeData);
      }

    public:

      /// @name Functions from CustomAdjointVectorEvaluationTapeInterface
      /// @{

      using Base::evaluate;

      /// \copydoc codi::CustomAdjointVectorEvaluationTapeInterface::evaluate()
      template<typename Adjoint>
      CODI_INLINE void evaluate(Position const& start, Position const& end, Adjoint* data) {
        internalEvaluateReverse<!TapeTypes::IsLinearIndexHandler>(start, end, data);
      }

      /// \copydoc codi::CustomAdjointVectorEvaluationTapeInterface::evaluateForward()
      template<typename Adjoint>
      CODI_INLINE void evaluateForward(Position const& start, Position const& end, Adjoint* data) {
        internalEvaluateForward<!TapeTypes::IsLinearIndexHandler>(start, end, data);
      }

      /// @}
      /*******************************************************************************/
      /// @name Functions from DataManagementTapeInterface
      /// @{

      /// \copydoc codi::DataManagementTapeInterface::swap()
      CODI_INLINE void swap(Impl& other) {
        // Index manager does not need to be swapped, it is either static or swapped with the vector data.
        // Vectors are swapped recursively in the base class.

        std::swap(adjoints, other.adjoints);
        std::swap(primals, other.primals);

        Base::swap(other);

        // Ensure that the primals vector of both tapes are sized according to the index manager.
        checkPrimalSize(true);
        other.checkPrimalSize(true);
      }

      /// \copydoc codi::DataManagementTapeInterface::deleteAdjointVector()
      void deleteAdjointVector() {
        adjoints.resize(1);
      }

      /// \copydoc codi::DataManagementTapeInterface::getParameter()
      size_t getParameter(TapeParameters parameter) const {
        switch (parameter) {
          case TapeParameters::AdjointSize:
            return adjoints.size();
            break;
          case TapeParameters::DynamicSizeDataSize:
            return dynamicSizeData.getDataSize();
            break;
          case TapeParameters::LargestIdentifier:
            return indexManager.get().getLargestCreatedIndex();
            break;
            break;
          case TapeParameters::PrimalSize:
            return primals.size();
            break;
          case TapeParameters::FixedSizeDataSize:
            return fixedSizeData.getDataSize();
            break;
          default:
            return Base::getParameter(parameter);
            break;
        }
      }

      /// \copydoc codi::DataManagementTapeInterface::setParameter()
      void setParameter(TapeParameters parameter, size_t value) {
        switch (parameter) {
          case TapeParameters::AdjointSize:
            adjoints.resize(value);
            break;
          case TapeParameters::DynamicSizeDataSize:
            dynamicSizeData.resize(value);
            break;
          case TapeParameters::LargestIdentifier:
            CODI_EXCEPTION("Tried to set a get only option.");
            break;
          case TapeParameters::PrimalSize:
            primals.resize(value);
            break;
          case TapeParameters::FixedSizeDataSize:
            fixedSizeData.resize(value);
            break;
          default:
            Base::setParameter(parameter, value);
            break;
        }
      }

      /// \copydoc codi::DataManagementTapeInterface::createVectorAccess()
      VectorAccess<Gradient>* createVectorAccess() {
        return createVectorAccessCustomAdjoints(adjoints.data());
      }

      /// \copydoc codi::DataManagementTapeInterface::createVectorAccessCustomAdjoints()
      template<typename Adjoint>
      VectorAccess<Adjoint>* createVectorAccessCustomAdjoints(Adjoint* data) {
        return new VectorAccess<Adjoint>(data, primals.data());
      }

      /// \copydoc codi::DataManagementTapeInterface::deleteVectorAccess()
      void deleteVectorAccess(VectorAccessInterface<Real, Identifier>* access) {
        delete access;
      }

      /// @}
      /*******************************************************************************/
      /// @name Functions from ExternalFunctionTapeInterface
      /// @{

      /// \copydoc codi::ExternalFunctionTapeInterface::registerExternalFunctionOutput()
      template<typename Lhs>
      Real registerExternalFunctionOutput(LhsExpressionInterface<Real, Gradient, Impl, Lhs>& value) {
        return internalRegisterInput(value, true);
      }

      /// @}
      /*******************************************************************************/
      /// @name Functions from ForwardEvaluationTapeInterface
      /// @{

      using Base::evaluateForward;

      /// \copydoc codi::ForwardEvaluationTapeInterface::evaluateForward()
      void evaluateForward(Position const& start, Position const& end) {
        checkAdjointSize(indexManager.get().getLargestCreatedIndex());

        cast().evaluateForward(start, end, adjoints.data());
      }

      /// @}
      /******************************************************************************
       * Protected helper function for ManualStatementPushTapeInterface
       */

    protected:

    public:

      /*******************************************************************************/
      /// @name Functions from ManualStatementPushTapeInterface
      /// @{

      /// \copydoc codi::ManualStatementPushTapeInterface::pushJacobiManual()
      void pushJacobiManual(Real const& jacobian, Real const& value, Identifier const& index) {
        CODI_UNUSED(value);

        manualStatementData.jacobians[manualStatementData.pos] = jacobian;
        manualStatementData.identifiers[manualStatementData.pos] = index;
        manualStatementData.pos += 1;
      }

      /// \copydoc codi::ManualStatementPushTapeInterface::storeManual()
      void storeManual(Real const& lhsValue, Identifier& lhsIndex, Config::ArgumentSize const& size) {
        size_t preAccSize = (size) * sizeof(Real)
            + (size) * sizeof(Identifier);
        size_t dynamicSize = preAccSize;
        if(!LinearIndexHandling) {
          dynamicSize += sizeof(Identifier) + sizeof(Real);
        }

        FixedSizeStatementData::reserve(fixedSizeData);
        dynamicSizeData.reserveItems(dynamicSize);

        char* dynamicPointer;
        dynamicSizeData.getDataPointer(dynamicPointer);
        dynamicSizeData.addDataSize(preAccSize);

        manualStatementData.pos = 0;
        manualStatementData.jacobians = (Real*)dynamicPointer;
        manualStatementData.identifiers = (Identifier*)(&dynamicPointer[size * sizeof(Real)]);

        bool generatedNewIndex = indexManager.get().assignIndex(lhsIndex);
        checkPrimalSize(generatedNewIndex);

        Real& primalEntry = primals[lhsIndex];
        if(!LinearIndexHandling) {
          dynamicSizeData.pushData(lhsIndex);
          dynamicSizeData.pushData(primalEntry);
        }

        FixedSizeStatementData::store(fixedSizeData, (Config::ArgumentSize)size, PrimalValueBaseTape::jacobianExpressions[size]);

        primalEntry = lhsValue;
      }

      /// @}
      /*******************************************************************************/
      /// @name Functions from PositionalEvaluationTapeInterface
      /// @{

      /// \copydoc codi::PositionalEvaluationTapeInterface::evaluate()
      CODI_INLINE void evaluate(Position const& start, Position const& end) {
        checkAdjointSize(indexManager.get().getLargestCreatedIndex());

        evaluate(start, end, adjoints.data());
      }

      /// \copydoc codi::PositionalEvaluationTapeInterface::resetTo()
      CODI_INLINE void resetTo(Position const& pos) {
        cast().internalResetPrimalValues(pos);

        Base::resetTo(pos);
      }

      /// @}
      /*******************************************************************************/
      /// @name Functions from PreaccumulationEvaluationTapeInterface
      /// @{

      /// \copydoc codi::PreaccumulationEvaluationTapeInterface::evaluateKeepState()
      void evaluateKeepState(Position const& start, Position const& end) {
        checkAdjointSize(indexManager.get().getLargestCreatedIndex());

        internalEvaluateReverse<false>(start, end, adjoints.data());

        if (!TapeTypes::IsLinearIndexHandler) {
          evaluatePrimal(end, start);
        }
      }

      /// \copydoc codi::PreaccumulationEvaluationTapeInterface::evaluateForwardKeepState()
      void evaluateForwardKeepState(Position const& start, Position const& end) {
        checkAdjointSize(indexManager.get().getLargestCreatedIndex());

        if (!TapeTypes::IsLinearIndexHandler) {
          cast().internalResetPrimalValues(end);
        }

        internalEvaluateForward<false>(start, end, adjoints.data());
      }

    protected:

      /******************************************************************************
       * Protected helper function for PrimalEvaluationTapeInterface
       */

      /// Wrapper helper for improved compiler optimizations.
      CODI_WRAP_FUNCTION(Wrap_internalEvaluatePrimal_Step3_EvalStatements,
                         Impl::internalEvaluatePrimal_Step3_EvalStatements);

      /// Start for primal evaluation between external function.
      CODI_INLINE static void internalEvaluatePrimal_Step2_DataExtraction(NestedPosition const& start,
                                                                          NestedPosition const& end, Real* primalData,
                                                                          DynamicSizeData& dynamicSizeData) {
        Wrap_internalEvaluatePrimal_Step3_EvalStatements evalFunc{};
        dynamicSizeData.evaluateForward(start, end, evalFunc, primalData);
      }

    public:

      /// @}
      /*******************************************************************************/
      /// @name Functions from PrimalEvaluationTapeInterface
      /// @{

      using Base::evaluatePrimal;

      /// \copydoc codi::PrimalEvaluationTapeInterface::evaluatePrimal()
      CODI_NO_INLINE void evaluatePrimal(Position const& start, Position const& end) {
        // TODO: implement primal value only accessor
        PrimalAdjointVectorAccess<Real, Identifier, Gradient> primalAdjointAccess(adjoints.data(), primals.data());

        Base::internalEvaluatePrimal_Step1_ExtFunc(start, end,
                                                   PrimalValueBaseTape::internalEvaluatePrimal_Step2_DataExtraction,
                                                   &primalAdjointAccess, primals.data(), dynamicSizeData);
      }

      /// \copydoc codi::PrimalEvaluationTapeInterface::primal(Identifier const&)
      Real& primal(Identifier const& identifier) {
        return primals[identifier];
      }

      /// \copydoc codi::PrimalEvaluationTapeInterface::primal(Identifier const&) const
      Real const& primal(Identifier const& identifier) const {
        return primals[identifier];
      }

      /// @}
      /*******************************************************************************/
      /// @name Function from StatementEvaluatorTapeInterface and StatementEvaluatorInnerTapeInterface
      /// @{

      /// Base class for statement call generators. Prepares the static construction context.
      template<typename Expr>
      struct StatementCallGenBase {
        public:
          using Lhs = typename Expr::Lhs;  ///< \copydoc codi::AssignExpression::Lhs
          using Rhs = typename Expr::Rhs;  ///< \copydoc codi::AssignExpression::Rhs
          using LhsReal = typename Lhs::Real;  ///< Primal value of lhs.
          using AggregateTraits = RealTraits::AggregatedTypeTraits<LhsReal>;  ///< Traits for aggregated type.

          using Constructor = ConstructStaticContextLogic<Rhs, Impl, 0, 0>;  ///< Static construction context.
          using StaticRhs = typename Constructor::ResultType;                ///< Static right hand side.

          template<size_t pos>
          using ExtractExpr = ExtractExpression<LhsReal, pos, StaticRhs>; ///< Extract expressions for aggregated types.

          /// Construct the statement in the static context.
          static StaticRhs constructStatic(Real* __restrict__ primalVector,
                                           PassiveReal const* const __restrict__ constantValues,
                                           Identifier const* const __restrict__ identifiers) {
            return Constructor::construct(primalVector, identifiers, constantValues);
          }
      };

      /// \copydoc StatementEvaluatorTapeInterface::StatementCallGen
      template<StatementCall type, typename Expr>
      struct StatementCallGen;

      /*******************************************************************************/
      /// ClearAdjoint implementation
      template<typename Expr>
      struct StatementCallGen<StatementCall::ClearAdjoint, Expr> {
          /// \copydoc codi::StatementEvaluatorInnerTapeInterface::StatementCallGen::evaluateInner()
          CODI_INLINE static void evaluateInner() {
            // Empty
          }

          /// \copydoc codi::StatementEvaluatorInnerTapeInterface::StatementCallGen::evaluateFull()
          template<typename Func>
          CODI_INLINE static void evaluateFull(
              Func const& evalInner, StatementSizes stmtSizes, STMT_ARGS,
              ADJOINT_VECTOR_TYPE* __restrict__ adjointVector, size_t adjointVectorSize
          ) {
            CODI_UNUSED(evalInner);

            StatementEvalArguments stmtArgs = STMT_ARGS_PACK;
            DynamicStatementData data = DynamicStatementData::readReverse(stmtSizes, stmtArgs);

            stmtArgs.updateAdjointPosReverse(stmtSizes.maxOutputArgs);

            for (size_t iLhs = 0; iLhs < stmtSizes.maxOutputArgs; iLhs += 1) {
              Identifier lhsIdentifier = stmtArgs.getLhsIdentifier(iLhs, data.lhsIdentifiers);

              if(lhsIdentifier < (Identifier)adjointVectorSize) {
#if CODI_VariableAdjointInterfaceInPrimalTapes
                adjointVector->resetAdjointVec(lhsIdentifier);
#else
                adjointVector[lhsIdentifier] = Gradient();
#endif
              }
            }
          }

          /// \copydoc codi::StatementEvaluatorTapeInterface::StatementCallGen::evaluate()
          CODI_INLINE static void evaluate(STMT_ARGS,
                                           ADJOINT_VECTOR_TYPE* __restrict__ adjointVector, size_t adjointVectorSize) {
            evaluateFull(evaluateInner, StatementSizes::create<Expr>(), STMT_ARGS_FORWARD,
                                      adjointVector, adjointVectorSize);
          }
      };

      /*******************************************************************************/
      /// Forward implementation
      template<typename Expr>
      struct StatementCallGen<StatementCall::Forward, Expr> : public StatementCallGenBase<Expr> {

          using Base = StatementCallGenBase<Expr>;
          using StaticRhs = typename Base::StaticRhs;
          template<size_t pos>
          using ExtractExpr = typename Base::template ExtractExpr<pos>;

          /// Perform the adjoint update based on the configuration in codi::Config::VariableAdjointInterfaceInPrimalTapes.
          struct IncrementForwardLogic : public JacobianComputationLogic<IncrementForwardLogic> {
            public:

              /// \copydoc codi::JacobianComputationLogic::handleJacobianOnActive()
              template<typename Node, typename Jacobian>
              CODI_INLINE void handleJacobianOnActive(Node const& node, Jacobian jacobianExpr, Gradient& lhsTangent,
                                                      ADJOINT_VECTOR_TYPE* adjointVector) {
                CODI_UNUSED(lhsTangent);

                Real jacobian = jacobianExpr;

                if (CODI_ENABLE_CHECK(Config::IgnoreInvalidJacobians, RealTraits::isTotalFinite(jacobian))) {
#if CODI_VariableAdjointInterfaceInPrimalTapes
                  adjointVector->updateTangentWithLhs(node.getIdentifier(), jacobian);
#else
                  lhsTangent += jacobian * adjointVector[node.getIdentifier()];
#endif
                }
              }
          };

          /// \copydoc codi::StatementEvaluatorInnerTapeInterface::StatementCallGen::evaluateInner()
          CODI_INLINE static void evaluateInner(Real* __restrict__ primalVector, ADJOINT_VECTOR_TYPE* adjointVector,
                                                    Real* __restrict__ lhsPrimals,
                                                    Gradient* __restrict__ lhsTangents,
                                                    PassiveReal const* const __restrict__ constantValues,
                                                    Identifier const* const __restrict__ identifiers) {
            StaticRhs staticsRhs = Base::constructStatic(primalVector, constantValues, identifiers);

            IncrementForwardLogic incrementForward;

            static_for<Base::AggregateTraits::Elements>([&](auto i) CODI_LAMBDA_INLINE {
#if CODI_VariableAdjointInterfaceInPrimalTapes
              adjointVector->setActiveViariableForIndirectAccess(i.value);
#endif
              ExtractExpr<i.value> expr(staticsRhs);

              incrementForward.eval(expr, Real(1.0), lhsTangents[i.value], adjointVector);
              lhsPrimals[i.value] = expr.getValue();
            });
          }

          /// \copydoc codi::StatementEvaluatorInnerTapeInterface::StatementCallGen::evaluateFull()
          template<typename Func>
          CODI_INLINE static void evaluateFull(Func const& evalInner, StatementSizes stmtSizes, STMT_ARGS,
                                   Real* __restrict__ primalVector,
                                   ADJOINT_VECTOR_TYPE* __restrict__ adjointVector,
                                   Real* __restrict__ lhsPrimals,
                                   Gradient* __restrict__ lhsTangents) {
            StatementEvalArguments stmtArgs = STMT_ARGS_PACK;
            DynamicStatementData data = DynamicStatementData::readForward(stmtSizes, stmtArgs);

            if(!TapeTypes::IsLinearIndexHandler) {
              data.updateOldPrimalValues(stmtSizes, stmtArgs, primalVector);
            }
            data.copyPassiveValuesIntoPrimalVector(stmtArgs, primalVector);

    #if CODI_VariableAdjointInterfaceInPrimalTapes
            adjointVector->setSizeForIndirectAccess(stmtSizes.maxOutputArgs);
    #endif

            evalInner(primalVector, adjointVector, lhsPrimals, lhsTangents, data.constantValues, data.rhsIdentifiers);

            for(size_t iLhs = 0; iLhs < stmtSizes.maxOutputArgs; iLhs += 1) {
              Identifier const lhsIdentifier = stmtArgs.getLhsIdentifier(iLhs, data.lhsIdentifiers);

              primalVector[lhsIdentifier] = lhsPrimals[iLhs];
    #if CODI_VariableAdjointInterfaceInPrimalTapes
              adjointVector->setActiveViariableForIndirectAccess(iLhs);
              adjointVector->setLhsTangent(lhsIdentifier);
    #else
              adjointVector[lhsIdentifier] = lhsTangents[iLhs];
              lhsTangents[iLhs] = Gradient();
    #endif
            }

            stmtArgs.updateAdjointPosForward(stmtSizes.maxOutputArgs);
          }

          /// \copydoc codi::StatementEvaluatorTapeInterface::StatementCallGen::evaluate()
          CODI_INLINE static void evaluate(STMT_ARGS, Real* __restrict__ primalVector,
                                           ADJOINT_VECTOR_TYPE* __restrict__ adjointVector,
                                           Real* __restrict__ lhsPrimals,
                                           Gradient* __restrict__ lhsTangents) {
            evaluateFull(evaluateInner, StatementSizes::create<Expr>(), STMT_ARGS_FORWARD, primalVector, adjointVector,
                         lhsPrimals, lhsTangents);
          }
      };

      /*******************************************************************************/
      /// Primal implementation
      template<typename Expr>
      struct StatementCallGen<StatementCall::Primal, Expr> : public StatementCallGenBase<Expr> {
        public:
          using Base = StatementCallGenBase<Expr>;
          using StaticRhs = typename Base::StaticRhs;
          template<size_t pos>
          using ExtractExpr = typename Base::template ExtractExpr<pos>;

          /// \copydoc codi::StatementEvaluatorInnerTapeInterface::StatementCallGen::evaluateInner()
          static void evaluateInner(Real* __restrict__ primalVector,
                                                   Real* __restrict__ lhsPrimals,
                                                   PassiveReal const* const __restrict__ constantValues,
                                                   Identifier const* const __restrict__ identifiers) {
            StaticRhs staticsRhs = Base::constructStatic(primalVector, constantValues, identifiers);

            static_for<Base::AggregateTraits::Elements>([&](auto i) CODI_LAMBDA_INLINE {
              ExtractExpr<i.value> expr(staticsRhs);
              lhsPrimals[i.value] = expr.getValue();
            });
          }

          /// \copydoc codi::StatementEvaluatorInnerTapeInterface::StatementCallGen::evaluateFull()
          template<typename Func>
          static void evaluateFull(Func const& evalInner, StatementSizes stmtSizes, STMT_ARGS,
                                   Real* __restrict__ primalVector,
                                   Real* __restrict__ lhsPrimals) {
            StatementEvalArguments stmtArgs = STMT_ARGS_PACK;
            DynamicStatementData data = DynamicStatementData::readForward(stmtSizes, stmtArgs);

            if(!TapeTypes::IsLinearIndexHandler) {
              data.updateOldPrimalValues(stmtSizes, stmtArgs, primalVector);
            }
            data.copyPassiveValuesIntoPrimalVector(stmtArgs, primalVector);

            evalInner(primalVector, lhsPrimals, data.constantValues, data.rhsIdentifiers);

            for(size_t iLhs = 0; iLhs < stmtSizes.maxOutputArgs; iLhs += 1) {
              Identifier const lhsIdentifier = stmtArgs.getLhsIdentifier(iLhs, data.lhsIdentifiers);

              primalVector[lhsIdentifier] = lhsPrimals[iLhs];
            }

            stmtArgs.updateAdjointPosForward(stmtSizes.maxOutputArgs);
          }

          /// \copydoc codi::StatementEvaluatorTapeInterface::StatementCallGen::evaluate()
          CODI_INLINE static void evaluate(STMT_ARGS,
                                           Real* __restrict__ primalVector,
                                           Real* __restrict__ lhsPrimals) {
            evaluateFull(evaluateInner, StatementSizes::create<Expr>(), STMT_ARGS_FORWARD, primalVector, lhsPrimals);
          }
      };

      /*******************************************************************************/
      /// ResetPrimal implementation
      template<typename Expr>
      struct StatementCallGen<StatementCall::ResetPrimal, Expr> {
          /// \copydoc codi::StatementEvaluatorInnerTapeInterface::StatementCallGen::evaluateInner()
          CODI_INLINE static void evaluateInner() {
            // Empty
          }

          /// \copydoc codi::StatementEvaluatorInnerTapeInterface::StatementCallGen::evaluateFull()
          template<typename Func>
          CODI_INLINE static void evaluateFull(
              Func const& evalInner, StatementSizes stmtSizes, STMT_ARGS,
              Real* __restrict__ primalVector
          ) {
            CODI_UNUSED(evalInner);

            StatementEvalArguments stmtArgs = STMT_ARGS_PACK;
            DynamicStatementData data = DynamicStatementData::readReverse(stmtSizes, stmtArgs);
            stmtArgs.updateAdjointPosReverse(stmtSizes.maxOutputArgs);

            for (size_t iLhs = 0; iLhs < stmtSizes.maxOutputArgs; iLhs += 1) {
              Identifier lhsIdentifier = stmtArgs.getLhsIdentifier(iLhs, data.lhsIdentifiers);

              if(!LinearIndexHandling) {
                primalVector[lhsIdentifier] = data.oldPrimalValues[iLhs];
              }
            }
          }

          /// \copydoc codi::StatementEvaluatorTapeInterface::StatementCallGen::evaluate()
          CODI_INLINE static void evaluate(STMT_ARGS, Real* __restrict__ primalVector) {
            evaluateFull(evaluateInner, StatementSizes::create<Expr>(), STMT_ARGS_FORWARD, primalVector);
          }
      };

      /*******************************************************************************/
      /// Reverse implementation
      template<typename Expr>
      struct StatementCallGen<StatementCall::Reverse, Expr> : public StatementCallGenBase<Expr> {
        public:
          using Base = StatementCallGenBase<Expr>;
          using StaticRhs = typename Base::StaticRhs;
          template<size_t pos>
          using ExtractExpr = typename Base::template ExtractExpr<pos>;

          /// Perform the adjoint update based on the configuration in codi::Config::VariableAdjointInterfaceInPrimalTapes.
          struct IncrementReversalLogic : public JacobianComputationLogic<IncrementReversalLogic> {
            public:

              /// See IncrementReversalLogic.
              template<typename Node, typename Jacobian>
              CODI_INLINE void handleJacobianOnActive(Node const& node, Jacobian jacobianExpr, Gradient const& lhsAdjoint,
                                                      ADJOINT_VECTOR_TYPE* adjointVector) {
                CODI_UNUSED(lhsAdjoint);

                Real jacobian = jacobianExpr;

                if (CODI_ENABLE_CHECK(Config::IgnoreInvalidJacobians, RealTraits::isTotalFinite(jacobian))) {
#if CODI_VariableAdjointInterfaceInPrimalTapes
                  adjointVector->updateAdjointWithLhs(node.getIdentifier(), jacobian);
#else
                  adjointVector[node.getIdentifier()] += jacobian * lhsAdjoint;
#endif
                }
              }
          };

          /// \copydoc codi::StatementEvaluatorInnerTapeInterface::StatementCallGen::evaluateInner()
          CODI_INLINE static void evaluateInner(Real* __restrict__ primalVector, ADJOINT_VECTOR_TYPE* __restrict__ adjointVector,
                                                Gradient* __restrict__ lhsAdjoints,
                                                PassiveReal const* const __restrict__ constantValues,
                                                Identifier const* const __restrict__ rhsIdentifiers) {
            StaticRhs staticsRhs = Base::constructStatic(primalVector, constantValues, rhsIdentifiers);

            IncrementReversalLogic incrementReverse;
            static_for<Base::AggregateTraits::Elements>([&](auto i) CODI_LAMBDA_INLINE {
#if CODI_VariableAdjointInterfaceInPrimalTapes
              adjointVector->setActiveViariableForIndirectAccess(i.value);
#endif
              ExtractExpr<i.value> expr(staticsRhs);
              incrementReverse.eval(expr, Real(1.0), const_cast<Gradient const&>(lhsAdjoints[i.value]), adjointVector);
            });
          }

          /// \copydoc codi::StatementEvaluatorInnerTapeInterface::StatementCallGen::evaluateFull()
          template<typename Func>
          CODI_INLINE static void evaluateFull(
              Func const& evalInner, StatementSizes stmtSizes, STMT_ARGS,
              Real* __restrict__ primalVector, ADJOINT_VECTOR_TYPE* __restrict__ adjointVector,
              Gradient* __restrict__ lhsAdjoints
          ) {

            StatementEvalArguments stmtArgs = STMT_ARGS_PACK;
            DynamicStatementData data = DynamicStatementData::readReverse(stmtSizes, stmtArgs);

            stmtArgs.updateAdjointPosReverse(stmtSizes.maxOutputArgs);

    #if CODI_VariableAdjointInterfaceInPrimalTapes
            adjointVector->setSizeForIndirectAccess(maxOutputArgs);
    #endif

            bool allZero = true;
            for (size_t iLhs = 0; iLhs < stmtSizes.maxOutputArgs; iLhs += 1) {
              Identifier lhsIdentifier = stmtArgs.getLhsIdentifier(iLhs, data.lhsIdentifiers);

              if (!LinearIndexHandling) {
                primalVector[lhsIdentifier] = data.oldPrimalValues[iLhs];
              }

    #if CODI_VariableAdjointInterfaceInPrimalTapes
              adjointVector->setActiveViariableForIndirectAccess(iLhs);
              adjointVector->setLhsAdjoint(lhsIdentifier);
              allZero &= adjointVector->isLhsZero();
    #else
              lhsAdjoints[iLhs] = adjointVector[lhsIdentifier];
              adjointVector[lhsIdentifier] = Gradient();
              allZero &= RealTraits::isTotalZero(lhsAdjoints[iLhs]);
    #endif
            }

            if (CODI_ENABLE_CHECK(Config::SkipZeroAdjointEvaluation, !allZero)) {
              data.copyPassiveValuesIntoPrimalVector(stmtArgs, primalVector);

              evalInner(primalVector, adjointVector, lhsAdjoints, data.constantValues, data.rhsIdentifiers);
            }
          }

          /// \copydoc codi::StatementEvaluatorTapeInterface::StatementCallGen::evaluate()
          CODI_INLINE static void evaluate(STMT_ARGS, Real* __restrict__ primalVector,
              ADJOINT_VECTOR_TYPE* __restrict__ adjointVector, Gradient* __restrict__ lhsAdjoints
          ) {
            evaluateFull(evaluateInner, StatementSizes::create<Expr>(), STMT_ARGS_FORWARD, primalVector, adjointVector,
                         lhsAdjoints);
          }
      };


    private:

      CODI_INLINE void checkAdjointSize(Identifier const& identifier) {
        if (identifier >= (Identifier)adjoints.size()) {
          resizeAdjointsVector();
        }
      }

      CODI_INLINE void checkPrimalSize(bool generatedNewIndex) {
        if (TapeTypes::IsLinearIndexHandler) {
          if (indexManager.get().getLargestCreatedIndex() >= (Identifier)primals.size()) {
            resizePrimalVector(primals.size() + Config::ChunkSize);
          }
        } else {
          if (generatedNewIndex) {
            resizePrimalVector(indexManager.get().getLargestCreatedIndex() + 1);
          }
        }
      }

      CODI_NO_INLINE void resizeAdjointsVector() {
        adjoints.resize(indexManager.get().getLargestCreatedIndex() + 1);
      }

      CODI_NO_INLINE void resizePrimalVector(size_t newSize) {
        primals.resize(newSize);
      }
  };

  /// Specialized for NumberOfActiveTypeArguments and NumberOfConstantTypeArguments.
  template<size_t size>
  struct JacobianExpression {};

  /// Specialization for manual statement pushes of the used expression type.
  template<size_t size>
  struct ExpressionTraits::NumberOfActiveTypeArguments<JacobianExpression<size>> {
      static size_t constexpr value = size;  ///< Number of arguments.
  };

  /// Specialization for manual statement pushes of the used expression type.
  template<size_t size>
  struct ExpressionTraits::NumberOfConstantTypeArguments<JacobianExpression<size>> {
      static size_t constexpr value = 0;  ///< Always zero.
  };

  /// Implements StatementEvaluatorTapeInterface and StatementEvaluatorInnerTapeInterface
  /// @tparam T_size  Number of arguments.
  template<typename T_Impl, size_t T_size>
  struct JacobianStatementGenerator :
      public StatementEvaluatorTapeInterface,
      public StatementEvaluatorInnerTapeInterface {
    public:

      using Impl = CODI_DD(T_Impl, CODI_T(PrimalValueBaseTape<
          PrimalValueTapeTypes<double, double, IndexManagerInterface<int>,
          StatementEvaluatorInterface, DefaultChunkedData>, void>));
      static size_t constexpr size = T_size;  ///< See JacobianStatementGenerator

      using Real = typename Impl::Real;
      using Gradient = typename Impl::Gradient;
      using Identifier = typename Impl::Identifier;
      using PassiveReal = typename Impl::PassiveReal;

      using DynamicStatementData = typename Impl::DynamicStatementData;
      using StmtPackHelper = typename Impl::StmtPackHelper;
      using StatementEvalArguments = typename Impl::StatementEvalArguments;

      static size_t constexpr LinearIndexHandling = Impl::LinearIndexHandling;

      /*******************************************************************************/
      /// @name Implementation of StatementEvaluatorTapeInterface and StatementEvaluatorInnerTapeInterface
      /// @{

      /// \copydoc StatementEvaluatorTapeInterface::StatementCallGen
      template<StatementCall type, typename Expr>
      struct StatementCallGen;

      template<typename Expr>
      struct StatementCallGen<StatementCall::ClearAdjoint, Expr> :
          public Impl::template StatementCallGen<StatementCall::ClearAdjoint, Expr> {};

      template<typename Expr>
      struct StatementCallGen<StatementCall::Forward, Expr> {
          /// Throws exception.
          static void evaluateInner() {
            CODI_EXCEPTION("Forward evaluation of jacobian statement not possible.");
          }

          /// Throws exception.
          static void evaluate() {
            CODI_EXCEPTION("Forward evaluation of jacobian statement not possible.");
          }
      };

      template<typename Expr>
      struct StatementCallGen<StatementCall::Primal, Expr> {
          /// Throws exception.
          static void evaluateInner() {
            CODI_EXCEPTION("Primal evaluation of jacobian statement not possible.");
          }

          /// Throws exception.
          static void evaluate() {
            CODI_EXCEPTION("Primal evaluation of jacobian statement not possible.");
          }
      };

      template<typename Expr>
      struct StatementCallGen<StatementCall::ResetPrimal, Expr> :
          public Impl::template StatementCallGen<StatementCall::ResetPrimal, Expr> {};

      template<typename Expr>
      struct StatementCallGen<StatementCall::Reverse, Expr> {
          /// \copydoc codi::StatementEvaluatorInnerTapeInterface::StatementCallGen::evaluateInner
          static void evaluateInner(
              Real* __restrict__ primalVector, ADJOINT_VECTOR_TYPE* __restrict__ adjointVector,
              Gradient* __restrict__ lhsAdjoints,
              PassiveReal const* const __restrict__ constantValues,
              Identifier const* const __restrict__ rhsIdentifiers
          ) {
            CODI_UNUSED(constantValues);

            evalJacobianReverse(adjointVector, lhsAdjoints[0], primalVector, rhsIdentifiers);
          }

          /// \copydoc codi::StatementEvaluatorTapeInterface::StatementCallGen::evaluate
          static void evaluate(
              STMT_ARGS, Real* __restrict__ primalVector,
              ADJOINT_VECTOR_TYPE* __restrict__ adjointVector, Gradient* __restrict__ lhsAdjoints) {

            StatementSizes stmtSizes = StatementSizes::create<Expr>();
            StatementEvalArguments stmtArgs = STMT_ARGS_PACK;
            DynamicStatementData data = DynamicStatementData::readReverse(stmtSizes, stmtArgs);

            stmtArgs.updateAdjointPosReverse(1);

#if CODI_VariableAdjointInterfaceInPrimalTapes
            adjointVector->setSizeForIndirectAccess(1);
#endif
            Identifier lhsIdentifier = stmtArgs.getLhsIdentifier(0, data.lhsIdentifiers);

            if (!LinearIndexHandling) {
              primalVector[lhsIdentifier] = data.oldPrimalValues[0];
            }

#if CODI_VariableAdjointInterfaceInPrimalTapes
            adjointVector->setActiveViariableForIndirectAccess(0);
            adjointVector->setLhsAdjoint(lhsIdentifier);
#else
            lhsAdjoints[0] = adjointVector[lhsIdentifier];
            adjointVector[lhsIdentifier] = Gradient();
#endif
            evalJacobianReverse(adjointVector, lhsAdjoints[0], data.passiveValues, data.rhsIdentifiers);
          }
      };

      /// @}

    private:

      static void evalJacobianReverse(ADJOINT_VECTOR_TYPE* adjointVector, Gradient lhsAdjoint,
                                      Real const* const passiveValues,
                                      Identifier const* const rhsIdentifiers) {
#if CODI_VariableAdjointInterfaceInPrimalTapes
        bool const lhsZero = adjointVector->isLhsZero();
#else
        bool const lhsZero = RealTraits::isTotalZero(lhsAdjoint);
#endif

        if (CODI_ENABLE_CHECK(Config::SkipZeroAdjointEvaluation, !lhsZero)) {
          for(size_t pos = 0; pos < size; pos += 1) {
            Real const& jacobian = passiveValues[pos];
#if CODI_VariableAdjointInterfaceInPrimalTapes
            adjointVector->updateAdjointWithLhs(rhsIdentifiers[pos], jacobian);
#else
            adjointVector[rhsIdentifiers[pos]] += jacobian * lhsAdjoint;
#endif
          }
        }
      }
  };


  template<typename TapeTypes, typename Impl>
  size_t PrimalValueBaseTape<TapeTypes, Impl>::ReuseStmtPackHelper::tempAdjointPos = 0;

#define CREATE_EXPRESSION(size)                                                                                        \
  TapeTypes::StatementEvaluator::template createHandle<Impl, JacobianStatementGenerator<Impl, size>, \
                                                       AssignExpression<ActiveType<Impl>, JacobianExpression<size>>>()

  /// Expressions for manual statement pushes.
  template<typename TapeTypes, typename Impl>
  const typename TapeTypes::EvalHandle
      PrimalValueBaseTape<TapeTypes, Impl>::jacobianExpressions[Config::MaxArgumentSize] = {
          CREATE_EXPRESSION(0),   CREATE_EXPRESSION(1),   CREATE_EXPRESSION(2),   CREATE_EXPRESSION(3),
          CREATE_EXPRESSION(4),   CREATE_EXPRESSION(5),   CREATE_EXPRESSION(6),   CREATE_EXPRESSION(7),
          CREATE_EXPRESSION(8),   CREATE_EXPRESSION(9),   CREATE_EXPRESSION(10),  CREATE_EXPRESSION(11),
          CREATE_EXPRESSION(12),  CREATE_EXPRESSION(13),  CREATE_EXPRESSION(14),  CREATE_EXPRESSION(15),
          CREATE_EXPRESSION(16),  CREATE_EXPRESSION(17),  CREATE_EXPRESSION(18),  CREATE_EXPRESSION(19),
          CREATE_EXPRESSION(20),  CREATE_EXPRESSION(21),  CREATE_EXPRESSION(22),  CREATE_EXPRESSION(23),
          CREATE_EXPRESSION(24),  CREATE_EXPRESSION(25),  CREATE_EXPRESSION(26),  CREATE_EXPRESSION(27),
          CREATE_EXPRESSION(28),  CREATE_EXPRESSION(29),  CREATE_EXPRESSION(30),  CREATE_EXPRESSION(31),
          CREATE_EXPRESSION(32),  CREATE_EXPRESSION(33),  CREATE_EXPRESSION(34),  CREATE_EXPRESSION(35),
          CREATE_EXPRESSION(36),  CREATE_EXPRESSION(37),  CREATE_EXPRESSION(38),  CREATE_EXPRESSION(39),
          CREATE_EXPRESSION(40),  CREATE_EXPRESSION(41),  CREATE_EXPRESSION(42),  CREATE_EXPRESSION(43),
          CREATE_EXPRESSION(44),  CREATE_EXPRESSION(45),  CREATE_EXPRESSION(46),  CREATE_EXPRESSION(47),
          CREATE_EXPRESSION(48),  CREATE_EXPRESSION(49),  CREATE_EXPRESSION(50),  CREATE_EXPRESSION(51),
          CREATE_EXPRESSION(52),  CREATE_EXPRESSION(53),  CREATE_EXPRESSION(54),  CREATE_EXPRESSION(55),
          CREATE_EXPRESSION(56),  CREATE_EXPRESSION(57),  CREATE_EXPRESSION(58),  CREATE_EXPRESSION(59),
          CREATE_EXPRESSION(60),  CREATE_EXPRESSION(61),  CREATE_EXPRESSION(62),  CREATE_EXPRESSION(63),
          CREATE_EXPRESSION(64),  CREATE_EXPRESSION(65),  CREATE_EXPRESSION(66),  CREATE_EXPRESSION(67),
          CREATE_EXPRESSION(68),  CREATE_EXPRESSION(69),  CREATE_EXPRESSION(70),  CREATE_EXPRESSION(71),
          CREATE_EXPRESSION(72),  CREATE_EXPRESSION(73),  CREATE_EXPRESSION(74),  CREATE_EXPRESSION(75),
          CREATE_EXPRESSION(76),  CREATE_EXPRESSION(77),  CREATE_EXPRESSION(78),  CREATE_EXPRESSION(79),
          CREATE_EXPRESSION(80),  CREATE_EXPRESSION(81),  CREATE_EXPRESSION(82),  CREATE_EXPRESSION(83),
          CREATE_EXPRESSION(84),  CREATE_EXPRESSION(85),  CREATE_EXPRESSION(86),  CREATE_EXPRESSION(87),
          CREATE_EXPRESSION(88),  CREATE_EXPRESSION(89),  CREATE_EXPRESSION(90),  CREATE_EXPRESSION(91),
          CREATE_EXPRESSION(92),  CREATE_EXPRESSION(93),  CREATE_EXPRESSION(94),  CREATE_EXPRESSION(95),
          CREATE_EXPRESSION(96),  CREATE_EXPRESSION(97),  CREATE_EXPRESSION(98),  CREATE_EXPRESSION(99),
          CREATE_EXPRESSION(100), CREATE_EXPRESSION(101), CREATE_EXPRESSION(102), CREATE_EXPRESSION(103),
          CREATE_EXPRESSION(104), CREATE_EXPRESSION(105), CREATE_EXPRESSION(106), CREATE_EXPRESSION(107),
          CREATE_EXPRESSION(108), CREATE_EXPRESSION(109), CREATE_EXPRESSION(110), CREATE_EXPRESSION(111),
          CREATE_EXPRESSION(112), CREATE_EXPRESSION(113), CREATE_EXPRESSION(114), CREATE_EXPRESSION(115),
          CREATE_EXPRESSION(116), CREATE_EXPRESSION(117), CREATE_EXPRESSION(118), CREATE_EXPRESSION(119),
          CREATE_EXPRESSION(120), CREATE_EXPRESSION(121), CREATE_EXPRESSION(122), CREATE_EXPRESSION(123),
          CREATE_EXPRESSION(124), CREATE_EXPRESSION(125), CREATE_EXPRESSION(126), CREATE_EXPRESSION(127),
          CREATE_EXPRESSION(128), CREATE_EXPRESSION(129), CREATE_EXPRESSION(130), CREATE_EXPRESSION(131),
          CREATE_EXPRESSION(132), CREATE_EXPRESSION(133), CREATE_EXPRESSION(134), CREATE_EXPRESSION(135),
          CREATE_EXPRESSION(136), CREATE_EXPRESSION(137), CREATE_EXPRESSION(138), CREATE_EXPRESSION(139),
          CREATE_EXPRESSION(140), CREATE_EXPRESSION(141), CREATE_EXPRESSION(142), CREATE_EXPRESSION(143),
          CREATE_EXPRESSION(144), CREATE_EXPRESSION(145), CREATE_EXPRESSION(146), CREATE_EXPRESSION(147),
          CREATE_EXPRESSION(148), CREATE_EXPRESSION(149), CREATE_EXPRESSION(150), CREATE_EXPRESSION(151),
          CREATE_EXPRESSION(152), CREATE_EXPRESSION(153), CREATE_EXPRESSION(154), CREATE_EXPRESSION(155),
          CREATE_EXPRESSION(156), CREATE_EXPRESSION(157), CREATE_EXPRESSION(158), CREATE_EXPRESSION(159),
          CREATE_EXPRESSION(160), CREATE_EXPRESSION(161), CREATE_EXPRESSION(162), CREATE_EXPRESSION(163),
          CREATE_EXPRESSION(164), CREATE_EXPRESSION(165), CREATE_EXPRESSION(166), CREATE_EXPRESSION(167),
          CREATE_EXPRESSION(168), CREATE_EXPRESSION(169), CREATE_EXPRESSION(170), CREATE_EXPRESSION(171),
          CREATE_EXPRESSION(172), CREATE_EXPRESSION(173), CREATE_EXPRESSION(174), CREATE_EXPRESSION(175),
          CREATE_EXPRESSION(176), CREATE_EXPRESSION(177), CREATE_EXPRESSION(178), CREATE_EXPRESSION(179),
          CREATE_EXPRESSION(180), CREATE_EXPRESSION(181), CREATE_EXPRESSION(182), CREATE_EXPRESSION(183),
          CREATE_EXPRESSION(184), CREATE_EXPRESSION(185), CREATE_EXPRESSION(186), CREATE_EXPRESSION(187),
          CREATE_EXPRESSION(188), CREATE_EXPRESSION(189), CREATE_EXPRESSION(190), CREATE_EXPRESSION(191),
          CREATE_EXPRESSION(192), CREATE_EXPRESSION(193), CREATE_EXPRESSION(194), CREATE_EXPRESSION(195),
          CREATE_EXPRESSION(196), CREATE_EXPRESSION(197), CREATE_EXPRESSION(198), CREATE_EXPRESSION(199),
          CREATE_EXPRESSION(200), CREATE_EXPRESSION(201), CREATE_EXPRESSION(202), CREATE_EXPRESSION(203),
          CREATE_EXPRESSION(204), CREATE_EXPRESSION(205), CREATE_EXPRESSION(206), CREATE_EXPRESSION(207),
          CREATE_EXPRESSION(208), CREATE_EXPRESSION(209), CREATE_EXPRESSION(210), CREATE_EXPRESSION(211),
          CREATE_EXPRESSION(212), CREATE_EXPRESSION(213), CREATE_EXPRESSION(214), CREATE_EXPRESSION(215),
          CREATE_EXPRESSION(216), CREATE_EXPRESSION(217), CREATE_EXPRESSION(218), CREATE_EXPRESSION(219),
          CREATE_EXPRESSION(220), CREATE_EXPRESSION(221), CREATE_EXPRESSION(222), CREATE_EXPRESSION(223),
          CREATE_EXPRESSION(224), CREATE_EXPRESSION(225), CREATE_EXPRESSION(226), CREATE_EXPRESSION(227),
          CREATE_EXPRESSION(228), CREATE_EXPRESSION(229), CREATE_EXPRESSION(230), CREATE_EXPRESSION(231),
          CREATE_EXPRESSION(232), CREATE_EXPRESSION(233), CREATE_EXPRESSION(234), CREATE_EXPRESSION(235),
          CREATE_EXPRESSION(236), CREATE_EXPRESSION(237), CREATE_EXPRESSION(238), CREATE_EXPRESSION(239),
          CREATE_EXPRESSION(240), CREATE_EXPRESSION(241), CREATE_EXPRESSION(242), CREATE_EXPRESSION(243),
          CREATE_EXPRESSION(244), CREATE_EXPRESSION(245), CREATE_EXPRESSION(246), CREATE_EXPRESSION(247),
          CREATE_EXPRESSION(248), CREATE_EXPRESSION(249), CREATE_EXPRESSION(250), CREATE_EXPRESSION(251),
          CREATE_EXPRESSION(252), CREATE_EXPRESSION(253)};

#undef CREATE_EXPRESSION
}
