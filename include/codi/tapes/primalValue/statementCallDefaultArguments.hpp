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

  /// Statement evaluation arguments for linear index handlers.
  /// @tparam T_Identifier Identifier type.
  template<typename T_Identifier>
  struct StatementCallDefaultArgumentsBase {
    public:
      using Identifier = CODI_DD(T_Identifier, int);  ///< See StatementCallDefaultArgumentsBase.

      /// Linear version of the pack helper.
      struct LinearPackHelper {
        public:
          size_t&  __restrict__ linearAdjointPos;  ///< Position of the lhs adjoint.
          Config::ArgumentSize numberOfPassiveArguments;  ///< Number of passive arguments.
      };

      /// Reuse version of the pack helper.
      struct ReusePackHelper {
        public:
          Config::ArgumentSize numberOfPassiveArguments;  ///< Number of passive arguments.
      };

      static size_t tempAdjointPos;  ///< Temporary position of the lhs adjoint.

      size_t&  __restrict__ linearAdjointPos;  ///< Position of the lhs adjoint.
      Config::ArgumentSize numberOfPassiveArguments;  ///< Number of passive arguments.
      size_t& __restrict__ curDynamicSizePos;  ///< Position for the dynamic size statement data.
      char* const __restrict__ dynamicSizeValues;  ///< Pointer to the dynamic size statement data.

      /// Constructor.
      StatementCallDefaultArgumentsBase(
          size_t&  __restrict__ linearAdjointPos,
          Config::ArgumentSize numberOfPassiveArguments,
          size_t& __restrict__ curDynamicSizePos,
          char* const __restrict__ dynamicSizeValues) :
        linearAdjointPos(linearAdjointPos),
        numberOfPassiveArguments(numberOfPassiveArguments),
        curDynamicSizePos(curDynamicSizePos),
        dynamicSizeValues(dynamicSizeValues)
      {}

      /// Constructor.
      StatementCallDefaultArgumentsBase(
          Config::ArgumentSize numberOfPassiveArguments,
          size_t& __restrict__ curDynamicSizePos,
          char* const __restrict__ dynamicSizeValues) :
        StatementCallDefaultArgumentsBase(tempAdjointPos, numberOfPassiveArguments, curDynamicSizePos, dynamicSizeValues) {}

      /// Constructor.
      template<typename ... Args>
      StatementCallDefaultArgumentsBase(LinearPackHelper packHelper, Args&& ... args) :
        StatementCallDefaultArgumentsBase(packHelper.linearAdjointPos, packHelper.numberOfPassiveArguments, std::forward<Args>(args)...) {}

      /// Constructor.
      template<typename ... Args>
      StatementCallDefaultArgumentsBase(ReusePackHelper packHelper, Args&& ... args) :
        StatementCallDefaultArgumentsBase(tempAdjointPos, packHelper.numberOfPassiveArguments, std::forward<Args>(args)...) {}


      /*******************************************************************************/
      /// Interface definition

      using PackHelper = CODI_DD(CODI_UNDEFINED, LinearPackHelper); ///< Defines which pack helper is used.

      /// Moves the linear adjoint position by the number of output arguments.
      CODI_INLINE void updateAdjointPosForward(size_t nOutputArgs);

      /// Moves the linear adjoint position by the number of output arguments.
      CODI_INLINE void updateAdjointPosReverse(size_t nOutputArgs);

      /// Get the lhs identifier.
      CODI_INLINE Identifier getLhsIdentifier(size_t pos, Identifier const* __restrict__ identifiers);

      /// Unpack the variadic part of the the data into the PackHelper.
      CODI_INLINE PackHelper unpackVariadic();
  };

  template<typename Identifier>
  size_t StatementCallDefaultArgumentsBase<Identifier>::tempAdjointPos = 0;

  /// Linear index handling implementation of the StatementCallDefaultArgumentsBase.
  template<typename T_Identifier>
  struct LinearStatementCallDefaultArguments : public StatementCallDefaultArgumentsBase<T_Identifier> {
    public:
      using Identifier = CODI_DD(T_Identifier, int);  ///< See StatementCallDefaultArgumentsBase.
      using Base = StatementCallDefaultArgumentsBase<Identifier>; ///< Base class abbreviation.

      using PackHelper = typename Base::LinearPackHelper; ///< See StatementCallDefaultArgumentsBase.

      using Base::Base; ///< Use all constructors.

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

      /// Packs the linear adjoint position and numberOfPassiveArguments.
      CODI_INLINE PackHelper unpackVariadic() {
        return PackHelper{this->linearAdjointPos, this->numberOfPassiveArguments};
      }
  };

  template<typename T_Identifier>
  struct ReuseStatementCallDefaultArguments : public StatementCallDefaultArgumentsBase<T_Identifier> {
    public:
      using Identifier = CODI_DD(T_Identifier, int); ///< See StatementCallDefaultArgumentsBase.
      using Base = StatementCallDefaultArgumentsBase<Identifier>; ///< Base class abbreviation.

      using PackHelper = typename Base::ReusePackHelper; ///< See StatementCallDefaultArgumentsBase.

      using Base::Base; ///< Use all constructors.

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

      /// Packs the numberOfPassiveArguments.
      CODI_INLINE PackHelper unpackVariadic() {
        return PackHelper{this->numberOfPassiveArguments};
      }
  };
}
