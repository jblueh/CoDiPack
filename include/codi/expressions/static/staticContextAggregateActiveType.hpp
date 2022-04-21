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

#include "../../config.h"
#include "../../misc/macros.hpp"
#include "../../misc/staticFor.hpp"
#include "../../traits/realTraits.hpp"
#include "../aggregate/aggregatedActiveType.hpp"
#include "../logic/constructStaticContext.hpp"

/** \copydoc codi::Namespace */
namespace codi {

  /**
   * @brief Static context implementation of the aggregated active type.
   *
   * See codi::AggregatedActiveType for details.
   *
   * @tparam T_Real Real value of the aggregated type.
   * @tparam T_ActiveInnerType CoDiPack type that composes the aggregated type.
   */
  template<typename T_Real, typename T_ActiveInnerType>
  struct StaticAggregatedActiveType
      : public AggregatedActiveTypeBase<T_Real, T_ActiveInnerType,
                                        StaticAggregatedActiveType<T_Real, T_ActiveInnerType>, true> {
      using Real = T_Real;  ///< See StaticAggregatedActiveType.
      using ActiveInnerType = CODI_DD(T_ActiveInnerType,
                                      CODI_T(ActiveType<void>));  ///< See StaticAggregatedActiveType.

      using InnerReal = typename ActiveInnerType::Real;              ///< Inner real type of the active type.
      using InnerIdentifier = typename ActiveInnerType::Identifier;  ///< Inner real type of the active type.

      using Base =
          AggregatedActiveTypeBase<Real, ActiveInnerType, StaticAggregatedActiveType, true>;  ///< Abbreviation for base
                                                                                              ///< class.

      CODI_INLINE StaticAggregatedActiveType() : Base() {}  ///< Constructor.

      CODI_INLINE StaticAggregatedActiveType(StaticAggregatedActiveType const&) = default;  ///< Constructor.
  };

#ifndef DOXYGEN_DISABLE

  template<typename T_Rhs, typename T_Tape, size_t T_primalValueOffset, size_t T_constantValueOffset>
  struct ConstructStaticContextLogic<T_Rhs, T_Tape, T_primalValueOffset, T_constantValueOffset,
                                     RealTraits::EnableIfAggregatedActiveType<T_Rhs>> {
    public:

      using Rhs = CODI_DD(T_Rhs, CODI_T(AggregatedActiveType<double, ActiveType<CODI_ANY>, CODI_ANY, 1>));
      using Tape = T_Tape;
      static constexpr size_t primalValueOffset = T_primalValueOffset;
      static constexpr size_t constantValueOffset = T_constantValueOffset;

      using Real = typename Rhs::Real;
      using ActiveInnerType = typename Rhs::ActiveInnerType;
      static int constexpr Elements = Rhs::Elements;

      using InnerConstructor = ConstructStaticContextLogic<ActiveInnerType, Tape, 0, 0>;
      using StaticInnerType = typename InnerConstructor::ResultType;

      using InnerReal = typename Tape::Real;
      using InnerIdentifier = typename Tape::Identifier;
      using PasiverInnerReal = typename Tape::PassiveReal;

      using ResultType = StaticAggregatedActiveType<Real, StaticInnerType>;

      /// Forwards to the static construction of the inner active type.
      CODI_INLINE static ResultType construct(InnerReal* primalVector, InnerIdentifier const* const identifiers,
                                              PasiverInnerReal const* const constantData) {
        CODI_UNUSED(constantData);

        ResultType value;

        static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
          new (&value.arrayValue[i.value]) StaticInnerType(InnerConstructor::construct(
              primalVector, &identifiers[primalValueOffset + i.value], &constantData[constantValueOffset + i.value]));
        });

        return value;
      }
  };

#endif
}
