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

#include <type_traits>

#include "../misc/macros.hpp"
#include "../config.h"
#include "../expressions/logic/compileTimeTraversalLogic.hpp"
#include "misc/enableIfHelpers.hpp"

/** \copydoc codi::Namespace */
namespace codi {

  template<typename T_Real, typename T_Impl>
  struct ExpressionInterface;

  template<typename T_Real, typename T_Gradient, typename T_Tape, typename T_Impl>
  struct LhsExpressionInterface;

  template<typename T_Real, template<typename> class T_ConversionOperator>
  struct ConstantExpression;

  template<typename T_Tape>
  struct StaticContextActiveType;

  namespace RealTraits {
    template<typename T_Real, typename>
    struct AggregatedTypeTraits;
  }

  /// Traits for everything that can be an expression e.g. codi::RealReverse, a + b, etc..
  namespace ExpressionTraits {

    /*******************************************************************************/
    /// @name Expression traits.
    /// @{

    /// Validates if the AD logic of two expressions are the same or compatible. `void` results are
    /// interpreted as the AD logic of a constant expression.
    template<typename LogicA, typename LogicB, typename = void>
    struct ValidateADLogicImpl {
      private:
        static bool constexpr isAVoid = std::is_same<void, LogicA>::value;
        static bool constexpr isBVoid = std::is_same<void, LogicB>::value;
        static bool constexpr isBothVoid = isAVoid & isBVoid;
        static bool constexpr isBothSame = std::is_same<LogicA, LogicB>::value;

        // Either one can be void (aka. constant value) but not both otherwise both need to be the same.
        static_assert((!isBothVoid) & (!isAVoid | !isBVoid | isBothSame), "AD logic types need to be the same.");

      public:

        /// The resulting active type of an expression.
        using ADLogic = typename std::conditional<isBVoid, LogicA, LogicB>::type;
    };

    /// \copydoc ValidateADLogicImpl
    template<typename ResultA, typename ResultB>
    using ValidateADLogic = ValidateADLogicImpl<ResultA, ResultB>;

    /// Create an CoDiPack active type that can capture an expression result. The ADLogic type definition in the
    /// expression is usually the tape type.
    /// @tparam T_Real Real value of the expression.
    /// @tparam T_Tape ADLogic of the expression.
    /// @tparam T_isStatic If a static context active type should be used.
    template<typename T_Real, typename T_Tape, bool T_isStatic = false, typename = void>
    struct ActiveResultImpl {

        using Real = CODI_DD(T_Real, CODI_ANY);
        using Tape = CODI_DD(T_Tape, CODI_ANY);

        /// The resulting active type of an expression.
        using ActiveResult = CODI_ANY;
    };

    /// \copydoc ActiveResultImpl
    template<typename Real, typename Tape, bool isStatic = false>
    using ActiveResult = typename ActiveResultImpl<Real, Tape, isStatic>::ActiveResult;

    /// Create an CoDiPack active type that can capture an expression result.
    template<typename T_Expr, bool isStatic = false, typename = void>
    struct ActiveResultFromExprImpl {

        using Expr = CODI_DD(T_Expr, CODI_ANY);

        /// The resulting active type of an expression.
        using ActiveResult = Expr;
    };

    /// \copydoc ActiveResultFromExprImpl
    template<typename Expr, bool isStatic = false>
    using ActiveResultFromExpr = typename ActiveResultFromExprImpl<Expr, isStatic>::ActiveResult;

    /// @}
    /*******************************************************************************/
    /// @name Detection of specific node types
    /// @{

    /// If the expression inherits from ExpressionInterface. Is either std::false_type or std::true_type
    template<typename Expr, typename = void>
    struct IsExpression : std::false_type {};

#ifndef DOXYGEN_DISABLE
    template<typename Expr>
    struct IsExpression<Expr, typename enable_if_base_of<ExpressionInterface<typename Expr::Real, Expr>, Expr>::type>
        : std::true_type {};
#endif

#if CODI_IS_CPP14
    /// Value entry of IsExpression
    template<typename Expr>
    bool constexpr isExpression = IsExpression<Expr>::value;
#endif

    /// Enable if wrapper for IsExpression
    template<typename Expr, typename T = void>
    using EnableIfExpression = typename std::enable_if<IsExpression<Expr>::value, T>::type;

    /// If the expression inherits from LhsExpressionInterface. Is either std::false_type or std::true_type
    template<typename Expr, typename = void>
    struct IsLhsExpression : std::false_type {};

#ifndef DOXYGEN_DISABLE
    template<typename Expr>
    struct IsLhsExpression<
        Expr, typename enable_if_base_of<
                  LhsExpressionInterface<typename Expr::Real, typename Expr::Gradient, typename Expr::Tape, Expr>,
                  Expr>::type> : std::true_type {};

    template<typename Tape>
    struct IsLhsExpression<StaticContextActiveType<Tape>> : std::true_type {};
#endif

#if CODI_IS_CPP14
    /// Value entry of IsLhsExpression
    template<typename Expr>
    bool constexpr isLhsExpression = IsLhsExpression<Expr>::value;
#endif

    /// Enable if wrapper for IsLhsExpression
    template<typename Expr, typename T = void>
    using EnableIfLhsExpression = typename std::enable_if<IsLhsExpression<Expr>::value, T>::type;

    /// If the expression inherits from ConstantExpression. Is either std::false_type or std::true_type
    template<typename Expr>
    struct IsConstantExpression : std::false_type {};

#ifndef DOXYGEN_DISABLE
    template<typename Real, template<typename> class ConversionOperator>
    struct IsConstantExpression<ConstantExpression<Real, ConversionOperator>> : std::true_type {};
#endif

#if CODI_IS_CPP14
    template<typename Expr>
    /// Value entry of IsConstantExpression
    bool constexpr isConstantExpression = IsConstantExpression<Expr>::value;
#endif

    /// Enable if wrapper for IsConstantExpression
    template<typename Expr, typename T = void>
    using EnableIfConstantExpression = typename std::enable_if<IsConstantExpression<Expr>::value, T>::type;

    /// If the expression inherits from StaticContextActiveType. Is either std::false_type or std::true_type
    template<typename Expr>
    struct IsStaticContextActiveType : std::false_type {};

#ifndef DOXYGEN_DISABLE
    template<typename Tape>
    struct IsStaticContextActiveType<StaticContextActiveType<Tape>> : std::true_type {};
#endif

#if CODI_IS_CPP14
    /// Value entry of IsStaticContextActiveType
    template<typename Expr>
    bool constexpr isStaticContextActiveType = IsStaticContextActiveType<Expr>::value;
#endif

    /// Enable if wrapper for IsStaticContextActiveType
    template<typename Expr, typename T = void>
    using EnableIfStaticContextActiveType = typename std::enable_if<IsStaticContextActiveType<Expr>::value, T>::type;

    /// @}
    /*******************************************************************************/
    /// @name Static values on expressions
    /// @{

    /// Counts the number of nodes that inherit from LhsExpressionInterface in the expression.
    template<typename Expr>
    struct NumberOfActiveTypeArguments : public CompileTimeTraversalLogic<size_t, NumberOfActiveTypeArguments<Expr>> {
      public:

        /// \copydoc CompileTimeTraversalLogic::leaf()
        template<typename Node, typename = ExpressionTraits::EnableIfLhsExpression<Node>>
        CODI_INLINE static size_t constexpr leaf() {
          return 1;
        }
        using CompileTimeTraversalLogic<size_t, NumberOfActiveTypeArguments>::leaf;

        /// See NumberOfActiveTypeArguments
        static size_t constexpr value = NumberOfActiveTypeArguments::template eval<Expr>();
    };

#if CODI_IS_CPP14
    /// Value entry of NumberOfActiveTypeArguments
    template<typename Expr>
    bool constexpr numberOfActiveTypeArguments = NumberOfActiveTypeArguments<Expr>::value;
#endif

    /// Counts the number of types that inherit from ConstantExpression in the expression.
    template<typename Expr>
    struct NumberOfConstantTypeArguments
        : public CompileTimeTraversalLogic<size_t, NumberOfConstantTypeArguments<Expr>> {
      public:

        /// \copydoc CompileTimeTraversalLogic::leaf()
        template<typename Node, typename = EnableIfConstantExpression<Node>>
        CODI_INLINE static size_t constexpr leaf() {
          return ::codi::RealTraits::AggregatedTypeTraits<typename Node::Real, void>::Elements;
        }
        using CompileTimeTraversalLogic<size_t, NumberOfConstantTypeArguments>::leaf;

        /// See NumberOfConstantTypeArguments
        static size_t constexpr value = NumberOfConstantTypeArguments::template eval<Expr>();
    };

#if CODI_IS_CPP14
    /// Value entry of NumberOfConstantTypeArguments
    template<typename Expr>
    bool constexpr numberOfConstantTypeArguments = NumberOfConstantTypeArguments<Expr>::value;
#endif

    /// @}
    /*******************************************************************************/
    /// @name Specialization of various definitions.
    /// @{

#ifndef DOXYGEN_DISABLE
    // Can not directly be specialized since EnableIfExpression is not available at the time of definition.
    template<typename T_Expr, bool isStatic>
    struct ActiveResultFromExprImpl<T_Expr, isStatic, EnableIfExpression<T_Expr>> {

        using Expr = CODI_DD(T_Expr, CODI_ANY);
        using Real = typename Expr::Real;
        using Tape = typename Expr::ADLogic;

        /// The resulting active type of an expression.
        using ActiveResult = typename ActiveResultImpl<Real, Tape, isStatic>::ActiveResult;
    };
#endif


    /// @}
  }
}
