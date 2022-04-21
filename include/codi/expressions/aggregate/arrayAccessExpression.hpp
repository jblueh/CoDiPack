#pragma once

#include "../../misc/macros.hpp"
#include "../../config.h"
#include "../../traits/realTraits.hpp"
#include "../unaryExpression.hpp"

/** \copydoc codi::Namespace */
namespace codi {


  /// Definitions for the ArrayAccessExpression.
  ///
  /// @tparam T_Real Return type of the expression.
  template<typename T_Real>
  struct ArrayAccessExpressionImpl {

      using Real = CODI_DD(T_Real, CODI_ANY); ///< See ArrayAccessExpressionImpl.

      using Traits = RealTraits::AggregatedTypeTraits<Real>; ///< Traits of the aggregated type.

      using InnerReal = typename Traits::InnerType; ///< Inner type of the aggregate.

      /// Implementation of the array access operator for a specific element.
      /// @tparam T_element Element that is accessed.
      template<size_t T_element>
      struct ArrayAccessOperationImpl {

          /// Operation for array access.
          /// @tparam T_OpReal Real value of the operator.
          template<typename T_OpReal>
          struct type : public UnaryOperation<T_OpReal> {
            public:

              using OpReal = CODI_DD(T_OpReal, double);  ///< See type.
              static size_t constexpr element = T_element; ///< See ArrayAccessOperationImpl.

              using Jacobian = Real; ///< Jacobian is the aggregated type.

              /// \copydoc UnaryOperation::primal().
              template<typename Arg>
              static CODI_INLINE OpReal primal(Arg const& arg) {
                return Traits::template arrayAccess<element>(arg);
              }

              /// \copydoc UnaryOperation::primal().
              template<typename Arg>
              static CODI_INLINE Jacobian gradient(Arg const& arg, OpReal const& result) {
                CODI_UNUSED(result);
                return Traits::template adjointOfArrayAccess<element>(arg, 1.0);
              }
          };
      };

      /// Definition of the array access expression.
      template<size_t element, typename Arg>
      using type = UnaryExpression<InnerReal, Arg, ArrayAccessOperationImpl<element>::template type>;
  };

  /// Expressions that performs a[element] in a compile time context.
  template<typename Real, size_t element, typename Arg>
  using ArrayAccessExpression = typename ArrayAccessExpressionImpl<Real>::template type<element, Arg>;

}
