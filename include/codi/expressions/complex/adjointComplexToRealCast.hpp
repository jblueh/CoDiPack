#pragma once

#include <complex>

#include "../../misc/macros.hpp"
#include "../../config.h"
#include "../../traits/expressionTraits.hpp"
#include "../unaryExpression.hpp"

/** \copydoc codi::Namespace */
namespace codi {

  struct ReduceToReal {}; ///< Placeholder to identify the operation on the Jacobian.

  /**
   * Returns a proxy object for the gradient that implements the operation in the multiplication of the proxy.
   *
   * @tparam T_Real  Original primal value of the statement/expression.
   */
  template<typename T_Real>
  struct OperationCastRealToComplex : public UnaryOperation<T_Real> {
    public:

      using Real = CODI_DD(T_Real, double); ///< See OperationCastRealToComplex.
      using Jacobian = ReduceToReal;   ///< See OperationCastRealToComplex.

      /// \copydoc codi::UnaryOperation::primal
      template<typename Arg>
      static CODI_INLINE Real primal(Arg const& arg) {
        return arg;
      }

      /// See OperationCastRealToComplex.
      template<typename Arg>
      static CODI_INLINE ReduceToReal gradient(Arg const& arg, Real const& result) {
        CODI_UNUSED(arg, result);

        return ReduceToReal {};
      }
  };

  /// Expression that converts in the adjoint evaluation a complex to the real part.
  template<typename T_Real, typename T_Arg>
  using AdjointComplexToRealCast = UnaryExpression<T_Real, T_Arg, OperationCastRealToComplex>;

  /// See codi::OperationCastRealToComplex.
  template<typename Real>
  CODI_INLINE Real operator *(ReduceToReal, std::complex<Real> const& adjoint) {
    return adjoint.real();
  }

  /// See codi::OperationCastRealToComplex.
  template<typename Real, typename Arg>
  CODI_INLINE ExpressionTraits::ActiveResult<Real, typename Arg::ADLogic> operator *(ReduceToReal, ExpressionInterface<std::complex<Real>, Arg> const& adjoint) {
    return adjoint.cast().real();
  }
}
