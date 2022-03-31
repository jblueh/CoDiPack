#pragma once

#include <complex>

#include "../../misc/macros.hpp"
#include "../../config.h"
#include "../../traits/expressionTraits.hpp"
#include "../unaryExpression.hpp"

/** \copydoc codi::Namespace */
namespace codi {

  struct ReduceToReal {};

  template<typename T_Real>
  struct OperationCastRealToComplex : public UnaryOperation<T_Real> {
    public:

      using Real = CODI_DD(T_Real, double);
      using Jacobian = ReduceToReal;

      template<typename Arg>
      static CODI_INLINE Real primal(Arg const& arg) {
        return arg;
      }

      template<typename Arg>
      static CODI_INLINE ReduceToReal gradient(Arg const& arg, Real const& result) {
        CODI_UNUSED(arg, result);

        return ReduceToReal {};
      }
  };

  template<typename T_Real, typename T_Arg>
  using AdjointComplexToRealCast = UnaryExpression<std::complex<T_Real>, T_Arg, OperationCastRealToComplex>;

  template<typename Real>
  CODI_INLINE Real operator *(ReduceToReal, std::complex<Real> const& adjoint) {
    return adjoint.real();
  }

  template<typename Real, typename Arg>
  CODI_INLINE ExpressionTraits::ActiveResultFromExpr<Arg> operator *(ReduceToReal, ExpressionInterface<std::complex<Real>, Arg> const& adjoint) {
    return adjoint.cast().real();
  }
}
