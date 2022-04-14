#pragma once

#include <complex>

#include "../../misc/macros.hpp"
#include "../../config.h"
#include "../expressionInterface.hpp"
#include "adjointComplexToRealCast.hpp"
#include "../real/allOperators.hpp"

/** \copydoc codi::Namespace */
namespace codi {

  using std::abs;
  using std::arg;
  using std::conj;
  using std::imag;
  using std::norm;
  using std::polar;
  using std::proj;
  using std::real;

  //  CREATE_JAC_TRAIT(std::complex<double>, 1.0);
  //  CREATE_JAC_TRANSPOSE(std::complex<double>, std::complex<double>, std::conj(jacobian));

  //  CREATE_JAC_MULTIPLY(double, std::complex<double>, std::complex<double>, lhs * rhs);
  //  CREATE_JAC_MULTIPLY(std::complex<double>, double, std::complex<double>, lhs * rhs);
  //  CREATE_JAC_MULTIPLY(std::complex<double>, std::complex<double>, std::complex<double>, lhs * rhs);

  //  CREATE_JAC_DEDUCE(std::complex<double>, std::complex<double>, std::complex<double>);
  //  CREATE_JAC_DEDUCE(double, std::complex<double>, std::complex<double>);
  //  CREATE_JAC_DEDUCE(std::complex<double>, double, std::complex<double>);

  //  CREATE_GRAD_UPDATE(double, std::complex<double>, double, lhs += rhs.real());

  /*******************************************************************************/
  /// @name Builtin binary operators
  /// @{

// Use the Logic form the real definition
#define OPERATION_LOGIC OperationAdd
#define FUNCTION operator+
#include "binaryMixedComplexAndRealOverloads.tpp"

#define OPERATION_LOGIC OperationSubstract
#define FUNCTION operator-
#include "binaryMixedComplexAndRealOverloads.tpp"

#define OPERATION_LOGIC OperationMultiply
#define FUNCTION operator*
#include "binaryMixedComplexAndRealOverloads.tpp"

#define OPERATION_LOGIC OperationDivide
#define FUNCTION operator/
#include "binaryMixedComplexAndRealOverloads.tpp"

  /// @}
  /*******************************************************************************/
  /// @name Standard math library binary operators
  /// @{

  template<typename T_Real>
  struct OperationComplexPolar : public BinaryOperation<T_Real> {
    public:

      using Real = CODI_DD(T_Real, double);
      using Jacobian = Real;

      template<typename ArgA, typename ArgB>
      static CODI_INLINE Real primal(ArgA const& argA, ArgB const& argB) {
        return polar(argA, argB);
      }

      template<typename ArgA, typename ArgB>
      static CODI_INLINE Real gradientA(ArgA const& argA, ArgB const& argB, Real const& result) {
        CODI_UNUSED(argA, result);

        return polar(1.0, argB);
      }

      /// \copydoc codi::BinaryOperation::gradientB()
      template<typename ArgA, typename ArgB>
      static CODI_INLINE Real gradientB(ArgA const& argA, ArgB const& argB, Real const& result) {
        CODI_UNUSED(argA, argB);

        return Real(-imag(result), real(result));
      }
  };

#define FUNCTION polar
#define OPERATION_LOGIC OperationComplexPolar
#include "binaryRealToComplexOverloads.tpp"

  /// BinaryOperation specialization for complex pow
  template<typename T_Real>
  struct OperationPow<std::complex<T_Real>> : public BinaryOperation<std::complex<T_Real>> {
    public:

      using Real = CODI_DD(std::complex<T_Real>, double);  ///< See BinaryOperation.

      /// \copydoc codi::BinaryOperation::primal()
      template<typename ArgA, typename ArgB>
      static CODI_INLINE Real primal(ArgA const& argA, ArgB const& argB) {
        return pow(argA, argB);
      }

      /// \copydoc codi::BinaryOperation::gradientA()
      template<typename ArgA, typename ArgB>
      static CODI_INLINE Real gradientA(ArgA const& argA, ArgB const& argB, Real const& result) {
        CODI_UNUSED(result);

        return argB * pow(argA, argB - 1.0);
      }

      /// \copydoc codi::BinaryOperation::gradientB()
      template<typename ArgA, typename ArgB>
      static CODI_INLINE Real gradientB(ArgA const& argA, ArgB const& argB, Real const& result) {
        CODI_UNUSED(argB);

        // Complex cast for argA, since the real log for negative numbers is not defined.
        return log(Real(argA)) * result;
      }
  };

#define OPERATION_LOGIC OperationPow
#define FUNCTION pow
#include "binaryMixedComplexAndRealOverloads.tpp"

  /// @}
  /*******************************************************************************/
  /// @name Builtin binary comparison operators
  /// @{

#define OPERATOR ==
#include "conditionalBinaryMixedComplexAndRealOverloads.tpp"

#define OPERATOR !=
#include "conditionalBinaryMixedComplexAndRealOverloads.tpp"

  /// @}
  /*******************************************************************************/
  /// @name Standard math library unary operators
  /// @{

  // Functions handled by the real definitions:
  // exp, log, log10, sqrt, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh,
  // Unary operators handled by the real definitions:
  // operator+, operator-

  template<typename T_Real>
  struct OperationComplexAbs : public UnaryOperation<T_Real> {
    public:

      using Real = CODI_DD(T_Real, double);
      using Jacobian = std::complex<Real>;

      template<typename Arg>
      static CODI_INLINE Real primal(Arg const& arg) {
        return abs(arg);
      }

      template<typename Arg>
      static CODI_INLINE Jacobian gradient(Arg const& arg, Real const& result) {

        checkResult(result);
        if(result != 0.0) {
          return Jacobian(real(arg) / result, -imag(arg) / result);
        } else {
          return Jacobian(1.0);
        }
      }

    private:
      static CODI_INLINE void checkResult(Real const& result) {
        if (Config::CheckExpressionArguments) {
          if (RealTraits::getPassiveValue(result) == 0.0) {
            CODI_EXCEPTION("Zero divisor for abs derivative.");
          }
        }
      }
  };

#define FUNCTION abs
#define OPERATION_LOGIC OperationComplexAbs
#include "unaryComplexToRealOverloads.tpp"

  template<typename T_Real>
  struct OperationComplexArg : public UnaryOperation<T_Real> {
    public:

      using Real = CODI_DD(T_Real, double);
      using Jacobian = std::complex<Real>;

      template<typename Arg>
      static CODI_INLINE Real primal(Arg const& argument) {
        return arg(argument);
      }

      template<typename Arg>
      static CODI_INLINE Jacobian gradient(Arg const& argument, Real const& result) {
        checkResult(result);

        Real divisor = real(argument) * real(argument) + imag(argument) * imag(argument);
        divisor = 1.0 / divisor;

        return Jacobian(-imag(argument) * divisor, -real(argument) * divisor);
      }

    private:
      static CODI_INLINE void checkResult(Real const& result) {
        if (Config::CheckExpressionArguments) {
          if (RealTraits::getPassiveValue(result) == 0.0) {
            CODI_EXCEPTION("Zero divisor for arg derivative.");
          }
        }
      }
  };

#define FUNCTION arg
#define OPERATION_LOGIC OperationComplexArg
#include "unaryComplexToRealOverloads.tpp"

  struct RevConj {};

  template<typename T_Real>
  struct OperationComplexConj : public UnaryOperation<T_Real> {
    public:



      using Real = CODI_DD(T_Real, double);
      using Jacobian = RevConj;

      template<typename Arg>
      static CODI_INLINE Real primal(Arg const& arg) {
        return conj(arg);
      }

      template<typename Arg>
      static CODI_INLINE Jacobian gradient(Arg const& arg, Real const& result) {
        CODI_UNUSED(arg, result);

        return RevConj();
      }
  };

  template<typename T>
  auto operator*(RevConj const& /*op*/, T const& adj) -> decltype(conj(std::declval<T>())) {
    return conj(adj);
  }

#define FUNCTION conj
#define OPERATION_LOGIC OperationComplexConj
#include "../real/unaryOverloads.tpp"

  template<typename T_Real>
  struct OperationComplexImag : public UnaryOperation<T_Real> {
    public:

      using Real = CODI_DD(T_Real, double);
      using Jacobian = std::complex<Real>;

      template<typename Arg>
      static CODI_INLINE Real primal(Arg const& arg) {
        return arg.imag();
      }

      template<typename Arg>
      static CODI_INLINE Jacobian gradient(Arg const& arg, Real const& result) {
        CODI_UNUSED(arg, result);

        return Jacobian(0.0, -1.0);
      }
  };

#define FUNCTION imag
#define OPERATION_LOGIC OperationComplexImag
#include "unaryComplexToRealOverloads.tpp"

  template<typename T_Real>
  struct OperationComplexNorm : public UnaryOperation<T_Real> {
    public:

      using Real = CODI_DD(T_Real, double);
      using Jacobian = std::complex<Real>;

      template<typename Arg>
      static CODI_INLINE Real primal(Arg const& arg) {
        return norm(arg);
      }

      template<typename Arg>
      static CODI_INLINE Jacobian gradient(Arg const& arg, Real const& result) {
        CODI_UNUSED(result);

        return Jacobian(2.0 * real(arg), -2.0 * imag(arg));
      }
  };

#define FUNCTION norm
#define OPERATION_LOGIC OperationComplexNorm
#include "unaryComplexToRealOverloads.tpp"

  template<typename T_Real>
  struct OperationComplexProj : public UnaryOperation<T_Real> {
    public:

      using Real = CODI_DD(T_Real, double);
      using Jacobian = double;

      template<typename Arg>
      static CODI_INLINE Real primal(Arg const& argument) {
        return proj(argument);
      }

      template<typename Arg>
      static CODI_INLINE Jacobian gradient(Arg const& argument, Real const& result) {
        CODI_UNUSED(argument, result);

        return 1.0;
      }
  };

#define FUNCTION proj
#define OPERATION_LOGIC OperationComplexProj
#include "../real/unaryOverloads.tpp"

  template<typename T_Real>
  struct OperationComplexReal : public UnaryOperation<T_Real> {
    public:

      using Real = CODI_DD(T_Real, double);
      using Jacobian = std::complex<Real>;

      template<typename Arg>
      static CODI_INLINE Real primal(Arg const& arg) {
        return arg.real();
      }

      template<typename Arg>
      static CODI_INLINE Jacobian gradient(Arg const& arg, Real const& result) {
        CODI_UNUSED(arg, result);

        return Jacobian(1.0, 0.0);
      }
  };

#define FUNCTION real
#define OPERATION_LOGIC OperationComplexReal
#include "unaryComplexToRealOverloads.tpp"

  /// @}
}

namespace std {
  bool isfinite(std::complex<double> arg) {
    return isfinite(arg.real()) && isfinite(arg.imag());
  }

  using codi::abs;
  using codi::imag;
  using codi::real;

}
