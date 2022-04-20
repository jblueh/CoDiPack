/*
 * In order to include this file the user has to define the preprocessor macros OPERATION_LOGIC and FUNCTION.
 * OPERATION_LOGIC contains the name of the operation logic class. FUNCTION represents the normal name of that function
 * e.g. 'operator -' or 'sin'.
 *
 * The defines OPERATION_LOGIC and FUNCTION will be undefined at the end of this template.
 *
 * Prior to including this file, the user has to implement the operation's primal and derivative logic according to
 * BinaryOpInterface.
 */

#ifndef OPERATOR
  #error Please define the operator for the comparison.
#endif

// Create a correct include environment for viewing and programming in an IDE
#ifndef OPERATOR
  #include <complex>

  #include "../../misc/macros.hpp"
  #include "../../config.h"
  #include "../../traits/realTraits.hpp"
  #include "../expressionInterface.hpp"

  #define OPERATOR ==

namespace codi {
#endif

  // Do not need to define complex complex binding, they are handled by the default real definitions

  /// Function overload for OPERATOR(complex, const real).
  template<typename Real, typename ArgA>
  CODI_INLINE bool operator OPERATOR(ExpressionInterface<std::complex<Real>, ArgA> const& argA, RealTraits::PassiveReal<Real> const& argB) {
    return RealTraits::getPassiveValue(argA.cast()) OPERATOR argB;
  }

  /// Function overload for OPERATOR(const real, complex).
  template<typename Real, typename ArgB>
  CODI_INLINE bool operator OPERATOR(RealTraits::PassiveReal<Real> const& argA, ExpressionInterface<std::complex<Real>, ArgB> const& argB) {
    return argA OPERATOR RealTraits::getPassiveValue(argB.cast());
  }

// Create a correct include environment for viewing and programming in an IDE
#ifndef OPERATOR
}
#endif

#undef OPERATOR
