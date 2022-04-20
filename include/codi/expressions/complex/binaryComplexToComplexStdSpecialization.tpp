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

#ifndef OPERATION_LOGIC
  #error Please define a name for the binary expression.
#endif
#ifndef FUNCTION
  #error Please define the primal function representation.
#endif

// Create a correct include environment for viewing and programming in an IDE
#ifndef FUNCTION
  #include <complex>

  #include "../../misc/macros.hpp"
  #include "../../config.h"
  #include "../activeType.hpp"
  #include "../constantExpression.hpp"
  #include "../binaryExpression.hpp"
  #include "adjointComplexToRealCast.hpp"

#define OPERATION_LOGIC codi::BinaryOperation
#define FUNCTION func

namespace std {
#endif

  // Define complex complex bindings

  /// Function overload for FUNCTION(complex, complex)
  template<typename Tape>
  CODI_INLINE codi::BinaryExpression<complex<typename Tape::Real>,
    complex<codi::ActiveType<Tape>>,
    complex<codi::ActiveType<Tape>>,
    OPERATION_LOGIC>
  FUNCTION(complex<codi::ActiveType<Tape>> const& argA, complex<codi::ActiveType<Tape>> const& argB) {
    return codi::BinaryExpression<
        complex<typename Tape::Real>,
        complex<codi::ActiveType<Tape>>,
        complex<codi::ActiveType<Tape>>,
        OPERATION_LOGIC>(argA, argB);
  }

  // Define complex real bindings

  /// Function overload for FUNCTION(complex, real)
  template<typename Tape>
  CODI_INLINE codi::BinaryExpression<complex<typename Tape::Real>,
    complex<codi::ActiveType<Tape>>,
    codi::AdjointComplexToRealCast<typename Tape::Real, codi::ActiveType<Tape>>,
    OPERATION_LOGIC>
  FUNCTION(complex<codi::ActiveType<Tape>> const& argA, codi::ActiveType<Tape> const& argB) {
    return codi::BinaryExpression<
        complex<typename Tape::Real>,
        complex<codi::ActiveType<Tape>>,
        codi::AdjointComplexToRealCast<typename Tape::Real, codi::ActiveType<Tape>>,
        OPERATION_LOGIC>(argA, codi::AdjointComplexToRealCast<typename Tape::Real, codi::ActiveType<Tape>>(argB));
  }

  /// Function overload for FUNCTION(complex, const real)
  template<typename Tape>
  CODI_INLINE codi::BinaryExpression<complex<typename Tape::Real>,
    complex<codi::ActiveType<Tape>>,
    codi::ConstantExpression<typename Tape::PassiveReal>,
    OPERATION_LOGIC>
  FUNCTION(complex<codi::ActiveType<Tape>> const& argA, typename Tape::PassiveReal const& argB) {
    return codi::BinaryExpression<
        complex<typename Tape::Real>,
        complex<codi::ActiveType<Tape>>,
        codi::ConstantExpression<typename Tape::PassiveReal>,
        OPERATION_LOGIC>(argA, codi::ConstantExpression<typename Tape::PassiveReal>(argB));
  }

  // Define real complex bindings

  /// Function overload for FUNCTION(real, complex)
  template<typename Tape>
  CODI_INLINE codi::BinaryExpression<complex<typename Tape::Real>,
    codi::AdjointComplexToRealCast<typename Tape::Real, codi::ActiveType<Tape>>,
    complex<codi::ActiveType<Tape>>,
    OPERATION_LOGIC>
  FUNCTION(codi::ActiveType<Tape> const& argA, complex<codi::ActiveType<Tape>> const& argB) {
    return codi::BinaryExpression<
        complex<typename Tape::Real>,
        codi::AdjointComplexToRealCast<typename Tape::Real, codi::ActiveType<Tape>>,
        complex<codi::ActiveType<Tape>>,
        OPERATION_LOGIC>(codi::AdjointComplexToRealCast<typename Tape::Real, codi::ActiveType<Tape>>(argA), argB);
  }

  /// Function overload for FUNCTION(const real, complex)
  template<typename Tape>
  CODI_INLINE codi::BinaryExpression<complex<typename Tape::Real>,
    codi::ConstantExpression<typename Tape::PassiveReal>,
    complex<codi::ActiveType<Tape>>,
    OPERATION_LOGIC>
  FUNCTION(typename Tape::PassiveReal const& argA, complex<codi::ActiveType<Tape>> const& argB) {
    return codi::BinaryExpression<
        complex<typename Tape::Real>,
        codi::ConstantExpression<typename Tape::PassiveReal>,
        complex<codi::ActiveType<Tape>>,
        OPERATION_LOGIC>(codi::ConstantExpression<typename Tape::PassiveReal>(argA), argB);
  }

// Create a correct include environment for viewing and programming in an IDE
#ifndef FUNCTION
}
#endif

#undef FUNCTION
#undef OPERATION_LOGIC
