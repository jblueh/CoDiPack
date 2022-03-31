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
  #include "../../traits/realTraits.hpp"
  #include "../activeType.hpp"
  #include "../unaryExpression.hpp"

  #define FUNCTION func
  #define OPERATION_LOGIC codi::UnaryOperation

namespace std {
#endif

  template<typename Tape>
  CODI_INLINE codi::UnaryExpression<complex<typename Tape::Real>,
    complex<codi::ActiveType<Tape>>,
    OPERATION_LOGIC>
  FUNCTION(complex<codi::ActiveType<Tape>> const& arg) {
    return codi::UnaryExpression<
        complex<typename Tape::Real>,
        complex<codi::ActiveType<Tape>>,
        OPERATION_LOGIC>(arg);
  }

// Create a correct include environment for viewing and programming in an IDE
#ifndef FUNCTION
}
#endif

#undef FUNCTION
#undef OPERATION_LOGIC
