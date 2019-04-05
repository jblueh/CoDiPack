/*
 * CoDiPack, a Code Differentiation Package
 *
 * Copyright (C) 2015-2019 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
 *
 * Lead developers: Max Sagebaum, Tim Albring (SciComp, TU Kaiserslautern)
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
 * Authors: Max Sagebaum, Tim Albring, (SciComp, TU Kaiserslautern)
 *          Prof. Robin Hogan, (Univ. of Reading).
 *
 * Originally based on Adept 1.0 (http://www.met.rdg.ac.uk/clouds/adept/)
 * released under GPL 3.0 (Copyright (C) 2012-2013 Robin Hogan and the University of Reading).
 */

/*
 * In order to include this file the user has to define the preprocessor macro OPERATOR_LOGIC and FUNCTION.
 * OPERATOR_LOGIC contains the name of the operator logic class. FUNCTION represents the normal name of that function
 * e.g. 'operator -' or 'sin'.
 *
 * The defines OPERATOR_LOGIC and FUNCTION will be undefined at the end of this template.
 *
 * Prior to including this file, the user has to implement the operator's primal and derivative logic according to UnaryOpInterface.
 */

#ifndef OPERATOR_LOGIC
  #error Please define a name for the unary expression.
#endif
#ifndef FUNCTION
  #error Please define the primal function representation.
#endif

#include "macros.h"

/**
 * @brief Overload for FUNC with the CoDiPack expressions.
 *
 * @param[in] a The argument of the operation.
 *
 * @return UnaryOp instanciated for IMPL.
 *
 * @tparam Real The real type used in the active types.
 * @tparam A The expression for the first argument of the function.
 */
template <typename Real, class A>
CODI_INLINE UnaryOp<Real, A, OPERATOR_LOGIC> FUNCTION(const Expression<Real, A>& a) {
  return UnaryOp<Real, A, OPERATOR_LOGIC>(a.cast());
}

#undef FUNCTION
#undef OPERATOR_LOGIC
