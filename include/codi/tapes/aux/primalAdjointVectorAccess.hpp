#pragma once

#include <cstddef>

#include "../../aux/macros.hpp"
#include "../../config.h"
#include "adjointVectorAccess.hpp"

/** \copydoc codi::Namespace */
namespace codi {

  /**
   * @brief Implementation of VectorAccessInterface for adjoint and primal vectors.
   *
   * Both vectors are used as is, they are assumed to have correct sizes. No bounds checking is performed.
   *
   * Inherits from AdjointVectorAccess and overwrites all methods specific to the primals.
   *
   * @tparam _Real        The computation type of a tape, usually chosen as ActiveType::Real.
   * @tparam _Identifier  The adjoint/tangent identification of a tape, usually chosen as ActiveType::Identifier.
   * @tparam _Gradient    The gradient type of a tape, usually chosen as ActiveType::Gradient.
   */
  template<typename _Real, typename _Identifier, typename _Gradient>
  struct PrimalAdjointVectorAccess : public AdjointVectorAccess<_Real, _Identifier, _Gradient> {
      using Real = CODI_DD(_Real, double);           ///< See PrimalAdjointVectorAccess.
      using Identifier = CODI_DD(_Identifier, int);  ///< See PrimalAdjointVectorAccess.
      using Gradient = CODI_DD(_Gradient, double);   ///< See PrimalAdjointVectorAccess.

      using Base = AdjointVectorAccess<Real, Identifier, Gradient>;  ///< Base class abbreviation.

    private:

      Real* primalVector;  ///< Pointer to the primal vector.

    public:

      /// Constructor. See interface documentation for details about the vectors.
      PrimalAdjointVectorAccess(Gradient* adjointVector, Real* primalVector)
          : Base(adjointVector), primalVector(primalVector) {}

      /*******************************************************************************/
      /// @name Primal access

      /// \copydoc VectorAccessInterface::setPrimal
      void setPrimal(Identifier const& index, Real const& primal) {
        primalVector[index] = primal;
      }

      /// \copydoc VectorAccessInterface::getPrimal
      Real getPrimal(Identifier const& index) {
        return primalVector[index];
      }

      /// \copydoc VectorAccessInterface::hasPrimals <br>
      /// Implementation: Always returns true.
      bool hasPrimals() {
        return true;
      }
  };
}