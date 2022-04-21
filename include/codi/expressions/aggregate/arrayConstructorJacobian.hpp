#pragma once

#include "../../config.h"
#include "../../misc/macros.hpp"
#include "../../traits/realTraits.hpp"
#include "../expressionInterface.hpp"

/** \copydoc codi::Namespace */
namespace codi {

  /// Helper class for the detection of a Jacobian application for an array constructor.
  /// @tparam T_Creator The expression creating the Jacobian.
  /// @tparam T_ReturnType Used in the multiplication detection for the return type.
  template<typename T_Creator, typename T_ReturnType, size_t T_index>
  struct ArrayConstructorJacobian {
      using Creator = CODI_DD(T_Creator, CODI_T(ExpressionInterface<double, void>));  ///< See ArrayConstructorJacobian.
      using ReturnType = CODI_DD(T_ReturnType, double);                               ///< See ArrayConstructorJacobian.

      static size_t constexpr index = CODI_DD(T_index, 0);  ///< The index that is accessed.

      Creator const& creator;  ///< Reference to the creator.

      /// Constructor.
      CODI_INLINE ArrayConstructorJacobian(Creator const& creator) : creator(creator) {}
  };

  /// Detection of the application of an Jacobian from an array constructor. See ArrayConstructorJacobian.
  template<typename Type, typename Creator, typename ReturnType, size_t index>
  CODI_INLINE ReturnType operator*(ArrayConstructorJacobian<Creator, ReturnType, index> const& reduce,
                                   Type const& jac) {
    return RealTraits::AggregatedTypeTraits<typename Creator::Real>::template adjointOfConstructor<index>(
        reduce.creator.getValue(), jac);
  }
}
