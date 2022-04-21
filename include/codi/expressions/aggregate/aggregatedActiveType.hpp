#pragma once

#include "../../config.h"
#include "../../misc/macros.hpp"
#include "../../misc/staticFor.hpp"
#include "../../traits/realTraits.hpp"
#include "../logic/constructStaticContext.hpp"
#include "arrayConstructorJacobian.hpp"

/** \copydoc codi::Namespace */
namespace codi {

  /**
   * @brief Defines the aggregated type via an array and implements the ExpressionInterface.
   *
   * @tparam T_Real Real value of the aggregated type.
   * @tparam T_ActiveInnerType CoDiPack type that composes the aggregated type.
   * @tparam T_Impl The final implementation of the aggregated type.
   * @tparam T_isStatic If the aggregated type is created in a static context.
   */
  template<typename T_Real, typename T_ActiveInnerType, typename T_Impl, bool T_isStatic>
  struct AggregatedActiveTypeBase : public ExpressionInterface<T_Real, T_Impl> {
    public:

      using Real = T_Real;                                                           ///< See AggregatedActiveTypeBase.
      using ActiveInnerType = CODI_DD(T_ActiveInnerType, CODI_T(ActiveType<void>));  ///< See AggregatedActiveTypeBase.
      using Impl = CODI_DD(T_Impl, CODI_T(AggregatedActiveTypeBase<Type, void>));    ///< See AggregatedActiveTypeBase.
      static bool constexpr isStatic = T_isStatic;                                   ///< See AggregatedActiveTypeBase.

      using Tape = typename ActiveInnerType::Tape;            ///< The tape of the inner active type.
      using Traits = RealTraits::AggregatedTypeTraits<Real>;  ///< The traits for the aggregated type.
      static int constexpr Elements = Traits::Elements;

      using InnerReal = typename Traits::InnerType;       ///< Inner real type of the active type.
      using InnerIdentifier = typename Tape::Identifier;  ///< Identifier of the tape.

      ActiveInnerType arrayValue[Elements];  ///< Array representation.

      CODI_INLINE AggregatedActiveTypeBase() = default;                                 ///< Constructor.
      CODI_INLINE AggregatedActiveTypeBase(AggregatedActiveTypeBase const&) = default;  ///< Constructor.

      /*******************************************************************************/
      /// Implementation of ExpressionInterface

      /// \copydoc codi::ExpressionInterface::StoreAs
      using StoreAs = typename std::conditional<isStatic, Impl, Impl const&>::type;
      using ADLogic = Tape;  ///< \copydoc codi::ExpressionInterface::ADLogic

      /// \copydoc codi::ExpressionInterface::getValue()
      CODI_INLINE Real const getValue() const {
        Real value{};
        static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
          Traits::template arrayAccess<i.value>(value) = arrayValue[i.value].getValue();
        });
        return value;
      }

      /// \copydoc codi::ExpressionInterface::getJacobian()
      template<size_t argNumber>
      CODI_INLINE ArrayConstructorJacobian<Impl, InnerReal, argNumber> getJacobian() const {
        return ArrayConstructorJacobian<Impl, InnerReal, argNumber>(cast());
      }

      /*******************************************************************************/
      /// Implementation of NodeInterface

      static bool constexpr EndPoint = false;  ///< \copydoc codi::NodeInterface::EndPoint

      /// \copydoc codi::NodeInterface::forEachLink
      template<typename Logic, typename... Args>
      CODI_INLINE void forEachLink(TraversalLogic<Logic>& logic, Args&&... args) const {
        static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
          logic.cast().template link<i.value>(arrayValue[i.value], *this, std::forward<Args>(args)...);
        });
      }

      /// \copydoc codi::NodeInterface::forEachLinkConstExpr
      template<typename Logic, typename... Args>
      CODI_INLINE static typename Logic::ResultType constexpr forEachLinkConstExpr(Args&&... args) {
        return Elements * Logic::template link<0, ActiveInnerType, Impl>(std::forward<Args>(args)...);
      }

    protected:

      /// Cast to implementation.
      CODI_INLINE Impl const& cast() const {
        return static_cast<Impl const&>(*this);
      }

      /// Cast to implementation.
      CODI_INLINE Impl& cast() {
        return static_cast<Impl&>(*this);
      }
  };

  /**
   * @brief Expression type to include aggregated types into the CoDiPack expression tree.
   *
   * In order to add an aggregated type to the CoDiPack expression tree. The traits
   * codi::RealTraits::AggregatedTypeTraits need to be specialized for the aggregated type. The helper
   * codi::RealTraits::ArrayAggregatedTypeTraitsBase can be used if the aggregated type can be interpreted as an array
   * of values. In addition this class needs to be extended and special constructors and assignment operators need
   * to be implemented.
   *
   * TODO: Link Example
   *
   * @tparam T_Real Real value of the aggregated type.
   * @tparam T_ActiveInnerType CoDiPack type that composes the aggregated type.
   * @tparam T_Impl The final implementation of the aggregated type.
   */
  template<typename T_Real, typename T_ActiveInnerType, typename T_Impl>
  struct AggregatedActiveType : public AggregatedActiveTypeBase<T_Real, T_ActiveInnerType, T_Impl, false> {
    public:
      using Real = T_Real;                                                           ///< See AggregatedActiveType.
      using ActiveInnerType = CODI_DD(T_ActiveInnerType, CODI_T(ActiveType<void>));  ///< See AggregatedActiveType.
      using Impl = CODI_DD(T_Impl, CODI_T(AggregatedActiveTypeBase<Type, void>));    ///< See AggregatedActiveType.

      using Base =
          AggregatedActiveTypeBase<T_Real, T_ActiveInnerType, T_Impl, false>;  ///< Abbreviation for base class.
      using Traits = RealTraits::AggregatedTypeTraits<Real>;                   ///< The traits for the aggregated type.
      using PassiveReal = RealTraits::PassiveReal<Real>;                       ///< Passive value type of the real.

      using Base::Base;  ///< Use base constructors.

      /// Constructor.
      template<typename Expr>
      CODI_INLINE AggregatedActiveType(ExpressionInterface<Real, Expr> const& expr) : Base() {
        store(expr);
      }

      /// Constructor.
      CODI_INLINE AggregatedActiveType(AggregatedActiveType const& expr) : Base() {
        store(expr);
      }

      /// Constructor.
      CODI_INLINE AggregatedActiveType(PassiveReal const& expr) : Base() {
        static_for<Base::Elements>([&](auto i) CODI_LAMBDA_INLINE {
          Base::arrayValue[i.value] = Traits::template arrayAccess<i.value>(expr);
        });
      }

      /// Assign operation.
      template<typename Expr>
      CODI_INLINE Impl& operator=(ExpressionInterface<Real, Expr> const& expr) {
        store(expr);

        return Base::cast();
      }

      /// Assign operation.
      CODI_INLINE Impl& operator=(AggregatedActiveType const& expr) {
        store(expr);

        return Base::cast();
      }

      /// Assign operation.
      CODI_INLINE Impl& operator=(PassiveReal const& expr) {
        static_for<Base::Elements>([&](auto i) CODI_LAMBDA_INLINE {
          Base::arrayValue[i.value] = Traits::template arrayAccess<i.value>(expr);
        });

        return Base::cast();
      }

    protected:

      /// \copydoc codi::InternalStatementRecordingTapeInterface::store()
      template<typename Rhs>
      CODI_INLINE void store(ExpressionInterface<Real, Rhs> const& rhs) {
        ActiveInnerType::getTape().store(*this, rhs);
      }
  };
}
