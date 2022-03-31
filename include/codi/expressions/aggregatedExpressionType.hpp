#pragma once

#include "../misc/macros.hpp"
#include "../misc/staticFor.hpp"
#include "../config.h"
#include "../tapes/interfaces/fullTapeInterface.hpp"
#include "../traits/realTraits.hpp"
#include "assignmentOperators.hpp"
#include "activeTypeWrapper.hpp"
#include "incrementOperators.hpp"
#include "lhsExpressionInterface.hpp"
#include "logic/constructStaticContext.hpp"
#include "../misc/self.hpp"
#include "../traits/computationTraits.hpp"
#include "../traits/misc/enableIfHelpers.hpp"
#include "../traits/realTraits.hpp"
#include "immutableActiveType.hpp"
#include "../tools/data/aggregatedTypeVectorAccessWrapper.hpp"

/** \copydoc codi::Namespace */
namespace codi {

  template<typename T_Creator, typename T_ReturnType, size_t T_index>
  struct ReduceJacobian {
      using Creator = CODI_DD(T_Creator, CODI_T(ExpressionInterface<double, void>));
      static size_t constexpr index = T_index;

      Creator const& creator;

      CODI_INLINE ReduceJacobian(Creator const& creator) : creator(creator) {}
  };

  template<typename Type, typename Creator, typename ReturnType, size_t index>
  CODI_INLINE ReturnType operator*(ReduceJacobian<Creator, ReturnType, index> const& reduce, Type const& jac) {
    return RealTraits::AggregatedTypeTraits<typename Creator::Real>::template adjointOfConstructor<index>(reduce.creator.getValue(), jac);
  }

  template<typename T_Real>
  struct ExtractExpressionHelper {

      using Real = CODI_DD(T_Real, CODI_ANY);

      using Traits = RealTraits::AggregatedTypeTraits<Real>;

      using InnerReal = typename Traits::InnerType;

      template<size_t T_element>
      struct ExtractOperationHelper {
          template<typename T_OpReal>
          struct type : public UnaryOperation<T_OpReal> {
            public:

              static size_t constexpr element = T_element;

              using OpReal = CODI_DD(T_OpReal, double);
              using Jacobian = Real;

              template<typename Arg>
              static CODI_INLINE OpReal primal(Arg const& arg) {
                return Traits::template arrayAccess<element>(arg);
              }

              template<typename Arg>
              static CODI_INLINE Jacobian gradient(Arg const& arg, OpReal const& result) {
                CODI_UNUSED(result);
                return Traits::template adjointOfArrayAccess<element>(arg, 1.0);
              }
          };
      };

      template<size_t element, typename Arg>
      using type = UnaryExpression<InnerReal, Arg, ExtractOperationHelper<element>::template type>;
  };

  template<typename Real, size_t element, typename Arg>
  using ExtractExpression = typename ExtractExpressionHelper<Real>::template type<element, Arg>;

  template<typename T_Real, typename T_ActiveInnerType, typename T_Impl, bool T_isStatic>
  struct AggregatedExpressionTypeBase
      : public ExpressionInterface<T_Real,T_Impl> {
    public:

      using Real = T_Real;
      using ActiveInnerType = CODI_DD(T_ActiveInnerType, CODI_T(ActiveType<void>));
      using Tape = typename ActiveInnerType::Tape;
      using Impl = CODI_DD(T_Impl, CODI_T(AggregatedExpressionTypeBase<Type, void>));
      static bool constexpr isStatic = T_isStatic;

      using Traits = RealTraits::AggregatedTypeTraits<Real>;

      using InnerReal = typename Traits::InnerType;
      static int constexpr Elements = Traits::Elements;
      using InnerIdentifier = typename Tape::Identifier;


      using ADLogic = Tape;

      ActiveInnerType arrayValue[Elements];

      CODI_INLINE AggregatedExpressionTypeBase() = default;
      CODI_INLINE AggregatedExpressionTypeBase(AggregatedExpressionTypeBase const&) = default;

      /*******************************************************************************/
      /// Implementation of ExpressionInterface

      using StoreAs = typename std::conditional<isStatic, Impl, Impl const&>::type;

      CODI_INLINE Real const getValue() const {
        Real value{};
        static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
          Traits::template arrayAccess<i.value>(value) = arrayValue[i.value].getValue();
        });
        return value;
      }

      template<size_t argNumber>
      CODI_INLINE ReduceJacobian<Impl, InnerReal, argNumber> getJacobian() const {
        return ReduceJacobian<Impl, InnerReal, argNumber>(cast());
      }

      /*******************************************************************************/
      /// Implementation of NodeInterface

      static bool constexpr EndPoint = false;

      template<typename Logic, typename... Args>
      CODI_INLINE void forEachLink(TraversalLogic<Logic>& logic, Args&&... args) const {
        static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
          logic.cast().template link<i.value>(arrayValue[i.value], *this,
                                              std::forward<Args>(args)...);
        });
      }

      template<typename Logic, typename... Args>
      CODI_INLINE static typename Logic::ResultType constexpr forEachLinkConstExpr(Args&&... args) {
        return Elements * Logic::template link<0, ActiveInnerType, Impl>(std::forward<Args>(args)...);
      }

    protected:

      template<int i>
      CODI_INLINE ActiveInnerType& array() {
        return arrayValue[i];
      }


      CODI_INLINE Impl const& cast() const {
        return static_cast<Impl const&>(*this);
      }

      CODI_INLINE Impl& cast() {
        return static_cast<Impl&>(*this);
      }
  };

  template<typename Expr>
  using EnableIfAggregatedExpressionType = typename enable_if_base_of<
      AggregatedExpressionTypeBase<
          typename Expr::Real,
          typename Expr::ActiveInnerType,
          typename Expr::Impl,
          Expr::isStatic
      >, Expr>::type;

  template<typename T_Real, typename T_ActiveInnerType, typename T_Impl>
  struct AggregatedExpressionType
      : public AggregatedExpressionTypeBase<T_Real,T_ActiveInnerType, T_Impl, false> {

      using Real = T_Real;
      using ActiveInnerType = CODI_DD(T_ActiveInnerType, CODI_T(ActiveType<void>));
      using Impl = CODI_DD(T_Impl, CODI_T(AggregatedExpressionTypeBase<Type, void>));

      using Base = AggregatedExpressionTypeBase<T_Real,T_ActiveInnerType, T_Impl, false>;
      using Traits = RealTraits::AggregatedTypeTraits<Real>;
      using PassiveReal = RealTraits::PassiveReal<Real>;

      using Base::Base;
      template<typename Expr>
      CODI_INLINE AggregatedExpressionType(ExpressionInterface<Real, Expr> const& expr) : Base() {
        store(expr);
      }

      CODI_INLINE AggregatedExpressionType(AggregatedExpressionType const& expr) : Base() {
        store(expr);
      }

      CODI_INLINE AggregatedExpressionType(PassiveReal const& expr) : Base() {
        static_for<Base::Elements>([&](auto i) CODI_LAMBDA_INLINE {
          Base::arrayValue[i.value] = Traits::template arrayAccess<i.value>(expr);
        });
      }

      template<typename Expr>
      CODI_INLINE Impl& operator=(ExpressionInterface<Real, Expr> const& expr) {
        store(expr);

        return Base::cast();
      }

      CODI_INLINE Impl& operator=(AggregatedExpressionType const& expr) {
        store(expr);

        return Base::cast();
      }

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

  template<typename T_Real, typename T_ActiveInnerType>
  struct StaticAggregatedExpressionType
      : public AggregatedExpressionTypeBase<T_Real,T_ActiveInnerType, StaticAggregatedExpressionType<T_Real, T_ActiveInnerType>, true> {

      using Real = T_Real;
      using ActiveInnerType = CODI_DD(T_ActiveInnerType, CODI_T(ActiveType<void>));

      using InnerIdentifier = typename ActiveInnerType::Identifier;
      using InnerReal = typename ActiveInnerType::Real;

      using Base = AggregatedExpressionTypeBase<Real, ActiveInnerType, StaticAggregatedExpressionType, true>;

      CODI_INLINE StaticAggregatedExpressionType() : Base() {}

      CODI_INLINE StaticAggregatedExpressionType(StaticAggregatedExpressionType const&) = default;
  };

  template<typename T_Rhs, typename T_Tape, size_t T_primalValueOffset, size_t T_constantValueOffset>
  struct ConstructStaticContextLogic<T_Rhs, T_Tape, T_primalValueOffset, T_constantValueOffset,
      EnableIfAggregatedExpressionType<T_Rhs>> {
    public:

      using Rhs = CODI_DD(T_Rhs, CODI_T(AggregatedExpressionType<double, ActiveType<CODI_ANY>, CODI_ANY, 1>));
      using Tape = T_Tape;
      static constexpr size_t primalValueOffset = T_primalValueOffset;
      static constexpr size_t constantValueOffset = T_constantValueOffset;

      using Real = typename Rhs::Real;
      using ActiveInnerType = typename Rhs::ActiveInnerType;
      static int constexpr Elements = Rhs::Elements;

      using InnerConstructor = ConstructStaticContextLogic<ActiveInnerType, Tape, 0, 0>;
      using StaticInnerType = typename InnerConstructor::ResultType;

      using InnerReal = typename Tape::Real;
      using InnerIdentifier = typename Tape::Identifier;
      using PasiverInnerReal = typename Tape::PassiveReal;

      using ResultType = StaticAggregatedExpressionType<Real, StaticInnerType>;

      CODI_INLINE static ResultType construct(InnerReal* primalVector, InnerIdentifier const* const identifiers,
                                  PasiverInnerReal const* const constantData) {
        CODI_UNUSED(constantData);

        ResultType value;

        static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
          new (&value.arrayValue[i.value]) StaticInnerType(InnerConstructor::construct(primalVector, &identifiers[primalValueOffset + i.value],
              &constantData[constantValueOffset + i.value]));
        });

        return value;
      }
  };

  /// Specialization of AggregatedTypeVectorAccessWrapper for std::complex.
  ///
  /// @tparam The nested type of the complex data type.
  template<typename T_Type>
  struct AggregatedTypeVectorAccessWrapper<T_Type, EnableIfAggregatedExpressionType<T_Type>> {
    public:

      using Type = CODI_DD(T_Type, CODI_T(AggregatedExpressionType<double, CODI_ANY, CODI_ANY>));
      using InnerType = typename Type::ActiveInnerType;
      static int constexpr Elements = Type::Elements;

      using InnerInterface = VectorAccessInterface<
          typename InnerType::Real,
          typename InnerType::Identifier>;

      using Real = typename RealTraits::DataExtraction<T_Type>::Real;
      using Identifier = typename RealTraits::DataExtraction<T_Type>::Identifier;

      using Traits = RealTraits::AggregatedTypeTraits<Real>;

      InnerInterface& innerInterface;
      int lhsOffset;

      /// Constructor
      CODI_INLINE AggregatedTypeVectorAccessWrapper(InnerInterface* innerInterface) : innerInterface(*innerInterface), lhsOffset(0) {
        innerInterface->setSizeForIndirectAccess(Elements);
      }

      /*******************************************************************************/
      /// @name Misc

      /// \copydoc VectorAccessInterface::getVectorSize()
      CODI_INLINE size_t getVectorSize() const {
        return innerInterface.getVectorSize();
      }

      /// \copydoc VectorAccessInterface::isLhsZero()
      CODI_INLINE bool isLhsZero() {
        bool isZero = true;

        static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
          innerInterface.setActiveViariableForIndirectAccess(i.value);
          isZero &= innerInterface.isLhsZero();
        });

        return isZero;
      }

      /*******************************************************************************/
      /// @name Indirect adjoint access

      /// \copydoc codi::VectorAccessInterface::setLhsAdjoint
      CODI_INLINE void setLhsAdjoint(Identifier const& index) {
        static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
          innerInterface.setActiveViariableForIndirectAccess(lhsOffset + i.value);
          innerInterface.setLhsAdjoint(index[i.value]);
        });
      }

      /// \copydoc codi::VectorAccessInterface::updateAdjointWithLhs
      CODI_INLINE void updateAdjointWithLhs(Identifier const& index, Real const& jacobian) {
        static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
          innerInterface.setActiveViariableForIndirectAccess(lhsOffset + i.value);
          innerInterface.updateAdjointWithLhs(index[i.value], Traits::template arrayAccess<i.value>(jacobian));
        });
      }

      /*******************************************************************************/
      /// @name Indirect tangent access

      /// \copydoc codi::VectorAccessInterface::setLhsTangent
      CODI_INLINE void setLhsTangent(Identifier const& index) {
        static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
          innerInterface.setActiveViariableForIndirectAccess(lhsOffset + i.value);
          innerInterface.setLhsTangent(index[i.value]);
        });
      }

      /// \copydoc codi::VectorAccessInterface::updateTangentWithLhs
      CODI_INLINE void updateTangentWithLhs(Identifier const& index, Real const& jacobian) {
        static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
          innerInterface.setActiveViariableForIndirectAccess(lhsOffset + i.value);
          innerInterface.updateTangentWithLhs(index[i.value], Traits::template arrayAccess<i.value>(jacobian));
        });
      }

      /*******************************************************************************/
      /// @name Indirect adjoint/tangent access for functions with multiple outputs

      /// \copydoc VectorAccessInterface::setSizeForIndirectAccess()
      CODI_INLINE void setSizeForIndirectAccess(size_t size) {
        innerInterface.setSizeForIndirectAccess(size * Elements);
      }

      /// \copydoc VectorAccessInterface::setActiveViariableForIndirectAccess()
      CODI_INLINE void setActiveViariableForIndirectAccess(size_t pos) {
        lhsOffset = pos * Elements;
      }

      /*******************************************************************************/
      /// @name Direct adjoint access

      /// \copydoc VectorAccessInterface::resetAdjoint()
      CODI_INLINE void resetAdjoint(Identifier const& index, size_t dim) {
        static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
          innerInterface.resetAdjoint(index[i.value], dim);
        });
      }

      /// \copydoc VectorAccessInterface::resetAdjointVec()
      CODI_INLINE void resetAdjointVec(Identifier const& index) {
        static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
          innerInterface.resetAdjointVec(index[i.value]);
        });
      }

      /// \copydoc VectorAccessInterface::getAdjoint()
      CODI_INLINE Real getAdjoint(Identifier const& index, size_t dim) {
        Real adjoint{};

        static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
          Traits::template arrayAccess<i.value>(adjoint) = innerInterface.getAdjoint(index[i.value], dim);
        });

        return adjoint;
      }

      /// \copydoc VectorAccessInterface::getAdjointVec()
      CODI_INLINE void getAdjointVec(Identifier const& index, Real* const vec) {
        for (size_t curDim = 0; curDim < innerInterface.getVectorSize(); curDim += 1) {
          vec[curDim] = this->getAdjoint(index, curDim);
        }
      }

      /// \copydoc VectorAccessInterface::updateAdjoint()
      CODI_INLINE void updateAdjoint(Identifier const& index, size_t dim, Real const& adjoint) {
        static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
          innerInterface.updateAdjoint(index[i.value], dim, Traits::template arrayAccess<i.value>(adjoint));
        });
      }

      /// \copydoc VectorAccessInterface::updateAdjointVec()
      CODI_INLINE void updateAdjointVec(Identifier const& index, Real const* const vec) {
        for (size_t curDim = 0; curDim < innerInterface.getVectorSize(); curDim += 1) {
          this->updateAdjoint(index, curDim, vec[curDim]);
        }
      }

      /*******************************************************************************/
      /// @name Primal access

      /// \copydoc VectorAccessInterface::hasPrimals()
      CODI_INLINE bool hasPrimals() {
        return innerInterface.hasPrimals();
      }

      /// \copydoc VectorAccessInterface::setPrimal()
      CODI_INLINE void setPrimal(Identifier const& index, Real const& primal) {
        static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
           innerInterface.setPrimal(index[i.value], Traits::template arrayAccess<i.value>(primal));
        });
      }

      /// \copydoc VectorAccessInterface::getPrimal()
      CODI_INLINE Real getPrimal(Identifier const& index) {
        Real primal{};

        static_for<Elements>([&](auto i) CODI_LAMBDA_INLINE {
          Traits::template arrayAccess<i.value>(primal) = innerInterface.getPrimal(index[i.value]);
        });

        return primal;
      }
  };
}
